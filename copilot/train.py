"""`make train`: mine the train set (admin replies) into models/ artefacts.

Artefacts (all derived from the confidential data -> models/ is git-ignored):
  rule_aliases.json      "ОП" -> "2.1"   (from the `violation` field)
  punishments.json       rule id -> most frequent punishments
  concept_rules.json     concept -> rule ids admins actually cite for it
  kb.jsonl.gz + kb_index.joblib   similar-case retrieval over past tickets with admin replies
  verdict.joblib         admin verdict model for complaints (supervised by real verdicts)
  lang.joblib            char-n-gram language id trained on the `lang` field
  category.joblib / priority.joblib   text classifiers on labels (real > LLM pseudo > keyword weak labels)
  templates_suggested.json   reply templates mined from clusters of similar admin answers (bonus)
"""

import gzip
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from .classify import Classifier, verdict_feature_tokens
from .concepts import ConceptMap, concept_scores, load_concepts
from .config import MODELS_DIR, data_file
from .extract import NICK_RE, extract_facts, parse_request
from .rules import REF_IN_TEXT_RE, RuleBook
from .taxonomy import Taxonomy


def iter_jsonl(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def admin_reply(rec: dict) -> str:
    replies = [m.get("text") or "" for m in rec.get("messages") or [] if (m.get("role") or "") == "admin"]
    return replies[-1] if replies else ""


_ADDR = re.compile(r"^\s*(уважаем(ый|ая|ые)|дорог(ой|ая)|dear|hello|hi)\s+[^,!.]{0,40}[,!.]\s*", re.I)


def canon_reply(text: str) -> str:
    t = NICK_RE.sub("{nick}", text or "")
    t = t.replace("<nick>", "{nick}")
    t = _ADDR.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def template_key(text: str) -> str:
    t = canon_reply(text).lower()
    t = re.sub(r"\d+", "{n}", t)
    return re.sub(r"[^\w{}\s]", "", t).strip()


def parse_violation(v: str, book: RuleBook) -> tuple[str | None, list[str]]:
    if not v:
        return None, []
    m = REF_IN_TEXT_RE.search(v)
    rid = book.resolve(m.group(1).replace(",", ".")) if m else None
    aliases = re.findall(r"[^\W\d_]{2,8}", v)
    return rid, aliases


def train(train_path: Path, valid_path: Path | None, models_dir: Path = MODELS_DIR, include_valid: bool = False,
          max_kb: int = 120_000, max_clf: int = 80_000, seed: int = 13):
    random.seed(seed)
    t_start = time.time()
    models_dir.mkdir(parents=True, exist_ok=True)
    book = RuleBook.load(data_file("rules.json"))
    tax = Taxonomy.load(data_file("taxonomy.json"))
    concepts = load_concepts()
    cmap = ConceptMap(tax, concepts)
    print(f"[train] rules={len(book.rules)} categories={len(tax.categories)} priorities={tax.priority_ids}")
    print("[train] taxonomy -> concepts:\n" + cmap.report())

    sources = [train_path] + ([valid_path] if include_valid and valid_path else [])
    rows = []
    alias_counts = defaultdict(Counter)
    punish_counts = defaultdict(Counter)
    for src in sources:
        for rec in iter_jsonl(src):
            t = parse_request(rec)
            reply = admin_reply(rec)
            rid, aliases = parse_violation(rec.get("violation") or "", book)
            if rid:
                for a in aliases:
                    alias_counts[a.lower()][rid] += 1
                if rec.get("punishment"):
                    punish_counts[rid][re.sub(r"\s+", " ", str(rec["punishment"]).strip().lower())] += 1
            rows.append({
                "id": rec.get("ticket_id"), "kind": t["kind"], "text": t["player_text"], "reported": t["reported_text"],
                "lang": rec.get("lang"), "verdict": rec.get("verdict"), "rule": rid, "reply": reply,
                "category": rec.get("category"), "priority": rec.get("priority"), "_t": t,
            })
    print(f"[train] loaded {len(rows)} tickets in {time.time() - t_start:.1f}s")

    # ---- rule aliases & punishments
    aliases = {}
    for a, cnt in alias_counts.items():
        rid, n = cnt.most_common(1)[0]
        if n >= 3 and n / sum(cnt.values()) >= 0.6 and not re.fullmatch(r"п|пункт|правил[оа]?", a):
            aliases[a] = rid
            book.add_alias(a, rid)
    punishments = {rid: cnt.most_common(3) for rid, cnt in punish_counts.items()}
    _dump(models_dir / "rule_aliases.json", aliases)
    _dump(models_dir / "punishments.json", punishments)
    print(f"[train] aliases={len(aliases)} rules with punishment stats={len(punishments)}")

    # ---- facts, weak concepts, concept -> rules
    cr_counts = defaultdict(Counter)
    for r in rows:
        facts = extract_facts(r["_t"], book)
        facts["images_count"] = r["_t"]["images_count"]
        facts["has_defense"] = bool(r["reported"])
        r["facts"] = facts
        sc = concept_scores(r["text"], concepts, r["kind"])
        r["concept"], r["concept_score"] = (max(sc.items(), key=lambda x: x[1]) if sc else (None, 0.0))
        if r["concept"] and r["rule"]:
            cr_counts[r["concept"]][r["rule"]] += 1
    concept_rules = {c: [rid for rid, n in cnt.most_common(5) if n >= 3] for c, cnt in cr_counts.items()}
    concept_rules = {c: v for c, v in concept_rules.items() if v}
    _dump(models_dir / "concept_rules.json", concept_rules)
    rule_concept = {}
    for c, cnt in cr_counts.items():
        for rid, n in cnt.items():
            if n > rule_concept.get(rid, (None, 0))[1]:
                rule_concept[rid] = (c, n)
    print(f"[train] concept->rules mined for {len(concept_rules)} concepts")

    # ---- knowledge base (similar cases)
    _build_kb(rows, models_dir, max_kb)

    # ---- suggested templates (bonus)
    _mine_templates(rows, models_dir)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import FeatureUnion, make_pipeline
    import joblib

    def text_clf():
        feats = FeatureUnion([
            ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=150_000, sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=200_000, sublinear_tf=True)),
        ])
        return make_pipeline(feats, SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=30, tol=1e-4,
                                                  class_weight="balanced", random_state=seed))

    # ---- verdict model (real supervision)
    comp = [r for r in rows if r["kind"] == "complaint" and r["verdict"]]
    vc = Counter(r["verdict"] for r in comp)
    keep = {v for v, n in vc.items() if n >= 20}
    comp = [r for r in comp if r["verdict"] in keep]
    if len(keep) >= 2 and len(comp) >= 200:
        random.shuffle(comp)
        comp = comp[:max_clf]
        X = [Classifier.model_input("complaint", r["text"], r["reported"]) + verdict_feature_tokens(r["facts"]) for r in comp]
        y = [r["verdict"] for r in comp]
        m = text_clf().fit(X, y)
        joblib.dump(m, models_dir / "verdict.joblib")
        print(f"[train] verdict model on {len(comp)} complaints, classes={dict(vc.most_common())}")
    else:
        print(f"[train] verdict model skipped (classes={dict(vc)})")

    # ---- language model on the organisers' `lang` field
    lang_rows = [r for r in rows if r["lang"] and r["text"]]
    if len({r["lang"] for r in lang_rows}) >= 2 and len(lang_rows) >= 200:
        random.shuffle(lang_rows)
        lang_rows = lang_rows[:60_000]
        lm = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=2, max_features=120_000, sublinear_tf=True),
            SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=20, random_state=seed))
        lm.fit([r["text"] for r in lang_rows], [r["lang"] for r in lang_rows])
        joblib.dump(lm, models_dir / "lang.joblib")
        print(f"[train] lang model on {len(lang_rows)} texts: {dict(Counter(r['lang'] for r in lang_rows))}")

    # ---- category / priority models
    pseudo = {}
    pl = models_dir / "pseudo_labels.jsonl"
    if pl.exists():
        for rec in iter_jsonl(pl):
            pseudo[rec["ticket_id"]] = rec
        print(f"[train] using {len(pseudo)} LLM pseudo-labels")
    X, y, yp, Xp = [], [], [], []
    for r in rows:
        lab, prio, weight = None, None, 0
        if r["category"] in tax.by_id:
            lab, prio, weight = r["category"], r["priority"], 3
        elif r["id"] in pseudo and pseudo[r["id"]].get("category") in tax.by_id:
            lab, prio, weight = pseudo[r["id"]]["category"], pseudo[r["id"]].get("priority"), 2
        elif r["concept"] and r["concept_score"] >= 1.5:
            # weak label; the admin's actual violation overrides the keyword guess
            c = rule_concept.get(r["rule"], (r["concept"], 0))[0] if r["rule"] else r["concept"]
            lab = cmap.category_for(c, r["kind"])
            weight = 1
        if not lab:
            continue
        x = Classifier.model_input(r["kind"], r["text"], r["reported"])
        X.extend([x] * weight)
        y.extend([lab] * weight)
        if prio and prio in tax.priority_ids:
            Xp.append(x + f" [cat] {lab}")
            yp.append(prio)
    if len(set(y)) >= 2 and len(y) >= 200:
        idx = list(range(len(X)))
        random.shuffle(idx)
        idx = idx[: max_clf * 2]
        m = text_clf().fit([X[i] for i in idx], [y[i] for i in idx])
        joblib.dump(m, models_dir / "category.joblib")
        print(f"[train] category model on {len(idx)} weighted examples, {len(set(y))} classes")
    else:
        print("[train] category model skipped (not enough labels) — keyword classifier will be used")
    if len(set(yp)) >= 2 and len(yp) >= 200:
        m = text_clf().fit(Xp, yp)
        joblib.dump(m, models_dir / "priority.joblib")
        print(f"[train] priority model on {len(yp)} examples")
    else:
        print("[train] priority model skipped (needs real or LLM pseudo labels; run `make label`)")
    print(f"[train] done in {time.time() - t_start:.1f}s -> {models_dir}")


def _build_kb(rows: list[dict], models_dir: Path, max_kb: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib
    kb = [r for r in rows if r["reply"] and len(r["text"]) >= 3]
    random.shuffle(kb)
    kb = kb[:max_kb]
    if not kb:
        print("[train] KB skipped (no admin replies)")
        return
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2 if len(kb) > 500 else 1,
                          max_features=250_000, sublinear_tf=True)
    mat = vec.fit_transform([r["text"] for r in kb])
    with gzip.open(models_dir / "kb.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in kb:
            f.write(json.dumps({"id": r["id"], "kind": r["kind"], "lang": r["lang"], "verdict": r["verdict"],
                                "rule": r["rule"], "text": r["text"][:400], "reply_canon": canon_reply(r["reply"])[:500]},
                               ensure_ascii=False) + "\n")
    joblib.dump({"vectorizer": vec, "matrix": mat}, models_dir / "kb_index.joblib", compress=3)
    print(f"[train] KB: {len(kb)} resolved tickets indexed")


def _mine_templates(rows: list[dict], models_dir: Path, top: int = 40):
    groups = defaultdict(list)
    for r in rows:
        if r["reply"]:
            groups[(r["kind"], r["verdict"] or "-", template_key(r["reply"]))].append(r)
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    out = []
    for (kind, verdict, _), items in ranked[:top]:
        if len(items) < 3:
            break
        concepts = Counter(i["concept"] for i in items if i["concept"])
        out.append({
            "kind": kind, "verdict": verdict, "count": len(items),
            "template": canon_reply(items[0]["reply"]),
            "top_concepts": concepts.most_common(3),
            "example_requests": [i["text"][:160] for i in items[:3]],
        })
    _dump(models_dir / "templates_suggested.json", out)
    print(f"[train] suggested templates: {len(out)} clusters")


def _dump(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    tr = Path(sys.argv[1]) if len(sys.argv) > 1 else data_file("tickets_train.jsonl.gz")
    train(tr, None)

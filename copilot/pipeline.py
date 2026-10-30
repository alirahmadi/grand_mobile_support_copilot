"""The triage pipeline: request -> language, category, priority, facts, draft with citations, routing."""

import json
import re
import time
from collections import Counter
from pathlib import Path

from . import templates as tpl
from .classify import Classifier
from .concepts import ConceptMap, load_concepts
from .config import AUTO_REPLY_MIN_CONFIDENCE, COMPLAINT_AUTO_MIN_CONFIDENCE, MODELS_DIR, data_file
from .dedup import DedupStore, spam_check
from .extract import extract_facts, parse_request, parse_time
from .guard import ALWAYS_HUMAN_CONCEPTS, ALWAYS_HUMAN_REASON, find_promises, validate_citations
from .langdetect import LangDetector
from .llm import SYSTEM_TEMPLATE, ClaudeClient, llm_enabled, response_schema
from .retrieval import KnowledgeBase, RuleRetriever
from .rules import RuleBook
from .taxonomy import Taxonomy

DRAFT_LANGS = ("ru", "kk", "uz", "en")
SERVER_LANG = {"uz": "uz", "kk": "ru", "ru": "ru"}  # kz-server admins answer in Russian unless the player wrote Kazakh
# Procedural points the offline templates cite (kept only if they exist in rules.json).
PROCEDURAL = {"need_evidence": "ЖБ 1.2.2", "invalid_profanity": "ЖБ 1.2.7", "invalid_description": "ЖБ 1.2.3",
              "not_a_complaint": "ЖБ 1.1.2", "duplicate": "ЖБ 1.2.9", "freebie": "3.1"}
APPEAL_RAW_RE = re.compile(r"не\s*согл|несогл|апелл|обжал|снять наказ|снимите|разбан|опру|rozi emas|nohaq", re.I)
# A player disputing their own punishment (launcher complaints about "my ban/mute/warn").
APPEAL_TEXT_RE = re.compile(
    r"не\s*соглас|несоглас|сн(ять|имите|ими)\s+(с меня\s+)?(наказ|бан|варн|мут|деморган|джайл|блок)|прошу снять|амнист|апелляц|обжал|"
    r"выдал\w*\s+(мне\s+)?(мут|бан|варн|деморган|джайл)|дал\w*\s+мне\s+(мут|бан|варн|деморган|джайл)|"
    r"меня\s+(наказ|забан|замут|посади)|наказал\w*\s+меня|наказани\w*\s+(выдан|было выдано)\s+(не|по ошибке)|"
    r"rozi emasman|jazoni olib|nohaq jazo|келіспеймін", re.I)
NOT_A_COMPLAINT = {"question", "payment", "bug", "stuck", "lost_items", "gratitude", "freebie"}
# Admin replies that mean the admin acts in game (watch / come / teleport) -> a human must send them.
ACTION_RE = re.compile(r"слеж|спеш|лечу|в пути|помог|ожида|телепорт|выдал|наказан|работаю|проверю|kuzat|yordamga|kuting|"
                       r"бақыла|келе жатыр|күтіңіз", re.I)
HELP_WORD_RE = re.compile(r"помог|памаг|хелп|help|yordam|ёрдам|көмек|адм[иі]н|admin", re.I)
_GREET_RE = re.compile(r"^\s*(приветствую|здравствуйте|привет|добрый день|доброго времени суток|assalomu\s+alaykum|assalamu\s+al\w*|"
                       r"salom|hello|сәлеметсіз бе)[\s,.!]*", re.I)


def _read_json(path: Path, default):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _load_joblib(path: Path):
    if not path.exists():
        return None
    import joblib
    return joblib.load(path)


class Copilot:
    def __init__(self, models_dir: Path = MODELS_DIR, use_llm: bool | None = None):
        self.book = RuleBook.load(data_file("rules.json"))
        self.tax = Taxonomy.load(data_file("taxonomy.json"))
        self.concepts = load_concepts()
        self.cmap = ConceptMap(self.tax, self.concepts)
        self.punishments: dict = _read_json(models_dir / "punishments.json", {})
        self.mined_concept_rules: dict = _read_json(models_dir / "concept_rules.json", {})
        self.retriever = RuleRetriever(self.book)
        self.kb = KnowledgeBase.load(models_dir)
        self.clf = Classifier(self.tax, self.cmap, models_dir)
        self.langdet = LangDetector(_load_joblib(models_dir / "lang.joblib"))
        self.dedup = DedupStore()
        self.procedural = {k: v for k, v in PROCEDURAL.items() if v in self.book.by_id}
        # which concept each rule belongs to (to sanity-check rule numbers typed by players)
        self.rule_concept: dict[str, str] = {}
        for cat in self.tax.categories:
            con = self.cmap.concept_for(cat.id)
            for ref in cat.rules:
                for rid in self.book.parse_refs(ref)[0]:
                    if con:
                        self.rule_concept.setdefault(rid, con)
        for name in self.concepts:
            for rid in self.concept_rules(name):
                self.rule_concept.setdefault(rid, name)
        use_llm = llm_enabled() if use_llm is None else use_llm
        self.llm = ClaudeClient() if use_llm else None
        self.system_prompt = SYSTEM_TEMPLATE.format(taxonomy=self.tax.as_prompt_block(), rules=self.book.as_prompt_block())
        self.schema = response_schema([c.id for c in self.tax.categories], self.tax.priority_ids, [r.id for r in self.book.rules])

    # ------------------------------------------------------------------ helpers
    def category_rules(self, category: str | None) -> list[str]:
        cat = self.tax.by_id.get(category or "")
        out = []
        for ref in (cat.rules if cat else []):
            for rid in self.book.parse_refs(ref)[0] or ([ref] if ref in self.book.by_id else []):
                if rid not in out:
                    out.append(rid)
        return out

    def concept_rules(self, concept: str | None, k: int = 3) -> list[str]:
        if not concept:
            return []
        mined = [r for r in self.mined_concept_rules.get(concept, []) if r in self.book.by_id]
        if mined:
            return mined[:k]
        c = self.concepts.get(concept)
        return self.retriever.by_hints(c.rule_hints, k) if c else []

    def rule_family(self, rid: str) -> str | None:
        if rid in self.rule_concept:
            return self.rule_concept[rid]
        doc = rid.split(" ")[0] if " " in rid else None
        return self.rule_concept.get(doc) if doc else None

    def typical_punishment(self, rid: str | None) -> str | None:
        if not rid:
            return None
        stats = self.punishments.get(rid)
        return stats[0][0] if stats else None

    def triage(self, req: dict) -> dict:
        try:
            return self.finish(self.prepare(req))
        except Exception as e:  # never fail the whole request/batch because of one odd ticket
            return self.fallback_response(req, e)

    def fallback_response(self, req, error: Exception) -> dict:
        kind = str((req or {}).get("kind") or "report").lower() if isinstance(req, dict) else "report"
        kind = kind if kind in ("complaint", "report") else "report"
        category = self.tax.fallback_category(kind)
        text = tpl.render("generic_complaint" if kind == "complaint" else "generic", "ru")
        return {
            "lang": "und", "category": category, "priority": "p2" if "p2" in self.tax.priority_ids else self.tax.priority_ids[-1],
            "is_spam_or_duplicate": False,
            "extracted": {"nickname": None, "rule_refs": [], "has_evidence": None, "server": None},
            "draft_reply": {"lang": "ru", "text": text, "cited": []},
            "needs_human": True, "confidence": 0.0,
            "reasons": ["обращение не удалось разобрать автоматически — нужен администратор"],
            "meta": {"mode": "fallback", "error": f"{type(error).__name__}: {error}"[:300], "latency_ms": 0,
                     "cost_usd": 0.0, "llm_error": None, "suggested_verdict": None},
        }

    def safe_prepare(self, req: dict):
        if isinstance(req, dict) and "_bad_line" in req:
            return {"_error": ValueError("строка не является корректным JSON"), "_req": req}
        try:
            return self.prepare(req)
        except Exception as e:
            return {"_error": e, "_req": req}

    def safe_finish(self, P: dict) -> dict:
        if "_error" in P:
            return self.fallback_response(P["_req"], P["_error"])
        try:
            return self.finish(P)
        except Exception as e:
            return self.fallback_response(P.get("t", {}), e)

    # ------------------------------------------------------------------ stage 1 (sequential, cheap)
    def prepare(self, req: dict) -> dict:
        t0 = time.perf_counter()
        t = parse_request(req)
        kind = t["kind"] if t["kind"] in ("complaint", "report") else "report"
        text = t["player_text"]
        facts = extract_facts(t, self.book)
        facts["images_count"] = t["images_count"]
        facts["has_defense"] = bool(t["reported_text"])

        hint = t["lang_hint"] if t["lang_hint"] in DRAFT_LANGS else None
        lang, lang_conf = self.langdet.detect(text, hint=hint)
        server_lang = SERVER_LANG.get(t["server_lang"] or "", "ru")
        if lang in DRAFT_LANGS and (lang_conf >= 0.7 or lang in ("kk", "uz")):
            draft_lang = lang
        else:
            draft_lang = server_lang
        cyr = sum(1 for ch in text if "Ѐ" <= ch <= "ӿ")
        lat = sum(1 for ch in text if "a" <= ch.lower() <= "z")
        uz_script = "cyrillic" if cyr > lat and lang == "uz" else "latin"

        rule_concepts = [c for c in (self.rule_family(r) for r in facts["rule_refs"]) if c]
        cls = self.clf.predict(kind, t["cls_text"], t["reported_text"], rule_concepts, facts)
        concept = cls["concept"]
        appeal_signal = kind == "complaint" and (concept == "ban_appeal" or bool(APPEAL_RAW_RE.search(str(t["rule_refs_raw"] or "")))
                                                 or bool(APPEAL_TEXT_RE.search(text)))
        is_spam, spam_reason = spam_check(text, kind, t["images_count"])
        if concept == "spam_ads" and cls["concept_scores"].get("spam_ads", 0) >= 1:
            is_spam, spam_reason = True, spam_reason or "реклама/флуд"
        is_dup, dup_of, dup_reason = self.dedup.check_and_add(
            reporter=t["reporter_hash"], nick=facts["nickname"], text=text, concept=concept,
            ts=parse_time(t["created_at"]), ticket_id=t["ticket_id"])

        # ---- verdict suggestion for complaints: the model trained on real admin verdicts decides,
        # duplicates come from the reporter's history, appeals need both wording and model agreement.
        verdict, verdict_p, variant, vprobs = None, 0.0, None, {}
        if kind == "complaint":
            mv, mp, vprobs = self.clf.predict_verdict(kind, t["cls_text"], t["reported_text"], facts)
            if is_dup:
                verdict, verdict_p = "duplicate", 0.85
            elif vprobs:
                if appeal_signal and vprobs.get("appeal", 0) >= 0.15:
                    verdict, verdict_p = "appeal", max(vprobs["appeal"], 0.6)
                else:
                    verdict, verdict_p = mv, mp
                    # "wrong format" without any formal reason (nick and a real description are present, no obscenity)
                    # is usually the model guessing on unfamiliar (kk/uz) text: take the next most likely verdict
                    if (mv == "invalid_format" and not facts["profanity"] and facts["nickname"]
                            and facts["_text_letters"] >= 15 and concept not in NOT_A_COMPLAINT):
                        alt = sorted(((p, v) for v, p in vprobs.items() if v not in ("invalid_format", "duplicate")), reverse=True)
                        if alt:
                            verdict, verdict_p = alt[0][1], alt[0][0]
            elif appeal_signal:
                verdict, verdict_p = "appeal", 0.6
            elif facts["profanity"]:
                verdict, verdict_p = "invalid_format", 0.6
            else:
                verdict, verdict_p = "approved", 0.5
            if verdict == "invalid_format":
                variant = ("invalid_profanity" if facts["profanity"] else
                           "not_a_complaint" if concept in NOT_A_COMPLAINT else "invalid_description")
        appeal = verdict == "appeal"
        if appeal:
            concept = "ban_appeal"
            cls["category"] = self.cmap.category_for("ban_appeal", kind) or cls["category"]
        if kind == "complaint" and variant == "not_a_complaint" and "complaint.invalid" in self.tax.by_id:
            cls["category"] = "complaint.invalid"
        cls["priority"], _ = self.clf.priority(cls["category"], kind, concept, cls["concept_scores"], text, facts)
        if verdict in ("need_evidence", "invalid_format") and verdict_p >= 0.5 and not appeal:
            cls["priority"] = self.clf.prules.get("complaint_without_evidence", "p3")

        # ---- rules for citation
        cat_rules = self.category_rules(cls["category"])
        consistent, inconsistent = [], []
        for rid in facts["rule_refs"]:
            fam = self.rule_family(rid)
            (consistent if (fam is None or concept is None or fam == concept or rid in cat_rules) else inconsistent).append(rid)
        # prefer the player's references that match the detected violation
        consistent.sort(key=lambda r: (self.rule_family(r) != concept, r not in cat_rules))
        bm25 = self.retriever.search(text + " " + " ".join(self.concepts[concept].rule_hints if concept else []), k=5)
        mined = self.concept_rules(concept)
        primary = None
        if kind == "complaint" and not appeal:
            def violation_rule(r):  # complaint-procedure rules (ЖБ) are never the violation itself
                return r in self.book.by_id and self.book.by_id[r].ref and self.book.by_id[r].doc != "ЖБ"
            primary = ([r for r in consistent if violation_rule(r)][:1] or [r for r in mined if violation_rule(r)][:1]
                       or [r for r in cat_rules if violation_rule(r)][:1] or [None])[0]
        candidates = []
        for rid in consistent + mined + cat_rules + [r for r, _ in bm25]:
            if rid and rid not in candidates:
                candidates.append(rid)

        neighbours = self.kb.search(t["cls_text"], kind=kind, k=8) if self.kb else []
        return {
            "t": t, "kind": kind, "facts": facts, "lang": lang, "lang_conf": lang_conf, "draft_lang": draft_lang,
            "uz_script": uz_script, "cls": cls, "concept": concept, "appeal": appeal,
            "is_spam": is_spam, "spam_reason": spam_reason, "is_dup": is_dup, "dup_of": dup_of, "dup_reason": dup_reason,
            "consistent_refs": consistent, "inconsistent_refs": inconsistent, "primary_rule": primary,
            "candidates": candidates[:10], "verdict": verdict, "verdict_p": verdict_p, "verdict_variant": variant,
            "verdict_probs": vprobs, "neighbours": neighbours, "t_prepare": time.perf_counter() - t0,
        }

    # ------------------------------------------------------------------ stage 2 (parallel, may call the LLM)
    def finish(self, P: dict) -> dict:
        t0 = time.perf_counter()
        draft = self.template_draft(P)
        mode, llm_meta = "template", None
        if self.llm is not None:
            out, llm_meta = self.llm.call_json(self.system_prompt, self.llm_user_message(P, draft), self.schema)
            if out:
                merged = self.merge_llm(P, draft, out)
                if merged:
                    draft, mode = merged, "llm"
        resp = self.finalize(P, draft, mode)
        resp["meta"].update({
            "latency_ms": int(((time.perf_counter() - t0) + P["t_prepare"]) * 1000),
            "cost_usd": round(llm_meta["cost_usd"], 6) if llm_meta else 0.0,
            "llm_error": llm_meta.get("error") if llm_meta else None,
        })
        return resp

    # ------------------------------------------------------------------ offline draft
    def template_draft(self, P: dict) -> dict:
        lang, kind, concept, f = P["draft_lang"], P["kind"], P["concept"], P["facts"]
        category = P["cls"]["category"]
        nick = f["nickname"] or ""
        conf = P["cls"]["category_conf"]
        reasons, cited, needs_human, key = [], [], True, None
        if kind == "complaint":
            v = P["verdict"]
            if v == "duplicate":
                key, needs_human, conf = "duplicate", not P["is_dup"], P["verdict_p"]
                reasons.append(P["dup_reason"] or "повторная жалоба")
            elif v == "appeal":
                key = "appeal"
                reasons.append("игрок оспаривает своё наказание — решает старшая администрация")
            elif concept == "admin_abuse":
                key = "admin_complaint"
            elif v == "invalid_format":
                key = P["verdict_variant"] or "invalid_description"
                needs_human, conf = P["verdict_p"] < COMPLAINT_AUTO_MIN_CONFIDENCE, P["verdict_p"]
                reasons.append({"invalid_profanity": "в тексте жалобы оскорбления/мат (п. 1.2.7 правил жалоб)",
                                "not_a_complaint": "это не жалоба на игрока, а вопрос или техническая проблема",
                                "invalid_description": "нет ника нарушителя или описания нарушения"}[key])
            elif v == "need_evidence":
                key, needs_human, conf = "need_evidence", P["verdict_p"] < COMPLAINT_AUTO_MIN_CONFIDENCE, P["verdict_p"]
                reasons.append("доказательств недостаточно" + ("" if P["verdict_p"] >= COMPLAINT_AUTO_MIN_CONFIDENCE
                                                                 else f" (уверенность модели {P['verdict_p']:.2f} — проверьте)"))
            elif v == "already_punished":
                key = "already_punished"
                reasons.append("по модели решений игрок, вероятно, уже наказан — проверьте историю наказаний")
            elif v == "rejected":
                key = "rejected"
                reasons.append("модель решений предлагает отклонить — проверьте доказательства")
            else:
                rid = P["primary_rule"]
                if rid:
                    pun = tpl.translate_punishment(self.typical_punishment(rid) or "", lang)
                    key = "approved" if pun else "approved_no_punishment"
                    text = tpl.render(key, lang, nick=nick, ref=self.book.label(rid, lang), punishment=pun or "")
                    reasons.append("решение о наказании подтверждает администратор после просмотра доказательств")
                    return self._draft(text, [rid], True, conf, reasons, "approved", lang, P)
                key = "generic_complaint"
                reasons.append("не удалось сопоставить жалобу с пунктом правил")
        else:
            key = self._report_key(P)
            if key not in ("account", "payment", "threat", "appeal_report", "freebie", "gratitude"):
                kb = self.kb_answer(P)
                if kb:
                    return kb
            needs_human = key in ("watching_player", "watching_you", "help", "question", "account", "payment", "threat", "generic")
            if not needs_human:
                conf = max(conf, 0.75)
        if key in self.procedural:
            cited = [self.procedural[key]]
        text = tpl.render(key, lang, nick=nick)
        verdict = P["verdict"] if kind == "complaint" else ("escalate" if needs_human else "answer")
        return self._draft(text, cited, needs_human, conf, reasons, verdict, lang, P)

    def _report_key(self, P: dict) -> str:
        category, concept = P["cls"]["category"], P["concept"]
        if P["is_spam"]:
            return "offtopic"
        if concept == "threat":
            return "threat"
        if concept == "ban_appeal":
            return "appeal_report"
        if concept == "freebie":
            return "freebie"
        if concept == "gratitude":
            return "gratitude"
        if category == "report.offtopic" and HELP_WORD_RE.search(P["t"]["player_text"]):
            return "watching_you"
        return {"report.offtopic": "offtopic", "report.help": "bug" if concept == "bug" else "help",
                "report.complaint": "watching_player", "report.question": "question", "report.account": "account",
                "report.payment": "payment"}.get(category, "generic")

    def _draft(self, text, cited, needs_human, conf, reasons, verdict, lang, P) -> dict:
        if lang == "uz" and P["uz_script"] == "cyrillic":
            text = tpl.uz_latin_to_cyrillic(text)
        return {"text": text, "cited": cited, "needs_human": needs_human, "confidence": conf, "reasons": reasons,
                "suggested_verdict": verdict, "category": P["cls"]["category"], "priority": P["cls"]["priority"]}

    def kb_answer(self, P: dict) -> dict | None:
        """Reuse the admins' own reply when similar past reports were answered the same way."""
        lang = P["draft_lang"]
        if lang not in ("ru", "uz"):
            return None
        pool = []
        for n in P["neighbours"]:
            reply = n.get("reply_canon") or ""
            is_cyr = bool(re.search(r"[а-яё]", reply, re.I))
            if n["sim"] >= 0.45 and reply and "{nick}" not in reply and len(reply) <= 160 and (is_cyr == (lang == "ru")):
                pool.append(reply)
        if len(pool) < 3:
            return None
        keys = [re.sub(r"[^\w ]", "", _GREET_RE.sub("", r.lower())).strip() for r in pool]
        key, cnt = Counter(keys).most_common(1)[0]
        if cnt < 3 or cnt / len(pool) < 0.4 or len(key) < 4:
            return None
        body = _GREET_RE.sub("", pool[keys.index(key)]).strip().rstrip(".!") + "."
        greet = "Приветствую" if lang == "ru" else "Assalomu alaykum"
        text = f"{greet}, {body[:1].lower()}{body[1:]}"
        human = bool(ACTION_RE.search(text))
        refs = [r for r in self.book.refs_in_text(text)]
        reason = f"типовой ответ админов на похожие репорты ({cnt} из {len(pool)})"
        return self._draft(text, refs, human, min(0.9, 0.5 + 0.1 * cnt), [reason] + (["админ должен выполнить действие в игре"] if human else []),
                           "escalate" if human else "answer", lang, P)

    # ------------------------------------------------------------------ LLM path
    def llm_user_message(self, P: dict, offline: dict) -> str:
        t, f, cls = P["t"], P["facts"], P["cls"]
        cand = [{"id": r, "text": self.book.describe(r, 320)} for r in P["candidates"]]
        pun = {r: self.typical_punishment(r) for r in P["candidates"] if self.typical_punishment(r)}
        similar = [{"player": n.get("text", "")[:200], "admin_reply": n.get("reply_canon", "")[:260],
                    "verdict": n.get("verdict"), "similarity": n["sim"]} for n in P["neighbours"][:4]]
        payload = {
            "kind": P["kind"], "server": t["region"], "target_lang": P["draft_lang"], "uz_script": P["uz_script"],
            "player_messages": t["player_text"][:1500], "reported_player_reply": t["reported_text"][:600] or None,
            "reported_nick": f["nickname"], "rule_field_as_typed": t["rule_refs_raw"], "images_count": t["images_count"],
            "extracted": {k: v for k, v in f.items() if not k.startswith("_") and k != "images_count"},
            "signals": {"keyword_category": cls["category"], "category_candidates": cls["category_top"],
                        "priority_guess": cls["priority"], "suggested_verdict": P["verdict"],
                        "verdict_model_probs": P["verdict_probs"], "is_appeal": P["appeal"],
                        "is_duplicate": P["is_dup"], "duplicate_reason": P["dup_reason"] or None, "is_spam": P["is_spam"],
                        "rule_refs_not_matching_text": P["inconsistent_refs"]},
            "candidate_rules": cand, "typical_punishment_by_rule": pun, "similar_cases": similar,
            "offline_draft": {"text": offline["text"], "cited": offline["cited"], "needs_human": offline["needs_human"]},
        }
        return "Обращение для разбора:\n" + json.dumps(payload, ensure_ascii=False)

    def merge_llm(self, P: dict, fallback: dict, out: dict) -> dict | None:
        text = (out.get("draft_text") or "").strip()
        if len(text) < 8:
            return None
        cat = out.get("category") if out.get("category") in self.tax.by_id else fallback["category"]
        if P["kind"] in ("complaint", "report") and self.tax.by_id[cat].kind not in (P["kind"], None):
            cat = fallback["category"]
        prio = out.get("priority") if out.get("priority") in self.tax.priority_ids else fallback["priority"]
        try:
            conf = float(out.get("confidence", 0.6))
        except (TypeError, ValueError):
            conf = 0.6
        return {"text": text, "cited": list(out.get("cited") or []), "needs_human": bool(out.get("needs_human", True)),
                "confidence": max(0.0, min(1.0, conf)), "reasons": [str(r) for r in (out.get("reasons") or [])][:3],
                "suggested_verdict": out.get("suggested_verdict"), "category": cat, "priority": prio}

    # ------------------------------------------------------------------ final guard + response
    def finalize(self, P: dict, d: dict, mode: str) -> dict:
        f = P["facts"]
        category = d["category"]
        concept = P["concept"]
        allowed = (set(P["candidates"]) | set(self.category_rules(category)) | set(self.concept_rules(concept, k=5))
                   | set(self.procedural.values()) | {r for r, _ in self.retriever.search(P["t"]["player_text"], k=10)})
        text, cited, problems = validate_citations(d["text"], d["cited"], self.book, allowed)
        reasons = list(d["reasons"])
        needs_human = d["needs_human"]
        if problems:
            reasons += problems[:2]
            needs_human = True
        if not text or len(text) < 8:
            fb = self.template_draft(P)
            text, cited, needs_human = fb["text"], fb["cited"], True
        promises = find_promises(text)
        if promises:
            needs_human = True
            if not any("наказан" in r for r in reasons) or set(promises) - {"punish"}:
                names = {"unban": "разбан", "refund": "возврат средств", "punish": "наказание игрока"}
                reasons.append("черновик содержит решение (" + ", ".join(names[p] for p in promises) + ") — подтверждает администратор")
        if category in self.tax.always_human:
            needs_human = True
            reasons.append(f"категория {category} всегда требует человека")
        for c in {concept} & ALWAYS_HUMAN_CONCEPTS:
            needs_human = True
            reasons.append(ALWAYS_HUMAN_REASON[c])
        if d.get("suggested_verdict") == "approved" and not cited:
            needs_human = True
            reasons.append("нет подходящей цитаты правил")
        if P["inconsistent_refs"]:
            reasons.append(f"игрок указал пункт {', '.join(P['inconsistent_refs'])}, но по тексту нарушение другое")
        if f["_unknown_rule_refs"]:
            reasons.append(f"указан несуществующий пункт правил: {', '.join(f['_unknown_rule_refs'][:3])}")
        if P["is_dup"] and P["dup_reason"] and not any("повтор" in r for r in reasons):
            reasons.append(P["dup_reason"])
        priority = self.clf.escalate(d["priority"], concept, P["cls"]["concept_scores"], P["t"]["player_text"])
        conf = round(float(d["confidence"]) * (0.85 if P["lang"] == "und" else 1.0), 2)
        if conf < AUTO_REPLY_MIN_CONFIDENCE and not needs_human:
            needs_human = True
            reasons.append(f"низкая уверенность ({conf:.2f})")
        if not reasons:
            reasons.append("типовой ответ, можно отправлять автоматически" if not needs_human else "требуется проверка администратора")
        uniq = list(dict.fromkeys(reasons))
        meta = {"mode": mode, "suggested_verdict": d.get("suggested_verdict"), "concept": concept,
                "category_candidates": P["cls"]["category_top"], "rule_candidates": P["candidates"],
                "lang_confidence": P["lang_conf"], "is_duplicate": P["is_dup"], "duplicate_of": P["dup_of"]}
        if P["draft_lang"] == "kk":
            from .textnorm import to_game_encoding
            meta["text_cp1251"] = to_game_encoding(text)
        return {
            "lang": P["lang"],
            "category": category,
            "priority": priority,
            "is_spam_or_duplicate": bool(P["is_spam"] or P["is_dup"]),
            "extracted": {"nickname": f["nickname"], "rule_refs": f["rule_refs"], "has_evidence": f["has_evidence"],
                          "server": f["server"]},
            "draft_reply": {"lang": P["draft_lang"], "text": text, "cited": cited},
            "needs_human": bool(needs_human),
            "confidence": conf,
            "reasons": uniq[:5],
            "meta": meta,
        }

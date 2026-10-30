"""The triage pipeline: request -> language, category, priority, facts, draft with citations, routing."""

import json
import re
import time
from pathlib import Path

from .classify import Classifier
from .concepts import ConceptMap, load_concepts
from .config import AUTO_REPLY_MIN_CONFIDENCE, MODELS_DIR, SUPPORTED_LANGS, data_file
from .dedup import DedupStore, spam_check
from .extract import extract_facts, parse_request, parse_time
from .guard import ALWAYS_HUMAN_CONCEPTS, ALWAYS_HUMAN_REASON, find_promises, validate_citations
from .langdetect import LangDetector
from .llm import SYSTEM_TEMPLATE, ClaudeClient, llm_enabled, response_schema
from .retrieval import KnowledgeBase, RuleRetriever
from .rules import RuleBook
from .taxonomy import Taxonomy
from . import templates as tpl

VIOLATION_RESPONSES = {"violation"}
NO_RULES_NEEDED = {"need_evidence", "invalid_format", "duplicate", "spam", "stuck", "bug", "lost_items",
                   "payment", "account", "ban_appeal", "question", "gratitude", "generic", "admin_complaint",
                   "threat", "kb"}
_ADDRESS_RE = re.compile(r"^\s*(уважаем(ый|ая|ые)|дорог(ой|ая)|здравствуйте|привет|добрый день)[^,!.]*[,!.]\s*", re.I)


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
        for alias, rid in _read_json(models_dir / "rule_aliases.json", {}).items():
            self.book.add_alias(alias, rid)
        self.punishments: dict = _read_json(models_dir / "punishments.json", {})
        self.mined_concept_rules: dict = _read_json(models_dir / "concept_rules.json", {})
        self.retriever = RuleRetriever(self.book)
        self.kb = KnowledgeBase.load(models_dir)
        self.clf = Classifier(self.tax, self.cmap, models_dir)
        self.langdet = LangDetector(_load_joblib(models_dir / "lang.joblib"))
        self.dedup = DedupStore()
        # which concept each rule belongs to (to sanity-check rule numbers typed by players)
        self.rule_concept: dict[str, str] = {}
        for name, c in self.concepts.items():
            for rid in self.concept_rules(name):
                self.rule_concept.setdefault(rid, name)
        use_llm = llm_enabled() if use_llm is None else use_llm
        self.llm = ClaudeClient() if use_llm else None
        self.system_prompt = SYSTEM_TEMPLATE.format(taxonomy=self.tax.as_prompt_block(),
                                                    rules=self.book.as_prompt_block())
        self.schema = response_schema([c.id for c in self.tax.categories], self.tax.priority_ids,
                                      [r.id for r in self.book.rules])

    # ------------------------------------------------------------------ helpers
    def concept_rules(self, concept: str | None, k: int = 3) -> list[str]:
        if not concept:
            return []
        mined = [r for r in self.mined_concept_rules.get(concept, []) if r in self.book.by_id]
        if mined:
            return mined[:k]
        c = self.concepts.get(concept)
        return self.retriever.by_hints(c.rule_hints, k) if c else []

    def typical_punishment(self, rid: str | None) -> str | None:
        if not rid:
            return None
        stats = self.punishments.get(rid)
        if stats:
            return stats[0][0]
        r = self.book.by_id.get(rid)
        if r and r.punishment and len(r.punishment) < 60:
            return r.punishment
        return None

    def triage(self, req: dict) -> dict:
        return self.finish(self.prepare(req))

    # ------------------------------------------------------------------ stage 1 (sequential, cheap)
    def prepare(self, req: dict) -> dict:
        t0 = time.perf_counter()
        t = parse_request(req)
        kind = t["kind"] if t["kind"] in ("complaint", "report") else "report"
        text = t["player_text"]
        facts = extract_facts(t, self.book)
        facts["images_count"] = t["images_count"]
        facts["has_defense"] = bool(t["reported_text"])

        lang, lang_conf = self.langdet.detect(text, hint=t["lang_hint"])
        draft_lang = lang if lang in SUPPORTED_LANGS else "ru"
        cyr = sum(1 for ch in text if "Ѐ" <= ch <= "ӿ")
        lat = sum(1 for ch in text if "a" <= ch.lower() <= "z")
        uz_script = "cyrillic" if cyr > lat else "latin"

        rule_concepts = [self.rule_concept[r] for r in facts["rule_refs"] if r in self.rule_concept]
        cls = self.clf.predict(kind, text, t["reported_text"], rule_concepts)
        concept = cls["concept"]

        is_spam, spam_reason = spam_check(text, kind, t["images_count"])
        if concept == "spam_ads" and cls["concept_scores"].get("spam_ads", 0) >= 1:
            is_spam, spam_reason = True, spam_reason or "реклама/флуд"
        is_dup, dup_of, dup_reason = self.dedup.check_and_add(
            reporter=t["reporter_hash"], nick=facts["nickname"], text=text, concept=concept,
            ts=parse_time(t["created_at"]), ticket_id=t["ticket_id"])

        # ---- rules for citation
        consistent, inconsistent = [], []
        for rid in facts["rule_refs"]:
            rc = self.rule_concept.get(rid)
            (consistent if (rc is None or concept is None or rc == concept) else inconsistent).append(rid)
        concept_rules = self.concept_rules(concept)
        bm25 = self.retriever.search(text + " " + " ".join(self.concepts[concept].rule_hints if concept else []), k=5)
        primary = None
        if kind == "complaint" or (concept in self.concepts and self.concepts[concept].response == "violation"):
            primary = (consistent[:1] or concept_rules[:1] or [None])[0]
        candidates = []
        for rid in consistent + concept_rules + [r for r, _ in bm25]:
            if rid and rid not in candidates:
                candidates.append(rid)

        # ---- verdict suggestion for complaints
        verdict, verdict_p = None, 0.0
        if kind == "complaint":
            if is_dup:
                verdict, verdict_p = "duplicate", 0.85
            elif is_spam:
                verdict, verdict_p = "invalid_format", 0.7
            elif not facts["nickname"] or (facts["_text_letters"] < 8 and not facts["rule_refs"]):
                verdict, verdict_p = "invalid_format", 0.7
            elif facts["has_evidence"] is False:
                verdict, verdict_p = "need_evidence", 0.75
            else:
                verdict, verdict_p = "approved", 0.55
            mv, mp = self.clf.predict_verdict(kind, text, t["reported_text"], facts)
            if mv and mv != verdict and mp >= 0.6 and not (mv == "duplicate" and not is_dup):
                verdict, verdict_p = mv, mp

        neighbours = self.kb.search(text, kind=kind, k=4) if self.kb else []
        return {
            "t": t, "kind": kind, "facts": facts, "lang": lang, "lang_conf": lang_conf,
            "draft_lang": draft_lang, "uz_script": uz_script, "cls": cls, "concept": concept,
            "is_spam": is_spam, "spam_reason": spam_reason, "is_dup": is_dup, "dup_of": dup_of,
            "dup_reason": dup_reason, "consistent_refs": consistent, "inconsistent_refs": inconsistent,
            "primary_rule": primary, "candidates": candidates[:8], "bm25": bm25, "verdict": verdict,
            "verdict_p": verdict_p, "neighbours": neighbours, "t_prepare": time.perf_counter() - t0,
        }

    # ------------------------------------------------------------------ stage 2 (parallel, may call the LLM)
    def finish(self, P: dict) -> dict:
        t0 = time.perf_counter()
        draft = self.template_draft(P)
        mode, llm_meta = "template", None
        if self.llm is not None:
            out, llm_meta = self.llm.call_json(self.system_prompt, self.llm_user_message(P), self.schema)
            if out:
                merged = self.merge_llm(P, draft, out)
                if merged:
                    draft, mode = merged, "llm"
        resp = self.finalize(P, draft, mode)
        latency = (time.perf_counter() - t0) + P["t_prepare"]
        resp["meta"].update({
            "latency_ms": int(latency * 1000),
            "cost_usd": round(llm_meta["cost_usd"], 6) if llm_meta else 0.0,
            "llm_error": llm_meta.get("error") if llm_meta else None,
        })
        return resp

    # ------------------------------------------------------------------ offline draft
    def template_draft(self, P: dict) -> dict:
        lang, kind, concept, f = P["draft_lang"], P["kind"], P["concept"], P["facts"]
        nick = f["nickname"] or ""
        con = self.concepts.get(concept)
        response = con.response if con else "generic"
        reasons, cited, needs_human, conf = [], [], True, P["cls"]["category_conf"]
        rid = P["primary_rule"]
        violation = tpl.VIOLATION.get(concept, tpl.GENERIC_VIOLATION).get(lang, tpl.GENERIC_VIOLATION["ru"])
        if lang == "ru" and rid:
            violation = f"«{self.book.by_id[rid].short}»"

        if P["is_spam"] and not P["is_dup"]:
            key, needs_human, conf = "spam", False, 0.8
            reasons.append(f"спам: {P['spam_reason']}")
        elif kind == "complaint":
            v = P["verdict"]
            if response == "admin_complaint":
                key = "admin_complaint"
            elif v == "duplicate":
                key, needs_human, conf = "duplicate", False, 0.85
                reasons.append(P["dup_reason"] or "повторная жалоба")
            elif v == "invalid_format":
                key, needs_human, conf = "invalid_format", False, 0.75
                reasons.append("не указан ник нарушителя или суть нарушения")
            elif v == "need_evidence":
                key, needs_human, conf = "need_evidence", False, 0.75
                reasons.append("к жалобе не приложены доказательства")
            elif v == "rejected":
                key = "rejected"
                reasons.append("модель решений предлагает отклонить — проверьте материалы")
            else:
                if rid:
                    key = "approved"
                    pun = tpl.translate_punishment(self.typical_punishment(rid) or "", lang)
                    punish = (tpl.render("punish_known", lang, nick=nick, punishment=pun) if pun
                              else tpl.T["punish_unknown"].get(lang, tpl.T["punish_unknown"]["ru"]))
                    cited = [rid]
                    reasons.append("решение о наказании подтверждает администратор")
                    text = tpl.render("approved", lang, nick=nick, rules=rid, violation=violation, punish=punish)
                    return self._draft(text, cited, True, conf, reasons, "approved", lang, P)
                key = "generic"
                reasons.append("не удалось сопоставить жалобу с пунктом правил")
        else:
            key = {"violation": "report_violation"}.get(response, response)
            if key == "report_violation":
                if rid:
                    cited = [rid]
                    text = tpl.render("report_violation", lang, rules=rid, violation=violation)
                    reasons.append("нужна проверка администратором (логи/чат)")
                    return self._draft(text, cited, True, conf, reasons, "escalate", lang, P)
                key = "generic"
            if key == "question":
                kb = self.kb_answer(P)
                if kb:
                    return kb
            if key in ("bug", "gratitude"):
                needs_human, conf = False, max(conf, 0.75)
            if key not in tpl.T:
                key = "generic"
        if concept in ALWAYS_HUMAN_CONCEPTS:
            needs_human = True
        text = tpl.render(key, lang, nick=nick)
        verdict = P["verdict"] if kind == "complaint" else ("escalate" if needs_human else "answer")
        return self._draft(text, cited, needs_human, conf, reasons, verdict, lang, P)

    def _draft(self, text, cited, needs_human, conf, reasons, verdict, lang, P) -> dict:
        if lang == "uz" and P["uz_script"] == "cyrillic":
            text = tpl.uz_latin_to_cyrillic(text)
        return {"text": text, "cited": cited, "needs_human": needs_human, "confidence": conf,
                "reasons": reasons, "suggested_verdict": verdict, "category": P["cls"]["category"],
                "priority": P["cls"]["priority"]}

    def kb_answer(self, P: dict) -> dict | None:
        """Reuse an admin reply when several near-identical past questions got the same generic answer."""
        nb = P["neighbours"]
        if P["draft_lang"] != "ru" or len(nb) < 2 or nb[0]["sim"] < 0.6:
            return None
        canon = [n.get("reply_canon", "") for n in nb if n["sim"] >= 0.5]
        if not canon or canon.count(canon[0]) < 2 or "{nick}" in canon[0] or len(canon[0]) < 15:
            return None
        text = _ADDRESS_RE.sub("", canon[0]).strip()
        text = f"{tpl.GREETING['ru']} {text[:1].upper()}{text[1:]}"
        refs = [r for r in self.book.refs_in_text(text) if r in self.book.by_id]
        return self._draft(text, refs, False, min(0.9, nb[0]["sim"]),
                           [f"типовой ответ администраторов на похожие вопросы (сходство {nb[0]['sim']:.2f})"],
                           "answer", "ru", P)

    # ------------------------------------------------------------------ LLM path
    def llm_user_message(self, P: dict) -> str:
        t, f, cls = P["t"], P["facts"], P["cls"]
        cand = [{"id": r, "text": self.book.describe(r, 300)} for r in P["candidates"]]
        pun = {r: self.typical_punishment(r) for r in P["candidates"] if self.typical_punishment(r)}
        similar = [{"player": n.get("text", "")[:280], "admin_reply": n.get("reply_canon", "")[:280],
                    "verdict": n.get("verdict"), "similarity": n["sim"]} for n in P["neighbours"][:3]]
        payload = {
            "kind": P["kind"],
            "target_lang": P["draft_lang"],
            "uz_script": P["uz_script"],
            "player_messages": t["player_text"][:1500],
            "reported_player_reply": t["reported_text"][:600] or None,
            "reported_nick": f["nickname"],
            "rule_field_as_typed": t["rule_refs_raw"],
            "images_count": t["images_count"],
            "extracted": {k: v for k, v in f.items() if not k.startswith("_") and k not in ("images_count",)},
            "signals": {
                "keyword_category": cls["category"], "category_candidates": cls["category_top"],
                "priority_guess": cls["priority"], "suggested_verdict": P["verdict"],
                "is_duplicate": P["is_dup"], "duplicate_reason": P["dup_reason"] or None,
                "is_spam": P["is_spam"], "unknown_rule_refs_typed": f["_unknown_rule_refs"],
                "rule_refs_not_matching_text": P["inconsistent_refs"],
            },
            "candidate_rules": cand,
            "typical_punishment_by_rule": pun,
            "similar_cases": similar,
        }
        return "Обращение для разбора:\n" + json.dumps(payload, ensure_ascii=False)

    def merge_llm(self, P: dict, fallback: dict, out: dict) -> dict | None:
        text = (out.get("draft_text") or "").strip()
        if len(text) < 10:
            return None
        cat = out.get("category") if out.get("category") in self.tax.by_id else fallback["category"]
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
        if P["is_spam"] and not P["is_dup"] and mode == "template":
            category = self.cmap.category_for("spam_ads", P["kind"]) or category
        concept = self.cmap.concept_for(category) or P["concept"]
        allowed = set(P["candidates"]) | set(self.concept_rules(concept, k=5)) | {r for r, _ in self.retriever.search(P["t"]["player_text"], k=10)}
        text, cited, problems = validate_citations(d["text"], d["cited"], self.book, allowed)
        reasons = list(d["reasons"])
        needs_human = d["needs_human"]
        if problems:
            reasons += problems[:2]
            needs_human = True
        if not text or len(text) < 10:
            fb = self.template_draft(P)
            text, cited = fb["text"], fb["cited"]
            needs_human = True
        promises = find_promises(text)
        if promises and not ("punish" in promises and any("наказан" in r for r in reasons) and len(promises) == 1):
            needs_human = True
            names = {"unban": "разбан", "refund": "возврат средств", "punish": "наказание игрока"}
            reasons.append("черновик содержит решение (" + ", ".join(names[p] for p in promises) + ") — подтверждает администратор")
        elif promises:
            needs_human = True
        for c in {concept, P["concept"]} & ALWAYS_HUMAN_CONCEPTS:
            needs_human = True
            reasons.append(ALWAYS_HUMAN_REASON[c])
        con = self.concepts.get(concept)
        needs_rules = bool(con and con.response in VIOLATION_RESPONSES) and d.get("suggested_verdict") in (None, "approved", "rejected", "escalate")
        if needs_rules and not cited:
            needs_human = True
            reasons.append("нет подходящей цитаты правил")
        if P["inconsistent_refs"]:
            reasons.append(f"игрок указал пункт {', '.join(P['inconsistent_refs'])}, но по тексту нарушение другое")
        if f["_unknown_rule_refs"]:
            reasons.append(f"указан несуществующий пункт правил: {', '.join(f['_unknown_rule_refs'][:3])}")
        if P["is_dup"] and P["dup_reason"] and not any("повтор" in r for r in reasons):
            reasons.append(P["dup_reason"])
        priority = self.clf._escalate(d["priority"], concept, P["cls"]["concept_scores"])
        if (P["is_spam"] or P["is_dup"]) and concept not in ALWAYS_HUMAN_CONCEPTS:
            priority = self.tax.priority_ids[-1] if self.tax.priority_ids else "p3"
        conf = round(float(d["confidence"]) * (0.85 if P["lang"] == "und" else 1.0), 2)
        if conf < AUTO_REPLY_MIN_CONFIDENCE and not needs_human:
            needs_human = True
            reasons.append(f"низкая уверенность ({conf:.2f})")
        if not reasons:
            reasons.append("типовой ответ, можно отправлять автоматически" if not needs_human else "требуется проверка администратора")
        seen, uniq = set(), []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        draft = {"lang": P["draft_lang"], "text": text, "cited": cited}
        meta = {"mode": mode, "suggested_verdict": d.get("suggested_verdict"), "concept": concept,
                "category_candidates": P["cls"]["category_top"], "rule_candidates": P["candidates"],
                "lang_confidence": P["lang_conf"], "is_duplicate": P["is_dup"], "duplicate_of": P["dup_of"]}
        if P["draft_lang"] == "kk":
            meta["text_cp1251"] = tpl_game(text)
        return {
            "lang": P["lang"],
            "category": category,
            "priority": priority,
            "is_spam_or_duplicate": bool(P["is_spam"] or P["is_dup"]),
            "extracted": {"nickname": f["nickname"], "rule_refs": f["rule_refs"], "has_evidence": f["has_evidence"],
                          "server": f["server"]},
            "draft_reply": draft,
            "needs_human": bool(needs_human),
            "confidence": conf,
            "reasons": uniq[:5],
            "meta": meta,
        }


def tpl_game(text: str) -> str:
    from .textnorm import to_game_encoding
    return to_game_encoding(text)

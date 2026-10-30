"""Category / priority / verdict prediction: keyword concepts + models trained on the train set."""

import json
import re
from pathlib import Path

from .concepts import ConceptMap, concept_scores
from .config import RESOURCES_DIR
from .taxonomy import Taxonomy
from .textnorm import letters_count

PRIO_ORDER = ["p0", "p1", "p2", "p3"]
# Short "content-free" reports: taxonomy puts «помогите», «ау», «привет» into offtopic.
OFFTOPIC_RE = re.compile(r"^\W*(\+|помогите|помоги|помогите пж|помогите пожалуйста|хелп|help|ау+|алло|привет|здравствуйте|"
                         r"админ\w*|адм[иі]н|салам|сәлем|salom|yordam|admin|<url>|\.+|\?+|пж|плиз)\W*$", re.I)


def _load(path: Path):
    if not path.exists():
        return None
    import joblib
    return joblib.load(path)


class Classifier:
    def __init__(self, taxonomy: Taxonomy, cmap: ConceptMap, models_dir: Path):
        self.tax = taxonomy
        self.cmap = cmap
        self.cat_model = _load(models_dir / "category.joblib")
        self.prio_model = _load(models_dir / "priority.joblib")
        self.verdict_model = _load(models_dir / "verdict.joblib")
        with open(RESOURCES_DIR / "priority_rules.json", encoding="utf-8") as f:
            self.prules = json.load(f)

    @staticmethod
    def model_input(kind: str, text: str, reported_text: str = "") -> str:
        return f"[{kind}] {text} [reported] {reported_text}".strip()

    def predict(self, kind: str, text: str, reported_text: str, rule_concepts: list[str], facts: dict | None = None) -> dict:
        facts = facts or {}
        scores = concept_scores(text, self.cmap.concepts, kind)
        for c in rule_concepts:  # what the player typed into the rule field is a (noisy) hint
            scores[c] = scores.get(c, 0.0) + 0.8
        # topics that always go to a human (account, payment, threats) win over anything else they co-occur with
        for urgent in ("threat", "account", "payment"):
            if scores.get(urgent, 0) >= 1:
                scores[urgent] = max(scores.values()) + 0.5
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        concept = ranked[0][0] if ranked else None
        kw_conf = 0.0
        if ranked:
            top = ranked[0][1]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            kw_conf = min(0.9, 0.45 + 0.15 * top - 0.1 * second)

        category = self.cmap.category_for(concept, kind) or self._no_concept_category(kind, text, facts)
        cat_conf = kw_conf if concept else (0.7 if category.endswith("offtopic") else 0.35)
        cat_top = [(category, round(cat_conf, 2))]

        if self.cat_model is not None:
            proba = self.cat_model.predict_proba([self.model_input(kind, text, reported_text)])[0]
            classes = list(self.cat_model.classes_)
            allowed = set(self.tax.ids_for_kind(kind))
            order = [i for i in proba.argsort()[::-1] if classes[i] in allowed][:3]
            ml_top = [(classes[i], round(float(proba[i]), 2)) for i in order]
            if ml_top:
                ml_cat, ml_p = ml_top[0]
                # The model learnt mostly from Russian texts: it decides when keywords found nothing,
                # but may overrule an explicit keyword match (often kk/uz/en) only when it is very sure.
                margin = 0.1 if not concept else max(0.25, 0.85 - cat_conf)
                if ml_cat == category:
                    cat_conf = max(cat_conf, ml_p)
                elif ml_p > cat_conf + margin and not (concept and ml_cat.endswith(("offtopic", "invalid"))):
                    category, cat_conf = ml_cat, ml_p
                    concept = self.cmap.concept_for(category) if self.cmap.concept_for(category) in scores else concept
                cat_top = ml_top

        priority, prio_conf = self.priority(category, kind, concept, scores, text, facts)
        if self.prio_model is not None:
            x = self.model_input(kind, text, reported_text) + f" [cat] {category}"
            proba = self.prio_model.predict_proba([x])[0]
            i = int(proba.argmax())
            if float(proba[i]) > max(prio_conf, 0.6):
                priority, prio_conf = str(self.prio_model.classes_[i]), float(proba[i])
        return {
            "category": category,
            "category_conf": round(cat_conf, 2),
            "category_top": cat_top,
            "concept": concept,
            "concept_scores": dict(ranked[:4]),
            "priority": self.escalate(priority, concept, scores, text),
            "priority_conf": round(prio_conf, 2),
        }

    def _no_concept_category(self, kind: str, text: str, facts: dict) -> str:
        if kind == "report":
            if OFFTOPIC_RE.match(text or "") or letters_count(text) < 4:
                return "report.offtopic" if "report.offtopic" in self.tax.by_id else self.tax.fallback_category(kind)
            if "report.question" in self.tax.by_id and re.search(r"\?|как |где |сколько|когда|можно ли|qanday|qayerda|қалай|қайда", text, re.I):
                return "report.question"
            return self.tax.fallback_category(kind)
        if "complaint.invalid" in self.tax.by_id and (not facts.get("nickname") or letters_count(text) < 15):
            return "complaint.invalid"
        return "complaint.other" if "complaint.other" in self.tax.by_id else self.tax.fallback_category(kind)

    def priority(self, category: str, kind: str, concept: str | None, scores: dict, text: str, facts: dict) -> tuple[str, float]:
        pr = self.prules
        low = (text or "").lower()
        prio = pr.get("category_default", {}).get(category)
        conf = 0.65 if prio else 0.4
        if not prio:
            prio = (self.cmap.concepts[concept].priority if concept in self.cmap.concepts else
                    ("p2" if kind == "complaint" else "p3"))
        if kind == "complaint" and facts.get("has_evidence") is False and category not in self.tax.always_human:
            prio = pr.get("complaint_without_evidence", "p3")
        if category in ("report.question",) and any(w in low for w in pr.get("rules_question_words", [])):
            prio = pr.get("rules_question_priority", "p2")
        if category.startswith("report.") and concept in pr.get("concept_overrides", {}):
            prio = min(prio, pr["concept_overrides"][concept], key=PRIO_ORDER.index)
        if any(w in low for w in pr.get("mass_bug_words", [])) and concept in ("bug", "stuck", None):
            prio = pr.get("mass_bug_priority", "p0")
        return prio, conf

    def escalate(self, priority: str, concept: str | None, scores: dict, text: str = "") -> str:
        valid = [p for p in PRIO_ORDER if p in self.tax.priority_ids] or PRIO_ORDER
        if scores.get("threat", 0) >= 1 and PRIO_ORDER.index(priority) > 0:
            priority = "p0"
        return priority if priority in valid else valid[min(len(valid) - 1, PRIO_ORDER.index(priority))]

    def predict_verdict(self, kind: str, text: str, reported_text: str, feats: dict) -> tuple[str | None, float, dict]:
        """Admin verdict model (trained on real verdicts from the train set)."""
        if self.verdict_model is None or kind != "complaint":
            return None, 0.0, {}
        x = self.model_input(kind, text, reported_text) + verdict_feature_tokens(feats)
        proba = self.verdict_model.predict_proba([x])[0]
        classes = [str(c) for c in self.verdict_model.classes_]
        i = int(proba.argmax())
        return classes[i], float(proba[i]), {c: round(float(p), 3) for c, p in zip(classes, proba)}


def verdict_feature_tokens(f: dict) -> str:
    """Structured features rendered as pseudo-tokens so one text pipeline can learn from them."""
    imgs = f.get("images_count", 0) or 0
    toks = [
        f"__img{min(imgs, 3)}__",
        "__nick_yes__" if f.get("nickname") else "__nick_no__",
        "__rule_yes__" if f.get("rule_refs") else "__rule_no__",
        "__badrule__" if f.get("_unknown_rule_refs") else "",
        "__defense__" if f.get("has_defense") else "",
        "__url__" if f.get("has_url") else "",
        "__profanity__" if f.get("profanity") else "",
        "__short__" if f.get("_text_letters", 0) < 25 else "",
    ]
    return " " + " ".join(t for t in toks if t)

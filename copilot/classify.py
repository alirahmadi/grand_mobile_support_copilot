"""Category / priority / verdict prediction: keyword concepts + optional trained models."""

from pathlib import Path

from .concepts import ConceptMap, concept_scores
from .taxonomy import Taxonomy

PRIO_ORDER = ["p0", "p1", "p2", "p3"]


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

    @staticmethod
    def model_input(kind: str, text: str, reported_text: str = "") -> str:
        return f"[{kind}] {text} [reported] {reported_text}".strip()

    def predict(self, kind: str, text: str, reported_text: str, rule_concepts: list[str]) -> dict:
        scores = concept_scores(text, self.cmap.concepts, kind)
        for c in rule_concepts:  # what the player typed into the rule field is a (noisy) hint
            scores[c] = scores.get(c, 0.0) + 0.8
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        concept = ranked[0][0] if ranked else None
        kw_conf = 0.0
        if ranked:
            top = ranked[0][1]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            kw_conf = min(0.9, 0.45 + 0.15 * top - 0.1 * second)

        category = self.cmap.category_for(concept, kind) or self.tax.fallback_category(kind)
        cat_conf = kw_conf if concept else 0.3
        cat_top = [(category, round(cat_conf, 2))]

        if self.cat_model is not None:
            x = self.model_input(kind, text, reported_text)
            proba = self.cat_model.predict_proba([x])[0]
            classes = list(self.cat_model.classes_)
            order = proba.argsort()[::-1][:3]
            ml_top = [(classes[i], round(float(proba[i]), 2)) for i in order]
            allowed = set(self.tax.ids_for_kind(kind))
            ml_top = [c for c in ml_top if c[0] in allowed] or ml_top
            ml_cat, ml_p = ml_top[0]
            # trust the model unless the keyword evidence is clearly stronger
            if ml_p >= cat_conf or not concept:
                category, cat_conf = ml_cat, max(ml_p, cat_conf * 0.9 if ml_cat == category else ml_p)
            cat_top = ml_top
            concept = self.cmap.concept_for(category) or concept

        priority, prio_conf = self._priority(category, concept, text, kind)
        if self.prio_model is not None:
            x = self.model_input(kind, text, reported_text) + f" [cat] {category}"
            proba = self.prio_model.predict_proba([x])[0]
            classes = list(self.prio_model.classes_)
            i = int(proba.argmax())
            if float(proba[i]) > prio_conf:
                priority, prio_conf = classes[i], float(proba[i])
        # hard escalation rules always win
        priority = self._escalate(priority, concept, scores)
        return {
            "category": category,
            "category_conf": round(cat_conf, 2),
            "category_top": cat_top,
            "concept": concept,
            "concept_scores": dict(ranked[:4]),
            "priority": priority,
            "priority_conf": round(prio_conf, 2),
        }

    def _priority(self, category: str, concept: str | None, text: str, kind: str) -> tuple[str, float]:
        cat = self.tax.by_id.get(category)
        if cat and cat.default_priority and cat.default_priority.lower() in PRIO_ORDER:
            return cat.default_priority.lower(), 0.6
        if concept and concept in self.cmap.concepts:
            return self.cmap.concepts[concept].priority, 0.55
        return ("p2" if kind == "complaint" else "p3"), 0.4

    def _escalate(self, priority: str, concept: str | None, scores: dict) -> str:
        valid = [p for p in PRIO_ORDER if p in self.tax.priority_ids] or PRIO_ORDER

        def at_least(p):
            return p if PRIO_ORDER.index(p) < PRIO_ORDER.index(priority) else priority
        if scores.get("threat", 0) >= 1:
            priority = at_least("p0")
        elif scores.get("account", 0) >= 1.5:
            priority = at_least("p1")
        return priority if priority in valid else valid[min(len(valid) - 1, PRIO_ORDER.index(priority))]

    def predict_verdict(self, kind: str, text: str, reported_text: str, feats: dict) -> tuple[str | None, float]:
        """Admin verdict model (trained on real verdicts from the train set)."""
        if self.verdict_model is None or kind != "complaint":
            return None, 0.0
        x = self.model_input(kind, text, reported_text) + verdict_feature_tokens(feats)
        proba = self.verdict_model.predict_proba([x])[0]
        i = int(proba.argmax())
        return str(self.verdict_model.classes_[i]), float(proba[i])


def verdict_feature_tokens(f: dict) -> str:
    """Structured features rendered as pseudo-tokens so one text pipeline can learn from them."""
    toks = [
        "__img_yes__" if f.get("images_count", 0) > 0 else "__img_no__",
        "__nick_yes__" if f.get("nickname") else "__nick_no__",
        "__rule_yes__" if f.get("rule_refs") else "__rule_no__",
        "__badrule__" if f.get("_unknown_rule_refs") else "",
        "__defense__" if f.get("has_defense") else "",
        "__short__" if f.get("_text_letters", 0) < 25 else "",
    ]
    return " " + " ".join(t for t in toks if t)

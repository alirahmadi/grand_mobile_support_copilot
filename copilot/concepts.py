"""Multilingual keyword 'concepts' and their mapping onto the official taxonomy."""

import json
import re
from dataclasses import dataclass, field

from .config import RESOURCES_DIR
from .taxonomy import Taxonomy
from .textnorm import tokens


@dataclass
class Concept:
    name: str
    kind: str
    priority: str
    human: bool
    response: str
    match: list
    rule_hints: list
    exact: set = field(default_factory=set)
    prefixes: list = field(default_factory=list)
    phrases: list = field(default_factory=list)

    def score(self, toks: list[str], low_text: str) -> float:
        s = 0.0
        for t in toks:
            if t in self.exact:
                s += 1.0
            else:
                for p in self.prefixes:
                    if t.startswith(p):
                        s += 1.0
                        break
        for ph in self.phrases:
            if ph in low_text:
                s += 1.5
        return s


def load_concepts(path=None) -> dict[str, Concept]:
    path = path or RESOURCES_DIR / "concepts.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for name, c in raw.items():
        if name.startswith("_"):
            continue
        con = Concept(name=name, kind=c.get("kind", "any"), priority=c.get("priority", "p2"),
                      human=bool(c.get("human")), response=c.get("response", "generic"),
                      match=[m.lower() for m in c.get("match", [])], rule_hints=c.get("rule_hints", []))
        for kw in c.get("keywords", []):
            kw = kw.lower().replace("ё", "е")
            if " " in kw or "." in kw or "-" in kw:
                con.phrases.append(kw.rstrip("*"))
            elif kw.endswith("*"):
                con.prefixes.append(kw[:-1])
            else:
                con.exact.add(kw)
        out[name] = con
    return out


def concept_scores(text: str, concepts: dict[str, Concept], kind: str | None = None) -> dict[str, float]:
    low = (text or "").lower().replace("ё", "е")
    toks = tokens(low)
    res = {}
    for name, c in concepts.items():
        if kind and c.kind not in ("any", kind):
            continue
        s = c.score(toks, low)
        if s > 0:
            res[name] = s
    # 'question' words are weak evidence: they only win when nothing else matched.
    if "question" in res and len(res) > 1:
        res["question"] *= 0.3
    return res


class ConceptMap:
    """Bidirectional mapping concept <-> taxonomy category ids."""

    def __init__(self, taxonomy: Taxonomy, concepts: dict[str, Concept]):
        self.taxonomy = taxonomy
        self.concepts = concepts
        self.cat_to_concept: dict[str, str] = {}
        self.concept_to_cats: dict[str, list[str]] = {}
        for cat in taxonomy.categories:
            best, best_s = None, 0.0
            hay = cat.text.lower()
            id_low = cat.id.lower()
            for name, c in concepts.items():
                s = 0.0
                for m in c.match:
                    if re.search(rf"(^|[._\s-]){re.escape(m)}", id_low):
                        s += 3.0
                    elif m in hay:
                        s += 1.0
                # also look at the category description through the keyword lists
                s += 0.3 * min(5.0, c.score(tokens(hay), hay))
                if s > best_s:
                    best, best_s = name, s
            if best and best_s >= 1.0:
                self.cat_to_concept[cat.id] = best
                self.concept_to_cats.setdefault(best, []).append(cat.id)

    def category_for(self, concept: str | None, kind: str) -> str | None:
        if not concept:
            return None
        cats = self.concept_to_cats.get(concept, [])
        same_kind = [c for c in cats if self.taxonomy.by_id[c].kind == kind]
        neutral = [c for c in cats if self.taxonomy.by_id[c].kind is None]
        for group in (same_kind, neutral, cats):
            if group:
                return group[0]
        return None

    def concept_for(self, category: str) -> str | None:
        return self.cat_to_concept.get(category)

    def report(self) -> str:
        lines = []
        for cat in self.taxonomy.categories:
            lines.append(f"  {cat.id:35s} -> {self.cat_to_concept.get(cat.id, '(не сопоставлено)')}")
        return "\n".join(lines)

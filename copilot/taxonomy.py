"""taxonomy.json: categories (with their rule references), priorities and the always-human list."""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

_ID_KEYS = ("code", "id", "key", "slug", "category")
_NAME_KEYS = ("title", "name", "label", "name_ru")
_DESC_KEYS = ("description", "desc", "definition", "criteria", "rule", "when")


@dataclass
class Category:
    id: str
    name: str = ""
    description: str = ""
    examples: list = field(default_factory=list)
    rules: list = field(default_factory=list)
    kind_field: str | None = None

    @property
    def kind(self) -> str | None:
        if self.kind_field in ("complaint", "report"):
            return self.kind_field
        head = self.id.split(".")[0].lower()
        if head.startswith("complaint"):
            return "complaint"
        if head.startswith("report"):
            return "report"
        return None

    @property
    def text(self) -> str:
        return " ".join([self.id.replace(".", " ").replace("_", " "), self.name, self.description,
                         " ".join(map(str, self.examples))])


@dataclass
class Priority:
    id: str
    name: str = ""
    criteria: str = ""


def _s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return "; ".join(_s(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_s(x)}" for k, x in v.items())
    return unicodedata.normalize("NFC", str(v))


def _first(d: dict, keys) -> str:
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return _s(d[k])
    return ""


class Taxonomy:
    def __init__(self, categories: list[Category], priorities: list[Priority], always_human: list[str], raw):
        self.categories = categories
        self.by_id = {c.id: c for c in categories}
        self.priorities = priorities or [Priority(p) for p in ("p0", "p1", "p2", "p3")]
        self.priority_ids = [p.id for p in self.priorities]
        self.always_human = set(always_human)
        self.raw = raw

    @classmethod
    def load(cls, path: Path) -> "Taxonomy":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        cats_node = raw.get("categories", []) if isinstance(raw, dict) else raw
        categories = []
        if isinstance(cats_node, dict):  # {"complaint.insult": {...}} or {"complaint.insult": "description"}
            cats_node = [{"code": k, **(v if isinstance(v, dict) else {"description": v})} for k, v in cats_node.items()]
        for c in cats_node or []:
            if isinstance(c, str):
                categories.append(Category(id=c))
                continue
            categories.append(Category(
                id=_first(c, _ID_KEYS), name=_first(c, _NAME_KEYS), description=_first(c, _DESC_KEYS),
                examples=[_s(x) for x in c.get("examples") or []], rules=[_s(x) for x in c.get("rules") or []],
                kind_field=c.get("kind")))
        prios = []
        pnode = raw.get("priorities", []) if isinstance(raw, dict) else []
        if isinstance(pnode, dict):
            pnode = [{"code": k, **(v if isinstance(v, dict) else {"rule": v})} for k, v in pnode.items()]
        for p in pnode:
            if isinstance(p, dict):
                prios.append(Priority(_first(p, ("code", "id", "level")), _first(p, _NAME_KEYS), _first(p, _DESC_KEYS)))
        prios = sorted([p for p in prios if re.fullmatch(r"p\d", p.id.lower())], key=lambda p: p.id)
        always = raw.get("needs_human_always", []) if isinstance(raw, dict) else []
        return cls(categories, prios, always, raw)

    def ids_for_kind(self, kind: str) -> list[str]:
        ids = [c.id for c in self.categories if c.kind in (kind, None)]
        return ids or [c.id for c in self.categories]

    def fallback_category(self, kind: str) -> str:
        cands = self.ids_for_kind(kind)
        for c in cands:
            if re.search(r"other|misc|general|проч|друг", c, re.I):
                return c
        return cands[-1] if cands else "other"

    def as_prompt_block(self) -> str:
        lines = ["Категории (код — название; пункты правил категории; примеры):"]
        for c in self.categories:
            line = f"- {c.id} [{c.kind or 'any'}] — {c.name}"
            if c.description:
                line += f": {c.description}"
            if c.rules:
                line += f"; пункты: {', '.join(c.rules)}"
            if c.examples:
                line += f"; примеры: {' | '.join(c.examples)}"
            lines.append(line)
        lines.append("Приоритеты:")
        for p in self.priorities:
            lines.append(f"- {p.id} {p.name}: {p.criteria}")
        if self.always_human:
            lines.append("Всегда нужен человек (needs_human=true): " + ", ".join(sorted(self.always_human)))
        return "\n".join(lines)

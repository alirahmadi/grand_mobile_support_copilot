"""Loading taxonomy.json (format-agnostic): categories, priorities and their criteria."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_ID_KEYS = ("id", "code", "key", "slug", "category", "name_en")
_NAME_KEYS = ("name", "title", "label", "name_ru", "название")
_DESC_KEYS = ("description", "desc", "definition", "criteria", "when", "описание", "критерии")
_EX_KEYS = ("examples", "example", "samples", "примеры")
_PRIO_KEYS = ("priority", "default_priority", "prio")


@dataclass
class Category:
    id: str
    name: str = ""
    description: str = ""
    examples: list = field(default_factory=list)
    default_priority: str = ""
    needs_rules: bool | None = None

    @property
    def kind(self) -> str | None:
        head = self.id.split(".")[0].lower()
        if head.startswith("complaint") or head in ("жалоба", "complaints"):
            return "complaint"
        if head.startswith("report") or head in ("репорт", "reports"):
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
    return str(v)


def _first(d: dict, keys) -> str:
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return _s(d[k])
    return ""


class Taxonomy:
    def __init__(self, categories: list[Category], priorities: list[Priority], raw: dict | list):
        self.categories = categories
        self.by_id = {c.id: c for c in categories}
        self.priorities = priorities or [Priority(p) for p in ("p0", "p1", "p2", "p3")]
        self.priority_ids = [p.id for p in self.priorities]
        self.raw = raw

    @classmethod
    def load(cls, path: Path) -> "Taxonomy":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        cats_node, prio_node = raw, None
        if isinstance(raw, dict):
            cats_node = next((raw[k] for k in ("categories", "category", "taxonomy", "категории") if k in raw), None)
            prio_node = next((raw[k] for k in ("priorities", "priority", "priority_levels", "приоритеты") if k in raw), None)
            if cats_node is None:
                cats_node = {k: v for k, v in raw.items() if k not in ("priorities", "priority", "version", "meta")}
        categories: list[Category] = []
        cls._walk_cats(cats_node, "", categories)
        priorities = cls._parse_priorities(prio_node)
        # categories may carry their own priority hints
        return cls(categories, priorities, raw)

    @classmethod
    def _walk_cats(cls, node, prefix: str, out: list):
        if isinstance(node, list):
            for x in node:
                if isinstance(x, str):
                    out.append(Category(id=f"{prefix}.{x}" if prefix and "." not in x else x))
                elif isinstance(x, dict):
                    cid = _first(x, _ID_KEYS) or _first(x, _NAME_KEYS)
                    sub = next((x[k] for k in ("subcategories", "children", "items", "categories") if k in x), None)
                    full = cid if (not prefix or "." in cid or cid.startswith(prefix)) else f"{prefix}.{cid}"
                    if sub:
                        cls._walk_cats(sub, full, out)
                    else:
                        out.append(Category(
                            id=full, name=_first(x, _NAME_KEYS), description=_first(x, _DESC_KEYS),
                            examples=x.get("examples") or x.get("example") or [],
                            default_priority=_first(x, _PRIO_KEYS),
                            needs_rules=x.get("needs_rules", x.get("requires_rules")),
                        ))
            return
        if isinstance(node, dict):
            for k, v in node.items():
                full = f"{prefix}.{k}" if prefix else k
                if isinstance(v, str):
                    out.append(Category(id=full, description=v))
                elif isinstance(v, dict) and (set(v) & set(_DESC_KEYS + _NAME_KEYS + _EX_KEYS)):
                    out.append(Category(
                        id=_first(v, ("id", "code")) or full, name=_first(v, _NAME_KEYS),
                        description=_first(v, _DESC_KEYS), examples=v.get("examples") or [],
                        default_priority=_first(v, _PRIO_KEYS),
                        needs_rules=v.get("needs_rules", v.get("requires_rules")),
                    ))
                else:
                    cls._walk_cats(v, full, out)

    @staticmethod
    def _parse_priorities(node) -> list[Priority]:
        out = []
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, dict):
                    out.append(Priority(k, _first(v, _NAME_KEYS), _first(v, _DESC_KEYS) or _s(v)))
                else:
                    out.append(Priority(k, "", _s(v)))
        elif isinstance(node, list):
            for x in node:
                if isinstance(x, dict):
                    pid = _first(x, ("id", "code", "key", "level", "priority"))
                    out.append(Priority(pid, _first(x, _NAME_KEYS), _first(x, _DESC_KEYS) or _s(x)))
                elif isinstance(x, str):
                    out.append(Priority(x))
        out = [p for p in out if re.fullmatch(r"p\d", p.id.lower())]
        return sorted(out, key=lambda p: p.id)

    def ids_for_kind(self, kind: str) -> list[str]:
        ids = [c.id for c in self.categories if c.kind in (kind, None)]
        return ids or [c.id for c in self.categories]

    def fallback_category(self, kind: str) -> str:
        """The 'other' bucket for this kind of ticket."""
        cands = self.ids_for_kind(kind)
        for c in cands:
            if re.search(r"other|misc|general|проч|друг", c, re.I):
                return c
        return cands[-1] if cands else "other"

    def as_prompt_block(self) -> str:
        lines = ["Категории:"]
        for c in self.categories:
            line = f"- {c.id}"
            if c.name:
                line += f" — {c.name}"
            if c.description:
                line += f": {c.description}"
            if c.examples:
                line += f" Примеры: {_s(c.examples)[:200]}"
            if c.default_priority:
                line += f" (обычный приоритет {c.default_priority})"
            lines.append(line)
        lines.append("Приоритеты:")
        for p in self.priorities:
            lines.append(f"- {p.id} {p.name}: {p.criteria}".rstrip(": "))
        return "\n".join(lines)

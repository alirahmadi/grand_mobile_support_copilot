"""Loading rules.json (format-agnostic), rule-reference parsing and validation."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .textnorm import normalize

RULE_ID_RE = re.compile(r"^\d{1,2}(?:\.\d{1,3}){0,3}$")
# A rule reference inside free text: 2.1 / 2.1.3 / п.2.1 / пункт 2,1
REF_IN_TEXT_RE = re.compile(r"(?<![\d.,])(\d{1,2}(?:[.,]\d{1,3}){1,3})(?![\d])")
_ID_KEYS = ("id", "rule_id", "ruleId", "code", "number", "num", "key", "point", "punkt", "пункт")
_TEXT_KEYS = ("text", "rule", "description", "content", "body", "desc", "value", "текст")
_TITLE_KEYS = ("title", "name", "header", "caption", "заголовок", "название")
_ALIAS_KEYS = ("short", "abbr", "abbreviation", "alias", "aliases", "tag", "tags", "short_name", "code_name")
_PUNISH_KEYS = ("punishment", "punishments", "penalty", "sanction", "наказание")
_CHILD_KEYS = ("items", "rules", "points", "children", "subrules", "sections", "clauses", "пункты")


@dataclass
class Rule:
    id: str
    text: str
    title: str = ""
    section_id: str = ""
    section_title: str = ""
    punishment: str = ""
    aliases: list = field(default_factory=list)

    @property
    def full_text(self) -> str:
        parts, seen = [], set()
        for p in (self.section_title, self.title, self.text):
            p = (p or "").strip().rstrip(".")
            if p and p.lower() not in seen:
                seen.add(p.lower())
                parts.append(p)
        s = ". ".join(parts)
        if self.punishment:
            s += f". Наказание: {self.punishment.strip().rstrip('.')}"
        return s

    @property
    def short(self) -> str:
        """Short human description used inside drafts."""
        s = self.title if len(self.title) >= 15 else (self.text or self.title)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.split(r"(?<=[.!?;])\s", s)[0].rstrip(" .;:")
        return s[:160]


def _as_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "; ".join(_as_str(x) for x in v if x is not None)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_as_str(x)}" for k, x in v.items())
    return str(v)


def _first(d: dict, keys) -> str:
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return _as_str(d[k])
    return ""


def _clean_id(raw: str) -> str:
    s = str(raw).strip().lower()
    s = re.sub(r"^(п\.|пункт|rule|правило|§)\s*", "", s)
    s = s.replace(",", ".").strip(" .")
    return s


class RuleBook:
    def __init__(self, rules: list[Rule]):
        self.rules = rules
        self.by_id = {r.id: r for r in rules}
        self.alias_to_ids: dict[str, list[str]] = {}
        for r in rules:
            for a in r.aliases:
                self.add_alias(a, r.id)

    # ---------- loading ----------
    @classmethod
    def load(cls, path: Path) -> "RuleBook":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        rules: list[Rule] = []
        cls._walk(raw, rules, section=("", ""), key_hint=None)
        # de-duplicate (keep the most informative entry per id)
        best: dict[str, Rule] = {}
        for r in rules:
            if r.id not in best or len(r.full_text) > len(best[r.id].full_text):
                best[r.id] = r
        ordered = sorted(best.values(), key=lambda r: [int(x) if x.isdigit() else 0 for x in r.id.split(".")])
        return cls(ordered)

    @classmethod
    def _walk(cls, node, out: list, section: tuple, key_hint):
        if isinstance(node, list):
            for x in node:
                cls._walk(x, out, section, None)
            return
        if isinstance(node, str):
            if key_hint is not None and RULE_ID_RE.match(_clean_id(key_hint)):
                rid = _clean_id(key_hint)
                out.append(Rule(id=rid, text=node, section_id=rid.split(".")[0], section_title=section[1]))
            return
        if not isinstance(node, dict):
            return
        rid = _first(node, _ID_KEYS)
        if not rid and key_hint is not None:
            rid = key_hint
        rid = _clean_id(rid) if rid else ""
        is_rule = bool(rid) and bool(RULE_ID_RE.match(rid))
        title = _first(node, _TITLE_KEYS)
        text = _first(node, _TEXT_KEYS)
        children_present = any(k in node and isinstance(node[k], (list, dict)) for k in _CHILD_KEYS)
        new_section = section
        if is_rule:
            aliases_raw = None
            for k in _ALIAS_KEYS:
                if k in node and node[k]:
                    aliases_raw = node[k]
                    break
            aliases = []
            if isinstance(aliases_raw, list):
                aliases = [str(a) for a in aliases_raw]
            elif aliases_raw:
                aliases = [a.strip() for a in re.split(r"[,/;]", str(aliases_raw)) if a.strip()]
            if "." not in rid:
                new_section = (rid, title or text)
            out.append(Rule(
                id=rid, text=text or title, title=title if text else "",
                section_id=rid.split(".")[0],
                section_title=new_section[1] if "." in rid else (title or text),
                punishment=_first(node, _PUNISH_KEYS), aliases=aliases,
            ))
        known = set(_TEXT_KEYS + _TITLE_KEYS + _ID_KEYS + _ALIAS_KEYS + _PUNISH_KEYS)
        for k, v in node.items():
            if isinstance(v, dict):
                cls._walk(v, out, new_section, k)
            elif isinstance(v, list) and k not in _ALIAS_KEYS:
                cls._walk(v, out, new_section, None)
            elif isinstance(v, str) and k not in known and not children_present and RULE_ID_RE.match(_clean_id(k)):
                # {"2.1": "text"} style
                cls._walk(v, out, new_section, k)

    # ---------- aliases / references ----------
    def add_alias(self, alias: str, rule_id: str):
        a = alias.strip().lower()
        if not a or rule_id not in self.by_id:
            return
        ids = self.alias_to_ids.setdefault(a, [])
        if rule_id not in ids:
            ids.append(rule_id)

    def exists(self, rid: str) -> bool:
        return _clean_id(rid) in self.by_id

    def resolve(self, rid: str) -> str | None:
        """Exact id, or the closest existing parent ("2.1.4" -> "2.1")."""
        rid = _clean_id(rid)
        while rid:
            if rid in self.by_id:
                return rid
            if "." not in rid:
                return None
            rid = rid.rsplit(".", 1)[0]
        return None

    def parse_refs(self, raw: str) -> tuple[list[str], list[str]]:
        """Parse what a player typed into rule ids. Returns (valid_ids, unknown_tokens)."""
        if not raw:
            return [], []
        text = normalize(str(raw))
        valid, unknown = [], []
        for m in REF_IN_TEXT_RE.finditer(text):
            tok = m.group(1).replace(",", ".")
            rid = self.resolve(tok)
            if rid:
                if rid not in valid:
                    valid.append(rid)
            else:
                unknown.append(tok)
        for w in re.findall(r"[^\W\d_]{2,10}", text):
            ids = self.alias_to_ids.get(w.lower())
            if ids and len(ids) <= 2:
                for rid in ids:
                    if rid not in valid:
                        valid.append(rid)
        # a bare section number ("2") typed into the rule field
        if not valid and re.fullmatch(r"\s*\d{1,2}\s*", text):
            rid = self.resolve(text.strip())
            if rid:
                valid.append(rid)
            else:
                unknown.append(text.strip())
        return valid, unknown

    def refs_in_text(self, text: str) -> list[str]:
        return [m.group(1).replace(",", ".") for m in REF_IN_TEXT_RE.finditer(text or "")]

    def describe(self, rid: str, max_len: int = 400) -> str:
        r = self.by_id.get(rid)
        if not r:
            return ""
        s = r.full_text
        return s if len(s) <= max_len else s[:max_len] + "…"

    def as_prompt_block(self, max_chars: int = 60000) -> str:
        lines = []
        for r in self.rules:
            line = f"[{r.id}] {r.full_text}"
            if r.aliases:
                line += f" (сокращения: {', '.join(r.aliases)})"
            lines.append(line)
        s = "\n".join(lines)
        return s[:max_chars]

"""rules.json loading, rule references ("2.1 ОП", "1.2 GZ", "ЖБ 1.2.7") parsing and validation.

Grand Mobile rules come as several documents (general rules, complaint rules, green zone, ...),
and point numbers repeat between documents. Citation ids therefore carry the document code,
following the convention of taxonomy.json:
    general rules          "2.1"            (bare number)
    other documents        "GZ 1.2", "ЖБ 1.2.7"
    a whole document       "GZ"
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .textnorm import normalize

PRIMARY = "ОП"
# (keywords in the document title, citation code, aliases players/admins use)
DOC_CODES = [
    (("общие правила",), "ОП", ("оп", "op", "оп.", "ор", "or")),
    (("жалоб",), "ЖБ", ("жб", "jb")),
    (("зелён", "зелен", "green"), "GZ", ("gz", "гз", "зз")),
    (("лидер",), "ПЛЗ", ("плз", "оплз", "лз")),
    (("государственных организаций",), "ПГО", ("пго", "pgo", "госс", "гос")),
    (("государственной волны", "департамент"), "ГВ", ("гв", "gw")),
    (("семейных мероприятий",), "FW", ("fw", "фв")),
    (("опг",), "ОПГ", ("опг", "opg")),
    (("войны за бизнес",), "BW", ("bw", "бв")),
    (("фургон",), "БЗФ", ("бзф", "fura", "фура")),
    (("rp термин", "термины"), "RP", ()),
]
# Words after a bare number that point to a specific document: "п. 1.2.7 правил подачи жалоб".
CONTEXT_HINTS = [
    (re.compile(r"^\s*(правил[а-я]*\s+)?(подачи\s+)?жалоб|^\s*\(?жб\)?", re.I), "ЖБ"),
    (re.compile(r"^\s*(правил[а-я]*\s+)?(нахождения\s+в\s+)?зел[её]н|^\s*\(?(gz|гз)\)?", re.I), "GZ"),
]
# ... and words before it: "шағым беру ережесінің 1.2.2-тармағы", "shikoyat berish qoidalarining 1.2.2-bandi"
CONTEXT_BEFORE = [
    (re.compile(r"(правил[а-я]*\s+(подачи\s+)?жалоб|шағым беру ережесінің|shikoyat berish qoidalarining|"
                r"шикоят бериш қоидаларининг|complaint rules,?)\s*(п\.|пункт[а-я]*)?\s*$", re.I), "ЖБ"),
]
NUM = r"\d{1,2}(?:\.\d{1,3}){1,3}"
_NUM_RE = re.compile(rf"(?<![\d.,])(\d{{1,2}}(?:\.\d{{1,3}}){{1,3}}|\d{{1,2}},\d{{1,2}})(?![\d])(?![,.]\d)")
_CODE_NUM_RE = re.compile(rf"(?<![\w])([A-Za-zА-Яа-яЁё]{{2,4}})\.?\s?({NUM})(?![\d])")
_TAIL_RE = re.compile(r"^\s?([A-Za-zА-Яа-яЁё]{2,4})(?![A-Za-zА-Яа-яЁё])")
_WORD_RE = re.compile(r"(?<![\w])([A-Za-zА-Яа-яЁё]{2,4})(?![\w])")
# kept for backwards compatibility with older imports
REF_IN_TEXT_RE = _NUM_RE


@dataclass
class Rule:
    id: str
    text: str
    doc: str = PRIMARY
    doc_title: str = ""
    ref: str = ""
    punishment: str = ""
    aliases: list = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return f"{self.doc_title}. {self.text}" if self.doc_title else self.text

    @property
    def short(self) -> str:
        s = re.sub(r"\s+", " ", self.text.split(" | ")[0]).strip()
        s = re.split(r"(?<=[.!?;])\s", s)[0].rstrip(" .;:")
        return s[:180]

    @property
    def label(self) -> str:
        """How admins write the reference in replies: "2.1 ОП", "1.2 GZ", "1.2.7 ЖБ"."""
        if self.ref:
            return f"{self.ref} {self.doc}"
        return self.doc


def _doc_code(title: str, used: set) -> tuple[str, tuple]:
    low = title.lower()
    for keys, code, aliases in DOC_CODES:
        if any(k in low for k in keys) and code not in used:
            return code, aliases
    initials = "".join(w[0] for w in re.findall(r"[А-Яа-яA-Za-z]{3,}", title)).upper()[:4] or "DOC"
    return initials, ()


def _punishment(text: str) -> str:
    """Everything after the first " | " is the sanctions table of the point."""
    parts = [p.strip().rstrip(".") for p in text.split(" | ")[1:]]
    return "; ".join(p for p in parts if p)


class RuleBook:
    def __init__(self, rules: list[Rule], docs: dict[str, str] | None = None, alias_to_doc: dict | None = None):
        self.rules = rules
        self.by_id = {r.id: r for r in rules}
        self.docs = docs or {PRIMARY: ""}
        self.alias_to_doc = alias_to_doc or {}
        self.alias_to_ids: dict[str, list[str]] = {}  # free-form aliases mined from data ("dm" -> ...)

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: Path) -> "RuleBook":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        docs_raw = raw.get("rules") if isinstance(raw, dict) else raw
        if isinstance(docs_raw, list) and docs_raw and isinstance(docs_raw[0], dict) and "points" in docs_raw[0]:
            return cls._load_documents(docs_raw)
        return cls._load_flat(raw)

    @classmethod
    def _load_documents(cls, docs_raw: list) -> "RuleBook":
        rules, docs, alias_to_doc, used = [], {}, {}, set()
        for d in docs_raw:
            title = unicodedata.normalize("NFC", str(d.get("category") or d.get("title") or ""))
            code, aliases = _doc_code(title, used)
            used.add(code)
            docs[code] = title
            alias_to_doc[code.lower()] = code
            for a in aliases:
                alias_to_doc.setdefault(a.lower(), code)
            intro = re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(d.get("text") or "")))[:400]
            rules.append(Rule(id=code, text=intro or title, doc=code, doc_title=title))
            for p in d.get("points") or []:
                ref = str(p.get("ref") or "").strip().rstrip(".")
                if not ref:
                    continue
                text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(p.get("text") or ""))).strip()
                rid = ref if code == PRIMARY else f"{code} {ref}"
                rules.append(Rule(id=rid, text=text, doc=code, doc_title=title, ref=ref, punishment=_punishment(text)))
        return cls(rules, docs, alias_to_doc)

    @classmethod
    def _load_flat(cls, raw) -> "RuleBook":
        """Fallback for a simple {"2.1": "text"} / [{"id","text"}] structure: everything is the primary document."""
        rules = []

        def walk(node, key=None):
            if isinstance(node, dict):
                rid = str(node.get("id") or node.get("ref") or key or "").strip()
                text = node.get("text") or node.get("title") or ""
                if re.fullmatch(r"\d{1,2}(\.\d{1,3}){0,3}", rid) and isinstance(text, str):
                    rules.append(Rule(id=rid, text=text, ref=rid, punishment=str(node.get("punishment") or "")))
                for k, v in node.items():
                    if isinstance(v, (dict, list)):
                        walk(v, k)
                    elif isinstance(v, str) and re.fullmatch(r"\d{1,2}(\.\d{1,3}){1,3}", k):
                        rules.append(Rule(id=k, text=v, ref=k))
            elif isinstance(node, list):
                for x in node:
                    walk(x)
        walk(raw)
        uniq = {r.id: r for r in rules}
        return cls(list(uniq.values()), {PRIMARY: "Правила"}, {"оп": PRIMARY, "op": PRIMARY})

    # ---------------------------------------------------------------- ids
    def resolve(self, num: str, doc: str = PRIMARY) -> str | None:
        """Exact point or the closest existing parent within the same document."""
        num = num.replace(",", ".").strip(" .")
        while num:
            rid = num if doc == PRIMARY else f"{doc} {num}"
            if rid in self.by_id:
                return rid
            if "." not in num:
                return None
            num = num.rsplit(".", 1)[0]
        return None

    def exists(self, rid: str) -> bool:
        return rid in self.by_id

    def add_alias(self, alias: str, rule_id: str):
        a = alias.strip().lower()
        if a and rule_id in self.by_id:
            ids = self.alias_to_ids.setdefault(a, [])
            if rule_id not in ids:
                ids.append(rule_id)

    def _doc_for_alias(self, word: str) -> str | None:
        return self.alias_to_doc.get(word.lower().strip("."))

    # ---------------------------------------------------------------- parsing
    def find_refs(self, text: str) -> list[dict]:
        """All rule references in a text: [{start, end, raw, id (None if it does not exist)}]."""
        text = text or ""
        out, taken = [], []

        def free(a, b):
            return all(b <= s or a >= e for s, e in taken)

        for m in _CODE_NUM_RE.finditer(text):  # "ЖБ 1.2.7", "GZ 1.2"
            doc = self._doc_for_alias(m.group(1))
            if doc and doc != PRIMARY:
                out.append({"start": m.start(), "end": m.end(), "raw": m.group(0), "id": self.resolve(m.group(2), doc)})
                taken.append((m.start(), m.end()))
        for m in _NUM_RE.finditer(text):  # "2.1", "2.1 ОП", "1.1gz", "1.2.7 правил подачи жалоб"
            if not free(m.start(), m.end()):
                continue
            num, end, doc = m.group(1), m.end(), PRIMARY
            tail = _TAIL_RE.match(text[end:end + 6])
            if tail and self._doc_for_alias(tail.group(1)):
                doc = self._doc_for_alias(tail.group(1))
                end += tail.end()
            else:
                after, before = text[end:end + 45], text[max(0, m.start() - 45):m.start()]
                for rx, code in CONTEXT_HINTS:
                    if rx.search(after) and code in self.docs:
                        doc = code
                        break
                for rx, code in CONTEXT_BEFORE:
                    if rx.search(before) and code in self.docs:
                        doc = code
                        break
            out.append({"start": m.start(), "end": end, "raw": text[m.start():end], "id": self.resolve(num, doc)})
            taken.append((m.start(), end))
        return sorted(out, key=lambda x: x["start"])

    def parse_refs(self, raw: str) -> tuple[list[str], list[str]]:
        """What a player typed into the rule field -> (existing ids, unknown references)."""
        if not raw:
            return [], []
        text = normalize(str(raw))
        valid, unknown = [], []
        refs = self.find_refs(text)
        for r in refs:
            if r["id"]:
                if r["id"] not in valid:
                    valid.append(r["id"])
            else:
                unknown.append(r["raw"].strip())
        # a bare document code ("gz") or a mined alias
        spans = [(r["start"], r["end"]) for r in refs]
        for m in _WORD_RE.finditer(text):
            if any(s <= m.start() < e for s, e in spans):
                continue
            doc = self._doc_for_alias(m.group(1))
            if doc and doc != PRIMARY and doc in self.by_id and doc not in valid:
                valid.append(doc)
        return valid, unknown

    def refs_in_text(self, text: str) -> list[str]:
        return [r["id"] for r in self.find_refs(text) if r["id"]]

    # ---------------------------------------------------------------- presentation
    LATIN_CODES = {"ОП": "OP", "ЖБ": "JB", "ПГО": "PGO", "ОПГ": "OPG", "ПЛЗ": "PLZ", "ГВ": "GV", "БЗФ": "BZF"}

    def label(self, rid: str, lang: str = "ru") -> str:
        r = self.by_id.get(rid)
        if not r:
            return rid
        if lang in ("uz", "en", "es", "pt") and r.doc in self.LATIN_CODES:
            return f"{r.ref} {self.LATIN_CODES[r.doc]}" if r.ref else self.LATIN_CODES[r.doc]
        return r.label

    def describe(self, rid: str, max_len: int = 400) -> str:
        r = self.by_id.get(rid)
        if not r:
            return ""
        s = r.full_text
        return s if len(s) <= max_len else s[:max_len] + "…"

    def as_prompt_block(self, full_docs=("ОП", "ЖБ", "GZ"), max_point_chars: int = 420) -> str:
        lines = []
        for code, title in self.docs.items():
            if code not in full_docs:
                continue
            how = "голый номер, например 2.1" if code == PRIMARY else f"«{code} <номер>», например {code} 1.1"
            lines.append(f"### {title} — код {code}; в cited писать {how}; в тексте ответа «<номер> {code}»")
            for r in self.rules:
                if r.doc == code and r.ref:
                    t = r.text if len(r.text) <= max_point_chars else r.text[:max_point_chars] + "…"
                    lines.append(f"[{r.id}] {t}")
        others = [f"{code} — {title}" for code, title in self.docs.items() if code not in full_docs]
        if others:
            lines.append("### Другие документы (их пункты приходят в candidate_rules, цитировать как «КОД номер»): "
                         + "; ".join(others))
        return "\n".join(lines)

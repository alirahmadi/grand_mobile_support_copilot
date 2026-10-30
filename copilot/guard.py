"""Safety layer: no invented rules, no promises the service cannot keep, human routing policy."""

import re

from .rules import RuleBook

# Promises a draft must not make on its own (brief: unban, refund, punishing another player).
PROMISE_PATTERNS = {
    "unban": re.compile(
        r"разбан|сним(ем|ут|у) (бан|блокировк|наказани)|наказание (будет )?снято|бан (будет )?снят|разблокир|"
        r"банды? (алып таста|аламыз|ашамыз)|жаза алынып тасталады|бұғаттан шығар|bandan chiqar|ban(ni)? olib tashla|"
        r"jazo(ni)? olib tashla|(will|we'll) (be )?unban|lift (the|your) (ban|penalty)|penalty (will be|has been) removed", re.I),
    "refund": re.compile(
        r"верн(ем|ём|ут|ули) (вам )?(деньги|средства|донат|оплат|имуществ|машин)|возврат средств (будет|произвед)|компенсир(уем|уют)|"
        r"восстанов(им|ят) (имуществ|машин|аккаунт)|ақша(ңызды)? қайтар(амыз|ылады)|pul(ingiz)? qaytariladi|pulni qaytaramiz|"
        r"(will|we'll) refund|refund (will|has been)|reembols|devolveremos", re.I),
    "punish": re.compile(
        r"получит:? (наказани|мут|бан|варн|деморган|джайл|предупрежд|кик|блокировк)|будет (наказан|забанен|заблокирован|замучен)|"
        r"будут применены меры|(выдан|выдал|выдали) (мут|бан|варн|деморган)|жаза (алады|қолданылады)|шара қолданылады|жазаланады|"
        r"jazo (oladi|qo'llaniladi)|choralar ko'riladi|jazolanadi|"
        r"will (be )?(banned|punished|muted|jailed)|will receive|action will be taken", re.I),
}
ALWAYS_HUMAN_CONCEPTS = {"payment", "ban_appeal", "account", "threat", "admin_abuse"}
ALWAYS_HUMAN_REASON = {
    "payment": "платежи всегда решает человек",
    "ban_appeal": "апелляции наказаний всегда решает человек",
    "account": "доступ к аккаунту / взлом всегда решает человек",
    "threat": "угрозы и реальная опасность — срочно человеку",
    "admin_abuse": "жалобы на администрацию разбирает старшая администрация",
}


def find_promises(text: str) -> list[str]:
    return [k for k, rx in PROMISE_PATTERNS.items() if rx.search(text or "")]


def _norm_cited(c: str, book: RuleBook) -> str | None:
    c = str(c).strip().replace(",", ".")
    if c in book.by_id:
        return c
    ids = book.parse_refs(c)[0]
    return ids[0] if len(ids) == 1 else None


def validate_citations(text: str, cited: list[str], book: RuleBook, allowed: set[str] | None) -> tuple[str, list[str], list[str]]:
    """Drop citations that do not exist (or are unrelated to the ticket) and scrub such references from the text.

    Returns (clean_text, clean_cited, problems).
    """
    problems, clean = [], []
    for c in cited or []:
        rid = _norm_cited(c, book)
        if not rid:
            problems.append(f"выдуманный пункт {c} удалён")
        elif allowed is not None and rid not in allowed:
            problems.append(f"пункт {rid} не относится к обращению — удалён")
        elif rid not in clean:
            clean.append(rid)

    bad_spans = []
    for ref in book.find_refs(text or ""):
        rid = ref["id"]
        if rid and (allowed is None or rid in allowed or rid in clean):
            if rid not in clean:
                clean.append(rid)
        else:
            bad_spans.append((ref["start"], ref["end"]))
            problems.append(f"упоминание несуществующего/нерелевантного пункта «{ref['raw'].strip()}» убрано из текста")
    if not bad_spans:
        return (text or "").strip(), clean, problems
    # drop every sentence that contained a bad reference
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, pos = [], 0
    for s in sentences:
        start = text.find(s, pos)
        end = start + len(s)
        pos = end
        if not any(start <= a < end for a, _ in bad_spans):
            out.append(s)
    return " ".join(out).strip(), clean, problems

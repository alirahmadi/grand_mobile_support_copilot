"""Safety layer: no invented rules, no promises the service cannot keep, human routing policy."""

import re

from .rules import REF_IN_TEXT_RE, RuleBook

# Promises a draft must not make on its own (brief: unban, refund, punishing another player).
PROMISE_PATTERNS = {
    "unban": re.compile(
        r"разбан|сним(ем|ут|у) (бан|блокировк|наказани)|бан (будет )?снят|разблокир|"
        r"банды? (алып таста|аламыз|ашамыз)|бұғаттан шығар|bandan chiqar|ban(ni)? olib tashla|"
        r"(will|we'll) (be )?unban|lift (the|your) ban|desbane|levantar (el|la) (ban|sanci)|remover o ban", re.I),
    "refund": re.compile(
        r"верн(ем|ём|ут|ули) (вам )?(деньги|средства|донат|оплат)|возврат средств (будет|произвед)|компенсир(уем|уют)|"
        r"ақша(ңызды)? қайтар(амыз|ылады)|pul(ingiz)? qaytariladi|pulni qaytaramiz|"
        r"(will|we'll) refund|refund (will|has been)|reembols(aremos|aremos)|devolveremos", re.I),
    "punish": re.compile(
        r"получит:? (наказани|мут|бан|варн|деморган|джайл|предупрежд|кик)|будет (наказан|забанен|заблокирован|замучен)|"
        r"(выдан|выдал|выдали) (мут|бан|варн|деморган)|жаза қолданылады|жазаланады|jazo qo'llaniladi|jazolanadi|"
        r"will (be )?(banned|punished|muted|jailed)|will receive a penalty|recibirá una sanción|receberá uma punição", re.I),
}
ALWAYS_HUMAN_CONCEPTS = {"payment", "ban_appeal", "account", "threat", "admin_abuse"}
ALWAYS_HUMAN_REASON = {
    "payment": "платежи всегда решает человек",
    "ban_appeal": "апелляции банов всегда решает человек",
    "account": "подозрение на взлом аккаунта всегда решает человек",
    "threat": "угрозы/реальная опасность — срочно человеку",
    "admin_abuse": "жалобы на администрацию разбирает старшая администрация",
}


def find_promises(text: str) -> list[str]:
    return [k for k, rx in PROMISE_PATTERNS.items() if rx.search(text or "")]


def validate_citations(text: str, cited: list[str], book: RuleBook, allowed: set[str] | None) -> tuple[str, list[str], list[str]]:
    """Drop citations that do not exist (or are irrelevant), and scrub unknown rule numbers from the text.

    Returns (clean_text, clean_cited, problems).
    """
    problems = []
    clean = []
    for c in cited or []:
        rid = str(c).strip().replace(",", ".")
        if rid not in book.by_id:
            problems.append(f"выдуманный пункт {rid} удалён")
            continue
        if allowed is not None and rid not in allowed:
            problems.append(f"пункт {rid} не относится к обращению — удалён")
            continue
        if rid not in clean:
            clean.append(rid)

    def scrub(m: re.Match) -> str:
        rid = m.group(1).replace(",", ".")
        if rid in book.by_id and (allowed is None or rid in allowed):
            if rid not in clean:
                clean.append(rid)
            return m.group(0)
        problems.append(f"упоминание несуществующего/нерелевантного пункта {rid} убрано из текста")
        return "…"

    new_text = REF_IN_TEXT_RE.sub(scrub, text or "")
    if "…" in new_text:
        # remove the whole clause that contained an invalid reference
        new_text = re.sub(r"[^.!?]*…[^.!?]*[.!?]?\s*", "", new_text).strip()
    return new_text, clean, problems

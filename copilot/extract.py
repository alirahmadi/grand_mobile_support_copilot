"""Structured fact extraction and request parsing (contract input or full ticket schema)."""

import re
from datetime import datetime

from .rules import RuleBook
from .textnorm import letters_count, normalize

NICK_RE = re.compile(r"\b([A-Z][a-z]{1,15}_[A-Z][a-zA-Z]{1,20}\d{0,4})\b")
# Complaint rules ЖБ 1.2.7: a complaint with obscenities/insults is not considered.
PROFANITY_RE = re.compile(
    r"\b(бля\w*|хуй\w*|хуе\w*|хуё\w*|нахуй|пизд\w*|еба\w*|ебл\w*|ёба\w*|сука|суки|сучк\w*|шлюх\w*|гандон\w*|"
    r"пидор\w*|пидар\w*|мудак\w*|долбо\w*|уеб\w*|залуп\w*|дебил\w*|даун\w*|чмо\w*|fuck\w*|bitch\w*|"
    r"qotaq\w*|jalab\w*|ам\w*ңды|сігейін)\b", re.I)
EVIDENCE_RE = re.compile(
    r"скрин|скриншот|видео|запис|док-?в|доказ|пруф|proof|screen|video|clip|клип|фрапс|imgur|youtu|"
    r"yadi\.sk|disk\.yandex|drive\.google|gyazo|prnt\.sc|lightshot|vk\.com/video|t\.me/|https?://|"
    r"сурет|бейне|дәлел|rasm|video|isbot|dalil|captura|prueba|print|vídeo|<url>",
    re.I,
)
NO_EVIDENCE_RE = re.compile(r"(нет|без|не (успел|сделал|могу)) (скрин|док|видео|запис)|no (proof|screens?)", re.I)
SERVER_RE = re.compile(
    r"(?:сервер[еау]?|server|serv|сервак[еау]?|с\.)\s*(?:№|#|n|номер)?\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:-?(?:й|ом|м))?\s*(?:сервер|server)",
    re.I,
)


def parse_request(req: dict) -> dict:
    """Normalise either the /triage contract or a full ticket record into one structure."""
    msgs = req.get("messages") or []
    if isinstance(msgs, str):
        msgs = [{"role": "player", "text": msgs}]
    if not msgs and req.get("text"):
        msgs = [{"role": "player", "text": req["text"]}]
    player, reported = [], []
    attachments = 0
    for m in msgs:
        role = (m.get("role") or "player").lower()
        text = m.get("text") or ""
        if role in ("admin", "administrator", "support", "moderator"):
            continue  # never look at the answer we are trying to predict
        if role in ("reported_player", "reported", "accused"):
            reported.append(text)
        else:
            player.append(text)
        a = m.get("attachments", 0)
        attachments += len(a) if isinstance(a, list) else int(a or 0)
    images = req.get("images_count")
    if images is None:
        images = attachments
    region = (req.get("region") or "").lower() or None
    server_lang = (req.get("server_lang") or "").lower() or None
    kazakh = region == "kz" or server_lang == "kk"
    raw_player = "\n".join(t for t in player if t)
    return {
        "ticket_id": req.get("ticket_id"),
        "kind": (req.get("kind") or "report").lower(),
        "created_at": req.get("created_at"),
        "region": region,
        "server_lang": server_lang,
        "lang_hint": req.get("lang"),
        "reporter_hash": req.get("reporter_hash"),
        "reported_hash": req.get("reported_hash"),
        "reported_nick": req.get("reported_nick"),
        "rule_refs_raw": req.get("rule_refs_raw"),
        "images_count": int(images or 0),
        "raw_player_text": raw_player,
        "player_text": normalize(raw_player, kazakh=kazakh),
        # what the player typed into the «rule» field is often the only description ("оск", "дм", "опру")
        "cls_text": normalize(raw_player + (" | " + str(req.get("rule_refs_raw")) if req.get("rule_refs_raw") else ""), kazakh=kazakh),
        "reported_text": normalize("\n".join(reported), kazakh=kazakh),
        "kazakh_server": kazakh,
    }


def parse_time(ts) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def extract_facts(t: dict, book: RuleBook) -> dict:
    text = t["player_text"]
    nick = t.get("reported_nick")
    if not nick:
        found = [n for n in NICK_RE.findall(text)]
        nick = found[0] if found else None

    raw_refs = t.get("rule_refs_raw")
    valid, unknown = book.parse_refs(raw_refs) if raw_refs else ([], [])
    # rule references written in the message body ("2.1 ОП", "1.2 GZ")
    for r in book.find_refs(text):
        if r["id"] and r["id"] not in valid:
            valid.append(r["id"])

    if t["images_count"] > 0 or EVIDENCE_RE.search(text):
        has_evidence = True
    elif NO_EVIDENCE_RE.search(text) or t["kind"] == "complaint":
        has_evidence = False
    else:
        has_evidence = None  # in-game reports rarely need evidence

    server = None
    m = SERVER_RE.search(text)
    if m:
        server = m.group(1) or m.group(2)

    return {
        "nickname": nick,
        "rule_refs": valid,
        "has_url": "<url>" in text or bool(re.search(r"https?://", text)),
        "profanity": bool(PROFANITY_RE.search(text)),
        "has_evidence": has_evidence,
        "server": server,
        "_unknown_rule_refs": unknown,
        "_text_letters": letters_count(text),
    }

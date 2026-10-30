"""Structured fact extraction and request parsing (contract input or full ticket schema)."""

import re
from datetime import datetime

from .rules import RuleBook
from .textnorm import letters_count, normalize

NICK_RE = re.compile(r"\b([A-Z][a-z]{1,15}_[A-Z][a-zA-Z]{1,20})\b")
EVIDENCE_RE = re.compile(
    r"скрин|скриншот|видео|запис|док-?в|доказ|пруф|proof|screen|video|clip|клип|фрапс|imgur|youtu|"
    r"yadi\.sk|disk\.yandex|drive\.google|gyazo|prnt\.sc|lightshot|vk\.com/video|t\.me/|https?://|"
    r"сурет|бейне|дәлел|rasm|video|isbot|dalil|captura|prueba|print|vídeo",
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
    # rule references typed into the message body
    v2, u2 = book.parse_refs(" ".join(book.refs_in_text(text)))
    for r in v2:
        if r not in valid:
            valid.append(r)
    unknown += [u for u in u2 if u not in unknown]

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
        "has_evidence": has_evidence,
        "server": server,
        "_unknown_rule_refs": unknown,
        "_text_letters": letters_count(text),
    }

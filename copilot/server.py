"""HTTP service: POST /triage (contract) + admin console (queue, draft, send/edit/escalate, feedback)."""

import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from .config import DATA_DIR, MODELS_DIR, ROOT, data_file
from .pipeline import Copilot

app = FastAPI(title="Grand Mobile Support Copilot", version="1.0.0")
_copilot: Copilot | None = None
_lock = threading.Lock()
DB_PATH = Path(ROOT / "copilot_queue.sqlite3")
FEEDBACK_PATH = DATA_DIR / "feedback.jsonl"  # derived from confidential data -> stays in git-ignored data/


def copilot() -> Copilot:
    global _copilot
    if _copilot is None:
        with _lock:
            if _copilot is None:
                _copilot = Copilot()
    return _copilot


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, request TEXT, result TEXT,
        status TEXT DEFAULT 'open', final_text TEXT, action TEXT, updated TEXT)""")
    return con


@app.on_event("startup")
def _warmup():
    copilot()


@app.get("/health")
def health():
    c = copilot()
    return {"status": "ok", "llm": c.llm.model if c.llm else None, "rules": len(c.book.rules),
            "categories": len(c.tax.categories), "kb": len(c.kb.records) if c.kb else 0}


@app.post("/triage")
def triage(req: dict = Body(...)):
    if not isinstance(req, dict) or not req.get("messages"):
        raise HTTPException(422, "expected {'kind': ..., 'messages': [{'role': 'player', 'text': ...}]}")
    res = copilot().triage(req)
    return JSONResponse(res)


# ------------------------------------------------------------------ admin console API
@app.get("/")
def admin_page():
    return FileResponse(Path(__file__).parent / "static" / "admin.html")


@app.post("/api/queue")
def enqueue(req: dict = Body(...)):
    items = req.get("items") if isinstance(req.get("items"), list) else [req]
    con = db()
    ids = []
    for item in items:
        res = copilot().triage(item)
        status = "open"
        cur = con.execute("INSERT INTO queue(created, request, result, status) VALUES (?,?,?,?)",
                          (datetime.now().isoformat(timespec="seconds"), json.dumps(item, ensure_ascii=False),
                           json.dumps(res, ensure_ascii=False), status))
        ids.append(cur.lastrowid)
    con.commit()
    return {"ids": ids}


@app.post("/api/queue/demo")
def enqueue_demo(n: int = 25):
    from .evaluate import strip_answers
    from .train import iter_jsonl
    src = DATA_DIR / "tickets_valid.jsonl"
    if not src.exists():
        src = data_file("tickets_demo.jsonl")
    recs = [strip_answers(r) for r in list(iter_jsonl(src))[:n]]
    return enqueue({"items": recs})


@app.get("/api/queue")
def list_queue(status: str = "open"):
    con = db()
    q = "SELECT * FROM queue" + ("" if status == "all" else " WHERE status = ?") + " ORDER BY id DESC LIMIT 500"
    rows = con.execute(q, () if status == "all" else (status,)).fetchall()
    out = []
    for r in rows:
        res = json.loads(r["result"])
        out.append({"id": r["id"], "created": r["created"], "status": r["status"], "action": r["action"],
                    "final_text": r["final_text"], "request": json.loads(r["request"]), "result": res})
    prio = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}
    out.sort(key=lambda x: (x["status"] != "open", prio.get(x["result"]["priority"], 9), not x["result"]["needs_human"], -x["id"]))
    return out


@app.get("/api/rules/{rid}")
def rule(rid: str):
    r = copilot().book.by_id.get(rid)
    if not r:
        raise HTTPException(404, "no such rule")
    return {"id": r.id, "text": r.full_text}


@app.post("/api/queue/{qid}/action")
def act(qid: int, body: dict = Body(...)):
    action = body.get("action")
    if action not in ("send", "edit", "escalate"):
        raise HTTPException(422, "action must be send | edit | escalate")
    con = db()
    row = con.execute("SELECT * FROM queue WHERE id = ?", (qid,)).fetchone()
    if not row:
        raise HTTPException(404, "not found")
    res = json.loads(row["result"])
    final_text = body.get("text") or res["draft_reply"]["text"]
    status = "escalated" if action == "escalate" else "sent"
    con.execute("UPDATE queue SET status=?, action=?, final_text=?, updated=? WHERE id=?",
                (status, action, final_text, datetime.now().isoformat(timespec="seconds"), qid))
    con.commit()
    # Every decision is a training signal: accepted drafts, edits (diff) and escalations.
    fb = {"ts": time.time(), "queue_id": qid, "action": action, "request": json.loads(row["request"]),
          "draft": res["draft_reply"]["text"], "final_text": final_text,
          "edited": action == "edit" and final_text.strip() != res["draft_reply"]["text"].strip(),
          "predicted": {k: res[k] for k in ("category", "priority", "needs_human")},
          "corrected": {k: body[k] for k in ("category", "priority") if body.get(k)}, "note": body.get("note")}
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(fb, ensure_ascii=False) + "\n")
    return {"ok": True, "status": status}


@app.get("/api/templates")
def templates():
    p = MODELS_DIR / "templates_suggested.json"
    mined = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    # edits made by admins in the console, grouped by final text -> candidate new templates
    edits = {}
    if FEEDBACK_PATH.exists():
        for line in FEEDBACK_PATH.read_text(encoding="utf-8").splitlines():
            fb = json.loads(line)
            if fb.get("edited"):
                key = fb["final_text"].strip()
                e = edits.setdefault(key, {"template": key, "count": 0, "category": fb["predicted"]["category"]})
                e["count"] += 1
    return {"mined_from_history": mined, "from_admin_edits": sorted(edits.values(), key=lambda x: -x["count"])[:30]}


@app.get("/api/meta")
def meta():
    c = copilot()
    return {"categories": [x.id for x in c.tax.categories], "priorities": c.tax.priority_ids,
            "llm": c.llm.model if c.llm else None}

"""`make label`: LLM pseudo-labels (category, priority) for a train sample and for the validation set.

The labeller sees the admin's real reply/verdict, so its labels are better than what the service can
infer online. Train labels feed the local classifiers (`make train`); valid labels become a *silver*
evaluation set (clearly reported as such).
"""

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import LLM_JUDGE_MODEL, LLM_MODEL, MODELS_DIR, data_file
from .train import admin_reply, iter_jsonl

SYSTEM = """Ты размечаешь обращения игроков мобильной RP-игры по официальной таксономии поддержки.
Для каждого обращения выбери ровно одну категорию и приоритет строго по описаниям и критериям ниже.
Учитывай канал (complaint — жалоба из лаунчера, report — репорт из игры), текст игрока, ответ обвиняемого,
а также реальный ответ администратора и его решение (они помогают понять суть, но категория описывает обращение, а не решение).

{taxonomy}"""


def _schema(cats, prios):
    return {"type": "object", "properties": {"category": {"type": "string", "enum": cats},
                                             "priority": {"type": "string", "enum": prios}},
            "required": ["category", "priority"], "additionalProperties": False}


def label(n_train: int = 3000, train_model: str = LLM_MODEL, valid_model: str = LLM_JUDGE_MODEL,
          models_dir: Path = MODELS_DIR, workers: int = 8, seed: int = 11):
    from .llm import ClaudeClient
    from .taxonomy import Taxonomy
    tax = Taxonomy.load(data_file("taxonomy.json"))
    system = SYSTEM.format(taxonomy=tax.as_prompt_block())
    schema = _schema([c.id for c in tax.categories], tax.priority_ids)
    random.seed(seed)

    def run(records, model, out_path):
        client = ClaudeClient(model=model, timeout=60)
        total = {"cost": 0.0, "ok": 0, "err": 0}

        def one(rec):
            user = json.dumps({
                "kind": rec.get("kind"),
                "player": "\n".join(m.get("text") or "" for m in rec.get("messages") or [] if m.get("role") == "player")[:1200],
                "reported_player": "\n".join(m.get("text") or "" for m in rec.get("messages") or [] if m.get("role") == "reported_player")[:400],
                "rule_field": rec.get("rule_refs_raw"), "images_count": rec.get("images_count"),
                "admin_reply": admin_reply(rec)[:500], "verdict": rec.get("verdict"), "violation": rec.get("violation"),
            }, ensure_ascii=False)
            out, meta = client.call_json(system, user, schema, max_tokens=200)
            return rec.get("ticket_id"), out, meta
        with ThreadPoolExecutor(max_workers=workers) as ex, open(out_path, "w", encoding="utf-8") as f:
            for tid, out, meta in ex.map(one, records):
                total["cost"] += meta["cost_usd"]
                if out:
                    total["ok"] += 1
                    f.write(json.dumps({"ticket_id": tid, **out, "model": model}, ensure_ascii=False) + "\n")
                else:
                    total["err"] += 1
        print(f"[label] {out_path.name}: ok={total['ok']} errors={total['err']} cost=${total['cost']:.4f} ({model})")

    models_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_file("tickets_train.jsonl.gz")
    if not train_path.exists():
        train_path = data_file("tickets_train.jsonl")
    if train_path.exists() and n_train > 0:
        recs = list(iter_jsonl(train_path))
        random.shuffle(recs)
        run(recs[:n_train], train_model, models_dir / "pseudo_labels.jsonl")
    valid_path = data_file("tickets_valid.jsonl")
    if valid_path.exists():
        run(list(iter_jsonl(valid_path)), valid_model, models_dir / "pseudo_labels_valid.jsonl")

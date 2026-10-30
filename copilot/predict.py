"""`make predict IN=tickets.jsonl OUT=pred.jsonl` — batch triage with the same code path as POST /triage."""

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import LLM_WORKERS
from .train import iter_jsonl


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def run_batch(copilot, records: list[dict], workers: int = LLM_WORKERS) -> list[dict]:
    # Stage 1 runs in chronological order so duplicate detection sees earlier tickets first.
    order = sorted(range(len(records)),
                   key=lambda i: (str(records[i].get("created_at") or "") if isinstance(records[i], dict) else "", i))
    prepared = [None] * len(records)
    for i in order:
        prepared[i] = copilot.safe_prepare(records[i])
    # Stage 2 (LLM) in parallel.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(copilot.safe_finish, prepared))
    out = []
    for rec, res in zip(records, results):
        out.append({"ticket_id": rec.get("ticket_id") if isinstance(rec, dict) else None, **res})
    return out


def summarize(results: list[dict], wall_s: float) -> dict:
    lat = [r["meta"]["latency_ms"] for r in results]
    cost = [r["meta"]["cost_usd"] for r in results]
    modes = {}
    for r in results:
        modes[r["meta"]["mode"]] = modes.get(r["meta"]["mode"], 0) + 1
    n = max(1, len(results))
    return {
        "tickets": len(results),
        "wall_time_s": round(wall_s, 2),
        "needs_human_share": round(sum(r["needs_human"] for r in results) / n, 4),
        "spam_or_duplicate_share": round(sum(r["is_spam_or_duplicate"] for r in results) / n, 4),
        "latency_ms_p50": round(percentile(lat, 0.5), 1),
        "latency_ms_p95": round(percentile(lat, 0.95), 1),
        "latency_ms_max": max(lat) if lat else 0,
        "cost_usd_per_ticket": round(statistics.mean(cost), 7) if cost else 0.0,
        "cost_usd_total": round(sum(cost), 5),
        "modes": modes,
        "llm_errors": sum(1 for r in results if r["meta"].get("llm_error")),
        "with_citations_share": round(sum(bool(r["draft_reply"]["cited"]) for r in results) / n, 4),
        "lang_distribution": _dist(r["lang"] for r in results),
        "category_distribution": _dist(r["category"] for r in results),
        "priority_distribution": _dist(r["priority"] for r in results),
    }


def _dist(values) -> dict:
    d = {}
    for v in values:
        d[v] = d.get(v, 0) + 1
    return dict(sorted(d.items(), key=lambda x: -x[1]))


def main(inp: Path, out: Path, workers: int = LLM_WORKERS, use_llm: bool | None = None, keep_meta: bool = True):
    from .config import MODELS_DIR
    from .pipeline import Copilot
    if not (MODELS_DIR / "verdict.joblib").exists():
        print("[predict] ВНИМАНИЕ: обученных моделей нет (models/ пуст) — сначала запустите `make train` "
              "(python -m copilot.cli train). Сейчас будут работать только словари и шаблоны.")
    copilot = Copilot(use_llm=use_llm)
    records = list(iter_jsonl(inp, lenient=True))
    bad = sum(1 for r in records if "_bad_line" in r)
    if bad:
        print(f"[predict] warning: {bad} lines are not valid JSON — answered with a safe 'needs human' fallback")
    print(f"[predict] {len(records)} tickets from {inp} | LLM: {'on (' + copilot.llm.model + ')' if copilot.llm else 'off (offline templates)'}")
    t0 = time.time()
    results = run_batch(copilot, records, workers)
    wall = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            if not keep_meta:
                r = {k: v for k, v in r.items() if k != "meta"}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = summarize(results, wall)
    with open(str(out) + ".stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"[predict] wrote {out} | " + ", ".join(f"{k}={stats[k]}" for k in
          ("needs_human_share", "latency_ms_p95", "cost_usd_per_ticket", "modes", "llm_errors")))
    if copilot.llm and stats["llm_errors"] >= max(1, len(results) // 2):
        first = next((r["meta"]["llm_error"] for r in results if r["meta"].get("llm_error")), "")
        print(f"[predict] ВНИМАНИЕ: LLM не отвечает ({stats['llm_errors']} ошибок из {len(results)}), "
              f"черновики сделаны шаблонами. Первая ошибка: {first}")
    return results, stats

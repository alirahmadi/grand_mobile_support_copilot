"""`make eval`: honest offline metrics on tickets_valid.jsonl + REPORT.md.

What can be measured without the hidden labels:
  * category accuracy / priority macro-F1 — only if valid has labels (or 'silver' LLM labels from `make label`);
  * agreement of our suggested verdict with the real admin verdict (complaints);
  * duplicate detection vs admin verdict == duplicate;
  * citations: existence (must be 100%), and whether we cite the rule the admin actually used;
  * draft similarity to the real admin reply (chrF) — a cheap proxy for the blind 1-5 rating;
  * optional LLM-judge 1-5 score of draft vs real reply (`make eval JUDGE=1`);
  * needs_human share, p50/p95 latency, cost per ticket.
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .config import LLM_JUDGE_MODEL, MODELS_DIR, REPORTS_DIR
from .predict import run_batch, summarize
from .train import VERDICT_MAP, VERDICT_SKIP, admin_reply, canon_reply, iter_jsonl, parse_violation


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    hyp, ref = re.sub(r"\s+", " ", hyp.lower()), re.sub(r"\s+", " ", ref.lower())
    if not hyp or not ref:
        return 0.0
    precs, recs = [], []
    for k in range(1, n + 1):
        h = Counter(hyp[i:i + k] for i in range(len(hyp) - k + 1))
        r = Counter(ref[i:i + k] for i in range(len(ref) - k + 1))
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precs.append(overlap / sum(h.values()))
        recs.append(overlap / sum(r.values()))
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    return round(100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r), 2)


def macro_f1(y_true: list, y_pred: list) -> float:
    labels = sorted(set(y_true))
    f1s = []
    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b != lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return round(sum(f1s) / len(f1s), 4) if f1s else 0.0


def strip_answers(rec: dict) -> dict:
    """What the service is allowed to see: no admin messages, no verdict/punishment/labels."""
    r = {k: v for k, v in rec.items() if k not in ("verdict", "punishment", "violation", "category", "priority")}
    r["messages"] = [m for m in rec.get("messages") or [] if (m.get("role") or "") != "admin"]
    return r


def judge_drafts(pairs: list[dict], model: str = LLM_JUDGE_MODEL) -> list[dict]:
    """LLM-as-judge: 1-5 score of our draft vs the real admin reply (proxy for the organisers' blind rating)."""
    from concurrent.futures import ThreadPoolExecutor
    from .llm import ClaudeClient
    client = ClaudeClient(model=model, timeout=60)
    system = ("Ты — старший администратор поддержки мобильной RP-игры. Оцени черновик ответа игроку по шкале 1–5 "
              "рядом с реальным ответом администратора: 5 — можно отправлять как есть и решение совпадает; "
              "4 — мелкие правки; 3 — суть верна, нужна заметная правка; 2 — решение спорное или ответ бесполезен; "
              "1 — неверное решение, выдуманные правила или обещания. Оценивай язык ответа (должен совпадать с языком игрока), "
              "корректность цитат правил и отсутствие ложных обещаний. Верни JSON.")
    schema = {"type": "object", "properties": {"score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                                               "comment": {"type": "string"}},
              "required": ["score", "comment"], "additionalProperties": False}

    def one(p):
        user = json.dumps({"player_request": p["request"][:800], "real_admin_reply": p["reference"][:600],
                           "draft": p["draft"][:800], "draft_cited_rules": p["cited"]}, ensure_ascii=False)
        out, meta = client.call_json(system, user, schema, max_tokens=300)
        return {**p, "judge_score": (out or {}).get("score"), "judge_comment": (out or {}).get("comment"),
                "judge_cost": meta["cost_usd"]}
    with ThreadPoolExecutor(max_workers=6) as ex:
        return list(ex.map(one, pairs))


def evaluate(valid_path: Path, out_dir: Path = REPORTS_DIR, use_llm: bool | None = None, judge: bool = False,
             judge_n: int = 100, limit: int | None = None, seed: int = 7) -> dict:
    from .pipeline import Copilot
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    gold = list(iter_jsonl(valid_path))
    if limit:
        gold = gold[:limit]
    copilot = Copilot(use_llm=use_llm)
    import time
    t0 = time.time()
    preds = run_batch(copilot, [strip_answers(r) for r in gold])
    stats = summarize(preds, time.time() - t0)
    metrics = {"n": len(gold), "mode": "llm:" + copilot.llm.model if copilot.llm else "offline", "ops": stats}

    # ---- labelled / silver classification metrics
    silver = {}
    sp = MODELS_DIR / "pseudo_labels_valid.jsonl"
    if sp.exists():
        silver = {r["ticket_id"]: r for r in iter_jsonl(sp)}
    for name, getter in (("gold", lambda g: (g.get("category"), g.get("priority"))),
                         ("silver_llm", lambda g: (silver.get(g.get("ticket_id"), {}).get("category"),
                                                   silver.get(g.get("ticket_id"), {}).get("priority")))):
        yc, pc, yp, pp = [], [], [], []
        for g, p in zip(gold, preds):
            c, pr = getter(g)
            if c:
                yc.append(c)
                pc.append(p["category"])
            if pr:
                yp.append(pr)
                pp.append(p["priority"])
        if yc:
            metrics[f"category_accuracy_{name}"] = round(sum(a == b for a, b in zip(yc, pc)) / len(yc), 4)
            metrics[f"category_n_{name}"] = len(yc)
        if yp:
            metrics[f"priority_macro_f1_{name}"] = macro_f1(yp, pp)

    # ---- verdict agreement, duplicates, citations, similarity
    book = copilot.book
    v_true, v_pred = [], []
    dup_tp = dup_fp = dup_fn = 0
    cite_exist = cite_total = 0
    cite_hit = cite_with_ref = 0
    sims = defaultdict(list)
    samples = []
    for g, p in zip(gold, preds):
        ref = admin_reply(g)
        gv = VERDICT_MAP.get(g.get("verdict"), g.get("verdict"))
        if g.get("kind") == "complaint" and gv not in VERDICT_SKIP:
            v_true.append(gv)
            v_pred.append(p["meta"].get("suggested_verdict") or "approved")
            is_dup_true = gv == "duplicate"
            if p["meta"].get("is_duplicate") and is_dup_true:
                dup_tp += 1
            elif p["meta"].get("is_duplicate") and not is_dup_true:
                dup_fp += 1
            elif is_dup_true and not p["meta"].get("is_duplicate"):
                dup_fn += 1
        for c in p["draft_reply"]["cited"]:
            cite_total += 1
            cite_exist += c in book.by_id
        rid, _ = parse_violation(g.get("violation") or "", book)
        if rid and p["draft_reply"]["cited"]:
            cite_with_ref += 1
            cite_hit += any(c == rid or rid.startswith(c + ".") or c.startswith(rid + ".") for c in p["draft_reply"]["cited"])
        if ref:
            s = chrf(p["draft_reply"]["text"], canon_reply(ref))
            sims[p["draft_reply"]["lang"]].append(s)
            samples.append({"ticket_id": g.get("ticket_id"), "kind": g.get("kind"), "lang": p["draft_reply"]["lang"],
                            "request": "\n".join(m.get("text") or "" for m in g.get("messages") or [] if m.get("role") != "admin"),
                            "reference": ref, "draft": p["draft_reply"]["text"], "cited": p["draft_reply"]["cited"],
                            "category": p["category"], "priority": p["priority"], "needs_human": p["needs_human"],
                            "verdict_true": g.get("verdict"), "verdict_pred": p["meta"].get("suggested_verdict"),
                            "chrf": s})
    if v_true:
        metrics["verdict_accuracy"] = round(sum(a == b for a, b in zip(v_true, v_pred)) / len(v_true), 4)
        metrics["verdict_macro_f1"] = macro_f1(v_true, v_pred)
        metrics["verdict_true_distribution"] = dict(Counter(v_true).most_common())
        conf = defaultdict(Counter)
        for a, b in zip(v_true, v_pred):
            conf[a][b] += 1
        metrics["verdict_confusion"] = {k: dict(v) for k, v in conf.items()}
    metrics["duplicate_precision"] = round(dup_tp / (dup_tp + dup_fp), 4) if dup_tp + dup_fp else None
    metrics["duplicate_recall"] = round(dup_tp / (dup_tp + dup_fn), 4) if dup_tp + dup_fn else None
    metrics["citations_total"] = cite_total
    metrics["citations_existing_share"] = round(cite_exist / cite_total, 4) if cite_total else None
    metrics["cited_rule_matches_admin_violation"] = round(cite_hit / cite_with_ref, 4) if cite_with_ref else None
    metrics["draft_chrf_vs_admin_by_lang"] = {k: round(sum(v) / len(v), 2) for k, v in sims.items() if v}
    all_s = [s for v in sims.values() for s in v]
    metrics["draft_chrf_vs_admin"] = round(sum(all_s) / len(all_s), 2) if all_s else None

    if judge and samples:
        pick = random.sample(samples, min(judge_n, len(samples)))
        judged = judge_drafts(pick)
        scores = [j["judge_score"] for j in judged if j["judge_score"]]
        metrics["llm_judge_mean_score"] = round(sum(scores) / len(scores), 3) if scores else None
        metrics["llm_judge_n"] = len(scores)
        metrics["llm_judge_model"] = LLM_JUDGE_MODEL
        metrics["llm_judge_cost_usd"] = round(sum(j["judge_cost"] for j in judged), 4)
        by_id = {j["ticket_id"]: j for j in judged}
        for s in samples:
            if s["ticket_id"] in by_id:
                s.update({k: by_id[s["ticket_id"]][k] for k in ("judge_score", "judge_comment")})

    with open(out_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    with open(out_dir / "valid_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    write_report(metrics, samples, out_dir / "REPORT.md")
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("verdict_confusion",)}, ensure_ascii=False, indent=1))
    print(f"[eval] report -> {out_dir / 'REPORT.md'}")
    return metrics


def write_report(m: dict, samples: list[dict], path: Path):
    ops = m["ops"]

    def fmt(v, pct=False):
        if v is None:
            return "—"
        return f"{100 * v:.1f}%" if pct else str(v)

    lines = [
        "# Support Copilot — отчёт по валидации",
        "",
        f"Валидация: {m['n']} обращений, режим: **{m['mode']}**.",
        "",
        "## Ключевые метрики",
        "",
        "| Метрика | Значение |",
        "|---|---|",
    ]
    if "category_accuracy_gold" in m:
        lines.append(f"| Accuracy категории (разметка организаторов) | {fmt(m['category_accuracy_gold'], True)} |")
    if "priority_macro_f1_gold" in m:
        lines.append(f"| Macro-F1 приоритета (разметка организаторов) | {m['priority_macro_f1_gold']} |")
    if "category_accuracy_silver_llm" in m:
        lines.append(f"| Accuracy категории (silver-разметка LLM, n={m.get('category_n_silver_llm')}) | {fmt(m['category_accuracy_silver_llm'], True)} |")
    if "priority_macro_f1_silver_llm" in m:
        lines.append(f"| Macro-F1 приоритета (silver-разметка LLM) | {m['priority_macro_f1_silver_llm']} |")
    lines += [
        f"| Совпадение предложенного вердикта с решением админа (жалобы) | {fmt(m.get('verdict_accuracy'), True)} (macro-F1 {m.get('verdict_macro_f1', '—')}) |",
        f"| Дубли: precision / recall (vs вердикт duplicate) | {fmt(m.get('duplicate_precision'), True)} / {fmt(m.get('duplicate_recall'), True)} |",
        f"| Цитаты: доля существующих пунктов | {fmt(m.get('citations_existing_share'), True)} (всего {m.get('citations_total')}) |",
        f"| Цитируемый пункт совпадает с пунктом админа | {fmt(m.get('cited_rule_matches_admin_violation'), True)} |",
        f"| chrF черновика к реальному ответу админа | {m.get('draft_chrf_vs_admin', '—')} |",
    ]
    if m.get("llm_judge_mean_score"):
        lines.append(f"| LLM-судья ({m['llm_judge_model']}), средняя оценка 1–5, n={m['llm_judge_n']} | {m['llm_judge_mean_score']} |")
    lines += [
        f"| Доля «нужен человек» | {fmt(ops['needs_human_share'], True)} |",
        f"| Доля спам/дубль | {fmt(ops['spam_or_duplicate_share'], True)} |",
        f"| Черновики с цитатами правил | {fmt(ops['with_citations_share'], True)} |",
        f"| Латентность p50 / p95 / max, мс | {ops['latency_ms_p50']} / {ops['latency_ms_p95']} / {ops['latency_ms_max']} |",
        f"| Стоимость на обращение, $ | {ops['cost_usd_per_ticket']} (всего {ops['cost_usd_total']}) |",
        f"| Режимы генерации | {ops['modes']} (ошибок LLM: {ops['llm_errors']}) |",
        "",
        f"chrF по языкам черновика: {m.get('draft_chrf_vs_admin_by_lang')}",
        "",
        f"Распределение языков: {ops['lang_distribution']}",
        "",
        f"Распределение категорий: {ops['category_distribution']}",
        "",
        f"Распределение приоритетов: {ops['priority_distribution']}",
        "",
    ]
    if m.get("verdict_confusion"):
        lines += ["## Матрица вердиктов (строка — решение админа, столбец — наш вердикт)", ""]
        cols = sorted({c for row in m["verdict_confusion"].values() for c in row})
        lines.append("| админ \\ мы | " + " | ".join(cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for k, row in m["verdict_confusion"].items():
            lines.append(f"| {k} | " + " | ".join(str(row.get(c, 0)) for c in cols) + " |")
        lines.append("")
    lines += ["## Примеры черновиков рядом с реальными ответами", ""]
    by_lang = defaultdict(list)
    for s in samples:
        by_lang[s["lang"]].append(s)
    picked = []
    for lang, quota in (("ru", 8), ("kk", 4), ("uz", 4), ("en", 2), ("es", 1), ("pt", 1)):
        pool = by_lang.get(lang, [])
        random.shuffle(pool)
        picked += pool[:quota]
    for s in picked:
        lines += [
            f"### {s['ticket_id']} · {s['kind']} · {s['lang']} · {s['category']} · {s['priority']} · нужен человек: {s['needs_human']}",
            "",
            f"**Обращение:** {s['request'][:400]}",
            "",
            f"**Черновик:** {s['draft']} — цитаты: {s['cited'] or 'нет'}",
            "",
            f"**Реальный ответ админа:** {s['reference'][:400]}",
            "",
            f"Вердикт админа: `{s['verdict_true']}`, наш: `{s['verdict_pred']}`, chrF={s['chrf']}"
            + (f", LLM-судья: {s['judge_score']} — {s.get('judge_comment', '')}" if s.get("judge_score") else ""),
            "",
        ]
    lines += [
        "## Как читать метрики",
        "",
        "* Категории и приоритеты в выданной выборке не размечены. Если их нет и в валидации, accuracy считается по silver-разметке "
        "(`make label`, более сильная модель с доступом к ответу админа) — это оценка, а не истинная метрика.",
        "* chrF и LLM-судья — автоматические прокси для слепой оценки администраторов, не замена ей.",
        "* Латентность — время обработки одного обращения сервисом (без очереди батча); стоимость — по фактическому usage API.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

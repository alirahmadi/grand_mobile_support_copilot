"""Single entry point (works on Windows without make): python -m copilot.cli <command>."""

import argparse
import json
import sys
from pathlib import Path

from .config import DATA_DIR, LLM_WORKERS, MODELS_DIR, data_file


def _utf8_console():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass


def cmd_check(_args):
    from .concepts import ConceptMap, load_concepts
    from .rules import RuleBook
    from .taxonomy import Taxonomy
    for name in ("rules.json", "taxonomy.json", "tickets_train.jsonl.gz", "tickets_valid.jsonl"):
        p = DATA_DIR / name
        print(f"{'OK ' if p.exists() else '-- '} {p}" + ("" if p.exists() else "  (нет — будет использована демо-фикстура, если есть)"))
    book = RuleBook.load(data_file("rules.json"))
    tax = Taxonomy.load(data_file("taxonomy.json"))
    print(f"\nПравила: {len(book.rules)} пунктов, разделы: {sorted({r.section_id for r in book.rules}, key=lambda x: int(x) if x.isdigit() else 0)}")
    for r in book.rules[:6]:
        print(f"  [{r.id}] {r.full_text[:110]}")
    print(f"\nКатегории ({len(tax.categories)}), приоритеты: {tax.priority_ids}")
    cmap = ConceptMap(tax, load_concepts())
    print(cmap.report())
    print("\nЕсли категория '(не сопоставлено)' — допишите её id/слова в copilot/resources/concepts.json (поле match).")


def cmd_train(args):
    from .train import train
    tr = Path(args.train) if args.train else data_file("tickets_train.jsonl.gz")
    if not tr.exists():
        tr = data_file("tickets_demo.jsonl")
        print(f"[train] нет tickets_train.jsonl.gz в data/, обучаюсь на демо-фикстуре {tr}")
    train(tr, data_file("tickets_valid.jsonl"), include_valid=args.include_valid, max_kb=args.max_kb)


def cmd_label(args):
    from .label import label
    label(n_train=args.n)


def cmd_predict(args):
    from .predict import main
    main(Path(args.inp), Path(args.out), workers=args.workers, use_llm=False if args.offline else None,
         keep_meta=not args.no_meta)


def cmd_eval(args):
    from .evaluate import evaluate
    vp = Path(args.valid) if args.valid else data_file("tickets_valid.jsonl")
    if not vp.exists():
        vp = data_file("tickets_demo.jsonl")
        print(f"[eval] нет tickets_valid.jsonl в data/, оцениваю на демо-фикстуре {vp} (метрики не показательны)")
    evaluate(vp, use_llm=False if args.offline else None, judge=args.judge, judge_n=args.judge_n, limit=args.limit)


def cmd_serve(args):
    import uvicorn
    uvicorn.run("copilot.server:app", host=args.host, port=args.port)


def cmd_one(args):
    from .pipeline import Copilot
    req = {"kind": args.kind, "messages": [{"role": "player", "text": args.text, "attachments": args.attachments}]}
    print(json.dumps(Copilot(use_llm=False if args.offline else None).triage(req), ensure_ascii=False, indent=1))


def main():
    _utf8_console()
    ap = argparse.ArgumentParser(prog="copilot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-data").set_defaults(fn=cmd_check)
    p = sub.add_parser("train")
    p.add_argument("--train")
    p.add_argument("--include-valid", action="store_true", help="добавить валидацию в базу знаний (для финального сабмита)")
    p.add_argument("--max-kb", type=int, default=120_000)
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("label")
    p.add_argument("--n", type=int, default=3000)
    p.set_defaults(fn=cmd_label)
    p = sub.add_parser("predict")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=LLM_WORKERS)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--no-meta", action="store_true", help="писать только поля контракта (+ticket_id)")
    p.set_defaults(fn=cmd_predict)
    p = sub.add_parser("eval")
    p.add_argument("--valid")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--judge-n", type=int, default=100)
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("serve")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("one", help="разобрать одно сообщение из командной строки")
    p.add_argument("text")
    p.add_argument("--kind", default="report")
    p.add_argument("--attachments", type=int, default=0)
    p.add_argument("--offline", action="store_true")
    p.set_defaults(fn=cmd_one)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

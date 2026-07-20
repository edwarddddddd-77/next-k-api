"""Backtest CLI.

  python -m backend.backtest download [--symbols ...] [--force] [--coverage]
  python -m backend.backtest run --sid S01_FUNDING_FADE [--params '{"tp_atr":2}']
  python -m backend.backtest optimize [--sid SXX] [--workers 6]
  python -m backend.backtest report
  python -m backend.backtest all [--workers 6]
"""
import argparse
import asyncio
import json
import logging

from . import data

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser(prog="backend.backtest")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download 30d history into backtest/data/")
    d.add_argument("--symbols", default="", help="comma list; default = full universe")
    d.add_argument("--force", action="store_true")
    d.add_argument("--skip-aggtrades", action="store_true")
    d.add_argument("--coverage", action="store_true", help="only print coverage report")

    r = sub.add_parser("run", help="single strategy backtest (default params)")
    r.add_argument("--sid", required=True)
    r.add_argument("--params", default="{}", help="JSON param overrides")
    r.add_argument("--oos", action="store_true", help="OOS window only")

    o = sub.add_parser("optimize", help="grid search + IS/OOS for all strategies")
    o.add_argument("--sid", default="", help="single strategy only")
    o.add_argument("--workers", type=int, default=6)

    sub.add_parser("report", help="render markdown reports from raw JSON")

    a = sub.add_parser("all", help="optimize everything then render reports")
    a.add_argument("--workers", type=int, default=6)

    args = ap.parse_args()
    if args.cmd == "download":
        if args.coverage:
            print(json.dumps(data.coverage_report(), indent=2, ensure_ascii=False))
            return
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
        asyncio.run(data.download_all(symbols=symbols, force=args.force,
                                      skip_aggtrades=args.skip_aggtrades))
        print(json.dumps(data.coverage_report(), indent=2, ensure_ascii=False))
    elif args.cmd == "run":
        from .runner import run
        res = run(args.sid, json.loads(args.params),
                  t0=data.IS_END if args.oos else None)
        print(json.dumps(res.metrics(), indent=2, ensure_ascii=False))
    elif args.cmd == "optimize":
        from .optimize import search, search_all
        if args.sid:
            search(args.sid, workers=args.workers)
        else:
            search_all(workers=args.workers)
    elif args.cmd == "report":
        from .report import write_all
        print(f"reports -> {write_all()}")
    elif args.cmd == "all":
        from .optimize import search_all
        from .report import write_all
        search_all(workers=args.workers)
        print(f"reports -> {write_all()}")


if __name__ == "__main__":
    main()

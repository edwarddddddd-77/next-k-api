#!/usr/bin/env python3
"""ORB 纸面扫描 CLI（ML Live Gate）。"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


_configure_logging()
logger = logging.getLogger("orb_scanner")

from orb.v2.paper import run_resolve_only_v2, run_scan_v2  # noqa: E402


def _scan_summary(out: dict) -> dict:
    return {
        "ok": out.get("ok"),
        "lane": out.get("lane"),
        "skipped": out.get("skipped"),
        "reason": out.get("reason"),
        "opens": len(out.get("opens") or []),
        "gate_skips": len(out.get("gate_skips") or []),
        "shadow": out.get("shadow"),
        "ml_ranker": out.get("ml_ranker"),
        "symbols": len(out.get("symbols") or []),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="ORB — ML Live Gate 纸面扫描")
    ap.add_argument("--resolve-only", action="store_true")
    ap.add_argument("--no-resolve", action="store_true")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()
    if args.resolve_only:
        out = run_resolve_only_v2()
    else:
        out = run_scan_v2(do_resolve=not args.no_resolve)
    if args.pretty:
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    else:
        logger.info("[orb] scan result %s", json.dumps(_scan_summary(out), ensure_ascii=False, default=str))
    return 0 if out.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())

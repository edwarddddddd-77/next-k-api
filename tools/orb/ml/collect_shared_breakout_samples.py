#!/usr/bin/env python3
"""43 标联合收集共享 ORB 样本（本地 K 线 + 宏观关闭）。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.core.backtest_ml import run_backtest  # noqa: E402
from orb.ml.samples import (  # noqa: E402
    MAX_LOOKBACK_DAYS,
    collect_rows,
    filter_low_sample_symbols,
    parse_symbol_list,
    recompute_sync,
)
from orb.core.config import OrbConfig  # noqa: E402
from orb.ml.model.paths import resolve_samples_path, resolve_symbols_path  # noqa: E402


def _chunks(items: list[str], size: int) -> list[list[str]]:
    if size <= 0 or size >= len(items):
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _ml_config() -> OrbConfig:
    os.environ["ORB_MACRO_FILTER"] = "0"
    cfg = OrbConfig.from_env()
    cfg.macro_filter = False
    return cfg


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    load_env_oi()
    ap = argparse.ArgumentParser(description="Collect shared ORB breakout samples from local kline cache")
    ap.add_argument("--days", type=float, default=MAX_LOOKBACK_DAYS)
    ap.add_argument("--symbols-file", type=str, default="")
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--json-out", type=str, default=str(resolve_samples_path()))
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=0, help="0=全部联合回测")
    args = ap.parse_args()

    sym_path = Path(args.symbols_file) if args.symbols_file.strip() else resolve_symbols_path()
    syms = parse_symbol_list(args.symbols) if args.symbols.strip() else parse_symbol_list(
        sym_path.read_text(encoding="utf-8")
    )
    if not syms:
        print("No symbols")
        return 1

    cfg = _ml_config()
    days = min(float(args.days), MAX_LOOKBACK_DAYS)
    batches = _chunks(syms, int(args.batch_size)) if int(args.batch_size) > 0 else [syms]
    t0 = time.time()
    print(
        f"[collect] cached klines | macro=off | {len(syms)} syms, {len(batches)} pass(es), {days}d",
        flush=True,
    )

    all_trades: list[dict] = []
    for i, batch in enumerate(batches, 1):
        bt0 = time.time()
        if len(batches) > 1:
            print(f"[collect] batch {i}/{len(batches)}: {len(batch)} symbols", flush=True)
        raw = run_backtest(days=days, symbols=batch, cfg=cfg, record_features=True)
        trades = list(raw.get("trades") or [])
        all_trades.extend(trades)
        print(f"[collect] pass {i}: +{len(trades)} trades ({round(time.time() - bt0, 1)}s)", flush=True)

    rows = collect_rows({"trades": all_trades})
    recompute_sync(rows)
    rows, dropped = filter_low_sample_symbols(rows, min_samples=max(1, int(args.min_samples)))

    sym_counts: dict[str, int] = {}
    for r in rows:
        s = str(r.get("symbol") or "")
        sym_counts[s] = sym_counts.get(s, 0) + 1
    fake_n = sum(int(r["fake_breakout"]) for r in rows)
    true_n = sum(int(r["true_breakout"]) for r in rows)
    summary = {
        "kind": "shared",
        "data_source": "local_kline_cache",
        "macro_filter": False,
        "symbols_requested": len(syms),
        "symbols_kept": len(sym_counts),
        "symbols_dropped_low_samples": dropped,
        "days_cap": days,
        "samples": len(rows),
        "fake_breakouts": fake_n,
        "true_breakouts": true_n,
        "fake_rate_pct": round(fake_n / len(rows) * 100, 1) if rows else 0.0,
        "true_rate_pct": round(true_n / len(rows) * 100, 1) if rows else 0.0,
        "elapsed_sec": round(time.time() - t0, 1),
        "per_symbol_counts": dict(sorted(sym_counts.items(), key=lambda x: -x[1])),
    }

    from orb.ml.model.paths import ensure_model_dirs

    ensure_model_dirs()
    json_out = Path(args.json_out)
    csv_out = json_out.with_suffix(".csv")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(rows, csv_out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"JSON -> {json_out}")
    print(f"CSV  -> {csv_out}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

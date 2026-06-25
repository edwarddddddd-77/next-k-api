#!/usr/bin/env python3
"""逐标 session sim（对齐 COIN 单标测试），汇总 CSV/JSON。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.core.config import OrbConfig  # noqa: E402
from orb.core.kline_cache import norm_symbol  # noqa: E402
from orb.ml.gate import LiveGateConfig, gate_with_ml_bypass  # noqa: E402
from orb.ml.samples import parse_symbol_list  # noqa: E402
from orb.v2.paths import resolve_gate_config_path, resolve_symbols_path  # noqa: E402
from orb.v2.robots import init_robot_wallets, robot_equity_from_env  # noqa: E402
from tools.orb.ml.eval_live_gate import _ml_cfg  # noqa: E402
from tools.orb.v2.backtest_universe import universe_session_dates  # noqa: E402
from tools.orb.v2.sim_live_session import (  # noqa: E402
    DEFAULT_FEE_BPS_PER_SIDE,
    _resolve_dates,
    simulate_live_sessions,
)

SUMMARY_FIELDS = [
    "symbol",
    "sessions",
    "opens",
    "fill_skips",
    "gross_pnl_usdt",
    "fees_usdt",
    "net_pnl_usdt",
    "return_pct",
    "win_trades",
    "loss_trades",
    "win_rate",
    "end_wallet_usdt",
    "avg_pnl_per_trade",
    "elapsed_sec",
]


def _run_one(
    sym: str,
    dates: List[str],
    *,
    gate: LiveGateConfig,
    ranker,
    cfg: OrbConfig,
    robot_equity: float,
    fee_bps: float,
    entry_fill: str,
) -> Dict[str, Any]:
    t0 = time.time()
    wallets = init_robot_wallets(count=1, equity_usdt=robot_equity)
    days = simulate_live_sessions(
        dates,
        [sym],
        gate=gate,
        ranker=ranker,
        cfg=cfg,
        robot_wallets=wallets,
        respect_env_filters=cfg.macro_filter,
        fee_bps_per_side=fee_bps,
        entry_fill=entry_fill,
        ml_enabled=ranker is not None,
    )
    trades = [t for d in days for t in (d.get("trades") or [])]
    wins = sum(1 for t in trades if float(t.get("pnl_usdt") or 0) > 0)
    losses = sum(1 for t in trades if float(t.get("pnl_usdt") or 0) < 0)
    opens = len(trades)
    gross = round(sum(float(d.get("gross_pnl_usdt") or 0) for d in days), 2)
    net = round(sum(float(d.get("net_pnl_usdt") or 0) for d in days), 2)
    fees = round(sum(float(d.get("fees_usdt") or 0) for d in days), 2)
    fill_skips = sum(int(d.get("fill_skips") or 0) for d in days)
    end_w = round(float(wallets[0]), 2)
    ret = round((end_w / robot_equity - 1) * 100, 1) if robot_equity > 0 else 0.0
    return {
        "symbol": sym.replace("USDT", ""),
        "sessions": len(dates),
        "opens": opens,
        "fill_skips": fill_skips,
        "gross_pnl_usdt": gross,
        "fees_usdt": fees,
        "net_pnl_usdt": net,
        "return_pct": ret,
        "win_trades": wins,
        "loss_trades": losses,
        "win_rate": round(wins / opens * 100, 1) if opens else 0.0,
        "end_wallet_usdt": end_w,
        "avg_pnl_per_trade": round(net / opens, 2) if opens else 0.0,
        "elapsed_sec": round(time.time() - t0, 1),
        "days": days,
    }


def main() -> int:
    load_env_oi()
    import os

    os.environ["ORB_V2_ROBOT_RESET_CAP"] = "0"
    ap = argparse.ArgumentParser(description="Batch per-symbol live session sim")
    ap.add_argument("--symbols-file", default=str(resolve_symbols_path()))
    ap.add_argument("--from-date", default="2026-02-09")
    ap.add_argument("--to-date", default="2026-06-24")
    ap.add_argument("--robot-equity", type=float, default=1000.0)
    ap.add_argument("--entry-fill", default="stoplimit_gap")
    ap.add_argument("--or-minutes", type=int, default=15)
    ap.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS_PER_SIDE)
    ap.add_argument("--no-live-filters", action="store_true")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--csv-out", default="")
    args = ap.parse_args()

    syms_raw = parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8"))
    syms = [norm_symbol(s) for s in syms_raw]

    gate = gate_with_ml_bypass(LiveGateConfig.from_json(Path(resolve_gate_config_path())))
    cfg = _ml_cfg(compound_per_symbol=True, respect_env_filters=not args.no_live_filters)
    if int(args.or_minutes) > 0:
        cfg.or_minutes = int(args.or_minutes)

    lo, hi = (args.from_date or "").strip(), (args.to_date or "").strip()
    ref_dates: List[str] = []
    for s in syms:
        cand = universe_session_dates([s], cfg)
        if len(cand) > len(ref_dates):
            ref_dates = cand
    if not ref_dates:
        print("No session dates in kline cache")
        return 1
    dates = [d for d in ref_dates if (not lo or d >= lo) and (not hi or d <= hi)]
    if not dates:
        print("No sessions in requested range")
        return 1

    tag = f"{dates[0]}_{dates[-1]}"
    out_dir = ROOT / "output" / "orb" / "v2" / "eval"
    json_path = Path(args.json_out) if args.json_out else out_dir / f"batch_sym_{args.entry_fill}_or{cfg.or_minutes}_{tag}_eq{int(args.robot_equity)}_no_filter_no_reset.json"
    csv_path = Path(args.csv_out) if args.csv_out else out_dir / f"batch_sym_{args.entry_fill}_or{cfg.or_minutes}_{tag}_eq{int(args.robot_equity)}_no_filter_no_reset.csv"

    print(
        f"[batch] {len(syms)} syms | {dates[0]}..{dates[-1]} ({len(dates)} sessions) | "
        f"fill={args.entry_fill} or={cfg.or_minutes}m | no ML/BS | eq={args.robot_equity} | "
        f"macro={'off' if args.no_live_filters else 'on'}",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    t_all = time.time()
    for i, sym in enumerate(syms, 1):
        sym_n = sym.replace("USDT", "")
        sym_dates = universe_session_dates([sym], cfg)
        sym_dates = [d for d in sym_dates if (not lo or d >= lo) and (not hi or d <= hi)]
        if not sym_dates:
            print(f"[{i}/{len(syms)}] {sym_n} skip (no kline)", flush=True)
            rows.append({"symbol": sym_n, "sessions": 0, "opens": 0, "net_pnl_usdt": 0, "note": "no_data"})
            continue
        print(f"[{i}/{len(syms)}] {sym_n} ({len(sym_dates)} sessions) ...", flush=True)
        row = _run_one(
            sym,
            sym_dates,
            gate=gate,
            ranker=None,
            cfg=cfg,
            robot_equity=float(args.robot_equity),
            fee_bps=float(args.fee_bps),
            entry_fill=str(args.entry_fill),
        )
        print(
            f"       opens={row['opens']} net={row['net_pnl_usdt']:+.1f}U ret={row['return_pct']:+.1f}% "
            f"win={row['win_rate']:.0f}% fill_skip={row['fill_skips']}",
            flush=True,
        )
        rows.append({k: row[k] for k in SUMMARY_FIELDS if k in row})

    rows.sort(key=lambda r: float(r.get("net_pnl_usdt") or 0), reverse=True)
    total_net = round(sum(float(r.get("net_pnl_usdt") or 0) for r in rows), 2)
    total_opens = sum(int(r.get("opens") or 0) for r in rows)
    profitable = sum(1 for r in rows if float(r.get("net_pnl_usdt") or 0) > 0)

    payload = {
        "rule": "batch_symbol_live_sim",
        "entry_fill": args.entry_fill,
        "or_minutes": cfg.or_minutes,
        "gate_ml": False,
        "robot_reset": False,
        "robot_equity_usdt": float(args.robot_equity),
        "fee_bps_per_side": float(args.fee_bps),
        "macro_filter": cfg.macro_filter,
        "date_range": {"from": dates[0], "to": dates[-1], "sessions": len(dates)},
        "summary": {
            "symbols": len(syms),
            "symbols_with_data": sum(1 for r in rows if int(r.get("sessions") or 0) > 0),
            "profitable_symbols": profitable,
            "total_opens": total_opens,
            "total_net_pnl_usdt": total_net,
            "elapsed_sec": round(time.time() - t_all, 1),
        },
        "symbols": rows,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"DONE {len(syms)} syms | total_net={total_net}U | opens={total_opens} | profitable={profitable}/{len(rows)}")
    print(f"json -> {json_path}")
    print(f"csv  -> {csv_path}")
    print()
    print(f"{'SYM':<6} {'opens':>5} {'net_U':>10} {'ret%':>8} {'win%':>6} {'fill_skip':>9}")
    for r in rows:
        if int(r.get("sessions") or 0) == 0:
            continue
        print(
            f"{r['symbol']:<6} {int(r['opens']):>5} {float(r['net_pnl_usdt']):>+10.1f} "
            f"{float(r['return_pct']):>+8.1f} {float(r['win_rate']):>6.1f} {int(r.get('fill_skips') or 0):>9}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

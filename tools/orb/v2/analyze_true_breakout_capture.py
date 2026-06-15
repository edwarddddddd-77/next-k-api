#!/usr/bin/env python3
"""统计 60 日 43 标：池内真突破 vs Gate 捕捉率。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.core.config import OrbConfig  # noqa: E402
from orb.core.kline_cache import load_klines  # noqa: E402
from orb.core.session import extended_fetch_anchor_ms  # noqa: E402
from orb.core.resolve import pnl_usdt  # noqa: E402
from orb.core.session import session_anchor_ms, session_close_ms, session_day_str  # noqa: E402
from orb.ml.features import label_is_true_breakout  # noqa: E402
from orb.ml.horizon import label_hold_30m, resolve_horizon  # noqa: E402
from orb.ml.samples import parse_symbol_list  # noqa: E402
from orb.core.backtest import _daily_df_asof, _iter_scan_ms, _resolve_open, _SimOpen  # noqa: E402
from orb.core.paper import analyze_at_ms, in_regular_session, is_actionable  # noqa: E402
from tools.orb.ml.eval_live_gate import _ml_cfg  # noqa: E402
from tools.orb.v2.backtest_symbol import session_dates_from_cache  # noqa: E402
from orb.v2.paths import resolve_symbols_path  # noqa: E402


def _first_breakout_of_day(
    session_date: str,
    sym: str,
    cfg: OrbConfig,
) -> Optional[Dict[str, Any]]:
    tz = cfg.session_tz
    ts = pd.Timestamp(f"{session_date} 12:00:00", tz=tz)
    anchor = session_anchor_ms(int(ts.value // 1_000_000), tz=tz, session_open_time=cfg.session_open_time)
    close = session_close_ms(anchor, tz=tz, session_close_time=cfg.session_close_time)
    if close is None:
        close = anchor + 6 * 60 * 60 * 1000
    bar = cfg.bar_step_ms()
    scans = [
        s
        for s in _iter_scan_ms(anchor, close, bar_step_ms=bar)
        if session_day_str(s, tz=tz, session_open_time=cfg.session_open_time) == session_date
    ]
    warmup = cfg.daily_atr_warmup_ms() + bar * 96
    fetch_start = extended_fetch_anchor_ms(anchor, cfg) - warmup
    end_ms = close + bar * 4
    df5 = load_klines(sym, cfg.signal_interval, start_ms=fetch_start, end_ms=end_ms)
    df1 = load_klines(sym, "1m", start_ms=fetch_start, end_ms=end_ms)
    ddf = None
    if (cfg.sl_mode or "").strip().lower() == "atr_pct":
        ddf = load_klines(sym, "1d", start_ms=fetch_start - cfg.daily_atr_warmup_ms(), end_ms=end_ms)
    if df5.empty:
        return None

    for scan_ms in scans:
        if not in_regular_session(cfg, now_ms=scan_ms):
            continue
        ddf_asof = _daily_df_asof(ddf if ddf is not None else pd.DataFrame(), scan_ms)
        sig = analyze_at_ms(
            sym,
            cfg=cfg,
            now_ms=scan_ms,
            session_traded=False,
            daily_df=ddf_asof if ddf_asof is not None and not ddf_asof.empty else None,
            bot_equity_usdt=cfg.per_symbol_bot_equity(),
            df5=df5,
        )
        if not is_actionable(sig, cfg):
            continue
        entry_bo = int(sig.entry_bar_open_ms or 0)
        if entry_bo <= 0 or df1.empty:
            return None
        pos = _SimOpen(
            symbol=sym,
            side=str(sig.side),
            play=str(sig.play),
            entry=float(sig.price),
            sl=float(sig.sl_price),
            tp=float(sig.tp_price) if sig.tp_price is not None else None,
            entry_bar_open_ms=entry_bo,
            notional=1000.0,
            session_date=session_date,
            scan_open_ms=int(scan_ms),
        )
        out_eod, ex_eod, _, _ = _resolve_open(pos, df1, scan_ms=close + bar, cfg=cfg)
        pnl_eod = float(pnl_usdt(pos.side, pos.entry, ex_eod, pos.notional)) if out_eod and out_eod != "supersede" else 0.0
        true_eod = label_is_true_breakout(str(out_eod or ""), pnl_eod)

        h_out, h_ex, h_pr, h_note = resolve_horizon(
            df1,
            entry=float(sig.price),
            entry_bar_open_ms=entry_bo,
            side=str(sig.side),
            sl=float(sig.sl_price),
            tp=float(sig.tp_price) if sig.tp_price is not None else None,
            cfg=cfg,
        )
        h_pnl = float(pnl_usdt(pos.side, pos.entry, h_ex, pos.notional)) if h_out else 0.0
        hold30 = label_hold_30m(h_out, h_pr)

        return {
            "session_date": session_date,
            "symbol": sym,
            "side": str(sig.side),
            "entry": float(sig.price),
            "scan_open_ms": int(scan_ms),
            "true_eod": int(true_eod),
            "hold30_true": int(hold30),
            "pnl_eod": round(pnl_eod, 4),
            "pnl_hold30": round(h_pnl, 4),
        }
    return None


def main() -> int:
    load_env_oi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest-json", default=str(ROOT / "output/orb/v2/eval/universe_60d_eight-robots_backtest.json"))
    ap.add_argument("--symbols-file", default=str(resolve_symbols_path()))
    ap.add_argument("--last-sessions", type=int, default=60)
    args = ap.parse_args()

    bt = json.loads(Path(args.backtest_json).read_text(encoding="utf-8"))
    dates = [d["session_date"] for d in bt.get("days") or []]
    if not dates:
        print("No dates in backtest json")
        return 1

    syms = parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8"))
    cfg = _ml_cfg(compound_per_symbol=True)

    opened_keys = set()
    opened_true_eod = 0
    opened_true_h30 = 0
    for day in bt.get("days") or []:
        sd = day["session_date"]
        for r in day.get("opened") or []:
            sym = str(r.get("symbol") or "").upper()
            opened_keys.add((sd, sym))
            if r.get("true_breakout"):
                opened_true_eod += 1

    # Re-label opened with hold30 from pool map later
    pool: List[Dict[str, Any]] = []
    print(f"Scanning pool: {len(dates)} days x {len(syms)} syms ...", flush=True)
    for i, sd in enumerate(dates, 1):
        if i % 10 == 0 or i == len(dates):
            print(f"  [{i}/{len(dates)}] {sd}", flush=True)
        for sym in syms:
            row = _first_breakout_of_day(sd, sym, cfg)
            if row:
                pool.append(row)

    pool_true_eod = sum(r["true_eod"] for r in pool)
    pool_true_h30 = sum(r["hold30_true"] for r in pool)
    pool_total = len(pool)

    pool_h30_map = {(r["session_date"], r["symbol"]): r for r in pool}

    captured_h30 = 0
    captured_eod = 0
    missed_h30: List[Dict[str, Any]] = []
    for day in bt.get("days") or []:
        sd = day["session_date"]
        for r in day.get("opened") or []:
            sym = str(r.get("symbol") or "").upper()
            p = pool_h30_map.get((sd, sym))
            if not p:
                continue
            if p["hold30_true"]:
                captured_h30 += 1
                opened_true_h30 = captured_h30  # recount properly below
            if p["true_eod"]:
                captured_eod += 1

    opened_true_h30 = captured_h30
    for r in pool:
        if r["hold30_true"] and (r["session_date"], r["symbol"]) not in opened_keys:
            missed_h30.append(r)

    # Gate blocked: pool breakouts that existed but sym not in opened (includes gate skip + max 8 + robot full)
    not_opened = [r for r in pool if (r["session_date"], r["symbol"]) not in opened_keys]
    not_opened_true_h30 = sum(r["hold30_true"] for r in not_opened)

    # Per day: pool true h30 vs opened true h30
    day_pool_h30: Dict[str, int] = defaultdict(int)
    day_open_h30: Dict[str, int] = defaultdict(int)
    for r in pool:
        if r["hold30_true"]:
            day_pool_h30[r["session_date"]] += 1
    for sd, sym in opened_keys:
        p = pool_h30_map.get((sd, sym))
        if p and p["hold30_true"]:
            day_open_h30[sd] += 1

    summary = {
        "date_range": {"from": dates[0], "to": dates[-1], "sessions": len(dates)},
        "symbols": len(syms),
        "pool_first_breakouts": pool_total,
        "pool_true_eod_pnl_gt_0": pool_true_eod,
        "pool_true_hold30": pool_true_h30,
        "gate_opened_trades": len(opened_keys),
        "gate_opened_true_eod_backtest_label": opened_true_eod,
        "gate_opened_true_hold30": opened_true_h30,
        "capture_rate_hold30": round(opened_true_h30 / pool_true_h30, 4) if pool_true_h30 else 0,
        "capture_rate_eod": round(opened_true_eod / pool_true_eod, 4) if pool_true_eod else 0,
        "recall_of_pool_breakouts": round(len(opened_keys) / pool_total, 4) if pool_total else 0,
        "missed_hold30_not_opened": not_opened_true_h30,
        "avg_pool_true_hold30_per_day": round(pool_true_h30 / len(dates), 2),
        "avg_captured_hold30_per_day": round(opened_true_h30 / len(dates), 2),
    }

    print("\n=== 真突破捕捉率 ===\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n说明:")
    print("- pool_* = 43 标各自「当日首次 actionable 突破」（不论 Gate）")
    print("- hold30_true = 30min 内未触 SL 且 pnl>0（ML 训练标签）")
    print("- gate_opened_true_eod = 回测 JSON 里 true_breakout（EoD 结算 pnl>0）")
    print(f"- 未捕捉的 hold30 真突破: {len(missed_h30)} 笔（Gate/8 单上限/robot 槽位）")

    stem = Path(args.backtest_json).stem
    out_path = Path(args.backtest_json).with_name(f"{stem}_capture_analysis.json")
    out_path.write_text(
        json.dumps({"summary": summary, "missed_hold30_sample": missed_h30[:30]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

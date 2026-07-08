#!/usr/bin/env python3
"""按组合日内图示回测：上午 ORB/KK + GTL | 午间跳过/VWAP | 尾盘 PowerHour。"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402

load_env_oi()

import pandas as pd  # noqa: E402

from orb.core.config import OrbConfig  # noqa: E402
from orb.core.fees import trade_fee_usdt  # noqa: E402
from orb.core.kline_cache import norm_symbol, session_dates_from_cache  # noqa: E402
from orb.core.resolve import pnl_usdt  # noqa: E402
from orb.core.session import is_trading_session, session_close_ms  # noqa: E402
from orb.core.signals import _session_vwap  # noqa: E402
from orb.core.symbols import parse_symbol_list  # noqa: E402
from orb.cta.execution import entry_fill_px, market_exit_fill_px, stop_exit_fill_px  # noqa: E402
from orb.gtl.engine import compute_gtl_dataframe  # noqa: E402
from orb.gtl.resample import resample_ohlcv  # noqa: E402
from orb.core.symbols_path import resolve_symbols_path  # noqa: E402
from tools.cta.compare_kk_orb_pool7 import (  # noqa: E402
    POOL7,
    _in_entry_window,
    _load_range,
    backtest_orb_honest,
    run_kk,
)
from tools.cta.research_gtl_downstream import _gtl_bias, _session_anchors  # noqa: E402

MORNING_START = (9, 45)
MORNING_END = (12, 0)
MIDDAY_START = (12, 0)
MIDDAY_END = (14, 0)
POWER_START = (15, 0)
POWER_END = (15, 55)


@dataclass(frozen=True)
class Lane:
    key: str
    label: str
    diagram: str


LANES: List[Lane] = [
    Lane("morning_orb_gtl", "① 上午 ORB + GTL", "9:45–12:00 突破，GTL 挡反向"),
    Lane("kk", "② 上午 KK 趋势", "9:30–12:00 后不开仓"),
    Lane("midday_skip", "③ 午间不交易", "12:00–14:00 空仓（默认）"),
    Lane("midday_vwap", "③ 午间 VWAP 轻仓", "12:00–14:00 偏离 VWAP 回归"),
    Lane("power_orb_gtl", "④ 尾盘 PowerHour", "15:00–15:55 ORB + GTL"),
]


def _wait_1m_fill(
    df_1m: pd.DataFrame,
    signal_ms: int,
    side: str,
    trigger: float,
    session_close_ms_val: int,
    slip_bps: float,
    bar_step_ms: int,
) -> Tuple[Optional[int], Optional[float]]:
    start_ms = int(signal_ms) + int(bar_step_ms)
    sub = df_1m[(df_1m["open_time"] >= start_ms) & (df_1m["open_time"] < session_close_ms_val)]
    side_u = str(side).upper()
    for _, row in sub.iterrows():
        h, l = float(row["high"]), float(row["low"])
        if side_u == "LONG" and h >= trigger:
            return int(row["open_time"]), entry_fill_px(1, trigger, slip_bps)
        if side_u == "SHORT" and l <= trigger:
            return int(row["open_time"]), entry_fill_px(-1, trigger, slip_bps)
    return None, None


def _exit_px_with_slip(side: str, outcome: str, raw_exit: float, bar_open: float, slip_bps: float) -> float:
    side_i = 1 if str(side).upper() == "LONG" else -1
    if outcome in ("loss", "win"):
        return stop_exit_fill_px(side_i, raw_exit, bar_open=bar_open, slip_bps=slip_bps)
    return market_exit_fill_px(side_i, raw_exit, slip_bps)


def _gtl_allows(side: str, bias: str) -> bool:
    return not (side == "LONG" and bias == "short") and not (side == "SHORT" and bias == "long")


def backtest_midday_vwap(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    gtl: pd.DataFrame,
    df_30m: pd.DataFrame,
    cfg: OrbConfig,
    *,
    equity: float,
    slip_bps: float,
    band_pct: float = 0.003,
    size_frac: float = 0.5,
) -> Dict[str, Any]:
    trades: List[float] = []
    filtered = 0
    bar_step = cfg.bar_step_ms()

    for anchor in _session_anchors(df_1m, cfg):
        close_ms = session_close_ms(anchor, tz=cfg.session_tz, session_close_time=cfg.session_close_time)
        if close_ms is None:
            continue
        bias = _gtl_bias(gtl, df_30m, anchor)
        session_traded = False
        sess_1m = df_1m[(df_1m["open_time"] >= anchor) & (df_1m["open_time"] < close_ms)]
        sess_5m = df_5m[(df_5m["open_time"] >= anchor) & (df_5m["open_time"] < close_ms)]

        for _, bar in sess_5m.iterrows():
            ms = int(bar["open_time"])
            if not _in_entry_window(ms, cfg.session_tz, start_hm=MIDDAY_START, end_hm=MIDDAY_END):
                continue
            if session_traded:
                break
            if not is_trading_session(
                ms,
                tz=cfg.session_tz,
                session_open_time=cfg.session_open_time,
                session_close_time=cfg.session_close_time,
                market=cfg.market,
            ):
                continue
            hist = sess_1m[sess_1m["open_time"] <= ms]
            if len(hist) < 5:
                continue
            vwap = _session_vwap(hist)
            if vwap <= 0:
                continue
            px = float(bar["close"])
            band = vwap * band_pct
            side = "LONG" if px <= vwap - band else "SHORT" if px >= vwap + band else ""
            if not side:
                continue
            if not _gtl_allows(side, bias):
                filtered += 1
                continue
            if side == "LONG":
                sl, tp = vwap - band * 2, vwap
            else:
                sl, tp = vwap + band * 2, vwap
            fill_ms, fill_px = _wait_1m_fill(df_1m, ms, side, px, close_ms, slip_bps, bar_step)
            if fill_ms is None or fill_px is None:
                continue
            exit_deadline = int(
                pd.Timestamp(ms, unit="ms", tz=cfg.session_tz)
                .replace(hour=MIDDAY_END[0], minute=MIDDAY_END[1], second=0, microsecond=0)
                .value
                // 1_000_000
            )
            sub = df_1m[(df_1m["open_time"] >= fill_ms) & (df_1m["open_time"] < min(close_ms, exit_deadline))]
            outcome, exit_px = "eod", float(fill_px)
            for _, row in sub.iterrows():
                h, l = float(row["high"]), float(row["low"])
                if side == "LONG":
                    if l <= sl:
                        outcome, exit_px = "loss", sl
                        break
                    if h >= tp:
                        outcome, exit_px = "win", tp
                        break
                elif h >= sl:
                    outcome, exit_px = "loss", sl
                    break
                elif l <= tp:
                    outcome, exit_px = "win", tp
                    break
            else:
                if not sub.empty:
                    exit_px = float(sub.iloc[-1]["close"])
            exit_adj = _exit_px_with_slip(side, outcome, float(exit_px), float(fill_px), slip_bps)
            notion = equity * size_frac * 0.1
            if notion <= 0:
                continue
            gross = pnl_usdt(side, float(fill_px), exit_adj, notion)
            fee = trade_fee_usdt(
                notional_usdt=notion,
                entry_mode="breakout",
                maker_bps=cfg.fee_maker_bps,
                taker_bps=cfg.fee_taker_bps,
            )
            trades.append(gross - fee)
            session_traded = True
            break

    wins = sum(1 for x in trades if x > 0)
    return {
        "trades": len(trades),
        "filtered": filtered,
        "sum_usd": round(float(sum(trades)), 2) if trades else 0.0,
        "win_rate": round(wins / len(trades), 3) if trades else 0.0,
    }


def load_symbol(sym: str, lo: str, hi: str, cfg: OrbConfig) -> pd.DataFrame:
    fetch_lo = (pd.Timestamp(lo) - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    df_1m = _load_range(sym, fetch_lo, hi, cfg)
    if df_1m.empty:
        return df_1m
    lo_ms = int(pd.Timestamp(lo, tz=cfg.session_tz).value // 1_000_000)
    hi_ms = int(
        (pd.Timestamp(hi, tz=cfg.session_tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).value
        // 1_000_000
    )
    return df_1m[(df_1m["open_time"] >= lo_ms - 30 * 86400 * 1000) & (df_1m["open_time"] <= hi_ms)].copy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagram-aligned combined intraday research")
    ap.add_argument("--from-date", default="2026-02-01")
    ap.add_argument("--to-date", default="2026-06-30")
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--kk-equity", type=float, default=14.0, help="KK 线默认每机器人 14U")
    ap.add_argument("--slip-bps", type=float, default=5.0)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--symbols-file", default="")
    args = ap.parse_args()

    cfg = OrbConfig.from_env()
    if args.symbols.strip():
        syms = [norm_symbol(s) for s in parse_symbol_list(args.symbols)]
    elif args.symbols_file.strip():
        syms = [norm_symbol(s) for s in parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8"))]
    else:
        ext = ROOT / "config" / "trading_orb" / "symbols.txt"
        src = ext if ext.is_file() else Path(resolve_symbols_path())
        syms = [norm_symbol(s) for s in parse_symbol_list(src.read_text(encoding="utf-8"))]
    if not syms:
        syms = [norm_symbol(s) for s in POOL7]

    lo, hi = args.from_date, args.to_date
    eq, kk_eq, slip = float(args.equity), float(args.kk_equity), float(args.slip_bps)

    orb_cfg = OrbConfig.from_env()
    orb_cfg.macro_filter = True
    orb_cfg.resolve_at_session_close = True
    orb_cfg.momentum_filter = False

    print(f"=== 组合日内（按图示）| {len(syms)} symbols | {lo}..{hi} ===")
    print("symbols:", ", ".join(s.replace("USDT", "") for s in syms))
    print()
    for lane in LANES:
        print(f"  {lane.label:22s}  {lane.diagram}")
    print()
    print("全程：ORB 用 GTL 挡反向 | 无 5 日动量 | EoD 15:55 强平")
    print(f"equity ORB={eq}U | KK={kk_eq}U/机器人 | slip={slip}bps\n")

    totals: Dict[str, float] = {lane.key: 0.0 for lane in LANES}
    counts: Dict[str, int] = {lane.key: 0 for lane in LANES}
    t0 = time.time()

    for sym in syms:
        label = sym.replace("USDT", "")
        df_1m = load_symbol(sym, lo, hi, cfg)
        if df_1m.empty:
            print(f"  {label:5s}  no_data", flush=True)
            continue
        df_5m = resample_ohlcv(df_1m, "5m")
        df_30m = resample_ohlcv(df_1m, "30m")
        gtl = compute_gtl_dataframe(df_30m, lookback=23, vol_window=500)

        morning = backtest_orb_honest(
            df_1m, df_5m, gtl, df_30m, orb_cfg,
            equity=eq, slip_bps=slip, gtl_mode="block",
            entry_start_hm=MORNING_START, entry_end_hm=MORNING_END,
        )
        totals["morning_orb_gtl"] += float(morning.get("sum_usd", 0) or 0)
        counts["morning_orb_gtl"] += int(morning.get("trades", 0) or 0)
        totals["midday_skip"] = totals["morning_orb_gtl"]
        counts["midday_skip"] = counts["morning_orb_gtl"]

        dates = [d for d in session_dates_from_cache(sym, cfg) if lo <= d <= hi]
        kk = run_kk(sym, dates, cfg, kk_eq) if dates else {"sum_usd": 0, "trades": 0}
        totals["kk"] += float(kk.get("sum_usd", 0) or 0)
        counts["kk"] += int(kk.get("trades", 0) or 0)

        vwap = backtest_midday_vwap(df_1m, df_5m, gtl, df_30m, orb_cfg, equity=eq, slip_bps=slip)
        totals["midday_vwap"] += float(vwap.get("sum_usd", 0) or 0)
        counts["midday_vwap"] += int(vwap.get("trades", 0) or 0)

        power = backtest_orb_honest(
            df_1m, df_5m, gtl, df_30m, orb_cfg,
            equity=eq, slip_bps=slip, gtl_mode="block",
            entry_start_hm=POWER_START, entry_end_hm=POWER_END,
        )
        totals["power_orb_gtl"] += float(power.get("sum_usd", 0) or 0)
        counts["power_orb_gtl"] += int(power.get("trades", 0) or 0)

        print(
            f"  {label:5s}  "
            f"①{morning.get('sum_usd', 0):+.0f}U  "
            f"②KK{kk.get('sum_usd', 0):+.0f}U  "
            f"③VWAP{vwap.get('sum_usd', 0):+.0f}U  "
            f"④Power{power.get('sum_usd', 0):+.0f}U",
            flush=True,
        )

    print()
    print(f"{'图示方案':22s} {'笔数':>6s} {'净盈亏U':>10s}")
    print("-" * 42)
    for lane in LANES:
        print(f"{lane.label:22s} {counts[lane.key]:6d} {totals[lane.key]:+10.2f}")
    print()
    print("推荐执行（按图）:")
    print("  · 上午：① ORB+GTL  或  ② KK（二选一，不叠加）")
    print("  · 午间：默认 ③ 不交易；VWAP 需单独验证后再加")
    print("  · 尾盘：④ 单独通过后，再考虑与上午叠加")
    print(f"\nelapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

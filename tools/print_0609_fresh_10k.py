#!/usr/bin/env python3
"""6/9 only: each bot starts 10,000U (fresh deploy)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orb.backtest import _load_range, _iter_scan_ms  # noqa: E402
from orb.config import DEFAULT_SYMBOLS, OrbConfig  # noqa: E402
from orb.paper import analyze_at_ms, is_actionable  # noqa: E402
from orb.resolve import pnl_usdt, resolve_forward  # noqa: E402
from orb.session import session_day_str  # noqa: E402

DAY = "2026-06-09"
INIT = 10_000.0


def main() -> None:
    import pandas as pd

    cfg = OrbConfig.for_backtest()
    tz = cfg.session_tz
    day0 = int(pd.Timestamp(f"{DAY} 00:00", tz=tz).value // 1_000_000)
    day1 = int(pd.Timestamp(f"{DAY} 23:59", tz=tz).value // 1_000_000)
    bar = cfg.bar_step_ms()
    fetch_start = day0 - bar * 96

    syms = [s.strip().upper() + ("USDT" if not s.strip().upper().endswith("USDT") else "")
            for s in DEFAULT_SYMBOLS.split(",") if s.strip()]

    oc = {"session_close": "收盘", "loss": "止损", "win": "止盈"}

    print(f"=== {DAY}  fresh deploy: wallet={INIT:,.0f}U each (no prior compound) ===\n")

    for sym in syms:
        base = sym.replace("USDT", "")
        df5 = _load_range(sym, cfg.signal_interval, fetch_start, day1)
        daily = _load_range(sym, "1d", day0 - cfg.daily_atr_warmup_ms(), day1)
        df1 = _load_range(sym, "1m", fetch_start, day1)
        scans = [s for s in _iter_scan_ms(day0, day1, bar_step_ms=bar)
                 if session_day_str(s, tz=tz, session_open_time=cfg.session_open_time) == DAY]

        opened = False
        for scan_ms in scans:
            ddf = daily[daily["open_time"] <= scan_ms] if not daily.empty else daily
            sig = analyze_at_ms(sym, cfg=cfg, now_ms=scan_ms, daily_df=ddf if not ddf.empty else None,
                                bot_equity_usdt=INIT, df5=df5, session_traded=False)
            if not is_actionable(sig, cfg):
                continue
            entry_bo = int(sig.entry_bar_open_ms or 0)
            if entry_bo <= 0:
                continue
            eod_ms = int(pd.Timestamp(f"{DAY} 16:00:05", tz=tz).value // 1_000_000)
            out, ex_px, _, _, _ = resolve_forward(
                df1, entry=float(sig.price), entry_bar_open_ms=entry_bo,
                side=str(sig.side), sl=float(sig.sl_price),
                tp=float(sig.tp_price) if sig.tp_price else None,
                hist_end_ms=eod_ms, bar_step_ms=bar, cfg=cfg,
            )
            if out is None:
                continue
            notion = float(sig.paper_notional_usdt or 0)
            pu = pnl_usdt(str(sig.side), float(sig.price), ex_px, notion)
            print(
                f"{base:5} {sig.side:5} {oc.get(out, out):4}  "
                f"entry={sig.price:.4f} sl={sig.sl_price:.4f} exit={ex_px:.4f}  "
                f"notional={notion:,.0f}  pnl={pu:+.2f}U  wallet_after={INIT + pu:,.2f}U"
            )
            opened = True
            break
        if not opened:
            print(f"{base:5} 无交易  wallet={INIT:,.0f}U")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify INTC 2026-06-09: aligned SL vs 1m wick (live/backtest parity)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orb.backtest import _load_range, _daily_df_asof, _resolve_open, _SimOpen  # noqa: E402
from orb.config import OrbConfig  # noqa: E402
from orb.paper import analyze_at_ms  # noqa: E402

SCAN_MS = 1781013005000  # 2026-06-09 09:50:05 ET
ENTRY_BO = 1781012700000  # 09:45 bar
SYM = "INTCUSDT"
LIVE_SL = 112.4469  # old live (misaligned ATR window)


def main() -> None:
    cfg = OrbConfig.for_backtest()
    tz = cfg.session_tz
    df5 = _load_range(SYM, cfg.signal_interval, SCAN_MS - 86400000, SCAN_MS + 86400000)
    daily = _load_range(SYM, "1d", SCAN_MS - 40 * 86400000, SCAN_MS + 86400000)
    ddf = _daily_df_asof(daily, SCAN_MS)
    sig = analyze_at_ms(SYM, cfg=cfg, now_ms=SCAN_MS, daily_df=ddf, bot_equity_usdt=9664, df5=df5)
    df1 = _load_range(SYM, "1m", ENTRY_BO, SCAN_MS + 8 * 3600000)
    max_high = float(df1["high"].max()) if not df1.empty else 0.0
    pos = _SimOpen(
        symbol=SYM,
        side="SHORT",
        play="ORB_BREAKOUT_SHORT",
        entry=float(sig.price),
        sl=float(sig.sl_price),
        tp=None,
        entry_bar_open_ms=ENTRY_BO,
        notional=21000.0,
        session_date="2026-06-09",
        scan_open_ms=SCAN_MS,
    )
    eod_ms = int(pd.Timestamp("2026-06-09 16:00:05", tz=tz).value // 1_000_000)
    out, ex_px, note, _ = _resolve_open(pos, df1, scan_ms=eod_ms, cfg=cfg)
    print(f"aligned SL: {sig.sl_price:.4f}  (old live SL: {LIVE_SL})")
    print(f"entry: {sig.price}  1m session max high: {max_high:.4f}")
    print(f"old live stopped: {max_high >= LIVE_SL}  aligned SL stopped: {max_high >= float(sig.sl_price)}")
    print(f"EOD resolve: {out} @ {ex_px:.4f} ({note})")


if __name__ == "__main__":
    main()

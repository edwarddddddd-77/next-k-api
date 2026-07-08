#!/usr/bin/env python3
"""对照实盘截图时段 vs 回测（ET 12:35-15:57 = 北京 00:35-03:57）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi

load_env_oi()

import pandas as pd  # noqa: E402

from binance_fapi import fetch_klines_forward, klines_to_df  # noqa: E402
from orb.core.config import OrbConfig  # noqa: E402
from orb.core.kline_cache import norm_symbol  # noqa: E402
from orb.core.session import session_anchor_ms  # noqa: E402
from orb.core.symbols import parse_symbol_list  # noqa: E402
from orb.cta.engine import run_cta_backtest  # noqa: E402
from orb.cta.registry import CTA_STRATEGIES, cta_config_for_strategy  # noqa: E402
from orb.core.symbols_path import resolve_symbols_path  # noqa: E402

KK = dict(
    compound=True,
    rth_only=True,
    eod_flat=True,
    exit_hour=15,
    exit_minute=55,
    slip_bps_entry=5.0,
    slip_bps_exit=5.0,
    max_notional_usdt=0.0,
)

# 用户截图里的实盘成交（北京时间 7/3）
LIVE_CLOSES = [
    ("00:40:39", "INTC", "SHORT", -0.10),
    ("00:40:04", "SNDK", "SHORT", +0.04),
    ("00:40:04", "SOXL", "SHORT", 0.00),
    ("00:42:19", "HOOD", "SHORT", -0.11),
    ("00:57:59", "MSTR", "LONG", -0.40),
    ("01:00:01", "SOXL", "SHORT", -0.03),
    ("01:25:33", "MSTR", "LONG", -0.42),
    ("01:54:03", "SOXL", "SHORT", -0.51),
    ("01:54:40", "CRCL", "SHORT", -0.32),
    ("01:54:35", "INTC", "SHORT", -0.27),
    ("02:06:02", "SNDK", "SHORT", +0.09),
    ("02:28:23", "SNDK", "SHORT", -0.21),
    ("02:35:01", "SOXL", "LONG", -0.77),
    ("03:03:20", "MSTR", "LONG", -0.41),
    ("03:05:10", "COIN", "LONG", -0.22),
    ("03:03:57", "HOOD", "LONG", -0.05),
    ("03:34:37", "HOOD", "SHORT", -0.37),
    ("03:42:55", "MSTR", "SHORT", -0.38),
    ("03:57:00", "SNDK", "LONG", -0.06),
]


def fetch(sym: str, d0: str, d1: str, cfg: OrbConfig) -> pd.DataFrame:
    tz = cfg.session_tz
    lo = pd.Timestamp(d0, tz=tz)
    hi = pd.Timestamp(d1, tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    df = klines_to_df(fetch_klines_forward(sym, "1m", int(lo.value // 1_000_000), int(hi.value // 1_000_000)))
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def sess_day(ms: int, cfg: OrbConfig) -> str:
    ts = pd.Timestamp(int(ms), unit="ms", tz=cfg.session_tz)
    a = session_anchor_ms(int(ts.value // 1_000_000), tz=cfg.session_tz, session_open_time=cfg.session_open_time)
    return pd.Timestamp(a, unit="ms", tz=cfg.session_tz).strftime("%Y-%m-%d")


def cn_hms(ms: int) -> str:
    return pd.Timestamp(int(ms), unit="ms", tz="America/New_York").tz_convert("Asia/Shanghai").strftime("%H:%M:%S")


def main() -> None:
    cfg = OrbConfig.from_env()
    meta = CTA_STRATEGIES["king_keltner"]
    symbols = [norm_symbol(s) for s in parse_symbol_list(Path(resolve_symbols_path()).read_text(encoding="utf-8"))]
    session = "2026-07-02"
    win_lo = pd.Timestamp("2026-07-02 12:35:00", tz="America/New_York")
    win_hi = pd.Timestamp("2026-07-02 15:57:00", tz="America/New_York")
    lo_ms, hi_ms = int(win_lo.value // 1_000_000), int(win_hi.value // 1_000_000)

    all_trades = []
    for sym in symbols:
        df = fetch(sym, "2026-06-25", session, cfg)
        out = run_cta_backtest(
            df,
            strategy_fn=meta["fn"],
            orb_cfg=cfg,
            cta_cfg=cta_config_for_strategy("king_keltner", equity_usdt=14, risk_pct=0.01, **KK),
            warmup=25,
        )
        for t in out["trades"]:
            if sess_day(int(t["ms"]), cfg) != session:
                continue
            if not (lo_ms <= int(t["ms"]) <= hi_ms):
                continue
            all_trades.append({**t, "sym": sym.replace("USDT", ""), "cn": cn_hms(int(t["ms"]))})

    closes = [t for t in all_trades if t["event"] == "close"]
    opens = [t for t in all_trades if t["event"] == "open"]
    net = sum(float(t["pnl_usdt"]) for t in closes)
    wins = sum(1 for t in closes if float(t["pnl_usdt"]) > 0)

    live_net = sum(p for _, _, _, p in LIVE_CLOSES)
    live_wins = sum(1 for _, _, _, p in LIVE_CLOSES if p > 0)

    print("=== 时段：北京 00:35-03:57（ET 12:35-15:57）session 2026-07-02 ===\n")
    print(f"实盘（截图 {len(LIVE_CLOSES)} 笔平仓）: net={live_net:+.2f}U  胜={live_wins}/{len(LIVE_CLOSES)}")
    print(f"回测（同窗口）:           net={net:+.4f}U  胜={wins}/{len(closes)}  开={len(opens)} 平={len(closes)}\n")

    print("--- 回测成交 ---")
    for t in sorted(all_trades, key=lambda x: int(x["ms"])):
        if t["event"] == "open":
            print(
                f"  {t['cn']} {t['sym']:4s} OPEN  {t['side']:5s} "
                f"entry={float(t['entry']):.4f} notional={float(t['notional_usdt']):.2f}U"
            )
        else:
            print(
                f"  {t['cn']} {t['sym']:4s} CLOSE {t['side']:5s} "
                f"{t.get('outcome', '?'):8s} pnl={float(t['pnl_usdt']):+.4f}U"
            )

    print("\n--- 实盘 vs 回测 笔数 ---")
    from collections import Counter

    live_c = Counter((s, d) for _, s, d, _ in LIVE_CLOSES)
    bt_c = Counter((t["sym"], t["side"]) for t in closes)
    all_syms = sorted(set(live_c) | set(bt_c))
    print(f"{'symbol':6s} {'side':5s}  live  backtest")
    for key in all_syms:
        print(f"{key[0]:6s} {key[1]:5s}  {live_c.get(key,0):4d}  {bt_c.get(key,0):4d}")


if __name__ == "__main__":
    main()

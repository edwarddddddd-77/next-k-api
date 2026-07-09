#!/usr/bin/env python3
"""Aberration 布林带突破 CTA — 币安 USDT 永续回测（对齐 orb/aberration/core + vnpy lane）。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orb.aberration.config import AberrationVnpyConfig  # noqa: E402
from orb.aberration.core import aberration_action, aberration_bands  # noqa: E402
from orb.aberration.paths import resolve_aberration_symbols_path  # noqa: E402
from orb.aberration.vnpy.sizing import fixed_size_for_aberration  # noqa: E402
from orb.core.symbols import parse_symbol_list  # noqa: E402

FAPI = "https://fapi.binance.com"
REPORT_PATH = ROOT / "aberration_backtest_report.json"


@dataclass
class Trade:
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    entry_px: float
    exit_px: float
    volume: float
    pnl_usdt: float
    pnl_pct: float
    exit_reason: str


def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    rows: List[list] = []
    cur = start
    while cur < end:
        resp = requests.get(
            f"{FAPI}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end, "limit": 1500},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1][0]) + 1
        if nxt <= cur:
            break
        cur = nxt

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote", "ignore",
        ],
    )
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.drop_duplicates("open_time").set_index("open_time").sort_index()


def _interval_hours(bar_hours: int) -> str:
    h = max(1, min(24, int(bar_hours)))
    if h == 1:
        return "1h"
    if h == 4:
        return "4h"
    if h >= 24:
        return "1d"
    return f"{h}h"


def run_symbol(
    ohlc: pd.DataFrame,
    symbol: str,
    cfg: AberrationVnpyConfig,
    *,
    fee_bps: float,
    slip_bps: float,
) -> tuple[List[Trade], float]:
    """单品种回测；返回 trades 与期末权益。"""
    if ohlc.empty or len(ohlc) < cfg.n_period + 2:
        return [], float(cfg.equity_usdt)

    fee = float(fee_bps) / 10_000.0
    slip = float(slip_bps) / 10_000.0
    equity = float(cfg.equity_usdt)
    pos: float = 0.0
    entry_px = 0.0
    vol = 0.0
    closes: List[float] = []
    trades: List[Trade] = []
    open_ts: Optional[pd.Timestamp] = None

    for ts, row in ohlc.iterrows():
        close = float(row["close"])
        prior = list(closes)
        bands = aberration_bands(
            prior,
            n_period=int(cfg.n_period),
            k_up=float(cfg.k_up),
            k_down=float(cfg.k_down),
        )
        closes.append(close)
        if bands is None:
            continue

        upper, middle, lower = bands
        act = aberration_action(pos, close, upper, middle, lower)
        if not act:
            continue

        ts_s = ts.strftime("%Y-%m-%d %H:%M UTC")

        def _close_position(reason: str, px: float) -> None:
            nonlocal pos, entry_px, vol, equity, open_ts
            if pos == 0 or vol <= 0:
                return
            side = "LONG" if pos > 0 else "SHORT"
            fill = px * (1.0 - slip if side == "LONG" else 1.0 + slip)
            notional = entry_px * vol
            if side == "LONG":
                gross = (fill - entry_px) / entry_px * notional
            else:
                gross = (entry_px - fill) / entry_px * notional
            fee_usd = notional * fee + fill * vol * fee
            pnl = gross - fee_usd
            equity += pnl
            trades.append(
                Trade(
                    symbol=symbol,
                    side=side,
                    entry_time=open_ts.strftime("%Y-%m-%d %H:%M UTC") if open_ts is not None else ts_s,
                    exit_time=ts_s,
                    entry_px=entry_px,
                    exit_px=fill,
                    volume=vol,
                    pnl_usdt=round(pnl, 4),
                    pnl_pct=round(pnl / max(1e-9, notional) * 100.0, 4),
                    exit_reason=reason,
                )
            )
            pos = 0.0
            entry_px = 0.0
            vol = 0.0
            open_ts = None

        def _open_position(side_sign: float, reason: str) -> None:
            nonlocal pos, entry_px, vol, equity, open_ts
            order_vol = fixed_size_for_aberration(cfg, close, equity_usdt=equity)
            if order_vol <= 0:
                return
            fill = close * (1.0 + slip if side_sign > 0 else 1.0 - slip)
            notional = fill * order_vol
            equity -= notional * fee
            pos = order_vol if side_sign > 0 else -order_vol
            entry_px = fill
            vol = order_vol
            open_ts = ts

        if act == "close_long":
            _close_position("middle_band", close)
        elif act == "close_short":
            _close_position("middle_band", close)
        elif act == "long":
            if pos < 0:
                _close_position("flip_to_long", close)
            if pos == 0:
                _open_position(1.0, "breakout_upper")
        elif act == "short":
            if pos > 0:
                _close_position("flip_to_short", close)
            if pos == 0:
                _open_position(-1.0, "breakout_lower")

    if pos != 0 and vol > 0:
        last_px = float(ohlc.iloc[-1]["close"])
        _close_position("end_of_data", last_px)

    return trades, equity


def summarize(trades: List[Trade], start_equity: float, end_equity: float, meta: Dict[str, Any]) -> Dict[str, Any]:
    if not trades:
        return {
            **meta,
            "trades": 0,
            "win_rate_pct": 0.0,
            "total_return_pct": 0.0,
            "total_pnl_usdt": 0.0,
            "profit_factor": 0.0,
            "avg_pnl_usdt": 0.0,
        }

    wins = sum(1 for t in trades if t.pnl_usdt > 0)
    gross_win = sum(t.pnl_usdt for t in trades if t.pnl_usdt > 0)
    gross_loss = abs(sum(t.pnl_usdt for t in trades if t.pnl_usdt < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)

    return {
        **meta,
        "trades": len(trades),
        "win_rate_pct": round(wins / len(trades) * 100.0, 2),
        "total_return_pct": round((end_equity / start_equity - 1.0) * 100.0, 2),
        "total_pnl_usdt": round(end_equity - start_equity, 2),
        "end_equity_usdt": round(end_equity, 2),
        "profit_factor": round(float(pf), 3) if math.isfinite(pf) else None,
        "avg_pnl_usdt": round(sum(t.pnl_usdt for t in trades) / len(trades), 4),
        "long_trades": sum(1 for t in trades if t.side == "LONG"),
        "short_trades": sum(1 for t in trades if t.side == "SHORT"),
    }


def load_symbols(args_symbols: Optional[List[str]]) -> List[str]:
    if args_symbols:
        return parse_symbol_list(",".join(args_symbols))
    path = resolve_aberration_symbols_path()
    if path.is_file():
        return parse_symbol_list(path.read_text(encoding="utf-8"))
    return parse_symbol_list("BTCUSDT,SOLUSDT,BNBUSDT,AVAXUSDT,LINKUSDT")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Aberration BB breakout backtest (Binance perp)")
    p.add_argument("--symbols", nargs="+", default=None, help="e.g. BTCUSDT SOLUSDT")
    p.add_argument("--days", type=int, default=90, help="回测天数")
    p.add_argument("--bar-hours", type=int, default=None, help="K 线周期小时数，默认读配置 1h")
    p.add_argument("--equity", type=float, default=None, help="每品种初始权益 USDT")
    p.add_argument("--fee-bps", type=float, default=4.0, help="单边手续费 bps")
    p.add_argument("--slip-bps", type=float, default=2.0, help="滑点 bps")
    p.add_argument("--report", type=Path, default=REPORT_PATH)
    args = p.parse_args(argv)

    cfg = AberrationVnpyConfig.from_env()
    if args.bar_hours is not None:
        cfg = AberrationVnpyConfig(**{**cfg.__dict__, "bar_hours": args.bar_hours})
    if args.equity is not None:
        cfg = AberrationVnpyConfig(**{**cfg.__dict__, "equity_usdt": args.equity})

    symbols = load_symbols(args.symbols)
    interval = _interval_hours(cfg.bar_hours)

    print(f"Aberration 回测 | {interval} | {args.days}d | 品种 {len(symbols)}")
    print(
        f"  n={cfg.n_period} k_up={cfg.k_up} k_down={cfg.k_down} "
        f"equity={cfg.equity_usdt} lev={cfg.leverage} pct={cfg.position_pct}"
    )
    print()

    per_symbol: List[Dict[str, Any]] = []
    all_trades: List[Dict[str, Any]] = []

    for sym in symbols:
        print(f"  fetch {sym} {interval} ...", flush=True)
        ohlc = fetch_klines(sym, interval, args.days)
        if ohlc.empty:
            print(f"    skip {sym}: no data")
            continue
        trades, end_eq = run_symbol(
            ohlc, sym, cfg, fee_bps=args.fee_bps, slip_bps=args.slip_bps
        )
        meta = summarize(
            trades,
            float(cfg.equity_usdt),
            end_eq,
            {
                "symbol": sym,
                "interval": interval,
                "bars": len(ohlc),
                "start": str(ohlc.index[0]),
                "end": str(ohlc.index[-1]),
            },
        )
        per_symbol.append(meta)
        for t in trades:
            all_trades.append({**t.__dict__})

        print(
            f"    {sym}: trades={meta['trades']} win%={meta['win_rate_pct']:.1f} "
            f"ret%={meta['total_return_pct']:.2f} pnl=${meta['total_pnl_usdt']:.2f}"
        )

    if not per_symbol:
        print("No results.")
        return 1

    total_start = float(cfg.equity_usdt) * len(per_symbol)
    total_end = sum(float(r["end_equity_usdt"]) for r in per_symbol)
    portfolio = {
        "symbols": len(per_symbol),
        "total_return_pct": round((total_end / total_start - 1.0) * 100.0, 2),
        "total_pnl_usdt": round(total_end - total_start, 2),
        "total_trades": sum(int(r["trades"]) for r in per_symbol),
        "avg_win_rate_pct": round(
            sum(float(r["win_rate_pct"]) for r in per_symbol) / len(per_symbol), 2
        ),
    }

    print()
    print(f"组合合计: {portfolio['symbols']} 品种 | trades={portfolio['total_trades']} | "
          f"ret%={portfolio['total_return_pct']:.2f} | pnl=${portfolio['total_pnl_usdt']:.2f}")

    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "n_period": cfg.n_period,
            "k_up": cfg.k_up,
            "k_down": cfg.k_down,
            "bar_hours": cfg.bar_hours,
            "interval": interval,
            "equity_usdt": cfg.equity_usdt,
            "leverage": cfg.leverage,
            "position_pct": cfg.position_pct,
            "days": args.days,
            "fee_bps": args.fee_bps,
            "slip_bps": args.slip_bps,
        },
        "portfolio": portfolio,
        "per_symbol": per_symbol,
        "trades": all_trades[-200:],
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

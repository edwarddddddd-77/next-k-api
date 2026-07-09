#!/usr/bin/env python3
"""复合 CTA v2 回测 — 移植自 fmz-composite-cta-reference/local/fmz_composite_cta.js

四子策略（趋势/回归/波段/另类）+ ADX 市场状态加权 + ATR 止损/移动保护。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAPI = "https://fapi.binance.com"
REPORT_PATH = ROOT / "composite_cta_backtest_report.json"

ENTRY_SCORE = 0.30
EXIT_SCORE = 0.22
STOP_ATR = 2.0
TRAIL_ATR = 3.5
TRAIL_ACTIVATE_ATR = 1.2
MAX_HOLD_BARS = 150
TREND_ADX_MIN = 20.0
RANGE_ADX_MAX = 18.0
MAX_POSITIONS = 4
RISK_PCT = 1.0
MAX_EXPOSURE_PCT = 30.0
W_TREND = {"trend": 0.42, "revert": 0.12, "band": 0.28, "alt": 0.18}
W_RANGE = {"trend": 0.10, "revert": 0.48, "band": 0.27, "alt": 0.15}
W_NEUTRAL = {"trend": 0.24, "revert": 0.26, "band": 0.32, "alt": 0.18}


@dataclass
class Position:
    symbol: str
    side: str
    entry: float
    qty: float
    atr: float
    bars: int = 0
    max_pnl_atr: float = 0.0
    trail_on: bool = False
    score: float = 0.0
    regime: str = ""


@dataclass
class Trade:
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    entry_px: float
    exit_px: float
    pnl_usdt: float
    reason: str


def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    rows: List[list] = []
    cur = start
    while cur < end:
        r = requests.get(
            f"{FAPI}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end, "limit": 1500},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
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
            "close_time", "quote_vol", "trades", "tb", "tbq", "ignore",
        ],
    )
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.drop_duplicates("open_time").set_index("open_time").sort_index()


def _ema(s: pd.Series, span: int) -> np.ndarray:
    return s.ewm(span=span, adjust=False).mean().values


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    s = pd.Series(close)
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    up = np.diff(high, prepend=high[0])
    dn = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_s = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr_s
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean().values


def _bollinger(close: np.ndarray, period: int = 20, std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = pd.Series(close)
    mid = s.rolling(period).mean()
    dev = s.rolling(period).std(ddof=0)
    return (mid + std * dev).values, mid.values, (mid - std * dev).values


def _cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
    tp = (high + low + close) / 3.0
    s = pd.Series(tp)
    ma = s.rolling(period).mean()
    md = s.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return ((s - ma) / (0.015 * md)).values


def sub_trend(closes: np.ndarray, adx: np.ndarray, daily_closes: Optional[np.ndarray]) -> float:
    n = len(closes)
    if n < 60:
        return 0.0
    ema12, ema26, ema55 = _ema(pd.Series(closes), 12), _ema(pd.Series(closes), 26), _ema(pd.Series(closes), 55)
    i = n - 1
    score = 0.0
    if ema12[i] > ema26[i] > ema55[i]:
        score += 0.45
    elif ema12[i] < ema26[i] < ema55[i]:
        score -= 0.45
    if ema12[i] > ema26[i] and ema12[i - 1] <= ema26[i - 1]:
        score += 0.35
    if ema12[i] < ema26[i] and ema12[i - 1] >= ema26[i - 1]:
        score -= 0.35
    roc5 = (closes[i] - closes[i - 5]) / closes[i - 5]
    roc20 = (closes[i] - closes[i - 20]) / closes[i - 20]
    if roc5 > 0 and roc20 > 0:
        score += 0.15
    if roc5 < 0 and roc20 < 0:
        score -= 0.15
    if adx[i] >= TREND_ADX_MIN:
        score *= 1.15
    elif adx[i] < RANGE_ADX_MAX:
        score *= 0.5
    if daily_closes is not None and len(daily_closes) >= 55:
        e20, e50 = _ema(pd.Series(daily_closes), 20), _ema(pd.Series(daily_closes), 50)
        j = len(daily_closes) - 1
        if e20[j] > e50[j] and score > 0:
            score += 0.12
        if e20[j] < e50[j] and score < 0:
            score -= 0.12
        if (e20[j] > e50[j] and score < 0) or (e20[j] < e50[j] and score > 0):
            score *= 0.6
    return float(np.clip(score, -1, 1))


def sub_revert(closes: np.ndarray, high: np.ndarray, low: np.ndarray) -> float:
    n = len(closes)
    if n < 45:
        return 0.0
    rsi = _rsi(closes)
    upper, mid, lower = _bollinger(closes)
    cci = _cci(high, low, closes)
    i = n - 1
    if np.isnan(rsi[i]) or np.isnan(upper[i]):
        return 0.0
    if rsi[i] < 25 and closes[i] <= lower[i] and cci[i] < -120:
        return 1.0
    if rsi[i] > 75 and closes[i] >= upper[i] and cci[i] > 120:
        return -1.0
    if rsi[i] < 32 and closes[i] < mid[i]:
        return 0.55
    if rsi[i] > 68 and closes[i] > mid[i]:
        return -0.55
    if rsi[i] < 40 and rsi[i] > rsi[i - 1]:
        return 0.25
    if rsi[i] > 60 and rsi[i] < rsi[i - 1]:
        return -0.25
    return 0.0


def sub_band(closes: np.ndarray) -> float:
    n = len(closes)
    if n < 55:
        return 0.0
    ema20, ema50 = _ema(pd.Series(closes), 20), _ema(pd.Series(closes), 50)
    rsi = _rsi(closes)
    i = n - 1
    up, dn = ema20[i] > ema50[i], ema20[i] < ema50[i]
    if up and rsi[i - 1] < 38 and rsi[i] > rsi[i - 1] and closes[i] >= ema20[i]:
        return 0.85
    if dn and rsi[i - 1] > 62 and rsi[i] < rsi[i - 1] and closes[i] <= ema20[i]:
        return -0.85
    if up and rsi[i] > 72:
        return -0.35
    if dn and rsi[i] < 28:
        return 0.35
    return 0.0


def sub_alt(closes: np.ndarray) -> float:
    n = len(closes)
    if n < 40:
        return 0.0
    i = n - 1
    window = closes[i - 29 : i + 1]
    mean = float(np.mean(window))
    std = float(np.std(window))
    if std <= 0:
        return 0.0
    z = (closes[i] - mean) / std
    if z < -2.0:
        return 0.9
    if z > 2.0:
        return -0.9
    if z < -1.2:
        return 0.45
    if z > 1.2:
        return -0.45
    return 0.0


def aggregate_score(
    closes: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    adx: np.ndarray,
    daily_closes: Optional[np.ndarray],
) -> Tuple[float, str, Dict[str, float]]:
    i = len(closes) - 1
    adx_v = float(adx[i])
    if adx_v >= TREND_ADX_MIN:
        regime = "trend"
        w = W_TREND
    elif adx_v <= RANGE_ADX_MAX:
        regime = "range"
        w = W_RANGE
    else:
        regime = "neutral"
        w = W_NEUTRAL
    parts = {
        "trend": sub_trend(closes, adx, daily_closes),
        "revert": sub_revert(closes, high, low),
        "band": sub_band(closes),
        "alt": sub_alt(closes),
    }
    score = sum(parts[k] * w[k] for k in parts)
    return score, regime, parts


def factor_exit_votes(parts: Dict[str, float], is_long: bool) -> int:
    votes = 0
    if is_long:
        if parts["trend"] < -0.25:
            votes += 1
        if parts["revert"] < -0.3:
            votes += 1
        if parts["band"] < -0.25:
            votes += 1
        if parts["alt"] < -0.3:
            votes += 1
    else:
        if parts["trend"] > 0.25:
            votes += 1
        if parts["revert"] > 0.3:
            votes += 1
        if parts["band"] > 0.25:
            votes += 1
        if parts["alt"] > 0.3:
            votes += 1
    return votes


def run_portfolio(
    data: Dict[str, pd.DataFrame],
    daily: Dict[str, pd.DataFrame],
    *,
    equity: float,
    fee_bps: float,
    slip_bps: float,
) -> Tuple[List[Trade], float]:
    fee = fee_bps / 10_000.0
    slip = slip_bps / 10_000.0
    positions: Dict[str, Position] = {}
    trades: List[Trade] = []
    cash = equity

    # align on common 4h index intersection
    idx = None
    for df in data.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    if idx is None or len(idx) < 100:
        return [], equity

    for ts in idx[80:]:
        exposure_pct = 0.0
        for p in positions.values():
            px = float(data[p.symbol].loc[ts, "close"])
            exposure_pct += px * p.qty / max(1e-9, cash + sum(
                (float(data[x.symbol].loc[ts, "close"]) - x.entry) * x.qty * (1 if x.side == "LONG" else -1)
                for x in positions.values()
            )) * 100

        # manage exits
        for sym in list(positions.keys()):
            pos = positions[sym]
            df = data[sym]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            hist = df.loc[:ts]
            closes = hist["close"].values
            high = hist["high"].values
            low = hist["low"].values
            adx = _adx(high, low, closes)
            d_closes = daily[sym]["close"].loc[:ts].values if sym in daily and ts in daily[sym].index else None
            if d_closes is None and sym in daily:
                sub = daily[sym].loc[:ts]
                d_closes = sub["close"].values if len(sub) else None
            score, regime, parts = aggregate_score(closes, high, low, adx, d_closes)
            px = float(row["close"])
            is_long = pos.side == "LONG"
            pnl_atr = (px - pos.entry) * (1 if is_long else -1) / max(1e-9, pos.atr)
            pos.bars += 1
            pos.max_pnl_atr = max(pos.max_pnl_atr, pnl_atr)
            if not pos.trail_on and pos.max_pnl_atr >= TRAIL_ACTIVATE_ATR:
                pos.trail_on = True
            reason = ""
            if pnl_atr <= -STOP_ATR:
                reason = "stop"
            elif pos.trail_on:
                giveback = max(TRAIL_ATR * 0.28, pos.max_pnl_atr * 0.38)
                if (pos.max_pnl_atr - pnl_atr) >= giveback:
                    reason = "trail"
            if not reason:
                votes = factor_exit_votes(parts, is_long)
                if votes >= 2 and ((is_long and score <= -EXIT_SCORE) or (not is_long and score >= EXIT_SCORE)):
                    reason = "factor_exit"
            if not reason and pos.bars >= MAX_HOLD_BARS:
                reason = "timeout"
            if not reason:
                continue
            fill = px * (1 - slip if is_long else 1 + slip)
            pnl = (fill - pos.entry) * pos.qty * (1 if is_long else -1)
            pnl -= pos.entry * pos.qty * fee + fill * pos.qty * fee
            cash += pnl
            trades.append(
                Trade(
                    symbol=sym,
                    side=pos.side,
                    entry_time=str(pos.entry),
                    exit_time=ts.strftime("%Y-%m-%d %H:%M UTC"),
                    entry_px=pos.entry,
                    exit_px=fill,
                    pnl_usdt=round(pnl, 4),
                    reason=reason,
                )
            )
            del positions[sym]

        if len(positions) >= MAX_POSITIONS or exposure_pct >= MAX_EXPOSURE_PCT:
            continue

        candidates: List[Tuple[str, float, float, str, float]] = []
        for sym, df in data.items():
            if sym in positions or ts not in df.index:
                continue
            hist = df.loc[:ts]
            if len(hist) < 80:
                continue
            closes = hist["close"].values
            high = hist["high"].values
            low = hist["low"].values
            adx = _adx(high, low, closes)
            atr_v = float(_atr(high, low, closes)[-1])
            if atr_v <= 0:
                continue
            d_closes = None
            if sym in daily:
                sub = daily[sym].loc[:ts]
                d_closes = sub["close"].values if len(sub) >= 55 else None
            score, regime, _ = aggregate_score(closes, high, low, adx, d_closes)
            if abs(score) < ENTRY_SCORE:
                continue
            candidates.append((sym, score, abs(score), regime, atr_v))
        candidates.sort(key=lambda x: -x[2])

        for sym, score, _, regime, atr_v in candidates:
            if len(positions) >= MAX_POSITIONS:
                break
            px = float(data[sym].loc[ts, "close"])
            stop_dist = atr_v * STOP_ATR
            risk_cash = cash * RISK_PCT / 100.0
            qty = risk_cash / max(1e-9, stop_dist)
            if qty * px < 5:
                continue
            is_long = score > 0
            fill = px * (1 + slip if is_long else 1 - slip)
            cash -= fill * qty * fee
            positions[sym] = Position(
                symbol=sym,
                side="LONG" if is_long else "SHORT",
                entry=fill,
                qty=qty,
                atr=atr_v,
                score=score,
                regime=regime,
            )

    # mark remaining at last bar
    last_ts = idx[-1]
    for sym, pos in list(positions.items()):
        px = float(data[sym].loc[last_ts, "close"])
        fill = px
        is_long = pos.side == "LONG"
        pnl = (fill - pos.entry) * pos.qty * (1 if is_long else -1)
        pnl -= pos.entry * pos.qty * fee + fill * pos.qty * fee
        cash += pnl
        trades.append(
            Trade(sym, pos.side, str(pos.entry), str(last_ts), pos.entry, fill, round(pnl, 4), "eod")
        )

    return trades, cash


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Composite CTA v2 backtest")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "LINKUSDT"])
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--equity", type=float, default=2500.0)
    p.add_argument("--fee-bps", type=float, default=4.0)
    p.add_argument("--slip-bps", type=float, default=2.0)
    p.add_argument("--report", type=Path, default=REPORT_PATH)
    args = p.parse_args(argv)

    print(f"复合 CTA v2 | 4h | {args.days}d | {len(args.symbols)} 品种")
    data: Dict[str, pd.DataFrame] = {}
    daily: Dict[str, pd.DataFrame] = {}
    for sym in args.symbols:
        print(f"  fetch {sym}...", flush=True)
        data[sym] = fetch_klines(sym, "4h", args.days + 30)
        daily[sym] = fetch_klines(sym, "1d", args.days + 60)

    trades, end_eq = run_portfolio(data, daily, equity=args.equity, fee_bps=args.fee_bps, slip_bps=args.slip_bps)
    ret = (end_eq / args.equity - 1) * 100
    wins = sum(1 for t in trades if t.pnl_usdt > 0)
    wr = wins / len(trades) * 100 if trades else 0
    pnl = end_eq - args.equity

    print(f"\n结果: trades={len(trades)} win%={wr:.1f} ret%={ret:.2f} pnl=${pnl:.2f} end=${end_eq:.2f}")

    report = {
        "strategy": "composite_cta_v2",
        "source": "fmz-composite-cta-reference/local/fmz_composite_cta.js",
        "days": args.days,
        "symbols": args.symbols,
        "trades": len(trades),
        "win_rate_pct": round(wr, 2),
        "total_return_pct": round(ret, 2),
        "total_pnl_usdt": round(pnl, 2),
        "end_equity_usdt": round(end_eq, 2),
        "trade_sample": [t.__dict__ for t in trades[-30:]],
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Desk-F style AVAX mean-reversion indicator v2 (research).

Reverse-engineered from ``0xa740…4d23``; v2 adds filters that lifted
standalone 90d equity ~1.23→~1.70 and F-entry precision ~83%→~86–94%:

  LONG  ← ret_4h ≤ -min  AND RSI14 < rsi_long_max
          AND ret_24h ≤ chase_block (no long chase)
          AND above 3d low ≤ ext_max_pct (near lows)
  SHORT ← ret_4h ≥ +min  AND RSI14 > rsi_short_min
          AND ret_24h ≥ -chase_block (no short chase)
          AND below 3d high ≤ ext_max_pct (near highs)
  FLAT  otherwise

Modes:
  trade  — default balanced (rsi 45/55, r4≥0.3%, ext≤4%, block 24h chase)
  gate   — stricter mirror filter (ext≤2.5%, higher precision vs F)

Not investment advice; mirror gate / paper research only.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Sequence

from utils.hl_wr_screen import _hl_info

BJ = timezone(timedelta(hours=8))
Side = Literal["long", "short", "flat"]
Mode = Literal["trade", "gate"]

REF_ADDR = "0xa7405ff2687cb83b8a8a08eeaa4e4bc249344d23"
VERSION = "f-mr-v2"


@dataclass(frozen=True)
class FMrParams:
    rsi_period: int = 14
    rsi_long_max: float = 45.0
    rsi_short_min: float = 55.0
    rsi_strong_long: float = 40.0
    rsi_strong_short: float = 60.0
    ret4h_min_abs: float = 0.3
    # block entries that chase the prior 24h move in trade direction
    ret24h_chase_block: float = 1.0
    # require price within ext_max_pct of 3d extreme (72x1h bars)
    use_ext3d: bool = True
    ext_lookback_bars: int = 72
    ext_max_pct: float = 4.0
    min_strength: int = 45


def params_for_mode(mode: Mode = "trade") -> FMrParams:
    if mode == "gate":
        return FMrParams(
            rsi_long_max=45.0,
            rsi_short_min=50.0,
            ret4h_min_abs=0.0,
            ret24h_chase_block=1.0,
            use_ext3d=True,
            ext_max_pct=2.5,
            min_strength=40,
        )
    return FMrParams()


@dataclass
class FMrSignal:
    coin: str
    side: Side
    strength: int
    price: float
    ret_4h_pct: float
    ret_24h_pct: float | None
    rsi14: float
    above_lo_3d_pct: float | None
    below_hi_3d_pct: float | None
    rules: dict[str, bool]
    ts_ms: int
    ts_bj: str
    params: dict[str, Any]
    version: str
    mode: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rsi(closes: Sequence[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        d = float(closes[i]) - float(closes[i - 1])
        if d >= 0:
            gains += d
        else:
            losses -= d
    avg_g = gains / period
    avg_l = losses / period
    if avg_l < 1e-12:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))


def _ret_pct(closes: Sequence[float], bars: int) -> float | None:
    if len(closes) < bars + 1:
        return None
    a = float(closes[-(bars + 1)])
    b = float(closes[-1])
    if a <= 0:
        return None
    return (b / a - 1.0) * 100.0


def _ext_3d(
    highs: Sequence[float] | None,
    lows: Sequence[float] | None,
    closes: Sequence[float],
    lookback: int,
) -> tuple[float | None, float | None]:
    """Return (below_hi_pct, above_lo_pct) over last lookback bars."""
    n = min(lookback, len(closes))
    if n < 3:
        return None, None
    px = float(closes[-1])
    if highs is not None and len(highs) >= n:
        hi = max(float(x) for x in highs[-n:])
    else:
        hi = max(float(x) for x in closes[-n:])
    if lows is not None and len(lows) >= n:
        lo = min(float(x) for x in lows[-n:])
    else:
        lo = min(float(x) for x in closes[-n:])
    below_hi = (hi - px) / hi * 100.0 if hi > 0 else None
    above_lo = (px - lo) / lo * 100.0 if lo > 0 else None
    return below_hi, above_lo


def compute_signal(
    closes_1h: Sequence[float],
    *,
    highs_1h: Sequence[float] | None = None,
    lows_1h: Sequence[float] | None = None,
    coin: str = "AVAX",
    ts_ms: int | None = None,
    params: FMrParams | None = None,
    mode: Mode = "trade",
) -> FMrSignal:
    """Compute F-MR v2 signal from ascending 1h OHLCs (last = current close)."""
    p = params or params_for_mode(mode)
    need = max(25, p.rsi_period + 2, p.ext_lookback_bars)
    if len(closes_1h) < max(5, p.rsi_period + 2):
        raise ValueError("need_more_1h_closes")

    px = float(closes_1h[-1])
    ret4 = _ret_pct(closes_1h, 4)
    ret24 = _ret_pct(closes_1h, 24) if len(closes_1h) >= 25 else None
    if ret4 is None:
        raise ValueError("ret4h_unavailable")
    rsi = _rsi(closes_1h, p.rsi_period)
    if rsi is None:
        raise ValueError("rsi_unavailable")

    below_hi, above_lo = _ext_3d(highs_1h, lows_1h, closes_1h, p.ext_lookback_bars)

    fade_long = ret4 < 0 and abs(ret4) >= p.ret4h_min_abs
    fade_short = ret4 > 0 and abs(ret4) >= p.ret4h_min_abs
    rsi_long_ok = rsi < p.rsi_long_max
    rsi_short_ok = rsi > p.rsi_short_min
    strong_long = rsi < p.rsi_strong_long
    strong_short = rsi > p.rsi_strong_short

    # no-chase: don't long after already +ret24, don't short after already -ret24
    chase_ok_long = ret24 is None or ret24 <= p.ret24h_chase_block
    chase_ok_short = ret24 is None or ret24 >= -p.ret24h_chase_block

    ext_ok_long = True
    ext_ok_short = True
    if p.use_ext3d:
        ext_ok_long = above_lo is not None and above_lo <= p.ext_max_pct
        ext_ok_short = below_hi is not None and below_hi <= p.ext_max_pct

    long_ok = fade_long and rsi_long_ok and chase_ok_long and ext_ok_long
    short_ok = fade_short and rsi_short_ok and chase_ok_short and ext_ok_short

    side: Side = "flat"
    strength = 0
    if long_ok and not short_ok:
        side = "long"
        s = 40 + min(25, int(abs(ret4) * 8))
        s += 15 if strong_long else 5
        if rsi < 35:
            s += 10
        if above_lo is not None and above_lo <= 2.0:
            s += 10
        strength = max(0, min(100, s))
    elif short_ok and not long_ok:
        side = "short"
        s = 40 + min(25, int(abs(ret4) * 8))
        s += 15 if strong_short else 5
        if rsi > 65:
            s += 10
        if below_hi is not None and below_hi <= 2.0:
            s += 10
        strength = max(0, min(100, s))

    if side != "flat" and strength < p.min_strength:
        side = "flat"
        strength = 0

    ts = ts_ms if ts_ms is not None else int(time.time() * 1000)
    return FMrSignal(
        coin=coin,
        side=side,
        strength=strength,
        price=px,
        ret_4h_pct=round(ret4, 4),
        ret_24h_pct=None if ret24 is None else round(ret24, 4),
        rsi14=round(rsi, 2),
        above_lo_3d_pct=None if above_lo is None else round(above_lo, 3),
        below_hi_3d_pct=None if below_hi is None else round(below_hi, 3),
        rules={
            "fade_4h_long": fade_long,
            "fade_4h_short": fade_short,
            "rsi_long_ok": rsi_long_ok,
            "rsi_short_ok": rsi_short_ok,
            "rsi_strong_long": strong_long,
            "rsi_strong_short": strong_short,
            "chase_ok_long": chase_ok_long,
            "chase_ok_short": chase_ok_short,
            "ext_ok_long": ext_ok_long,
            "ext_ok_short": ext_ok_short,
            "long_ok": long_ok,
            "short_ok": short_ok,
        },
        ts_ms=ts,
        ts_bj=datetime.fromtimestamp(ts / 1000, BJ).isoformat(),
        params=asdict(p),
        version=VERSION,
        mode=mode if params is None else "custom",
        note=(
            "f-mr-v2: 4h fade + RSI band + 24h no-chase + 3d extreme proximity; "
            "use as mirror gate or paper signal, not auto-size."
        ),
    )


def _ohlc_from_candles(
    candles: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float], list[int]]:
    closes, highs, lows, ts = [], [], [], []
    for c in candles:
        closes.append(float(c["c"]))
        highs.append(float(c.get("h") or c["c"]))
        lows.append(float(c.get("l") or c["c"]))
        ts.append(int(c["t"]))
    return closes, highs, lows, ts


def fetch_1h_closes(coin: str = "AVAX", days: int = 60) -> tuple[list[dict[str, Any]], list[float]]:
    """Fetch HL 1h candles; return (candles, closes)."""
    end = int(time.time() * 1000)
    start = end - days * 86400 * 1000
    batch = _hl_info(
        {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": "1h",
                "startTime": start,
                "endTime": end,
            },
        }
    )
    if not isinstance(batch, list) or not batch:
        raise RuntimeError(f"no_candles:{coin}")
    candles = sorted(
        [c for c in batch if isinstance(c, dict) and c.get("t") is not None],
        key=lambda c: int(c["t"]),
    )
    closes = [float(c["c"]) for c in candles]
    return candles, closes


def live_signal(
    coin: str = "AVAX",
    params: FMrParams | None = None,
    mode: Mode = "trade",
) -> dict[str, Any]:
    candles, _ = fetch_1h_closes(coin=coin, days=40)
    closes, highs, lows, ts = _ohlc_from_candles(candles)
    sig = compute_signal(
        closes,
        highs_1h=highs,
        lows_1h=lows,
        coin=coin,
        ts_ms=ts[-1],
        params=params,
        mode=mode,
    )
    out = sig.to_dict()
    out["ok"] = True
    out["ref_addr"] = REF_ADDR
    out["candle_n"] = len(candles)
    out["source"] = "hl_candleSnapshot_1h"
    return out


def backtest_1h(
    closes: Sequence[float],
    candle_ts: list[int] | None = None,
    *,
    highs: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
    params: FMrParams | None = None,
    mode: Mode = "trade",
    hold_bars: int = 72,
    fee_bps: float = 2.0,
) -> dict[str, Any]:
    """Bar-close signal; enter next bar; exit after hold_bars or flip."""
    p = params or params_for_mode(mode)
    need = max(25, p.rsi_period + 2)
    if len(closes) < need + hold_bars + 2:
        return {"ok": False, "error": "not_enough_bars"}

    trades: list[dict[str, Any]] = []
    i = need
    equity = 1.0
    while i < len(closes) - 1:
        h_slice = None if highs is None else highs[: i + 1]
        l_slice = None if lows is None else lows[: i + 1]
        sig = compute_signal(
            closes[: i + 1],
            highs_1h=h_slice,
            lows_1h=l_slice,
            params=p,
            mode=mode,
        )
        if sig.side == "flat" or sig.strength < p.min_strength:
            i += 1
            continue
        entry_i = i + 1
        if entry_i >= len(closes):
            break
        entry = float(closes[entry_i])
        side = sig.side
        exit_i = min(len(closes) - 1, entry_i + hold_bars)
        for j in range(entry_i + 1, exit_i + 1):
            s2 = compute_signal(
                closes[: j + 1],
                highs_1h=None if highs is None else highs[: j + 1],
                lows_1h=None if lows is None else lows[: j + 1],
                params=p,
                mode=mode,
            )
            if side == "long" and s2.side == "short" and s2.strength >= 50:
                exit_i = j
                break
            if side == "short" and s2.side == "long" and s2.strength >= 50:
                exit_i = j
                break
        exit_px = float(closes[exit_i])
        raw = (exit_px / entry - 1.0) if side == "long" else (entry / exit_px - 1.0)
        pnl = raw - 2 * (fee_bps / 10_000.0)
        equity *= 1.0 + pnl
        trades.append(
            {
                "side": side,
                "strength": sig.strength,
                "entry_i": entry_i,
                "exit_i": exit_i,
                "hold_h": exit_i - entry_i,
                "ret_pct": round(pnl * 100, 3),
                "entry_ts": None if not candle_ts else candle_ts[entry_i],
                "exit_ts": None if not candle_ts else candle_ts[exit_i],
                "ret_4h_at_sig": sig.ret_4h_pct,
                "rsi_at_sig": sig.rsi14,
            }
        )
        i = exit_i + 1

    wins = [t for t in trades if t["ret_pct"] > 0]
    losses = [t for t in trades if t["ret_pct"] <= 0]
    rets = [t["ret_pct"] for t in trades]
    return {
        "ok": True,
        "version": VERSION,
        "mode": mode if params is None else "custom",
        "n_trades": len(trades),
        "wr": None if not trades else round(len(wins) / len(trades), 4),
        "avg_ret_pct": None if not rets else round(sum(rets) / len(rets), 4),
        "sum_ret_pct": round(sum(rets), 3),
        "equity_mult": round(equity, 4),
        "hold_bars": hold_bars,
        "fee_bps": fee_bps,
        "params": asdict(p),
        "trades_tail": trades[-12:],
        "wins": len(wins),
        "losses": len(losses),
    }


def snapshot(
    coin: str = "AVAX",
    *,
    with_backtest: bool = True,
    mode: Mode = "trade",
) -> dict[str, Any]:
    candles, _ = fetch_1h_closes(coin=coin, days=90)
    closes, highs, lows, ts = _ohlc_from_candles(candles)
    sig = compute_signal(
        closes,
        highs_1h=highs,
        lows_1h=lows,
        coin=coin,
        ts_ms=ts[-1],
        mode=mode,
    )
    out: dict[str, Any] = {
        "ok": True,
        "version": VERSION,
        "mode": mode,
        "signal": sig.to_dict(),
        "ref_addr": REF_ADDR,
        "candle_n": len(candles),
    }
    if with_backtest:
        out["backtest_90d"] = backtest_1h(
            closes, ts, highs=highs, lows=lows, mode=mode, hold_bars=72
        )
        out["backtest_gate_90d"] = backtest_1h(
            closes, ts, highs=highs, lows=lows, mode="gate", hold_bars=72
        )
    return out


def _ms_to_utc_iso(ms: int | None) -> str:
    if not ms:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def strategy_signal_feed(
    *,
    coin: str = "AVAX",
    mode: Mode = "gate",
    limit: int = 100,
) -> dict[str, Any]:
    """Shape compatible with ``/api/strategy/signals`` for 观盘台纸面模拟.

    Live non-flat bar → first row (status=emitted / 纸面信号).
    Recent backtest entries → historical rows (status=shadow / 回测纸面).
    No hard SL/TP (null); exit is max-hold / reverse signal.
    """
    lim = max(1, min(int(limit or 100), 200))
    candles, _ = fetch_1h_closes(coin=coin, days=90)
    closes, highs, lows, ts = _ohlc_from_candles(candles)
    live = compute_signal(
        closes,
        highs_1h=highs,
        lows_1h=lows,
        coin=coin,
        ts_ms=ts[-1] if ts else None,
        mode=mode,
    )
    trades_all = _backtest_trades_all(
        closes, ts, highs=highs, lows=lows, mode=mode, hold_bars=72
    )
    wins = sum(1 for t in trades_all if float(t.get("ret_pct") or 0) > 0)
    wr = (wins / len(trades_all)) if trades_all else None
    equity = 1.0
    for t in trades_all:
        equity *= 1.0 + float(t.get("ret_pct") or 0) / 100.0

    symbol = f"{coin.upper()}USDT"
    signals: list[dict[str, Any]] = []
    if live.side in ("long", "short"):
        signals.append(
            {
                "id": 0,
                "lane": "avax_f_mr",
                "symbol": symbol,
                "side": live.side.upper(),
                "action": "open",
                "entry_price": live.price,
                "sl_price": None,
                "tp_price": None,
                "status": "emitted",
                "skip_reason": None,
                "detail": {
                    "version": VERSION,
                    "mode": mode,
                    "strength": live.strength,
                    "ret_4h_pct": live.ret_4h_pct,
                    "ret_24h_pct": live.ret_24h_pct,
                    "rsi14": live.rsi14,
                    "live": True,
                },
                "bar_ms": live.ts_ms,
                "received_at": _ms_to_utc_iso(live.ts_ms),
            }
        )

    # newest first
    for i, t in enumerate(reversed(trades_all)):
        if len(signals) >= lim:
            break
        raw_side = str(t.get("side") or "").lower()
        side = "LONG" if raw_side == "long" else "SHORT"
        entry_ts = t.get("entry_ts")
        signals.append(
            {
                "id": -(i + 1),
                "lane": "avax_f_mr",
                "symbol": symbol,
                "side": side,
                "action": "open",
                "entry_price": t.get("entry_px"),
                "sl_price": None,
                "tp_price": None,
                "status": "shadow",
                "skip_reason": f"hold={t.get('hold_h')}h ret={t.get('ret_pct')}%",
                "detail": {
                    "version": VERSION,
                    "mode": mode,
                    "strength": t.get("strength"),
                    "ret_pct": t.get("ret_pct"),
                    "hold_h": t.get("hold_h"),
                    "rsi_at_sig": t.get("rsi_at_sig"),
                    "ret_4h_at_sig": t.get("ret_4h_at_sig"),
                    "backtest": True,
                },
                "bar_ms": entry_ts,
                "received_at": _ms_to_utc_iso(int(entry_ts) if entry_ts else None),
            }
        )

    return {
        "ok": True,
        "lane": "avax_f_mr",
        "count": len(signals),
        "signals": signals[:lim],
        "live": live.to_dict(),
        "backtest_summary": {
            "n_trades": len(trades_all),
            "wr": None if wr is None else round(wr, 4),
            "equity_mult": round(equity, 4),
            "mode": mode,
            "version": VERSION,
        },
        "ref_addr": REF_ADDR,
    }


def _backtest_trades_all(
    closes: Sequence[float],
    candle_ts: list[int] | None,
    *,
    highs: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
    mode: Mode = "gate",
    hold_bars: int = 72,
    fee_bps: float = 2.0,
) -> list[dict[str, Any]]:
    """Same as backtest_1h but returns full trade list (with entry_px)."""
    p = params_for_mode(mode)
    need = max(25, p.rsi_period + 2)
    if len(closes) < need + hold_bars + 2:
        return []
    trades: list[dict[str, Any]] = []
    i = need
    while i < len(closes) - 1:
        sig = compute_signal(
            closes[: i + 1],
            highs_1h=None if highs is None else highs[: i + 1],
            lows_1h=None if lows is None else lows[: i + 1],
            params=p,
            mode=mode,
        )
        if sig.side == "flat" or sig.strength < p.min_strength:
            i += 1
            continue
        entry_i = i + 1
        if entry_i >= len(closes):
            break
        entry = float(closes[entry_i])
        side = sig.side
        exit_i = min(len(closes) - 1, entry_i + hold_bars)
        for j in range(entry_i + 1, exit_i + 1):
            s2 = compute_signal(
                closes[: j + 1],
                highs_1h=None if highs is None else highs[: j + 1],
                lows_1h=None if lows is None else lows[: j + 1],
                params=p,
                mode=mode,
            )
            if side == "long" and s2.side == "short" and s2.strength >= 50:
                exit_i = j
                break
            if side == "short" and s2.side == "long" and s2.strength >= 50:
                exit_i = j
                break
        exit_px = float(closes[exit_i])
        raw = (exit_px / entry - 1.0) if side == "long" else (entry / exit_px - 1.0)
        pnl = raw - 2 * (fee_bps / 10_000.0)
        trades.append(
            {
                "side": side,
                "strength": sig.strength,
                "entry_i": entry_i,
                "exit_i": exit_i,
                "hold_h": exit_i - entry_i,
                "entry_px": round(entry, 6),
                "ret_pct": round(pnl * 100, 3),
                "entry_ts": None if not candle_ts else candle_ts[entry_i],
                "exit_ts": None if not candle_ts else candle_ts[exit_i],
                "ret_4h_at_sig": sig.ret_4h_pct,
                "rsi_at_sig": sig.rsi14,
            }
        )
        i = exit_i + 1
    return trades

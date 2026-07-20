"""RTM pattern backtest — win rate and expectancy from detected signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from quant.rtm_patterns.config import RTMConfig
from quant.rtm_patterns.scanner import scan_rtm_patterns
from quant.rtm_patterns.types import PatternHit

TradeOutcome = Literal["win", "loss", "timeout"]


@dataclass(frozen=True)
class RTMTrade:
    pattern: str
    direction: str
    signal_bar: int
    entry_bar: int
    entry_price: float
    stop_price: float
    target_price: float
    exit_bar: int
    exit_price: float
    outcome: TradeOutcome
    r_multiple: float
    quality: float
    zone_source: str | None


@dataclass
class PatternStats:
    pattern: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    total_r: float = 0.0


@dataclass
class BacktestSummary:
    trades: list[RTMTrade] = field(default_factory=list)
    by_pattern: dict[str, PatternStats] = field(default_factory=dict)
    total_trades: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    profit_factor: float = 0.0


@dataclass(frozen=True)
class BacktestParams:
    target_r: float = 2.0
    max_hold_bars: int = 30
    entry_on_next_bar: bool = True


def _resolve_target(hit: PatternHit, entry: float, stop: float, target_r: float) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return entry
    default_long = entry + risk * target_r
    default_short = entry - risk * target_r

    if hit.target_level is not None and not np.isnan(hit.target_level):
        t = float(hit.target_level)
        if hit.direction == "long" and t > entry:
            return t
        if hit.direction == "short" and t < entry:
            return t
    return default_long if hit.direction == "long" else default_short


def simulate_hit(
    hit: PatternHit,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    params: BacktestParams,
) -> RTMTrade | None:
    n = len(close)
    if hit.stop_level is None or hit.entry_level is None:
        return None

    entry_bar = hit.bar_index + (1 if params.entry_on_next_bar else 0)
    if entry_bar >= n:
        return None

    entry = float(close[entry_bar] if params.entry_on_next_bar else hit.entry_level)
    stop = float(hit.stop_level)
    target = _resolve_target(hit, entry, stop, params.target_r)
    risk = abs(entry - stop)
    if risk <= 0:
        return None

    if hit.direction == "long" and not (stop < entry < target or stop < entry):
        if stop >= entry:
            return None
    if hit.direction == "short" and not (target < entry < stop or entry < stop):
        if stop <= entry:
            return None

    end = min(n - 1, entry_bar + params.max_hold_bars)
    outcome: TradeOutcome = "timeout"
    exit_bar = end
    exit_price = float(close[end])

    for j in range(entry_bar + 1, end + 1):
        if hit.direction == "long":
            if low[j] <= stop:
                outcome, exit_bar, exit_price = "loss", j, stop
                break
            if high[j] >= target:
                outcome, exit_bar, exit_price = "win", j, target
                break
        else:
            if high[j] >= stop:
                outcome, exit_bar, exit_price = "loss", j, stop
                break
            if low[j] <= target:
                outcome, exit_bar, exit_price = "win", j, target
                break

    if hit.direction == "long":
        r_mult = (exit_price - entry) / risk
    else:
        r_mult = (entry - exit_price) / risk

    return RTMTrade(
        pattern=hit.pattern,
        direction=hit.direction,
        signal_bar=hit.bar_index,
        entry_bar=entry_bar,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        exit_bar=exit_bar,
        exit_price=exit_price,
        outcome=outcome,
        r_multiple=float(r_mult),
        quality=float(hit.meta.get("quality", 0.0)),
        zone_source=hit.meta.get("zone_source"),
    )


def backtest_rtm_patterns(
    df: pd.DataFrame,
    *,
    config: RTMConfig | None = None,
    params: BacktestParams | None = None,
) -> BacktestSummary:
    """Run simple stop/target backtest on RTM pattern hits."""
    cfg = config or RTMConfig()
    p = params or BacktestParams()
    hits = scan_rtm_patterns(df, config=cfg)

    cols = {c.lower(): c for c in df.columns}
    norm = df.rename(columns={cols[k]: k for k in ("open", "high", "low", "close")})
    high = norm["high"].to_numpy(dtype=float)
    low = norm["low"].to_numpy(dtype=float)
    close = norm["close"].to_numpy(dtype=float)

    trades: list[RTMTrade] = []
    for hit in hits:
        t = simulate_hit(hit, high, low, close, params=p)
        if t is not None:
            trades.append(t)

    by_pattern: dict[str, PatternStats] = {}
    for t in trades:
        st = by_pattern.setdefault(t.pattern, PatternStats(pattern=t.pattern))
        st.trades += 1
        st.total_r += t.r_multiple
        if t.outcome == "win":
            st.wins += 1
        elif t.outcome == "loss":
            st.losses += 1
        else:
            st.timeouts += 1

    for st in by_pattern.values():
        decided = st.wins + st.losses
        st.win_rate = st.wins / decided if decided else 0.0
        st.avg_r = st.total_r / st.trades if st.trades else 0.0

    wins = sum(1 for t in trades if t.outcome == "win")
    losses = sum(1 for t in trades if t.outcome == "loss")
    decided = wins + losses
    gross_win = sum(t.r_multiple for t in trades if t.outcome == "win")
    gross_loss = abs(sum(t.r_multiple for t in trades if t.outcome == "loss"))

    return BacktestSummary(
        trades=trades,
        by_pattern=by_pattern,
        total_trades=len(trades),
        win_rate=wins / decided if decided else 0.0,
        avg_r=float(np.mean([t.r_multiple for t in trades])) if trades else 0.0,
        profit_factor=gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
    )


def summary_to_dataframe(summary: BacktestSummary) -> pd.DataFrame:
    rows = [
        {
            "pattern": st.pattern,
            "trades": st.trades,
            "wins": st.wins,
            "losses": st.losses,
            "timeouts": st.timeouts,
            "win_rate": round(st.win_rate * 100, 1),
            "avg_r": round(st.avg_r, 2),
            "total_r": round(st.total_r, 2),
        }
        for st in sorted(summary.by_pattern.values(), key=lambda x: -x.trades)
    ]
    return pd.DataFrame(rows)

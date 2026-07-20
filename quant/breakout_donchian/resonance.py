"""1D execute + 1W confirm + 1H bonus (aligned with breakoutscanner)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from quant.breakout_donchian.bars import BarRow, drop_incomplete_bars, resample_weekly_from_daily
from quant.breakout_donchian.core import BreakoutDirection, DonchianSignal, detect_donchian_signal

if TYPE_CHECKING:
    from quant.breakout_donchian.config import BreakoutDonchianConfig


@dataclass(frozen=True)
class ResonanceResult:
    weekly_ok: bool
    hourly_bonus: bool
    tier: str
    risk_mult: float


def _direction_filter(cfg: "BreakoutDonchianConfig") -> Optional[BreakoutDirection]:
    return "bullish" if cfg.long_only else None


def _mode(cfg: "BreakoutDonchianConfig", *, weekly: bool = False) -> str:
    if weekly:
        raw = str(getattr(cfg, "weekly_breakout_mode", "") or "standard").lower()
        return "strict" if raw == "strict" else "standard"
    return "strict" if str(cfg.breakout_mode).lower() == "strict" else "standard"


def _detect_kwargs(cfg: "BreakoutDonchianConfig", *, weekly: bool = False) -> dict:
    return {
        "mode": _mode(cfg, weekly=weekly),
        "strict_vol_mult": float(cfg.strict_vol_mult),
        "atr_period": int(cfg.atr_period),
        "direction_filter": _direction_filter(cfg),
        "tp1_rr": float(cfg.tp1_rr),
        "tp2_rr": float(cfg.tp2_rr),
        "tp3_rr": float(cfg.tp3_rr),
        "sl_atr_mult": float(cfg.sl_atr_mult),
        "sl_level_buffer": float(cfg.sl_level_buffer),
    }


def weekly_bars_from_daily(daily_bars: Sequence[BarRow]) -> list[BarRow]:
    return resample_weekly_from_daily(daily_bars)


def weekly_trend_ok(bars: Sequence[BarRow], cfg: "BreakoutDonchianConfig") -> bool:
    """1W 验：收盘价在周线均线之上（趋势过滤，非重复 Strict 突破）。"""
    clean = list(bars)
    period = max(2, int(cfg.weekly_trend_ma_period))
    if len(clean) < period + 1:
        return False
    closes = [float(b[4]) for b in clean]
    ma = sum(closes[-(period + 1) : -1]) / period
    return float(closes[-1]) > ma


def detect_weekly_confirm(
    bars: Sequence[BarRow],
    cfg: "BreakoutDonchianConfig",
) -> Optional[DonchianSignal]:
    clean = drop_incomplete_bars(list(bars), "1w")
    return detect_donchian_signal(
        clean,
        lookback=int(cfg.weekly_lookback),
        vol_lookback=int(cfg.weekly_vol_lookback),
        vol_mult=float(cfg.weekly_vol_mult),
        strong_close_pct=float(cfg.weekly_strong_close_pct),
        strict_atr_mult=float(cfg.strict_atr_mult),
        **_detect_kwargs(cfg, weekly=True),
    )


def detect_weekly_pool_confirm(
    bars: Sequence[BarRow],
    cfg: "BreakoutDonchianConfig",
) -> bool:
    """
    Scanner 池对齐：1W 用 standard 突破确认（1D 仍 strict）。
    等价于 watchlist 里 1D∩1W 的 1W 侧，但放宽 ATR 扩张要求。
    """
    return detect_weekly_confirm(bars, cfg) is not None


def detect_hourly_bonus(
    bars: Sequence[BarRow],
    cfg: "BreakoutDonchianConfig",
) -> bool:
    clean = drop_incomplete_bars(list(bars), "1h")
    sig = detect_donchian_signal(
        clean,
        lookback=int(cfg.hourly_lookback),
        vol_lookback=int(cfg.hourly_vol_lookback),
        vol_mult=float(cfg.hourly_vol_mult),
        strong_close_pct=float(cfg.hourly_strong_close_pct),
        strict_atr_mult=float(cfg.hourly_strict_atr_mult),
        **_detect_kwargs(cfg),
    )
    return sig is not None


def _weekly_ok(cfg: "BreakoutDonchianConfig", weekly_bars: Sequence[BarRow]) -> bool:
    mode = str(cfg.weekly_confirm_mode or "trend").strip().lower()
    if mode in ("off", "none", "0", "false"):
        return True
    if mode in ("trend", "ma"):
        return weekly_trend_ok(weekly_bars, cfg)
    if mode in ("pool", "scanner"):
        return detect_weekly_pool_confirm(weekly_bars, cfg)
    if mode in ("strict", "breakout_strict"):
        return detect_weekly_confirm(weekly_bars, cfg) is not None
    return weekly_trend_ok(weekly_bars, cfg)


def evaluate_resonance(
    cfg: "BreakoutDonchianConfig",
    *,
    weekly_bars: Sequence[BarRow],
    hourly_bars: Sequence[BarRow] | None = None,
) -> ResonanceResult:
    weekly_ok = _weekly_ok(cfg, weekly_bars)
    hourly_bonus = bool(cfg.check_1h_bonus and hourly_bars and detect_hourly_bonus(hourly_bars, cfg))

    if weekly_ok and hourly_bonus:
        return ResonanceResult(True, True, "triple", float(cfg.risk_mult_triple))
    if weekly_ok:
        return ResonanceResult(True, False, "dual", float(cfg.risk_mult_base))
    return ResonanceResult(False, False, "none", 0.0)


def preload_days_for_interval(interval: str, *, min_bars: int) -> int:
    key = interval.strip().lower()
    if key == "1h":
        return max(14, (min_bars // 24) + 7)
    return max(120, min_bars + 30)

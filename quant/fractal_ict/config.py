"""Fractal ICT backtest / strategy configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _interval_minutes(interval: str) -> int:
    s = interval.strip().lower()
    if s.endswith("m") and s[:-1].isdigit():
        return int(s[:-1])
    if s.endswith("h") and s[:-1].isdigit():
        return int(s[:-1]) * 60
    if s in ("1d", "d", "day"):
        return 24 * 60
    raise ValueError(f"unsupported interval: {interval}")


def auto_htf_interval(ltf_interval: str) -> str:
    """Mirror Pine auto HTF1 mapping."""
    m = _interval_minutes(ltf_interval)
    if m <= 1:
        return "15m"
    if m <= 3:
        return "30m"
    if m <= 5:
        return "1h"
    if m <= 15:
        return "4h"
    if m <= 60:
        return "1d"
    return "1w"


@dataclass
class FractalIctConfig:
    ltf_interval: str = "5m"
    htf_interval: str = ""
    bias: str = "none"  # none | bullish | bearish
    use_body_for_confirmation: bool = True
    entry_mode: str = "cisd_c2"  # cisd_c2 | c3_touch
    rr_ratio: float = 2.0
    require_fractal_touch: bool = False
    allowed_patterns: tuple[str, ...] | None = None  # e.g. ("c2_sweep",)
    range_only: bool = False
    range_max_pct: float = 0.015
    allowed_sessions: tuple[str, ...] | None = None  # asia | london | ny | late
    equity_usdt: float = 10_000.0
    risk_pct: float = 1.0
    cooldown_bars: int = 3
    max_setup_bars: int = 0  # 0 = one HTF period
    slippage_bps: float = 5.0

    def resolved_htf(self) -> str:
        return self.htf_interval.strip() or auto_htf_interval(self.ltf_interval)

    @classmethod
    def from_env(cls) -> "FractalIctConfig":
        return cls(
            ltf_interval=os.getenv("FRACTAL_LTF_INTERVAL", "5m"),
            htf_interval=os.getenv("FRACTAL_HTF_INTERVAL", ""),
            bias=os.getenv("FRACTAL_BIAS", "none").lower(),
            use_body_for_confirmation=os.getenv("FRACTAL_USE_BODY", "1") != "0",
            entry_mode=os.getenv("FRACTAL_ENTRY_MODE", "cisd_c2").lower(),
            rr_ratio=float(os.getenv("FRACTAL_RR_RATIO", "2.0")),
            require_fractal_touch=os.getenv("FRACTAL_REQUIRE_TOUCH", "0") == "1",
            equity_usdt=float(os.getenv("FRACTAL_EQUITY_USDT", "10000")),
            risk_pct=float(os.getenv("FRACTAL_RISK_PCT", "1.0")),
            cooldown_bars=int(os.getenv("FRACTAL_COOLDOWN_BARS", "3")),
            max_setup_bars=int(os.getenv("FRACTAL_MAX_SETUP_BARS", "0")),
            slippage_bps=float(os.getenv("FRACTAL_SLIPPAGE_BPS", "5")),
        )

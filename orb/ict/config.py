"""ICT 2022 vnpy lane 配置（ICT_VNPY_*）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from orb.core.symbols import parse_symbol_list


def _truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except ValueError:
        return float(default)


@dataclass
class IctVnpyConfig:
    lane: str = "ict_2022"
    engine: str = "vnpy"
    enabled: bool = False
    shadow: bool = False
    symbols: List[str] | None = None
    equity_usdt: float = 1000.0
    position_pct: float = 1.0
    leverage: float = 2.0
    rr: float = 1.5
    compound: bool = True
    hmm_filter: bool = True
    hmm_stick: float = 0.97
    hmm_confirm: int = 3
    live_enabled: bool = False
    live_leverage: float = 2.0
    max_notional_usdt: float = 0.0
    limit_ttl_hours: float = 4.0
    setup_ttl_hours: float = 6.0
    breakout_ttl_hours: float = 2.0
    init_bar_days: int = 3

    @classmethod
    def from_env(cls) -> "IctVnpyConfig":
        inline = (os.getenv("ICT_VNPY_SYMBOLS") or "ETHUSDT").strip()
        symbols = parse_symbol_list(inline) if inline else ["ETHUSDT"]
        return cls(
            enabled=_truthy("ICT_VNPY_ENABLED", default=False),
            shadow=_truthy("ICT_VNPY_SHADOW", default=False),
            symbols=symbols,
            equity_usdt=_float_env("ICT_VNPY_EQUITY_USDT", 1000.0),
            position_pct=_float_env("ICT_VNPY_POSITION_PCT", 1.0),
            leverage=_float_env("ICT_VNPY_LEVERAGE", 2.0),
            rr=_float_env("ICT_VNPY_RR", 1.5),
            compound=_truthy("ICT_VNPY_COMPOUND", default=True),
            hmm_filter=_truthy("ICT_VNPY_HMM_FILTER", default=True),
            hmm_stick=_float_env("ICT_VNPY_HMM_STICK", 0.97),
            hmm_confirm=int(_float_env("ICT_VNPY_HMM_CONFIRM", 3)),
            live_enabled=_truthy("ICT_VNPY_LIVE_ENABLED", default=False),
            live_leverage=_float_env("ICT_VNPY_LIVE_LEVERAGE", 2.0),
            max_notional_usdt=_float_env("ICT_VNPY_MAX_NOTIONAL_USDT", 0.0),
            limit_ttl_hours=_float_env("ICT_VNPY_LIMIT_TTL_HOURS", 4.0),
            setup_ttl_hours=_float_env("ICT_VNPY_SETUP_TTL_HOURS", 6.0),
            breakout_ttl_hours=_float_env("ICT_VNPY_BREAKOUT_TTL_HOURS", 2.0),
            init_bar_days=max(1, min(10, int(_float_env("ICT_VNPY_INIT_BAR_DAYS", 3)))),
        )

    def is_vnpy_engine(self) -> bool:
        return str(self.engine).strip().lower() == "vnpy"

    def symbol_list(self) -> List[str]:
        return list(self.symbols or [])

    def orb_session_cfg(self):
        from orb.core.config import OrbConfig

        return OrbConfig.from_env()

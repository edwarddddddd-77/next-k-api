"""Aberration vnpy lane 配置（ABERRATION_VNPY_*）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from orb.core.symbols import parse_symbol_list
from orb.aberration.paths import resolve_aberration_symbols_path


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


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return int(default)
    try:
        return int(float(str(raw).strip()))
    except ValueError:
        return int(default)


def _load_symbols() -> List[str]:
    inline = (os.getenv("ABERRATION_VNPY_SYMBOLS") or "").strip()
    if inline:
        return parse_symbol_list(inline)
    path_raw = (os.getenv("ABERRATION_VNPY_SYMBOLS_FILE") or "").strip()
    path = Path(path_raw) if path_raw else resolve_aberration_symbols_path()
    if path.is_file():
        return parse_symbol_list(path.read_text(encoding="utf-8"))
    return parse_symbol_list("BTCUSDT,SOLUSDT,BNBUSDT")


@dataclass
class AberrationVnpyConfig:
    lane: str = "aberration"
    engine: str = "vnpy"
    enabled: bool = False
    shadow: bool = False
    symbols: List[str] | None = None
    n_period: int = 35
    k_up: float = 2.0
    k_down: float = 2.0
    bar_hours: int = 1
    equity_usdt: float = 500.0
    position_pct: float = 1.0
    leverage: float = 2.0
    compound: bool = True
    live_enabled: bool = False
    live_leverage: float = 2.0
    max_notional_usdt: float = 0.0
    init_bar_days: int = 14

    @classmethod
    def from_env(cls) -> "AberrationVnpyConfig":
        return cls(
            enabled=_truthy("ABERRATION_VNPY_ENABLED", default=False),
            shadow=_truthy("ABERRATION_VNPY_SHADOW", default=False),
            symbols=_load_symbols(),
            n_period=_int_env("ABERRATION_VNPY_N_PERIOD", 35),
            k_up=_float_env("ABERRATION_VNPY_K_UP", 2.0),
            k_down=_float_env("ABERRATION_VNPY_K_DOWN", 2.0),
            bar_hours=max(1, min(24, _int_env("ABERRATION_VNPY_BAR_HOURS", 1))),
            equity_usdt=_float_env("ABERRATION_VNPY_EQUITY_USDT", 500.0),
            position_pct=_float_env("ABERRATION_VNPY_POSITION_PCT", 1.0),
            leverage=_float_env("ABERRATION_VNPY_LEVERAGE", 2.0),
            compound=_truthy("ABERRATION_VNPY_COMPOUND", default=True),
            live_enabled=_truthy("ABERRATION_VNPY_LIVE_ENABLED", default=False),
            live_leverage=_float_env("ABERRATION_VNPY_LIVE_LEVERAGE", 2.0),
            max_notional_usdt=_float_env("ABERRATION_VNPY_MAX_NOTIONAL_USDT", 0.0),
            init_bar_days=max(7, _int_env("ABERRATION_VNPY_INIT_BAR_DAYS", 14)),
        )

    def is_vnpy_engine(self) -> bool:
        return str(self.engine).strip().lower() == "vnpy"

    def symbol_list(self) -> List[str]:
        return list(self.symbols or [])

    def orb_session_cfg(self):
        from orb.core.config import OrbConfig

        return OrbConfig.from_env()

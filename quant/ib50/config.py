"""IB50 vnpy lane 配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from quant.common.config import OrbConfig
from quant.common.exchange_env import resolve_live_exchange_id, resolve_market_data_exchange_id
from quant.common.symbols import parse_symbol_list
from quant.ib50.core import normalize_direction_mode, parse_weekday_filter
from quant.ib50.paths import resolve_ib50_symbols_path
from quant.ib50.switches import IB50_SWITCH


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


def _parse_hm(raw: str, default_h: int, default_m: int) -> tuple[int, int]:
    text = (raw or "").strip()
    if not text:
        return default_h, default_m
    if ":" in text:
        parts = text.split(":", 1)
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return default_h, default_m
    try:
        return int(float(text)), 0
    except ValueError:
        return default_h, default_m


@dataclass
class Ib50Config:
    lane: str = "ib50"
    engine: str = "vnpy"
    enabled: bool = False
    shadow: bool = False
    symbols_file: str = ""
    symbols: List[str] | None = None
    equity_usdt: float = 50.0
    risk_pct: float = 0.01
    risk_per_trade_usdt: float = 0.0
    compound: bool = True
    ib_minutes: int = 60
    direction_mode: str = "continuation"
    allowed_weekdays: str = ""
    rth_only: bool = True
    eod_flat: bool = True
    exit_hour: int = 15
    exit_minute: int = 50
    entry_end_hour: int = 15
    entry_end_minute: int = 0
    fee_maker_bps: float = 2.0
    fee_taker_bps: float = 4.0
    macro_filter: bool = True
    one_trade_per_session: bool = True
    live_enabled: bool = False
    live_exchange: str = "binance"
    market_data_exchange: str = "binance"
    live_leverage: float = 5.0
    max_notional_usdt: float = 0.0
    max_open_positions: int = 7
    vnpy_idle_outside_rth: bool = True

    @classmethod
    def from_env(cls) -> "Ib50Config":
        prefix = "IB50_VNPY_"
        alt = "IB50_"
        sym_file = (os.getenv(f"{prefix}SYMBOLS_FILE") or "").strip() or str(resolve_ib50_symbols_path())
        inline = (os.getenv(f"{prefix}SYMBOLS") or os.getenv(f"{alt}SYMBOLS") or "").strip()
        symbols: List[str] | None = None
        if inline:
            symbols = parse_symbol_list(inline)
        elif Path(sym_file).is_file():
            symbols = parse_symbol_list(Path(sym_file).read_text(encoding="utf-8"))
        ee_h, ee_m = _parse_hm(
            os.getenv(f"{prefix}ENTRY_END") or os.getenv(f"{alt}ENTRY_END") or "15:00",
            15,
            0,
        )
        return cls(
            enabled=IB50_SWITCH.enabled(),
            shadow=IB50_SWITCH.shadow(),
            symbols_file=sym_file,
            symbols=symbols,
            equity_usdt=_float_env(f"{prefix}EQUITY_USDT", _float_env(f"{alt}EQUITY_USDT", 50.0)),
            risk_pct=_float_env(f"{prefix}RISK_PCT", _float_env(f"{alt}RISK_PCT", 0.01)),
            risk_per_trade_usdt=_float_env(
                f"{prefix}RISK_PER_TRADE",
                _float_env(f"{alt}RISK_PER_TRADE_USDT", 0.0),
            ),
            compound=_truthy(f"{prefix}COMPOUND", default=True),
            ib_minutes=max(15, _int_env(f"{prefix}IB_MINUTES", _int_env(f"{alt}IB_MINUTES", 60))),
            direction_mode=normalize_direction_mode(
                os.getenv(f"{prefix}DIRECTION") or os.getenv(f"{alt}DIRECTION") or "continuation"
            ),
            allowed_weekdays=(
                os.getenv(f"{prefix}WEEKDAYS") or os.getenv(f"{alt}WEEKDAYS") or ""
            ).strip(),
            rth_only=_truthy(f"{prefix}RTH_ONLY", default=True),
            eod_flat=_truthy(f"{prefix}EOD_FLAT", default=True),
            exit_hour=_int_env(f"{prefix}EXIT_HOUR", _int_env(f"{alt}EXIT_HOUR", 15)),
            exit_minute=_int_env(f"{prefix}EXIT_MINUTE", _int_env(f"{alt}EXIT_MINUTE", 50)),
            entry_end_hour=ee_h,
            entry_end_minute=ee_m,
            fee_maker_bps=_float_env(f"{alt}FEE_MAKER_BPS", 2.0),
            fee_taker_bps=_float_env(f"{alt}FEE_TAKER_BPS", 4.0),
            macro_filter=_truthy(f"{alt}MACRO_FILTER", default=True),
            one_trade_per_session=_truthy(f"{alt}ONE_TRADE_PER_SESSION", default=True),
            live_enabled=IB50_SWITCH.live(),
            live_exchange=resolve_live_exchange_id(),
            market_data_exchange=resolve_market_data_exchange_id(),
            live_leverage=_float_env(f"{prefix}LIVE_LEVERAGE", _float_env(f"{alt}LIVE_LEVERAGE", 5.0)),
            max_notional_usdt=_float_env(f"{prefix}MAX_NOTIONAL_USDT", 0.0),
            max_open_positions=max(
                0,
                _int_env(f"{prefix}MAX_OPEN_POSITIONS", _int_env(f"{alt}MAX_OPEN_POSITIONS", 7)),
            ),
            vnpy_idle_outside_rth=_truthy(f"{prefix}IDLE_OUTSIDE_RTH", default=True),
        )

    def symbol_list(self) -> List[str]:
        if self.symbols:
            return list(self.symbols)
        p = Path(self.symbols_file)
        if p.is_file():
            return parse_symbol_list(p.read_text(encoding="utf-8"))
        return []

    def is_vnpy_engine(self) -> bool:
        return str(self.engine).lower() == "vnpy"

    def session_cfg(self) -> OrbConfig:
        return OrbConfig.from_env()

    def weekday_filter(self):
        return parse_weekday_filter(self.allowed_weekdays)

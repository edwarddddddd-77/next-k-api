"""Donchian breakout lane config."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from quant.breakout_donchian.paths import resolve_breakout_donchian_symbols_path
from quant.breakout_donchian.switches import BREAKOUT_DONCHIAN_SWITCH
from quant.common.exchange_env import resolve_live_exchange_id, resolve_market_data_exchange_id
from quant.common.scanner_watchlist import merge_symbol_pools
from quant.common.symbols import parse_symbol_list


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


def _bool_env(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


@dataclass
class BreakoutDonchianConfig:
    lane: str = "breakout_donchian"
    engine: str = "vnpy"
    enabled: bool = False
    shadow: bool = False
    symbols_file: str = ""
    symbols: List[str] | None = None
    use_scanner_watchlist: bool = True
    scanner_watchlist_max: int = 20
    equity_usdt: float = 100.0
    risk_pct: float = 0.01
    compound: bool = True
    live_enabled: bool = False
    live_exchange: str = "binance"
    market_data_exchange: str = "binance"
    live_leverage: float = 5.0
    max_notional_usdt: float = 0.0
    max_open_positions: int = 3
    init_bar_days: int = 120
    signal_minutes: int = 1440
    require_weekly_confirm: bool = True
    weekly_confirm_mode: str = "trend"
    weekly_trend_ma_period: int = 10
    weekly_breakout_mode: str = "standard"
    check_1h_bonus: bool = True
    risk_mult_base: float = 1.0
    risk_mult_triple: float = 1.25
    weekly_lookback: int = 10
    weekly_vol_lookback: int = 10
    weekly_vol_mult: float = 1.20
    weekly_strong_close_pct: float = 0.55
    hourly_lookback: int = 24
    hourly_vol_lookback: int = 24
    hourly_vol_mult: float = 1.35
    hourly_strong_close_pct: float = 0.60
    hourly_strict_atr_mult: float = 1.2
    lookback: int = 20
    vol_lookback: int = 20
    vol_mult: float = 1.30
    strong_close_pct: float = 0.60
    strict_vol_mult: float = 1.6
    strict_atr_mult: float = 1.3
    sl_atr_mult: float = 1.2
    sl_level_buffer: float = 0.015
    atr_period: int = 14
    breakout_mode: str = "strict"
    exit_target: str = "tp1"
    signal_flip_exit: bool = False
    tp1_rr: float = 2.0
    tp2_rr: float = 2.5
    tp3_rr: float = 3.0
    heavy_symbols: tuple[str, ...] = ("BTC", "ETH")
    heavy_symbol_risk_mult: float = 0.5
    long_only: bool = True
    rth_only: bool = False
    eod_flat: bool = False
    vnpy_idle_outside_rth: bool = False

    @classmethod
    def from_env(cls) -> "BreakoutDonchianConfig":
        prefix = "BREAKOUT_DONCHIAN_VNPY_"
        alt = "BREAKOUT_DONCHIAN_"
        sym_file = (os.getenv(f"{prefix}SYMBOLS_FILE") or "").strip() or str(
            resolve_breakout_donchian_symbols_path()
        )
        inline = (os.getenv(f"{prefix}SYMBOLS") or os.getenv(f"{alt}SYMBOLS") or "").strip()
        symbols: List[str] | None = None
        if inline:
            symbols = parse_symbol_list(inline)
        elif Path(sym_file).is_file():
            symbols = parse_symbol_list(Path(sym_file).read_text(encoding="utf-8"))
        return cls(
            enabled=BREAKOUT_DONCHIAN_SWITCH.enabled(),
            shadow=BREAKOUT_DONCHIAN_SWITCH.shadow(),
            symbols_file=sym_file,
            symbols=symbols,
            use_scanner_watchlist=_bool_env("SCANNER_WATCHLIST_ENABLED", True),
            equity_usdt=_float_env(f"{prefix}EQUITY_USDT", _float_env(f"{alt}EQUITY_USDT", 100.0)),
            risk_pct=_float_env(f"{prefix}RISK_PCT", _float_env(f"{alt}RISK_PCT", 0.01)),
            compound=_bool_env(f"{prefix}COMPOUND", _bool_env(f"{alt}COMPOUND", True)),
            live_enabled=BREAKOUT_DONCHIAN_SWITCH.live(),
            live_exchange=resolve_live_exchange_id(),
            market_data_exchange=resolve_market_data_exchange_id(),
            live_leverage=_float_env(f"{prefix}LIVE_LEVERAGE", _float_env(f"{alt}LIVE_LEVERAGE", 5.0)),
            max_notional_usdt=_float_env(f"{prefix}MAX_NOTIONAL_USDT", 0.0),
            max_open_positions=max(0, _int_env(f"{prefix}MAX_OPEN_POSITIONS", 3)),
        )

    def symbol_list(self) -> List[str]:
        base: List[str]
        if self.symbols:
            base = list(self.symbols)
        else:
            p = Path(self.symbols_file)
            base = parse_symbol_list(p.read_text(encoding="utf-8")) if p.is_file() else []
        return merge_symbol_pools(
            base,
            use_watchlist=self.use_scanner_watchlist,
            watchlist_max=self.scanner_watchlist_max,
        )

    def is_vnpy_engine(self) -> bool:
        return str(self.engine).lower() == "vnpy"

    def orb_session_cfg(self):
        from quant.common.config import OrbConfig

        return OrbConfig.from_env()

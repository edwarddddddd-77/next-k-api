"""KAMA Trend lane 配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from quant.common.exchange_env import resolve_live_exchange_id, resolve_market_data_exchange_id
from quant.common.symbols import parse_symbol_list
from quant.kama_trend.paths import resolve_kama_trend_symbols_path
from quant.kama_trend.switches import KAMA_TREND_SWITCH


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


@dataclass
class KamaTrendConfig:
    lane: str = "kama_trend"
    engine: str = "vnpy"
    enabled: bool = False
    shadow: bool = False
    symbols_file: str = ""
    symbols: List[str] | None = None
    equity_usdt: float = 100.0
    risk_pct: float = 0.03
    position_size_mult: float = 3.0
    compound: bool = True
    live_enabled: bool = False
    live_exchange: str = "binance"
    market_data_exchange: str = "binance"
    live_leverage: float = 5.0
    max_notional_usdt: float = 0.0
    max_open_positions: int = 1
    init_bar_days: int = 30
    signal_minutes: int = 15
    kama_period: int = 14
    adx_period: int = 14
    adx_min: float = 50.0
    chop_period: int = 14
    chop_max: float = 50.0
    bb_period: int = 20
    bb_width_max_pct: float = 7.0
    cooldown_bars: int = 10
    stop_atr: float = 2.5
    tp_atr: float = 2.5
    rth_only: bool = False
    eod_flat: bool = False
    vnpy_idle_outside_rth: bool = False

    @classmethod
    def from_env(cls) -> "KamaTrendConfig":
        sym_file = (os.getenv("KAMA_TREND_VNPY_SYMBOLS_FILE") or "").strip() or str(
            resolve_kama_trend_symbols_path()
        )
        inline = (os.getenv("KAMA_TREND_VNPY_SYMBOLS") or os.getenv("KAMA_TREND_SYMBOLS") or "").strip()
        symbols: List[str] | None = None
        if inline:
            symbols = parse_symbol_list(inline)
        elif Path(sym_file).is_file():
            symbols = parse_symbol_list(Path(sym_file).read_text(encoding="utf-8"))
        return cls(
            enabled=KAMA_TREND_SWITCH.enabled(),
            shadow=KAMA_TREND_SWITCH.shadow(),
            symbols_file=sym_file,
            symbols=symbols,
            equity_usdt=_float_env("KAMA_TREND_VNPY_EQUITY_USDT", _float_env("KAMA_TREND_EQUITY_USDT", 100.0)),
            risk_pct=_float_env("KAMA_TREND_VNPY_RISK_PCT", _float_env("KAMA_TREND_RISK_PCT", 0.03)),
            position_size_mult=_float_env(
                "KAMA_TREND_VNPY_SIZE_MULT",
                _float_env("KAMA_TREND_SIZE_MULT", 3.0),
            ),
            compound=(os.getenv("KAMA_TREND_VNPY_COMPOUND") or os.getenv("KAMA_TREND_COMPOUND") or "1")
            .strip()
            .lower()
            not in ("0", "false", "no", "off"),
            live_enabled=KAMA_TREND_SWITCH.live(),
            live_exchange=resolve_live_exchange_id(),
            market_data_exchange=resolve_market_data_exchange_id(),
            live_leverage=_float_env("KAMA_TREND_VNPY_LIVE_LEVERAGE", _float_env("KAMA_TREND_LIVE_LEVERAGE", 5.0)),
            max_notional_usdt=_float_env("KAMA_TREND_VNPY_MAX_NOTIONAL_USDT", 0.0),
            max_open_positions=max(0, _int_env("KAMA_TREND_VNPY_MAX_OPEN_POSITIONS", 1)),
            init_bar_days=max(7, _int_env("KAMA_TREND_VNPY_INIT_BAR_DAYS", 30)),
            signal_minutes=max(1, _int_env("KAMA_TREND_VNPY_SIGNAL_MINUTES", 15)),
            kama_period=max(2, _int_env("KAMA_TREND_VNPY_KAMA_PERIOD", 14)),
            adx_period=max(2, _int_env("KAMA_TREND_VNPY_ADX_PERIOD", 14)),
            adx_min=_float_env("KAMA_TREND_VNPY_ADX_MIN", 50.0),
            chop_period=max(2, _int_env("KAMA_TREND_VNPY_CHOP_PERIOD", 14)),
            chop_max=_float_env("KAMA_TREND_VNPY_CHOP_MAX", 50.0),
            bb_period=max(5, _int_env("KAMA_TREND_VNPY_BB_PERIOD", 20)),
            bb_width_max_pct=_float_env("KAMA_TREND_VNPY_BB_WIDTH_MAX_PCT", 7.0),
            cooldown_bars=max(0, _int_env("KAMA_TREND_VNPY_COOLDOWN_BARS", 10)),
            stop_atr=_float_env("KAMA_TREND_VNPY_STOP_ATR", 2.5),
            tp_atr=_float_env("KAMA_TREND_VNPY_TP_ATR", 2.5),
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

    def orb_session_cfg(self):
        from quant.common.config import OrbConfig

        return OrbConfig.from_env()

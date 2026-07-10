"""Smart Breakout Targets lane 配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from quant.common.exchange_env import resolve_live_exchange_id, resolve_market_data_exchange_id
from quant.common.symbols import parse_symbol_list
from quant.squeeze_breakout.paths import resolve_squeeze_breakout_symbols_path
from quant.squeeze_breakout.switches import SQUEEZE_BREAKOUT_SWITCH


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
class SqueezeBreakoutConfig:
    lane: str = "squeeze_breakout"
    engine: str = "vnpy"
    enabled: bool = False
    shadow: bool = False
    symbols_file: str = ""
    symbols: List[str] | None = None
    equity_usdt: float = 100.0
    risk_pct: float = 0.02
    compound: bool = True
    live_enabled: bool = False
    live_exchange: str = "binance"
    market_data_exchange: str = "binance"
    live_leverage: float = 5.0
    max_notional_usdt: float = 0.0
    max_open_positions: int = 2
    init_bar_days: int = 30
    signal_minutes: int = 15
    squeeze_length: int = 20
    bb_mult: float = 2.0
    squeeze_threshold: float = 0.6
    atr_compress_ratio: float = 0.75
    min_squeeze_bars: int = 5
    impulse_mult: float = 0.8
    sl_atr_buffer: float = 0.5
    tp1_rr: float = 1.0
    tp2_rr: float = 2.0
    tp3_rr: float = 3.0
    prevent_overlap: bool = True
    volume_filter: bool = False
    volume_mult: float = 1.5
    rth_only: bool = False
    eod_flat: bool = False
    vnpy_idle_outside_rth: bool = False

    @classmethod
    def from_env(cls) -> "SqueezeBreakoutConfig":
        prefix = "SQUEEZE_BREAKOUT_VNPY_"
        alt = "SQUEEZE_BREAKOUT_"
        sym_file = (os.getenv(f"{prefix}SYMBOLS_FILE") or "").strip() or str(
            resolve_squeeze_breakout_symbols_path()
        )
        inline = (os.getenv(f"{prefix}SYMBOLS") or os.getenv(f"{alt}SYMBOLS") or "").strip()
        symbols: List[str] | None = None
        if inline:
            symbols = parse_symbol_list(inline)
        elif Path(sym_file).is_file():
            symbols = parse_symbol_list(Path(sym_file).read_text(encoding="utf-8"))
        return cls(
            enabled=SQUEEZE_BREAKOUT_SWITCH.enabled(),
            shadow=SQUEEZE_BREAKOUT_SWITCH.shadow(),
            symbols_file=sym_file,
            symbols=symbols,
            equity_usdt=_float_env(f"{prefix}EQUITY_USDT", _float_env(f"{alt}EQUITY_USDT", 100.0)),
            risk_pct=_float_env(f"{prefix}RISK_PCT", _float_env(f"{alt}RISK_PCT", 0.02)),
            compound=_bool_env(f"{prefix}COMPOUND", _bool_env(f"{alt}COMPOUND", True)),
            live_enabled=SQUEEZE_BREAKOUT_SWITCH.live(),
            live_exchange=resolve_live_exchange_id(),
            market_data_exchange=resolve_market_data_exchange_id(),
            live_leverage=_float_env(f"{prefix}LIVE_LEVERAGE", _float_env(f"{alt}LIVE_LEVERAGE", 5.0)),
            max_notional_usdt=_float_env(f"{prefix}MAX_NOTIONAL_USDT", 0.0),
            max_open_positions=max(0, _int_env(f"{prefix}MAX_OPEN_POSITIONS", 2)),
            init_bar_days=max(7, _int_env(f"{prefix}INIT_BAR_DAYS", 30)),
            signal_minutes=max(1, _int_env(f"{prefix}SIGNAL_MINUTES", 15)),
            squeeze_length=max(5, _int_env(f"{prefix}SQUEEZE_LENGTH", 20)),
            bb_mult=_float_env(f"{prefix}BB_MULT", 2.0),
            squeeze_threshold=_float_env(f"{prefix}SQUEEZE_THRESHOLD", 0.6),
            atr_compress_ratio=_float_env(f"{prefix}ATR_COMPRESS_RATIO", 0.75),
            min_squeeze_bars=max(1, _int_env(f"{prefix}MIN_SQUEEZE_BARS", 5)),
            impulse_mult=_float_env(f"{prefix}IMPULSE_MULT", 0.8),
            sl_atr_buffer=_float_env(f"{prefix}SL_ATR_BUFFER", 0.5),
            tp1_rr=_float_env(f"{prefix}TP1_RR", 1.0),
            tp2_rr=_float_env(f"{prefix}TP2_RR", 2.0),
            tp3_rr=_float_env(f"{prefix}TP3_RR", 3.0),
            prevent_overlap=_bool_env(f"{prefix}PREVENT_OVERLAP", True),
            volume_filter=_bool_env(f"{prefix}VOLUME_FILTER", False),
            volume_mult=_float_env(f"{prefix}VOLUME_MULT", 1.5),
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

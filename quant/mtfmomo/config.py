"""MtfMomo2xA lane 配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from quant.common.exchange_env import resolve_live_exchange_id, resolve_market_data_exchange_id
from quant.common.symbols import parse_symbol_list
from quant.mtfmomo.paths import resolve_mtfmomo_symbols_path
from quant.mtfmomo.switches import MTFMOMO_SWITCH


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
class MtfMomoConfig:
    lane: str = "mtfmomo"
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
    entry_lb: int = 26
    ema_exit: int = 35
    ema_4h: int = 21
    ema_1d: int = 16
    stop_atr: float = 3.295829874337854
    tp_atr: float = 8.681332636811806
    rth_only: bool = False
    eod_flat: bool = False
    vnpy_idle_outside_rth: bool = False

    @classmethod
    def from_env(cls) -> "MtfMomoConfig":
        sym_file = (os.getenv("MTFMOMO_VNPY_SYMBOLS_FILE") or "").strip() or str(resolve_mtfmomo_symbols_path())
        inline = (os.getenv("MTFMOMO_VNPY_SYMBOLS") or os.getenv("MTFMOMO_SYMBOLS") or "").strip()
        symbols: List[str] | None = None
        if inline:
            symbols = parse_symbol_list(inline)
        elif Path(sym_file).is_file():
            symbols = parse_symbol_list(Path(sym_file).read_text(encoding="utf-8"))
        return cls(
            enabled=MTFMOMO_SWITCH.enabled(),
            shadow=MTFMOMO_SWITCH.shadow(),
            symbols_file=sym_file,
            symbols=symbols,
            equity_usdt=_float_env("MTFMOMO_VNPY_EQUITY_USDT", _float_env("MTFMOMO_EQUITY_USDT", 100.0)),
            risk_pct=_float_env("MTFMOMO_VNPY_RISK_PCT", _float_env("MTFMOMO_RISK_PCT", 0.02)),
            compound=(os.getenv("MTFMOMO_VNPY_COMPOUND") or os.getenv("MTFMOMO_COMPOUND") or "1").strip().lower()
            not in ("0", "false", "no", "off"),
            live_enabled=MTFMOMO_SWITCH.live(),
            live_exchange=resolve_live_exchange_id(),
            market_data_exchange=resolve_market_data_exchange_id(),
            live_leverage=_float_env("MTFMOMO_VNPY_LIVE_LEVERAGE", _float_env("MTFMOMO_LIVE_LEVERAGE", 5.0)),
            max_notional_usdt=_float_env("MTFMOMO_VNPY_MAX_NOTIONAL_USDT", 0.0),
            max_open_positions=max(0, _int_env("MTFMOMO_VNPY_MAX_OPEN_POSITIONS", 2)),
            init_bar_days=max(7, _int_env("MTFMOMO_VNPY_INIT_BAR_DAYS", 30)),
            entry_lb=_int_env("MTFMOMO_VNPY_ENTRY_LB", 26),
            ema_exit=_int_env("MTFMOMO_VNPY_EMA_EXIT", 35),
            ema_4h=_int_env("MTFMOMO_VNPY_EMA_4H", 21),
            ema_1d=_int_env("MTFMOMO_VNPY_EMA_1D", 16),
            stop_atr=_float_env("MTFMOMO_VNPY_STOP_ATR", 3.295829874337854),
            tp_atr=_float_env("MTFMOMO_VNPY_TP_ATR", 8.681332636811806),
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
        """lane 辅助接口兼容；crypto 24h 不使用 RTH。"""
        from quant.common.config import OrbConfig

        return OrbConfig.from_env()

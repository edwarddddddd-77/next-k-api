"""Anchor Drift lane 配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from quant.anchor_drift.paths import resolve_anchor_drift_symbols_path
from quant.anchor_drift.switches import ANCHOR_DRIFT_SWITCH
from quant.common.config import OrbConfig
from quant.common.exchange_env import resolve_live_exchange_id, resolve_market_data_exchange_id
from quant.common.symbols import parse_symbol_list


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


@dataclass
class AnchorDriftConfig:
    lane: str = "anchor_drift"
    engine: str = "vnpy"
    enabled: bool = False
    shadow: bool = False
    symbols_file: str = ""
    symbols: List[str] | None = None
    equity_usdt: float = 14.0
    risk_pct: float = 0.01
    compound: bool = True
    live_enabled: bool = False
    live_exchange: str = "binance"
    market_data_exchange: str = "binance"
    live_leverage: float = 5.0
    max_notional_usdt: float = 0.0
    max_open_positions: int = 3
    non_rth_only: bool = True
    rth_only: bool = False
    eod_flat: bool = False
    vnpy_idle_outside_rth: bool = False
    signal_threshold: float = 0.015
    converge_threshold: float = 0.003
    max_adverse_extension: float = 0.025
    preopen_flat_minutes: int = 5
    tick_interval_sec: float = 30.0

    @classmethod
    def from_env(cls) -> "AnchorDriftConfig":
        sym_file = (os.getenv("ANCHOR_DRIFT_VNPY_SYMBOLS_FILE") or "").strip() or str(
            resolve_anchor_drift_symbols_path()
        )
        inline = (
            os.getenv("ANCHOR_DRIFT_VNPY_SYMBOLS") or os.getenv("ANCHOR_DRIFT_SYMBOLS") or ""
        ).strip()
        symbols: List[str] | None = None
        if inline:
            symbols = parse_symbol_list(inline)
        elif Path(sym_file).is_file():
            symbols = parse_symbol_list(Path(sym_file).read_text(encoding="utf-8"))
        return cls(
            enabled=ANCHOR_DRIFT_SWITCH.enabled(),
            shadow=ANCHOR_DRIFT_SWITCH.shadow(),
            symbols_file=sym_file,
            symbols=symbols,
            equity_usdt=_float_env(
                "ANCHOR_DRIFT_VNPY_EQUITY_USDT",
                _float_env("ANCHOR_DRIFT_EQUITY_USDT", 14.0),
            ),
            risk_pct=_float_env(
                "ANCHOR_DRIFT_VNPY_RISK_PCT",
                _float_env("ANCHOR_DRIFT_RISK_PCT", 0.01),
            ),
            compound=_truthy("ANCHOR_DRIFT_VNPY_COMPOUND", default=True),
            live_enabled=ANCHOR_DRIFT_SWITCH.live(),
            live_exchange=resolve_live_exchange_id(),
            market_data_exchange=resolve_market_data_exchange_id(),
            live_leverage=_float_env(
                "ANCHOR_DRIFT_VNPY_LIVE_LEVERAGE",
                _float_env("ANCHOR_DRIFT_LIVE_LEVERAGE", 5.0),
            ),
            max_notional_usdt=_float_env("ANCHOR_DRIFT_VNPY_MAX_NOTIONAL_USDT", 0.0),
            max_open_positions=max(0, _int_env("ANCHOR_DRIFT_VNPY_MAX_OPEN_POSITIONS", 3)),
            signal_threshold=_float_env("ANCHOR_DRIFT_VNPY_SIGNAL_THRESHOLD", 0.015),
            converge_threshold=_float_env("ANCHOR_DRIFT_VNPY_CONVERGE_THRESHOLD", 0.003),
            max_adverse_extension=_float_env("ANCHOR_DRIFT_VNPY_MAX_ADVERSE_EXTENSION", 0.025),
            preopen_flat_minutes=max(1, _int_env("ANCHOR_DRIFT_VNPY_PREOPEN_FLAT_MINUTES", 5)),
            tick_interval_sec=max(5.0, _float_env("ANCHOR_DRIFT_VNPY_TICK_INTERVAL_SEC", 30.0)),
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
        cfg = OrbConfig.from_env()
        cfg.risk_pct = float(self.risk_pct)
        return cfg

    def orb_session_cfg(self):
        return self.session_cfg()

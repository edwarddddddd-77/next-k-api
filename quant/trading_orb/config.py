"""Trading ORB vnpy lane 配置（ORB_VNPY_* / 兼容 ORB_*）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from quant.common.config import OrbConfig
from quant.common.exchange_env import resolve_live_exchange_id, resolve_market_data_exchange_id
from quant.common.symbols import parse_symbol_list
from quant.trading_orb.paths import resolve_orb_vnpy_symbols_path
from quant.trading_orb.switches import TRADING_ORB_SWITCH


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
class OrbVnpyConfig:
    lane: str = "trading_orb"
    engine: str = "vnpy"
    enabled: bool = True
    shadow: bool = False
    symbols_file: str = ""
    symbols: List[str] | None = None
    equity_usdt: float = 14.0
    risk_pct: float = 0.01
    risk_per_trade_usdt: float = 0.0
    compound: bool = True
    rth_only: bool = True
    eod_flat: bool = True
    exit_hour: int = 15
    exit_minute: int = 50
    entry_start_hour: int = 10
    entry_start_minute: int = 0
    entry_end_hour: int = 11
    entry_end_minute: int = 30
    or_minutes: int = 20
    vol_thresh: float = 1.2
    vol_lookback_days: int = 20
    stop_or_mult: float = 0.5
    target_or_mult: float = 0.75
    breakeven_or_mult: float = 1.0
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
    def from_env(cls) -> "OrbVnpyConfig":
        sym_file = (os.getenv("ORB_VNPY_SYMBOLS_FILE") or "").strip() or str(resolve_orb_vnpy_symbols_path())
        inline = (os.getenv("ORB_VNPY_SYMBOLS") or os.getenv("ORB_SYMBOLS") or "").strip()
        symbols: List[str] | None = None
        if inline:
            symbols = parse_symbol_list(inline)
        elif Path(sym_file).is_file():
            symbols = parse_symbol_list(Path(sym_file).read_text(encoding="utf-8"))
        live_on = TRADING_ORB_SWITCH.live()
        es_h, es_m = _parse_hm(
            os.getenv("ORB_VNPY_ENTRY_START") or os.getenv("ORB_ENTRY_START") or "10:00",
            10,
            0,
        )
        ee_h, ee_m = _parse_hm(
            os.getenv("ORB_VNPY_ENTRY_END")
            or os.getenv("ORB_ENTRY_END")
            or os.getenv("ORB_NO_ENTRY_AFTER")
            or "11:30",
            11,
            30,
        )
        return cls(
            enabled=TRADING_ORB_SWITCH.enabled(),
            shadow=TRADING_ORB_SWITCH.shadow(),
            symbols_file=sym_file,
            symbols=symbols,
            equity_usdt=_float_env("ORB_VNPY_EQUITY_USDT", _float_env("ORB_ACCOUNT_EQUITY", 50.0)),
            risk_pct=_float_env("ORB_VNPY_RISK_PCT", _float_env("ORB_RISK_PCT", 0.01)),
            risk_per_trade_usdt=_float_env(
                "ORB_VNPY_RISK_PER_TRADE",
                _float_env("ORB_RISK_PER_TRADE_USDT", 0.0),
            ),
            compound=_truthy("ORB_VNPY_COMPOUND", default=True),
            rth_only=_truthy("ORB_VNPY_RTH_ONLY", default=True),
            eod_flat=_truthy("ORB_VNPY_EOD_FLAT", default=True),
            exit_hour=_int_env("ORB_VNPY_EXIT_HOUR", _int_env("ORB_EXIT_HOUR", 15)),
            exit_minute=_int_env("ORB_VNPY_EXIT_MINUTE", _int_env("ORB_EXIT_MINUTE", 50)),
            entry_start_hour=es_h,
            entry_start_minute=es_m,
            entry_end_hour=ee_h,
            entry_end_minute=ee_m,
            or_minutes=_int_env("ORB_VNPY_OR_MINUTES", _int_env("ORB_OR_MINUTES", 20)),
            vol_thresh=_float_env(
                "ORB_VNPY_VOL_THRESH",
                _float_env("ORB_VOL_THRESH", _float_env("ORB_VOL_MULT", 1.2)),
            ),
            vol_lookback_days=_int_env("ORB_VNPY_VOL_LOOKBACK_DAYS", _int_env("ORB_VOL_LOOKBACK_DAYS", 20)),
            stop_or_mult=_float_env("ORB_VNPY_STOP_OR_MULT", _float_env("ORB_STOP_OR_MULT", 0.5)),
            target_or_mult=_float_env("ORB_VNPY_TARGET_OR_MULT", _float_env("ORB_TARGET_OR_MULT", 0.75)),
            breakeven_or_mult=_float_env(
                "ORB_VNPY_BREAKEVEN_OR_MULT",
                _float_env("ORB_BREAKEVEN_OR_MULT", 1.0),
            ),
            fee_maker_bps=_float_env("ORB_FEE_MAKER_BPS", 2.0),
            fee_taker_bps=_float_env("ORB_FEE_TAKER_BPS", 4.0),
            macro_filter=_truthy("ORB_MACRO_FILTER", default=True),
            one_trade_per_session=_truthy("ORB_ONE_TRADE_PER_SESSION", default=True),
            live_enabled=live_on,
            live_exchange=resolve_live_exchange_id(),
            market_data_exchange=resolve_market_data_exchange_id(),
            live_leverage=_float_env("ORB_VNPY_LIVE_LEVERAGE", _float_env("ORB_LIVE_LEVERAGE", 5.0)),
            max_notional_usdt=_float_env("ORB_VNPY_MAX_NOTIONAL_USDT", 0.0),
            max_open_positions=max(
                0,
                _int_env("ORB_VNPY_MAX_OPEN_POSITIONS", _int_env("ORB_MAX_OPEN_POSITIONS", 7)),
            ),
            vnpy_idle_outside_rth=_truthy("ORB_VNPY_IDLE_OUTSIDE_RTH", default=True),
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

    def orb_session_cfg(self) -> OrbConfig:
        cfg = OrbConfig.from_env()
        cfg.risk_pct = float(self.risk_pct)
        cfg.fixed_notional_usdt = 0.0
        cfg.or_minutes = int(self.or_minutes)
        return cfg

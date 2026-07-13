"""IBS lane 共享配置加载。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from quant.common.config import OrbConfig
from quant.common.exchange_env import resolve_live_exchange_id, resolve_market_data_exchange_id
from quant.common.strategy_switch import StrategySwitchSpec
from quant.common.symbols import parse_symbol_list
from quant.ibs.profile import PROFILE_DEFAULTS, PROFILE_TV, IbsProfileDefaults
from quant.ibs.symbols import resolve_ibs_trading_symbol
from quant.ibs.core import VALID_TRADE_TYPES, normalize_trade_type


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
class IbsLaneConfig:
    lane: str
    profile: str
    engine: str = "vnpy"
    enabled: bool = False
    shadow: bool = False
    symbols_file: str = ""
    symbols: List[str] | None = None
    equity_usdt: float = 100.0
    risk_pct: float = 0.01
    compound: bool = True
    live_enabled: bool = False
    live_exchange: str = "bitget_spot"
    market_data_exchange: str = "bitget_spot"
    product_type: str = "spot"
    live_leverage: float = 1.0
    max_notional_usdt: float = 0.0
    max_open_positions: int = 3
    init_bar_days: int = 400
    entry_threshold: float = 0.20
    exit_threshold: float = 0.50
    position_pct: float = 0.10
    stop_loss_pct: float = 0.05
    sma_period: int = 200
    trend_ma_type: str = "sma"
    trend_ma_period: int = 200
    trend_price_mode: str = "prev_close"
    min_entry_distance_pct: float = 0.0
    max_trade_duration_days: int = 0
    eval_after_close_minutes: int = 5
    exec_after_open_minutes: int = 1
    daily_bar_source: str = "session_5m"
    execute_at_next_open: bool = True
    signal_minutes: int = 5
    rth_only: bool = False
    eod_flat: bool = False
    vnpy_idle_outside_rth: bool = False
    trade_type: str = "long_only"

    @classmethod
    def from_env(
        cls,
        *,
        lane: str,
        profile: str,
        switch: StrategySwitchSpec,
        resolve_symbols_path: Callable[[], Path],
        env_prefix: str,
    ) -> "IbsLaneConfig":
        defaults = PROFILE_DEFAULTS[profile]
        product_type = (
            os.getenv(f"{env_prefix}_VNPY_PRODUCT_TYPE")
            or os.getenv(f"{env_prefix}_PRODUCT_TYPE")
            or defaults.product_type
            or "spot"
        ).strip().lower()
        if product_type not in ("spot", "perp"):
            product_type = defaults.product_type or "spot"
        default_live = "bitget_spot" if product_type == "spot" else "bitget"
        live_exchange = resolve_live_exchange_id(
            (os.getenv(f"{env_prefix}_VNPY_LIVE_EXCHANGE") or "").strip() or default_live
        )
        market_data_exchange = resolve_market_data_exchange_id(
            (os.getenv(f"{env_prefix}_VNPY_MARKET_DATA_EXCHANGE") or "").strip() or default_live
        )
        sym_file = (os.getenv(f"{env_prefix}_VNPY_SYMBOLS_FILE") or "").strip() or str(resolve_symbols_path())
        inline = (os.getenv(f"{env_prefix}_VNPY_SYMBOLS") or os.getenv(f"{env_prefix}_SYMBOLS") or "").strip()
        symbols: List[str] | None = None
        if inline:
            symbols = parse_symbol_list(inline)
        elif Path(sym_file).is_file():
            symbols = parse_symbol_list(Path(sym_file).read_text(encoding="utf-8"))
        trade_type = normalize_trade_type(
            os.getenv(f"{env_prefix}_VNPY_TRADE_TYPE") or defaults.trade_type or "long_only"
        )
        allowed = VALID_TRADE_TYPES if profile == PROFILE_TV else ("long_only",)
        if trade_type not in allowed:
            trade_type = normalize_trade_type(defaults.trade_type)
        return cls(
            lane=lane,
            profile=profile,
            enabled=switch.enabled(),
            shadow=switch.shadow(),
            symbols_file=sym_file,
            symbols=symbols,
            equity_usdt=_float_env(
                f"{env_prefix}_VNPY_EQUITY_USDT",
                _float_env(f"{env_prefix}_EQUITY_USDT", 100.0),
            ),
            risk_pct=_float_env(
                f"{env_prefix}_VNPY_RISK_PCT",
                _float_env(f"{env_prefix}_RISK_PCT", defaults.risk_pct),
            ),
            compound=_truthy(f"{env_prefix}_VNPY_COMPOUND", default=True),
            live_enabled=switch.live(),
            live_exchange=live_exchange,
            market_data_exchange=market_data_exchange,
            product_type=product_type,
            live_leverage=_float_env(
                f"{env_prefix}_VNPY_LIVE_LEVERAGE",
                _float_env(f"{env_prefix}_LIVE_LEVERAGE", 1.0 if product_type == "spot" else 5.0),
            ),
            max_notional_usdt=_float_env(f"{env_prefix}_VNPY_MAX_NOTIONAL_USDT", 0.0),
            max_open_positions=max(0, _int_env(f"{env_prefix}_VNPY_MAX_OPEN_POSITIONS", 3)),
            init_bar_days=max(30, _int_env(f"{env_prefix}_VNPY_INIT_BAR_DAYS", defaults.init_bar_days)),
            entry_threshold=_float_env(f"{env_prefix}_VNPY_ENTRY_THRESHOLD", defaults.entry_threshold),
            exit_threshold=_float_env(f"{env_prefix}_VNPY_EXIT_THRESHOLD", defaults.exit_threshold),
            position_pct=_float_env(f"{env_prefix}_VNPY_POSITION_PCT", defaults.position_pct),
            stop_loss_pct=_float_env(f"{env_prefix}_VNPY_STOP_LOSS_PCT", defaults.stop_loss_pct),
            sma_period=max(0, _int_env(f"{env_prefix}_VNPY_SMA_PERIOD", defaults.sma_period)),
            trend_ma_type=(
                os.getenv(f"{env_prefix}_VNPY_TREND_MA_TYPE") or defaults.trend_ma_type
            ).strip().lower(),
            trend_ma_period=max(
                0,
                _int_env(f"{env_prefix}_VNPY_TREND_MA_PERIOD", defaults.trend_ma_period),
            ),
            trend_price_mode=(
                os.getenv(f"{env_prefix}_VNPY_TREND_PRICE_MODE") or defaults.trend_price_mode
            ).strip().lower(),
            min_entry_distance_pct=_float_env(
                f"{env_prefix}_VNPY_MIN_ENTRY_DISTANCE_PCT",
                defaults.min_entry_distance_pct,
            ),
            max_trade_duration_days=max(
                0,
                _int_env(
                    f"{env_prefix}_VNPY_MAX_TRADE_DURATION_DAYS",
                    defaults.max_trade_duration_days,
                ),
            ),
            eval_after_close_minutes=max(1, _int_env(f"{env_prefix}_VNPY_EVAL_AFTER_CLOSE_MINUTES", 5)),
            exec_after_open_minutes=max(
                0,
                _int_env(f"{env_prefix}_VNPY_EXEC_AFTER_OPEN_MINUTES", defaults.exec_after_open_minutes),
            ),
            daily_bar_source=(
                os.getenv(f"{env_prefix}_VNPY_DAILY_BAR_SOURCE") or defaults.daily_bar_source
            ).strip().lower(),
            execute_at_next_open=_truthy(
                f"{env_prefix}_VNPY_EXECUTE_AT_NEXT_OPEN",
                default=defaults.execute_at_next_open,
            ),
            signal_minutes=max(1, _int_env(f"{env_prefix}_VNPY_SIGNAL_MINUTES", 5)),
            trade_type=trade_type,
        )

    def symbol_list(self) -> List[str]:
        raw: List[str]
        if self.symbols:
            raw = list(self.symbols)
        else:
            p = Path(self.symbols_file)
            if p.is_file():
                raw = parse_symbol_list(p.read_text(encoding="utf-8"))
            else:
                return []
        out: List[str] = []
        seen: set[str] = set()
        for item in raw:
            sym = resolve_ibs_trading_symbol(item, self.product_type)
            if sym and sym not in seen:
                seen.add(sym)
                out.append(sym)
        return out

    def is_vnpy_engine(self) -> bool:
        return str(self.engine).lower() == "vnpy"

    def orb_session_cfg(self) -> OrbConfig:
        return OrbConfig.from_env()

    def profile_defaults(self) -> IbsProfileDefaults:
        return PROFILE_DEFAULTS[self.profile]

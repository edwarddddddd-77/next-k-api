#!/usr/bin/env python3
"""ORB 策略配置（环境变量 ORB_*）。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from orb.tz import normalize_session_tz


def _truthy(raw: str, *, default: bool = False) -> bool:
    v = (raw if raw is not None else "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off")


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(float(str(raw).strip()))
    except ValueError:
        return default


DEFAULT_CRYPTO_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT"
DEFAULT_US_EQUITY_SYMBOLS = "COINUSDT,MSTRUSDT,GOOGLUSDT,QQQUSDT,TSMUSDT,EWYUSDT"
DEFAULT_SYMBOLS = DEFAULT_US_EQUITY_SYMBOLS
DEFAULT_MARKET = "us_equity"


def _market_defaults(market: str) -> dict:
    m = (market or "crypto").strip().lower()
    if m in ("us_equity", "equity", "stock", "stocks"):
        return {
            "market": "us_equity",
            "session_tz": "America/New_York",
            "session_open_time": "09:30",
            "session_close_time": "16:00",
            "regular_session_only": True,
            "symbols": DEFAULT_US_EQUITY_SYMBOLS,
            "max_open_positions": 6,
            # 上线默认：15m OR + 5m 突破 + 5%ATR + EoD + 每标 bot 10k + 1% 风险定仓
            "signal_interval": "5m",
            "or_minutes": 15,
            "entry_mode": "breakout",
            "confirm_bars": 1,
            "confirm_no_soften": False,
            "trade_window_minutes": 0,
            "min_or_width_pct": 0.0,
            "max_or_width_pct": 0.0,
            "vol_ma_period": 20,
            "vol_mult": 0.0,
            "sl_mode": "atr_pct",
            "atr_period": 14,
            "atr_sl_fraction": 0.05,
            "exit_mode": "eod",
            "tp_r_multiple": 0.0,
            "min_sl_pct": 0.0,
            "vwap_filter": False,
            "risk_pct": 0.01,
            "symbol_bot_equity_usdt": 10_000.0,
            "account_equity_usdt": 10_000.0,
            "fixed_notional_usdt": 0.0,
            "position_safety_pct": 0.15,
            "entry_tick_offset": 2,
            "tick_size": 0.01,
            "early_exit_minutes": 0,
            "macro_filter": True,
            "resolve_max_hold_ms": 0,
            "resolve_max_bars": 0,
            "resolve_at_session_close": True,
            "leverage": 5.0,
        }
    return {
        "market": "crypto",
        "session_tz": "UTC",
        "session_open_time": "",
        "session_close_time": "",
        "regular_session_only": False,
        "symbols": DEFAULT_CRYPTO_SYMBOLS,
    }


US_EQUITY_DEFAULTS = _market_defaults("us_equity")


def scan_interval_minutes_for_signal(signal_interval: str) -> int:
    """实盘扫描间隔（分钟）= 信号 K 线周期，与 walk-forward 回测同频。"""
    return {
        "1m": 1,
        "2m": 2,
        "3m": 3,
        "5m": 5,
        "15m": 15,
    }.get((signal_interval or "5m").strip().lower(), 5)


DEFAULT_SCAN_INTERVAL_MINUTES = scan_interval_minutes_for_signal(US_EQUITY_DEFAULTS["signal_interval"])


def default_scan_interval_minutes() -> int:
    """未设 ORB_SCAN_INTERVAL_MINUTES 时，跟 ORB_SIGNAL_INTERVAL（或内置 5m）对齐。"""
    iv = (os.getenv("ORB_SIGNAL_INTERVAL") or US_EQUITY_DEFAULTS["signal_interval"] or "5m").strip()
    return scan_interval_minutes_for_signal(iv)


@dataclass
class OrbConfig:
    """开盘区间突破 + 量价过滤 + 纸面/回测共用参数。"""

    enabled: bool = True
    market: str = DEFAULT_MARKET
    signal_interval: str = "5m"
    or_minutes: int = 15
    session_tz: str = "America/New_York"
    session_open_time: str = "09:30"
    session_close_time: str = "16:00"
    regular_session_only: bool = True
    entry_mode: str = "breakout"
    confirm_bars: int = 1
    confirm_no_soften: bool = False
    trade_window_minutes: int = 0
    one_trade_per_session: bool = True
    min_or_width_pct: float = 0.0
    max_or_width_pct: float = 0.0
    vol_ma_period: int = 20
    vol_mult: float = 0.0
    sl_mode: str = "atr_pct"
    atr_period: int = 14
    atr_sl_fraction: float = 0.05
    exit_mode: str = "eod"
    tp_r_multiple: float = 0.0
    sl_buffer_bps: float = 5.0
    min_sl_pct: float = 0.0
    vwap_filter: bool = False
    risk_pct: float = 0.01
    symbol_bot_equity_usdt: float = 10_000.0
    account_equity_usdt: float = 10_000.0
    fixed_notional_usdt: float = 0.0
    position_safety_pct: float = 0.15
    entry_tick_offset: int = 2
    tick_size: float = 0.01
    early_exit_minutes: int = 0
    macro_filter: bool = True
    margin_usdt: float = 100.0
    leverage: float = 5.0
    max_open_positions: int = 6
    resolve_max_bars: int = 0
    resolve_max_hold_ms: int = 0
    resolve_at_session_close: bool = True
    same_bar_rule: str = "pessimistic"
    db_skip_flat: bool = True
    symbols: str = DEFAULT_US_EQUITY_SYMBOLS

    @property
    def virtual_notional_usdt(self) -> float:
        return self.margin_usdt * self.leverage

    def per_symbol_bot_equity(self) -> float:
        """单标机器人虚拟本金（对齐 MOSS profile capital）。"""
        bot = float(getattr(self, "symbol_bot_equity_usdt", 0.0) or 0.0)
        if bot > 0:
            return bot
        return float(self.account_equity_usdt or 0.0)

    def default_paper_notional(self) -> float:
        """DB 缺省名义回退（仅当 virtual_notional_usdt 为空）；正常开仓应写入真实名义。"""
        if self.fixed_notional_usdt > 0:
            return float(self.fixed_notional_usdt)
        return float(self.virtual_notional_usdt)

    def uses_risk_sizing(self) -> bool:
        if self.fixed_notional_usdt > 0:
            return False
        eq = self.per_symbol_bot_equity()
        return self.risk_pct > 0 and eq > 0

    @classmethod
    def from_env(cls) -> "OrbConfig":
        md = _market_defaults(os.getenv("ORB_MARKET", DEFAULT_MARKET) or DEFAULT_MARKET)
        mode = (os.getenv("ORB_ENTRY_MODE") or md.get("entry_mode", "breakout") or "breakout").strip().lower()
        iv = (os.getenv("ORB_SIGNAL_INTERVAL") or md.get("signal_interval", "5m") or "5m").strip().lower()
        sbr = (os.getenv("ORB_SAME_BAR_RULE", "pessimistic") or "pessimistic").strip().lower()
        market = str(md["market"])
        session_tz = normalize_session_tz(
            os.getenv("ORB_SESSION_TZ") or md["session_tz"],
            market=market,
        )
        session_open = (os.getenv("ORB_SESSION_OPEN") or md["session_open_time"]).strip()
        session_close = (os.getenv("ORB_SESSION_CLOSE") or md["session_close_time"]).strip()
        if os.getenv("ORB_REGULAR_SESSION_ONLY") is None:
            regular_only = bool(md["regular_session_only"])
        else:
            regular_only = _truthy(os.getenv("ORB_REGULAR_SESSION_ONLY"), default=False)
        symbols_default = md["symbols"]
        exit_mode = (
            os.getenv("ORB_EXIT_MODE") or md.get("exit_mode", "fixed_r") or "fixed_r"
        ).strip().lower()
        if exit_mode not in ("eod", "fixed_r"):
            exit_mode = "eod" if md.get("exit_mode") == "eod" else "fixed_r"
        sl_mode = (os.getenv("ORB_SL_MODE") or md.get("sl_mode", "or_range") or "or_range").strip().lower()
        if sl_mode not in ("atr_pct", "or_range"):
            sl_mode = "atr_pct" if md.get("sl_mode") == "atr_pct" else "or_range"
        tp_default = float(md.get("tp_r_multiple", 1.5))
        tp_raw = _float_env("ORB_TP_R", tp_default)
        tp_r = tp_raw if exit_mode == "fixed_r" else 0.0
        if exit_mode == "fixed_r" and tp_r <= 0:
            tp_r = max(0.5, tp_default if tp_default > 0 else 1.5)
        return cls(
            enabled=_truthy(os.getenv("ORB_ENABLED", "1"), default=True),
            market=market,
            signal_interval=iv if iv in ("1m", "2m", "3m", "5m", "15m") else "5m",
            or_minutes=max(1, _int_env("ORB_OR_MINUTES", int(md.get("or_minutes", 15)))),
            session_tz=session_tz,
            session_open_time=session_open,
            session_close_time=session_close,
            regular_session_only=regular_only,
            entry_mode=mode if mode in ("breakout", "retest") else "breakout",
            confirm_bars=max(1, _int_env("ORB_CONFIRM_BARS", int(md.get("confirm_bars", 1)))),
            confirm_no_soften=(
                _truthy(os.getenv("ORB_CONFIRM_NO_SOFTEN"), default=False)
                if os.getenv("ORB_CONFIRM_NO_SOFTEN") is not None
                else bool(md.get("confirm_no_soften", False))
            ),
            trade_window_minutes=max(
                0, _int_env("ORB_TRADE_WINDOW_MINUTES", int(md.get("trade_window_minutes", 0)))
            ),
            one_trade_per_session=_truthy(os.getenv("ORB_ONE_TRADE_PER_SESSION", "1"), default=True),
            min_or_width_pct=_float_env(
                "ORB_MIN_RANGE_WIDTH_PCT", float(md.get("min_or_width_pct", 0.0))
            ),
            max_or_width_pct=_float_env(
                "ORB_MAX_RANGE_WIDTH_PCT", float(md.get("max_or_width_pct", 0.0))
            ),
            vol_ma_period=max(2, _int_env("ORB_VOL_MA_PERIOD", int(md.get("vol_ma_period", 20)))),
            vol_mult=_float_env("ORB_VOL_MULT", float(md.get("vol_mult", 0.0))),
            sl_mode=sl_mode,
            atr_period=max(2, _int_env("ORB_ATR_PERIOD", int(md.get("atr_period", 14)))),
            atr_sl_fraction=max(0.0, _float_env("ORB_ATR_SL_FRACTION", float(md.get("atr_sl_fraction", 0.05)))),
            exit_mode=exit_mode,
            tp_r_multiple=tp_r,
            risk_pct=max(0.0, _float_env("ORB_RISK_PCT", float(md.get("risk_pct", 0.01)))),
            account_equity_usdt=max(
                0.0,
                _float_env(
                    "ORB_ACCOUNT_EQUITY",
                    float(md.get("account_equity_usdt", md.get("symbol_bot_equity_usdt", 10_000.0))),
                ),
            ),
            symbol_bot_equity_usdt=max(
                0.0,
                _float_env(
                    "ORB_SYMBOL_BOT_EQUITY",
                    float(md.get("symbol_bot_equity_usdt", md.get("account_equity_usdt", 0.0))),
                ),
            ),
            fixed_notional_usdt=max(
                0.0, _float_env("ORB_FIXED_NOTIONAL", float(md.get("fixed_notional_usdt", 0.0)))
            ),
            position_safety_pct=min(0.9, max(0.0, _float_env("ORB_POSITION_SAFETY_PCT", float(md.get("position_safety_pct", 0.15))))),
            vwap_filter=(
                _truthy(os.getenv("ORB_VWAP_FILTER"), default=False)
                if os.getenv("ORB_VWAP_FILTER") is not None
                else bool(md.get("vwap_filter", False))
            ),
            entry_tick_offset=max(0, _int_env("ORB_ENTRY_TICK_OFFSET", int(md.get("entry_tick_offset", 2)))),
            tick_size=max(0.0, _float_env("ORB_TICK_SIZE", float(md.get("tick_size", 0.01)))),
            early_exit_minutes=max(
                0, _int_env("ORB_EARLY_EXIT_MINUTES", int(md.get("early_exit_minutes", 0)))
            ),
            macro_filter=(
                _truthy(os.getenv("ORB_MACRO_FILTER"), default=False)
                if os.getenv("ORB_MACRO_FILTER") is not None
                else bool(md.get("macro_filter", True))
            ),
            sl_buffer_bps=_float_env("ORB_SL_BUFFER_BPS", 5.0),
            min_sl_pct=_float_env("ORB_MIN_SL_PCT", float(md.get("min_sl_pct", 0.0))),
            margin_usdt=_float_env("ORB_MARGIN_USDT", 100.0),
            leverage=_float_env("ORB_LEVERAGE", float(md.get("leverage", 5.0))),
            max_open_positions=max(
                0,
                _int_env(
                    "ORB_MAX_OPEN_POSITIONS",
                    int(md.get("max_open_positions", 6)),
                ),
            ),
            resolve_max_bars=max(
                0, _int_env("ORB_RESOLVE_MAX_BARS", int(md.get("resolve_max_bars", 0)))
            ),
            resolve_max_hold_ms=max(
                0,
                _int_env(
                    "ORB_RESOLVE_MAX_HOLD_MS",
                    int(md.get("resolve_max_hold_ms", 0)),
                ),
            ),
            resolve_at_session_close=(
                _truthy(os.getenv("ORB_RESOLVE_AT_SESSION_CLOSE"), default=False)
                if os.getenv("ORB_RESOLVE_AT_SESSION_CLOSE") is not None
                else bool(md.get("resolve_at_session_close", False))
            ),
            same_bar_rule=sbr if sbr in ("pessimistic", "optimistic") else "pessimistic",
            db_skip_flat=_truthy(os.getenv("ORB_DB_SKIP_FLAT", "1"), default=True),
            symbols=(os.getenv("ORB_SYMBOLS") or symbols_default).strip(),
        )

    @classmethod
    def for_backtest(cls) -> "OrbConfig":
        return cls.from_env()

    def bar_step_ms(self) -> int:
        return {
            "1m": 60_000,
            "2m": 120_000,
            "3m": 180_000,
            "5m": 300_000,
            "15m": 900_000,
        }.get(self.signal_interval.strip().lower(), 300_000)

    def symbol_list(self) -> list[str]:
        return [x.strip().upper() for x in self.symbols.split(",") if x.strip()]

#!/usr/bin/env python3
"""ZCT VWAP 策略配置：环境变量收拢为可注入的 dataclass（实盘 / 回测 / 单测）。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional


def _truthy(raw: str, *, default: bool = False) -> bool:
    v = (raw if raw is not None else "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off", "disabled")


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


def _parse_resolve_play_hold(
    *,
    hours_env: str,
    ms_env: str,
    default_hours: float,
) -> tuple[int, int]:
    """按 play 族解析 resolve 持仓上限；hold_ms=0 表示回退全局 ZCT_RESOLVE_MAX_HOLD_MS。"""
    ms_raw = os.getenv(ms_env, "").strip()
    if ms_raw:
        try:
            hold_ms = max(0, int(float(ms_raw)))
        except ValueError:
            hold_ms = max(0, int(default_hours * 3_600_000))
    else:
        try:
            h = float(os.getenv(hours_env, str(default_hours)).strip() or default_hours)
        except ValueError:
            h = default_hours
        hold_ms = 0 if h <= 0 else int(h * 3_600_000)
    if hold_ms > 0:
        bars = max(1, int(round(hold_ms / 60_000.0)))
    else:
        bars = 0
    return hold_ms, bars


@dataclass
class StrategyConfig:
    """策略闸门与风控参数；`btc_macro_state` 为扫描轮次内可变缓存。"""

    band_sigma: float = 1.0
    vwap_slope_bars: int = 20
    slope_steep_bps: float = 2.5
    slope_flat_bps: float = 0.8
    wide_band_mult: float = 1.15
    tight_band_mult: float = 0.88
    choppy_cross_min: int = 7
    ma_period: int = 30
    ma_choppy_cross_min: int = 10
    ma_lookback: int = 120
    band_touch_frac: float = 0.35
    db_skip_flat: bool = False

    btc_macro_filter_enabled: bool = True
    btc_macro_slope_threshold_bps: float = 3.0
    btc_macro_rs_min_ratio: float = 0.5
    btc_macro_long_fuse_slope_bps: float = 8.0
    btc_macro_state: Dict[str, Any] = field(
        default_factory=lambda: {"slope_bps": 0.0, "chop": "high"}
    )

    strict_pa_filters: bool = True
    vol_ma_period: int = 10
    spike_lookback: int = 5
    spike_range_ratio: float = 0.004
    spike_use_atr_15m: bool = True
    spike_atr_interval: str = "15m"
    spike_atr_period: int = 14
    spike_atr_mult: float = 1.25
    spike_atr_ratio_floor: float = 0.0009
    spike_atr_ratio_cap: float = 0.025
    spike_atr_kline_limit: int = 64
    grind_lookback: int = 6
    grind_max_net_move_pct: float = 0.0035
    level_touch_lookback_bars: int = 480
    level_fresh_min_bars: int = 360
    level_recycle_touch_min: int = 3
    level_fresh_min_hours: float = 0.0
    play03_tp_1r: bool = False
    koroush_min_stop_distance_pct: float = 0.01
    psych_levels_enabled: bool = False
    breakout_max_ma_crosses: int = 0
    recycled_near_veto_enabled: bool = False
    recycled_near_max_dist_pct: float = 0.2
    vwap_cross_max_low: int = 3
    vwap_cross_max_mid: int = 6
    enforce_setup_level: bool = True
    min_setup_level_for_side: int = 3

    account_equity_usdt: float = 10000.0
    risk_pct_per_trade: float = 0.005
    use_risk_sized_notional: bool = False
    max_notional_cap_usdt: float = 0.0
    max_daily_loss_pct: float = 0.05
    max_open_positions: int = 8
    max_open_play01: int = 5
    max_open_play02: int = 5

    cooldown_after_loss_ms: int = 30 * 60 * 1000
    cooldown_after_win_ms: int = 30 * 60 * 1000
    cooldown_after_close_ms: int = 0
    use_db_cooldown: bool = True

    max_band_width_pct: float = 15.0
    swing_lookback: int = 20
    min_sl_pct: float = 0.003
    sl_buffer_bps: float = 2.0
    max_sl_widen_pct: float = 0.05

    resolve_max_bars: int = 240
    resolve_max_hold_ms: int = 4 * 60 * 60 * 1000
    resolve_max_hold_ms_play01: int = 5 * 60 * 60 * 1000
    resolve_max_bars_play01: int = 300
    resolve_max_hold_ms_play02: int = 4 * 60 * 60 * 1000
    resolve_max_bars_play02: int = 240
    resolve_max_hold_ms_play03: int = 3 * 60 * 60 * 1000
    resolve_max_bars_play03: int = 180

    liquidity_oi_filter_enabled: bool = False
    zct_margin_usdt: float = 100.0
    zct_leverage: float = 10.0

    @property
    def virtual_notional_usdt(self) -> float:
        return self.zct_margin_usdt * self.zct_leverage

    @classmethod
    def from_env(cls) -> "StrategyConfig":
        play03_raw = os.getenv("ZCT_PLAY03_TP_MODE", "vwap").strip().lower()
        kms_raw = os.getenv("ZCT_KOROUSH_MIN_STOP_DISTANCE_PCT", "0.01").strip()
        try:
            koroush = 0.01 if kms_raw == "" else float(kms_raw)
        except ValueError:
            koroush = 0.01

        _default_resolve_bars = 4 * 60
        _default_resolve_hold_ms = 4 * 60 * 60 * 1000

        resolve_bars_raw = os.getenv("ZCT_RESOLVE_MAX_BARS")
        try:
            if resolve_bars_raw is None or str(resolve_bars_raw).strip() == "":
                resolve_max_bars = _default_resolve_bars
            else:
                resolve_max_bars = int(float(str(resolve_bars_raw).strip()))
        except ValueError:
            resolve_max_bars = _default_resolve_bars

        resolve_hold_raw = os.getenv("ZCT_RESOLVE_MAX_HOLD_MS")
        try:
            if resolve_hold_raw is None or str(resolve_hold_raw).strip() == "":
                resolve_max_hold_ms = _default_resolve_hold_ms
            else:
                resolve_max_hold_ms = max(0, int(float(str(resolve_hold_raw).strip())))
        except ValueError:
            resolve_max_hold_ms = _default_resolve_hold_ms

        resolve_max_hold_ms_play01, resolve_max_bars_play01 = _parse_resolve_play_hold(
            hours_env="ZCT_RESOLVE_MAX_HOLD_HOURS_PLAY01",
            ms_env="ZCT_RESOLVE_MAX_HOLD_MS_PLAY01",
            default_hours=5.0,
        )
        resolve_max_hold_ms_play02, resolve_max_bars_play02 = _parse_resolve_play_hold(
            hours_env="ZCT_RESOLVE_MAX_HOLD_HOURS_PLAY02",
            ms_env="ZCT_RESOLVE_MAX_HOLD_MS_PLAY02",
            default_hours=4.0,
        )
        resolve_max_hold_ms_play03, resolve_max_bars_play03 = _parse_resolve_play_hold(
            hours_env="ZCT_RESOLVE_MAX_HOLD_HOURS_PLAY03",
            ms_env="ZCT_RESOLVE_MAX_HOLD_MS_PLAY03",
            default_hours=3.0,
        )

        try:
            max_open = max(0, int(os.getenv("ZCT_MAX_OPEN_POSITIONS", "8").strip() or "8"))
        except ValueError:
            max_open = 8
        try:
            max_p1 = max(0, int(os.getenv("ZCT_MAX_OPEN_PLAY01", "5").strip() or "5"))
        except ValueError:
            max_p1 = 5
        try:
            max_p2 = max(0, int(os.getenv("ZCT_MAX_OPEN_PLAY02", "5").strip() or "5"))
        except ValueError:
            max_p2 = 5

        try:
            recycled_dist = float(os.getenv("ZCT_RECYCLED_NEAR_MAX_DIST_PCT", "0.2").strip() or "0.2")
        except ValueError:
            recycled_dist = 0.2
        if recycled_dist <= 0:
            recycled_dist = 0.2

        try:
            level_fresh_h = float(os.getenv("ZCT_LEVEL_FRESH_MIN_HOURS", "0").strip() or "0")
        except ValueError:
            level_fresh_h = 0.0

        bma_raw = os.getenv("ZCT_BREAKOUT_MAX_MA_CROSSES", "0").strip()
        try:
            breakout_ma = int(bma_raw) if bma_raw else 0
        except ValueError:
            breakout_ma = 0

        return cls(
            band_sigma=_float_env("ZCT_VWAP_BAND_SIGMA", 1.0),
            vwap_slope_bars=_int_env("ZCT_VWAP_SLOPE_BARS", 20),
            slope_steep_bps=_float_env("ZCT_SLOPE_STEEP_BPS", 2.5),
            slope_flat_bps=_float_env("ZCT_SLOPE_FLAT_BPS", 0.8),
            wide_band_mult=_float_env("ZCT_WIDE_BAND_MULT", 1.15),
            tight_band_mult=_float_env("ZCT_TIGHT_BAND_MULT", 0.88),
            choppy_cross_min=_int_env("ZCT_CHOPPY_CROSS_MIN", 7),
            ma_period=_int_env("ZCT_MA_PERIOD", 30),
            ma_choppy_cross_min=_int_env("ZCT_MA_CHOPPY_CROSS_MIN", 10),
            ma_lookback=_int_env("ZCT_MA_LOOKBACK", 120),
            band_touch_frac=_float_env("ZCT_BAND_TOUCH_FRAC", 0.35),
            db_skip_flat=_truthy(os.getenv("ZCT_VWAP_DB_SKIP_FLAT", "")),
            btc_macro_filter_enabled=_truthy(
                os.getenv("ZCT_BTC_MACRO_FILTER_ENABLED", "1"), default=True
            ),
            btc_macro_slope_threshold_bps=_float_env(
                "ZCT_BTC_MACRO_SLOPE_THRESHOLD_BPS", 3.0
            ),
            btc_macro_rs_min_ratio=max(
                0.0, _float_env("ZCT_BTC_MACRO_RS_MIN_RATIO", 0.5)
            ),
            btc_macro_long_fuse_slope_bps=_float_env(
                "ZCT_BTC_MACRO_LONG_FUSE_SLOPE_BPS", 8.0
            ),
            strict_pa_filters=_truthy(os.getenv("ZCT_STRICT_PA_FILTERS", ""), default=True),
            vol_ma_period=_int_env("ZCT_VOL_MA_PERIOD", 10),
            spike_lookback=_int_env("ZCT_SPIKE_LOOKBACK", 5),
            spike_range_ratio=_float_env("ZCT_SPIKE_RANGE_RATIO", 0.004),
            spike_use_atr_15m=_truthy(os.getenv("ZCT_SPIKE_USE_ATR_15M", "1"), default=True),
            spike_atr_interval=(os.getenv("ZCT_SPIKE_ATR_INTERVAL", "15m").strip() or "15m"),
            spike_atr_period=_int_env("ZCT_SPIKE_ATR_PERIOD", 14),
            spike_atr_mult=_float_env("ZCT_SPIKE_ATR_MULT", 1.25),
            spike_atr_ratio_floor=_float_env("ZCT_SPIKE_ATR_RATIO_FLOOR", 0.0009),
            spike_atr_ratio_cap=_float_env("ZCT_SPIKE_ATR_RATIO_CAP", 0.025),
            spike_atr_kline_limit=_int_env("ZCT_SPIKE_ATR_KLINE_LIMIT", 64),
            grind_lookback=_int_env("ZCT_GRIND_LOOKBACK", 6),
            grind_max_net_move_pct=_float_env("ZCT_GRIND_MAX_NET_MOVE_PCT", 0.0035),
            level_touch_lookback_bars=_int_env("ZCT_LEVEL_TOUCH_LOOKBACK_BARS", 480),
            level_fresh_min_bars=_int_env("ZCT_LEVEL_FRESH_MIN_BARS", 360),
            level_recycle_touch_min=_int_env("ZCT_LEVEL_RECYCLE_TOUCH_MIN", 3),
            level_fresh_min_hours=level_fresh_h,
            play03_tp_1r=play03_raw in ("1r", "one_r", "risk1"),
            koroush_min_stop_distance_pct=koroush,
            psych_levels_enabled=_truthy(os.getenv("ZCT_PSYCH_LEVELS", "")),
            breakout_max_ma_crosses=breakout_ma,
            recycled_near_veto_enabled=_truthy(
                os.getenv("ZCT_RECYCLED_NEAR_VETO_ENABLED", ""), default=False
            ),
            recycled_near_max_dist_pct=recycled_dist,
            vwap_cross_max_low=_int_env("ZCT_VWAP_CROSS_MAX_LOW", 3),
            vwap_cross_max_mid=_int_env("ZCT_VWAP_CROSS_MAX_MID", 6),
            enforce_setup_level=_truthy(os.getenv("ZCT_ENFORCE_SETUP_LEVEL", "1"), default=True),
            min_setup_level_for_side=_int_env("ZCT_MIN_SETUP_LEVEL", 3),
            account_equity_usdt=_float_env("ZCT_ACCOUNT_EQUITY_USDT", 10000.0),
            risk_pct_per_trade=_float_env("ZCT_RISK_PCT_PER_TRADE", 0.005),
            use_risk_sized_notional=_truthy(os.getenv("ZCT_USE_RISK_SIZED_NOTIONAL", "")),
            max_notional_cap_usdt=_float_env("ZCT_MAX_NOTIONAL_CAP_USDT", 0.0),
            max_daily_loss_pct=_float_env("ZCT_MAX_DAILY_LOSS_PCT", 0.05),
            max_open_positions=max_open,
            max_open_play01=max_p1,
            max_open_play02=max_p2,
            max_band_width_pct=_float_env("ZCT_MAX_BAND_WIDTH_PCT", 15.0),
            swing_lookback=_int_env("ZCT_SWING_LOOKBACK", 20),
            min_sl_pct=_float_env("ZCT_MIN_SL_PCT", 0.003),
            sl_buffer_bps=_float_env("ZCT_SL_BUFFER_BPS", 2.0),
            max_sl_widen_pct=_float_env("ZCT_MAX_SL_WIDEN_PCT", 0.05),
            resolve_max_bars=resolve_max_bars,
            resolve_max_hold_ms=resolve_max_hold_ms,
            resolve_max_hold_ms_play01=resolve_max_hold_ms_play01,
            resolve_max_bars_play01=resolve_max_bars_play01,
            resolve_max_hold_ms_play02=resolve_max_hold_ms_play02,
            resolve_max_bars_play02=resolve_max_bars_play02,
            resolve_max_hold_ms_play03=resolve_max_hold_ms_play03,
            resolve_max_bars_play03=resolve_max_bars_play03,
            liquidity_oi_filter_enabled=False,
            zct_margin_usdt=_float_env("ZCT_VIRTUAL_NOTIONAL_USDT", 100.0),
            zct_leverage=_float_env("ZCT_LEVERAGE", 10.0),
            use_db_cooldown=True,
        )

    @classmethod
    def for_backtest(
        cls,
        *,
        use_db_cooldown: bool = False,
        btc_macro_filter_enabled: bool = False,
    ) -> "StrategyConfig":
        base = cls.from_env()
        return replace(
            base,
            use_db_cooldown=use_db_cooldown,
            btc_macro_filter_enabled=btc_macro_filter_enabled,
            btc_macro_state={"slope_bps": 0.0, "chop": "high"},
        )

    def reset_btc_macro_state(self) -> None:
        self.btc_macro_state = {"slope_bps": 0.0, "chop": "high"}

    def copy_for_scan(self) -> "StrategyConfig":
        """每轮扫描独立副本，避免 DEFAULT 单例上 btc_macro_state 被并发改写。"""
        return replace(self, btc_macro_state={"slope_bps": 0.0, "chop": "high"})

    @classmethod
    def for_live_scan(cls, base: Optional["StrategyConfig"] = None) -> "StrategyConfig":
        return (base or cls.from_env()).copy_for_scan()

    def cooldown_blocks(self, symbol: str, repo: Any = None) -> bool:
        if not self.use_db_cooldown:
            return False
        if (
            self.cooldown_after_loss_ms <= 0
            and self.cooldown_after_win_ms <= 0
            and self.cooldown_after_close_ms <= 0
        ):
            return False
        from zct_db_repositories import CooldownRepository

        r = repo if repo is not None else CooldownRepository()
        return r.is_symbol_in_cooldown(symbol)

    def cooldown_blocks_batch(
        self, symbols: list[str], repo: Any = None
    ) -> set[str]:
        """返回当前处于冷却中的 symbol 集合（单次 DB 往返）。"""
        if not self.use_db_cooldown:
            return set()
        from zct_db_repositories import CooldownRepository

        r = repo if repo is not None else CooldownRepository()
        return r.symbols_in_cooldown(symbols)


def export_strategy_module_aliases(g: Dict[str, Any], cfg: StrategyConfig) -> None:
    """将配置同步到模块级名称，兼容旧代码与 scan_params JSON。"""
    g["BAND_SIGMA"] = cfg.band_sigma
    g["VWAP_SLOPE_BARS"] = cfg.vwap_slope_bars
    g["SLOPE_STEEP_BPS"] = cfg.slope_steep_bps
    g["SLOPE_FLAT_BPS"] = cfg.slope_flat_bps
    g["WIDE_BAND_MULT"] = cfg.wide_band_mult
    g["TIGHT_BAND_MULT"] = cfg.tight_band_mult
    g["CHOPPY_CROSS_MIN"] = cfg.choppy_cross_min
    g["MA_PERIOD"] = cfg.ma_period
    g["MA_CHOPPY_CROSS_MIN"] = cfg.ma_choppy_cross_min
    g["MA_LOOKBACK"] = cfg.ma_lookback
    g["BAND_TOUCH_FRAC"] = cfg.band_touch_frac
    g["DB_SKIP_FLAT"] = cfg.db_skip_flat
    g["BTC_MACRO_FILTER_ENABLED"] = cfg.btc_macro_filter_enabled
    g["BTC_MACRO_SLOPE_THRESHOLD_BPS"] = cfg.btc_macro_slope_threshold_bps
    g["BTC_MACRO_RS_MIN_RATIO"] = cfg.btc_macro_rs_min_ratio
    g["BTC_MACRO_LONG_FUSE_SLOPE_BPS"] = cfg.btc_macro_long_fuse_slope_bps
    g["_BTC_MACRO_STATE"] = cfg.btc_macro_state
    g["STRICT_PA_FILTERS"] = cfg.strict_pa_filters
    g["VOL_MA_PERIOD"] = cfg.vol_ma_period
    g["SPIKE_LOOKBACK"] = cfg.spike_lookback
    g["SPIKE_RANGE_RATIO"] = cfg.spike_range_ratio
    g["SPIKE_USE_ATR_15M"] = cfg.spike_use_atr_15m
    g["SPIKE_ATR_INTERVAL"] = cfg.spike_atr_interval
    g["SPIKE_ATR_PERIOD"] = cfg.spike_atr_period
    g["SPIKE_ATR_MULT"] = cfg.spike_atr_mult
    g["SPIKE_ATR_RATIO_FLOOR"] = cfg.spike_atr_ratio_floor
    g["SPIKE_ATR_RATIO_CAP"] = cfg.spike_atr_ratio_cap
    g["SPIKE_ATR_KLINE_LIMIT"] = cfg.spike_atr_kline_limit
    g["GRIND_LOOKBACK"] = cfg.grind_lookback
    g["GRIND_MAX_NET_MOVE_PCT"] = cfg.grind_max_net_move_pct
    g["LEVEL_TOUCH_LOOKBACK_BARS"] = cfg.level_touch_lookback_bars
    g["LEVEL_FRESH_MIN_BARS"] = cfg.level_fresh_min_bars
    g["LEVEL_RECYCLE_TOUCH_MIN"] = cfg.level_recycle_touch_min
    g["LEVEL_FRESH_MIN_HOURS"] = cfg.level_fresh_min_hours
    g["PLAY03_TP_1R"] = cfg.play03_tp_1r
    g["KOROUSH_MIN_STOP_DISTANCE_PCT"] = cfg.koroush_min_stop_distance_pct
    g["PSYCH_LEVELS_ENABLED"] = cfg.psych_levels_enabled
    g["BREAKOUT_MAX_MA_CROSSES"] = cfg.breakout_max_ma_crosses
    g["RECYCLED_NEAR_VETO_ENABLED"] = cfg.recycled_near_veto_enabled
    g["RECYCLED_NEAR_MAX_DIST_PCT"] = cfg.recycled_near_max_dist_pct
    g["VWAP_CROSS_MAX_LOW"] = cfg.vwap_cross_max_low
    g["VWAP_CROSS_MAX_MID"] = cfg.vwap_cross_max_mid
    g["ENFORCE_SETUP_LEVEL"] = cfg.enforce_setup_level
    g["MIN_SETUP_LEVEL_FOR_SIDE"] = cfg.min_setup_level_for_side
    g["ACCOUNT_EQUITY_USDT"] = cfg.account_equity_usdt
    g["RISK_PCT_PER_TRADE"] = cfg.risk_pct_per_trade
    g["USE_RISK_SIZED_NOTIONAL"] = cfg.use_risk_sized_notional
    g["MAX_NOTIONAL_CAP_USDT"] = cfg.max_notional_cap_usdt
    g["MAX_DAILY_LOSS_PCT"] = cfg.max_daily_loss_pct
    g["MAX_OPEN_POSITIONS"] = cfg.max_open_positions
    g["MAX_OPEN_PLAY01"] = cfg.max_open_play01
    g["MAX_OPEN_PLAY02"] = cfg.max_open_play02
    g["COOLDOWN_AFTER_LOSS_MS"] = cfg.cooldown_after_loss_ms
    g["COOLDOWN_AFTER_WIN_MS"] = cfg.cooldown_after_win_ms
    g["COOLDOWN_AFTER_CLOSE_MS"] = cfg.cooldown_after_close_ms
    g["MAX_BAND_WIDTH_PCT"] = cfg.max_band_width_pct
    g["SWING_LOOKBACK"] = cfg.swing_lookback
    g["MIN_SL_PCT"] = cfg.min_sl_pct
    g["SL_BUFFER_BPS"] = cfg.sl_buffer_bps
    g["MAX_SL_WIDEN_PCT"] = cfg.max_sl_widen_pct
    g["RESOLVE_MAX_BARS"] = cfg.resolve_max_bars
    g["RESOLVE_MAX_HOLD_MS"] = cfg.resolve_max_hold_ms
    g["RESOLVE_MAX_HOLD_MS_PLAY01"] = cfg.resolve_max_hold_ms_play01
    g["RESOLVE_MAX_BARS_PLAY01"] = cfg.resolve_max_bars_play01
    g["RESOLVE_MAX_HOLD_MS_PLAY02"] = cfg.resolve_max_hold_ms_play02
    g["RESOLVE_MAX_BARS_PLAY02"] = cfg.resolve_max_bars_play02
    g["RESOLVE_MAX_HOLD_MS_PLAY03"] = cfg.resolve_max_hold_ms_play03
    g["RESOLVE_MAX_BARS_PLAY03"] = cfg.resolve_max_bars_play03
    g["_DEFAULT_RESOLVE_HOLD_HOURS"] = (
        cfg.resolve_max_hold_ms / 3_600_000.0 if cfg.resolve_max_hold_ms > 0 else 4.0
    )
    g["LIQUIDITY_OI_FILTER_ENABLED"] = cfg.liquidity_oi_filter_enabled
    g["_ZCT_MARGIN_USDT"] = cfg.zct_margin_usdt
    g["ZCT_LEVERAGE"] = cfg.zct_leverage
    g["VIRTUAL_NOTIONAL_USDT"] = cfg.virtual_notional_usdt

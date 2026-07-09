"""trading-orb 同时刻相对成交量基准（5m 历史，行情源见 MARKET_DATA_EXCHANGE）。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict

import pandas as pd

from quant.market import fetch_klines_forward, klines_to_df
from quant.common.config import OrbConfig
from quant.common.exchange_env import resolve_market_data_exchange_id
from quant.common.kline_cache import norm_symbol
from quant.common.session_paper import in_regular_session
from quant.common.session import session_day_str
from quant.common.resample import resample_ohlcv

logger = logging.getLogger(__name__)

_BASELINE_CACHE: Dict[str, Dict[str, float]] = {}


def clear_baseline_cache() -> None:
    _BASELINE_CACHE.clear()


def load_volume_baselines(
    symbol: str,
    *,
    cfg: OrbConfig,
    lookback_days: int = 20,
    use_cache: bool = True,
    market_data_exchange: str | None = None,
) -> Dict[str, float]:
    """过去 N 个交易日各 5m 时刻的平均成交量（不含今日）。"""
    sym = norm_symbol(symbol)
    if use_cache and sym in _BASELINE_CACHE:
        return dict(_BASELINE_CACHE[sym])
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - int(lookback_days + 10) * 86_400_000
    md_exchange = resolve_market_data_exchange_id(market_data_exchange)
    rows = fetch_klines_forward(sym, "1m", start_ms, end_ms, exchange_id=md_exchange)
    df_1m = klines_to_df(rows)
    if df_1m.empty:
        logger.warning("[orb-vnpy] no 1m history for vol baseline %s", sym)
        return {}
    df_5m = resample_ohlcv(df_1m, "5m")
    if df_5m.empty:
        return {}
    today = session_day_str(end_ms, tz=cfg.session_tz, session_open_time=cfg.session_open_time)
    buckets: Dict[str, list[float]] = {}
    for _, row in df_5m.iterrows():
        ms = int(row["open_time"])
        if not in_regular_session(cfg, now_ms=ms):
            continue
        day = session_day_str(ms, tz=cfg.session_tz, session_open_time=cfg.session_open_time)
        if day >= today:
            continue
        key = _time_key_from_ms(ms, cfg.session_tz)
        buckets.setdefault(key, []).append(float(row["volume"]))
    baselines = {k: sum(v) / len(v) for k, v in buckets.items() if v}
    _BASELINE_CACHE[sym] = baselines
    return baselines


def _time_key_from_ms(open_ms: int, tz: str) -> str:
    ts = pd.Timestamp(int(open_ms), unit="ms", tz=tz)
    return ts.strftime("%H:%M")


def preload_pool_baselines(
    symbols: list[str],
    *,
    cfg: OrbConfig,
    lookback_days: int = 20,
    pause_sec: float = 2.5,
    market_data_exchange: str | None = None,
) -> Dict[str, Dict[str, float]]:
    """启动前串行预载 vol 基准，避免 7 策略并行打爆 fapi 权重。"""
    import time

    out: Dict[str, Dict[str, float]] = {}
    for raw in symbols:
        sym = norm_symbol(raw)
        logger.info("[orb-vnpy] vol baseline begin %s", sym)
        t0 = time.time()
        out[sym] = load_volume_baselines(
            sym,
            cfg=cfg,
            lookback_days=lookback_days,
            use_cache=False,
            market_data_exchange=market_data_exchange,
        )
        logger.info(
            "[orb-vnpy] vol baseline done %s keys=%d elapsed=%.1fs",
            sym,
            len(out[sym]),
            time.time() - t0,
        )
        if pause_sec > 0:
            time.sleep(float(pause_sec))
    return out

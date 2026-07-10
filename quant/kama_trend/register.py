"""KAMA Trend — vnpy 策略注册。"""

from __future__ import annotations

from typing import List

from quant.common.exchange_env import resolve_live_exchange_id, resolve_market_data_exchange_id
from quant.common.kline_cache import norm_symbol
from quant.common.register_sizing import recent_atr
from quant.common.vnpy_wallet import lane_equity_usdt, migrate_vnpy_lane_tables
from quant.engine.exchanges.registry import vnpy_vt_symbol
from quant.engine.registry import VnpyLanePlugin
from quant.kama_trend.config import KamaTrendConfig
from quant.kama_trend.kama_vnpy import KamaTrendVnpyStrategy
from quant.kama_trend.sizing import size_for_kama
from quant.kama_trend.switches import KAMA_TREND_SWITCH
from quant.market import fetch_mark_price


def register_vnpy_strategies(cta_engine, cfg: KamaTrendConfig, wallet_cur) -> List[str]:
    if wallet_cur is not None:
        migrate_vnpy_lane_tables(wallet_cur)
    live_ex = resolve_live_exchange_id(cfg.live_exchange)
    md_exchange = resolve_market_data_exchange_id(cfg.market_data_exchange)
    names: List[str] = []
    settings = KamaTrendVnpyStrategy.from_kama_config(cfg)
    interval = f"{max(1, int(cfg.signal_minutes))}m"
    for sym in cfg.symbol_list():
        sym = norm_symbol(sym)
        px = fetch_mark_price(sym, exchange_id=md_exchange) or 100.0
        eq = lane_equity_usdt(cfg, sym, cur=wallet_cur) if wallet_cur is not None else float(cfg.equity_usdt)
        atr = recent_atr(sym, interval, exchange_id=md_exchange, atr_period=int(cfg.adx_period))
        stop_dist = float(cfg.stop_atr) * (atr if atr and atr > 0 else px * 0.015)
        vol = size_for_kama(cfg, px, stop_distance=stop_dist, equity_usdt=eq)
        name = f"kama_{sym.lower()}"
        cta_engine.add_strategy(
            class_name="KamaTrendVnpyStrategy",
            strategy_name=name,
            vt_symbol=vnpy_vt_symbol(sym, exchange_id=live_ex),
            setting={**settings, "fixed_size": max(vol, 0.001)},
        )
        names.append(name)
    return names


KAMA_TREND_VNPY_PLUGIN = VnpyLanePlugin(
    name="kama_trend",
    load_config=KamaTrendConfig.from_env,
    strategy_class=KamaTrendVnpyStrategy,
    class_name="KamaTrendVnpyStrategy",
    sync_prefix="kama",
    register=register_vnpy_strategies,
    switch=KAMA_TREND_SWITCH,
    uses_kline_stream=True,
)

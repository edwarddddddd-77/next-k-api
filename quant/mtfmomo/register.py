"""MtfMomo2xA — vnpy 策略注册。"""

from __future__ import annotations

from typing import List

from quant.common.exchange_env import resolve_live_exchange_id, resolve_market_data_exchange_id
from quant.common.kline_cache import norm_symbol
from quant.common.register_sizing import recent_atr
from quant.common.vnpy_wallet import lane_equity_usdt, migrate_vnpy_lane_tables
from quant.engine.exchanges.registry import vnpy_vt_symbol
from quant.engine.registry import VnpyLanePlugin
from quant.market import fetch_mark_price
from quant.mtfmomo.config import MtfMomoConfig
from quant.mtfmomo.mtfmomo_vnpy import MtfMomoVnpyStrategy
from quant.mtfmomo.sizing import size_for_momo
from quant.mtfmomo.switches import MTFMOMO_SWITCH


def register_vnpy_strategies(cta_engine, cfg: MtfMomoConfig, wallet_cur) -> List[str]:
    if wallet_cur is not None:
        migrate_vnpy_lane_tables(wallet_cur)
    live_ex = resolve_live_exchange_id(cfg.live_exchange)
    md_exchange = resolve_market_data_exchange_id(cfg.market_data_exchange)
    names: List[str] = []
    settings = MtfMomoVnpyStrategy.from_mtfmomo_config(cfg)
    for sym in cfg.symbol_list():
        sym = norm_symbol(sym)
        px = fetch_mark_price(sym, exchange_id=md_exchange) or 100.0
        eq = lane_equity_usdt(cfg, sym, cur=wallet_cur) if wallet_cur is not None else float(cfg.equity_usdt)
        atr = recent_atr(sym, "1h", exchange_id=md_exchange)
        stop_dist = float(cfg.stop_atr) * (atr if atr and atr > 0 else px * 0.02)
        vol = size_for_momo(cfg, px, stop_distance=stop_dist, equity_usdt=eq)
        name = f"momo_{sym.lower()}"
        cta_engine.add_strategy(
            class_name="MtfMomoVnpyStrategy",
            strategy_name=name,
            vt_symbol=vnpy_vt_symbol(sym, exchange_id=live_ex),
            setting={**settings, "fixed_size": max(vol, 0.001)},
        )
        names.append(name)
    return names


MTFMOMO_VNPY_PLUGIN = VnpyLanePlugin(
    name="mtfmomo",
    load_config=MtfMomoConfig.from_env,
    strategy_class=MtfMomoVnpyStrategy,
    class_name="MtfMomoVnpyStrategy",
    sync_prefix="momo",
    register=register_vnpy_strategies,
    switch=MTFMOMO_SWITCH,
    uses_kline_stream=True,
)

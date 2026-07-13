"""IB50 — vnpy 策略注册。"""

from __future__ import annotations

from typing import List

from quant.common.exchange_env import resolve_live_exchange_id, resolve_market_data_exchange_id
from quant.common.kline_cache import norm_symbol
from quant.common.vnpy_wallet import lane_equity_usdt, migrate_vnpy_lane_tables
from quant.engine.exchanges.registry import vnpy_vt_symbol
from quant.engine.registry import VnpyLanePlugin
from quant.ib50.config import Ib50Config
from quant.ib50.ib50_vnpy import Ib50VnpyStrategy
from quant.ib50.sizing import fixed_size_for_ib50
from quant.ib50.switches import IB50_SWITCH
from quant.market import fetch_mark_price


def register_vnpy_strategies(cta_engine, cfg: Ib50Config, wallet_cur) -> List[str]:
    if wallet_cur is not None:
        migrate_vnpy_lane_tables(wallet_cur)

    live_ex = resolve_live_exchange_id(cfg.live_exchange)
    md_exchange = resolve_market_data_exchange_id(cfg.market_data_exchange)
    names: List[str] = []
    settings = Ib50VnpyStrategy.from_ib50_config(cfg)
    for sym in cfg.symbol_list():
        sym = norm_symbol(sym)
        px = fetch_mark_price(sym, exchange_id=md_exchange) or 100.0
        eq = lane_equity_usdt(cfg, sym, cur=wallet_cur) if wallet_cur is not None else float(cfg.equity_usdt)
        # 假定 IB 半宽 ≈ 0.5% 用于预注册定仓；实盘按真实 IB 止损距离重算
        stop_dist = px * 0.005
        vol = fixed_size_for_ib50(cfg, sym, px, stop_distance=stop_dist, equity_usdt=eq)
        name = f"ib50_{sym.lower()}"
        cta_engine.add_strategy(
            class_name="Ib50VnpyStrategy",
            strategy_name=name,
            vt_symbol=vnpy_vt_symbol(sym, exchange_id=live_ex),
            setting={**settings, "fixed_size": max(vol, 0.001)},
        )
        names.append(name)
    return names


IB50_VNPY_PLUGIN = VnpyLanePlugin(
    name="ib50",
    load_config=Ib50Config.from_env,
    strategy_class=Ib50VnpyStrategy,
    class_name="Ib50VnpyStrategy",
    sync_prefix="ib50",
    register=register_vnpy_strategies,
    switch=IB50_SWITCH,
    uses_kline_stream=True,
)

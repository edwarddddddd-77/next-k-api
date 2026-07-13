"""IBS lane vnpy 插件工厂。"""

from __future__ import annotations

from typing import Callable, List

from quant.common.exchange_env import resolve_live_exchange_id, resolve_market_data_exchange_id
from quant.common.kline_cache import norm_symbol
from quant.common.strategy_switch import StrategySwitchSpec
from quant.common.vnpy_wallet import lane_equity_usdt, migrate_vnpy_lane_tables
from quant.engine.exchanges.registry import vnpy_vt_symbol
from quant.engine.registry import VnpyLanePlugin
from quant.ibs.config import IbsLaneConfig
from quant.ibs.ibs_vnpy import IbsVnpyStrategy
from quant.ibs.sizing import size_for_ibs
from quant.market import fetch_mark_price


def register_ibs_vnpy_strategies(
    cta_engine,
    cfg: IbsLaneConfig,
    wallet_cur,
    *,
    name_prefix: str,
    lane_config_for_symbol: Callable[[IbsLaneConfig, str], IbsLaneConfig] | None = None,
) -> List[str]:
    if wallet_cur is not None:
        migrate_vnpy_lane_tables(wallet_cur)
    live_ex = resolve_live_exchange_id(cfg.live_exchange)
    md_exchange = resolve_market_data_exchange_id(cfg.market_data_exchange)
    names: List[str] = []
    for sym in cfg.symbol_list():
        sym = norm_symbol(sym)
        sym_cfg = (
            lane_config_for_symbol(cfg, sym)
            if lane_config_for_symbol is not None
            else cfg
        )
        settings = IbsVnpyStrategy.from_ibs_config(sym_cfg)
        px = fetch_mark_price(sym, exchange_id=md_exchange) or 100.0
        eq = lane_equity_usdt(sym_cfg, sym, cur=wallet_cur) if wallet_cur is not None else float(sym_cfg.equity_usdt)
        vol = size_for_ibs(sym_cfg, px, equity_usdt=eq)
        name = f"{name_prefix}_{sym.lower()}"
        cta_engine.add_strategy(
            class_name="IbsVnpyStrategy",
            strategy_name=name,
            vt_symbol=vnpy_vt_symbol(sym, exchange_id=live_ex),
            setting={**settings, "fixed_size": max(vol, 0.001)},
        )
        names.append(name)
    return names


def build_ibs_plugin(
    *,
    lane: str,
    profile: str,
    switch: StrategySwitchSpec,
    load_config: Callable[[], IbsLaneConfig],
    sync_prefix: str,
    name_prefix: str,
    register: Callable[..., List[str]] | None = None,
) -> VnpyLanePlugin:
    def _register_default(cta_engine, cfg: IbsLaneConfig, wallet_cur) -> List[str]:
        return register_ibs_vnpy_strategies(
            cta_engine,
            cfg,
            wallet_cur,
            name_prefix=name_prefix,
        )

    return VnpyLanePlugin(
        name=lane,
        load_config=load_config,
        strategy_class=IbsVnpyStrategy,
        class_name="IbsVnpyStrategy",
        sync_prefix=sync_prefix,
        register=register or _register_default,
        switch=switch,
        uses_kline_stream=True,
    )

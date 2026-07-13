"""Anchor Drift — vnpy 策略注册。"""

from __future__ import annotations

from typing import List

from quant.anchor_drift.anchor_drift_vnpy import AnchorDriftVnpyStrategy
from quant.anchor_drift.config import AnchorDriftConfig
from quant.anchor_drift.sizing import size_for_drift
from quant.anchor_drift.switches import ANCHOR_DRIFT_SWITCH
from quant.common.exchange_env import resolve_live_exchange_id, resolve_market_data_exchange_id
from quant.common.kline_cache import norm_symbol
from quant.common.vnpy_wallet import lane_equity_usdt, migrate_vnpy_lane_tables
from quant.engine.exchanges.registry import vnpy_vt_symbol
from quant.engine.registry import VnpyLanePlugin
from quant.market import fetch_mark_price


def register_vnpy_strategies(cta_engine, cfg: AnchorDriftConfig, wallet_cur) -> List[str]:
    if wallet_cur is not None:
        migrate_vnpy_lane_tables(wallet_cur)
        from quant.anchor_drift.db import migrate_anchor_drift_tables

        migrate_anchor_drift_tables(wallet_cur)

    live_ex = resolve_live_exchange_id(cfg.live_exchange)
    md_exchange = resolve_market_data_exchange_id(cfg.market_data_exchange)
    names: List[str] = []
    settings = AnchorDriftVnpyStrategy.from_drift_config(cfg)
    for sym in cfg.symbol_list():
        sym = norm_symbol(sym)
        px = fetch_mark_price(sym, exchange_id=md_exchange) or 100.0
        eq = lane_equity_usdt(cfg, sym, cur=wallet_cur) if wallet_cur is not None else float(cfg.equity_usdt)
        vol = size_for_drift(cfg, px, anchor_price=px, equity_usdt=eq)
        name = f"drift_{sym.lower()}"
        cta_engine.add_strategy(
            class_name="AnchorDriftVnpyStrategy",
            strategy_name=name,
            vt_symbol=vnpy_vt_symbol(sym, exchange_id=live_ex),
            setting={**settings, "fixed_size": max(vol, 0.001)},
        )
        names.append(name)
    return names


ANCHOR_DRIFT_VNPY_PLUGIN = VnpyLanePlugin(
    name="anchor_drift",
    load_config=AnchorDriftConfig.from_env,
    strategy_class=AnchorDriftVnpyStrategy,
    class_name="AnchorDriftVnpyStrategy",
    sync_prefix="drift",
    register=register_vnpy_strategies,
    switch=ANCHOR_DRIFT_SWITCH,
    uses_kline_stream=False,
)

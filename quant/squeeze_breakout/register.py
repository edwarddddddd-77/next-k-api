"""Smart Breakout — vnpy 策略注册。"""

from __future__ import annotations

from typing import List

from quant.common.exchange_env import resolve_live_exchange_id, resolve_market_data_exchange_id
from quant.common.kline_cache import norm_symbol
from quant.common.register_sizing import recent_atr
from quant.common.vnpy_wallet import lane_equity_usdt, migrate_vnpy_lane_tables
from quant.engine.exchanges.registry import vnpy_vt_symbol
from quant.engine.registry import VnpyLanePlugin
from quant.market import fetch_mark_price
from quant.squeeze_breakout.config import SqueezeBreakoutConfig
from quant.squeeze_breakout.sizing import size_for_breakout
from quant.squeeze_breakout.squeeze_vnpy import SqueezeBreakoutVnpyStrategy
from quant.squeeze_breakout.switches import SQUEEZE_BREAKOUT_SWITCH


def register_vnpy_strategies(cta_engine, cfg: SqueezeBreakoutConfig, wallet_cur) -> List[str]:
    if wallet_cur is not None:
        migrate_vnpy_lane_tables(wallet_cur)
    live_ex = resolve_live_exchange_id(cfg.live_exchange)
    md_exchange = resolve_market_data_exchange_id(cfg.market_data_exchange)
    names: List[str] = []
    settings = SqueezeBreakoutVnpyStrategy.from_breakout_config(cfg)
    interval = f"{max(1, int(cfg.signal_minutes))}m"
    for sym in cfg.symbol_list():
        sym = norm_symbol(sym)
        px = fetch_mark_price(sym, exchange_id=md_exchange) or 100.0
        eq = lane_equity_usdt(cfg, sym, cur=wallet_cur) if wallet_cur is not None else float(cfg.equity_usdt)
        atr = recent_atr(sym, interval, exchange_id=md_exchange, atr_period=max(2, int(cfg.squeeze_length) // 2))
        stop_dist = (float(cfg.sl_atr_buffer) + 1.0) * (atr if atr and atr > 0 else px * 0.015)
        vol = size_for_breakout(cfg, px, stop_distance=stop_dist, equity_usdt=eq)
        name = f"sqz_{sym.lower()}"
        cta_engine.add_strategy(
            class_name="SqueezeBreakoutVnpyStrategy",
            strategy_name=name,
            vt_symbol=vnpy_vt_symbol(sym, exchange_id=live_ex),
            setting={**settings, "fixed_size": max(vol, 0.001)},
        )
        names.append(name)
    return names


SQUEEZE_BREAKOUT_VNPY_PLUGIN = VnpyLanePlugin(
    name="squeeze_breakout",
    load_config=SqueezeBreakoutConfig.from_env,
    strategy_class=SqueezeBreakoutVnpyStrategy,
    class_name="SqueezeBreakoutVnpyStrategy",
    sync_prefix="sqz",
    register=register_vnpy_strategies,
    switch=SQUEEZE_BREAKOUT_SWITCH,
    uses_kline_stream=True,
)

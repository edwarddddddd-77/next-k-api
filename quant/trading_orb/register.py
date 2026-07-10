"""Trading ORB — vnpy 策略注册（执行层调用）。"""

from __future__ import annotations

from typing import Any, List

from quant.common.kline_cache import norm_symbol
from quant.common.macro_calendar import is_macro_skip_day
from quant.common.session_paper import _session_date_now
from quant.trading_orb.config import OrbVnpyConfig
from quant.trading_orb.trading_orb_vnpy import TradingOrbVnpyStrategy
from quant.common.exchange_env import resolve_live_exchange_id, resolve_market_data_exchange_id
from quant.engine.exchanges.registry import vnpy_vt_symbol
from quant.engine.registry import VnpyLanePlugin
from quant.trading_orb.switches import TRADING_ORB_SWITCH

from quant.market import fetch_mark_price


def register_vnpy_strategies(cta_engine, cfg: OrbVnpyConfig, wallet_cur) -> List[str]:
    session_date = _session_date_now(cfg.orb_session_cfg())
    if cfg.macro_filter and is_macro_skip_day(session_date):
        import logging

        logging.getLogger(__name__).info("[vnpy] ORB skipped today (macro_skip)")
        return []

    from quant.trading_orb.rel_volume import clear_baseline_cache, preload_pool_baselines
    from quant.trading_orb.sizing import fixed_size_for_orb

    live_ex = resolve_live_exchange_id(cfg.live_exchange)
    md_exchange = resolve_market_data_exchange_id(cfg.market_data_exchange)
    clear_baseline_cache()
    preload_pool_baselines(
        cfg.symbol_list(),
        cfg=cfg.orb_session_cfg(),
        lookback_days=int(cfg.vol_lookback_days),
        pause_sec=2.5,
        market_data_exchange=md_exchange,
    )
    names: List[str] = []
    settings = TradingOrbVnpyStrategy.from_orb_config(cfg)
    for sym in cfg.symbol_list():
        sym = norm_symbol(sym)
        px = fetch_mark_price(sym, exchange_id=md_exchange) or 100.0
        eq = float(cfg.equity_usdt)
        if wallet_cur is not None:
            from quant.trading_orb.equity import symbol_equity_usdt

            eq = symbol_equity_usdt(cfg, sym, cur=wallet_cur)
        stop_dist = float(cfg.stop_or_mult) * px * 0.01
        vol = fixed_size_for_orb(cfg, sym, px, stop_distance=stop_dist, equity_usdt=eq)
        name = f"orb_{sym.lower()}"
        cta_engine.add_strategy(
            class_name="TradingOrbVnpyStrategy",
            strategy_name=name,
            vt_symbol=vnpy_vt_symbol(sym, exchange_id=live_ex),
            setting={**settings, "fixed_size": vol},
        )
        names.append(name)
    return names


TRADING_ORB_VNPY_PLUGIN = VnpyLanePlugin(
    name="trading_orb",
    load_config=OrbVnpyConfig.from_env,
    strategy_class=TradingOrbVnpyStrategy,
    class_name="TradingOrbVnpyStrategy",
    sync_prefix="orb",
    register=register_vnpy_strategies,
    switch=TRADING_ORB_SWITCH,
    uses_kline_stream=True,
)

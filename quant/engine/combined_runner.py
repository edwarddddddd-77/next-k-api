"""vnpy 执行层 — 单引擎多 lane，策略注册由 quant.engine.registry 提供。"""

from __future__ import annotations

import logging
import time
from threading import Event
from typing import Any, Dict, List, Optional

from quant.common.kline_cache import norm_symbol
from quant.common.exchange_env import (
    resolve_lanes_live_exchange,
    resolve_market_data_exchange_id,
)
from quant.engine.exchanges.context import clear_runtime_live_exchange, set_runtime_live_exchange
from quant.engine.exchanges.registry import get_adapter
from quant.market.context import clear_runtime_market_data_exchange, set_runtime_market_data_exchange
from quant.engine.bootstrap import ensure_vnpy_path
from quant.engine.cta_rth_patch import apply_cta_engine_patches
from quant.engine.lane import find_symbol_pool_overlaps, get_enabled_vnpy_lanes, lane_live_enabled
from quant.engine.position_sync import sync_cta_positions
from quant.engine.registry import plugin_for_lane, register_strategy_classes

ensure_vnpy_path()

from vnpy.event import EventEngine  # noqa: E402
from vnpy.trader.engine import MainEngine  # noqa: E402
from vnpy.trader.setting import SETTINGS  # noqa: E402
from vnpy_ctastrategy import CtaStrategyApp  # noqa: E402
from vnpy_ctastrategy.base import EVENT_CTA_LOG  # noqa: E402

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    SETTINGS["log.active"] = True
    SETTINGS["log.console"] = True


def _wait_contracts(gateway, symbols: List[str], *, timeout_sec: float) -> bool:
    names = {norm_symbol(s) for s in symbols}
    deadline = time.time() + max(10.0, float(timeout_sec))
    while time.time() < deadline:
        if names.issubset(set(gateway.name_contract_map.keys())):
            return True
        time.sleep(0.5)
    missing = sorted(names - set(gateway.name_contract_map.keys()))
    if missing:
        logger.error("[combined-vnpy] contracts not ready: %s", missing)
    return not missing


class CombinedVnpyEngine:
    def __init__(self) -> None:
        self._event_engine: Optional[EventEngine] = None
        self._main_engine: Optional[MainEngine] = None
        self._cta_engine = None
        self._started: List[str] = []

    def _bootstrap_abort(self, out: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        clear_runtime_live_exchange()
        clear_runtime_market_data_exchange()
        out.update({"ok": False, **extra})
        self.shutdown()
        return out

    def bootstrap(self, *, init_wait_sec: float = 30.0) -> Dict[str, Any]:
        lanes = get_enabled_vnpy_lanes()
        out: Dict[str, Any] = {
            "ok": False,
            "engine": "vnpy",
            "exchange": None,
            "gateway": None,
            "lane": None,
            "lanes": [],
            "symbols": [],
            "strategies": [],
            "reason": None,
        }
        if not lanes:
            out.update({"ok": True, "skipped": True, "reason": "no_vnpy_lanes"})
            return out

        try:
            live_ex = resolve_lanes_live_exchange(lanes)
        except ValueError as exc:
            out.update({"ok": False, "reason": "multiple_live_exchanges", "detail": str(exc)})
            return out

        first_cfg = lanes[0][1]
        market_ex = resolve_market_data_exchange_id(getattr(first_cfg, "market_data_exchange", None))
        set_runtime_live_exchange(live_ex)
        set_runtime_market_data_exchange(market_ex)

        adapter = get_adapter(live_ex)
        lane_names = [n for n, _ in lanes]
        out.update(
            {
                "exchange": adapter.id,
                "gateway": adapter.gateway_name,
                "market_data_exchange": market_ex,
                "lane": "combined" if len(lanes) > 1 else (lane_names[0] if lane_names else None),
                "lanes": lane_names,
            }
        )

        need_live = any(lane_live_enabled(cfg) for _, cfg in lanes)
        if need_live and not adapter.credentials_configured():
            return self._bootstrap_abort(out, reason=adapter.credentials_missing_reason)

        all_symbols: List[str] = []
        for _, cfg in lanes:
            all_symbols.extend(cfg.symbol_list())
        all_symbols = sorted({norm_symbol(s) for s in all_symbols})
        out["symbols"] = all_symbols
        if not all_symbols:
            return self._bootstrap_abort(out, reason="no_symbols")

        overlaps = find_symbol_pool_overlaps(lanes)
        if overlaps:
            logger.error("[combined-vnpy] lane symbol overlap not allowed: %s", overlaps)
            return self._bootstrap_abort(out, reason="symbol_pool_overlap", overlap=overlaps)

        _configure_logging()
        self._event_engine = EventEngine()
        self._main_engine = MainEngine(self._event_engine)
        self._main_engine.add_gateway(adapter.gateway_class, adapter.gateway_name)
        self._cta_engine = self._main_engine.add_app(CtaStrategyApp)
        apply_cta_engine_patches()
        register_strategy_classes(self._cta_engine)
        self._event_engine.register(EVENT_CTA_LOG, lambda e: logger.info("[cta] %s", e.data))

        gateway = self._main_engine.get_gateway(adapter.gateway_name)
        if gateway:
            gateway.connect(adapter.connect_setting())

        contract_wait = max(90.0, float(init_wait_sec) * max(1, len(all_symbols)))
        if gateway and not _wait_contracts(gateway, all_symbols, timeout_sec=contract_wait):
            return self._bootstrap_abort(out, reason=adapter.contracts_not_ready_reason)

        if need_live and adapter.credentials_configured():
            try:
                for _, cfg in lanes:
                    adapter.ensure_pool_leverage(cfg.symbol_list(), cfg)
            except Exception as exc:
                logger.warning("[combined-vnpy] leverage setup failed: %s", exc)

        self._cta_engine.init_engine()
        self._started = []

        wallet_cur = None
        wallet_conn = None
        try:
            from accumulation_radar import init_db
            from quant.trading_orb.db import migrate_orb_vnpy_tables

            wallet_conn = init_db()
            wallet_cur = wallet_conn.cursor()
            migrate_orb_vnpy_tables(wallet_cur)
        except Exception as exc:
            logger.warning("[combined-vnpy] wallet load skipped: %s", exc)
            wallet_cur = None

        try:
            for lane_name, cfg in lanes:
                plugin = plugin_for_lane(lane_name)
                if plugin is None:
                    continue
                self._started.extend(plugin.register(self._cta_engine, cfg, wallet_cur))
        finally:
            if wallet_conn is not None:
                wallet_conn.close()

        for name in self._started:
            logger.info("[combined-vnpy] init begin %s", name)
            fut = self._cta_engine.init_strategy(name)
            try:
                fut.result(timeout=max(120.0, float(init_wait_sec) * 3))
            except Exception as exc:
                logger.warning("[combined-vnpy] init %s failed: %s", name, exc)

        inited = [n for n, s in self._cta_engine.strategies.items() if getattr(s, "inited", False)]
        not_ready = [n for n, s in self._cta_engine.strategies.items() if not getattr(s, "inited", False)]
        if not inited:
            return self._bootstrap_abort(out, reason="strategies_not_inited", not_ready=not_ready)
        if not_ready:
            out["not_ready"] = not_ready

        if need_live and adapter.credentials_configured() and gateway:
            try:
                gateway.hydrate_from_exchange(all_symbols)
            except Exception as exc:
                logger.warning("[combined-vnpy] gateway hydrate failed: %s", exc)
            for lane_name, cfg in lanes:
                plugin = plugin_for_lane(lane_name)
                if plugin is None:
                    continue
                if getattr(cfg, "shadow", False):
                    continue
                try:
                    synced = sync_cta_positions(
                        self._cta_engine,
                        cfg.symbol_list(),
                        strategy_prefix=plugin.sync_prefix,
                        restore_levels=True,
                    )
                    if synced:
                        logger.info("[combined-vnpy] position sync %s: %s", plugin.sync_prefix, synced)
                except Exception as exc:
                    logger.warning("[combined-vnpy] position sync %s failed: %s", plugin.sync_prefix, exc)

        self._cta_engine.start_all_strategies()

        out["strategies"] = inited
        out["ok"] = True
        logger.info("[combined-vnpy] started %d strategies lanes=%s", len(inited), lane_names)
        return out

    def run_until(self, stop_event: Event) -> None:
        while not stop_event.is_set():
            stop_event.wait(1.0)

    def shutdown(self) -> None:
        try:
            if self._cta_engine is not None:
                self._cta_engine.stop_all_strategies()
        except Exception as exc:
            logger.warning("[combined-vnpy] stop: %s", exc)
        try:
            if self._main_engine is not None:
                self._main_engine.close()
        except Exception as exc:
            logger.warning("[combined-vnpy] close: %s", exc)
        self._cta_engine = None
        self._main_engine = None
        self._event_engine = None
        self._started = []
        clear_runtime_live_exchange()
        clear_runtime_market_data_exchange()


def run_combined_vnpy(
    *,
    run_seconds: Optional[float] = None,
    init_wait_sec: float = 60.0,
    stop_event: Optional[Event] = None,
) -> Dict[str, Any]:
    engine = CombinedVnpyEngine()
    out = engine.bootstrap(init_wait_sec=init_wait_sec)
    if not out.get("ok") or out.get("skipped"):
        return out
    evt = stop_event or Event()
    t0 = time.time()
    try:
        while not evt.is_set():
            if run_seconds is not None and (time.time() - t0) >= float(run_seconds):
                break
            evt.wait(1.0)
    except KeyboardInterrupt:
        logger.info("[combined-vnpy] interrupted")
    finally:
        engine.shutdown()
    return out

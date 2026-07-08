"""ICT 2022 vnpy 引擎：BinanceLinearGateway + TradingIct2022VnpyStrategy。"""

from __future__ import annotations

import logging
import time
from threading import Event
from typing import Any, Dict, List, Optional

from orb.core.kline_cache import norm_symbol
from orb.ict.config import IctVnpyConfig
from orb.ict.vnpy.bootstrap import ensure_vnpy_path
from orb.ict.vnpy.strategies.trading_ict_2022_vnpy import TradingIct2022VnpyStrategy
from orb.vnpy.binance_gateway import (
    GATEWAY_NAME,
    VnpyBinanceLinearGateway,
    binance_connect_setting,
    binance_credentials_configured,
    vnpy_vt_symbol,
)
from orb.vnpy.cta_rth_patch import apply_cta_engine_patches
from orb.vnpy.position_sync import sync_cta_positions

ensure_vnpy_path()

from vnpy.event import EventEngine  # noqa: E402
from vnpy.trader.engine import MainEngine  # noqa: E402
from vnpy.trader.setting import SETTINGS  # noqa: E402
from vnpy_ctastrategy import CtaStrategyApp  # noqa: E402
from vnpy_ctastrategy.base import EVENT_CTA_LOG  # noqa: E402

from binance_fapi import fetch_mark_price  # noqa: E402

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    SETTINGS["log.active"] = True
    SETTINGS["log.console"] = True


def _wait_contracts(gateway: VnpyBinanceLinearGateway, symbols: List[str], *, timeout_sec: float) -> bool:
    names = {norm_symbol(s) for s in symbols}
    deadline = time.time() + max(10.0, float(timeout_sec))
    while time.time() < deadline:
        if names.issubset(set(gateway.name_contract_map.keys())):
            return True
        time.sleep(0.5)
    missing = sorted(names - set(gateway.name_contract_map.keys()))
    if missing:
        logger.error("[ict-vnpy] contracts not ready: %s", missing)
    return not missing


class IctVnpyEngine:
    def __init__(self) -> None:
        self._event_engine: Optional[EventEngine] = None
        self._main_engine: Optional[MainEngine] = None
        self._cta_engine = None
        self._started: List[str] = []

    def bootstrap(self, *, init_wait_sec: float = 30.0) -> Dict[str, Any]:
        ict = IctVnpyConfig.from_env()
        out: Dict[str, Any] = {
            "ok": False,
            "engine": "vnpy",
            "gateway": GATEWAY_NAME,
            "lane": ict.lane,
            "symbols": [],
            "strategies": [],
            "reason": None,
        }
        if not ict.enabled:
            out.update({"ok": True, "skipped": True, "reason": "ict_vnpy_disabled"})
            return out
        if not ict.is_vnpy_engine():
            out.update({"ok": True, "skipped": True, "reason": "not_vnpy_engine"})
            return out
        if ict.live_enabled and not binance_credentials_configured():
            out.update({"ok": False, "reason": "binance_credentials_missing"})
            return out

        symbols = ict.symbol_list()
        out["symbols"] = symbols
        if not symbols:
            out.update({"ok": False, "reason": "no_symbols"})
            return out

        _configure_logging()
        self._event_engine = EventEngine()
        self._main_engine = MainEngine(self._event_engine)
        self._main_engine.add_gateway(VnpyBinanceLinearGateway, GATEWAY_NAME)
        self._cta_engine = self._main_engine.add_app(CtaStrategyApp)
        apply_cta_engine_patches()
        self._cta_engine.classes["TradingIct2022VnpyStrategy"] = TradingIct2022VnpyStrategy
        self._event_engine.register(EVENT_CTA_LOG, lambda e: logger.info("[ict-cta] %s", e.data))

        gateway = self._main_engine.get_gateway(GATEWAY_NAME)
        if gateway:
            gateway.connect(binance_connect_setting())

        contract_wait = max(60.0, float(init_wait_sec) * max(1, len(symbols)))
        if gateway and not _wait_contracts(gateway, symbols, timeout_sec=contract_wait):
            out.update({"ok": False, "reason": "binance_contracts_not_ready"})
            return out

        if ict.live_enabled and binance_credentials_configured():
            try:
                from orb.vnpy.binance_account import ensure_pool_leverage

                ensure_pool_leverage(symbols, ict)
            except Exception as exc:
                logger.warning("[ict-vnpy] leverage setup failed: %s", exc)

        self._cta_engine.init_engine()
        base_settings = TradingIct2022VnpyStrategy.from_ict_config(ict)
        self._started = []

        from orb.ict.vnpy.sizing import fixed_size_for_ict

        wallet_cur = None
        wallet_conn = None
        if ict.compound:
            try:
                from accumulation_radar import init_db
                from orb.trading_orb.db import migrate_orb_vnpy_tables

                wallet_conn = init_db()
                wallet_cur = wallet_conn.cursor()
                migrate_orb_vnpy_tables(wallet_cur)
            except Exception as exc:
                logger.warning("[ict-vnpy] wallet load skipped: %s", exc)
                wallet_cur = None

        try:
            for sym in symbols:
                sym = norm_symbol(sym)
                px = fetch_mark_price(sym) or 100.0
                eq = float(ict.equity_usdt)
                if wallet_cur is not None and ict.compound:
                    from orb.ict.equity import symbol_equity_usdt

                    eq = symbol_equity_usdt(ict, sym, cur=wallet_cur)
                vol = fixed_size_for_ict(ict, px, equity_usdt=eq)
                name = f"ict2022_{sym.lower()}"
                self._cta_engine.add_strategy(
                    class_name="TradingIct2022VnpyStrategy",
                    strategy_name=name,
                    vt_symbol=vnpy_vt_symbol(sym),
                    setting={**base_settings, "fixed_size": vol},
                )
                self._started.append(name)
        finally:
            if wallet_conn is not None:
                wallet_conn.close()

        for name in self._started:
            logger.info("[ict-vnpy] strategy init begin %s", name)
            fut = self._cta_engine.init_strategy(name)
            try:
                fut.result(timeout=max(120.0, float(init_wait_sec) * 3))
            except Exception as exc:
                logger.warning("[ict-vnpy] strategy init %s failed: %s", name, exc)
        inited = [n for n, s in self._cta_engine.strategies.items() if getattr(s, "inited", False)]
        if not inited:
            out.update({"ok": False, "reason": "strategies_not_inited"})
            return out

        gateway = self._main_engine.get_gateway(GATEWAY_NAME) if self._main_engine else None
        if ict.live_enabled and binance_credentials_configured() and not ict.shadow:
            if gateway:
                try:
                    gateway.hydrate_from_exchange(symbols)
                except Exception as exc:
                    logger.warning("[ict-vnpy] gateway hydrate failed: %s", exc)
            try:
                synced = sync_cta_positions(
                    self._cta_engine,
                    symbols,
                    strategy_prefix="ict2022",
                    restore_levels=True,
                )
                if synced:
                    logger.info("[ict-vnpy] position sync: %s", synced)
            except Exception as exc:
                logger.warning("[ict-vnpy] position sync failed: %s", exc)

        self._cta_engine.start_all_strategies()

        out["strategies"] = inited
        out["ok"] = True
        logger.info("[ict-vnpy] started %d strategies: %s", len(inited), inited)
        return out

    def run_until(self, stop_event: Event) -> None:
        while not stop_event.is_set():
            stop_event.wait(1.0)

    def shutdown(self) -> None:
        try:
            if self._cta_engine is not None:
                self._cta_engine.stop_all_strategies()
        except Exception as exc:
            logger.warning("[ict-vnpy] stop strategies: %s", exc)
        try:
            if self._main_engine is not None:
                self._main_engine.close()
        except Exception as exc:
            logger.warning("[ict-vnpy] main_engine close: %s", exc)
        self._cta_engine = None
        self._main_engine = None
        self._event_engine = None
        self._started = []


def run_vnpy_ict(
    *,
    run_seconds: Optional[float] = None,
    init_wait_sec: float = 30.0,
    stop_event: Optional[Event] = None,
) -> Dict[str, Any]:
    engine = IctVnpyEngine()
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
        logger.info("[ict-vnpy] interrupted")
    finally:
        engine.shutdown()
    return out

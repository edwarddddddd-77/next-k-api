"""单 vnpy 引擎同时跑 ORB + ICT 2022 + Aberration（共享币安 Gateway）。"""

from __future__ import annotations

import logging
import time
from threading import Event
from typing import Any, Dict, List, Optional

from orb.aberration.config import AberrationVnpyConfig
from orb.aberration.vnpy.strategies.aberration_vnpy import AberrationVnpyStrategy
from orb.core.kline_cache import norm_symbol
from orb.core.macro_calendar import is_macro_skip_day
from orb.core.session_paper import _session_date_now
from orb.ict.config import IctVnpyConfig
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
from orb.trading_orb.config import OrbVnpyConfig
from orb.trading_orb.vnpy.bootstrap import ensure_vnpy_path
from orb.trading_orb.vnpy.strategies.trading_orb_vnpy import TradingOrbVnpyStrategy
from orb.vnpy.lane import find_symbol_pool_overlaps, get_enabled_vnpy_lanes, lane_live_enabled

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
        logger.error("[combined-vnpy] contracts not ready: %s", missing)
    return not missing


class CombinedVnpyEngine:
    def __init__(self) -> None:
        self._event_engine: Optional[EventEngine] = None
        self._main_engine: Optional[MainEngine] = None
        self._cta_engine = None
        self._started: List[str] = []

    def bootstrap(self, *, init_wait_sec: float = 30.0) -> Dict[str, Any]:
        lanes = get_enabled_vnpy_lanes()
        lane_names = [n for n, _ in lanes]
        out: Dict[str, Any] = {
            "ok": False,
            "engine": "vnpy",
            "gateway": GATEWAY_NAME,
            "lane": "combined" if len(lanes) > 1 else (lane_names[0] if lane_names else None),
            "lanes": lane_names,
            "symbols": [],
            "strategies": [],
            "reason": None,
        }
        if not lanes:
            out.update({"ok": True, "skipped": True, "reason": "no_vnpy_lanes"})
            return out

        need_live = any(lane_live_enabled(cfg) for _, cfg in lanes)
        if need_live and not binance_credentials_configured():
            out.update({"ok": False, "reason": "binance_credentials_missing"})
            return out

        all_symbols: List[str] = []
        for _, cfg in lanes:
            all_symbols.extend(cfg.symbol_list())
        all_symbols = sorted({norm_symbol(s) for s in all_symbols})
        out["symbols"] = all_symbols
        if not all_symbols:
            out.update({"ok": False, "reason": "no_symbols"})
            return out

        overlaps = find_symbol_pool_overlaps(lanes)
        if overlaps:
            out.update({"ok": False, "reason": "symbol_pool_overlap", "overlap": overlaps})
            logger.error("[combined-vnpy] lane symbol overlap not allowed: %s", overlaps)
            return out

        _configure_logging()
        self._event_engine = EventEngine()
        self._main_engine = MainEngine(self._event_engine)
        self._main_engine.add_gateway(VnpyBinanceLinearGateway, GATEWAY_NAME)
        self._cta_engine = self._main_engine.add_app(CtaStrategyApp)
        apply_cta_engine_patches()
        self._cta_engine.classes["TradingOrbVnpyStrategy"] = TradingOrbVnpyStrategy
        self._cta_engine.classes["TradingIct2022VnpyStrategy"] = TradingIct2022VnpyStrategy
        self._cta_engine.classes["AberrationVnpyStrategy"] = AberrationVnpyStrategy
        self._event_engine.register(EVENT_CTA_LOG, lambda e: logger.info("[cta] %s", e.data))

        gateway = self._main_engine.get_gateway(GATEWAY_NAME)
        if gateway:
            gateway.connect(binance_connect_setting())

        contract_wait = max(90.0, float(init_wait_sec) * max(1, len(all_symbols)))
        if gateway and not _wait_contracts(gateway, all_symbols, timeout_sec=contract_wait):
            out.update({"ok": False, "reason": "binance_contracts_not_ready"})
            return out

        if any(lane_live_enabled(cfg) for _, cfg in lanes) and binance_credentials_configured():
            try:
                from orb.vnpy.binance_account import ensure_pool_leverage

                for _, cfg in lanes:
                    ensure_pool_leverage(cfg.symbol_list(), cfg)
            except Exception as exc:
                logger.warning("[combined-vnpy] leverage setup failed: %s", exc)

        self._cta_engine.init_engine()
        self._started = []

        wallet_cur = None
        wallet_conn = None
        try:
            from accumulation_radar import init_db
            from orb.trading_orb.db import migrate_orb_vnpy_tables

            wallet_conn = init_db()
            wallet_cur = wallet_conn.cursor()
            migrate_orb_vnpy_tables(wallet_cur)
        except Exception as exc:
            logger.warning("[combined-vnpy] wallet load skipped: %s", exc)
            wallet_cur = None

        try:
            self._started.extend(self._add_orb_strategies(wallet_cur, lanes))
            self._started.extend(self._add_ict_strategies(wallet_cur, lanes))
            self._started.extend(self._add_aberration_strategies(wallet_cur, lanes))
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
            out.update({"ok": False, "reason": "strategies_not_inited", "not_ready": not_ready})
            return out
        if not_ready:
            out["not_ready"] = not_ready

        if need_live and binance_credentials_configured() and gateway:
            try:
                gateway.hydrate_from_exchange(all_symbols)
            except Exception as exc:
                logger.warning("[combined-vnpy] gateway hydrate failed: %s", exc)
            ict_cfg = next((c for n, c in lanes if n == "ict_2022"), None)
            ab_cfg = next((c for n, c in lanes if n == "aberration"), None)
            for prefix, syms, restore in self._sync_groups(lanes):
                if prefix == "ict2022" and ict_cfg and getattr(ict_cfg, "shadow", False):
                    continue
                if prefix == "aberration" and ab_cfg and getattr(ab_cfg, "shadow", False):
                    continue
                try:
                    synced = sync_cta_positions(
                        self._cta_engine,
                        syms,
                        strategy_prefix=prefix,
                        restore_levels=restore,
                    )
                    if synced:
                        logger.info("[combined-vnpy] position sync %s: %s", prefix, synced)
                except Exception as exc:
                    logger.warning("[combined-vnpy] position sync %s failed: %s", prefix, exc)

        self._cta_engine.start_all_strategies()

        out["strategies"] = inited
        out["ok"] = True
        logger.info("[combined-vnpy] started %d strategies lanes=%s", len(inited), lane_names)
        return out

    def _sync_groups(self, lanes) -> List[tuple[str, List[str], bool]]:
        groups: List[tuple[str, List[str], bool]] = []
        for name, cfg in lanes:
            if name == "trading_orb":
                groups.append(("orb", cfg.symbol_list(), True))
            elif name == "ict_2022":
                groups.append(("ict2022", cfg.symbol_list(), True))
            elif name == "aberration":
                groups.append(("aberration", cfg.symbol_list(), True))
        return groups

    def _add_orb_strategies(self, wallet_cur, lanes) -> List[str]:
        orb = next((cfg for n, cfg in lanes if n == "trading_orb"), None)
        if orb is None:
            return []
        session_date = _session_date_now(orb.orb_session_cfg())
        if orb.macro_filter and is_macro_skip_day(session_date):
            logger.info("[combined-vnpy] ORB skipped today (macro_skip); ICT still runs")
            return []

        from orb.trading_orb.vnpy.rel_volume import clear_baseline_cache, preload_pool_baselines
        from orb.trading_orb.vnpy.sizing import fixed_size_for_orb

        clear_baseline_cache()
        preload_pool_baselines(
            orb.symbol_list(),
            cfg=orb.orb_session_cfg(),
            lookback_days=int(orb.vol_lookback_days),
            pause_sec=2.5,
        )
        names: List[str] = []
        settings = TradingOrbVnpyStrategy.from_orb_config(orb)
        for sym in orb.symbol_list():
            sym = norm_symbol(sym)
            px = fetch_mark_price(sym) or 100.0
            eq = float(orb.equity_usdt)
            if wallet_cur is not None:
                from orb.trading_orb.equity import symbol_equity_usdt

                eq = symbol_equity_usdt(orb, sym, cur=wallet_cur)
            stop_dist = float(orb.stop_or_mult) * px * 0.01
            vol = fixed_size_for_orb(orb, sym, px, stop_distance=stop_dist, equity_usdt=eq)
            name = f"orb_{sym.lower()}"
            self._cta_engine.add_strategy(
                class_name="TradingOrbVnpyStrategy",
                strategy_name=name,
                vt_symbol=vnpy_vt_symbol(sym),
                setting={**settings, "fixed_size": vol},
            )
            names.append(name)
        return names

    def _add_ict_strategies(self, wallet_cur, lanes) -> List[str]:
        ict = next((cfg for n, cfg in lanes if n == "ict_2022"), None)
        if ict is None:
            return []
        from orb.ict.vnpy.sizing import fixed_size_for_ict

        names: List[str] = []
        settings = TradingIct2022VnpyStrategy.from_ict_config(ict)
        for sym in ict.symbol_list():
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
                setting={**settings, "fixed_size": vol},
            )
            names.append(name)
        return names

    def _add_aberration_strategies(self, wallet_cur, lanes) -> List[str]:
        ab = next((cfg for n, cfg in lanes if n == "aberration"), None)
        if ab is None:
            return []
        from orb.aberration.vnpy.sizing import fixed_size_for_aberration

        names: List[str] = []
        settings = AberrationVnpyStrategy.from_aberration_config(ab)
        for sym in ab.symbol_list():
            sym = norm_symbol(sym)
            px = fetch_mark_price(sym) or 100.0
            eq = float(ab.equity_usdt)
            if wallet_cur is not None and ab.compound:
                from orb.aberration.equity import symbol_equity_usdt

                eq = symbol_equity_usdt(ab, sym, cur=wallet_cur)
            vol = fixed_size_for_aberration(ab, px, equity_usdt=eq)
            name = f"aberration_{sym.lower()}"
            self._cta_engine.add_strategy(
                class_name="AberrationVnpyStrategy",
                strategy_name=name,
                vt_symbol=vnpy_vt_symbol(sym),
                setting={**settings, "fixed_size": vol},
            )
            names.append(name)
        return names

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

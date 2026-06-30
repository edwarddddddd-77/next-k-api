"""ORB 2.0 纸面扫描：ML Live Gate + 8-robot 资金池。"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from orb.ml.model import BreakoutModelBundle
from orb.core.config import OrbConfig
from orb.core.db import (
    count_open_positions,
    ensure_symbol_bots,
    fetch_open_hold,
    migrate_orb_tables,
    symbol_bot_enabled,
    symbol_bot_wallet_balance,
)
from orb.core.signals import OrbSignal, compute_position_notional
from orb.ml.features import extract_features
from orb.ml.gate import (
    LiveGateConfig,
    evaluate_open_decision,
    evaluate_open_decision_without_ml,
    rollback_open_decision,
)
from orb.core.macro_calendar import is_macro_skip_day, macro_calendar_status
from orb.core.paper import (
    _idle_scan_skip_reason,
    _live_open,
    _load_1m_df,
    _load_daily_df,
    _load_signal_df,
    _scan_params,
    _session_date_now,
    _upsert_signal,
    analyze_at_ms,
    in_regular_session,
    is_actionable,
    resolve_open_positions,
)
from orb.core.fvg import find_fvg_limit_entry, first_or_reclaim_bar_ms, synthesize_fvg_fill_from_protocol, uses_fvg_entry
from orb.core.session import session_anchor_ms, session_close_ms
from orb.v2.config import OrbV2Config
from orb.core.live_exec import (
    cancel_fvg_live_limit,
    fvg_api_signal_id,
    live_enabled,
    live_ingest_succeeded,
    live_open_is_pending,
    notify_open,
    protocol_fvg_entry_price,
    protocol_fvg_open_done,
    protocol_fvg_open_status,
    sync_live_pending_entries,
)
from orb.core.protocol_client import LIVE_PENDING_NOTE, live_pending_note
from orb.v2.db import (
    delete_fvg_watch,
    list_fvg_watches,
    mark_breakout_seen,
    migrate_orb_v2_tables,
    rollback_breakout_opened,
    upsert_fvg_watch,
)
from orb.v2.gate_state import load_gate_day_state, persist_gate_day_state, v2_session_traded
from orb.v2.robots import (
    bound_robot_id_for_open,
    busy_robot_ids,
    ensure_orb_robots,
    list_robot_wallet_balances,
    next_free_robot_id,
    resolve_robot_pool_size,
    robot_bound_mode,
    robot_equity_for_signals,
    robot_equity_from_env,
    robot_symbol_bindings,
    robot_wallet_balance,
    symbol_to_robot_id,
)

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rollback_failed_live_open(
    cur,
    *,
    session_day: str,
    sym: str,
    gate_state,
    stats: Dict[str, Any],
    live_open: Optional[Dict[str, Any]],
) -> None:
    rollback_breakout_opened(cur, session_day, sym)
    rollback_open_decision(gate_state, symbol=sym)
    cur.execute(
        """
        DELETE FROM orb_signals
        WHERE symbol = ? AND outcome IS NULL AND side IN ('LONG', 'SHORT')
        """,
        (str(sym).strip().upper(),),
    )
    stats["written"] = max(0, int(stats.get("written") or 0) - 1)
    stats["opens"] = [row for row in stats.get("opens") or [] if row.get("symbol") != sym]
    fail_reason = "live_open_failed"
    if isinstance(live_open, dict):
        if live_open.get("error"):
            fail_reason = str(live_open["error"])
        else:
            for detail in live_open.get("details") or []:
                if detail.get("error"):
                    fail_reason = str(detail["error"])
                    break
                if str(detail.get("action") or "").lower() == "error":
                    fail_reason = str(detail.get("error") or fail_reason)
                    break
    stats["skipped"].append({"symbol": sym, "reason": fail_reason, "live": live_open})
    logger.warning("[orb_v2] live open failed %s: %s", sym, live_open)


def _scan_params_v2(
    cfg: OrbConfig,
    *,
    gate: LiveGateConfig,
    model: Optional[BreakoutModelBundle],
    ml_enabled: bool,
    shadow: bool,
    use_robots: bool,
    robot_bound: bool,
    robot_count: int,
    robot_equity: float,
) -> Dict[str, Any]:
    base = _scan_params(cfg)
    if robot_bound:
        sizing = "robot_bound"
    elif use_robots:
        sizing = "eight_robots"
    else:
        sizing = "per_symbol"
    base.update(
        {
            "strategy": "orb_v2",
            "orb_version": 2,
            "ml_ranker": model.kind if model is not None else ("disabled" if not ml_enabled else None),
            "ml_enabled": ml_enabled,
            "gate_min_p_true": gate.min_p_true,
            "gate_min_breakout_score": gate.min_breakout_score,
            "gate_max_opens": gate.max_opens_per_day,
            "gate_shadow": shadow,
            "sizing": sizing,
            "robot_bound": robot_bound,
            "robot_count": robot_count if use_robots else None,
            "robot_equity_usdt": robot_equity if use_robots else None,
            "entry_fill": cfg.entry_fill,
        }
    )
    return base


def _paper_breakout_score(
    sym: str,
    sig: OrbSignal,
    cfg: OrbConfig,
    *,
    session_day: str,
    now_ms: int,
    df5_cache: Dict[str, Any],
) -> Optional[float]:
    from orb.core.breakout_score import breakout_score_for_signal, df5_for_breakout_score

    df5 = df5_for_breakout_score(
        sym,
        sig,
        cfg,
        session_day=session_day,
        now_ms=now_ms,
        df5_cache=df5_cache,
    )
    if df5 is None or getattr(df5, "empty", True):
        logger.warning(
            "[breakout_score] %s score unavailable (empty klines entry_bar=%s)",
            sym,
            sig.entry_bar_open_ms,
        )
        return None
    score = round(breakout_score_for_signal(sig, df5, cfg, now_ms=now_ms), 2)
    logger.info(
        "[breakout_score] %s score=%.1f side=%s entry_bar=%s",
        sym,
        score,
        sig.side,
        sig.entry_bar_open_ms,
    )
    return score


def _load_model(v2: OrbV2Config) -> Optional[BreakoutModelBundle]:
    bundle = BreakoutModelBundle.load(
        gbm_path=v2.gbm_path,
        profiles_path=v2.profiles_path,
    )
    if not bundle.is_ready or bundle.ranker.gbm is None:
        logger.error(
            "[orb_v2] production GBM required: gbm_path=%s exists=%s kind=%s",
            bundle.gbm_path,
            bundle.gbm_path.is_file(),
            bundle.kind,
        )
        return None
    return bundle


def _record_v2_run(
    cur,
    *,
    now_utc: str,
    symbols_scanned: int,
    opens: int,
    gate_skips: int,
    detail: Dict[str, Any],
) -> None:
    cur.execute(
        """
        INSERT INTO orb_v2_runs (ran_at_utc, symbols_scanned, opens, gate_skips, detail_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (now_utc, symbols_scanned, opens, gate_skips, json.dumps(detail, default=str)),
    )


def _apply_robot_notional(sig: OrbSignal, *, entry: float, sl: float, cfg: OrbConfig, bot_equity: float) -> None:
    sig.paper_notional_usdt = round(
        compute_position_notional(entry=entry, sl=sl, cfg=cfg, bot_equity_usdt=bot_equity),
        4,
    )


def _orb_signal_to_json(sig: OrbSignal) -> str:
    from dataclasses import asdict

    return json.dumps(asdict(sig), default=str)


def _orb_signal_from_json(raw: str) -> OrbSignal:
    d = json.loads(raw or "{}")
    return OrbSignal(
        symbol=str(d.get("symbol") or ""),
        price=float(d.get("price") or 0),
        side=str(d.get("side") or "FLAT"),
        play=str(d.get("play") or ""),
        confidence=str(d.get("confidence") or "low"),
        reasons=list(d.get("reasons") or []),
        or_high=float(d.get("or_high") or 0),
        or_low=float(d.get("or_low") or 0),
        or_mid=float(d.get("or_mid") or 0),
        or_width_pct=float(d.get("or_width_pct") or 0),
        session_date=str(d.get("session_date") or ""),
        entry_bar_open_ms=int(d["entry_bar_open_ms"]) if d.get("entry_bar_open_ms") else None,
        fvg_confirm_bar_ms=int(d["fvg_confirm_bar_ms"]) if d.get("fvg_confirm_bar_ms") else None,
        sl_price=float(d["sl_price"]) if d.get("sl_price") is not None else None,
        tp_price=float(d["tp_price"]) if d.get("tp_price") is not None else None,
        r_unit=float(d["r_unit"]) if d.get("r_unit") is not None else None,
        paper_notional_usdt=float(d["paper_notional_usdt"])
        if d.get("paper_notional_usdt") is not None
        else None,
        volume=float(d.get("volume") or 0),
        vol_ma=float(d.get("vol_ma") or 0),
    )


def _merge_fvg_fill(sig: OrbSignal, fill_sig: OrbSignal) -> None:
    confirm = sig.entry_bar_open_ms
    sig.fvg_confirm_bar_ms = getattr(fill_sig, "fvg_confirm_bar_ms", None) or confirm
    sig.price = float(fill_sig.price)
    sig.sl_price = fill_sig.sl_price
    sig.tp_price = fill_sig.tp_price
    sig.entry_bar_open_ms = fill_sig.entry_bar_open_ms
    if fill_sig.r_unit is not None:
        sig.r_unit = fill_sig.r_unit


def _pending_note_for_open(sig: OrbSignal, cfg: OrbConfig) -> str:
    if uses_fvg_entry(cfg):
        return live_pending_note(fvg_api_signal_id(sig))
    return LIVE_PENDING_NOTE


def _cancel_fvg_watch_limit(
    sig: OrbSignal,
    watch: Dict[str, Any],
    cfg: OrbConfig,
    *,
    reason: str,
) -> None:
    if str(watch.get("reason") or "") != "fvg_limit_pending":
        return
    cancel_fvg_live_limit(sig, cfg, reason=reason)


def _preview_fvg_live_notional(
    sig: OrbSignal,
    conn,
    cfg: OrbConfig,
    *,
    use_robots: bool,
    robot_count: int,
    robot_init: float,
    bot_equity: float,
    signal_equity: float,
) -> None:
    if use_robots:
        rid = next_free_robot_id(conn.cursor(), count=robot_count, initial_equity_usdt=robot_init)
        if rid is not None:
            rw = robot_wallet_balance(conn, rid, initial_equity_usdt=robot_init, sync=False)
            _apply_robot_notional(
                sig,
                entry=float(sig.price),
                sl=float(sig.sl_price or sig.price),
                cfg=cfg,
                bot_equity=rw,
            )
            return
        pool_eq = float(signal_equity or 0)
        if pool_eq > 0:
            _apply_robot_notional(
                sig,
                entry=float(sig.price),
                sl=float(sig.sl_price or sig.price),
                cfg=cfg,
                bot_equity=pool_eq,
            )
            return
    bot_wallet = symbol_bot_wallet_balance(conn, sig.symbol, initial_equity_usdt=bot_equity, sync=False)
    _apply_robot_notional(
        sig,
        entry=float(sig.price),
        sl=float(sig.sl_price or sig.price),
        cfg=cfg,
        bot_equity=bot_wallet,
    )


def _submit_fvg_live_limit(
    sig: OrbSignal,
    cfg: OrbConfig,
    *,
    sym: str,
    stats: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not cfg.live_enabled or not live_enabled(cfg):
        return None
    if protocol_fvg_open_done(sig, cfg):
        return {"skipped": True, "reason": "protocol_already_open", "details": [{"action": "duplicate"}]}
    result = notify_open(sig, cfg)
    stats["live"].append({"action": "fvg_limit", "symbol": sym, "result": result})
    return result


def _live_open_after_fvg(sig: OrbSignal, cfg: OrbConfig) -> Optional[Dict[str, Any]]:
    if uses_fvg_entry(cfg) and protocol_fvg_open_done(sig, cfg):
        return {"skipped": True, "reason": "protocol_already_open", "details": [{"action": "traded"}]}
    return _live_open(sig, cfg)


def _daily_atr_for_scan(
    sym: str,
    cfg: OrbConfig,
    now_ms: int,
    daily_cache: Dict[str, Any],
) -> Optional[float]:
    if (cfg.sl_mode or "").strip().lower() != "atr_pct":
        return None
    if sym not in daily_cache:
        daily_cache[sym] = _load_daily_df(sym, cfg, now_ms=now_ms)
    ddf = daily_cache.get(sym)
    if ddf is None or getattr(ddf, "empty", True):
        return None
    from orb.core.indicators import daily_atr_asof

    return daily_atr_asof(ddf, int(now_ms), period=cfg.atr_period, tz=cfg.session_tz)


def _fvg_session_close_ms(cfg: OrbConfig, scan_ms: int) -> int:
    anchor = session_anchor_ms(
        int(scan_ms), tz=cfg.session_tz, session_open_time=cfg.session_open_time
    )
    close_ms = session_close_ms(anchor, tz=cfg.session_tz, session_close_time=cfg.session_close_time)
    if close_ms is None:
        close_ms = anchor + 6 * 60 * 60 * 1000
    return int(close_ms)


def _ensure_sig_session(sig: OrbSignal, session_day: str) -> None:
    if not (sig.session_date or "").strip():
        sig.session_date = session_day


def _fvg_zone_from_sig_json(raw: str) -> Optional[int]:
    try:
        d = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    zms = d.get("fvg_zone_form_ms")
    return int(zms) if zms is not None else None


def _fvg_watch_sig_json(sig: OrbSignal, *, zone_form_ms: Optional[int] = None) -> str:
    from dataclasses import asdict

    payload = asdict(sig)
    if zone_form_ms is not None:
        payload["fvg_zone_form_ms"] = int(zone_form_ms)
    return json.dumps(payload, default=str)


def _resolve_fvg_entry(
    sig: OrbSignal,
    *,
    sym: str,
    cfg: OrbConfig,
    now_ms: int,
    confirm_scan_ms: int,
    df5_cache: Dict[str, Any],
    df1_cache: Dict[str, Any],
    daily_cache: Optional[Dict[str, Any]] = None,
    df5_fvg_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[OrbSignal], str, Optional[Any]]:
    fvg5 = df5_fvg_cache if df5_fvg_cache is not None else df5_cache
    if sym not in df1_cache:
        df1_cache[sym] = _load_1m_df(sym, cfg, now_ms=now_ms)
    if sym not in fvg5:
        fvg5[sym] = _load_signal_df(sym, cfg, now_ms=now_ms)
    daily_atr = None
    if daily_cache is not None:
        daily_atr = _daily_atr_for_scan(sym, cfg, now_ms, daily_cache)
    fill_sig, reason, zone = find_fvg_limit_entry(
        sig,
        df1_cache[sym],
        fvg5[sym],
        scan_ms=int(confirm_scan_ms),
        close_ms=_fvg_session_close_ms(cfg, confirm_scan_ms),
        bar=cfg.bar_step_ms(),
        cfg=cfg,
        asof_ms=now_ms,
        daily_atr=daily_atr,
    )
    return fill_sig, reason, zone


def run_scan_conn_v2(conn, *, do_resolve: bool = True, cfg: Optional[OrbV2Config] = None) -> Dict[str, Any]:
    v2 = cfg or OrbV2Config.from_env()
    c = v2.base
    if not v2.enabled:
        return {"ok": True, "lane": v2.lane, "skipped": True, "reason": "orb_v2_disabled"}

    now_utc = _utc_now()
    session_day = _session_date_now(c)
    syms = v2.symbol_list()
    if not syms:
        return {
            "ok": True,
            "lane": v2.lane,
            "skipped": True,
            "reason": "orb_v2_no_symbols",
            "symbols_file": str(v2.symbols_file),
        }
    gate = v2.load_gate()
    ml_enabled = v2.gate_ml_enabled()
    model = _load_model(v2) if ml_enabled else None
    robot_count = resolve_robot_pool_size(gate=gate, symbol_count=len(syms))
    robot_bound = robot_bound_mode(symbol_count=len(syms), robot_count=robot_count)
    if robot_bound:
        use_robots = True
    else:
        use_robots = bool(gate.robot_reuse_after_exit)
    robot_init = robot_equity_from_env()

    stats: Dict[str, Any] = {
        "ok": True,
        "lane": v2.lane,
        "ran_at_utc": now_utc,
        "symbols": syms,
        "symbols_file": str(v2.symbols_file),
        "symbols_source": "orb_v2",
        "written": 0,
        "skipped": [],
        "opens": [],
        "gate_skips": [],
        "live": [],
        "shadow": v2.shadow,
        "ml_ranker": model.kind if model else ("disabled" if not ml_enabled else None),
        "robot_bound": robot_bound,
        "sizing": "robot_bound" if robot_bound else ("eight_robots" if use_robots else "per_symbol"),
        "gate": {
            "ml_enabled": ml_enabled,
            "min_p_true": gate.min_p_true,
            "min_breakout_score": gate.min_breakout_score,
            "max_opens_per_day": gate.max_opens_per_day,
            "robot_reuse_after_exit": gate.robot_reuse_after_exit,
            "day_abort_enabled": gate.day_abort_enabled,
        },
    }

    if ml_enabled and model is None:
        stats["ok"] = False
        stats["skipped"] = True
        stats["reason"] = "ml_model_missing"
        from orb.ml.live_bundle import resolve_live_gbm_path

        gbm_p = resolve_live_gbm_path()
        logger.error("[orb_v2] ML model not found: gbm=%s exists=%s", gbm_p, gbm_p.is_file())
        return stats
    if model is not None:
        stats["ml_model"] = model.status()

    conn.row_factory = __import__("sqlite3").Row
    migrate_orb_tables(conn.cursor())
    migrate_orb_v2_tables(conn.cursor())
    conn.commit()
    cur = conn.cursor()
    now_ms = int(time.time() * 1000)
    bot_equity = c.per_symbol_bot_equity()

    robot_wallets: Optional[List[float]] = None
    signal_equity = bot_equity
    if use_robots:
        ensure_orb_robots(cur, count=robot_count, initial_equity_usdt=robot_init)
        robot_wallets = list_robot_wallet_balances(conn, count=robot_count, initial_equity_usdt=robot_init)
        signal_equity = robot_equity_for_signals(robot_wallets, c)
        stats["robot_wallets"] = {f"R{i + 1}": round(w, 2) for i, w in enumerate(robot_wallets)}
        if robot_bound:
            stats["robot_bindings"] = robot_symbol_bindings(syms)
    else:
        ensure_symbol_bots(cur, syms, initial_equity_usdt=bot_equity)
    conn.commit()

    if c.macro_filter:
        macro_meta = macro_calendar_status()
        stats["macro_calendar"] = macro_meta
        logger.info(
            "[orb_v2] macro filter on: total=%s fomc_live=%s(%s) cpi_live=%s(%s) cache_age_s=%s",
            macro_meta["total_dates"],
            macro_meta["fomc_live"],
            macro_meta["fomc_live_count"],
            macro_meta["cpi_live"],
            macro_meta["cpi_live_count"],
            macro_meta["cache_age_seconds"],
        )
        if is_macro_skip_day(session_day):
            logger.info("[orb_v2] macro skip day %s — new entries blocked in signal layer", session_day)

    idle_reason = _idle_scan_skip_reason(c, cur, now_ms=now_ms)
    if idle_reason:
        logger.info("[orb_v2] idle skip: %s", idle_reason)
        return {**stats, "skipped": True, "reason": idle_reason, "opens": []}

    if not in_regular_session(c, now_ms=now_ms):
        if not do_resolve:
            return {**stats, "skipped": True, "reason": "outside_regular_session_resolve_disabled"}
        resolve_pre = resolve_open_positions(conn, cfg=c, now_ms=now_ms)
        stats["live"] = list(resolve_pre.get("live") or [])
        stats["mode"] = "resolve_only"
        stats["reason"] = "outside_regular_session_has_open_positions"
        stats["resolve_pre"] = resolve_pre
        _record_v2_run(
            cur,
            now_utc=now_utc,
            symbols_scanned=0,
            opens=0,
            gate_skips=0,
            detail={"stats": stats},
        )
        conn.commit()
        return stats

    scan_params = _scan_params_v2(
        c,
        gate=gate,
        model=model,
        ml_enabled=ml_enabled,
        shadow=v2.shadow,
        use_robots=use_robots,
        robot_bound=robot_bound,
        robot_count=robot_count,
        robot_equity=robot_init,
    )
    if c.live_enabled and live_enabled(c):
        synced = sync_live_pending_entries(conn, c)
        if synced:
            stats["live_pending_synced"] = synced
    resolve_pre = resolve_open_positions(conn, cfg=c, now_ms=now_ms) if do_resolve else {}
    stats["live"].extend(resolve_pre.get("live") or [])

    if use_robots and robot_wallets is not None:
        robot_wallets = list_robot_wallet_balances(conn, count=robot_count, initial_equity_usdt=robot_init)
        signal_equity = robot_equity_for_signals(robot_wallets, c)

    gate_state = load_gate_day_state(cur, session_day, v2)
    daily_cache: Dict[str, Any] = {}
    df5_cache: Dict[str, Any] = {}
    df5_fvg_cache: Dict[str, Any] = {}
    df1_cache: Dict[str, Any] = {}
    fvg_watched: set[str] = set()
    if uses_fvg_entry(c):
        stats["entry_fill"] = c.entry_fill
        fvg_watched = {str(w["symbol"]).strip().upper() for w in list_fvg_watches(cur, session_day)}
        stats["fvg_watches"] = len(fvg_watched)
        for watch in list_fvg_watches(cur, session_day):
            wsym = str(watch["symbol"]).strip().upper()
            wsig = _orb_signal_from_json(str(watch.get("sig_json") or "{}"))
            wsig.symbol = wsym
            _ensure_sig_session(wsig, session_day)
            if v2_session_traded(cur, wsym, session_day, v2):
                _cancel_fvg_watch_limit(wsig, watch, c, reason="session_traded")
                delete_fvg_watch(cur, session_day, wsym)
                fvg_watched.discard(wsym)
                continue
            if fetch_open_hold(cur, wsym, default_notional=c.default_paper_notional()) is not None:
                _cancel_fvg_watch_limit(wsig, watch, c, reason="open_hold_exists")
                delete_fvg_watch(cur, session_day, wsym)
                fvg_watched.discard(wsym)
                continue
            fill_sig, fvg_reason, zone = _resolve_fvg_entry(
                wsig,
                sym=wsym,
                cfg=c,
                now_ms=now_ms,
                confirm_scan_ms=int(watch.get("confirm_scan_ms") or now_ms),
                df5_cache=df5_cache,
                df5_fvg_cache=df5_fvg_cache,
                df1_cache=df1_cache,
                daily_cache=daily_cache,
            )
            if fvg_reason != "ok" and protocol_fvg_open_status(wsig, c) == "traded":
                daily_atr = _daily_atr_for_scan(wsym, c, now_ms, daily_cache)
                proto_px = protocol_fvg_entry_price(wsig, c)
                synth = synthesize_fvg_fill_from_protocol(
                    wsig,
                    c,
                    now_ms=now_ms,
                    quote=fill_sig,
                    daily_atr=daily_atr,
                    protocol_entry_px=proto_px,
                )
                if synth is not None:
                    fill_sig = synth
                    fvg_reason = "ok"
            if fvg_reason == "fvg_pending":
                # 实盘：fvg_pending 仅表示本档 scan 尚未确认成交，LIMIT 应继续在交易所挂着。
                # 仅在 5m 收盘明确 OR reclaim 且 Protocol 未成交时撤单（突破失效）。
                if (
                    str(watch.get("reason") or "") == "fvg_limit_pending"
                    and c.live_enabled
                    and live_enabled(c)
                    and float(wsig.or_high or 0) > 0
                    and float(wsig.or_low or 0) > 0
                ):
                    confirm_ms = int(watch.get("confirm_scan_ms") or now_ms)
                    if wsym not in df5_fvg_cache:
                        df5_fvg_cache[wsym] = _load_signal_df(wsym, c, now_ms=now_ms)
                    df5 = df5_fvg_cache.get(wsym) or df5_cache.get(wsym)
                    reclaim_ms = (
                        first_or_reclaim_bar_ms(
                            df5,
                            after_ms=confirm_ms,
                            before_ms=now_ms,
                            or_high=float(wsig.or_high),
                            or_low=float(wsig.or_low),
                        )
                        if df5 is not None
                        else None
                    )
                    if reclaim_ms is not None and protocol_fvg_open_status(wsig, c) != "traded":
                        cancel_fvg_live_limit(wsig, c, reason="fvg_or_reclaim")
                        delete_fvg_watch(cur, session_day, wsym)
                        fvg_watched.discard(wsym)
                        mark_breakout_seen(
                            cur,
                            session_date=session_day,
                            symbol=wsym,
                            now_utc=now_utc,
                            scan_open_ms=now_ms,
                            p_true=float(watch.get("p_true") or 0),
                            opened=False,
                            reason="fvg_or_reclaim",
                        )
                        stats["skipped"].append(
                            {"symbol": wsym, "reason": "fvg_or_reclaim", "source": "fvg_watch"}
                        )
                        logger.info(
                            "[orb_v2] fvg_or_reclaim %s — cancelled live LIMIT (5m close back in OR)",
                            wsym,
                        )
                continue
            if fvg_reason == "fvg_limit_pending":
                quote = fill_sig
                stored_zone_ms = _fvg_zone_from_sig_json(str(watch.get("sig_json") or "{}"))
                zone_form_ms = int(zone.form_bar_open_ms) if zone is not None else None
                if (
                    quote is not None
                    and stored_zone_ms is not None
                    and zone_form_ms is not None
                    and zone_form_ms != stored_zone_ms
                ):
                    cancel_fvg_live_limit(wsig, c, reason="fvg_zone_advance")
                if quote is not None:
                    _ensure_sig_session(quote, session_day)
                    quote.symbol = wsym
                    _preview_fvg_live_notional(
                        quote,
                        conn,
                        c,
                        use_robots=use_robots,
                        robot_count=robot_count,
                        robot_init=robot_init,
                        bot_equity=bot_equity,
                        signal_equity=signal_equity,
                    )
                    _submit_fvg_live_limit(quote, c, sym=wsym, stats=stats)
                upsert_fvg_watch(
                    cur,
                    session_date=session_day,
                    symbol=wsym,
                    now_utc=now_utc,
                    confirm_scan_ms=int(watch.get("confirm_scan_ms") or now_ms),
                    sig_json=_fvg_watch_sig_json(wsig, zone_form_ms=zone_form_ms),
                    p_true=float(watch.get("p_true") or 0),
                    sync_n=int(watch.get("sync_n") or 0),
                    breakout_score=watch.get("breakout_score"),
                    reason="fvg_limit_pending",
                )
                continue
            delete_fvg_watch(cur, session_day, wsym)
            fvg_watched.discard(wsym)
            if fill_sig is None:
                _cancel_fvg_watch_limit(wsig, watch, c, reason=str(fvg_reason or "fvg_watch_end"))
                mark_breakout_seen(
                    cur,
                    session_date=session_day,
                    symbol=wsym,
                    now_utc=now_utc,
                    scan_open_ms=now_ms,
                    p_true=float(watch.get("p_true") or 0),
                    opened=False,
                    reason=fvg_reason,
                )
                stats["skipped"].append({"symbol": wsym, "reason": fvg_reason, "source": "fvg_watch"})
                continue
            _merge_fvg_fill(wsig, fill_sig)
            if use_robots and not robot_bound and len(busy_robot_ids(cur)) >= gate.max_opens_per_day:
                upsert_fvg_watch(
                    cur,
                    session_date=session_day,
                    symbol=wsym,
                    now_utc=now_utc,
                    confirm_scan_ms=int(watch.get("confirm_scan_ms") or now_ms),
                    sig_json=_orb_signal_to_json(wsig),
                    p_true=float(watch.get("p_true") or 0),
                    sync_n=int(watch.get("sync_n") or 0),
                    breakout_score=watch.get("breakout_score"),
                    reason="fvg_fill_no_slot",
                )
                fvg_watched.add(wsym)
                continue
            gate_state.opens += 1
            try:
                assigned_robot: Optional[int] = None
                wdecision = {
                    "p_true": watch.get("p_true"),
                    "breakout_score": watch.get("breakout_score"),
                }
                if use_robots:
                    if robot_bound:
                        assigned_robot = bound_robot_id_for_open(
                            cur, wsym, syms, initial_equity_usdt=robot_init
                        )
                    else:
                        assigned_robot = next_free_robot_id(
                            cur, count=robot_count, initial_equity_usdt=robot_init
                        )
                    if assigned_robot is None:
                        gate_state.opens = max(0, gate_state.opens - 1)
                        upsert_fvg_watch(
                            cur,
                            session_date=session_day,
                            symbol=wsym,
                            now_utc=now_utc,
                            confirm_scan_ms=int(watch.get("confirm_scan_ms") or now_ms),
                            sig_json=_orb_signal_to_json(wsig),
                            p_true=float(watch.get("p_true") or 0),
                            sync_n=int(watch.get("sync_n") or 0),
                            breakout_score=watch.get("breakout_score"),
                            reason="fvg_fill_no_robot",
                        )
                        fvg_watched.add(wsym)
                        continue
                    rw = robot_wallet_balance(
                        conn, assigned_robot, initial_equity_usdt=robot_init, sync=False
                    )
                    _apply_robot_notional(
                        wsig,
                        entry=float(wsig.price),
                        sl=float(wsig.sl_price),
                        cfg=c,
                        bot_equity=rw,
                    )
                else:
                    bot_wallet = symbol_bot_wallet_balance(
                        conn, wsym, initial_equity_usdt=bot_equity, sync=False
                    )
                    _apply_robot_notional(
                        wsig,
                        entry=float(wsig.price),
                        sl=float(wsig.sl_price),
                        cfg=c,
                        bot_equity=bot_wallet,
                    )
                mark_breakout_seen(
                    cur,
                    session_date=session_day,
                    symbol=wsym,
                    now_utc=now_utc,
                    scan_open_ms=now_ms,
                    p_true=float(watch.get("p_true") or 0),
                    opened=True,
                    reason="fvg_fill_ok",
                )
                _upsert_signal(
                    cur,
                    ts=now_utc,
                    sig=wsig,
                    scan_params=scan_params,
                    cfg=c,
                    robot_id=assigned_robot,
                )
                stats["written"] += 1
                open_row = {
                    "symbol": wsym,
                    "side": wsig.side,
                    "entry": wsig.price,
                    "sl": wsig.sl_price,
                    "tp": wsig.tp_price,
                    "p_true": wdecision.get("p_true"),
                    "breakout_score": wdecision.get("breakout_score"),
                    "notional_usdt": wsig.paper_notional_usdt,
                    "entry_mode": "fvg_prox",
                }
                if assigned_robot is not None:
                    open_row["robot_id"] = assigned_robot
                stats["opens"].append(open_row)
                logger.info(
                    "[orb_v2] fvg watch fill open %s %s entry=%.4f robot=%s",
                    wsym,
                    wsig.side,
                    float(wsig.price),
                    assigned_robot,
                )
                live_open = _live_open_after_fvg(wsig, c)
                if live_open is not None:
                    stats["live"].append({"action": "open", "symbol": wsym, "result": live_open})
                if c.live_enabled and not live_ingest_succeeded(live_open):
                    _rollback_failed_live_open(
                        cur,
                        session_day=session_day,
                        sym=wsym,
                        gate_state=gate_state,
                        stats=stats,
                        live_open=live_open,
                    )
                elif c.live_enabled and live_open_is_pending(live_open):
                    cur.execute(
                        """
                        UPDATE orb_signals SET notes=?
                        WHERE symbol=? AND outcome IS NULL AND side IN ('LONG', 'SHORT')
                        """,
                        (_pending_note_for_open(wsig, c), wsym),
                    )
            except Exception as exc:
                gate_state.opens = max(0, gate_state.opens - 1)
                stats["skipped"].append(
                    {"symbol": wsym, "reason": "fvg_watch_open_error", "error": str(exc)}
                )
                logger.warning("[orb_v2] fvg watch open %s failed: %s", wsym, exc)
    if (c.sl_mode or "").strip().lower() == "atr_pct":
        for sym in syms:
            if c.one_trade_per_session and v2_session_traded(cur, sym, session_day, v2):
                continue
            daily_cache[sym] = _load_daily_df(sym, c, now_ms=now_ms)

    candidates: List[Tuple[str, OrbSignal]] = []
    for sym in syms:
        try:
            if not symbol_bot_enabled(cur, sym):
                stats["skipped"].append({"symbol": sym, "reason": "bot_disabled"})
                continue
            if c.one_trade_per_session and v2_session_traded(cur, sym, session_day, v2):
                stats["skipped"].append({"symbol": sym, "reason": "session_traded"})
                continue
            if use_robots and robot_bound:
                bound_rid = symbol_to_robot_id(sym, syms)
                if bound_rid is None:
                    stats["skipped"].append({"symbol": sym, "reason": "no_robot_binding"})
                    continue
                bot_wallet = robot_wallet_balance(
                    conn, bound_rid, initial_equity_usdt=robot_init, sync=False
                )
                if bot_wallet <= 0:
                    stats["skipped"].append({"symbol": sym, "reason": "robot_wallet_depleted", "robot_id": bound_rid})
                    continue
            elif use_robots:
                if signal_equity <= 0:
                    stats["skipped"].append({"symbol": sym, "reason": "robot_pool_depleted"})
                    continue
                bot_wallet = signal_equity
            else:
                bot_wallet = symbol_bot_wallet_balance(
                    conn, sym, initial_equity_usdt=bot_equity, sync=True
                )
                if bot_wallet <= 0:
                    stats["skipped"].append({"symbol": sym, "reason": "bot_wallet_depleted"})
                    continue
            if sym not in df5_cache:
                df5_cache[sym] = _load_signal_df(sym, c, now_ms=now_ms)
            sig = analyze_at_ms(
                sym,
                cfg=c,
                now_ms=now_ms,
                session_traded=False,
                daily_df=daily_cache.get(sym),
                bot_equity_usdt=bot_wallet,
                df5=df5_cache[sym],
            )
            if not is_actionable(sig, c):
                continue
            _ensure_sig_session(sig, session_day)
            sig.symbol = sym
            candidates.append((sym, sig))
        except Exception as exc:
            logger.warning("[orb_v2] candidate %s failed: %s", sym, exc)
            stats["skipped"].append({"symbol": sym, "reason": "error", "error": str(exc)})

    sync_by_sym: Dict[str, int] = {}
    for sym, sig in candidates:
        side = str(sig.side)
        sync_by_sym[sym] = sum(1 for s2, g2 in candidates if s2 != sym and str(g2.side) == side)

    scored: List[Tuple[float, str, OrbSignal, int, Dict[str, float]]] = []
    for sym, sig in candidates:
        sync_n = int(sync_by_sym.get(sym, 0))
        feat = extract_features(sig, c, sync_same_side=sync_n)
        if ml_enabled and model is not None:
            p_true = float(model.predict_true(feat, symbol=sym))
        else:
            p_true = 1.0
        scored.append((p_true, sym, sig, sync_n, feat))
    scored.sort(key=lambda x: x[0], reverse=True)

    gate_skips = 0
    need_breakout_score = ml_enabled and float(gate.min_breakout_score or 0) > 0
    if need_breakout_score and candidates:
        logger.info(
            "[orb_v2] breakout score filter on min_bs=%.0f candidates=%d session=%s",
            gate.min_breakout_score,
            len(candidates),
            session_day,
        )
    for p_true, sym, sig, sync_n, feat in scored:
        if sym in fvg_watched:
            continue
        if use_robots and not robot_bound:
            if len(busy_robot_ids(cur)) >= gate.max_opens_per_day:
                break
        elif not use_robots and gate_state.opens >= gate.max_opens_per_day:
            break

        breakout_score: Optional[float] = None
        if need_breakout_score:
            breakout_score = _paper_breakout_score(
                sym,
                sig,
                c,
                session_day=session_day,
                now_ms=now_ms,
                df5_cache=df5_cache,
            )

        if v2.shadow:
            from orb.ml.gate import record_scored_signal, should_open

            record_scored_signal(gate_state, p_true=p_true, gate=gate)
            gate_pass, reason = should_open(
                p_true=p_true,
                symbol=sym,
                feat=feat,
                sync=sync_n,
                state=gate_state,
                gate=gate,
                profiles=model.ranker.profiles if ml_enabled and model is not None else {},
                breakout_score=breakout_score,
            )
            decision = {
                "symbol": sym,
                "p_true": p_true,
                "p_fake": float(model.predict_fake(feat, symbol=sym)) if ml_enabled and model is not None else 0.0,
                "sync_same_side": sync_n,
                "minutes_after_or": round(float(feat.get("minutes_after_or", 0) or 0), 1),
                "opened": gate_pass,
                "reason": reason,
            }
            if breakout_score is not None:
                decision["breakout_score"] = breakout_score
        elif ml_enabled and model is not None:
            decision = evaluate_open_decision(
                model.ranker,
                symbol=sym,
                feat=feat,
                sync=sync_n,
                state=gate_state,
                gate=gate,
                p_true=p_true,
                p_fake=float(model.predict_fake(feat, symbol=sym)),
                breakout_score=breakout_score,
            )
        else:
            decision = evaluate_open_decision_without_ml(
                symbol=sym,
                feat=feat,
                sync=sync_n,
                state=gate_state,
                gate=gate,
                breakout_score=breakout_score,
            )

        gate_pass = bool(decision.get("opened"))
        reason = str(decision.get("reason") or "")

        if not gate_pass and not v2.shadow:
            logger.info(
                "[orb_v2] gate skip %s %s p=%.3f bs=%s min_bs=%s sync=%d reason=%s",
                sym,
                sig.side,
                float(decision.get("p_true") or 0),
                decision.get("breakout_score"),
                f"{gate.min_breakout_score:.0f}" if need_breakout_score else "off",
                sync_n,
                reason,
            )
            gate_skips += 1
            mark_breakout_seen(
                cur,
                session_date=session_day,
                symbol=sym,
                now_utc=now_utc,
                scan_open_ms=now_ms,
                p_true=float(decision.get("p_true") or 0),
                opened=False,
                reason=reason,
            )
            stats["gate_skips"].append(
                {
                    "symbol": sym,
                    "p_true": decision.get("p_true"),
                    "breakout_score": decision.get("breakout_score"),
                    "reason": reason,
                    "sync": sync_n,
                    "minutes_after_or": decision.get("minutes_after_or"),
                }
            )
            continue

        try:
            if gate_pass and not v2.shadow and uses_fvg_entry(c):
                _ensure_sig_session(sig, session_day)
                sig.symbol = sym
                fill_sig, fvg_reason, zone = _resolve_fvg_entry(
                    sig,
                    sym=sym,
                    cfg=c,
                    now_ms=now_ms,
                    confirm_scan_ms=now_ms,
                    df5_cache=df5_cache,
                    df5_fvg_cache=df5_fvg_cache,
                    df1_cache=df1_cache,
                    daily_cache=daily_cache,
                )
                if fvg_reason == "fvg_pending":
                    rollback_open_decision(gate_state, symbol=sym)
                    upsert_fvg_watch(
                        cur,
                        session_date=session_day,
                        symbol=sym,
                        now_utc=now_utc,
                        confirm_scan_ms=now_ms,
                        sig_json=_orb_signal_to_json(sig),
                        p_true=float(decision.get("p_true") or 0),
                        sync_n=sync_n,
                        breakout_score=decision.get("breakout_score"),
                        reason="fvg_pending",
                    )
                    fvg_watched.add(sym)
                    mark_breakout_seen(
                        cur,
                        session_date=session_day,
                        symbol=sym,
                        now_utc=now_utc,
                        scan_open_ms=now_ms,
                        p_true=float(decision.get("p_true") or 0),
                        opened=False,
                        reason="fvg_pending",
                    )
                    stats["gate_skips"].append(
                        {
                            "symbol": sym,
                            "p_true": decision.get("p_true"),
                            "breakout_score": decision.get("breakout_score"),
                            "reason": "fvg_pending",
                            "sync": sync_n,
                        }
                    )
                    gate_skips += 1
                    logger.info("[orb_v2] fvg pending %s %s — waiting for zone", sym, sig.side)
                    continue
                if fvg_reason == "fvg_limit_pending" and fill_sig is not None:
                    rollback_open_decision(gate_state, symbol=sym)
                    quote = fill_sig
                    _ensure_sig_session(quote, session_day)
                    quote.symbol = sym
                    zone_form_ms = int(zone.form_bar_open_ms) if zone is not None else None
                    _preview_fvg_live_notional(
                        quote,
                        conn,
                        c,
                        use_robots=use_robots,
                        robot_count=robot_count,
                        robot_init=robot_init,
                        bot_equity=bot_equity,
                        signal_equity=signal_equity,
                    )
                    _submit_fvg_live_limit(quote, c, sym=sym, stats=stats)
                    upsert_fvg_watch(
                        cur,
                        session_date=session_day,
                        symbol=sym,
                        now_utc=now_utc,
                        confirm_scan_ms=now_ms,
                        sig_json=_fvg_watch_sig_json(sig, zone_form_ms=zone_form_ms),
                        p_true=float(decision.get("p_true") or 0),
                        sync_n=sync_n,
                        breakout_score=decision.get("breakout_score"),
                        reason="fvg_limit_pending",
                    )
                    fvg_watched.add(sym)
                    mark_breakout_seen(
                        cur,
                        session_date=session_day,
                        symbol=sym,
                        now_utc=now_utc,
                        scan_open_ms=now_ms,
                        p_true=float(decision.get("p_true") or 0),
                        opened=False,
                        reason="fvg_limit_pending",
                    )
                    stats["gate_skips"].append(
                        {
                            "symbol": sym,
                            "p_true": decision.get("p_true"),
                            "breakout_score": decision.get("breakout_score"),
                            "reason": "fvg_limit_pending",
                            "sync": sync_n,
                            "limit_px": quote.price,
                        }
                    )
                    gate_skips += 1
                    logger.info(
                        "[orb_v2] fvg limit pending %s %s @ %.4f — protocol LIMIT",
                        sym,
                        sig.side,
                        float(quote.price),
                    )
                    continue
                if fill_sig is None:
                    rollback_open_decision(gate_state, symbol=sym)
                    mark_breakout_seen(
                        cur,
                        session_date=session_day,
                        symbol=sym,
                        now_utc=now_utc,
                        scan_open_ms=now_ms,
                        p_true=float(decision.get("p_true") or 0),
                        opened=False,
                        reason=fvg_reason,
                    )
                    stats["skipped"].append({"symbol": sym, "reason": fvg_reason})
                    continue
                signal_entry_px = float(sig.price)
                _merge_fvg_fill(sig, fill_sig)
                logger.info(
                    "[orb_v2] fvg fill %s %s entry=%.4f (signal=%.4f)",
                    sym,
                    sig.side,
                    float(sig.price),
                    signal_entry_px,
                )

            open_persisted = False
            open_marked = False
            hold = fetch_open_hold(cur, sym, default_notional=c.default_paper_notional())
            if hold is not None:
                if gate_pass and not v2.shadow:
                    rollback_open_decision(gate_state, symbol=sym)
                reason = "same_side_open" if str(hold["side"]) == sig.side else "open_hold_exists"
                mark_breakout_seen(
                    cur,
                    session_date=session_day,
                    symbol=sym,
                    now_utc=now_utc,
                    scan_open_ms=now_ms,
                    p_true=float(decision.get("p_true") or 0),
                    opened=False,
                    reason=reason,
                )
                stats["skipped"].append({"symbol": sym, "reason": reason})
                continue

            if c.max_open_positions > 0 and count_open_positions(cur) >= c.max_open_positions:
                if gate_pass and not v2.shadow:
                    rollback_open_decision(gate_state, symbol=sym)
                mark_breakout_seen(
                    cur,
                    session_date=session_day,
                    symbol=sym,
                    now_utc=now_utc,
                    scan_open_ms=now_ms,
                    p_true=float(decision.get("p_true") or 0),
                    opened=False,
                    reason="max_open_cap",
                )
                stats["skipped"].append({"symbol": sym, "reason": "max_open_cap"})
                continue

            assigned_robot: Optional[int] = None
            if gate_pass and use_robots and not v2.shadow:
                if robot_bound:
                    assigned_robot = bound_robot_id_for_open(
                        cur, sym, syms, initial_equity_usdt=robot_init
                    )
                    if assigned_robot is None:
                        rollback_open_decision(gate_state, symbol=sym)
                        busy = symbol_to_robot_id(sym, syms)
                        reason = "robot_busy" if busy and busy in busy_robot_ids(cur) else "no_robot_slot"
                        mark_breakout_seen(
                            cur,
                            session_date=session_day,
                            symbol=sym,
                            now_utc=now_utc,
                            scan_open_ms=now_ms,
                            p_true=float(decision.get("p_true") or 0),
                            opened=False,
                            reason=reason,
                        )
                        stats["skipped"].append({"symbol": sym, "reason": reason, "robot_id": busy})
                        continue
                else:
                    assigned_robot = next_free_robot_id(
                        cur, count=robot_count, initial_equity_usdt=robot_init
                    )
                    if assigned_robot is None:
                        rollback_open_decision(gate_state, symbol=sym)
                        mark_breakout_seen(
                            cur,
                            session_date=session_day,
                            symbol=sym,
                            now_utc=now_utc,
                            scan_open_ms=now_ms,
                            p_true=float(decision.get("p_true") or 0),
                            opened=False,
                            reason="no_robot_slot",
                        )
                        stats["skipped"].append({"symbol": sym, "reason": "no_robot_slot"})
                        continue
                rw = robot_wallet_balance(
                    conn, assigned_robot, initial_equity_usdt=robot_init, sync=False
                )
                _apply_robot_notional(
                    sig,
                    entry=float(sig.price),
                    sl=float(sig.sl_price),
                    cfg=c,
                    bot_equity=rw,
                )
            elif gate_pass and not use_robots and not v2.shadow:
                bot_wallet = symbol_bot_wallet_balance(
                    conn, sym, initial_equity_usdt=bot_equity, sync=False
                )
                _apply_robot_notional(
                    sig,
                    entry=float(sig.price),
                    sl=float(sig.sl_price),
                    cfg=c,
                    bot_equity=bot_wallet,
                )

            if v2.shadow:
                mark_breakout_seen(
                    cur,
                    session_date=session_day,
                    symbol=sym,
                    now_utc=now_utc,
                    scan_open_ms=now_ms,
                    p_true=float(decision.get("p_true") or 0),
                    opened=False,
                    reason="shadow_pass",
                )
                stats["gate_skips"].append(
                    {
                        "symbol": sym,
                        "p_true": decision.get("p_true"),
                        "reason": "shadow_would_open",
                        "side": sig.side,
                        "entry": sig.price,
                    }
                )
                gate_skips += 1
                continue

            mark_breakout_seen(
                cur,
                session_date=session_day,
                symbol=sym,
                now_utc=now_utc,
                scan_open_ms=now_ms,
                p_true=float(decision.get("p_true") or 0),
                opened=True,
                reason=reason or "open_ok",
            )
            open_marked = True
            _upsert_signal(
                cur,
                ts=now_utc,
                sig=sig,
                scan_params=scan_params,
                cfg=c,
                robot_id=assigned_robot,
            )
            open_persisted = True
            stats["written"] += 1
            open_row = {
                "symbol": sym,
                "side": sig.side,
                "entry": sig.price,
                "sl": sig.sl_price,
                "tp": sig.tp_price,
                "p_true": decision.get("p_true"),
                "breakout_score": decision.get("breakout_score"),
                "notional_usdt": sig.paper_notional_usdt,
            }
            if assigned_robot is not None:
                open_row["robot_id"] = assigned_robot
            if uses_fvg_entry(c):
                open_row["entry_mode"] = "fvg_prox"
            stats["opens"].append(open_row)
            logger.info(
                "[orb_v2] gate open %s %s p=%.3f bs=%s sync=%d robot=%s",
                sym,
                sig.side,
                float(decision.get("p_true") or 0),
                decision.get("breakout_score"),
                sync_n,
                assigned_robot,
            )
            live_open = _live_open_after_fvg(sig, c)
            if live_open is not None:
                stats["live"].append({"action": "open", "symbol": sym, "result": live_open})
            if c.live_enabled and not live_ingest_succeeded(live_open):
                _rollback_failed_live_open(
                    cur,
                    session_day=session_day,
                    sym=sym,
                    gate_state=gate_state,
                    stats=stats,
                    live_open=live_open,
                )
                continue
            if c.live_enabled and live_open_is_pending(live_open):
                cur.execute(
                    """
                    UPDATE orb_signals SET notes=?
                    WHERE symbol=? AND outcome IS NULL AND side IN ('LONG', 'SHORT')
                    """,
                    (_pending_note_for_open(sig, c), sym),
                )
                logger.info("[orb_v2] live pending entry %s — paper resolve deferred", sym)
        except Exception as exc:
            if gate_pass and not v2.shadow:
                if open_persisted or open_marked:
                    _rollback_failed_live_open(
                        cur,
                        session_day=session_day,
                        sym=sym,
                        gate_state=gate_state,
                        stats=stats,
                        live_open={"error": str(exc)},
                    )
                else:
                    rollback_open_decision(gate_state, symbol=sym)
                    stats["skipped"].append({"symbol": sym, "reason": "open_error", "error": str(exc)})
            else:
                stats["skipped"].append({"symbol": sym, "reason": "open_error", "error": str(exc)})
            logger.warning("[orb_v2] open %s failed: %s", sym, exc)

    if need_breakout_score:
        bs_skips = sum(
            1
            for row in stats.get("gate_skips") or []
            if str(row.get("reason") or "").startswith("breakout_score")
        )
        logger.info(
            "[orb_v2] scan summary session=%s candidates=%d opens=%d gate_skips=%d bs_skips=%d min_bs=%.0f",
            session_day,
            len(candidates),
            len(stats["opens"]),
            gate_skips,
            bs_skips,
            gate.min_breakout_score,
        )

    persist_gate_day_state(cur, session_day, gate_state)
    conn.commit()
    resolve_post = resolve_open_positions(conn, cfg=c, now_ms=now_ms) if do_resolve else {}
    stats["live"].extend(resolve_post.get("live") or [])
    if use_robots and robot_wallets is not None:
        stats["robot_wallets"] = {
            f"R{i + 1}": round(
                robot_wallet_balance(conn, i + 1, initial_equity_usdt=robot_init, sync=False),
                2,
            )
            for i in range(robot_count)
        }
    _record_v2_run(
        cur,
        now_utc=now_utc,
        symbols_scanned=len(syms),
        opens=len(stats["opens"]),
        gate_skips=gate_skips,
        detail={"stats": stats, "resolve_pre": resolve_pre, "resolve_post": resolve_post},
    )
    conn.commit()
    stats["resolve_pre"] = resolve_pre
    stats["resolve_post"] = resolve_post
    stats["gate_opens_today"] = gate_state.opens
    return stats


def run_scan_v2(*, do_resolve: bool = True) -> Dict[str, Any]:
    from accumulation_radar import init_db

    conn = init_db()
    try:
        return run_scan_conn_v2(conn, do_resolve=do_resolve)
    finally:
        conn.close()


def run_resolve_only_v2() -> Dict[str, Any]:
    from accumulation_radar import init_db

    v2 = OrbV2Config.from_env()
    conn = init_db()
    try:
        return resolve_open_positions(conn, cfg=v2.base)
    finally:
        conn.close()

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
    rollback_open_decision,
)
from orb.core.macro_calendar import is_macro_skip_day, macro_calendar_status
from orb.core.paper import (
    _idle_scan_skip_reason,
    _live_open,
    _load_daily_df,
    _scan_params,
    _session_date_now,
    _upsert_signal,
    analyze_at_ms,
    in_regular_session,
    is_actionable,
    resolve_open_positions,
)
from orb.v2.config import OrbV2Config
from orb.core.live_exec import live_ingest_succeeded
from orb.v2.db import mark_breakout_seen, migrate_orb_v2_tables, rollback_breakout_opened
from orb.v2.gate_state import load_gate_day_state, persist_gate_day_state, v2_session_traded
from orb.v2.robots import (
    busy_robot_ids,
    ensure_orb_robots,
    list_robot_wallet_balances,
    next_free_robot_id,
    robot_count_from_env,
    robot_equity_for_signals,
    robot_equity_from_env,
    robot_wallet_balance,
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
    model: BreakoutModelBundle,
    shadow: bool,
    use_robots: bool,
    robot_count: int,
    robot_equity: float,
) -> Dict[str, Any]:
    base = _scan_params(cfg)
    base.update(
        {
            "strategy": "orb_v2",
            "orb_version": 2,
            "ml_ranker": model.kind,
            "gate_min_p_true": gate.min_p_true,
            "gate_min_breakout_score": gate.min_breakout_score,
            "gate_max_opens": gate.max_opens_per_day,
            "gate_shadow": shadow,
            "sizing": "eight_robots" if use_robots else "per_symbol",
            "robot_count": robot_count if use_robots else None,
            "robot_equity_usdt": robot_equity if use_robots else None,
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
    from orb.core.breakout_score import breakout_kline_range_ms, breakout_score_for_signal
    from orb.core.kline_cache import load_klines

    if sym not in df5_cache:
        fetch_start, end_ms = breakout_kline_range_ms(session_day, cfg)
        df5_cache[sym] = load_klines(
            sym,
            cfg.signal_interval,
            start_ms=fetch_start,
            end_ms=end_ms,
        )
    df5 = df5_cache.get(sym)
    if df5 is None or getattr(df5, "empty", True):
        return None
    return round(breakout_score_for_signal(sig, df5, cfg, now_ms=now_ms), 2)


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
    model = _load_model(v2)
    use_robots = bool(gate.robot_reuse_after_exit)
    robot_count = robot_count_from_env()
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
        "ml_ranker": model.kind if model else None,
        "sizing": "eight_robots" if use_robots else "per_symbol",
        "gate": {
            "min_p_true": gate.min_p_true,
            "min_breakout_score": gate.min_breakout_score,
            "max_opens_per_day": gate.max_opens_per_day,
            "robot_reuse_after_exit": gate.robot_reuse_after_exit,
            "day_abort_enabled": gate.day_abort_enabled,
        },
    }

    if model is None:
        stats["ok"] = False
        stats["skipped"] = True
        stats["reason"] = "ml_model_missing"
        from orb.ml.live_bundle import resolve_live_gbm_path

        gbm_p = resolve_live_gbm_path()
        logger.error("[orb_v2] ML model not found: gbm=%s exists=%s", gbm_p, gbm_p.is_file())
        return stats
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
        shadow=v2.shadow,
        use_robots=use_robots,
        robot_count=robot_count,
        robot_equity=robot_init,
    )
    resolve_pre = resolve_open_positions(conn, cfg=c, now_ms=now_ms) if do_resolve else {}
    stats["live"].extend(resolve_pre.get("live") or [])

    if use_robots and robot_wallets is not None:
        robot_wallets = list_robot_wallet_balances(conn, count=robot_count, initial_equity_usdt=robot_init)
        signal_equity = robot_equity_for_signals(robot_wallets, c)

    gate_state = load_gate_day_state(cur, session_day, v2)
    daily_cache: Dict[str, Any] = {}
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
            if use_robots:
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
            sig = analyze_at_ms(
                sym,
                cfg=c,
                now_ms=now_ms,
                session_traded=False,
                daily_df=daily_cache.get(sym),
                bot_equity_usdt=bot_wallet,
            )
            if not is_actionable(sig, c):
                continue
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
        p_true = float(model.predict_true(feat, symbol=sym))
        scored.append((p_true, sym, sig, sync_n, feat))
    scored.sort(key=lambda x: x[0], reverse=True)

    gate_skips = 0
    need_breakout_score = float(gate.min_breakout_score or 0) > 0
    df5_cache: Dict[str, Any] = {}
    for p_true, sym, sig, sync_n, feat in scored:
        if use_robots:
            if len(busy_robot_ids(cur)) >= gate.max_opens_per_day:
                break
        elif gate_state.opens >= gate.max_opens_per_day:
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
                profiles=model.ranker.profiles,
                breakout_score=breakout_score,
            )
            decision = {
                "symbol": sym,
                "p_true": p_true,
                "p_fake": model.predict_fake(feat, symbol=sym),
                "sync_same_side": sync_n,
                "minutes_after_or": round(float(feat.get("minutes_after_or", 0) or 0), 1),
                "opened": gate_pass,
                "reason": reason,
            }
            if breakout_score is not None:
                decision["breakout_score"] = breakout_score
        else:
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

        gate_pass = bool(decision.get("opened"))
        reason = str(decision.get("reason") or "")

        if not gate_pass and not v2.shadow:
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

            if not use_robots and c.max_open_positions > 0 and count_open_positions(cur) >= c.max_open_positions:
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
            stats["opens"].append(open_row)
            live_open = _live_open(sig, c)
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

    persist_gate_day_state(cur, session_day, gate_state)
    conn.commit()
    resolve_post = resolve_open_positions(conn, cfg=c, now_ms=now_ms) if do_resolve else {}
    stats["live"].extend(resolve_post.get("live") or [])
    stats["robot_resets"] = resolve_post.get("robot_resets") or []
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

"""接针策略（独立 lane）— 热度+OI 标的池纸面扫描 + 移动止盈止损。"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import jiezhen_config as cfg
from binance_fapi import fetch_klines, fetch_mark_price
from jiezhen_db import (
    archive_settlement,
    count_open,
    fetch_all_open,
    fetch_open_by_symbol_side,
    last_close_info,
    migrate_jz_tables,
    peak_profit_from_row,
)
from jiezhen_logic import build_spike_plan
from jiezhen_signals import resolve_jiezhen_universe
from momentum_trail import TIER_LABELS, evaluate_trail

logger = logging.getLogger(__name__)

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")


def _verbose_log() -> bool:
    return cfg.JIEZHEN_VERBOSE_LOG


def _log_jz(msg: str, *args: Any) -> None:
    logger.info("[jz] " + msg, *args)


def _log_jz_trail(msg: str, *args: Any) -> None:
    logger.info("[jz-trail] " + msg, *args)


def _universe_brief(symbols: List[str], *, limit: int = 12) -> str:
    if not symbols:
        return "(空)"
    head = ",".join(symbols[:limit])
    if len(symbols) > limit:
        return f"{head}…+{len(symbols) - limit}"
    return head


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def send_tg(text: str) -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return
    try:
        import requests

        requests.post(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
    except Exception:
        pass


def pnl_usdt(side: str, entry: float, exit_px: float, notional: float) -> float:
    if entry <= 0 or notional <= 0:
        return 0.0
    if side.upper() == "LONG":
        return notional * (exit_px - entry) / entry
    return notional * (entry - exit_px) / entry


def _outcome_for_pnl(pnl: float) -> str:
    if pnl > 0:
        return "win"
    if pnl < 0:
        return "loss"
    return "flat"


def _reopen_cooldown_kind(cur, *, symbol: str, side: str) -> str:
    last_ms, exit_rule = last_close_info(cur, symbol=symbol, side=side)
    if last_ms is None:
        return ""
    elapsed = (_now_ms() - last_ms) / 1000.0
    rule = str(exit_rule or "")
    if (
        cfg.JIEZHEN_TRAIL_REOPEN_BLOCK
        and rule.startswith("trail")
        and elapsed < cfg.JIEZHEN_TRAIL_REOPEN_COOLDOWN_SEC
    ):
        return "trail_reopen_cooldown"
    if cfg.JIEZHEN_COOLDOWN_SEC > 0 and elapsed < cfg.JIEZHEN_COOLDOWN_SEC:
        return "cooldown"
    return ""


def _settle_row(
    cur,
    row: sqlite3.Row,
    *,
    exit_price: float,
    exit_rule: str,
    now_utc: str,
) -> Dict[str, Any]:
    side = str(row["side"])
    entry = float(row["entry_price"] or 0)
    notional = float(row["virtual_notional_usdt"] or cfg.JIEZHEN_NOTIONAL_USDT)
    pnl = pnl_usdt(side, entry, exit_price, notional)
    outcome = _outcome_for_pnl(pnl)
    cur.execute(
        """
        UPDATE jz_signals SET
            outcome = ?, outcome_at_utc = ?, exit_price = ?,
            pnl_usdt = ?, exit_rule = ?, mark_price = ?, unrealized_pnl_usdt = NULL
        WHERE id = ?
        """,
        (outcome, now_utc, exit_price, pnl, exit_rule, exit_price, row["id"]),
    )
    archive_settlement(
        cur,
        signal_id=int(row["id"]),
        symbol=str(row["symbol"]),
        side=side,
        outcome=outcome,
        entry_price=entry,
        exit_price=exit_price,
        pnl_usdt=pnl,
        notional=notional,
        exit_rule=exit_rule,
        settled_at_utc=now_utc,
    )
    return {
        "symbol": row["symbol"],
        "side": side,
        "pnl_usdt": pnl,
        "outcome": outcome,
        "exit_rule": exit_rule,
    }


def _open_row(
    cur,
    *,
    side: str,
    symbol: str,
    entry_price: float,
    now_utc: str,
    meta: Dict[str, Any],
) -> None:
    cur.execute(
        """
        INSERT INTO jz_signals (
            recorded_at_utc, side, symbol, signal_type, entry_price,
            virtual_notional_usdt, mark_price,
            unrealized_pnl_usdt, meta_json, updated_at_utc,
            peak_profit_pct, trail_tier
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now_utc,
            side.upper(),
            symbol.upper(),
            "JZ_SPIKE",
            entry_price,
            cfg.JIEZHEN_NOTIONAL_USDT,
            entry_price,
            0.0,
            json.dumps(meta, default=str),
            now_utc,
            0.0,
            "none",
        ),
    )


def _apply_trail(
    cur,
    row: sqlite3.Row,
    *,
    mark: float,
    now_utc: str,
    stats: Dict[str, Any],
    events: List[str],
    log_tag: str = "trail",
) -> bool:
    side = str(row["side"])
    sym = str(row["symbol"])
    entry = float(row["entry_price"] or 0)
    notional = float(row["virtual_notional_usdt"] or cfg.JIEZHEN_NOTIONAL_USDT)
    u = pnl_usdt(side, entry, mark, notional)
    trail_cfg = cfg.jz_trail_config()
    ev = evaluate_trail(
        side=side,
        entry=entry,
        mark=mark,
        peak_profit_pct=peak_profit_from_row(row),
        cfg=trail_cfg,
    )
    log_fn = _log_jz_trail if log_tag == "trail" else _log_jz
    if _verbose_log():
        tier_cn = TIER_LABELS.get(ev.trail_tier, ev.trail_tier)
        log_fn(
            "%s %s entry=%.8g mark=%.8g profit=%.3f%% peak=%.3f%% tier=%s(%s)",
            side,
            sym,
            entry,
            mark,
            ev.profit_pct,
            ev.peak_profit_pct,
            ev.trail_tier,
            tier_cn,
        )
        if ev.exit_rule and trail_cfg.enabled:
            log_fn("%s %s 触发平仓 rule=%s", side, sym, ev.exit_rule)
    cur.execute(
        """
        UPDATE jz_signals SET
            mark_price = ?, unrealized_pnl_usdt = ?, updated_at_utc = ?,
            peak_profit_pct = ?, trail_tier = ?
        WHERE id = ? AND outcome IS NULL
        """,
        (mark, u, now_utc, ev.peak_profit_pct, ev.trail_tier, row["id"]),
    )
    if not trail_cfg.enabled or not ev.exit_rule:
        return False
    closed = _settle_row(
        cur, row, exit_price=mark, exit_rule=ev.exit_rule, now_utc=now_utc
    )
    stats["closes"] += 1
    tier_cn = TIER_LABELS.get(ev.trail_tier, ev.trail_tier)
    events.append(
        f"平{side[0]} {closed['symbol']} {ev.exit_rule}/{tier_cn} "
        f"pnl={closed['pnl_usdt']:.4f}U ({closed['outcome']})"
    )
    log_fn(
        "%s %s 平仓 rule=%s tier=%s pnl=%.4fU (%s)",
        side,
        sym,
        ev.exit_rule,
        tier_cn,
        closed["pnl_usdt"],
        closed["outcome"],
    )
    return True


def _run_trail_pass(
    cur,
    *,
    now_utc: str,
    stats: Dict[str, Any],
    events: List[str],
    log_tag: str = "trail",
) -> None:
    open_rows = fetch_all_open(cur)
    if _verbose_log() and not open_rows:
        log_fn = _log_jz_trail if log_tag == "trail" else _log_jz
        log_fn("止盈扫描：无持仓")
    for row in open_rows:
        sym = str(row["symbol"])
        mark = fetch_mark_price(sym)
        if mark is None:
            stats["skipped"].append(f"trail:no_mark:{sym}")
            log_fn = _log_jz_trail if log_tag == "trail" else _log_jz
            log_fn("%s 取 mark 失败，跳过止盈", sym)
            continue
        _apply_trail(
            cur,
            row,
            mark=mark,
            now_utc=now_utc,
            stats=stats,
            events=events,
            log_tag=log_tag,
        )


def _try_open_spike(
    cur,
    *,
    symbol: str,
    side: str,
    mark: float,
    plan_meta: Dict[str, Any],
    now_utc: str,
    stats: Dict[str, Any],
    events: List[str],
) -> None:
    side = side.upper()
    sym = symbol.upper()
    if fetch_open_by_symbol_side(cur, symbol=sym, side=side):
        if _verbose_log():
            _log_jz("%s %s 已持仓，跳过开仓", side, sym)
        return
    open_total = count_open(cur)
    if open_total >= cfg.JIEZHEN_MAX_OPEN_TOTAL:
        stats["skipped"].append(f"cap:total:{sym}")
        _log_jz("跳过开仓 %s %s：总持仓 cap %s/%s", side, sym, open_total, cfg.JIEZHEN_MAX_OPEN_TOTAL)
        return
    side_n = count_open(cur, side=side)
    if side_n >= cfg.JIEZHEN_MAX_OPEN_PER_SIDE:
        stats["skipped"].append(f"cap:{side}:{sym}")
        _log_jz(
            "跳过开仓 %s %s：%s 腿 cap %s/%s",
            side,
            sym,
            side,
            side_n,
            cfg.JIEZHEN_MAX_OPEN_PER_SIDE,
        )
        return
    cd = _reopen_cooldown_kind(cur, symbol=sym, side=side)
    if cd:
        stats["skipped"].append(f"cooldown:{side}:{sym}:{cd}")
        _log_jz("跳过开仓 %s %s：%s", side, sym, cd)
        return
    _open_row(
        cur,
        side=side,
        symbol=sym,
        entry_price=mark,
        now_utc=now_utc,
        meta=plan_meta,
    )
    stats["opens"] += 1
    events.append(f"开{side[0]} {sym} @ {mark:.8g} (JZ_SPIKE)")
    _log_jz("%s 纸面开仓 %s @ %.8g 名义=%.0fU", side, sym, mark, cfg.JIEZHEN_NOTIONAL_USDT)


def _process_symbol(
    cur,
    *,
    symbol: str,
    now_utc: str,
    stats: Dict[str, Any],
    events: List[str],
) -> None:
    sym = symbol.upper()
    mark = fetch_mark_price(sym)
    if mark is None:
        stats["skipped"].append(f"no_mark:{sym}")
        logger.warning("[jz] %s 取 mark 失败", sym)
        return
    klines = fetch_klines(
        sym,
        cfg.JIEZHEN_KLINE_INTERVAL,
        cfg.JIEZHEN_KLINE_LIMIT,
    )
    if not klines:
        stats["skipped"].append(f"no_klines:{sym}")
        logger.warning("[jz] %s K线为空 interval=%s", sym, cfg.JIEZHEN_KLINE_INTERVAL)
        return
    plan = build_spike_plan(
        mark=mark,
        klines_raw=klines,
        ema_period=cfg.JIEZHEN_EMA_PERIOD,
        atr_period=cfg.JIEZHEN_ATR_PERIOD,
        amplitude_period=cfg.JIEZHEN_AMPLITUDE_PERIOD,
        value_multiplier=cfg.JIEZHEN_VALUE_MULTIPLIER,
        min_distance_pct=cfg.JIEZHEN_MIN_DISTANCE_PCT,
        distance_mode=cfg.JIEZHEN_DISTANCE_MODE,
        touch_lookback_bars=cfg.JIEZHEN_TOUCH_LOOKBACK_BARS,
    )
    if plan is None:
        stats["skipped"].append(f"no_plan:{sym}")
        if _verbose_log():
            _log_jz("%s 无接针计划（距离/振幅过滤）", sym)
        return
    meta = {
        "lane": "jiezhen",
        "ema": plan.ema,
        "atr": plan.atr,
        "avg_amp_pct": plan.average_amplitude_pct,
        "selected_pct": plan.selected_distance_pct,
        "target_long": plan.target_long,
        "target_short": plan.target_short,
    }
    if _verbose_log():
        _log_jz(
            "%s dist=%.2f%% long_fill=%s short_fill=%s tgtL=%.8g tgtS=%.8g",
            sym,
            plan.selected_distance_pct,
            plan.long_fill,
            plan.short_fill,
            plan.target_long,
            plan.target_short,
        )
    if plan.long_fill:
        if _verbose_log():
            _log_jz("%s 触价接多 long_fill mark=%.8g tgt=%.8g", sym, mark, plan.target_long)
        if not fetch_open_by_symbol_side(cur, symbol=sym, side="LONG"):
            _try_open_spike(
                cur,
                symbol=sym,
                side="LONG",
                mark=mark,
                plan_meta={**meta, "fill": "long_spike"},
                now_utc=now_utc,
                stats=stats,
                events=events,
            )
    if plan.short_fill:
        if _verbose_log():
            _log_jz("%s 触价接空 short_fill mark=%.8g tgt=%.8g", sym, mark, plan.target_short)
        if not fetch_open_by_symbol_side(cur, symbol=sym, side="SHORT"):
            _try_open_spike(
                cur,
                symbol=sym,
                side="SHORT",
                mark=mark,
                plan_meta={**meta, "fill": "short_spike"},
                now_utc=now_utc,
                stats=stats,
                events=events,
            )


def _persist_run(
    cur,
    *,
    now_utc: str,
    universe_size: int,
    stats: Dict[str, Any],
    events: List[str],
    universe_meta: dict,
) -> None:
    cur.execute(
        """
        INSERT INTO jz_runs (
            ran_at_utc, universe_size, opens, closes, skipped, detail_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            now_utc,
            universe_size,
            stats["opens"],
            stats["closes"],
            ",".join(stats["skipped"][:40]),
            json.dumps(
                {
                    "ok": stats.get("ok", True),
                    "events": events[:40],
                    "universe_meta": universe_meta,
                    "notional_usdt": cfg.JIEZHEN_NOTIONAL_USDT,
                },
                default=str,
            ),
        ),
    )


def run_trail_checks_conn(
    conn: sqlite3.Connection, *, notify: bool = True
) -> Dict[str, Any]:
    now_utc = _utc_now()
    stats: Dict[str, Any] = {
        "ok": True,
        "closes": 0,
        "skipped": [],
        "task": "trail",
    }
    events: List[str] = []
    conn.row_factory = sqlite3.Row
    migrate_jz_tables(conn.cursor())
    conn.commit()
    cur = conn.cursor()
    trail_cfg = cfg.jz_trail_config()
    if not trail_cfg.enabled:
        stats["skipped"].append("trail_disabled")
        _log_jz_trail("移动止盈未启用 (JIEZHEN_TRAIL_ENABLED=0)")
        return stats
    if cfg.JIEZHEN_NOTIONAL_USDT <= 0:
        stats["ok"] = False
        stats["error"] = "zero_notional"
        logger.warning("[jz-trail] JIEZHEN_NOTIONAL_USDT<=0，跳过")
        return stats
    open_n = count_open(cur)
    _log_jz_trail("=== 止盈检查开始 %s | 持仓=%s 名义=%.0fU ===", now_utc, open_n, cfg.JIEZHEN_NOTIONAL_USDT)
    if open_n == 0:
        _log_jz_trail("无持仓")
    _run_trail_pass(cur, now_utc=now_utc, stats=stats, events=events)
    conn.commit()
    _log_jz_trail(
        "=== 止盈检查结束 %s closes=%s skipped=%s ===",
        now_utc,
        stats["closes"],
        stats["skipped"] or "—",
    )
    for e in events:
        _log_jz_trail("  · %s", e)
    if notify and events and cfg.JIEZHEN_TG_NOTIFY:
        send_tg("*接针止盈*\n" + "\n".join(events[:12]))
    stats["events"] = events
    return stats


def run_trail_checks(*, notify: bool = True) -> Dict[str, Any]:
    from accumulation_radar import init_db

    conn = init_db()
    try:
        return run_trail_checks_conn(conn, notify=notify)
    finally:
        conn.close()


def run_scan_conn(
    conn: sqlite3.Connection, *, notify: bool = True
) -> Dict[str, Any]:
    now_utc = _utc_now()
    stats: Dict[str, Any] = {
        "ok": True,
        "opens": 0,
        "closes": 0,
        "skipped": [],
    }
    events: List[str] = []
    conn.row_factory = sqlite3.Row
    migrate_jz_tables(conn.cursor())
    conn.commit()
    cur = conn.cursor()

    if cfg.JIEZHEN_NOTIONAL_USDT <= 0:
        stats["ok"] = False
        stats["error"] = "zero_notional"
        logger.warning("[jz] JIEZHEN_NOTIONAL_USDT<=0，跳过扫描")
        _persist_run(
            cur,
            now_utc=now_utc,
            universe_size=0,
            stats=stats,
            events=events,
            universe_meta={"error": "zero_notional"},
        )
        conn.commit()
        return stats

    universe, u_meta = resolve_jiezhen_universe()
    warn = u_meta.get("warning") if isinstance(u_meta, dict) else None
    _log_jz(
        "=== 扫描开始 %s | universe=%s cap=%s/%s 名义=%.0fU trail=%s ===",
        now_utc,
        len(universe),
        cfg.JIEZHEN_MAX_OPEN_PER_SIDE,
        cfg.JIEZHEN_MAX_OPEN_TOTAL,
        cfg.JIEZHEN_NOTIONAL_USDT,
        cfg.jz_trail_config().enabled,
    )
    if _verbose_log():
        _log_jz("标的池: %s", _universe_brief(universe))
        if warn:
            _log_jz("标的池提示: %s", warn)
    if not universe:
        stats["skipped"].append("empty_universe")
        logger.warning("[jz] 标的池为空（需 worth_watch_hot_oi / 先跑 oi） meta=%s", u_meta)
        _persist_run(
            cur,
            now_utc=now_utc,
            universe_size=0,
            stats=stats,
            events=events,
            universe_meta=u_meta,
        )
        conn.commit()
        return stats

    trail_cfg = cfg.jz_trail_config()
    open_before = count_open(cur)
    if trail_cfg.enabled:
        if _verbose_log() and open_before:
            _log_jz("扫描内止盈 pass：持仓=%s", open_before)
        _run_trail_pass(cur, now_utc=now_utc, stats=stats, events=events, log_tag="scan")
    elif _verbose_log():
        _log_jz("扫描跳过内嵌止盈（trail 未启用）")

    for sym in universe:
        try:
            _process_symbol(
                cur, symbol=sym, now_utc=now_utc, stats=stats, events=events
            )
        except Exception as e:
            stats["skipped"].append(f"err:{sym}")
            logger.exception("[jz] %s failed: %s", sym, e)

    _persist_run(
        cur,
        now_utc=now_utc,
        universe_size=len(universe),
        stats=stats,
        events=events,
        universe_meta=u_meta,
    )
    conn.commit()
    summary = (
        f"[jz] {now_utc} universe={len(universe)} "
        f"opens={stats['opens']} closes={stats['closes']}"
    )
    print(summary)
    _log_jz(
        "=== 扫描结束 opens=%s closes=%s universe=%s open_now=%s ===",
        stats["opens"],
        stats["closes"],
        len(universe),
        count_open(cur),
    )
    for s in stats["skipped"]:
        _log_jz("跳过: %s", s)
    for e in events:
        print(f"  · {e}")
        _log_jz("执行: %s", e)
    if notify and events and cfg.JIEZHEN_TG_NOTIFY:
        send_tg("*接针扫描*\n" + "\n".join(events[:12]))
    stats["events"] = events
    stats["universe"] = universe
    stats["universe_meta"] = u_meta
    return stats


def run_scan(*, notify: bool = True) -> Dict[str, Any]:
    from accumulation_radar import init_db

    conn = init_db()
    try:
        return run_scan_conn(conn, notify=notify)
    finally:
        conn.close()

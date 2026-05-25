"""动量多一空一 — topMovers 纸面调仓。

- 多：最新 PULLBACK（默认）
- 空：最新 RALLY（默认）
- 每腿最多 1 个持仓；标的变化时市价纸面平仓再开仓

CLI:
  python momentum_scanner.py
  python momentum_scanner.py --no-tg
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import momentum_config as cfg
from binance_fapi import fetch_mark_price
from momentum_db import (
    archive_settlement,
    fetch_open_by_side,
    last_close_info,
    last_close_utc_ms,
    migrate_mom_tables,
)
from momentum_db import peak_profit_from_row
from momentum_filters import inspect_open_filter
from momentum_signals import fetch_momentum_targets
from momentum_trail import evaluate_trail, TIER_LABELS

logger = logging.getLogger(__name__)

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")


def _verbose_log() -> bool:
    return cfg.MOM_VERBOSE_LOG


def _log_mom(msg: str, *args: Any) -> None:
    logger.info("[mom] " + msg, *args)


def _log_trail(msg: str, *args: Any) -> None:
    logger.info("[mom-trail] " + msg, *args)


def _event_brief(event_raw: Dict[str, Any] | None) -> str:
    if not event_raw:
        return "无事件原始数据"
    et = event_raw.get("eventType") or "?"
    pc = event_raw.get("priceChange")
    ts = event_raw.get("createTimestamp")
    return f"eventType={et} priceChange={pc} createTimestamp={ts}"


def _log_filter_report(side: str, report: Dict[str, Any]) -> None:
    sym = report.get("symbol") or "?"
    evt = report.get("_event_raw") or {}
    if report.get("allowed"):
        if not _verbose_log():
            return
        _log_mom(
            "%s %s 过滤通过 | %s | pc=%s age=%sm | VP=%s 允许=%s",
            side,
            sym,
            _event_brief(evt if isinstance(evt, dict) else {}),
            report.get("price_change"),
            report.get("event_age_min"),
            report.get("vp_scheme") or "—",
            ",".join(report.get("vp_allowed_schemes") or []),
        )
    else:
        _log_mom(
            "%s %s 开仓拒绝: %s (%s) | %s | pc=%s age=%sm VP=%s 允许=%s",
            side,
            sym,
            report.get("reason_cn"),
            report.get("reason"),
            _event_brief(evt if isinstance(evt, dict) else {}),
            report.get("price_change"),
            report.get("event_age_min"),
            report.get("vp_scheme") or "—",
            ",".join(report.get("vp_allowed_schemes") or []),
        )


def _append_filter_report(
    stats: Dict[str, Any], *, side: str, report: Dict[str, Any], event_raw: Dict[str, Any]
) -> None:
    report = dict(report)
    report["_event_raw"] = event_raw
    stats.setdefault("filter_reports", []).append(report)
    _log_filter_report(side, report)


TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")


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
    """
    若应禁止再次开仓，返回原因片段：trail_reopen_cooldown | cooldown；否则空串。
    移动止盈(exit_rule 以 trail 开头)后用更长冷却，避免下轮扫描立刻重开。
    """
    last_ms, exit_rule = last_close_info(cur, symbol=symbol, side=side)
    if last_ms is None:
        return ""
    elapsed = (_now_ms() - last_ms) / 1000.0
    rule = str(exit_rule or "")
    if (
        cfg.MOM_TRAIL_REOPEN_BLOCK
        and rule.startswith("trail")
        and elapsed < cfg.MOM_TRAIL_REOPEN_COOLDOWN_SEC
    ):
        return "trail_reopen_cooldown"
    if cfg.MOM_COOLDOWN_SEC > 0 and elapsed < cfg.MOM_COOLDOWN_SEC:
        return "cooldown"
    return ""


def _cooldown_blocks(cur, *, symbol: str, side: str) -> bool:
    return bool(_reopen_cooldown_kind(cur, symbol=symbol, side=side))


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
    notional = float(row["virtual_notional_usdt"] or cfg.MOM_NOTIONAL_USDT)
    pnl = pnl_usdt(side, entry, exit_price, notional)
    outcome = _outcome_for_pnl(pnl)
    cur.execute(
        """
        UPDATE mom_signals SET
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
        "entry": entry,
        "exit": exit_price,
        "pnl_usdt": pnl,
        "outcome": outcome,
        "exit_rule": exit_rule,
    }


def _open_row(
    cur,
    *,
    side: str,
    symbol: str,
    signal_type: str,
    entry_price: float,
    event_timestamp_ms: Optional[int],
    now_utc: str,
    meta: Dict[str, Any],
) -> None:
    cur.execute(
        """
        INSERT INTO mom_signals (
            recorded_at_utc, side, symbol, signal_type, entry_price,
            virtual_notional_usdt, event_timestamp_ms, mark_price,
            unrealized_pnl_usdt, meta_json, updated_at_utc,
            peak_profit_pct, trail_tier
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now_utc,
            side.upper(),
            symbol.upper(),
            signal_type,
            entry_price,
            cfg.MOM_NOTIONAL_USDT,
            event_timestamp_ms,
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
    stats: Optional[Dict[str, Any]] = None,
    events: Optional[List[str]] = None,
    log_tag: str = "mom",
) -> bool:
    """更新 mark / 浮盈 / 峰值；若触发移动止盈则平仓。返回是否已平仓。"""
    side = str(row["side"])
    sym = str(row["symbol"])
    entry = float(row["entry_price"] or 0)
    notional = float(row["virtual_notional_usdt"] or cfg.MOM_NOTIONAL_USDT)
    u = pnl_usdt(side, entry, mark, notional)
    trail_cfg = cfg.mom_trail_config()
    ev = evaluate_trail(
        side=side,
        entry=entry,
        mark=mark,
        peak_profit_pct=peak_profit_from_row(row),
        cfg=trail_cfg,
    )
    if _verbose_log():
        log_fn = _log_trail if log_tag == "trail" else _log_mom
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
        UPDATE mom_signals SET
            mark_price = ?, unrealized_pnl_usdt = ?, updated_at_utc = ?,
            peak_profit_pct = ?, trail_tier = ?
        WHERE id = ? AND outcome IS NULL
        """,
        (mark, u, now_utc, ev.peak_profit_pct, ev.trail_tier, row["id"]),
    )
    if not trail_cfg.enabled or not ev.exit_rule:
        return False
    closed = _settle_row(
        cur,
        row,
        exit_price=mark,
        exit_rule=ev.exit_rule,
        now_utc=now_utc,
    )
    if stats is not None:
        stats["closes"] += 1
    if events is not None:
        events.append(
            f"平{side[0]} {closed['symbol']} {ev.exit_rule}/{ev.tier_label} "
            f"pnl={closed['pnl_usdt']:.4f}U ({closed['outcome']})"
        )
    return True


def _mark_open_row(
    cur,
    row: sqlite3.Row,
    *,
    mark: float,
    now_utc: str,
    stats: Optional[Dict[str, Any]] = None,
    events: Optional[List[str]] = None,
) -> bool:
    return _apply_trail(
        cur,
        row,
        mark=mark,
        now_utc=now_utc,
        stats=stats,
        events=events,
        log_tag="scan",
    )


def _try_trail_exit(
    cur,
    *,
    side: str,
    now_utc: str,
    stats: Dict[str, Any],
    events: List[str],
    log_tag: str = "trail",
) -> None:
    """持仓腿：分档移动止盈 / 止损（优先于 rotate）。"""
    open_row = fetch_open_by_side(cur, side)
    if not open_row:
        if _verbose_log():
            (_log_trail if log_tag == "trail" else _log_mom)("%s 无持仓", side)
        return
    sym = str(open_row["symbol"])
    mark = fetch_mark_price(sym)
    if mark is None:
        stats["skipped"].append(f"{side}:trail:no_mark:{sym}")
        if _verbose_log():
            (_log_trail if log_tag == "trail" else _log_mom)(
                "%s %s 取 mark 失败", side, sym
            )
        return
    _apply_trail(
        cur,
        open_row,
        mark=mark,
        now_utc=now_utc,
        stats=stats,
        events=events,
        log_tag=log_tag,
    )


def _run_trail_pass(
    cur,
    *,
    now_utc: str,
    stats: Dict[str, Any],
    events: List[str],
    log_tag: str = "trail",
) -> None:
    for leg in ("LONG", "SHORT"):
        _try_trail_exit(
            cur,
            side=leg,
            now_utc=now_utc,
            stats=stats,
            events=events,
            log_tag=log_tag,
        )


def _adjust_side(
    cur,
    *,
    side: str,
    target_symbol: Optional[str],
    signal_type: str,
    event_timestamp_ms: Optional[int],
    now_utc: str,
    stats: Dict[str, Any],
    events: List[str],
    signal_meta: Dict[str, Any],
    peer_symbol: Optional[str] = None,
) -> None:
    open_row = fetch_open_by_side(cur, side)
    current_sym = str(open_row["symbol"]).upper() if open_row else None

    if not target_symbol:
        if _verbose_log():
            _log_mom(
                "%s 本腿 topMovers 无目标；当前持仓=%s",
                side,
                current_sym or "无",
            )
        if open_row:
            sym = str(open_row["symbol"])
            px = fetch_mark_price(sym)
            if px is not None:
                _mark_open_row(
                    cur,
                    open_row,
                    mark=px,
                    now_utc=now_utc,
                    stats=stats,
                    events=events,
                )
            else:
                stats["skipped"].append(f"{side}:no_mark_price:{sym}")
        return

    target_symbol = target_symbol.upper()
    if _verbose_log():
        _log_mom(
            "%s 调仓 目标=%s 当前=%s 信号=%s 对腿=%s",
            side,
            target_symbol,
            current_sym or "无",
            _event_brief(signal_meta),
            peer_symbol or "—",
        )

    if current_sym == target_symbol:
        px = fetch_mark_price(target_symbol)
        if px is None:
            stats["skipped"].append(f"{side}:no_mark_price:{target_symbol}")
            _log_mom("%s %s 标的未变但取 mark 失败", side, target_symbol)
            return
        if _verbose_log():
            _log_mom("%s %s 标的未变，更新 mark=%.8g", side, target_symbol, px)
        if _mark_open_row(
            cur,
            open_row,
            mark=px,
            now_utc=now_utc,
            stats=stats,
            events=events,
        ):
            return
        return

    closed_without_reopen = False

    if open_row and current_sym:
        exit_px = fetch_mark_price(current_sym)
        if exit_px is None:
            stats["skipped"].append(f"{side}:no_exit_price:{current_sym}")
            _log_mom("%s rotate 平仓 %s 失败: 无 exit 价", side, current_sym)
            return
        closed = _settle_row(
            cur,
            open_row,
            exit_price=exit_px,
            exit_rule="rotate",
            now_utc=now_utc,
        )
        stats["closes"] += 1
        closed_without_reopen = True
        events.append(
            f"平{side[0]} {closed['symbol']} pnl={closed['pnl_usdt']:.4f}U ({closed['outcome']})"
        )
        _log_mom(
            "%s rotate 平仓 %s @ %.8g pnl=%.4fU → 目标 %s",
            side,
            closed["symbol"],
            exit_px,
            closed["pnl_usdt"],
            target_symbol,
        )

    cd_kind = _reopen_cooldown_kind(cur, symbol=target_symbol, side=side)
    if cd_kind:
        tag = (
            f"{side}:closed_without_reopen:{cd_kind}:{target_symbol}"
            if closed_without_reopen
            else f"{side}:{cd_kind}:{target_symbol}"
        )
        stats["skipped"].append(tag)
        if cd_kind == "trail_reopen_cooldown":
            _log_mom(
                "%s 动态止盈后冷却中不可开 %s（%s 分钟内）",
                side,
                target_symbol,
                cfg.MOM_TRAIL_REOPEN_COOLDOWN_MIN,
            )
        else:
            _log_mom("%s 冷却中不可开 %s (%s)", side, target_symbol, tag)
        return

    report = inspect_open_filter(
        side=side,
        symbol=target_symbol,
        event_raw=signal_meta,
        peer_symbol=peer_symbol,
    )
    _append_filter_report(stats, side=side, report=report, event_raw=signal_meta)
    if not report["allowed"]:
        filter_reason = str(report["reason"])
        if closed_without_reopen:
            stats["skipped"].append(
                f"{side}:closed_without_reopen:{filter_reason}:{target_symbol}"
            )
        else:
            stats["skipped"].append(f"{side}:{filter_reason}:{target_symbol}")
        return

    entry_px = fetch_mark_price(target_symbol)
    if entry_px is None:
        if closed_without_reopen:
            stats["skipped"].append(
                f"{side}:closed_without_reopen:no_entry_price:{target_symbol}"
            )
        else:
            stats["skipped"].append(f"{side}:no_entry_price:{target_symbol}")
        _log_mom("%s 过滤已通过但取入场价失败: %s", side, target_symbol)
        return

    filter_reason = str(report.get("reason") or "")
    meta = {
        "lane": "momentum_top_movers",
        "signal": signal_meta,
        "notional_usdt": cfg.MOM_NOTIONAL_USDT,
        "equity_usdt": cfg.MOM_ACCOUNT_EQUITY_USDT,
        "leverage": cfg.MOM_LEVERAGE,
        "open_filter": filter_reason or "ok",
        "filter_report": {k: v for k, v in report.items() if k != "_event_raw"},
    }
    _open_row(
        cur,
        side=side,
        symbol=target_symbol,
        signal_type=signal_type,
        entry_price=entry_px,
        event_timestamp_ms=event_timestamp_ms,
        now_utc=now_utc,
        meta=meta,
    )
    stats["opens"] += 1
    events.append(f"开{side[0]} {target_symbol} @ {entry_px:.8g} ({signal_type})")
    _log_mom(
        "%s 开仓 %s @ %.8g (%s) 名义=%.0fU",
        side,
        target_symbol,
        entry_px,
        signal_type,
        cfg.MOM_NOTIONAL_USDT,
    )


def _persist_mom_run(
    cur,
    *,
    now_utc: str,
    long_sym: Optional[str],
    short_sym: Optional[str],
    stats: Dict[str, Any],
    events: List[str],
    sig_meta: Dict[str, Any],
) -> None:
    cur.execute(
        """
        INSERT INTO mom_runs (
            ran_at_utc, long_target, short_target, opens, closes, skipped, detail_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now_utc,
            long_sym,
            short_sym,
            stats["opens"],
            stats["closes"],
            ",".join(stats["skipped"][:20]),
            json.dumps(
                {
                    "ok": stats.get("ok", True),
                    "error": stats.get("error"),
                    "events": events[:30],
                    "signal_meta": sig_meta,
                    "filter_reports": stats.get("filter_reports", [])[:10],
                    "notional_usdt": cfg.MOM_NOTIONAL_USDT,
                },
                default=str,
            ),
        ),
    )


def run_trail_checks_conn(
    conn: sqlite3.Connection, *, notify: bool = True
) -> Dict[str, Any]:
    """仅检查持仓移动止盈 / 止损（不调 topMovers、不开换仓）。"""
    now_utc = _utc_now()
    stats: Dict[str, Any] = {
        "ok": True,
        "closes": 0,
        "skipped": [],
        "task": "trail",
    }
    events: List[str] = []

    conn.row_factory = sqlite3.Row
    migrate_mom_tables(conn.cursor())
    conn.commit()
    cur = conn.cursor()

    if not cfg.MOM_TRAIL_ENABLED:
        stats["skipped"].append("trail_disabled")
        return stats

    if cfg.MOM_NOTIONAL_USDT <= 0:
        stats["ok"] = False
        stats["error"] = "zero_notional"
        stats["skipped"].append("zero_notional")
        return stats

    if _verbose_log():
        _log_trail("=== 止盈检查开始 %s ===", now_utc)
    _run_trail_pass(cur, now_utc=now_utc, stats=stats, events=events)
    conn.commit()

    _log_trail(
        "=== 止盈检查结束 %s closes=%s skipped=%s ===",
        now_utc,
        stats["closes"],
        stats["skipped"] or "—",
    )
    for e in events:
        _log_trail("  · %s", e)

    if notify and events and cfg.MOM_TG_NOTIFY:
        send_tg("*动量止盈*\n" + "\n".join(events[:12]))

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
    sig_meta: Dict[str, Any] = {}

    conn.row_factory = sqlite3.Row
    migrate_mom_tables(conn.cursor())
    conn.commit()
    cur = conn.cursor()

    if cfg.MOM_NOTIONAL_USDT <= 0:
        stats["ok"] = False
        stats["error"] = "zero_notional"
        logger.warning("[mom] MOM_NOTIONAL_USDT<=0，跳过调仓")
        _persist_mom_run(
            cur,
            now_utc=now_utc,
            long_sym=None,
            short_sym=None,
            stats=stats,
            events=events,
            sig_meta={"error": "zero_notional"},
        )
        conn.commit()
        return stats

    if _verbose_log():
        _log_mom(
            "=== 扫描开始 %s | filter=%s vp=%s trail=%s 名义=%.0fU ===",
            now_utc,
            cfg.mom_filter_enabled(),
            cfg.MOM_VP_FILTER,
            cfg.MOM_TRAIL_ENABLED,
            cfg.MOM_NOTIONAL_USDT,
        )
    _run_trail_pass(
        cur, now_utc=now_utc, stats=stats, events=events, log_tag="scan"
    )

    long_sym, short_sym, sig_meta = fetch_momentum_targets()
    if sig_meta.get("error"):
        stats["ok"] = False
        stats["error"] = sig_meta["error"]
        logger.warning("[mom] topMovers 失败: %s", sig_meta)
        _persist_mom_run(
            cur,
            now_utc=now_utc,
            long_sym=long_sym,
            short_sym=short_sym,
            stats=stats,
            events=events,
            sig_meta=sig_meta,
        )
        conn.commit()
        return stats

    long_evt = sig_meta.get("long_event_raw") or {}
    short_evt = sig_meta.get("short_event_raw") or {}
    long_ts = int(long_evt.get("createTimestamp") or 0) or None
    short_ts = int(short_evt.get("createTimestamp") or 0) or None

    _log_mom(
        "topMovers movers=%s valid=%s → long=%s short=%s",
        sig_meta.get("movers_total"),
        sig_meta.get("movers_valid"),
        long_sym,
        short_sym,
    )
    if _verbose_log():
        _log_mom("LONG 事件: %s", _event_brief(long_evt))
        _log_mom("SHORT 事件: %s", _event_brief(short_evt))

    _adjust_side(
        cur,
        side="LONG",
        target_symbol=long_sym,
        signal_type=cfg.MOM_LONG_EVENT,
        event_timestamp_ms=long_ts,
        now_utc=now_utc,
        stats=stats,
        events=events,
        signal_meta=long_evt,
        peer_symbol=short_sym,
    )
    _adjust_side(
        cur,
        side="SHORT",
        target_symbol=short_sym,
        signal_type=cfg.MOM_SHORT_EVENT,
        event_timestamp_ms=short_ts,
        now_utc=now_utc,
        stats=stats,
        events=events,
        signal_meta=short_evt,
        peer_symbol=long_sym,
    )
    _persist_mom_run(
        cur,
        now_utc=now_utc,
        long_sym=long_sym,
        short_sym=short_sym,
        stats=stats,
        events=events,
        sig_meta=sig_meta,
    )
    conn.commit()

    summary = (
        f"[mom] {now_utc} long={long_sym} short={short_sym} "
        f"opens={stats['opens']} closes={stats['closes']}"
    )
    print(summary)
    _log_mom(
        "=== 扫描结束 long=%s short=%s opens=%s closes=%s ===",
        long_sym,
        short_sym,
        stats["opens"],
        stats["closes"],
    )
    for s in stats["skipped"]:
        _log_mom("跳过: %s", s)
    for e in events:
        print(f"  · {e}")
        _log_mom("执行: %s", e)

    if notify and events and cfg.MOM_TG_NOTIFY:
        send_tg("*动量纸面*\n" + "\n".join(events[:12]))

    stats["long_target"] = long_sym
    stats["short_target"] = short_sym
    stats["events"] = events
    return stats


def run_scan(*, notify: bool = True) -> Dict[str, Any]:
    from accumulation_radar import init_db

    conn = init_db()
    try:
        return run_scan_conn(conn, notify=notify)
    finally:
        conn.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    ap = argparse.ArgumentParser(description="动量多一空一纸面扫描")
    ap.add_argument("--no-tg", action="store_true")
    ap.add_argument(
        "--trail-only",
        action="store_true",
        help="仅跑移动止盈检查（与定时任务 mom_trail 相同）",
    )
    args = ap.parse_args()
    if args.trail_only:
        run_trail_checks(notify=not args.no_tg)
    else:
        run_scan(notify=not args.no_tg)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Supertrend 量化信号（币安 U 本位永续）

- 标的：worth_watch_hot_oi（🔥⚡ 热度+OI，由 OI 雷达写入）
- 平仓：利润保护（浮盈回撤 giveback → ATR 跟踪 trail_atr）→ 反转信号 reverse_signal
- 开仓过滤：ADX / 15m 同向 / 箱体·ATR% / 确认 K / 翻转冷却等（ST_FILTER_ENABLED，仅挡新开）
- 入场窗口：翻转后 ST_ENTRY_WINDOW_BARS 根内可入场（配合确认 K，非仅 flip 当根）
- 亏损冷却：仅挡同向再开，反转反手不受阻
- 掉出热度+OI 池：不按市价强平；有仓则继续扫描至反转平仓，且不再开新仓/反手
- 定时：APScheduler cron（K 线收盘后 +30s）；见 ST_SCHEDULER_ENABLED

用法：
  python supertrend_signal_scanner.py
  python supertrend_signal_scanner.py --resolve-only
  python supertrend_signal_scanner.py --no-tg
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging

import pandas as pd
import requests

from binance_fapi import fetch_klines, klines_to_df
import supertrend_config as cfg
from supertrend_db import (
    archive_settlement,
    cooldown_blocks_entry,
    count_open_positions,
    count_symbol_losses_today,
    fetch_open_row,
    get_indicator_state,
    list_open_position_symbols,
    migrate_st_tables,
    purge_expired_cooldowns,
    upsert_indicator_state,
    upsert_symbol_cooldown,
)
from supertrend_filters import (
    build_filter_context,
    chop_cooldown_until_bar,
    closed_bars_df,
    compute_entry_intent,
    evaluate_entry_filters,
    htf_trend_for_symbol,
    record_filter_reject,
)
from supertrend_indicator import compute_supertrend, last_closed_bar_signals
from supertrend_profit_protect import run_profit_protection
from supertrend_universe import resolve_symbols

logger = logging.getLogger(__name__)

_env_file = Path(__file__).parent / ".env.oi"
if _env_file.exists():
    with open(_env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def send_tg(text: str) -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
    except Exception:
        pass


def _timeframe_ms() -> int:
    return cfg.st_timeframe_ms(cfg.ST_TIMEFRAME)


def _now_utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _apply_close_cooldown(
    cur,
    closed_evt: Dict[str, Any],
    *,
    symbol: str,
    bar_open_ms: int,
    now_utc: str,
) -> None:
    pnl = float(closed_evt.get("pnl_usdt") or 0)
    now_ms = _now_utc_ms()
    if pnl < 0 and cfg.ST_COOLDOWN_AFTER_LOSS_MIN > 0:
        until_ms = now_ms + cfg.ST_COOLDOWN_AFTER_LOSS_MIN * 60_000
        upsert_symbol_cooldown(
            cur,
            symbol=symbol,
            until_bar_open_ms=bar_open_ms,
            until_utc_ms=until_ms,
            reason="loss_cooldown",
            updated_at_utc=now_utc,
            blocked_side=str(closed_evt.get("side") or "").upper() or None,
        )
    elif pnl > 0 and cfg.ST_COOLDOWN_AFTER_WIN_MIN > 0:
        until_ms = now_ms + cfg.ST_COOLDOWN_AFTER_WIN_MIN * 60_000
        upsert_symbol_cooldown(
            cur,
            symbol=symbol,
            until_bar_open_ms=bar_open_ms,
            until_utc_ms=until_ms,
            reason="win_cooldown",
            updated_at_utc=now_utc,
            blocked_side=str(closed_evt.get("side") or "").upper() or None,
        )


def _try_open_side(
    cur,
    *,
    symbol: str,
    side: str,
    signal_type: str,
    open_row: Optional[sqlite3.Row],
    st_df: pd.DataFrame,
    last_bar: pd.Series,
    bar_open_ms: int,
    trend: int,
    close_px: float,
    st_up: float,
    st_dn: float,
    st_atr: float,
    now_utc: str,
    stats: Dict[str, Any],
    events: List[str],
    htf_cache: Dict[str, Optional[int]],
    tf_ms: int,
) -> None:
    if open_row is not None and str(open_row["side"]) == side:
        return

    cd_reason = cooldown_blocks_entry(
        cur,
        symbol,
        bar_open_ms=bar_open_ms,
        now_utc_ms=_now_utc_ms(),
        entry_side=side,
    )
    if cd_reason:
        stats["skipped"].append(f"{symbol}:cooldown:{cd_reason}")
        stats["filter_blocked"] = stats.get("filter_blocked", 0) + 1
        record_filter_reject(stats, cd_reason)
        return

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if cfg.ST_MAX_LOSSES_PER_SYMBOL_PER_DAY > 0:
        losses = count_symbol_losses_today(cur, symbol, day)
        if losses >= cfg.ST_MAX_LOSSES_PER_SYMBOL_PER_DAY:
            stats["skipped"].append(f"{symbol}:symbol_daily_loss_cap")
            stats["filter_blocked"] = stats.get("filter_blocked", 0) + 1
            record_filter_reject(stats, "symbol_daily_loss_cap")
            return

    htf_trend: Optional[int] = None
    if cfg.ST_FILTER_ENABLED and (cfg.ST_HTF_TIMEFRAME or "").strip():
        if symbol not in htf_cache:
            htf_ms = cfg.st_timeframe_ms(cfg.ST_HTF_TIMEFRAME)
            htf_cache[symbol] = htf_trend_for_symbol(
                symbol,
                fetch_klines_fn=fetch_klines,
                klines_to_df_fn=klines_to_df,
                timeframe_ms=htf_ms,
            )
        htf_trend = htf_cache[symbol]

    ctx = build_filter_context(
        symbol, side, st_df, last_bar, timeframe_ms=tf_ms, htf_trend=htf_trend
    )

    until_chop = chop_cooldown_until_bar(ctx, tf_ms)
    if until_chop is not None:
        upsert_symbol_cooldown(
            cur,
            symbol=symbol,
            until_bar_open_ms=until_chop,
            until_utc_ms=None,
            reason="chop_flips",
            updated_at_utc=now_utc,
            blocked_side=None,
        )
        stats["skipped"].append(f"{symbol}:chop_flips:{ctx.flip_count}")
        stats["filter_blocked"] = stats.get("filter_blocked", 0) + 1
        record_filter_reject(stats, "chop_flips")
        return

    allowed, reject = evaluate_entry_filters(ctx)
    if not allowed:
        stats["skipped"].append(f"{symbol}:filter:{reject}")
        stats["filter_blocked"] = stats.get("filter_blocked", 0) + 1
        record_filter_reject(stats, reject)
        return

    meta = {
        "lane": "supertrend",
        "universe": cfg.ST_UNIVERSE_MODE,
        "filter": {
            "adx": ctx.adx,
            "htf_trend": ctx.htf_trend,
            "range_pct": ctx.range_pct,
            "atr_pct": ctx.atr_pct,
            "flip_count": ctx.flip_count,
        },
    }
    _open_position(
        cur,
        symbol=symbol,
        side=side,
        signal_type=signal_type,
        entry_price=close_px,
        bar_open_ms=bar_open_ms,
        trend=trend,
        st_up=st_up,
        st_dn=st_dn,
        st_atr=st_atr,
        now_utc=now_utc,
        meta=meta,
    )
    stats["opens"] += 1
    events.append(f"开{'多' if side == 'LONG' else '空'} {symbol} @ {close_px:.8g}")


def _pnl_usdt(side: str, entry: float, exit_px: float, notional: float) -> float:
    if entry <= 0:
        return 0.0
    if side == "LONG":
        return notional * (exit_px - entry) / entry
    return notional * (entry - exit_px) / entry


def _daily_profit_lock_exceeded(conn) -> bool:
    """当日已实现净利 ≥ 权益×比例 时禁止新开仓（利润保护）。"""
    if cfg.ST_DAILY_PROFIT_LOCK_PCT <= 0:
        return False
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(SUM(pnl_usdt), 0) FROM st_settlements
        WHERE settled_at_utc >= ? AND pnl_usdt IS NOT NULL
        """,
        (f"{day}T00:00:00Z",),
    )
    net = float(cur.fetchone()[0] or 0)
    cap = cfg.ST_ACCOUNT_EQUITY_USDT * cfg.ST_DAILY_PROFIT_LOCK_PCT
    return net >= cap


def _daily_loss_exceeded(conn) -> bool:
    """当日已实现净 PnL ≤ -权益×比例 时禁止新开仓。"""
    if cfg.ST_MAX_DAILY_LOSS_PCT <= 0:
        return False
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(SUM(pnl_usdt), 0) FROM st_settlements
        WHERE settled_at_utc >= ? AND pnl_usdt IS NOT NULL
        """,
        (f"{day}T00:00:00Z",),
    )
    net = float(cur.fetchone()[0] or 0)
    cap = cfg.ST_ACCOUNT_EQUITY_USDT * cfg.ST_MAX_DAILY_LOSS_PCT
    return net <= -cap


def _outcome_for_exit(exit_rule: str, pnl: float) -> str:
    if exit_rule == "reverse_signal":
        return "reverse"
    if exit_rule in ("trail_atr", "giveback"):
        return "win" if pnl >= 0 else "loss"
    if exit_rule == "universe_removed":
        return "pruned"
    return "win" if pnl >= 0 else "loss"


def _persist_position_meta(cur, signal_id: int, meta: Dict[str, Any]) -> None:
    cur.execute(
        "UPDATE st_signals SET meta_json = ? WHERE id = ?",
        (json.dumps(meta, default=str), signal_id),
    )


def _emit_close(
    cur,
    closed_evt: Dict[str, Any],
    *,
    symbol: str,
    bar_open_ms: int,
    now_utc: str,
    stats: Dict[str, Any],
    events: List[str],
) -> None:
    _apply_close_cooldown(
        cur, closed_evt, symbol=symbol, bar_open_ms=bar_open_ms, now_utc=now_utc
    )
    stats["closes"] += 1
    events.append(
        f"平仓 {closed_evt['symbol']} {closed_evt['side']} "
        f"pnl={closed_evt['pnl_usdt']:.2f}U ({closed_evt['outcome']}/{closed_evt['exit_rule']})"
    )
    if cfg.ST_TG_NOTIFY_RESOLVE:
        send_tg(
            f"*ST 平仓* `{closed_evt['symbol']}` {closed_evt['side']}\n"
            f"rule={closed_evt['exit_rule']} exit={closed_evt['exit']:.8g} "
            f"pnl={closed_evt['pnl_usdt']:.2f} USDT"
        )


def _try_profit_protect_exit(
    cur,
    open_row: sqlite3.Row,
    *,
    high_px: float,
    low_px: float,
    close_px: float,
    st_atr: float,
    now_utc: str,
    symbol: str,
    bar_open_ms: int,
    stats: Dict[str, Any],
    events: List[str],
) -> Optional[Dict[str, Any]]:
    if not cfg.st_profit_protect_enabled():
        return None
    exit_rule, meta, fill_px = run_profit_protection(
        open_row,
        high=high_px,
        low=low_px,
        close=close_px,
        atr=st_atr,
    )
    _persist_position_meta(cur, int(open_row["id"]), meta)
    if not exit_rule:
        return None
    stats["exit_protect"] = stats.get("exit_protect") or {}
    if isinstance(stats["exit_protect"], dict):
        stats["exit_protect"][exit_rule] = int(stats["exit_protect"].get(exit_rule, 0)) + 1
    closed = _close_position(
        cur,
        open_row,
        exit_price=fill_px,
        exit_rule=exit_rule,
        now_utc=now_utc,
    )
    _emit_close(
        cur, closed, symbol=symbol, bar_open_ms=bar_open_ms, now_utc=now_utc, stats=stats, events=events
    )
    return closed


def _settle_position(
    cur,
    row,
    *,
    exit_price: float,
    exit_rule: str,
    now_utc: str,
    outcome: Optional[str] = None,
) -> Dict[str, Any]:
    side = str(row["side"])
    entry = float(row["entry_price"] or 0)
    notional = float(row["virtual_notional_usdt"] or cfg.ST_NOTIONAL_USDT)
    pnl = _pnl_usdt(side, entry, exit_price, notional)
    outcome = outcome or _outcome_for_exit(exit_rule, pnl)
    cur.execute(
        """
        UPDATE st_signals SET
            outcome = ?, outcome_at_utc = ?, exit_price = ?,
            pnl_usdt = ?, exit_rule = ?, side = 'FLAT', signal_type = 'CLOSE'
        WHERE id = ?
        """,
        (outcome, now_utc, exit_price, pnl, exit_rule, row["id"]),
    )
    archive_settlement(
        cur,
        signal_id=int(row["id"]),
        symbol=str(row["symbol"]),
        side=side,
        outcome=outcome,
        entry_price=entry,
        exit_price=exit_price,
        pnl_r=None,
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


def _close_position(
    cur,
    row,
    *,
    exit_price: float,
    exit_rule: str,
    now_utc: str,
    outcome: Optional[str] = None,
) -> Dict[str, Any]:
    return _settle_position(
        cur,
        row,
        exit_price=exit_price,
        exit_rule=exit_rule,
        now_utc=now_utc,
        outcome=outcome,
    )


def _bar_already_handled(
    open_row: Optional[sqlite3.Row],
    *,
    bar_open_ms: int,
    state: Optional[Any],
    buy: bool,
    sell: bool,
) -> bool:
    """同根 K 已处理且无未完成的反向动作时跳过（崩溃重跑仍可平仓/反手）。"""
    if not state or int(state[0]) != bar_open_ms:
        return False
    if open_row is not None:
        pos = str(open_row["side"])
        entry_ms = open_row["entry_bar_open_ms"]
        if (pos == "LONG" and sell) or (pos == "SHORT" and buy):
            return False
        if entry_ms is not None and int(entry_ms) == bar_open_ms:
            return True
    if buy or sell:
        return False
    return True


def _open_position(
    cur,
    *,
    symbol: str,
    side: str,
    signal_type: str,
    entry_price: float,
    bar_open_ms: int,
    trend: int,
    st_up: float,
    st_dn: float,
    st_atr: float,
    now_utc: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    notional = cfg.ST_NOTIONAL_USDT
    payload = meta if meta is not None else {"lane": "supertrend", "universe": cfg.ST_UNIVERSE_MODE}
    payload.setdefault(
        "protect",
        {
            "mfe_price": entry_price,
            "peak_pnl_pct": 0.0,
            "trail_armed": False,
            "trail_stop": None,
        },
    )
    cur.execute(
        """
        INSERT INTO st_signals (
            recorded_at_utc, symbol, side, trend, signal_type,
            entry_price, sl_price, tp_price, st_up, st_dn, st_atr,
            timeframe, st_period, st_multiplier, entry_bar_open_ms,
            virtual_notional_usdt, meta_json
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            recorded_at_utc = excluded.recorded_at_utc,
            side = excluded.side,
            trend = excluded.trend,
            signal_type = excluded.signal_type,
            entry_price = excluded.entry_price,
            sl_price = NULL,
            tp_price = NULL,
            st_up = excluded.st_up,
            st_dn = excluded.st_dn,
            st_atr = excluded.st_atr,
            timeframe = excluded.timeframe,
            st_period = excluded.st_period,
            st_multiplier = excluded.st_multiplier,
            entry_bar_open_ms = excluded.entry_bar_open_ms,
            outcome = NULL,
            outcome_at_utc = NULL,
            exit_price = NULL,
            pnl_usdt = NULL,
            exit_rule = NULL,
            virtual_notional_usdt = excluded.virtual_notional_usdt,
            meta_json = excluded.meta_json
        """,
        (
            now_utc,
            symbol,
            side,
            trend,
            signal_type,
            entry_price,
            st_up,
            st_dn,
            st_atr,
            cfg.ST_TIMEFRAME,
            cfg.ST_ATR_PERIOD,
            cfg.ST_ATR_MULTIPLIER,
            bar_open_ms,
            notional,
            json.dumps(payload, default=str),
        ),
    )


def _process_symbol(
    conn,
    symbol: str,
    *,
    now_utc: str,
    stats: Dict[str, Any],
    events: List[str],
    in_universe: bool = True,
    htf_cache: Optional[Dict[str, Optional[int]]] = None,
) -> None:
    if htf_cache is None:
        htf_cache = {}
    cur = conn.cursor()
    tf_ms = _timeframe_ms()
    rows = fetch_klines(symbol, cfg.ST_TIMEFRAME, cfg.ST_KLINE_LIMIT)
    if len(rows) < cfg.ST_ATR_PERIOD + 5:
        stats["skipped"].append(f"{symbol}:insufficient_klines")
        return

    df = klines_to_df(rows)
    st_df = compute_supertrend(
        df,
        period=cfg.ST_ATR_PERIOD,
        multiplier=cfg.ST_ATR_MULTIPLIER,
        source=cfg.ST_SOURCE,
        atr_method=cfg.ST_ATR_METHOD,
    )
    last_bar, _ = last_closed_bar_signals(st_df, timeframe_ms=tf_ms)
    if last_bar is None:
        stats["skipped"].append(f"{symbol}:no_closed_bar")
        return

    bar_open_ms = int(last_bar["open_time"])
    purge_expired_cooldowns(cur, bar_open_ms=bar_open_ms, now_utc_ms=_now_utc_ms())
    trend = int(last_bar["st_trend"])
    buy = bool(last_bar.get("buy_signal", False))
    sell = bool(last_bar.get("sell_signal", False))
    close_px = float(last_bar["close"])
    high_px = float(last_bar["high"])
    low_px = float(last_bar["low"])
    st_up = float(last_bar["st_up"])
    st_dn = float(last_bar["st_dn"])
    st_atr = float(last_bar["st_atr"]) if not pd.isna(last_bar["st_atr"]) else 0.0

    open_row = fetch_open_row(cur, symbol)
    state = get_indicator_state(cur, symbol)
    if _bar_already_handled(open_row, bar_open_ms=bar_open_ms, state=state, buy=buy, sell=sell):
        stats["skipped"].append(f"{symbol}:bar_already_processed")
        return

    if open_row is not None:
        closed_pp = _try_profit_protect_exit(
            cur,
            open_row,
            high_px=high_px,
            low_px=low_px,
            close_px=close_px,
            st_atr=st_atr,
            now_utc=now_utc,
            symbol=symbol,
            bar_open_ms=bar_open_ms,
            stats=stats,
            events=events,
        )
        if closed_pp:
            open_row = None

    if open_row is not None and "reverse_signal" in cfg.st_exit_modes_enabled():
        pos_side = str(open_row["side"])
        if pos_side == "LONG" and sell:
            closed_evt = _close_position(
                cur,
                open_row,
                exit_price=close_px,
                exit_rule="reverse_signal",
                now_utc=now_utc,
            )
            _emit_close(
                cur,
                closed_evt,
                symbol=symbol,
                bar_open_ms=bar_open_ms,
                now_utc=now_utc,
                stats=stats,
                events=events,
            )
            open_row = None
        elif pos_side == "SHORT" and buy:
            closed_evt = _close_position(
                cur,
                open_row,
                exit_price=close_px,
                exit_rule="reverse_signal",
                now_utc=now_utc,
            )
            _emit_close(
                cur,
                closed_evt,
                symbol=symbol,
                bar_open_ms=bar_open_ms,
                now_utc=now_utc,
                stats=stats,
                events=events,
            )
            open_row = None

    if buy or sell:
        stats["flips"] += 1

    closed = closed_bars_df(st_df, timeframe_ms=tf_ms)
    want_long, want_short = compute_entry_intent(
        trend=trend,
        buy=buy,
        sell=sell,
        closed=closed,
        open_row=open_row,
    )

    if want_long or want_short:
        if not in_universe:
            stats["skipped"].append(f"{symbol}:off_universe_no_new_entries")
            upsert_indicator_state(
                cur, symbol=symbol, bar_open_ms=bar_open_ms, trend=trend, updated_at_utc=now_utc
            )
            return
        if _daily_loss_exceeded(conn):
            stats["skipped"].append(f"{symbol}:daily_loss_cap")
            upsert_indicator_state(
                cur, symbol=symbol, bar_open_ms=bar_open_ms, trend=trend, updated_at_utc=now_utc
            )
            return
        if _daily_profit_lock_exceeded(conn):
            stats["skipped"].append(f"{symbol}:daily_profit_lock")
            upsert_indicator_state(
                cur, symbol=symbol, bar_open_ms=bar_open_ms, trend=trend, updated_at_utc=now_utc
            )
            return

        open_count = count_open_positions(cur)
        if (
            cfg.ST_MAX_OPEN_POSITIONS > 0
            and open_row is None
            and open_count >= cfg.ST_MAX_OPEN_POSITIONS
        ):
            stats["skipped"].append(f"{symbol}:max_open_positions")
            upsert_indicator_state(
                cur, symbol=symbol, bar_open_ms=bar_open_ms, trend=trend, updated_at_utc=now_utc
            )
            return

        if want_long:
            _try_open_side(
                cur,
                symbol=symbol,
                side="LONG",
                signal_type="BUY" if buy else "ENTRY_LONG",
                open_row=open_row,
                st_df=st_df,
                last_bar=last_bar,
                bar_open_ms=bar_open_ms,
                trend=trend,
                close_px=close_px,
                st_up=st_up,
                st_dn=st_dn,
                st_atr=st_atr,
                now_utc=now_utc,
                stats=stats,
                events=events,
                htf_cache=htf_cache,
                tf_ms=tf_ms,
            )
        elif want_short:
            _try_open_side(
                cur,
                symbol=symbol,
                side="SHORT",
                signal_type="SELL" if sell else "ENTRY_SHORT",
                open_row=open_row,
                st_df=st_df,
                last_bar=last_bar,
                bar_open_ms=bar_open_ms,
                trend=trend,
                close_px=close_px,
                st_up=st_up,
                st_dn=st_dn,
                st_atr=st_atr,
                now_utc=now_utc,
                stats=stats,
                events=events,
                htf_cache=htf_cache,
                tf_ms=tf_ms,
            )

    upsert_indicator_state(
        cur, symbol=symbol, bar_open_ms=bar_open_ms, trend=trend, updated_at_utc=now_utc
    )


def _scan_symbol_list(universe: List[str], conn: sqlite3.Connection) -> List[str]:
    """池内标的 + 池外仍持仓（仅续扫反转平仓，不新开）。"""
    allow = set(universe)
    cur = conn.cursor()
    carry = [s for s in list_open_position_symbols(cur) if s not in allow]
    ordered: List[str] = []
    seen: set[str] = set()
    for s in universe + carry:
        u = str(s).strip().upper()
        if u and u not in seen:
            seen.add(u)
            ordered.append(u)
    if carry:
        logger.info("[st] 池外续扫(仅平仓) %s", ",".join(carry))
    return ordered


def run_scan(*, notify: bool = True) -> Dict[str, Any]:
    from accumulation_radar import init_db

    universe = resolve_symbols()
    now_utc = _utc_now()
    stats: Dict[str, Any] = {
        "ok": True,
        "symbols": 0,
        "flips": 0,
        "opens": 0,
        "closes": 0,
        "skipped": [],
        "filter_blocked": 0,
        "filter_rejects": {},
        "exit_protect": {},
    }
    events: List[str] = []
    htf_cache: Dict[str, Optional[int]] = {}

    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        migrate_st_tables(conn.cursor())
        conn.commit()
        symbols = _scan_symbol_list(universe, conn)
        stats["symbols"] = len(symbols)
        if not symbols:
            msg = "[st] worth_watch_hot_oi 为空且无未平仓，跳过"
            print(msg)
            return {"ok": False, "error": "empty_universe", "message": msg}
        allow = set(universe)
        for i, sym in enumerate(symbols):
            if i > 0 and cfg.ST_INTER_SYMBOL_SLEEP_SEC > 0:
                time.sleep(cfg.ST_INTER_SYMBOL_SLEEP_SEC)
            try:
                _process_symbol(
                    conn,
                    sym,
                    now_utc=now_utc,
                    stats=stats,
                    events=events,
                    in_universe=sym in allow,
                    htf_cache=htf_cache,
                )
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.exception("[st] %s failed: %s", sym, e)
                stats["skipped"].append(f"{sym}:error:{type(e).__name__}")
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO st_runs (ran_at_utc, symbols_scanned, flips, opens, closes, skipped, detail_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_utc,
                len(symbols),
                stats["flips"],
                stats["opens"],
                stats["closes"],
                ",".join(stats["skipped"][:20]),
                json.dumps(
                    {
                        "events": events[:30],
                        "filter_blocked": stats.get("filter_blocked", 0),
                        "filter_rejects": stats.get("filter_rejects", {}),
                        "exit_protect": stats.get("exit_protect", {}),
                        "filter_enabled": cfg.ST_FILTER_ENABLED,
                    }
                ),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    summary = (
        f"ST scan {now_utc} symbols={len(symbols)} "
        f"opens={stats['opens']} closes={stats['closes']} flips={stats['flips']} "
        f"filtered={stats.get('filter_blocked', 0)}"
    )
    print(summary)
    for e in events:
        print(f"  · {e}")

    if notify and events and cfg.ST_TG_PUSH_MODE in ("actionable", "summary", "all"):
        if cfg.ST_TG_PUSH_MODE == "actionable" and stats["opens"] == 0 and stats["closes"] == 0:
            pass
        else:
            body = summary + "\n" + "\n".join(events[:15])
            send_tg(body)

    stats["events"] = events
    return stats


def run_resolve_only() -> Dict[str, Any]:
    """反转平仓在 scan 完成；resolve 仅做占位（可扩展 SL/TP）。"""
    return {"ok": True, "resolved": 0, "note": "reverse_signal exits on scan"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Supertrend signal scanner")
    ap.add_argument("--resolve-only", action="store_true")
    ap.add_argument("--no-tg", action="store_true")
    args = ap.parse_args()
    if args.resolve_only:
        out = run_resolve_only()
        print(out)
        return
    run_scan(notify=not args.no_tg)


if __name__ == "__main__":
    main()

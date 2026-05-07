#!/usr/bin/env python3
"""
Groq AI 交易计划：针对「重点关注 focus_watch ∪ 热度+收筹 worth_watch_heat_accum」标的拉上下文，
调用 groq_trading_plan，结果写入 ai_groq_trade_plan。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from groq_trading_plan import get_ai_trading_plan_groq

from accumulation_radar import (
    WORTH_HIGHLIGHT_CATEGORY_ORDER,
    WORTH_WATCH_TABLE_BY_CATEGORY,
    _parse_bpc_for_item,
    api_get,
    union_focus_watch_and_heat_accum_symbols,
)


def _now_cst_label() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M") + " CST"


def fetch_last_price(symbol: str) -> float:
    j = api_get("/fapi/v1/ticker/24hr", {"symbol": symbol})
    if not j:
        return 0.0
    try:
        return float(j.get("lastPrice") or 0)
    except (TypeError, ValueError):
        return 0.0


def resolve_archived_row(
    conn: sqlite3.Connection, symbol: str
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    返回 (coin, summary_line, detail_json, bpc_json, bpc_updated_cst)，来自 focus_watch 或任一 worth 表。
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT coin, summary_line, detail_json, bpc_json, bpc_updated_cst FROM focus_watch WHERE symbol = ? LIMIT 1",
        (symbol,),
    )
    r = cur.fetchone()
    if r:
        return (
            str(r[0] or ""),
            str(r[1] or "") or None,
            str(r[2] or "") if r[2] else None,
            str(r[3] or "") if r[3] else None,
            str(r[4] or "") if r[4] else None,
        )
    tbl_h = WORTH_WATCH_TABLE_BY_CATEGORY.get("heat_accum")
    if tbl_h:
        try:
            cur.execute(
                f"SELECT coin, summary_line, detail_json, bpc_json, bpc_updated_cst FROM {tbl_h} WHERE symbol = ? LIMIT 1",
                (symbol,),
            )
            rh = cur.fetchone()
            if rh:
                return (
                    str(rh[0] or ""),
                    str(rh[1] or "") or None,
                    str(rh[2] or "") if rh[2] else None,
                    str(rh[3] or "") if rh[3] else None,
                    str(rh[4] or "") if rh[4] else None,
                )
        except sqlite3.OperationalError:
            pass
    for cat in WORTH_HIGHLIGHT_CATEGORY_ORDER:
        tbl = WORTH_WATCH_TABLE_BY_CATEGORY.get(cat)
        if not tbl:
            continue
        try:
            cur.execute(
                f"SELECT coin, summary_line, detail_json, bpc_json, bpc_updated_cst FROM {tbl} WHERE symbol = ? LIMIT 1",
                (symbol,),
            )
            r2 = cur.fetchone()
            if r2:
                return (
                    str(r2[0] or ""),
                    str(r2[1] or "") or None,
                    str(r2[2] or "") if r2[2] else None,
                    str(r2[3] or "") if r2[3] else None,
                    str(r2[4] or "") if r2[4] else None,
                )
        except sqlite3.OperationalError:
            continue
    return None, None, None, None, None


def build_summary_context(
    symbol: str,
    coin: str,
    summary_line: Optional[str],
    detail_json: Optional[str],
    bpc_json: Optional[str],
    bpc_updated_cst: Optional[str],
) -> Tuple[str, str]:
    """返回 (phase_str, kline_summary_text)。"""
    bpc = _parse_bpc_for_item(bpc_json, bpc_updated_cst)
    phase = "unknown"
    if bpc and isinstance(bpc, dict):
        phase = str(bpc.get("phase") or "unknown")
        parts: List[str] = []
        if summary_line:
            parts.append(f"摘要: {summary_line}")
        if detail_json:
            try:
                dj = json.loads(detail_json)
                if isinstance(dj, dict) and dj:
                    parts.append(f"快照指标(节选): {json.dumps(dj, ensure_ascii=False)[:1200]}")
            except Exception:
                parts.append(f"快照: {detail_json[:400]}")
        pzh = bpc.get("phase_zh")
        crzh = bpc.get("continuation_reason_zh")
        parts.append(
            f"1H结构: phase={phase}"
            + (f", {pzh}" if pzh else "")
            + (f", 延续形态={crzh}" if crzh and phase == "continuation" else "")
        )
        if bpc.get("breakout_level") is not None:
            parts.append(f"突破参考位: {bpc.get('breakout_level')}")
        text = "\n".join(parts) if parts else f"标的 {symbol} ({coin or '?'}), 无归档摘要，仅依赖现价。"
        return phase, text
    parts2: List[str] = []
    if summary_line:
        parts2.append(f"摘要: {summary_line}")
    if detail_json:
        parts2.append(f"快照: {detail_json[:800]}")
    text2 = "\n".join(parts2) if parts2 else f"标的 {symbol} ({coin or '?'}), 库内无 BPC/摘要，仅市价上下文。"
    return "unknown", text2


def persist_plan(
    conn: sqlite3.Connection,
    symbol: str,
    coin: str,
    price: float,
    phase: str,
    summary_ctx: str,
    result: Dict[str, Any],
) -> None:
    cur = conn.cursor()
    now = _now_cst_label()
    if result.get("ok"):
        cur.execute(
            """
            INSERT OR REPLACE INTO ai_groq_trade_plan (
                symbol, coin, generated_cst, current_price, phase,
                buy_zone_bottom, buy_zone_top, stop_loss, take_profit_1,
                reasoning, model, ok, error_detail, summary_context
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, ?)
            """,
            (
                symbol,
                coin,
                now,
                price,
                phase,
                float(result["buy_zone_bottom"]),
                float(result["buy_zone_top"]),
                float(result["stop_loss"]),
                float(result["take_profit_1"]),
                str(result.get("reasoning") or ""),
                str(result.get("model") or ""),
                summary_ctx[:8000] if summary_ctx else "",
            ),
        )
    else:
        cur.execute(
            """
            INSERT OR REPLACE INTO ai_groq_trade_plan (
                symbol, coin, generated_cst, current_price, phase,
                buy_zone_bottom, buy_zone_top, stop_loss, take_profit_1,
                reasoning, model, ok, error_detail, summary_context
            ) VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, 0, ?, ?)
            """,
            (
                symbol,
                coin,
                now,
                price,
                phase,
                json.dumps({k: result.get(k) for k in ("error", "detail")}, ensure_ascii=False)[:2000],
                summary_ctx[:8000] if summary_ctx else "",
            ),
        )
    conn.commit()


def run_ai_plan_for_symbol(conn: sqlite3.Connection, symbol: str) -> Dict[str, Any]:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return {"symbol": sym, "ok": False, "error": "empty_symbol"}
    coin_t, summ, det, bj, bu = resolve_archived_row(conn, sym)
    coin = coin_t or (sym[:-4] if sym.endswith("USDT") else sym)
    phase, summary_text = build_summary_context(sym, coin, summ, det, bj, bu)
    price = fetch_last_price(sym)
    if price <= 0:
        err = {"ok": False, "error": "no_last_price", "symbol": sym}
        persist_plan(conn, sym, coin, 0.0, phase, summary_text, err)
        return {"symbol": sym, **err}

    out = get_ai_trading_plan_groq(sym, price, summary_text, phase)
    persist_plan(conn, sym, coin, price, phase, summary_text, out)
    return {"symbol": sym, **out}


def run_ai_plans_for_focus_and_heat_accum(conn: sqlite3.Connection, *, sleep_s: float = 1.0) -> Dict[str, Any]:
    """批量：仅 focus_watch ∪ worth_watch_heat_accum（热度+收筹）。
    每批最多 GROQ_AI_BATCH_SIZE（默认 10）个标的；若还有剩余，间隔 GROQ_AI_BATCH_GAP_SEC（默认 120）秒再跑下一批。
    """
    batch_size = max(1, int(os.getenv("GROQ_AI_BATCH_SIZE", "10")))
    batch_gap_s = max(0.0, float(os.getenv("GROQ_AI_BATCH_GAP_SEC", "120")))
    sleep_between = float(os.getenv("GROQ_AI_SLEEP_SEC", str(sleep_s)))

    syms = union_focus_watch_and_heat_accum_symbols(conn)
    items: List[Dict[str, Any]] = []
    n = len(syms)
    for start in range(0, n, batch_size):
        chunk = syms[start : start + batch_size]
        for sym in chunk:
            items.append(run_ai_plan_for_symbol(conn, sym))
            time.sleep(max(0.05, sleep_between))
        if start + batch_size < n:
            done = min(start + batch_size, n)
            print(
                f"[Groq AI] 已分析 {done}/{n} 个标的，{batch_gap_s:.0f}s 后继续下一批（每批≤{batch_size}）…",
                flush=True,
            )
            time.sleep(batch_gap_s)

    ok_n = sum(1 for x in items if x.get("ok"))
    batch_count = (n + batch_size - 1) // batch_size if n else 0
    return {
        "ok": True,
        "generated_at_cst": _now_cst_label(),
        "symbols": syms,
        "success_count": ok_n,
        "fail_count": len(items) - ok_n,
        "items": items,
        "batch_size": batch_size,
        "batch_gap_sec": batch_gap_s,
        "batch_count": batch_count,
    }


def load_ai_trade_plans_from_db(conn: sqlite3.Connection) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT symbol, coin, generated_cst, current_price, phase,
               buy_zone_bottom, buy_zone_top, stop_loss, take_profit_1,
               reasoning, model, ok, error_detail, summary_context
        FROM ai_groq_trade_plan
        ORDER BY symbol ASC
        """
    )
    rows = []
    for r in cur.fetchall():
        rows.append(
            {
                "symbol": r[0],
                "coin": r[1],
                "generated_cst": r[2],
                "current_price": r[3],
                "phase": r[4],
                "buy_zone_bottom": r[5],
                "buy_zone_top": r[6],
                "stop_loss": r[7],
                "take_profit_1": r[8],
                "reasoning": r[9],
                "model": r[10],
                "ok": bool(r[11]),
                "error_detail": r[12],
                "summary_context": r[13],
            }
        )
    return {"ok": True, "items": rows, "updated_hint": _now_cst_label()}

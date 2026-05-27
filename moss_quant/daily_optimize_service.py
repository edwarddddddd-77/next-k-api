"""每日全宇宙标的自动寻优 + 同步 daily_auto Profile。"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from moss_quant import config as cfg
from moss_quant.db import (
    DAILY_PROFILE_SOURCE,
    _utc_now,
    daily_profile_name,
    get_daily_profile_by_symbol,
    get_profile,
    row_to_profile,
)
from moss_quant.optimize_service import run_strategy_optimize
from moss_quant.params import build_initial_params
from moss_quant.universe import list_universe

logger = logging.getLogger(__name__)


def _open_db():
    from accumulation_radar import init_db

    return init_db()


def _create_batch(
    *,
    symbols_total: int,
    capital: float,
    now: str,
) -> int:
    conn = _open_db()
    try:
        cur = conn.execute(
            """INSERT INTO moss_daily_optimize_batches(
                   ran_at_utc, status, symbols_total, capital, data_source)
               VALUES (?,?,?,?,?)""",
            (
                now,
                "running",
                symbols_total,
                capital,
                cfg.MOSS_QUANT_DATA_SOURCE,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _insert_batch_item(
    batch_id: int,
    *,
    symbol: str,
    template: Optional[str],
    tactical: dict,
    summary: dict,
    score: float,
) -> None:
    conn = _open_db()
    try:
        conn.execute(
            """INSERT INTO moss_daily_optimize_items(
                   batch_id, symbol, template, tactical_params_json,
                   summary_json, score)
               VALUES (?,?,?,?,?,?)""",
            (
                batch_id,
                symbol,
                template,
                json.dumps(tactical, ensure_ascii=False),
                json.dumps(summary, ensure_ascii=False),
                score,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _finalize_batch(
    batch_id: int,
    *,
    status: str,
    symbols_ok: int,
    kline_start: Optional[str],
    kline_end: Optional[str],
    error: Optional[str] = None,
) -> None:
    conn = _open_db()
    try:
        if error:
            conn.execute(
                """UPDATE moss_daily_optimize_batches SET
                   status=?, finished_at_utc=?, symbols_ok=?, error=?
                   WHERE id=?""",
                (status, _utc_now(), symbols_ok, error, batch_id),
            )
        else:
            conn.execute(
                """UPDATE moss_daily_optimize_batches SET
                   status=?, finished_at_utc=?, symbols_ok=?,
                   kline_start=?, kline_end=?
                   WHERE id=?""",
                (
                    status,
                    _utc_now(),
                    symbols_ok,
                    kline_start,
                    kline_end,
                    batch_id,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def run_daily_optimize_batch(
    *,
    capital: Optional[float] = None,
    refresh_klines: Optional[bool] = None,
    apply_profiles: Optional[bool] = None,
) -> Dict[str, Any]:
    """对 universe 全部标的寻优；每标的单独 commit，避免长时间锁库。"""
    capital = float(capital or cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    refresh = (
        cfg.MOSS_QUANT_DAILY_OPTIMIZE_REFRESH
        if refresh_klines is None
        else refresh_klines
    )
    apply = (
        cfg.MOSS_QUANT_DAILY_OPTIMIZE_APPLY_PROFILES
        if apply_profiles is None
        else apply_profiles
    )
    symbols = [u["symbol"] for u in list_universe()]
    now = _utc_now()
    batch_id = _create_batch(symbols_total=len(symbols), capital=capital, now=now)

    items: List[Dict[str, Any]] = []
    kline_start = None
    kline_end = None
    try:
        for i, sym in enumerate(symbols):
            sym = str(sym).upper()
            logger.info(
                "[moss] daily optimize %s/%s %s",
                i + 1,
                len(symbols),
                sym,
            )
            try:
                out = run_strategy_optimize(
                    symbol=sym,
                    capital=capital,
                    refresh_klines=refresh,
                    top_n=1,
                )
                best = out.get("best")
                if not best or not best.get("summary"):
                    items.append(
                        {
                            "symbol": sym,
                            "error": "no_valid_result",
                            "score": -999.0,
                        }
                    )
                    _insert_batch_item(
                        batch_id,
                        symbol=sym,
                        template=None,
                        tactical={},
                        summary={"error": "no_valid_result"},
                        score=-999.0,
                    )
                    continue
                if kline_start is None and out.get("kline_start"):
                    kline_start = out.get("kline_start")
                    kline_end = out.get("kline_end")
                summary = best["summary"]
                tact = best.get("tactical_params") or {}
                score = float(best.get("score") or 0)
                _insert_batch_item(
                    batch_id,
                    symbol=sym,
                    template=best.get("template"),
                    tactical=tact,
                    summary=summary,
                    score=score,
                )
                items.append(
                    {
                        "symbol": sym,
                        "template": best.get("template"),
                        "tactical_params": tact,
                        "summary": summary,
                        "score": score,
                    }
                )
            except Exception as e:
                logger.warning("[moss] daily optimize %s failed: %s", sym, e)
                _insert_batch_item(
                    batch_id,
                    symbol=sym,
                    template=None,
                    tactical={},
                    summary={"error": str(e)},
                    score=-999.0,
                )
                items.append({"symbol": sym, "error": str(e)})

        profile_map: Dict[str, int] = {}
        if apply:
            conn = _open_db()
            try:
                profile_map = sync_daily_profiles(conn, batch_id)
            finally:
                conn.close()

        symbols_ok = len([x for x in items if x.get("summary")])
        _finalize_batch(
            batch_id,
            status="completed",
            symbols_ok=symbols_ok,
            kline_start=kline_start,
            kline_end=kline_end,
        )
        return {
            "ok": True,
            "batch_id": batch_id,
            "symbols_total": len(symbols),
            "symbols_ok": symbols_ok,
            "items": items,
            "profiles": profile_map,
            "apply_profiles": apply,
        }
    except Exception as e:
        logger.exception("daily optimize batch failed")
        _finalize_batch(
            batch_id,
            status="failed",
            symbols_ok=len([x for x in items if x.get("summary")]),
            kline_start=kline_start,
            kline_end=kline_end,
            error=str(e),
        )
        raise


def sync_daily_profiles(conn, batch_id: int) -> Dict[str, int]:
    """根据 batch 结果为每个标的 upsert 一个启用的 daily_auto Profile。"""
    import sqlite3

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT * FROM moss_daily_optimize_items
           WHERE batch_id = ? AND summary_json IS NOT NULL""",
        (int(batch_id),),
    ).fetchall()
    out: Dict[str, int] = {}
    now = _utc_now()
    for row in rows:
        sym = str(row["symbol"]).upper()
        summary = json.loads(row["summary_json"] or "{}")
        if summary.get("error"):
            continue
        template = str(row["template"] or "balanced")
        tactical = json.loads(row["tactical_params_json"] or "{}")
        initial = build_initial_params(template=template)

        existing = get_daily_profile_by_symbol(conn, sym)
        if not existing:
            any_row = conn.execute(
                "SELECT * FROM moss_profiles WHERE symbol = ? ORDER BY id DESC LIMIT 1",
                (sym,),
            ).fetchone()
            existing = row_to_profile(any_row) if any_row else None
        equity = cfg.MOSS_QUANT_DEFAULT_CAPITAL
        if existing:
            conn.execute(
                """UPDATE moss_profiles SET
                   name=?, symbol=?, template=?, enabled=1, profile_source=?,
                   initial_params_json=?, tactical_params_json=?,
                   updated_at_utc=?
                   WHERE id=?""",
                (
                    daily_profile_name(sym),
                    sym,
                    template,
                    DAILY_PROFILE_SOURCE,
                    json.dumps(initial, ensure_ascii=False),
                    json.dumps(tactical, ensure_ascii=False),
                    now,
                    int(existing["id"]),
                ),
            )
            pid = int(existing["id"])
        else:
            cur = conn.execute(
                """INSERT INTO moss_profiles(
                       name, symbol, template, enabled, profile_source,
                       initial_params_json, tactical_params_json,
                       virtual_equity_usdt, evolution_enabled,
                       created_at_utc, updated_at_utc)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    daily_profile_name(sym),
                    sym,
                    template,
                    1,
                    DAILY_PROFILE_SOURCE,
                    json.dumps(initial, ensure_ascii=False),
                    json.dumps(tactical, ensure_ascii=False),
                    float(equity),
                    0,
                    now,
                    now,
                ),
            )
            pid = int(cur.lastrowid)
        conn.execute(
            """UPDATE moss_profiles SET enabled=0, updated_at_utc=?
               WHERE symbol=? AND id<>? AND enabled=1""",
            (now, sym, pid),
        )
        conn.execute(
            "UPDATE moss_daily_optimize_items SET profile_id=? WHERE id=?",
            (pid, int(row["id"])),
        )
        out[sym] = pid
        conn.commit()
    return out


def get_latest_daily_batch(conn) -> Optional[Dict[str, Any]]:
    import sqlite3

    conn.row_factory = sqlite3.Row
    batch = conn.execute(
        """SELECT * FROM moss_daily_optimize_batches
           ORDER BY id DESC LIMIT 1"""
    ).fetchone()
    if not batch:
        return None
    b = dict(batch)
    items = conn.execute(
        """SELECT * FROM moss_daily_optimize_items
           WHERE batch_id = ? ORDER BY score DESC, symbol ASC""",
        (int(b["id"]),),
    ).fetchall()
    rows_out: List[Dict[str, Any]] = []
    for r in items:
        d = dict(r)
        d["tactical_params"] = json.loads(d.pop("tactical_params_json") or "{}")
        d["summary"] = json.loads(d.pop("summary_json") or "{}")
        if d.get("profile_id"):
            prof = get_profile(conn, int(d["profile_id"]))
            d["profile"] = prof
        rows_out.append(d)
    b["items"] = rows_out
    return b


def is_daily_batch_running(conn) -> bool:
    row = conn.execute(
        """SELECT id FROM moss_daily_optimize_batches
           WHERE status = 'running' ORDER BY id DESC LIMIT 1"""
    ).fetchone()
    return row is not None

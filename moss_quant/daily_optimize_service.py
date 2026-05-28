"""每日全宇宙标的自动寻优 + 同步 daily_auto Profile。"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from moss_quant import config as cfg
from moss_quant.db import (
    FROM_DAILY_PROFILE_SOURCE,
    _utc_now,
    get_profile,
    get_profile_by_symbol,
    row_to_profile,
)
from moss_quant.daily_auto_enable import evaluate_profile_auto_enable
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
            sym_refresh = refresh
            if (
                refresh
                and cfg.MOSS_QUANT_DATA_SOURCE == "binance"
                and not cfg.MOSS_QUANT_DAILY_OPTIMIZE_BINANCE_REFRESH_ALL
                and i > 0
            ):
                sym_refresh = False
            try:
                out = run_strategy_optimize(
                    symbol=sym,
                    capital=capital,
                    refresh_klines=sym_refresh,
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
                summary = {
                    **best["summary"],
                    **evaluate_profile_auto_enable(best["summary"]),
                }
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

        annotate_stats: Dict[str, Any] = {}
        if apply:
            conn = _open_db()
            try:
                annotate_stats = annotate_daily_batch_items(conn, batch_id)
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
            "annotate": annotate_stats,
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


def annotate_daily_batch_items(conn, batch_id: int) -> Dict[str, Any]:
    """为寻优批次写入达标/不达标与原因；不创建或修改纸面 Profile。"""
    import sqlite3

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT * FROM moss_daily_optimize_items
           WHERE batch_id = ? AND summary_json IS NOT NULL""",
        (int(batch_id),),
    ).fetchall()
    pass_n = 0
    fail_n = 0
    for row in rows:
        sym = str(row["symbol"]).upper()
        summary = json.loads(row["summary_json"] or "{}")
        gate = evaluate_profile_auto_enable(summary)
        summary = {**summary, **gate}
        if gate.get("auto_enabled"):
            pass_n += 1
        else:
            fail_n += 1
        linked = get_profile_by_symbol(conn, sym)
        profile_id = int(linked["id"]) if linked else None
        conn.execute(
            """UPDATE moss_daily_optimize_items
               SET summary_json=?, profile_id=? WHERE id=?""",
            (
                json.dumps(summary, ensure_ascii=False),
                profile_id,
                int(row["id"]),
            ),
        )
        conn.commit()
    logger.info(
        "[moss] daily annotate batch=%s pass=%s fail=%s",
        batch_id,
        pass_n,
        fail_n,
    )
    return {"pass": pass_n, "fail": fail_n, "total": pass_n + fail_n}


def get_latest_daily_item_for_symbol(
    conn, symbol: str
) -> Optional[Dict[str, Any]]:
    import sqlite3

    sym = str(symbol).strip().upper()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT i.* FROM moss_daily_optimize_items i
           INNER JOIN moss_daily_optimize_batches b ON b.id = i.batch_id
           WHERE i.symbol = ? AND b.status = 'completed'
             AND i.summary_json IS NOT NULL
           ORDER BY b.id DESC, i.id DESC LIMIT 1""",
        (sym,),
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["tactical_params"] = json.loads(d.pop("tactical_params_json") or "{}")
    d["summary"] = json.loads(d.pop("summary_json") or "{}")
    return d


def import_profile_from_daily(
    conn,
    symbol: str,
    *,
    enabled: bool = True,
    name: Optional[str] = None,
    update_existing: bool = True,
) -> Dict[str, Any]:
    """从最近一次每日寻优结果创建或更新纸面 Profile（用户主动操作）。"""
    from moss_quant import config as mq_cfg
    from moss_quant.universe import active_symbols_taken, is_symbol_allowed

    sym = str(symbol).strip().upper()
    if not is_symbol_allowed(sym):
        raise ValueError("symbol_not_allowed")
    item = get_latest_daily_item_for_symbol(conn, sym)
    if not item or item.get("summary", {}).get("error"):
        raise ValueError("daily_item_not_found")
    template = str(item.get("template") or "balanced")
    tactical = item.get("tactical_params") or {}
    initial = build_initial_params(template=template)
    now = _utc_now()
    equity = float(mq_cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    prof_name = (name or "").strip() or ("from-daily-" + sym)

    existing = get_profile_by_symbol(conn, sym)
    if existing and not update_existing:
        raise ValueError("profile_already_exists")

    if enabled:
        from moss_quant.db import count_enabled_profiles

        if not existing or not existing.get("enabled"):
            if count_enabled_profiles(conn) >= mq_cfg.MOSS_QUANT_MAX_ACTIVE_PROFILES:
                raise ValueError("max_active_profiles_reached")
        taken = active_symbols_taken(
            conn, exclude_profile_id=int(existing["id"]) if existing else None
        )
        if sym in taken and (not existing or not existing.get("enabled")):
            raise ValueError("symbol_already_active")

    if existing:
        conn.execute(
            """UPDATE moss_profiles SET
               name=?, template=?, profile_source=?,
               initial_params_json=?, tactical_params_json=?,
               enabled=?, updated_at_utc=?
               WHERE id=?""",
            (
                prof_name,
                template,
                FROM_DAILY_PROFILE_SOURCE,
                json.dumps(initial, ensure_ascii=False),
                json.dumps(tactical, ensure_ascii=False),
                1 if enabled else 0,
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
                prof_name,
                sym,
                template,
                1 if enabled else 0,
                FROM_DAILY_PROFILE_SOURCE,
                json.dumps(initial, ensure_ascii=False),
                json.dumps(tactical, ensure_ascii=False),
                equity,
                0,
                now,
                now,
            ),
        )
        pid = int(cur.lastrowid)

    conn.execute(
        """UPDATE moss_daily_optimize_items SET profile_id=?
           WHERE id=?""",
        (pid, int(item["id"])),
    )
    conn.commit()
    return get_profile(conn, pid) or {}


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


def reconcile_stale_daily_batches(conn) -> int:
    """DB 仍为 running 但本进程未持有寻优锁 → 后台已停，收尾为 failed。"""
    try:
        import worker_tasks as wt

        if wt.moss_daily_optimize_busy():
            return 0
    except Exception:
        pass

    rows = conn.execute(
        "SELECT id FROM moss_daily_optimize_batches WHERE status = 'running'"
    ).fetchall()
    if not rows:
        return 0

    now = _utc_now()
    n = 0
    for row in rows:
        bid = int(row["id"] if hasattr(row, "keys") else row[0])
        ok = int(
            conn.execute(
                """SELECT COUNT(*) FROM moss_daily_optimize_items
                   WHERE batch_id=? AND summary_json IS NOT NULL""",
                (bid,),
            ).fetchone()[0]
            or 0
        )
        conn.execute(
            """UPDATE moss_daily_optimize_batches SET
               status='failed', finished_at_utc=?, symbols_ok=?, error=?
               WHERE id=?""",
            (now, ok, "后台已停止或进程中断", bid),
        )
        n += 1
    if n:
        conn.commit()
        logger.warning("[moss] reconciled %s stale daily optimize batch(es)", n)
    return n


def is_daily_optimize_in_progress(conn) -> bool:
    """寻优是否进行中（进程锁或 DB running；会先清理僵死 running）。"""
    try:
        import worker_tasks as wt

        if wt.moss_daily_optimize_busy():
            return True
    except Exception:
        pass
    reconcile_stale_daily_batches(conn)
    return is_daily_batch_running(conn)

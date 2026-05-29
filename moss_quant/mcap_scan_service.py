"""币安市值 Top100（排除稳定币与每日宇宙）扩展寻优。"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from moss_quant import config as cfg
from moss_quant.binance_mcap_universe import build_mcap_scan_candidates
from moss_quant.daily_auto_enable import evaluate_profile_auto_enable
from moss_quant.db import _utc_now
from moss_quant.optimize_service import run_strategy_optimize

logger = logging.getLogger(__name__)

DISPLAY_TOP_N = 15


def _open_db():
    from accumulation_radar import init_db

    return init_db()


def _create_batch(*, symbols_total: int, capital: float, now: str) -> int:
    conn = _open_db()
    try:
        cur = conn.execute(
            """INSERT INTO moss_mcap_scan_batches(
                   ran_at_utc, status, symbols_total, capital, data_source,
                   display_top_n, mcap_pool_limit)
               VALUES (?,?,?,?,?,?,?)""",
            (
                now,
                "running",
                symbols_total,
                capital,
                cfg.MOSS_QUANT_DATA_SOURCE,
                DISPLAY_TOP_N,
                cfg.MOSS_QUANT_MCAP_SCAN_POOL_LIMIT,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _insert_item(
    batch_id: int,
    *,
    symbol: str,
    market_cap_usd: float,
    template: Optional[str],
    tactical: dict,
    summary: dict,
    score: float,
    mcap_rank: int,
) -> None:
    conn = _open_db()
    try:
        conn.execute(
            """INSERT INTO moss_mcap_scan_items(
                   batch_id, symbol, market_cap_usd, mcap_rank,
                   template, tactical_params_json, summary_json, score)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                batch_id,
                symbol,
                market_cap_usd,
                mcap_rank,
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
                """UPDATE moss_mcap_scan_batches SET
                   status=?, finished_at_utc=?, symbols_ok=?, error=?
                   WHERE id=?""",
                (status, _utc_now(), symbols_ok, error, batch_id),
            )
        else:
            conn.execute(
                """UPDATE moss_mcap_scan_batches SET
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


def run_mcap_scan_batch(
    *,
    capital: Optional[float] = None,
    refresh_klines: Optional[bool] = None,
    mcap_pool_limit: Optional[int] = None,
) -> Dict[str, Any]:
    capital = float(capital or cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    refresh = (
        cfg.MOSS_QUANT_MCAP_SCAN_REFRESH
        if refresh_klines is None
        else refresh_klines
    )
    pool_limit = int(
        mcap_pool_limit or cfg.MOSS_QUANT_MCAP_SCAN_POOL_LIMIT
    )
    now = _utc_now()

    try:
        candidates = build_mcap_scan_candidates(mcap_limit=pool_limit)
    except Exception as e:
        logger.exception("[moss] mcap scan candidate build failed")
        raise

    batch_id = _create_batch(
        symbols_total=len(candidates), capital=capital, now=now
    )
    items: List[Dict[str, Any]] = []
    kline_start = None
    kline_end = None

    try:
        for i, cand in enumerate(candidates):
            sym = str(cand["symbol"]).upper()
            mcap_rank = i + 1
            sym_refresh = refresh
            if (
                refresh
                and cfg.MOSS_QUANT_DATA_SOURCE == "binance"
                and i > 0
            ):
                sym_refresh = False
            logger.info(
                "[moss] mcap scan %s/%s %s mcap_rank=%s",
                i + 1,
                len(candidates),
                sym,
                mcap_rank,
            )
            try:
                out = run_strategy_optimize(
                    symbol=sym,
                    capital=capital,
                    refresh_klines=sym_refresh,
                    top_n=1,
                )
                best = out.get("best")
                if not best or not best.get("summary"):
                    _insert_item(
                        batch_id,
                        symbol=sym,
                        market_cap_usd=float(cand.get("market_cap_usd") or 0),
                        template=None,
                        tactical={},
                        summary={"error": "no_valid_result"},
                        score=-999.0,
                        mcap_rank=mcap_rank,
                    )
                    items.append(
                        {
                            "symbol": sym,
                            "market_cap_usd": cand.get("market_cap_usd"),
                            "mcap_rank": mcap_rank,
                            "summary": {"error": "no_valid_result"},
                            "error": "no_valid_result",
                        }
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
                _insert_item(
                    batch_id,
                    symbol=sym,
                    market_cap_usd=float(cand.get("market_cap_usd") or 0),
                    template=best.get("template"),
                    tactical=tact,
                    summary=summary,
                    score=score,
                    mcap_rank=mcap_rank,
                )
                items.append(
                    {
                        "symbol": sym,
                        "market_cap_usd": cand.get("market_cap_usd"),
                        "mcap_rank": mcap_rank,
                        "template": best.get("template"),
                        "tactical_params": tact,
                        "summary": summary,
                        "score": score,
                    }
                )
            except Exception as e:
                logger.warning("[moss] mcap scan %s failed: %s", sym, e)
                _insert_item(
                    batch_id,
                    symbol=sym,
                    market_cap_usd=float(cand.get("market_cap_usd") or 0),
                    template=None,
                    tactical={},
                    summary={"error": str(e)},
                    score=-999.0,
                    mcap_rank=mcap_rank,
                )
                items.append(
                    {
                        "symbol": sym,
                        "market_cap_usd": cand.get("market_cap_usd"),
                        "mcap_rank": mcap_rank,
                        "summary": {"error": str(e)},
                        "error": str(e),
                    }
                )

        symbols_ok = len(
            [
                x
                for x in items
                if (x.get("summary") or {}).get("error") is None and not x.get("error")
            ]
        )
        _finalize_batch(
            batch_id,
            status="completed",
            symbols_ok=symbols_ok,
            kline_start=kline_start,
            kline_end=kline_end,
        )
        top = top_qualified_mcap_items(items, DISPLAY_TOP_N)
        return {
            "ok": True,
            "batch_id": batch_id,
            "symbols_total": len(candidates),
            "symbols_ok": symbols_ok,
            "qualified_count": len(
                [x for x in items if _item_passes_daily_gate(x)]
            ),
            "display_top_n": DISPLAY_TOP_N,
            "top": top,
            "items": items,
        }
    except Exception as e:
        logger.exception("mcap scan batch failed")
        _finalize_batch(
            batch_id,
            status="failed",
            symbols_ok=len([x for x in items if x.get("summary")]),
            kline_start=kline_start,
            kline_end=kline_end,
            error=str(e),
        )
        raise


def _item_passes_daily_gate(item: Dict[str, Any]) -> bool:
    if item.get("error"):
        return False
    summary = item.get("summary") or {}
    if summary.get("error"):
        return False
    if summary.get("auto_enabled") is True:
        return True
    if summary.get("auto_enabled") is False:
        return False
    return bool(evaluate_profile_auto_enable(summary).get("auto_enabled"))


def top_qualified_mcap_items(
    items: List[Dict[str, Any]], n: int
) -> List[Dict[str, Any]]:
    """达标（同每日寻优门槛）后按得分取前 n。"""
    qualified = [x for x in items if _item_passes_daily_gate(x)]
    qualified.sort(key=lambda r: -float(r.get("score") or -999))
    return qualified[: max(0, int(n))]


def _top_items_from_list(
    items: List[Dict[str, Any]], n: int
) -> List[Dict[str, Any]]:
    return top_qualified_mcap_items(items, n)


def get_latest_mcap_scan_batch(conn) -> Optional[Dict[str, Any]]:
    import sqlite3

    conn.row_factory = sqlite3.Row
    batch = conn.execute(
        """SELECT * FROM moss_mcap_scan_batches ORDER BY id DESC LIMIT 1"""
    ).fetchone()
    if not batch:
        return None
    b = dict(batch)
    items = conn.execute(
        """SELECT * FROM moss_mcap_scan_items
           WHERE batch_id = ? ORDER BY score DESC, mcap_rank ASC""",
        (int(b["id"]),),
    ).fetchall()
    rows_out: List[Dict[str, Any]] = []
    for r in items:
        d = dict(r)
        d["tactical_params"] = json.loads(d.pop("tactical_params_json") or "{}")
        summary = json.loads(d.pop("summary_json") or "{}")
        if summary and not summary.get("error") and "auto_enabled" not in summary:
            summary = {**summary, **evaluate_profile_auto_enable(summary)}
        d["summary"] = summary
        rows_out.append(d)
    b["items"] = rows_out
    limit = int(b.get("display_top_n") or DISPLAY_TOP_N)
    flat = [
        {
            "symbol": d["symbol"],
            "market_cap_usd": d.get("market_cap_usd"),
            "mcap_rank": d.get("mcap_rank"),
            "template": d.get("template"),
            "tactical_params": d.get("tactical_params"),
            "summary": d.get("summary"),
            "score": d.get("score"),
        }
        for d in rows_out
    ]
    b["qualified_count"] = len([x for x in flat if _item_passes_daily_gate(x)])
    b["top"] = top_qualified_mcap_items(flat, limit)
    return b


def is_mcap_batch_running(conn) -> bool:
    row = conn.execute(
        """SELECT id FROM moss_mcap_scan_batches
           WHERE status = 'running' ORDER BY id DESC LIMIT 1"""
    ).fetchone()
    return row is not None


def reconcile_stale_mcap_batches(conn) -> int:
    try:
        import worker_tasks as wt

        if wt.moss_mcap_scan_busy():
            return 0
    except Exception:
        pass

    rows = conn.execute(
        "SELECT id FROM moss_mcap_scan_batches WHERE status = 'running'"
    ).fetchall()
    if not rows:
        return 0
    now = _utc_now()
    n = 0
    for row in rows:
        bid = int(row[0])
        ok = int(
            conn.execute(
                """SELECT COUNT(*) FROM moss_mcap_scan_items
                   WHERE batch_id=? AND summary_json IS NOT NULL""",
                (bid,),
            ).fetchone()[0]
            or 0
        )
        conn.execute(
            """UPDATE moss_mcap_scan_batches SET
               status='failed', finished_at_utc=?, symbols_ok=?, error=?
               WHERE id=?""",
            (now, ok, "后台已停止或进程中断", bid),
        )
        n += 1
    if n:
        conn.commit()
    return n


def is_mcap_scan_in_progress(conn) -> bool:
    try:
        import worker_tasks as wt

        if wt.moss_mcap_scan_busy():
            return True
    except Exception:
        pass
    reconcile_stale_mcap_batches(conn)
    return is_mcap_batch_running(conn)

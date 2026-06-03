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
from moss_quant.optimize_policy import (
    can_sync_profile_params,
    enrich_summary,
    risk_scale_for_rank,
)
from moss_quant.optimize_service import run_strategy_optimize
from moss_quant.params import TACTICAL_FLOAT_FIELDS, build_initial_params
from moss_quant.universe import list_daily_core_universe

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
    """对 moss_daily_core_symbols 全目录逐标的寻优；每标的单独 commit。"""
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
    now = _utc_now()
    conn0 = _open_db()
    try:
        symbols = [u["symbol"] for u in list_daily_core_universe(conn0)]
    finally:
        conn0.close()

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
                summary = dict(best.get("summary") or {})
                tact = best.get("tactical_params") or {}
                score = float(best.get("score") or summary.get("train_score") or 0)
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
        sync_stats: Dict[str, Any] = {}
        governance_stats: Dict[str, Any] = {}
        if apply:
            conn = _open_db()
            try:
                annotate_stats = annotate_daily_batch_items(conn, batch_id)
                sync_stats = sync_enabled_profiles_from_batch(conn, batch_id)
                from moss_quant.pool_governance import apply_pool_governance

                governance_stats = apply_pool_governance(conn, batch_id)
            finally:
                conn.close()

        symbols_ok = len(
            [
                x
                for x in items
                if x.get("summary") and not (x.get("summary") or {}).get("error")
            ]
        )
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
            "sync_profiles": sync_stats,
            "pool_governance": governance_stats,
            "apply_profiles": apply,
        }
    except Exception as e:
        logger.exception("daily optimize batch failed")
        _finalize_batch(
            batch_id,
            status="failed",
            symbols_ok=len(
                [
                    x
                    for x in items
                    if x.get("summary") and not (x.get("summary") or {}).get("error")
                ]
            ),
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
        linked = get_profile_by_symbol(conn, sym)
        profile_id = int(linked["id"]) if linked else None
        cap = None
        if linked:
            cap = float(linked.get("virtual_equity_usdt") or 0) or None
        summary = enrich_summary(
            json.loads(row["summary_json"] or "{}"),
            conn=conn,
            profile_id=profile_id,
            profile_capital=cap,
        )
        if summary.get("auto_enabled"):
            pass_n += 1
        else:
            fail_n += 1
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


def _norm_tactical_for_compare(tactical: Optional[dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in TACTICAL_FLOAT_FIELDS:
        if key not in (tactical or {}):
            continue
        try:
            out[key] = round(float(tactical[key]), 6)
        except (TypeError, ValueError):
            continue
    return out


def _profile_matches_batch_optimal(
    profile: Dict[str, Any], item: Dict[str, Any]
) -> bool:
    opt_tpl = str(item.get("template") or "").strip().lower()
    cur_tpl = str(profile.get("template") or "").strip().lower()
    if opt_tpl != cur_tpl:
        return False
    prof_tact = profile.get("tactical_params") or {}
    item_tact = item.get("tactical_params") or {}
    if bool(prof_tact.get("trailing_enabled")) != bool(item_tact.get("trailing_enabled")):
        return False
    return _norm_tactical_for_compare(prof_tact) == _norm_tactical_for_compare(item_tact)


def _batch_items_by_symbol(conn, batch_id: int) -> Dict[str, Dict[str, Any]]:
    import sqlite3

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT * FROM moss_daily_optimize_items WHERE batch_id = ?""",
        (int(batch_id),),
    ).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        d = dict(row)
        sym = str(d.get("symbol") or "").strip().upper()
        if not sym:
            continue
        d["tactical_params"] = json.loads(d.pop("tactical_params_json") or "{}")
        summary = json.loads(d.pop("summary_json") or "{}")
        if summary and not summary.get("error"):
            summary = enrich_summary(summary)
        d["summary"] = summary
        out[sym] = d
    return out


def _run_paper_scan_after_param_sync(
    conn, updated_count: int
) -> Optional[Dict[str, Any]]:
    """参数变更后立即纸面扫描，使持仓按最新 SL/TP/出场阈值判平。"""
    if updated_count <= 0 or not cfg.MOSS_QUANT_PAPER_ENABLED:
        return None
    from moss_quant.paper_scanner import run_paper_scan

    try:
        out = run_paper_scan(conn)
        return {
            "opens": int(out.get("opens") or 0),
            "closes": int(out.get("closes") or 0),
            "profiles_scanned": int(out.get("profiles_scanned") or 0),
        }
    except Exception as e:
        logger.warning("[moss] paper scan after param sync failed: %s", e)
        return {"error": str(e)}


def _build_initial_for_sync(template: str, risk_scale: float = 1.0) -> dict:
    initial = build_initial_params(template=template)
    scale = max(0.1, min(1.0, float(risk_scale)))
    if scale < 1.0:
        for key in ("risk_per_trade", "max_position_pct"):
            if key in initial:
                initial[key] = round(float(initial[key]) * scale, 4)
    return initial


def _sync_rank_index(items_by_sym: Dict[str, Dict[str, Any]], sym: str) -> int:
    def _val_sharpe(item: Dict[str, Any]) -> float:
        sm = item.get("summary") or {}
        return float(sm.get("val_sharpe") or 0)

    syncable = [
        (s, _val_sharpe(items_by_sym[s]))
        for s in items_by_sym
        if can_sync_profile_params(items_by_sym[s].get("summary") or {})
    ]
    syncable.sort(key=lambda x: -x[1])
    for i, (s, _) in enumerate(syncable):
        if s == sym:
            return i
    return 999


def sync_enabled_profiles_from_batch(
    conn, batch_id: int, *, trigger_paper_scan: bool = True
) -> Dict[str, Any]:
    """将已启用或有持仓 Profile 的模板+战术参数同步为本批次寻优最优（不改启用状态）。"""
    from moss_quant.db import list_profiles_for_strategy_sync, profile_has_open_position

    items_by_sym = _batch_items_by_symbol(conn, batch_id)
    targets = list_profiles_for_strategy_sync(conn)
    now = _utc_now()
    stats: Dict[str, Any] = {
        "checked": 0,
        "updated": 0,
        "already_optimal": 0,
        "no_batch_item": 0,
        "invalid_batch_item": 0,
        "skipped_sync_gate": 0,
        "skipped_recent_loss": 0,
        "updated_with_open_position": 0,
        "updated_profiles": [],
    }
    for prof in targets:
        stats["checked"] += 1
        sym = str(prof.get("symbol") or "").strip().upper()
        pid = int(prof["id"])
        item = items_by_sym.get(sym)
        if not item:
            stats["no_batch_item"] += 1
            continue
        summary = item.get("summary") or {}
        if summary.get("error") or not item.get("template"):
            stats["invalid_batch_item"] += 1
            continue
        from moss_quant.optimize_policy import paper_recent_pnl_block_reason

        paper_block = paper_recent_pnl_block_reason(
            conn,
            pid,
            profile_capital=float(prof.get("virtual_equity_usdt") or 0) or None,
        )
        if paper_block:
            stats["skipped_recent_loss"] += 1
            continue
        if not can_sync_profile_params(summary):
            stats["skipped_sync_gate"] += 1
            continue
        if _profile_matches_batch_optimal(prof, item):
            stats["already_optimal"] += 1
            continue
        template = str(item.get("template") or "balanced")
        tactical = dict(item.get("tactical_params") or {})
        rank_idx = _sync_rank_index(items_by_sym, sym)
        pool_sharpes = [
            float((items_by_sym[s].get("summary") or {}).get("val_sharpe") or 0)
            for s in items_by_sym
            if can_sync_profile_params(items_by_sym[s].get("summary") or {})
        ]
        pool_max_sharpe = max(pool_sharpes) if pool_sharpes else 0.0
        val_sh = float(summary.get("val_sharpe") or 0)
        risk_scale = risk_scale_for_rank(
            rank_idx,
            val_sharpe=val_sh,
            pool_max_val_sharpe=pool_max_sharpe,
        )
        initial = _build_initial_for_sync(template, risk_scale)
        conn.execute(
            """UPDATE moss_profiles SET
               template=?, initial_params_json=?, tactical_params_json=?,
               updated_at_utc=?
               WHERE id=?""",
            (
                template,
                json.dumps(initial, ensure_ascii=False),
                json.dumps(tactical, ensure_ascii=False),
                now,
                pid,
            ),
        )
        conn.execute(
            """UPDATE moss_daily_optimize_items SET profile_id=?
               WHERE batch_id=? AND symbol=?""",
            (pid, int(batch_id), sym),
        )
        had_open = profile_has_open_position(conn, pid)
        stats["updated"] += 1
        if had_open:
            stats["updated_with_open_position"] += 1
        stats["updated_profiles"].append(
            {
                "profile_id": pid,
                "symbol": sym,
                "template": template,
                "tactical_params": tactical,
                "had_open_position": had_open,
                "risk_scale": risk_scale,
                "pool_tier": summary.get("pool_tier"),
            }
        )
    if stats["updated"]:
        conn.commit()
    stats["paper_scan"] = (
        _run_paper_scan_after_param_sync(conn, int(stats["updated"]))
        if trigger_paper_scan
        else None
    )
    logger.info(
        "[moss] daily sync profiles batch=%s checked=%s updated=%s open=%s optimal=%s",
        batch_id,
        stats["checked"],
        stats["updated"],
        stats["updated_with_open_position"],
        stats["already_optimal"],
    )
    return stats


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
    summary = json.loads(d.pop("summary_json") or "{}")
    if summary and not summary.get("error"):
        summary = enrich_summary(summary)
    d["summary"] = summary
    return d


def import_profile_from_daily(
    conn,
    symbol: str,
    *,
    enabled: bool = True,
    name: Optional[str] = None,
    update_existing: bool = True,
    profile_source: Optional[str] = None,
) -> Dict[str, Any]:
    """从最近一次每日寻优结果创建或更新纸面 Profile（用户主动操作）。"""
    item = get_latest_daily_item_for_symbol(conn, str(symbol).strip().upper())
    if not item:
        raise ValueError("daily_item_not_found")
    return import_profile_from_batch_item(
        conn,
        item,
        enabled=enabled,
        name=name,
        update_existing=update_existing,
        profile_source=profile_source or FROM_DAILY_PROFILE_SOURCE,
    )


def import_profile_from_batch_item(
    conn,
    item: Dict[str, Any],
    *,
    enabled: bool = True,
    name: Optional[str] = None,
    update_existing: bool = True,
    profile_source: Optional[str] = None,
    commit: bool = True,
) -> Dict[str, Any]:
    """从指定每日寻优 item 创建或更新纸面 Profile。"""
    from moss_quant import config as mq_cfg
    from moss_quant.universe import active_symbols_taken, is_symbol_allowed

    sym = str(item.get("symbol") or "").strip().upper()
    if not sym:
        raise ValueError("invalid_symbol")
    if not is_symbol_allowed(sym, conn=conn):
        raise ValueError("symbol_not_allowed")
    summary = item.get("summary") or {}
    if summary.get("error"):
        raise ValueError("daily_item_not_found")
    if str(summary.get("pool_tier") or "C") == "C":
        raise ValueError("daily_pool_rejected")
    template = str(item.get("template") or "balanced")
    tactical = item.get("tactical_params") or {}
    initial = build_initial_params(template=template)
    now = _utc_now()
    equity = float(mq_cfg.MOSS_QUANT_PROFILE_CAPITAL)
    prof_name = (name or "").strip() or ("from-daily-" + sym)
    src = profile_source or FROM_DAILY_PROFILE_SOURCE

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
               enabled=?, governance_manual_lock=0, updated_at_utc=?
               WHERE id=?""",
            (
                prof_name,
                template,
                src,
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
                   governance_manual_lock, created_at_utc, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                prof_name,
                sym,
                template,
                1 if enabled else 0,
                src,
                json.dumps(initial, ensure_ascii=False),
                json.dumps(tactical, ensure_ascii=False),
                equity,
                0,
                0,
                now,
                now,
            ),
        )
        pid = int(cur.lastrowid)

    item_id = item.get("id")
    if item_id is not None:
        conn.execute(
            """UPDATE moss_daily_optimize_items SET profile_id=?
               WHERE id=?""",
            (pid, int(item_id)),
        )
    if commit:
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
        summary = json.loads(d.pop("summary_json") or "{}")
        if summary and not summary.get("error") and not summary.get("pool_tier"):
            summary = enrich_summary(summary)
        d["summary"] = summary
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


def summarize_latest_daily_pools(conn) -> Dict[str, Any]:
    """最新每日寻优批次的币池 / 验证 KPI（供 summary API）。"""
    batch = get_latest_daily_batch(conn)
    if not batch:
        return {}
    pools = {"A": 0, "B": 0, "C": 0}
    val_pass = val_fail = sync_ok = 0
    for it in batch.get("items") or []:
        s = it.get("summary") or {}
        if s.get("error"):
            pools["C"] += 1
            continue
        tier = str(s.get("pool_tier") or "C")
        pools[tier] = pools.get(tier, 0) + 1
        if s.get("validation_passed"):
            val_pass += 1
        elif "validation_passed" in s:
            val_fail += 1
        if s.get("sync_allowed"):
            sync_ok += 1
    return {
        "batch_id": int(batch.get("id") or 0),
        "batch_status": batch.get("status"),
        "symbols_total": int(batch.get("symbols_total") or 0),
        "symbols_ok": int(batch.get("symbols_ok") or 0),
        "pool_counts": pools,
        "validation_passed": val_pass,
        "validation_failed": val_fail,
        "sync_allowed": sync_ok,
        "train_bars": next(
            (
                (it.get("summary") or {}).get("train_bars")
                for it in batch.get("items") or []
                if (it.get("summary") or {}).get("train_bars")
            ),
            None,
        ),
    }


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

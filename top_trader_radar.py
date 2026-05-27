#!/usr/bin/env python3
"""
大户多空 + Taker 雷达（公开 Binance fapi/futures/data）。

数据源（无需 API Key）：
  - /futures/data/topLongShortAccountRatio
  - /futures/data/topLongShortPositionRatio
  - /futures/data/takerlongshortRatio

不含 Smart Money bapi 的盈利人数 / 巨鲸均价（公开 API 无此字段）。
"""

from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from accumulation_radar import FAPI, get_all_perp_symbols, init_db
from top_trader_config import (
    VALID_UNIVERSES,
    top_trader_params,
    top_trader_snapshot_path_name,
    top_trader_trend_note,
)
from trend_5m import (
    TREND_5M_SOURCE_TABLES,
    assess_trend_5m,
    partition_trend_items,
)
from watchlist_symbols import drop_blacklisted_symbols, filter_symbols_to_binance_usdt_perps

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))
_last_binance_req_at: float = 0.0

_WORTH_UNION_TABLES: frozenset[str] = frozenset({
    "worth_watch_heat_accum",
    "worth_watch_patrick_core",
    "worth_watch_hot_oi",
    "worth_watch_chase_fire",
    "worth_watch_dual_list",
    "worth_watch_ambush_dark",
    "worth_watch_ambush_gem",
    "focus_watch",
    "heat_accum_watch",
    "patrick_core_watch",
    "ambush_watch",
    "watchlist",
})


def _now_cst() -> datetime:
    return datetime.now(CST)


def _now_cst_label() -> str:
    return _now_cst().strftime("%Y-%m-%d %H:%M:%S") + " CST"


def _run_id_from_cst(run_cst: str) -> str:
    return str(run_cst).replace(" CST", "").strip()


def _data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parent)))


def snapshot_path() -> Path:
    return _data_dir() / top_trader_snapshot_path_name()


def _binance_get(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    min_interval_sec: float = 0.12,
) -> Any:
    global _last_binance_req_at
    gap = float(min_interval_sec)
    if gap > 0:
        elapsed = time.monotonic() - _last_binance_req_at
        if elapsed < gap:
            time.sleep(gap - elapsed)
    url = f"{FAPI}{endpoint}"
    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, timeout=12)
            _last_binance_req_at = time.monotonic()
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait_s = float(retry_after) if retry_after else min(8.0, 2.0 * (attempt + 1))
                except (TypeError, ValueError):
                    wait_s = min(8.0, 2.0 * (attempt + 1))
                logger.warning("binance 429 %s wait=%.1fs", endpoint, wait_s)
                time.sleep(wait_s)
                continue
            logger.warning("binance %s status=%s", endpoint, resp.status_code)
            return None
        except requests.RequestException as exc:
            logger.warning("binance %s error=%s attempt=%s", endpoint, exc, attempt + 1)
            time.sleep(1.0 * (attempt + 1))
    return None


def ensure_top_trader_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS top_trader_snapshots (
        run_id TEXT NOT NULL,
        run_at_ms INTEGER NOT NULL,
        generated_date TEXT NOT NULL,
        symbol TEXT NOT NULL,
        period TEXT NOT NULL,
        ts INTEGER NOT NULL,
        top_account_long_pct REAL,
        top_account_short_pct REAL,
        top_account_lsr REAL,
        top_position_long_pct REAL,
        top_position_short_pct REAL,
        top_position_lsr REAL,
        taker_buy_vol REAL,
        taker_sell_vol REAL,
        taker_bsr REAL,
        signal_tags TEXT,
        summary_line TEXT,
        PRIMARY KEY (run_id, symbol)
    )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_top_trader_run_at ON top_trader_snapshots(run_at_ms)"
    )
    conn.commit()


def _symbols_from_table(cur: sqlite3.Cursor, table: str) -> Set[str]:
    allowed = _WORTH_UNION_TABLES | frozenset(TREND_5M_SOURCE_TABLES.keys())
    if table not in allowed:
        return set()
    out: Set[str] = set()
    try:
        rows = cur.execute(f"SELECT DISTINCT symbol FROM {table}").fetchall()
    except sqlite3.Error:
        return out
    for row in rows:
        if row and row[0]:
            out.add(str(row[0]).strip().upper())
    return out


def resolve_universe_symbols(
    conn: sqlite3.Connection,
    *,
    universe: Optional[str] = None,
    pool_max: Optional[int] = None,
    explicit_symbols: Optional[List[str]] = None,
) -> Tuple[List[str], str, Dict[str, List[str]]]:
    """
    解析扫描标的池。返回 (symbols, universe_label, sources_by_symbol)。
    pool_max=0 表示不截断。
    """
    p = top_trader_params()
    uni = (universe or p.universe).strip().lower()
    if uni not in VALID_UNIVERSES:
        uni = p.universe

    cap = p.pool_max if pool_max is None else max(0, int(pool_max))
    symbols: Set[str] = set()
    sources: Dict[str, List[str]] = {}

    cur = conn.cursor()
    if uni == "explicit":
        raw = list(explicit_symbols or p.explicit_symbols)
        for s in raw:
            u = s.strip().upper()
            if u:
                symbols.add(u)
                sources.setdefault(u, []).append("explicit")
    elif uni == "all":
        symbols.update(get_all_perp_symbols())
    elif uni == "watchlist":
        for sym in _symbols_from_table(cur, "watchlist"):
            symbols.add(sym)
            sources.setdefault(sym, []).append("watchlist")
    elif uni == "focus":
        for sym in _symbols_from_table(cur, "focus_watch"):
            symbols.add(sym)
            sources.setdefault(sym, []).append("focus")
    elif uni == "hot_oi":
        for sym in _symbols_from_table(cur, "worth_watch_hot_oi"):
            symbols.add(sym)
            sources.setdefault(sym, []).append("hot_oi")
    elif uni == "trend_5m":
        for tbl, label in TREND_5M_SOURCE_TABLES.items():
            for sym in _symbols_from_table(cur, tbl):
                symbols.add(sym)
                if label not in sources.setdefault(sym, []):
                    sources[sym].append(label)
    elif uni == "worth_union":
        for tbl in sorted(_WORTH_UNION_TABLES):
            if tbl == "watchlist":
                continue
            for sym in _symbols_from_table(cur, tbl):
                symbols.add(sym)
                if "worth" not in sources.setdefault(sym, []):
                    sources[sym].append("worth")

    filtered = drop_blacklisted_symbols(filter_symbols_to_binance_usdt_perps(sorted(symbols)))
    if cap > 0:
        filtered = filtered[:cap]
    src_out = {s: sources[s] for s in filtered if s in sources}
    return filtered, uni, src_out


def rehydrate_pool_sources(
    items: List[Dict[str, Any]],
    universe: Optional[str] = None,
) -> None:
    """DB 回退或旧快照缺 pool_sources 时，从 trend_5m 看盘表补全。"""
    uni = (universe or top_trader_params().universe or "").strip().lower()
    if uni != "trend_5m":
        return
    if not any(not row.get("pool_sources") for row in items):
        return
    conn = init_db()
    try:
        _, _, src_map = resolve_universe_symbols(conn, universe="trend_5m")
        for row in items:
            sym = str(row.get("symbol") or "").strip().upper()
            if sym and not row.get("pool_sources") and sym in src_map:
                row["pool_sources"] = list(src_map[sym])
    finally:
        conn.close()


def enrich_items_trend_5m(
    items: List[Dict[str, Any]],
    sources_by_symbol: Optional[Dict[str, List[str]]] = None,
    *,
    universe: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """为每条快照附加 5m 趋势判定，并返回 long/short/avoid/neutral 分组。"""
    from trend_5m import load_oi_coin_context

    rehydrate_pool_sources(items, universe)
    oi_ctx = load_oi_coin_context()
    src_map = sources_by_symbol or {}
    for row in items:
        sym = str(row.get("symbol") or "").strip().upper()
        pool_src = row.get("pool_sources") or src_map.get(sym) or []
        trend = assess_trend_5m(
            row,
            oi_ctx.get(sym),
            pool_sources=pool_src if isinstance(pool_src, list) else [],
        )
        row.update(trend)
        if pool_src:
            row["pool_sources"] = list(pool_src)
    return partition_trend_items(items)


def _parse_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _fetch_ratio(path: str, symbol: str, period: str) -> Optional[Dict[str, Any]]:
    data = _binance_get(
        path,
        {"symbol": symbol, "period": period, "limit": 1},
        min_interval_sec=top_trader_params().min_interval_sec,
    )
    if not isinstance(data, list) or not data:
        return None
    row = data[0]
    return row if isinstance(row, dict) else None


def fetch_top_trader_snapshot(symbol: str, period: str) -> Optional[Dict[str, Any]]:
    """单 symbol：Top 账户/持仓多空比 + Taker 比。"""
    sym = str(symbol or "").strip().upper()
    if not sym:
        return None

    acc = _fetch_ratio("/futures/data/topLongShortAccountRatio", sym, period)
    pos = _fetch_ratio("/futures/data/topLongShortPositionRatio", sym, period)
    taker = _fetch_ratio("/futures/data/takerlongshortRatio", sym, period)
    if not acc or not pos or not taker:
        return None

    # Binance 在 positionRatio 接口复用 longAccount/shortAccount 字段名
    snap = {
        "symbol": sym,
        "period": period,
        "ts": int(taker.get("timestamp") or pos.get("timestamp") or acc.get("timestamp") or time.time() * 1000),
        "top_account_long_pct": _parse_float(acc.get("longAccount")),
        "top_account_short_pct": _parse_float(acc.get("shortAccount")),
        "top_account_lsr": _parse_float(acc.get("longShortRatio")),
        "top_position_long_pct": _parse_float(pos.get("longAccount")),
        "top_position_short_pct": _parse_float(pos.get("shortAccount")),
        "top_position_lsr": _parse_float(pos.get("longShortRatio")),
        "taker_buy_vol": _parse_float(taker.get("buyVol")),
        "taker_sell_vol": _parse_float(taker.get("sellVol")),
        "taker_bsr": _parse_float(taker.get("buySellRatio")),
    }
    tags, summary = derive_signal_tags(snap)
    snap["signal_tags"] = tags
    snap["summary_line"] = summary
    return snap


def derive_signal_tags(snap: Dict[str, Any]) -> Tuple[List[str], str]:
    """由公开比率推导简短标签（非 Smart Money 盈利面）。"""
    tags: List[str] = []
    pos_lsr = float(snap.get("top_position_lsr") or 0)
    acc_long = float(snap.get("top_account_long_pct") or 0)
    pos_long = float(snap.get("top_position_long_pct") or 0)
    taker_bsr = float(snap.get("taker_bsr") or 0)

    if pos_lsr >= 1.25:
        tags.append("大户持仓偏多")
    elif pos_lsr > 0 and pos_lsr <= 0.8:
        tags.append("大户持仓偏空")

    if taker_bsr >= 1.08:
        tags.append("主动买盘")
    elif taker_bsr > 0 and taker_bsr <= 0.92:
        tags.append("主动卖盘")

    if acc_long - pos_long >= 12:
        tags.append("账户偏多/持仓偏空")
    elif pos_long - acc_long >= 12:
        tags.append("账户偏空/持仓偏多")

    if pos_lsr >= 1.15 and taker_bsr >= 1.05:
        tags.append("大户+Taker同向多")
    elif pos_lsr > 0 and pos_lsr <= 0.85 and taker_bsr > 0 and taker_bsr <= 0.95:
        tags.append("大户+Taker同向空")

    parts = []
    if pos_lsr > 0:
        parts.append(f"PosLSR {pos_lsr:.2f}")
    if taker_bsr > 0:
        parts.append(f"Taker {taker_bsr:.2f}")
    if tags:
        parts.append(" · ".join(tags[:2]))
    summary = " · ".join(parts) if parts else "—"
    return tags, summary


def _sleep_batch(spacing_sec: float, jitter_sec: float) -> None:
    if spacing_sec <= 0:
        return
    jitter = (random.random() * 2.0 - 1.0) * jitter_sec if jitter_sec > 0 else 0.0
    time.sleep(max(0.0, spacing_sec + jitter))


def run_top_trader_batch(
    symbols: List[str],
    *,
    period: Optional[str] = None,
    spacing_sec: Optional[float] = None,
    jitter_sec: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    p = top_trader_params()
    per = period or p.period
    space = p.spacing_sec if spacing_sec is None else spacing_sec
    jitter = p.jitter_sec if jitter_sec is None else jitter_sec
    out: Dict[str, Dict[str, Any]] = {}
    for i, sym in enumerate(symbols):
        snap = fetch_top_trader_snapshot(sym, per)
        if snap:
            out[sym] = snap
        if i + 1 < len(symbols):
            _sleep_batch(space, jitter)
    return out


def _prune_old_runs(conn: sqlite3.Connection, retention_days: int) -> int:
    cutoff_ms = int((_now_cst() - timedelta(days=retention_days)).timestamp() * 1000)
    cur = conn.cursor()
    row = cur.execute(
        "SELECT COUNT(DISTINCT run_id) FROM top_trader_snapshots WHERE run_at_ms < ?",
        (cutoff_ms,),
    ).fetchone()
    cur.execute("DELETE FROM top_trader_snapshots WHERE run_at_ms < ?", (cutoff_ms,))
    conn.commit()
    return int(row[0] if row and row[0] is not None else 0)


def _persist_run(
    conn: sqlite3.Connection,
    run_id: str,
    run_at_ms: int,
    generated_date: str,
    period: str,
    items: Dict[str, Dict[str, Any]],
) -> None:
    ensure_top_trader_schema(conn)
    cur = conn.cursor()
    for sym, snap in items.items():
        tags_json = json.dumps(snap.get("signal_tags") or [], ensure_ascii=False)
        cur.execute(
            """INSERT OR REPLACE INTO top_trader_snapshots
            (run_id, run_at_ms, generated_date, symbol, period, ts,
             top_account_long_pct, top_account_short_pct, top_account_lsr,
             top_position_long_pct, top_position_short_pct, top_position_lsr,
             taker_buy_vol, taker_sell_vol, taker_bsr, signal_tags, summary_line)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                run_at_ms,
                generated_date,
                sym,
                period,
                int(snap.get("ts") or run_at_ms),
                snap.get("top_account_long_pct"),
                snap.get("top_account_short_pct"),
                snap.get("top_account_lsr"),
                snap.get("top_position_long_pct"),
                snap.get("top_position_short_pct"),
                snap.get("top_position_lsr"),
                snap.get("taker_buy_vol"),
                snap.get("taker_sell_vol"),
                snap.get("taker_bsr"),
                tags_json,
                snap.get("summary_line"),
            ),
        )
    conn.commit()


def _write_disk_snapshot(payload: Dict[str, Any]) -> None:
    path = snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_top_trader_data(conn: sqlite3.Connection) -> Dict[str, Any]:
    """清空大户多空 SQLite 表与磁盘快照 top_trader_snapshot.json。"""
    ensure_top_trader_schema(conn)
    cur = conn.cursor()
    count_row = cur.execute("SELECT COUNT(*) FROM top_trader_snapshots").fetchone()
    deleted_rows = int(count_row[0] if count_row else 0)
    cur.execute("DELETE FROM top_trader_snapshots")
    conn.commit()

    disk_removed = False
    path = snapshot_path()
    if path.is_file():
        path.unlink()
        disk_removed = True

    return {
        "deleted_top_trader_rows": deleted_rows,
        "disk_snapshot_removed": disk_removed,
    }


def apply_trend_5m_to_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """读快照时用最新 OI 上下文重算 trend 分组（GET 时调用）。"""
    data = dict(data)
    period = str(data.get("period") or top_trader_params().period or "15m")
    data["note"] = top_trader_trend_note(period)

    items = data.get("items")
    if not isinstance(items, list) or not items:
        return data
    out_items = [dict(x) for x in items if isinstance(x, dict)]
    uni = data.get("universe") or top_trader_params().universe
    buckets = enrich_items_trend_5m(out_items, universe=uni)
    data["items"] = out_items
    data["trend_long"] = buckets["trend_long"]
    data["trend_short"] = buckets["trend_short"]
    data["trend_avoid"] = buckets["trend_avoid"]
    data["trend_neutral"] = buckets["trend_neutral"]
    data["strategy"] = "trend_5m" if uni == "trend_5m" else data.get("strategy") or "top_trader"
    return data


def load_top_trader_snapshot_auto(*, limit: int = 200) -> Dict[str, Any]:
    """优先磁盘 ok 快照，否则 SQLite 最新 run，再否则磁盘错误信息。"""
    disk = load_top_trader_snapshot_from_disk()
    if disk.get("ok"):
        return apply_trend_5m_to_payload(disk)
    db = load_latest_top_trader_from_db(limit=limit)
    if db.get("ok"):
        return apply_trend_5m_to_payload(db)
    if disk.get("error") not in (None, "no_snapshot"):
        return disk
    return db


def run_top_trader_radar_once(
    *,
    universe: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """完整一轮：解析标的 → 批量拉取 → 落库 + 写 JSON 快照。"""
    p = top_trader_params()
    started = time.time()
    run_cst = _now_cst_label()
    run_id = _run_id_from_cst(run_cst)
    run_at_ms = int(_now_cst().timestamp() * 1000)
    generated_date = _now_cst().strftime("%Y-%m-%d")

    conn = init_db()
    try:
        ensure_top_trader_schema(conn)
        if symbols:
            pool = drop_blacklisted_symbols(
                filter_symbols_to_binance_usdt_perps(
                    [s.strip().upper() for s in symbols if s and str(s).strip()]
                )
            )
            uni_label = "request"
            src_map = {s: ["request"] for s in pool}
        else:
            pool, uni_label, src_map = resolve_universe_symbols(
                conn,
                universe=universe,
                pool_max=p.pool_max,
            )

        if not pool:
            payload = {
                "ok": False,
                "error": "empty_universe",
                "message": (
                    f"标的池为空（universe={uni_label}）。"
                    "trend_5m 需 worth_watch_hot_oi / focus_watch / worth_watch_chase_fire 有数据；"
                    "请先跑 OI 雷达（:30）再刷新大户多空。"
                ),
                "universe": uni_label,
                "period": p.period,
                "run_cst": run_cst,
            }
            return payload

        if not quiet:
            eta = len(pool) * max(p.spacing_sec, p.min_interval_sec) * 3
            logger.info(
                "top_trader start universe=%s pool=%s period=%s eta~%.0fs",
                uni_label,
                len(pool),
                p.period,
                eta,
            )

        captured = run_top_trader_batch(pool, period=p.period)

        if not captured:
            return {
                "ok": False,
                "error": "fetch_failed",
                "message": f"未获取到任何有效数据（requested={len(pool)}）。请稍后重试或缩小标的池。",
                "universe": uni_label,
                "period": p.period,
                "run_cst": run_cst,
                "requested": len(pool),
                "captured": 0,
            }

        _persist_run(conn, run_id, run_at_ms, generated_date, p.period, captured)
        pruned_runs = _prune_old_runs(conn, p.retention_days)

        items = []
        for sym in sorted(captured.keys()):
            row = dict(captured[sym])
            row["coin"] = sym.replace("USDT", "")
            row["pool_sources"] = src_map.get(sym, [])
            items.append(row)

        buckets = enrich_items_trend_5m(items, src_map, universe=uni_label)

        payload: Dict[str, Any] = {
            "ok": True,
            "run_id": run_id,
            "run_cst": run_cst,
            "generated_date": generated_date,
            "universe": uni_label,
            "period": p.period,
            "strategy": "trend_5m" if uni_label == "trend_5m" else "top_trader",
            "requested": len(pool),
            "captured": len(captured),
            "missed": max(0, len(pool) - len(captured)),
            "elapsed_sec": round(time.time() - started, 2),
            "pruned_old_runs": pruned_runs,
            "items": items,
            "trend_long": buckets["trend_long"],
            "trend_short": buckets["trend_short"],
            "trend_avoid": buckets["trend_avoid"],
            "trend_neutral": buckets["trend_neutral"],
            "note": top_trader_trend_note(p.period),
        }
        _write_disk_snapshot(payload)
        if not quiet:
            logger.info(
                "top_trader done universe=%s captured=%s/%s elapsed=%.1fs",
                uni_label,
                len(captured),
                len(pool),
                payload["elapsed_sec"],
            )
        return payload
    finally:
        conn.close()


def load_top_trader_snapshot_from_disk() -> Dict[str, Any]:
    path = snapshot_path()
    if not path.is_file():
        return {
            "ok": False,
            "error": "no_snapshot",
            "message": (
                "尚无大户多空快照。维护面板触发 top_trader，"
                "或设置 TOP_TRADER_SCHEDULER_ENABLED=1。"
            ),
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("snapshot root must be object")
        data["snapshot_source"] = "disk"
        return data
    except Exception as exc:
        logger.warning("top_trader snapshot read failed: %s", exc)
        return {"ok": False, "error": "snapshot_corrupt", "message": str(exc)}


def load_latest_top_trader_from_db(limit: int = 200) -> Dict[str, Any]:
    conn = init_db()
    try:
        ensure_top_trader_schema(conn)
        cur = conn.cursor()
        row = cur.execute(
            "SELECT run_id, run_at_ms, generated_date, period FROM top_trader_snapshots "
            "ORDER BY run_at_ms DESC LIMIT 1"
        ).fetchone()
        if not row:
            disk = load_top_trader_snapshot_from_disk()
            if disk.get("error") == "no_snapshot":
                return disk
            return {"ok": False, "error": "no_snapshot", "message": "数据库与磁盘均无大户多空快照。"}

        run_id, run_at_ms, generated_date, period = row
        rows = cur.execute(
            """SELECT symbol, ts,
                      top_account_long_pct, top_account_short_pct, top_account_lsr,
                      top_position_long_pct, top_position_short_pct, top_position_lsr,
                      taker_buy_vol, taker_sell_vol, taker_bsr,
                      signal_tags, summary_line
               FROM top_trader_snapshots
               WHERE run_id = ?
               ORDER BY top_position_lsr DESC
               LIMIT ?""",
            (run_id, max(1, int(limit))),
        ).fetchall()

        items: List[Dict[str, Any]] = []
        for r in rows:
            tags_raw = r[12]
            try:
                tags = json.loads(tags_raw) if tags_raw else []
            except json.JSONDecodeError:
                tags = []
            sym = str(r[0])
            items.append(
                {
                    "symbol": sym,
                    "coin": sym.replace("USDT", ""),
                    "ts": int(r[1]),
                    "top_account_long_pct": r[2],
                    "top_account_short_pct": r[3],
                    "top_account_lsr": r[4],
                    "top_position_long_pct": r[5],
                    "top_position_short_pct": r[6],
                    "top_position_lsr": r[7],
                    "taker_buy_vol": r[8],
                    "taker_sell_vol": r[9],
                    "taker_bsr": r[10],
                    "signal_tags": tags,
                    "summary_line": r[13],
                    "period": period,
                }
            )

        run_cst = datetime.fromtimestamp(run_at_ms / 1000, tz=CST).strftime(
            "%Y-%m-%d %H:%M:%S"
        ) + " CST"
        return {
            "ok": True,
            "run_id": run_id,
            "run_cst": run_cst,
            "generated_date": generated_date,
            "universe": top_trader_params().universe,
            "period": period,
            "captured": len(items),
            "items": items,
            "snapshot_source": "db",
            "note": top_trader_trend_note(period),
        }
    finally:
        conn.close()

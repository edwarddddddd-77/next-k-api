"""
接针严选标的池 — 表 jz_universe（自动刷新）。

方案 v2（curated）原料：
  主池：patrick_core_watch ∪ worth_watch_patrick_core（收筹池 + |OI| 异动）
  扩池：worth_watch_heat_accum（热度 + 收筹，蓄力接针）
  hot_oi：仅作加分，不作为唯一入选来源

过滤（共用）：
  |d6h| ∈ [MIN, MAX]（默认 3%～22%）
  px_chg ∈ [MIN, MAX]；价涨且 d6h 弱 → 否决
  可选 VP：scheme 在允许集合内（无快照默认放行）

刷新：oi 小时雷达后；接针扫描前表空/过期时。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import jiezhen_config as cfg
from watchlist_symbols import (
    drop_blacklisted_symbols,
    filter_symbols_to_binance_usdt_perps,
    symbol_blacklist,
)

logger = logging.getLogger(__name__)

JZ_UNIVERSE_TABLE = "jz_universe"

# 仅允许读取这些 worth_watch 物理表（防 SQL 标识符注入）
_WORTH_WATCH_TABLES = frozenset(
    {
        "worth_watch_patrick_core",
        "worth_watch_heat_accum",
        "worth_watch_hot_oi",
    }
)


def migrate_jz_universe_table(c: sqlite3.Cursor) -> None:
    c.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {JZ_UNIVERSE_TABLE} (
            symbol TEXT PRIMARY KEY,
            selected_at_utc TEXT NOT NULL,
            rank INTEGER NOT NULL,
            score REAL NOT NULL,
            sources TEXT NOT NULL,
            heat REAL,
            d6h REAL,
            px_chg REAL,
            hot_oi_rank INTEGER,
            vp_scheme TEXT,
            filter_note TEXT
        )
        """
    )
    c.execute(
        f"""
        CREATE INDEX IF NOT EXISTS ix_jz_universe_selected
        ON {JZ_UNIVERSE_TABLE}(selected_at_utc DESC)
        """
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_detail(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        obj = json.loads(str(raw))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _float_or_none(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _empty_candidate(sym: str) -> Dict[str, Any]:
    return {
        "symbol": sym,
        "sources": [],
        "heat": None,
        "d6h": None,
        "px_chg": None,
        "sw_days": None,
        "patrick_rank": 999,
        "heat_rank": 999,
        "hot_oi_rank": 0,
        "summary_line": "",
    }


def _merge_into(
    pool: Dict[str, Dict[str, Any]], sym: str, source: str, **fields: Any
) -> None:
    """
    合并候选字段。后写入的非空字段覆盖先写入的（patrick：先 worth 后 core 表，以 core 为准）。
    """
    sym = sym.strip().upper()
    if not sym:
        return
    row = pool.get(sym)
    if row is None:
        row = _empty_candidate(sym)
        pool[sym] = row
    if source not in row["sources"]:
        row["sources"].append(source)
    for k, v in fields.items():
        if v is None:
            continue
        if k == "summary_line" and v:
            row[k] = str(v)
        elif k in ("patrick_rank", "heat_rank", "hot_oi_rank"):
            row[k] = min(int(row.get(k) or 999), int(v))
        else:
            row[k] = v


def _load_worth_watch_rows(
    cur: sqlite3.Cursor, table: str
) -> List[Dict[str, Any]]:
    if table not in _WORTH_WATCH_TABLES:
        logger.warning("[jz-universe] unknown worth_watch table: %s", table)
        return []
    try:
        cur.execute(
            f"""
            SELECT symbol, rank_in_category, detail_json, summary_line
            FROM {table}
            ORDER BY COALESCE(rank_in_category, 999) ASC, symbol ASC
            """
        )
    except sqlite3.OperationalError:
        return []
    out: List[Dict[str, Any]] = []
    for row in cur.fetchall():
        sym = str(row[0] or "").strip().upper()
        if not sym:
            continue
        det = _parse_detail(row[2])
        out.append(
            {
                "symbol": sym,
                "rank": int(row[1] or 999),
                "detail": det,
                "summary_line": str(row[3] or ""),
            }
        )
    return out


def _patrick_core_symbol_set(cur: sqlite3.Cursor) -> Set[str]:
    """patrick_core_watch（2 日留存）为 Patrick 主池权威来源。"""
    try:
        cur.execute("SELECT symbol FROM patrick_core_watch")
    except sqlite3.OperationalError:
        return set()
    return {
        str(row[0] or "").strip().upper()
        for row in cur.fetchall()
        if row and row[0]
    }


def _load_patrick_candidates(
    cur: sqlite3.Cursor, core_syms: Set[str]
) -> Dict[str, Dict[str, Any]]:
    pool: Dict[str, Dict[str, Any]] = {}
    require_core = cfg.JIEZHEN_SEL_PATRICK_REQUIRE_CORE_ROW

    for item in _load_worth_watch_rows(cur, "worth_watch_patrick_core"):
        sym = item["symbol"]
        if require_core and sym not in core_syms:
            continue
        det = item["detail"]
        _merge_into(
            pool,
            sym,
            "patrick",
            d6h=_float_or_none(det.get("d6h")),
            px_chg=_float_or_none(det.get("px_chg")),
            heat=_float_or_none(det.get("heat")),
            sw_days=_float_or_none(det.get("sw_days")),
            patrick_rank=item["rank"],
            summary_line=item["summary_line"],
        )
    try:
        cur.execute(
            """
            SELECT symbol, d6h, px_chg, est_mcap, sideways_days, summary_line
            FROM patrick_core_watch
            """
        )
        for row in cur.fetchall():
            sym = str(row[0] or "").strip().upper()
            if not sym:
                continue
            _merge_into(
                pool,
                sym,
                "patrick",
                d6h=_float_or_none(row[1]),
                px_chg=_float_or_none(row[2]),
                sw_days=_float_or_none(row[4]),
                summary_line=str(row[5] or ""),
            )
    except sqlite3.OperationalError:
        pass
    return pool


def _load_heat_accum_candidates(cur: sqlite3.Cursor) -> Dict[str, Dict[str, Any]]:
    if not cfg.JIEZHEN_SEL_INCLUDE_HEAT_ACCUM:
        return {}
    pool: Dict[str, Dict[str, Any]] = {}
    for item in _load_worth_watch_rows(cur, "worth_watch_heat_accum"):
        det = item["detail"]
        heat = _float_or_none(det.get("heat"))
        if heat is not None and heat < cfg.JIEZHEN_SEL_MIN_HEAT:
            continue
        _merge_into(
            pool,
            item["symbol"],
            "heat_accum",
            heat=heat,
            d6h=_float_or_none(det.get("d6h")),
            px_chg=_float_or_none(det.get("px_chg")),
            sw_days=_float_or_none(det.get("sw_days")),
            heat_rank=item["rank"],
            summary_line=item["summary_line"],
        )
    return pool


def _load_hot_oi_rank_map(cur: sqlite3.Cursor) -> Dict[str, int]:
    ranks: Dict[str, int] = {}
    for item in _load_worth_watch_rows(cur, "worth_watch_hot_oi"):
        ranks[item["symbol"]] = item["rank"]
    return ranks


def _build_candidate_pool(
    cur: sqlite3.Cursor,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int], int]:
    core_syms = _patrick_core_symbol_set(cur)
    patrick = _load_patrick_candidates(cur, core_syms)
    heat = _load_heat_accum_candidates(cur)
    hot_ranks = _load_hot_oi_rank_map(cur)
    pool: Dict[str, Dict[str, Any]] = {}
    for sym, row in patrick.items():
        pool[sym] = dict(row)
    for sym, row in heat.items():
        if sym in pool:
            for s in row["sources"]:
                if s not in pool[sym]["sources"]:
                    pool[sym]["sources"].append(s)
            for k in ("heat", "d6h", "px_chg", "sw_days", "heat_rank", "summary_line"):
                if row.get(k) is not None and (
                    pool[sym].get(k) is None or k == "heat_rank"
                ):
                    if k == "heat_rank":
                        pool[sym][k] = min(
                            int(pool[sym].get(k) or 999), int(row[k])
                        )
                    else:
                        pool[sym][k] = row[k]
        else:
            pool[sym] = dict(row)
    for sym in pool:
        r = hot_ranks.get(sym)
        if r is not None:
            pool[sym]["hot_oi_rank"] = r
            if "hot_oi" not in pool[sym]["sources"]:
                pool[sym]["sources"].append("hot_oi_bonus")
    return pool, hot_ranks, len(core_syms)


def _load_vp_schemes(cur: sqlite3.Cursor, symbols: List[str]) -> Dict[str, str]:
    if not symbols or not cfg.JIEZHEN_SEL_VP_FILTER:
        return {}
    table = cfg.jiezhen_sel_vp_table()
    try:
        cur.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        if not cur.fetchone():
            return {}
    except sqlite3.OperationalError:
        return {}
    out: Dict[str, str] = {}
    chunk = 80
    for i in range(0, len(symbols), chunk):
        part = symbols[i : i + chunk]
        placeholders = ",".join("?" * len(part))
        cur.execute(
            f"SELECT symbol, scheme FROM {table} WHERE symbol IN ({placeholders})",
            part,
        )
        for row in cur.fetchall():
            sym = str(row[0] or "").strip().upper()
            if sym:
                out[sym] = str(row[1] or "").strip().upper()
    return out


def _has_primary_source(sources: List[str]) -> bool:
    if "patrick" in sources:
        return True
    if cfg.JIEZHEN_SEL_INCLUDE_HEAT_ACCUM and "heat_accum" in sources:
        return True
    return False


def _reject_reason(row: Dict[str, Any]) -> str:
    sources = row.get("sources") or []
    if not _has_primary_source(sources):
        return "not_primary_source"
    d6h = row.get("d6h")
    if d6h is None:
        return "missing_d6h"
    ad6 = abs(float(d6h))
    if ad6 < cfg.JIEZHEN_SEL_MIN_D6H:
        return f"|d6h|<{cfg.JIEZHEN_SEL_MIN_D6H}"
    if ad6 > cfg.JIEZHEN_SEL_MAX_D6H:
        return f"|d6h|>{cfg.JIEZHEN_SEL_MAX_D6H}"
    px = row.get("px_chg")
    if px is not None:
        if px > cfg.JIEZHEN_SEL_MAX_PX_CHG:
            return f"px_chg>{cfg.JIEZHEN_SEL_MAX_PX_CHG}"
        if px < cfg.JIEZHEN_SEL_MIN_PX_CHG:
            return f"px_chg<{cfg.JIEZHEN_SEL_MIN_PX_CHG}"
        if px > 0 and float(d6h) < cfg.JIEZHEN_SEL_VETO_D6H_WHEN_PX_UP:
            return "veto_px_up_oi_weak"
    return ""


def _score_row(row: Dict[str, Any], *, vp_scheme: Optional[str]) -> float:
    d6h = float(row.get("d6h") or 0)
    heat = float(row.get("heat") or 0)
    sources = row.get("sources") or []
    score = abs(d6h) * 1.2 + heat * 0.5
    if "patrick" in sources:
        score += cfg.JIEZHEN_SEL_PATRICK_BONUS
        pr = int(row.get("patrick_rank") or 99)
        if pr == 1:
            score += 6.0
        elif pr == 2:
            score += 3.0
    if "heat_accum" in sources:
        score += cfg.JIEZHEN_SEL_HEAT_ACCUM_BONUS
        hr = int(row.get("heat_rank") or 99)
        if hr == 1:
            score += 4.0
    sw = row.get("sw_days")
    if sw is not None and float(sw) >= cfg.JIEZHEN_SEL_SW_DAYS_BONUS_MIN:
        score += 3.0
    ho = int(row.get("hot_oi_rank") or 0)
    if ho == 1:
        score += cfg.JIEZHEN_SEL_HOT_OI_BONUS
    elif ho == 2:
        score += cfg.JIEZHEN_SEL_HOT_OI_BONUS * 0.6
    elif ho == 3:
        score += cfg.JIEZHEN_SEL_HOT_OI_BONUS * 0.3
    if vp_scheme and vp_scheme in cfg.jiezhen_sel_vp_allowed_schemes():
        score += 5.0
    return score


def _evaluate_candidates(
    cur: sqlite3.Cursor,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pool, hot_ranks, patrick_core_rows = _build_candidate_pool(cur)
    meta: Dict[str, Any] = {
        "scheme": "curated_v2",
        "candidate_pool": len(pool),
        "patrick_table": sum(1 for r in pool.values() if "patrick" in r["sources"]),
        "heat_accum_table": sum(
            1 for r in pool.values() if "heat_accum" in r["sources"]
        ),
        "hot_oi_bonus_tags": sum(
            1 for r in pool.values() if int(r.get("hot_oi_rank") or 0) > 0
        ),
        "hot_oi_listed": len(hot_ranks),
        "patrick_core_rows": patrick_core_rows,
        "rejected": {},
        "trimmed_by_cap": 0,
    }
    passed: List[Dict[str, Any]] = []
    for row in pool.values():
        reason = _reject_reason(row)
        if reason:
            meta["rejected"][reason] = meta["rejected"].get(reason, 0) + 1
            continue
        passed.append(row)

    syms_for_vp = [r["symbol"] for r in passed]
    vp_map = _load_vp_schemes(cur, syms_for_vp)
    meta["vp_snapshots"] = len(vp_map)
    allowed_vp = cfg.jiezhen_sel_vp_allowed_schemes()
    final: List[Dict[str, Any]] = []

    for row in passed:
        sym = row["symbol"]
        vp = vp_map.get(sym)
        if cfg.JIEZHEN_SEL_VP_FILTER and vp:
            if vp not in allowed_vp:
                key = f"vp_scheme:{vp}"
                meta["rejected"][key] = meta["rejected"].get(key, 0) + 1
                continue
        elif cfg.JIEZHEN_SEL_VP_FILTER and cfg.JIEZHEN_SEL_VP_STRICT:
            meta["rejected"]["vp_missing"] = meta["rejected"].get("vp_missing", 0) + 1
            continue
        final.append(
            {
                **row,
                "vp_scheme": vp,
                "score": _score_row(row, vp_scheme=vp),
            }
        )

    final.sort(key=lambda x: (-float(x["score"]), x["symbol"]))
    cap = cfg.JIEZHEN_UNIVERSE_MAX
    if len(final) > cap:
        meta["trimmed_by_cap"] = len(final) - cap
        final = final[:cap]

    meta["passed"] = len(final)
    return final, meta


def refresh_jiezhen_universe(conn: sqlite3.Connection) -> Dict[str, Any]:
    """全量重算并写入 jz_universe。"""
    migrate_jz_universe_table(conn.cursor())
    cur = conn.cursor()
    selected, meta = _evaluate_candidates(cur)
    symbols = drop_blacklisted_symbols(
        filter_symbols_to_binance_usdt_perps([r["symbol"] for r in selected])
    )
    allowed: Set[str] = set(symbols)
    selected = [r for r in selected if r["symbol"] in allowed]
    meta["blacklist"] = list(symbol_blacklist()) if symbol_blacklist() else []
    meta["after_perp_filter"] = len(selected)

    now = _utc_now()
    cur.execute(f"DELETE FROM {JZ_UNIVERSE_TABLE}")
    for i, row in enumerate(selected, start=1):
        cur.execute(
            f"""
            INSERT INTO {JZ_UNIVERSE_TABLE} (
                symbol, selected_at_utc, rank, score, sources,
                heat, d6h, px_chg, hot_oi_rank, vp_scheme, filter_note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["symbol"],
                now,
                i,
                float(row["score"]),
                json.dumps(row["sources"], ensure_ascii=False),
                row.get("heat"),
                row.get("d6h"),
                row.get("px_chg"),
                int(row.get("hot_oi_rank") or 0),
                row.get("vp_scheme"),
                row.get("summary_line") or "",
            ),
        )
    conn.commit()
    meta["selected_at_utc"] = now
    meta["symbols"] = [r["symbol"] for r in selected]
    logger.info(
        "[jz-universe] v2 refreshed n=%s pool=%s rejected=%s",
        len(selected),
        meta.get("candidate_pool"),
        meta.get("rejected"),
    )
    return meta


def _latest_selected_ms(cur: sqlite3.Cursor) -> Optional[int]:
    cur.execute(f"SELECT MAX(selected_at_utc) FROM {JZ_UNIVERSE_TABLE}")
    row = cur.fetchone()
    raw = row[0] if row else None
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        return None


def universe_is_stale(cur: sqlite3.Cursor) -> bool:
    cur.execute(f"SELECT COUNT(*) FROM {JZ_UNIVERSE_TABLE}")
    n = int(cur.fetchone()[0] or 0)
    if n == 0:
        return True
    ms = _latest_selected_ms(cur)
    if ms is None:
        return True
    age_sec = (datetime.now(timezone.utc).timestamp() * 1000 - ms) / 1000.0
    return age_sec > cfg.JIEZHEN_SEL_REFRESH_MAX_AGE_SEC


def refresh_jiezhen_universe_if_stale(
    conn: sqlite3.Connection, *, force: bool = False
) -> Dict[str, Any]:
    migrate_jz_universe_table(conn.cursor())
    cur = conn.cursor()
    if not force and not universe_is_stale(cur):
        return {"skipped": True, "reason": "fresh"}
    return refresh_jiezhen_universe(conn)


def load_jiezhen_universe_rows(
    cur: sqlite3.Cursor,
) -> List[Dict[str, Any]]:
    migrate_jz_universe_table(cur)
    cur.execute(
        f"""
        SELECT symbol, selected_at_utc, rank, score, sources,
               heat, d6h, px_chg, hot_oi_rank, vp_scheme, filter_note
        FROM {JZ_UNIVERSE_TABLE}
        ORDER BY rank ASC
        """
    )
    out: List[Dict[str, Any]] = []
    for row in cur.fetchall():
        d = dict(row)
        try:
            d["sources"] = json.loads(str(d.get("sources") or "[]"))
        except json.JSONDecodeError:
            d["sources"] = []
        out.append(d)
    return out


def load_jiezhen_universe_symbols(cur: sqlite3.Cursor) -> List[str]:
    return [r["symbol"] for r in load_jiezhen_universe_rows(cur)]

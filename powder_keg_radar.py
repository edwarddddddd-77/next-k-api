#!/usr/bin/env python3
"""
火药桶宏观雷达（币安）：收筹池内 OI 激增 + |费率|极端 + 横盘；负费率→仅多、正费率→仅空。

用法:
  python powder_keg_radar.py --once
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from accumulation_radar import FAPI, init_db
from powder_keg_config import powder_keg_params

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))

_last_binance_req_at: float = 0.0


def _now_cst() -> datetime:
    return datetime.now(CST)


def _now_cst_label() -> str:
    return _now_cst().strftime("%Y-%m-%d %H:%M:%S") + " CST"


def _run_id_from_cst(run_cst: str) -> str:
    return str(run_cst).replace(" CST", "").strip()


def _binance_get(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    min_interval_sec: float = 0.12,
) -> Any:
    """币安 GET：请求间隔 + 429 退避（减轻权重突发）。"""
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
                wait_s = float(retry_after) if retry_after else min(8.0, 2.0 * (attempt + 1))
                logger.warning(
                    "binance 429 %s wait=%.1fs attempt=%s",
                    endpoint,
                    wait_s,
                    attempt + 1,
                )
                time.sleep(wait_s)
                continue
            logger.warning("binance %s status=%s", endpoint, resp.status_code)
            return None
        except requests.RequestException as exc:
            logger.warning("binance %s error=%s attempt=%s", endpoint, exc, attempt + 1)
            time.sleep(1.0 * (attempt + 1))
    return None


def _retention_cutoff_ms(*, retention_hours: int, now: Optional[datetime] = None) -> int:
    t0 = now or _now_cst()
    return int((t0 - timedelta(hours=int(retention_hours))).timestamp() * 1000)


def _migrate_powder_keg_schema_v2(cur: sqlite3.Cursor) -> None:
    """旧版 symbol PRIMARY KEY 单表快照 → 按 run 保留多轮。"""
    row = cur.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='powder_keg_watchlist'"
    ).fetchone()
    if not row or not row[0]:
        return
    ddl = str(row[0])
    if "run_id" in ddl:
        return
    cur.execute("ALTER TABLE powder_keg_watchlist RENAME TO powder_keg_watchlist__v0")
    cur.execute("DROP TABLE IF EXISTS powder_keg_watchlist__v0")


def ensure_powder_keg_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    _migrate_powder_keg_schema_v2(cur)
    cur.execute(
        """CREATE TABLE IF NOT EXISTS powder_keg_watchlist (
        run_id TEXT NOT NULL,
        run_at_ms INTEGER NOT NULL,
        generated_date TEXT NOT NULL,
        symbol TEXT NOT NULL,
        coin TEXT NOT NULL,
        run_cst TEXT NOT NULL,
        rank_in_list INTEGER NOT NULL,
        score REAL NOT NULL,
        oi_usd REAL,
        oi_delta_1h_pct REAL,
        oi_delta_6h_pct REAL,
        funding_rate_pct REAL,
        px_chg_24h_pct REAL,
        range_6h_pct REAL,
        price REAL,
        vol_24h_usd REAL,
        summary_line TEXT,
        detail_json TEXT,
        PRIMARY KEY (run_id, symbol)
    )"""
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_powder_keg_run_at ON powder_keg_watchlist(run_at_ms)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_powder_keg_generated_date "
        "ON powder_keg_watchlist(generated_date)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_powder_keg_symbol "
        "ON powder_keg_watchlist(symbol)"
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS powder_keg_runs (
        run_id TEXT PRIMARY KEY,
        run_at_ms INTEGER NOT NULL,
        run_cst TEXT NOT NULL,
        watchlist_count INTEGER NOT NULL,
        scanned_pre INTEGER NOT NULL,
        matched INTEGER NOT NULL,
        inserted INTEGER NOT NULL,
        api_mode TEXT
    )"""
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_powder_keg_runs_at ON powder_keg_runs(run_at_ms)"
    )
    conn.commit()


def _prune_powder_keg_watchlist(
    conn: sqlite3.Connection,
    *,
    retention_hours: int,
    now: Optional[datetime] = None,
) -> int:
    cutoff_ms = _retention_cutoff_ms(retention_hours=retention_hours, now=now)
    cur = conn.cursor()
    cur.execute("DELETE FROM powder_keg_watchlist WHERE run_at_ms < ?", (cutoff_ms,))
    return int(cur.rowcount or 0)


def _dedupe_items_for_insert(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """同一轮入库前去重：同 symbol 保留 score 更高的一条。"""
    best: Dict[str, Dict[str, Any]] = {}
    for row in items:
        sym = str(row.get("symbol") or "").strip().upper()
        if not sym:
            continue
        prev = best.get(sym)
        if prev is None or float(row.get("score") or 0) >= float(prev.get("score") or 0):
            best[sym] = {**row, "symbol": sym}
    return sorted(best.values(), key=lambda r: (-float(r.get("score") or 0), r["symbol"]))


def _delete_watchlist_symbols(conn: sqlite3.Connection, symbols: List[str]) -> int:
    """删除这些 symbol 的旧记录（为「每币仅保留最新」腾位）。"""
    syms = sorted({str(s).strip().upper() for s in symbols if s})
    if not syms:
        return 0
    cur = conn.cursor()
    placeholders = ",".join("?" * len(syms))
    cur.execute(
        f"DELETE FROM powder_keg_watchlist WHERE symbol IN ({placeholders})",
        syms,
    )
    return int(cur.rowcount or 0)


def _prune_duplicate_symbols_in_watchlist(conn: sqlite3.Connection) -> int:
    """表内按 symbol 去重：仅保留 run_at_ms 最新的一条。"""
    cur = conn.cursor()
    cur.execute(
        """
        DELETE FROM powder_keg_watchlist
        WHERE rowid NOT IN (
            SELECT p.rowid
            FROM powder_keg_watchlist AS p
            INNER JOIN (
                SELECT symbol, MAX(run_at_ms) AS mx
                FROM powder_keg_watchlist
                GROUP BY symbol
            ) AS t ON p.symbol = t.symbol AND p.run_at_ms = t.mx
        )
        """
    )
    return int(cur.rowcount or 0)


def _load_watchlist_universe(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """收筹池标的（status != removed），按 pool score 降序。"""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT symbol, coin, score, sideways_days, status
        FROM watchlist
        WHERE status != 'removed'
        ORDER BY score DESC, symbol ASC
        """
    )
    rows: List[Dict[str, Any]] = []
    for r in cur.fetchall():
        sym = str(r[0]).strip().upper()
        if not sym:
            continue
        rows.append(
            {
                "symbol": sym,
                "coin": str(r[1] or "").strip() or sym.replace("USDT", ""),
                "pool_score": float(r[2] or 0),
                "sideways_days": int(r[3] or 0),
                "status": str(r[4] or ""),
            }
        )
    return rows


def _fetch_maps_for_symbols(
    symbols: List[str],
    p: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], str]:
    """
    仅服务收筹池 symbol 的行情/费率。
    小规模：按 symbol 询价（约 2N weight）；大规模：bulk 再过滤（约 50 weight）。
    """
    sym_set: Set[str] = {str(s).strip().upper() for s in symbols if s}
    if not sym_set:
        return {}, {}, "none"

    interval = float(p.get("api_min_interval_sec") or 0.12)
    threshold = int(p.get("bulk_ticker_threshold") or 40)
    ticker_map: Dict[str, Dict[str, float]] = {}
    funding_map: Dict[str, float] = {}

    if len(sym_set) <= threshold:
        mode = "per_symbol"
        for sym in sorted(sym_set):
            t = _binance_get(
                "/fapi/v1/ticker/24hr",
                {"symbol": sym},
                min_interval_sec=interval,
            )
            t_row = t[0] if isinstance(t, list) and t else t
            if isinstance(t_row, dict) and t_row.get("symbol"):
                ticker_map[sym] = {
                    "px_chg": float(t_row.get("priceChangePercent") or 0),
                    "vol": float(t_row.get("quoteVolume") or 0),
                    "price": float(t_row.get("lastPrice") or 0),
                }
            prem = _binance_get(
                "/fapi/v1/premiumIndex",
                {"symbol": sym},
                min_interval_sec=interval,
            )
            p_row = prem[0] if isinstance(prem, list) and prem else prem
            if isinstance(p_row, dict):
                funding_map[sym] = float(p_row.get("lastFundingRate") or 0)
        return ticker_map, funding_map, mode

    mode = "bulk_filter"
    tickers = _binance_get("/fapi/v1/ticker/24hr", min_interval_sec=interval) or []
    premiums = _binance_get("/fapi/v1/premiumIndex", min_interval_sec=interval) or []
    for p_row in premiums:
        sym = str(p_row.get("symbol") or "")
        if sym in sym_set:
            funding_map[sym] = float(p_row.get("lastFundingRate") or 0)
    for t in tickers:
        sym = str(t.get("symbol") or "")
        if sym not in sym_set:
            continue
        ticker_map[sym] = {
            "px_chg": float(t.get("priceChangePercent") or 0),
            "vol": float(t.get("quoteVolume") or 0),
            "price": float(t.get("lastPrice") or 0),
        }
    return ticker_map, funding_map, mode


def _range_6h_pct(symbol: str, *, min_interval_sec: float) -> Optional[float]:
    kl = _binance_get(
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": "1h", "limit": 6},
        min_interval_sec=min_interval_sec,
    )
    if not kl or len(kl) < 2:
        return None
    highs = [float(k[2]) for k in kl]
    lows = [float(k[3]) for k in kl]
    hi, lo = max(highs), min(lows)
    mid = (hi + lo) / 2.0
    if mid <= 0:
        return None
    return (hi - lo) / mid * 100.0


def _oi_metrics(
    symbol: str,
    *,
    min_interval_sec: float,
    oi_period: str = "1h",
) -> Optional[Dict[str, float]]:
    """默认仅 1 次 openInterestHist（1h×6），避免先拉 15m×24 再回退。"""
    period = str(oi_period or "1h").strip().lower()
    if period == "15m":
        hist = _binance_get(
            "/futures/data/openInterestHist",
            {"symbol": symbol, "period": "15m", "limit": 24},
            min_interval_sec=min_interval_sec,
        )
        if not hist or len(hist) < 5:
            return None
        curr = float(hist[-1]["sumOpenInterestValue"])
        prev_1h = float(hist[-4]["sumOpenInterestValue"])
        prev_15m = float(hist[-2]["sumOpenInterestValue"])
        base_6h = float(hist[0]["sumOpenInterestValue"])
        return {
            "oi_usd": curr,
            "oi_delta_15m_pct": ((curr - prev_15m) / prev_15m * 100.0) if prev_15m > 0 else 0.0,
            "oi_delta_1h_pct": ((curr - prev_1h) / prev_1h * 100.0) if prev_1h > 0 else 0.0,
            "oi_delta_6h_pct": ((curr - base_6h) / base_6h * 100.0) if base_6h > 0 else 0.0,
        }

    hist = _binance_get(
        "/futures/data/openInterestHist",
        {"symbol": symbol, "period": "1h", "limit": 6},
        min_interval_sec=min_interval_sec,
    )
    if not hist or len(hist) < 2:
        return None
    curr = float(hist[-1]["sumOpenInterestValue"])
    prev_1h = float(hist[-2]["sumOpenInterestValue"])
    base_6h = float(hist[0]["sumOpenInterestValue"])
    d1h = ((curr - prev_1h) / prev_1h * 100.0) if prev_1h > 0 else 0.0
    d6h = ((curr - base_6h) / base_6h * 100.0) if base_6h > 0 else 0.0
    return {"oi_usd": curr, "oi_delta_1h_pct": d1h, "oi_delta_6h_pct": d6h}


def _depth_scan_symbol(
    row: Dict[str, Any],
    p: Dict[str, Any],
) -> None:
    """单标的：1h K 线 + 1 次 OI（可配置），写回 row。"""
    sym = row["symbol"]
    interval = float(p.get("api_min_interval_sec") or 0.12)
    rng = _range_6h_pct(sym, min_interval_sec=interval)
    if rng is not None:
        row["range_6h_pct"] = rng
    oi = _oi_metrics(
        sym,
        min_interval_sec=interval,
        oi_period=str(p.get("oi_period") or "1h"),
    )
    if oi:
        row.update(oi)


def _funding_side_metadata(fr_pct: float) -> Dict[str, Any]:
    """
    费率极端保留 |fr| 门槛；方向用于执行侧过滤：
    负费率 → 仅 LONG；正费率 → 仅 SHORT。
    """
    fr = float(fr_pct)
    abs_fr = abs(fr)
    if fr < 0:
        return {
            "funding_sign": "negative",
            "funding_sign_label": "负费率",
            "funding_extreme_label": "负费率极端",
            "allowed_side": "LONG",
            "allowed_side_label": "仅做多",
            "funding_rate_abs_pct": round(abs_fr, 4),
        }
    if fr > 0:
        return {
            "funding_sign": "positive",
            "funding_sign_label": "正费率",
            "funding_extreme_label": "正费率极端",
            "allowed_side": "SHORT",
            "allowed_side_label": "仅做空",
            "funding_rate_abs_pct": round(abs_fr, 4),
        }
    return {
        "funding_sign": "zero",
        "funding_sign_label": "零费率",
        "funding_extreme_label": "费率中性",
        "allowed_side": None,
        "allowed_side_label": None,
        "funding_rate_abs_pct": 0.0,
    }


def _attach_funding_side(row: Dict[str, Any]) -> None:
    """写入 funding_sign / allowed_side 等（就地修改 row）。"""
    meta = _funding_side_metadata(float(row.get("fr_pct") or 0))
    row.update(meta)


def _enrich_powder_keg_item(item: Dict[str, Any]) -> Dict[str, Any]:
    fr = item.get("funding_rate_pct")
    if fr is None and item.get("detail"):
        fr = item["detail"].get("fr_pct")
    if fr is not None:
        item.update(_funding_side_metadata(float(fr)))
    return item


def _passes_hard_filters(row: Dict[str, Any], p: Dict[str, Any]) -> bool:
    if float(row["vol"]) < float(p["min_vol_24h_usd"]):
        return False
    if abs(float(row["px_chg"])) > float(p["max_px_chg_24h_pct"]):
        return False
    rng = row.get("range_6h_pct")
    if rng is None or float(rng) > float(p["max_range_6h_pct"]):
        return False
    if abs(float(row["fr_pct"])) < float(p["min_fr_abs_pct"]):
        return False
    oi = row.get("oi_usd")
    if oi is None or float(oi) < float(p["min_oi_usd"]):
        return False
    d1h = float(row.get("oi_delta_1h_pct") or 0)
    d6h = float(row.get("oi_delta_6h_pct") or 0)
    if d1h < float(p["min_oi_delta_1h_pct"]) and d6h < float(p["min_oi_delta_6h_pct"]):
        return False
    return True


def _score_row(row: Dict[str, Any], p: Dict[str, Any]) -> float:
    d1h = float(row.get("oi_delta_1h_pct") or 0)
    d6h = float(row.get("oi_delta_6h_pct") or 0)
    oi_surge = max(d1h if d1h > 0 else 0.0, d6h if d6h > 0 else 0.0)
    oi_part = min(oi_surge, 25.0) * 1.2
    fr_part = min(abs(float(row["fr_pct"])), 0.25) * 120.0
    rng = float(row.get("range_6h_pct") or p["max_range_6h_pct"])
    flat_part = max(0.0, float(p["max_range_6h_pct"]) - rng) * 4.0
    px_part = max(0.0, float(p["max_px_chg_24h_pct"]) - abs(float(row["px_chg"]))) * 2.0
    return round(oi_part + fr_part + flat_part + px_part, 2)


def _summary_line(row: Dict[str, Any]) -> str:
    coin = row["coin"]
    d6h = float(row.get("oi_delta_6h_pct") or 0)
    d1h = float(row.get("oi_delta_1h_pct") or 0)
    fr = float(row["fr_pct"])
    px = float(row["px_chg"])
    rng = float(row.get("range_6h_pct") or 0)
    side = str(row.get("allowed_side_label") or "")
    if fr < 0:
        fr_tag = f"负费率{fr:+.3f}%|{abs(fr):.3f}%"
    elif fr > 0:
        fr_tag = f"正费率{fr:+.3f}%|{abs(fr):.3f}%"
    else:
        fr_tag = f"费率{fr:+.3f}%"
    return (
        f"🧨{coin} OI1h{d1h:+.1f}%/6h{d6h:+.1f}% "
        f"{fr_tag}{('→' + side) if side else ''} 24h{px:+.1f}% 6h振幅{rng:.1f}%"
    )


def _log_powder_keg_scan_summary(out: Dict[str, Any]) -> None:
    """统一 INFO/WARNING，便于确认是否已向币安拉数。"""
    if not out.get("ok"):
        logger.warning(
            "powder_keg 扫描中止 error=%s watchlist_count=%s message=%s",
            out.get("error"),
            out.get("watchlist_count"),
            out.get("message") or out.get("error"),
        )
        return
    picked = list(out.get("items") or [])
    sym_sample = ", ".join(
        f"{i.get('symbol')}({i.get('funding_sign', '?')})" for i in picked[:5]
    )
    stats = out.get("scan_stats") or {}
    logger.info(
        "powder_keg 扫描完成 pool=%s binance_ticker=%s binance_funding=%s "
        "api_mode=%s pre=%s depth=%s oi_ok=%s kline_ok=%s matched=%s top_n=%s "
        "picked=%s elapsed_sec=%.1f %s",
        out.get("watchlist_count"),
        stats.get("ticker_rows"),
        stats.get("funding_rows"),
        out.get("api_mode"),
        out.get("scanned_pre"),
        stats.get("depth_scanned"),
        stats.get("oi_fetched"),
        stats.get("kline_fetched"),
        out.get("matched"),
        out.get("top_n"),
        len(picked),
        float(out.get("elapsed_sec") or 0),
        f"symbols=[{sym_sample}]" if sym_sample else "",
    )


def scan_powder_keg_candidates(
    *,
    params: Optional[Dict[str, Any]] = None,
    quiet: bool = False,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """扫描并返回候选（不写库）；宇宙默认收筹池 watchlist。"""
    p = dict(powder_keg_params())
    if params:
        p.update(params)

    universe = str(p.get("universe") or "watchlist").strip().lower()
    if universe != "watchlist":
        return {
            "ok": False,
            "error": "unsupported_universe",
            "message": f"POWDER_KEG_UNIVERSE={universe!r} 未实现，仅支持 watchlist",
            "candidates": [],
        }

    own_conn = conn is None
    db = conn or init_db()
    pool_rows: List[Dict[str, Any]] = []
    pre: List[Dict[str, Any]] = []
    api_mode = "unknown"
    t_scan = time.monotonic()
    try:
        pool_rows = _load_watchlist_universe(db)
        logger.info(
            "powder_keg 开始扫描 universe=%s 收筹池=%s",
            universe,
            len(pool_rows),
        )
        if not pool_rows:
            out_empty: Dict[str, Any] = {
                "ok": False,
                "error": "watchlist_empty",
                "message": "收筹池为空，请先运行 accumulation_radar pool（每日收筹扫描）",
                "universe": "watchlist",
                "watchlist_count": 0,
                "candidates": [],
            }
            _log_powder_keg_scan_summary(out_empty)
            return out_empty

        pool_syms = [ent["symbol"] for ent in pool_rows]
        logger.info(
            "powder_keg 拉取币安行情/费率 symbols=%s …",
            len(pool_syms),
        )
        ticker_map, funding_map, api_mode = _fetch_maps_for_symbols(pool_syms, p)
        logger.info(
            "powder_keg 币安返回 ticker=%s funding=%s api_mode=%s",
            len(ticker_map),
            len(funding_map),
            api_mode,
        )
        if not ticker_map and not funding_map:
            out_api: Dict[str, Any] = {
                "ok": False,
                "error": "ticker_api",
                "message": "币安 24hr/premiumIndex 无有效返回",
                "watchlist_count": len(pool_rows),
                "candidates": [],
            }
            _log_powder_keg_scan_summary(out_api)
            return out_api

        pre = []
        for ent in pool_rows:
            sym = ent["symbol"]
            tk = ticker_map.get(sym)
            if not tk:
                continue
            if float(tk["vol"]) < float(p["min_vol_24h_usd"]):
                continue
            if abs(float(tk["px_chg"])) > float(p["max_px_chg_24h_pct"]) * 1.5:
                continue
            fr = funding_map.get(sym, 0.0)
            fr_pct = fr * 100.0
            if abs(fr_pct) < float(p["min_fr_abs_pct"]) * 0.5:
                continue
            pre.append(
                {
                    "symbol": sym,
                    "coin": ent["coin"],
                    "pool_score": float(ent["pool_score"]),
                    "sideways_days": int(ent["sideways_days"]),
                    "px_chg": float(tk["px_chg"]),
                    "vol": float(tk["vol"]),
                    "price": float(tk["price"]),
                    "fr_pct": fr_pct,
                }
            )

        pre.sort(
            key=lambda r: (
                float(r.get("pool_score") or 0),
                abs(r["fr_pct"]),
                r["vol"],
            ),
            reverse=True,
        )
        cap = int(p["oi_scan_max_symbols"])
        if cap > 0 and len(pre) > cap:
            pre = pre[:cap]

        logger.info(
            "powder_keg 预筛完成 pool=%s → pre=%s (cap=%s) 开始深度扫描 OI+K线 …",
            len(pool_rows),
            len(pre),
            cap if cap > 0 else "none",
        )
        if not quiet:
            est_calls = 2 * len(pre) if api_mode == "per_symbol" else 52
            est_calls += 2 * len(pre)
            print(
                f"[powder_keg] 收筹池 {len(pool_rows)} → 预筛 {len(pre)} "
                f"(行情 {api_mode} ~{est_calls} req) 深度扫描…",
                flush=True,
            )
    finally:
        if own_conn:
            db.close()

    candidates: List[Dict[str, Any]] = []
    oi_fetched = 0
    kline_fetched = 0
    sleep_s = float(p["sleep_per_symbol_sec"])
    for row in pre:
        _depth_scan_symbol(row, p)
        if row.get("oi_usd") is not None:
            oi_fetched += 1
        if row.get("range_6h_pct") is not None:
            kline_fetched += 1
        if _passes_hard_filters(row, p):
            _attach_funding_side(row)
            row["score"] = _score_row(row, p)
            row["summary_line"] = _summary_line(row)
            candidates.append(row)
        if sleep_s > 0:
            time.sleep(sleep_s)

    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    top_n = int(p["top_n"])
    picked = candidates[:top_n]

    out_ok: Dict[str, Any] = {
        "ok": True,
        "run_cst": _now_cst_label(),
        "universe": universe,
        "watchlist_count": len(pool_rows),
        "scanned_pre": len(pre),
        "matched": len(candidates),
        "top_n": top_n,
        "items": picked,
        "params": p,
        "api_mode": api_mode,
        "scan_stats": {
            "ticker_rows": len(ticker_map),
            "funding_rows": len(funding_map),
            "depth_scanned": len(pre),
            "oi_fetched": oi_fetched,
            "kline_fetched": kline_fetched,
        },
        "elapsed_sec": round(time.monotonic() - t_scan, 2),
    }
    _log_powder_keg_scan_summary(out_ok)
    return out_ok


def _record_powder_keg_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    run_cst: str,
    run_at_ms: int,
    watchlist_count: int,
    scanned_pre: int,
    matched: int,
    inserted: int,
    api_mode: str,
) -> None:
    ensure_powder_keg_schema(conn)
    conn.execute(
        """
        INSERT OR REPLACE INTO powder_keg_runs (
            run_id, run_at_ms, run_cst, watchlist_count,
            scanned_pre, matched, inserted, api_mode
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            run_at_ms,
            run_cst,
            int(watchlist_count),
            int(scanned_pre),
            int(matched),
            int(inserted),
            str(api_mode),
        ),
    )


def persist_powder_keg_watchlist(
    conn: sqlite3.Connection,
    items: List[Dict[str, Any]],
    *,
    run_cst: str,
    retention_hours: int = 24,
    now: Optional[datetime] = None,
    run_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """入库 Top N：同 symbol 覆盖旧记录；表内保证每币仅一条（最新 run_at_ms）。"""
    ensure_powder_keg_schema(conn)
    t0 = now or _now_cst()
    run_id = _run_id_from_cst(run_cst)
    run_at_ms = int(t0.timestamp() * 1000)
    generated_date = t0.strftime("%Y-%m-%d")
    items = _dedupe_items_for_insert(items)
    removed_same_symbol = _delete_watchlist_symbols(
        conn, [r["symbol"] for r in items]
    )
    cur = conn.cursor()
    for rank, row in enumerate(items, start=1):
        detail = {
            k: row.get(k)
            for k in (
                "pool_score",
                "sideways_days",
                "oi_delta_15m_pct",
                "oi_delta_1h_pct",
                "oi_delta_6h_pct",
                "fr_pct",
                "funding_sign",
                "funding_sign_label",
                "funding_extreme_label",
                "allowed_side",
                "allowed_side_label",
                "funding_rate_abs_pct",
                "px_chg",
                "range_6h_pct",
                "vol",
            )
            if row.get(k) is not None
        }
        cur.execute(
            """
            INSERT OR REPLACE INTO powder_keg_watchlist (
                run_id, run_at_ms, generated_date,
                symbol, coin, run_cst, rank_in_list, score,
                oi_usd, oi_delta_1h_pct, oi_delta_6h_pct,
                funding_rate_pct, px_chg_24h_pct, range_6h_pct,
                price, vol_24h_usd, summary_line, detail_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                run_at_ms,
                generated_date,
                row["symbol"],
                row["coin"],
                run_cst,
                rank,
                float(row["score"]),
                float(row.get("oi_usd") or 0),
                float(row.get("oi_delta_1h_pct") or 0),
                float(row.get("oi_delta_6h_pct") or 0),
                float(row["fr_pct"]),
                float(row["px_chg"]),
                float(row.get("range_6h_pct") or 0),
                float(row.get("price") or 0),
                float(row.get("vol") or 0),
                str(row.get("summary_line") or ""),
                json.dumps(detail, ensure_ascii=False),
            ),
        )
    deduped_rows = _prune_duplicate_symbols_in_watchlist(conn)
    pruned = _prune_powder_keg_watchlist(
        conn, retention_hours=retention_hours, now=t0
    )
    meta = run_meta or {}
    _record_powder_keg_run(
        conn,
        run_id=run_id,
        run_cst=run_cst,
        run_at_ms=run_at_ms,
        watchlist_count=int(meta.get("watchlist_count") or 0),
        scanned_pre=int(meta.get("scanned_pre") or 0),
        matched=int(meta.get("matched") or 0),
        inserted=len(items),
        api_mode=str(meta.get("api_mode") or ""),
    )
    _prune_powder_keg_runs(conn, retention_hours=retention_hours, now=t0)
    conn.commit()
    return {
        "run_id": run_id,
        "run_cst": run_cst,
        "inserted": len(items),
        "replaced_symbol_rows": removed_same_symbol,
        "deduped_symbol_rows": deduped_rows,
        "pruned_rows": pruned,
        "retention_hours": int(retention_hours),
    }


def _prune_powder_keg_runs(
    conn: sqlite3.Connection,
    *,
    retention_hours: int,
    now: Optional[datetime] = None,
) -> None:
    cutoff_ms = _retention_cutoff_ms(retention_hours=retention_hours, now=now)
    conn.execute("DELETE FROM powder_keg_runs WHERE run_at_ms < ?", (cutoff_ms,))


def _row_to_item(r: Tuple[Any, ...]) -> Dict[str, Any]:
    det: Dict[str, Any] = {}
    if r[17]:
        try:
            det = json.loads(r[17])
        except json.JSONDecodeError:
            det = {}
    item = {
        "run_id": r[0],
        "run_at_ms": r[1],
        "generated_date": r[2],
        "symbol": r[3],
        "coin": r[4],
        "run_cst": r[5],
        "rank": r[6],
        "score": r[7],
        "oi_usd": r[8],
        "oi_delta_1h_pct": r[9],
        "oi_delta_6h_pct": r[10],
        "funding_rate_pct": r[11],
        "px_chg_24h_pct": r[12],
        "range_6h_pct": r[13],
        "price": r[14],
        "vol_24h_usd": r[15],
        "summary_line": r[16],
        "detail": det,
    }
    return _enrich_powder_keg_item(item)


_SELECT_COLS = """
        run_id, run_at_ms, generated_date,
        symbol, coin, run_cst, rank_in_list, score,
        oi_usd, oi_delta_1h_pct, oi_delta_6h_pct,
        funding_rate_pct, px_chg_24h_pct, range_6h_pct,
        price, vol_24h_usd, summary_line, detail_json
"""


def load_powder_keg_watchlist(
    conn: sqlite3.Connection,
    *,
    latest_only: bool = False,
    retention_hours: Optional[int] = None,
) -> Dict[str, Any]:
    """读取火药桶名单（按 symbol 去重后每币一条）；默认按 score 排序。"""
    ensure_powder_keg_schema(conn)
    _prune_duplicate_symbols_in_watchlist(conn)
    conn.commit()
    p = powder_keg_params()
    hours = int(retention_hours if retention_hours is not None else p["retention_hours"])
    cutoff_ms = _retention_cutoff_ms(retention_hours=hours)
    cur = conn.cursor()

    cur.execute(
        f"""
        SELECT {_SELECT_COLS.strip()}
        FROM powder_keg_watchlist
        WHERE run_at_ms >= ?
        ORDER BY score DESC, run_at_ms DESC, symbol ASC
        """,
        (cutoff_ms,),
    )
    all_rows = [_row_to_item(tuple(r)) for r in cur.fetchall()]

    if latest_only:
        latest_run_id: Optional[str] = None
        if all_rows:
            latest_run_id = str(max(all_rows, key=lambda x: int(x["run_at_ms"]))["run_id"])
            items = [it for it in all_rows if str(it["run_id"]) == latest_run_id]
        else:
            items = []
    else:
        latest_run_id = all_rows[0]["run_id"] if all_rows else None
        items = all_rows

    updated = items[0]["run_cst"] if items else None
    cur.execute(
        """
        SELECT run_id, run_cst, run_at_ms, watchlist_count, scanned_pre, matched, inserted, api_mode
        FROM powder_keg_runs
        WHERE run_at_ms >= ?
        ORDER BY run_at_ms DESC
        LIMIT 20
        """,
        (cutoff_ms,),
    )
    run_audit = [
        {
            "run_id": r[0],
            "run_cst": r[1],
            "run_at_ms": r[2],
            "watchlist_count": r[3],
            "scanned_pre": r[4],
            "matched": r[5],
            "inserted": r[6],
            "api_mode": r[7],
        }
        for r in cur.fetchall()
    ]
    fr_thr = float(p.get("min_fr_abs_pct") or 0)
    return {
        "ok": True,
        "items": items,
        "count": len(items),
        "unique_symbols": len(items),
        "updated_at_cst": updated,
        "latest_run_id": latest_run_id,
        "retention_hours": hours,
        "rows_in_window": len(all_rows),
        "run_audit": run_audit,
        "universe": "watchlist",
        "dedupe_by_symbol": True,
        "funding_extreme_rules": {
            "threshold_abs_pct": fr_thr,
            "negative": {"allowed_side": "LONG", "label": "负费率极端 → 仅做多"},
            "positive": {"allowed_side": "SHORT", "label": "正费率极端 → 仅做空"},
        },
    }


def run_powder_keg_radar_once(*, quiet: bool = False) -> Dict[str, Any]:
    conn = init_db()
    try:
        out = scan_powder_keg_candidates(quiet=quiet, conn=conn)
        if not out.get("ok"):
            return out
        p = out.get("params") or powder_keg_params()
        logger.info(
            "powder_keg 开始入库 matched=%s top_n=%s",
            out.get("matched"),
            out.get("top_n"),
        )
        meta = persist_powder_keg_watchlist(
            conn,
            list(out.get("items") or []),
            run_cst=str(out.get("run_cst") or _now_cst_label()),
            retention_hours=int(p.get("retention_hours") or 24),
            run_meta={
                "watchlist_count": out.get("watchlist_count"),
                "scanned_pre": out.get("scanned_pre"),
                "matched": out.get("matched"),
                "api_mode": out.get("api_mode"),
            },
        )
        out["persist"] = meta
        payload = load_powder_keg_watchlist(conn)
        out["watchlist"] = payload
        logger.info(
            "powder_keg 入库完成 inserted=%s replaced=%s deduped=%s 表内=%s",
            meta.get("inserted"),
            meta.get("replaced_symbol_rows"),
            meta.get("deduped_symbol_rows"),
            payload.get("count"),
        )
        if not quiet:
            print(
                f"[powder_keg] 入库 {payload.get('count', 0)} 个: "
                + ", ".join(
                    i["symbol"] for i in (payload.get("items") or [])[:10]
                ),
                flush=True,
            )
        return out
    finally:
        conn.close()


def powder_keg_side_by_symbol(*, within_retention: bool = True) -> Dict[str, str]:
    """当前火药桶表：symbol → 允许方向 LONG/SHORT（费率正负）。"""
    conn = init_db()
    try:
        ensure_powder_keg_schema(conn)
        p = powder_keg_params()
        cur = conn.cursor()
        if within_retention:
            cutoff_ms = _retention_cutoff_ms(retention_hours=int(p["retention_hours"]))
            cur.execute(
                """
                SELECT symbol, funding_rate_pct FROM powder_keg_watchlist
                WHERE run_at_ms >= ?
                """,
                (cutoff_ms,),
            )
        else:
            cur.execute("SELECT symbol, funding_rate_pct FROM powder_keg_watchlist")
        out: Dict[str, str] = {}
        for sym, fr in cur.fetchall():
            s = str(sym or "").strip().upper()
            if not s:
                continue
            allowed = _funding_side_metadata(float(fr or 0)).get("allowed_side")
            if allowed in ("LONG", "SHORT"):
                out[s] = str(allowed)
        return out
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()


def powder_keg_symbols(*, within_retention: bool = True) -> List[str]:
    """保留窗口内出现过的 symbol（去重）；微观扫描用。"""
    conn = init_db()
    try:
        ensure_powder_keg_schema(conn)
        p = powder_keg_params()
        cur = conn.cursor()
        if within_retention:
            cutoff_ms = _retention_cutoff_ms(retention_hours=int(p["retention_hours"]))
            cur.execute(
                """
                SELECT DISTINCT symbol FROM powder_keg_watchlist
                WHERE run_at_ms >= ?
                ORDER BY symbol ASC
                """,
                (cutoff_ms,),
            )
        else:
            cur.execute(
                "SELECT DISTINCT symbol FROM powder_keg_watchlist ORDER BY symbol ASC"
            )
        return [str(x[0]).strip().upper() for x in cur.fetchall() if x and x[0]]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="火药桶宏观雷达（币安）")
    ap.add_argument("--once", action="store_true", help="扫描并写入 DB")
    args = ap.parse_args()
    if not args.once:
        ap.print_help()
        sys.exit(0)
    run_powder_keg_radar_once(quiet=False)


if __name__ == "__main__":
    main()

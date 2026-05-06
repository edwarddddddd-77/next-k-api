#!/usr/bin/env python3
"""
庄家收筹雷达 v1 — 发现庄家横盘吸筹 + OI异动

核心逻辑（Patrick教的）：
1. 庄家拉盘前必须先收筹 → 长期横盘+低量 = 收筹中
2. OI暴涨 = 大资金进场建仓 = 即将拉盘
3. 两个信号叠加 = 最强信号

两个模块：
A. 横盘收筹标的池（每天扫一次）→ 找正在被庄家收筹的币
B. OI异动监控（每小时扫）→ 标的池内的币有OI异动立即报警

数据源：币安合约API（免费公开，零成本）
"""

import json
import os
import sys
import time
import requests
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from datetime import date as date_cls, datetime, timezone, timedelta
from pathlib import Path

# 兼容 Windows 控制台编码，避免 emoji 打印导致崩溃，并启用行缓冲便于看进度
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

# === 加载 .env ===
env_file = Path(__file__).parent / ".env.oi"
if env_file.exists():
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# === 配置 ===
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
FAPI = "https://fapi.binance.com"
db_dir = os.getenv("DATA_DIR", Path(__file__).parent)
DB_PATH = Path(db_dir) / "accumulation.db"
OI_RADAR_SNAPSHOT_PATH = Path(db_dir) / "oi_radar_snapshot.json"
HEAT_ACCUM_RETENTION_DAYS = 7  # 含今天在内共 7 个日历日
AMBUSH_WATCH_RETENTION_DAYS = 7  # 暗流 / 低市值埋伏看盘，与热度收筹一致
AMBUSH_WATCH_TOP_N = 2  # 暗流 / 低市值：全埋伏榜（已按 total 降序）命中条件后取分数最高的前 N 条入库
_LEGACY_HEAT_ACCUM_JSON = Path(db_dir) / "heat_accum_watchlist.json"
# 热度收筹表：突破—回踩—延续状态机（默认 4h K 线，不含 OI）
HEAT_ACCUM_BPC_INTERVAL = "4h"
HEAT_ACCUM_BPC_KLINE_LIMIT = 120
BPC_PHASE_ZH: Dict[str, str] = {
    "idle": "待突破",
    "post_breakout": "突破后",
    "pullback": "回踩中",
    "continuation": "回踩结束",
}


def _persist_oi_radar_snapshot(payload: Dict[str, Any]) -> None:
    """供 GET /api/accumulation/oi-radar 读盘；定时任务与后台 refresh 写入同一路径。"""
    if not payload.get("ok"):
        return
    tmp = OI_RADAR_SNAPSHOT_PATH.with_suffix(".json.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp.replace(OI_RADAR_SNAPSHOT_PATH)
        print(f"  💾 OI 快照已写入 {OI_RADAR_SNAPSHOT_PATH}")
    except Exception as e:
        print(f"⚠️ OI 快照写入失败: {e}")


def _heat_accum_summary_line(sig: Dict[str, Any]) -> str:
    """与「值得关注」热力+收筹条一致的单行摘要（CST 语境）。"""
    tags = list(sig.get("tags") or [])
    coin = sig.get("coin") or ""
    sw = sig.get("sideways_days") or 0
    return f"🔥💤 {coin} 热度({'+'.join(tags)})+收筹{sw}天=OI将涨"


def _heat_accum_now_cst(now: datetime) -> datetime:
    cst = timezone(timedelta(hours=8))
    if now.tzinfo is None:
        return now.replace(tzinfo=cst)
    return now.astimezone(cst)


def _heat_accum_cutoff_iso(now: datetime) -> str:
    """早于该生成日（不含）的行删除；含今天在内共 RETENTION_DAYS 个日历日。"""
    now_cst = _heat_accum_now_cst(now)
    today = now_cst.date()
    cutoff = today - timedelta(days=HEAT_ACCUM_RETENTION_DAYS - 1)
    return cutoff.isoformat()


def _heat_accum_prune(conn: sqlite3.Connection, now: datetime) -> None:
    cutoff_s = _heat_accum_cutoff_iso(now)
    conn.execute("DELETE FROM heat_accum_watch WHERE generated_date < ?", (cutoff_s,))


def _parse_bpc_for_item(bpc_json: Optional[str], bpc_updated_cst: Optional[str]) -> Optional[Dict[str, Any]]:
    if not bpc_json:
        return None
    try:
        d = json.loads(bpc_json)
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    ph = str(d.get("phase") or "idle")
    return {
        "ok": d.get("ok", True),
        "phase": ph,
        "phase_zh": BPC_PHASE_ZH.get(ph, ph),
        "reason": d.get("reason"),
        "continuation_reason": d.get("continuation_reason"),
        "pullback_vol_contracted": d.get("pullback_vol_contracted"),
        "breakout_level": d.get("breakout_level"),
        "peak_after_breakout": d.get("peak_after_breakout"),
        "last_invalid_reason": d.get("last_invalid_reason"),
        "interval": d.get("interval") or HEAT_ACCUM_BPC_INTERVAL,
        "evaluated_at_cst": bpc_updated_cst,
    }


def _sqlite_row_to_watch_item(row: Tuple[Any, ...]) -> Dict[str, Any]:
    (
        symbol,
        coin,
        generated_date,
        last_seen_cst,
        heat,
        sideways_days,
        tags_json,
        low_price,
        high_price,
        price,
        zone_json,
        zone_reason,
        summary_line,
        bpc_json,
        bpc_updated_cst,
    ) = row
    tags: List[Any] = []
    if tags_json:
        try:
            t = json.loads(tags_json)
            if isinstance(t, list):
                tags = t
        except Exception:
            pass
    zone: Optional[Any] = None
    if zone_json:
        try:
            zone = json.loads(zone_json)
        except Exception:
            zone = None
    return {
        "symbol": symbol,
        "coin": coin,
        "generated_date": generated_date,
        "last_seen_cst": last_seen_cst,
        "heat": heat,
        "tags": tags,
        "sideways_days": sideways_days,
        "low_price": low_price,
        "high_price": high_price,
        "price": price,
        "zone": zone,
        "zone_reason": zone_reason,
        "summary_line": summary_line,
        "bpc": _parse_bpc_for_item(
            str(bpc_json) if bpc_json else None,
            str(bpc_updated_cst) if bpc_updated_cst else None,
        ),
    }


def _heat_accum_fetch_payload(conn: sqlite3.Connection, now: datetime) -> Dict[str, Any]:
    now_cst = _heat_accum_now_cst(now)
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"
    cur = conn.cursor()
    cur.execute(
        """
        SELECT symbol, coin, generated_date, last_seen_cst, heat, sideways_days, tags_json,
               low_price, high_price, price, zone_json, zone_reason, summary_line,
               bpc_json, bpc_updated_cst
        FROM heat_accum_watch
        ORDER BY generated_date DESC, symbol DESC
        """
    )
    rows = cur.fetchall()
    items = [_sqlite_row_to_watch_item(tuple(r)) for r in rows]
    seen_times = [it.get("last_seen_cst") for it in items if isinstance(it.get("last_seen_cst"), str)]
    updated_at = max(seen_times) if seen_times else now_label
    bpc_times: List[str] = []
    for it in items:
        b = it.get("bpc")
        if isinstance(b, dict) and b.get("evaluated_at_cst"):
            bpc_times.append(str(b["evaluated_at_cst"]))
    bpc_snapshot_cst = max(bpc_times) if bpc_times else None
    return {
        "ok": True,
        "items": items,
        "updated_at_cst": updated_at,
        "bpc_snapshot_cst": bpc_snapshot_cst,
        "bpc_interval": HEAT_ACCUM_BPC_INTERVAL,
        "retention_days": HEAT_ACCUM_RETENTION_DAYS,
        "storage": "sqlite",
    }


def load_heat_accum_watchlist_from_db(
    conn: sqlite3.Connection,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """供 HTTP GET：按保留策略清理过期行后返回当前列表。"""
    if now is None:
        now = datetime.now(timezone(timedelta(hours=8)))
    _heat_accum_prune(conn, now)
    conn.commit()
    return _heat_accum_fetch_payload(conn, now)


def refresh_all_heat_accum_watch_prices(
    conn: sqlite3.Connection,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    对 heat_accum_watch 全表刷新现价、last_seen、摘要；清空 zone（已不再计算买入/进场区间）。
    每日 12:00 CST 定时与维护面板调用。
    """
    if now is None:
        now = datetime.now(timezone(timedelta(hours=8)))
    now_cst = _heat_accum_now_cst(now)
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"

    _heat_accum_prune(conn, now)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT symbol, coin, heat, sideways_days, tags_json, low_price, high_price, price
        FROM heat_accum_watch
        """
    )
    rows = cur.fetchall()
    if not rows:
        conn.commit()
        return _heat_accum_fetch_payload(conn, now)

    ticker_map: Dict[str, float] = {}
    t24 = api_get("/fapi/v1/ticker/24hr")
    if isinstance(t24, list):
        for r in t24:
            try:
                sym = str(r.get("symbol") or "")
                px = float(r.get("lastPrice") or 0.0)
                if sym and px > 0:
                    ticker_map[sym] = px
            except Exception:
                continue

    recalculated = 0
    for row in rows:
        symbol = str(row[0] or "")
        if not symbol:
            continue
        coin = row[1]
        heat = row[2]
        sideways_days = row[3]
        tags_raw = row[4]
        low_price = float(row[5] or 0.0)
        high_price = float(row[6] or 0.0)
        old_price = float(row[7] or 0.0)
        price = float(ticker_map.get(symbol) or old_price or 0.0)

        tags: List[Any] = []
        if isinstance(tags_raw, str) and tags_raw:
            try:
                parsed = json.loads(tags_raw)
                if isinstance(parsed, list):
                    tags = parsed
            except Exception:
                tags = []

        sig = {
            "coin": coin,
            "symbol": symbol,
            "heat": heat,
            "tags": tags,
            "sideways_days": sideways_days,
            "low_price": low_price,
            "high_price": high_price,
            "price": price,
        }
        summary = _heat_accum_summary_line(sig)
        cur.execute(
            """
            UPDATE heat_accum_watch SET
                price = ?, last_seen_cst = ?, zone_json = ?, zone_reason = ?, summary_line = ?
            WHERE symbol = ?
            """,
            (price, now_label, None, None, summary, symbol),
        )
        recalculated += 1

    conn.commit()
    try:
        patch_oi_radar_snapshot_watchlists_from_db(conn)
    except Exception as e:
        print(f"⚠️ patch oi_radar snapshot after heat watch price refresh failed: {e}")

    payload = _heat_accum_fetch_payload(conn, now)
    payload["recalculated"] = recalculated
    return payload


def refresh_all_heat_accum_bpc_states(
    conn: sqlite3.Connection,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    对 heat_accum_watch 全表按 4h K 线重算「突破—回踩—延续」状态（不含 OI），写入 bpc_json / bpc_updated_cst。
    供每 4 小时定时任务与维护面板手动刷新。
    """
    from breakout_pullback_fsm import BPCParams, evaluate_breakout_pullback_continuation

    if now is None:
        now = datetime.now(timezone(timedelta(hours=8)))
    now_cst = _heat_accum_now_cst(now)
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"

    _heat_accum_prune(conn, now)
    cur = conn.cursor()
    cur.execute("SELECT symbol FROM heat_accum_watch")
    syms = [str(r[0] or "") for r in cur.fetchall() if r and r[0]]
    if not syms:
        conn.commit()
        return _heat_accum_fetch_payload(conn, now)

    params = BPCParams()
    recalculated = 0
    failed_klines = 0
    for sym in syms:
        if not sym:
            continue
        kl = api_get(
            "/fapi/v1/klines",
            {"symbol": sym, "interval": HEAT_ACCUM_BPC_INTERVAL, "limit": HEAT_ACCUM_BPC_KLINE_LIMIT},
        )
        time.sleep(0.06)
        if not kl:
            err_payload = {
                "ok": False,
                "reason": "no_klines",
                "phase": "idle",
                "interval": HEAT_ACCUM_BPC_INTERVAL,
            }
            cur.execute(
                """
                UPDATE heat_accum_watch SET bpc_json = ?, bpc_updated_cst = ?
                WHERE symbol = ?
                """,
                (json.dumps(err_payload, ensure_ascii=False), now_label, sym),
            )
            failed_klines += 1
            continue
        ev = evaluate_breakout_pullback_continuation(kl, params)
        ev["interval"] = HEAT_ACCUM_BPC_INTERVAL
        cur.execute(
            """
            UPDATE heat_accum_watch SET bpc_json = ?, bpc_updated_cst = ?
            WHERE symbol = ?
            """,
            (json.dumps(ev, ensure_ascii=False), now_label, sym),
        )
        recalculated += 1

    conn.commit()
    try:
        patch_oi_radar_snapshot_watchlists_from_db(conn)
    except Exception as e:
        print(f"⚠️ patch oi_radar snapshot after heat BPC refresh failed: {e}")

    payload = _heat_accum_fetch_payload(conn, now)
    payload["bpc_recalculated"] = recalculated
    payload["bpc_failed_klines"] = failed_klines
    return payload


def merge_and_persist_heat_accum_watchlist(
    conn: sqlite3.Connection,
    hot_pool_signals: List[Dict[str, Any]],
    now: datetime,
) -> Dict[str, Any]:
    """
    增量写入 accumulation.db：当前轮 hot_pool（热度+收筹）；按 symbol 主键去重；
    首次出现记 generated_date（CST，精确到分），再次命中刷新区间等字段但保留生成时间；
    仅保留生成日在最近 HEAT_ACCUM_RETENTION_DAYS 个日历日内的条目。
    """
    now_cst = _heat_accum_now_cst(now)
    generated_at_s = now_cst.strftime("%Y-%m-%d %H:%M")
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"

    _heat_accum_prune(conn, now)
    cur = conn.cursor()

    for sig in hot_pool_signals:
        if not isinstance(sig, dict):
            continue
        sym = sig.get("symbol")
        if not sym:
            continue
        sym = str(sym)
        summary = _heat_accum_summary_line(sig)
        tags_json = json.dumps(sig.get("tags") or [], ensure_ascii=False)

        cur.execute("SELECT generated_date FROM heat_accum_watch WHERE symbol = ?", (sym,))
        ex = cur.fetchone()
        if ex:
            cur.execute(
                """
                UPDATE heat_accum_watch SET
                    coin = ?, last_seen_cst = ?, heat = ?, sideways_days = ?, tags_json = ?,
                    low_price = ?, high_price = ?, price = ?, zone_json = ?, zone_reason = ?, summary_line = ?
                WHERE symbol = ?
                """,
                (
                    sig.get("coin"),
                    now_label,
                    sig.get("heat"),
                    sig.get("sideways_days"),
                    tags_json,
                    sig.get("low_price"),
                    sig.get("high_price"),
                    sig.get("price"),
                    None,
                    None,
                    summary,
                    sym,
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO heat_accum_watch (
                    symbol, coin, generated_date, last_seen_cst, heat, sideways_days, tags_json,
                    low_price, high_price, price, zone_json, zone_reason, summary_line,
                    bpc_json, bpc_updated_cst
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    sym,
                    sig.get("coin"),
                    generated_at_s,
                    now_label,
                    sig.get("heat"),
                    sig.get("sideways_days"),
                    tags_json,
                    sig.get("low_price"),
                    sig.get("high_price"),
                    sig.get("price"),
                    None,
                    None,
                    summary,
                ),
            )

    conn.commit()
    print(f"  💾 热度+收筹看盘已写入 SQLite ({DB_PATH})")
    return _heat_accum_fetch_payload(conn, now)


def _migrate_legacy_heat_accum_json(conn: sqlite3.Connection) -> None:
    """一次性：旧 heat_accum_watchlist.json → DB，成功后改名为 .bak。"""
    path = _LEGACY_HEAT_ACCUM_JSON
    if not path.is_file():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            return
        cur = conn.cursor()
        migrated = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            sym = it.get("symbol")
            if not sym:
                continue
            gd = it.get("generated_date")
            if not gd or not isinstance(gd, str):
                continue
            cur.execute("SELECT 1 FROM heat_accum_watch WHERE symbol = ?", (str(sym),))
            if cur.fetchone():
                continue
            tags_json = json.dumps(it.get("tags") or [], ensure_ascii=False)
            zone = it.get("zone")
            zone_json = json.dumps(zone, ensure_ascii=False) if zone else None
            cur.execute(
                """
                INSERT OR IGNORE INTO heat_accum_watch (
                    symbol, coin, generated_date, last_seen_cst, heat, sideways_days, tags_json,
                    low_price, high_price, price, zone_json, zone_reason, summary_line,
                    bpc_json, bpc_updated_cst
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    str(sym),
                    it.get("coin"),
                    gd[:10],
                    str(it.get("last_seen_cst") or "") or "(迁移)",
                    it.get("heat"),
                    it.get("sideways_days"),
                    tags_json,
                    it.get("low_price"),
                    it.get("high_price"),
                    it.get("price"),
                    zone_json,
                    str(it.get("zone_reason")) if it.get("zone_reason") is not None else None,
                    str(it.get("summary_line") or ""),
                ),
            )
            if cur.rowcount == 1:
                migrated += 1
        conn.commit()
        bak = path.with_suffix(".json.bak")
        path.rename(bak)
        print(f"  📦 已迁移热度+收筹看盘 JSON → DB（新增 {migrated} 条），原文件改为 {bak.name}")
    except Exception as e:
        print(f"⚠️ 迁移 heat_accum_watchlist.json 跳过: {e}")


def _ambush_watch_cutoff_iso(now: datetime) -> str:
    now_cst = _heat_accum_now_cst(now)
    today = now_cst.date()
    cutoff = today - timedelta(days=AMBUSH_WATCH_RETENTION_DAYS - 1)
    return cutoff.isoformat()


def _ambush_watch_prune(conn: sqlite3.Connection, now: datetime) -> None:
    cutoff_s = _ambush_watch_cutoff_iso(now)
    conn.execute("DELETE FROM ambush_watch WHERE generated_date < ?", (cutoff_s,))


def _sqlite_row_to_ambush_item(row: Tuple[Any, ...]) -> Dict[str, Any]:
    (
        symbol,
        signal_type,
        coin,
        generated_date,
        last_seen_cst,
        d6h,
        px_chg,
        est_mcap,
        ambush_total,
        summary_line,
    ) = row
    return {
        "symbol": symbol,
        "signal_type": signal_type,
        "coin": coin,
        "generated_date": generated_date,
        "last_seen_cst": last_seen_cst,
        "d6h": d6h,
        "px_chg": px_chg,
        "est_mcap": est_mcap,
        "ambush_total": ambush_total,
        "summary_line": summary_line,
    }


def _ambush_watch_fetch_payload(conn: sqlite3.Connection, now: datetime) -> Dict[str, Any]:
    now_cst = _heat_accum_now_cst(now)
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"
    cur = conn.cursor()
    cur.execute(
        """
        SELECT symbol, signal_type, coin, generated_date, last_seen_cst,
               d6h, px_chg, est_mcap, ambush_total, summary_line
        FROM ambush_watch
        ORDER BY signal_type ASC, (ambush_total IS NULL) ASC, ambush_total DESC, generated_date DESC, symbol ASC
        """
    )
    rows = cur.fetchall()
    items = [_sqlite_row_to_ambush_item(tuple(r)) for r in rows]
    seen_times = [it.get("last_seen_cst") for it in items if isinstance(it.get("last_seen_cst"), str)]
    updated_at = max(seen_times) if seen_times else now_label
    return {
        "ok": True,
        "items": items,
        "updated_at_cst": updated_at,
        "retention_days": AMBUSH_WATCH_RETENTION_DAYS,
        "storage": "sqlite",
    }


def load_ambush_watchlist_from_db(
    conn: sqlite3.Connection,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """供 HTTP GET：按保留策略清理过期行后返回当前列表。"""
    if now is None:
        now = datetime.now(timezone(timedelta(hours=8)))
    _ambush_watch_prune(conn, now)
    conn.commit()
    return _ambush_watch_fetch_payload(conn, now)


def patch_oi_radar_snapshot_watchlists_from_db(conn: sqlite3.Connection) -> bool:
    """
    将磁盘 oi_radar_snapshot.json 内的嵌套看盘列表与当前 SQLite 对齐。
    仅清库而未重跑雷达时调用，避免前端仍读到快照里的陈旧副本。
    """
    if not OI_RADAR_SNAPSHOT_PATH.is_file():
        return False
    try:
        raw = json.loads(OI_RADAR_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(raw, dict) or not raw.get("ok"):
        return False
    now = datetime.now(timezone(timedelta(hours=8)))
    raw["ambush_watchlist"] = load_ambush_watchlist_from_db(conn, now=now)
    raw["heat_accum_watchlist"] = load_heat_accum_watchlist_from_db(conn, now=now)
    _persist_oi_radar_snapshot(raw)
    return True


def clear_ambush_watch_table(conn: sqlite3.Connection) -> int:
    """清空表 ambush_watch。返回清空前行数。"""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ambush_watch")
    n = int(cur.fetchone()[0] or 0)
    cur.execute("DELETE FROM ambush_watch")
    conn.commit()
    return n


def clear_heat_accum_watch_table(conn: sqlite3.Connection) -> int:
    """清空表 heat_accum_watch。返回清空前行数。"""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM heat_accum_watch")
    n = int(cur.fetchone()[0] or 0)
    cur.execute("DELETE FROM heat_accum_watch")
    conn.commit()
    return n


def _sync_ambush_watch_kept_symbols(
    conn: sqlite3.Connection,
    signal_type: str,
    kept_symbols: List[str],
) -> None:
    """本轮仅保留 kept_symbols；同类型其余行删除，避免旧的前 10 命中残留。"""
    cur = conn.cursor()
    if not kept_symbols:
        cur.execute("DELETE FROM ambush_watch WHERE signal_type = ?", (signal_type,))
        return
    placeholders = ",".join("?" * len(kept_symbols))
    cur.execute(
        f"DELETE FROM ambush_watch WHERE signal_type = ? AND symbol NOT IN ({placeholders})",
        (signal_type, *kept_symbols),
    )


def merge_and_persist_ambush_watchlist(
    conn: sqlite3.Connection,
    ambush_dark: List[Dict[str, Any]],
    ambush_gem: List[Dict[str, Any]],
    now: datetime,
    mcap_str_fn,
) -> Dict[str, Any]:
    """
    埋伏榜内 🎯 暗流（OI 涨、价格横盘）与 💎 低市值+OI；调用方已截断为每类至多 AMBUSH_WATCH_TOP_N 条。
    按 (symbol, signal_type) upsert；首次命中记入 generated_date（CST，精确到分）。
    写入后删除该 signal_type 下不在本轮保留列表中的行。
    """
    now_cst = _heat_accum_now_cst(now)
    generated_at_s = now_cst.strftime("%Y-%m-%d %H:%M")
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"

    _ambush_watch_prune(conn, now)
    cur = conn.cursor()

    def upsert(sig: Dict[str, Any], signal_type: str, summary: str) -> None:
        sym = sig.get("sym") or sig.get("symbol")
        if not sym:
            return
        sym = str(sym)
        cur.execute(
            "SELECT generated_date FROM ambush_watch WHERE symbol = ? AND signal_type = ?",
            (sym, signal_type),
        )
        ex = cur.fetchone()
        atotal = float(sig.get("total") or 0)
        row = (
            sig.get("coin"),
            now_label,
            float(sig.get("d6h") or 0),
            float(sig.get("px_chg") or 0),
            float(sig.get("est_mcap") or 0),
            atotal,
            summary,
        )
        if ex:
            cur.execute(
                """
                UPDATE ambush_watch SET
                    coin = ?, last_seen_cst = ?, d6h = ?, px_chg = ?, est_mcap = ?, ambush_total = ?, summary_line = ?
                WHERE symbol = ? AND signal_type = ?
                """,
                row + (sym, signal_type),
            )
        else:
            cur.execute(
                """
                INSERT INTO ambush_watch (
                    symbol, signal_type, coin, generated_date, last_seen_cst,
                    d6h, px_chg, est_mcap, ambush_total, summary_line
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sym,
                    signal_type,
                    sig.get("coin"),
                    generated_at_s,
                    now_label,
                    float(sig.get("d6h") or 0),
                    float(sig.get("px_chg") or 0),
                    float(sig.get("est_mcap") or 0),
                    atotal,
                    summary,
                ),
            )

    for s in ambush_dark:
        if not isinstance(s, dict):
            continue
        summ = (
            f"🎯 {s['coin']} 暗流！OI{s['d6h']:+.0f}%但价格没动，市值仅{mcap_str_fn(s['est_mcap'])}"
        )
        upsert(s, "dark_flow", summ)

    for s in ambush_gem:
        if not isinstance(s, dict):
            continue
        summ = (
            f"💎 {s['coin']} 低市值{mcap_str_fn(s['est_mcap'])}+OI{s['d6h']:+.0f}%，埋伏首选"
        )
        upsert(s, "low_mcap_oi", summ)

    kept_dark = [
        str(s.get("sym") or s.get("symbol") or "")
        for s in ambush_dark
        if isinstance(s, dict) and (s.get("sym") or s.get("symbol"))
    ]
    kept_gem = [
        str(s.get("sym") or s.get("symbol") or "")
        for s in ambush_gem
        if isinstance(s, dict) and (s.get("sym") or s.get("symbol"))
    ]
    _sync_ambush_watch_kept_symbols(conn, "dark_flow", kept_dark)
    _sync_ambush_watch_kept_symbols(conn, "low_mcap_oi", kept_gem)

    conn.commit()
    print(f"  💾 暗流/低市值埋伏看盘已写入 SQLite ({DB_PATH})")
    return _ambush_watch_fetch_payload(conn, now)


# 收筹标的池参数
MIN_SIDEWAYS_DAYS = 45        # 至少横盘45天
MAX_RANGE_PCT = 80            # 横盘期价格波动<80%（宽松点，庄家盘波动可以大）
MAX_AVG_VOL_USD = 20_000_000  # 日均成交<$20M（低量才是收筹）
MIN_DATA_DAYS = 50            # 至少50天数据

# OI异动参数
MIN_OI_DELTA_PCT = 3.0        # OI变化至少3%
MIN_OI_USD = 2_000_000        # 最低OI门槛 $2M

# 放量突破参数
VOL_BREAKOUT_MULT = 3.0       # 当日Vol > 3x均值 = 放量

def api_get(endpoint, params=None):
    """币安API请求"""
    url = f"{FAPI}{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(2)
            else:
                return None
        except:
            time.sleep(1)
    return None


def init_db():
    """初始化数据库"""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist (
        symbol TEXT PRIMARY KEY,
        coin TEXT,
        added_date TEXT,
        sideways_days INT,
        range_pct REAL,
        avg_vol REAL,
        low_price REAL,
        high_price REAL,
        current_price REAL,
        score REAL,
        status TEXT DEFAULT 'watching',
        last_oi_alert TEXT,
        notes TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        alert_type TEXT,
        alert_time TEXT,
        price REAL,
        oi_delta_pct REAL,
        vol_ratio REAL,
        details TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS heat_accum_watch (
        symbol TEXT PRIMARY KEY,
        coin TEXT,
        generated_date TEXT NOT NULL,
        last_seen_cst TEXT NOT NULL,
        heat REAL,
        sideways_days INTEGER,
        tags_json TEXT,
        low_price REAL,
        high_price REAL,
        price REAL,
        zone_json TEXT,
        zone_reason TEXT,
        summary_line TEXT,
        bpc_json TEXT,
        bpc_updated_cst TEXT
    )""")
    try:
        c.execute("ALTER TABLE heat_accum_watch ADD COLUMN bpc_json TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE heat_accum_watch ADD COLUMN bpc_updated_cst TEXT")
    except sqlite3.OperationalError:
        pass
    c.execute("""CREATE TABLE IF NOT EXISTS ambush_watch (
        symbol TEXT NOT NULL,
        signal_type TEXT NOT NULL,
        coin TEXT,
        generated_date TEXT NOT NULL,
        last_seen_cst TEXT NOT NULL,
        d6h REAL,
        px_chg REAL,
        est_mcap REAL,
        ambush_total REAL,
        summary_line TEXT,
        PRIMARY KEY (symbol, signal_type)
    )""")
    try:
        c.execute("ALTER TABLE ambush_watch ADD COLUMN ambush_total REAL")
    except sqlite3.OperationalError:
        pass
    c.execute("""CREATE TABLE IF NOT EXISTS s2_funding_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recorded_at TEXT NOT NULL,
        symbol TEXT NOT NULL,
        coin TEXT,
        price REAL,
        price_chg_24h REAL,
        prev_fr REAL,
        current_fr REAL,
        oi_change_pct REAL,
        oi_segment_avgs_json TEXT,
        volume_usd REAL,
        est_mcap_usd REAL,
        has_spot INTEGER DEFAULT 0,
        square_posts INTEGER DEFAULT 0,
        square_views INTEGER DEFAULT 0
    )""")
    c.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_s2_recorded_symbol "
        "ON s2_funding_signals(recorded_at, symbol)"
    )
    conn.commit()
    _migrate_legacy_heat_accum_json(conn)
    return conn


def get_all_perp_symbols():
    """获取所有USDT永续合约"""
    info = api_get("/fapi/v1/exchangeInfo")
    if not info:
        return []
    return [s["symbol"] for s in info["symbols"]
            if s["quoteAsset"] == "USDT" 
            and s["contractType"] == "PERPETUAL"
            and s["status"] == "TRADING"]


def analyze_accumulation(symbol, klines):
    """分析单个币的收筹特征"""
    if len(klines) < MIN_DATA_DAYS:
        return None
    
    data = []
    for k in klines:
        data.append({
            "ts": k[0],
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "vol": float(k[7]),  # quote volume (USDT)
        })
    
    coin = symbol.replace("USDT", "")
    
    # === 排除稳定币和指数 ===
    EXCLUDE = {"USDC", "USDP", "TUSD", "FDUSD", "BTCDOM", "DEFI", "USDM"}
    if coin in EXCLUDE:
        return None
    
    # === 排除已经暴涨过+崩盘的币 ===
    # 最近7天vs之前的均价，如果已经涨>300%就跳过（来不及了）
    recent_7d = data[-7:]
    prior = data[:-7]
    if not prior:
        return None
    
    recent_avg_px = sum(d["close"] for d in recent_7d) / len(recent_7d)
    prior_avg_px = sum(d["close"] for d in prior) / len(prior)
    
    if prior_avg_px > 0 and ((recent_avg_px - prior_avg_px) / prior_avg_px) > 3.0:
        return None  # 已经涨了300%+，来不及了
    
    # === 寻找横盘区间 ===
    # 从最近往回找，找最长的横盘期（价格波动<MAX_RANGE_PCT%）
    # 关键：必须是真横盘（斜率接近零），阴跌不算横盘！
    best_sideways = 0
    best_range = 0
    best_low = 0
    best_high = 0
    best_avg_vol = 0
    best_slope_pct = 0
    
    # 用滑动窗口从60天到全部
    for window in range(MIN_SIDEWAYS_DAYS, len(prior) + 1):
        window_data = prior[-window:]
        lows = [d["low"] for d in window_data]
        highs = [d["high"] for d in window_data]
        
        w_low = min(lows)
        w_high = max(highs)
        
        if w_low <= 0:
            continue
        
        range_pct = ((w_high - w_low) / w_low) * 100
        
        if range_pct <= MAX_RANGE_PCT:
            avg_vol = sum(d["vol"] for d in window_data) / len(window_data)
            if avg_vol <= MAX_AVG_VOL_USD:
                # 线性回归算斜率：阴跌/暴涨不算横盘
                closes = [d["close"] for d in window_data]
                n = len(closes)
                x_mean = (n - 1) / 2.0
                y_mean = sum(closes) / n
                num = sum((i - x_mean) * (c - y_mean) for i, c in enumerate(closes))
                den = sum((i - x_mean) ** 2 for i in range(n))
                slope = num / den if den > 0 else 0
                # 累计变化占起始价的百分比
                slope_pct = (slope * n / closes[0] * 100) if closes[0] > 0 else 0
                
                # 斜率过滤：累计变化超过±20%不算横盘
                if abs(slope_pct) > 20:
                    continue
                
                if window > best_sideways:
                    best_sideways = window
                    best_range = range_pct
                    best_low = w_low
                    best_high = w_high
                    best_avg_vol = avg_vol
                    best_slope_pct = slope_pct
    
    if best_sideways < MIN_SIDEWAYS_DAYS:
        return None
    
    # === 计算收筹评分 ===
    # 横盘越久越好（庄家需要时间吸筹）
    days_score = min(best_sideways / 90, 1.0) * 25  # 90天满分25
    
    # 区间越窄越好（控盘紧）
    range_score = max(0, (1 - best_range / MAX_RANGE_PCT)) * 20  # 越窄越高，满分20
    
    # 成交量越低越好（死水一潭 = 筹码集中）
    vol_score = max(0, (1 - best_avg_vol / MAX_AVG_VOL_USD)) * 20  # 越低越高，满分20
    
    # 最近是否开始放量？（放量是启动信号）
    recent_vol = sum(d["vol"] for d in recent_7d) / len(recent_7d)
    vol_breakout = recent_vol / best_avg_vol if best_avg_vol > 0 else 0
    breakout_score = min(vol_breakout / VOL_BREAKOUT_MULT, 1.0) * 15  # 放量加分，满分15
    
    # 市值越低空间越大（核心！Patrick: 低市值=大空间）
    # 用当前价格*日均成交量/换手率来粗估市值排名
    # 实际市值在推送时用CoinGecko补充
    est_mcap = data[-1]["close"] * best_avg_vol * 30  # 粗略估算
    if est_mcap > 0 and est_mcap < 50_000_000:
        mcap_score = 20  # <$50M 满分
    elif est_mcap < 100_000_000:
        mcap_score = 15
    elif est_mcap < 200_000_000:
        mcap_score = 10
    elif est_mcap < 500_000_000:
        mcap_score = 5
    else:
        mcap_score = 0
    
    total_score = days_score + range_score + vol_score + breakout_score + mcap_score
    
    # 横盘质量加分：斜率越接近零越好（真横盘bonus，满分+5）
    flatness_bonus = max(0, (1 - abs(best_slope_pct) / 20)) * 5
    total_score += flatness_bonus
    
    # 状态判断
    if vol_breakout >= VOL_BREAKOUT_MULT:
        status = "🔥放量启动"
    elif vol_breakout >= 1.5:
        status = "⚡开始放量"
    else:
        status = "💤收筹中"
    
    return {
        "symbol": symbol,
        "coin": coin,
        "sideways_days": best_sideways,
        "range_pct": best_range,
        "slope_pct": best_slope_pct,
        "low_price": best_low,
        "high_price": best_high,
        "avg_vol": best_avg_vol,
        "current_price": data[-1]["close"],
        "recent_vol": recent_vol,
        "vol_breakout": vol_breakout,
        "score": total_score,
        "status": status,
        "data_days": len(data),
    }


def scan_accumulation_pool():
    """扫描全市场，找正在被收筹的币"""
    print("📊 扫描全市场收筹标的...")
    
    symbols = get_all_perp_symbols()
    print(f"  共 {len(symbols)} 个合约")
    
    results = []
    
    for i, sym in enumerate(symbols):
        klines = api_get("/fapi/v1/klines", {
            "symbol": sym, "interval": "1d", "limit": 180
        })
        
        if klines and isinstance(klines, list):
            r = analyze_accumulation(sym, klines)
            if r:
                results.append(r)
        
        if (i + 1) % 10 == 0:
            time.sleep(0.5)
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(symbols)}... 已发现{len(results)}个")
    
    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"  ✅ 发现 {len(results)} 个收筹标的")
    return results


def scan_oi_changes(watchlist_symbols):
    """对标的池内的币扫描OI异动"""
    print(f"📊 扫描OI异动（{len(watchlist_symbols)}个标的）...")
    
    alerts = []
    
    for sym in watchlist_symbols:
        # OI历史
        oi_hist = api_get("/futures/data/openInterestHist", {
            "symbol": sym, "period": "1h", "limit": 3
        })
        
        if not oi_hist or len(oi_hist) < 2:
            continue
        
        prev_oi = float(oi_hist[-2]["sumOpenInterestValue"])
        curr_oi = float(oi_hist[-1]["sumOpenInterestValue"])
        
        if prev_oi <= 0 or curr_oi < MIN_OI_USD:
            continue
        
        delta_pct = ((curr_oi - prev_oi) / prev_oi) * 100
        
        if abs(delta_pct) >= MIN_OI_DELTA_PCT:
            # 拿当前价格
            ticker = api_get("/fapi/v1/ticker/24hr", {"symbol": sym})
            if not ticker:
                continue
            
            price = float(ticker["lastPrice"])
            vol_24h = float(ticker["quoteVolume"])
            px_chg = float(ticker["priceChangePercent"])
            
            # 拿费率
            funding = api_get("/fapi/v1/fundingRate", {"symbol": sym, "limit": 1})
            fr = float(funding[0]["fundingRate"]) if funding else 0
            
            coin = sym.replace("USDT", "")
            
            alerts.append({
                "symbol": sym,
                "coin": coin,
                "price": price,
                "oi_usd": curr_oi,
                "oi_delta_pct": delta_pct,
                "oi_delta_usd": curr_oi - prev_oi,
                "vol_24h": vol_24h,
                "px_chg_pct": px_chg,
                "funding_rate": fr,
            })
        
        time.sleep(0.3)
    
    alerts.sort(key=lambda x: abs(x["oi_delta_pct"]), reverse=True)
    print(f"  ✅ 发现 {len(alerts)} 个OI异动")
    return alerts


def format_usd(v):
    if v >= 1e9: return f"${v/1e9:.1f}B"
    if v >= 1e6: return f"${v/1e6:.1f}M"
    if v >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


def build_pool_report(results, top_n=25):
    """生成收筹标的池报告"""
    if not results:
        return ""
    
    now = datetime.now(timezone(timedelta(hours=8)))
    
    lines = [
        f"🏦 **庄家收筹雷达** — 标的池更新",
        f"⏰ {now.strftime('%Y-%m-%d %H:%M')} CST",
        f"━━━━━━━━━━━━━━━━━━",
        f"扫描 {len(results)} 个合约，发现标的：",
        "",
    ]
    
    # 分组：放量启动 > 开始放量 > 收筹中
    firing = [r for r in results if "放量启动" in r["status"]]
    warming = [r for r in results if "开始放量" in r["status"]]
    sleeping = [r for r in results if "收筹中" in r["status"]]
    
    if firing:
        lines.append(f"🔥 **放量启动** ({len(firing)}个) — 最高优先级！")
        for r in firing[:10]:
            lines.append(
                f"  🔥 **{r['coin']}** | 分:{r['score']:.0f} | "
                f"横盘{r['sideways_days']}天 | 波动{r['range_pct']:.0f}% | "
                f"Vol放大{r['vol_breakout']:.1f}x"
            )
            lines.append(
                f"     ${r['current_price']:.6f} | "
                f"区间: ${r['low_price']:.6f}~${r['high_price']:.6f} | "
                f"日均Vol: {format_usd(r['avg_vol'])}"
            )
        lines.append("")
    
    if warming:
        lines.append(f"⚡ **开始放量** ({len(warming)}个) — 关注中")
        for r in warming[:10]:
            lines.append(
                f"  ⚡ {r['coin']} | 分:{r['score']:.0f} | "
                f"横盘{r['sideways_days']}天 | 波动{r['range_pct']:.0f}% | "
                f"Vol{r['vol_breakout']:.1f}x"
            )
        lines.append("")
    
    if sleeping:
        lines.append(f"💤 **收筹中** ({len(sleeping)}个) — 持续监控")
        for r in sleeping[:15]:
            lines.append(
                f"  💤 {r['coin']} | 分:{r['score']:.0f} | "
                f"横盘{r['sideways_days']}天 | 波动{r['range_pct']:.0f}% | "
                f"日均Vol {format_usd(r['avg_vol'])}"
            )
    
    return "\n".join(lines)


def build_oi_alert_report(alerts, watchlist_coins):
    """生成OI异动报告（只报标的池内的）"""
    if not alerts:
        return ""
    
    now = datetime.now(timezone(timedelta(hours=8)))
    
    # 区分：池内 vs 池外
    in_pool = [a for a in alerts if a["symbol"] in watchlist_coins]
    out_pool = [a for a in alerts if a["symbol"] not in watchlist_coins]
    
    lines = [
        f"📊 **OI异动扫描** [收筹池]",
        f"⏰ {now.strftime('%Y-%m-%d %H:%M')} CST",
        f"━━━━━━━━━━━━━━━━━━",
        "",
    ]
    
    if in_pool:
        lines.append(f"🎯 **收筹池内异动** ({len(in_pool)}个) ⚠️ 重点关注!")
        for a in in_pool[:10]:
            emoji = "🟢" if a["oi_delta_pct"] > 0 else "🔴"
            lines.append(
                f"  {emoji} **{a['coin']}** | OI: {a['oi_delta_pct']:+.1f}% "
                f"({format_usd(a['oi_usd'])}) | 价格: {a['px_chg_pct']:+.1f}%"
            )
            # 信号解读
            if a["oi_delta_pct"] > 0 and abs(a["px_chg_pct"]) < 3:
                lines.append(f"     ⚡ 暗流涌动！OI涨但价格平 = 庄家建仓中")
            elif a["oi_delta_pct"] > 0 and a["px_chg_pct"] > 3:
                lines.append(f"     🚀 放量拉升！OI+价格同涨 = 启动中")
        lines.append("")
    
    if out_pool:
        lines.append(f"📋 池外异动 ({len(out_pool)}个)")
        for a in out_pool[:8]:
            emoji = "🟢" if a["oi_delta_pct"] > 0 else "🔴"
            lines.append(
                f"  {emoji} {a['coin']} | OI: {a['oi_delta_pct']:+.1f}% | "
                f"价格: {a['px_chg_pct']:+.1f}%"
            )
    
    return "\n".join(lines)


def send_telegram(text):
    """发送TG消息"""
    if not TG_BOT_TOKEN:
        print("\n[TG] No token, stdout:\n")
        print(text)
        return
    
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    
    # 分段发送（TG限制4096字）
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > 3800:
            chunks.append(current)
            current = line
        else:
            current += "\n" + line if current else line
    if current:
        chunks.append(current)
    
    for chunk in chunks:
        try:
            resp = requests.post(url, json={
                "chat_id": TG_CHAT_ID,
                "text": chunk,
                "parse_mode": "Markdown"
            }, timeout=10)
            if resp.status_code == 200:
                print(f"[TG] Sent ✓ ({len(chunk)} chars)")
            else:
                # Markdown失败就用纯文本
                resp2 = requests.post(url, json={
                    "chat_id": TG_CHAT_ID,
                    "text": chunk.replace("*", "").replace("_", ""),
                }, timeout=10)
                print(f"[TG] Sent plain ({'✓' if resp2.status_code == 200 else '✗'})")
        except Exception as e:
            print(f"[TG] Error: {e}")
        time.sleep(0.5)


def save_watchlist(conn, results):
    """保存标的池到数据库"""
    c = conn.cursor()
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    
    for r in results:
        c.execute("""INSERT OR REPLACE INTO watchlist 
            (symbol, coin, added_date, sideways_days, range_pct, avg_vol, 
             low_price, high_price, current_price, score, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (r["symbol"], r["coin"], now, r["sideways_days"], r["range_pct"],
             r["avg_vol"], r["low_price"], r["high_price"], r["current_price"],
             r["score"], r["status"]))
    
    conn.commit()
    print(f"  💾 保存 {len(results)} 个标的到数据库")


def load_watchlist_symbols(conn):
    """从数据库加载标的池"""
    c = conn.cursor()
    c.execute("SELECT symbol FROM watchlist WHERE status != 'removed'")
    return [row[0] for row in c.fetchall()]


def scan_short_fuel():
    """策略2: 空头燃料 — 涨了+费率负+OI大 = 庄家拉盘爆空单"""
    print("📊 扫描空头燃料（费率为负+在涨的币）...")
    
    tickers = api_get("/fapi/v1/ticker/24hr")
    premiums = api_get("/fapi/v1/premiumIndex")
    
    if not tickers or not premiums:
        return [], []
    
    funding_map = {p["symbol"]: float(p["lastFundingRate"]) 
                   for p in premiums if p["symbol"].endswith("USDT")}
    
    fuel_targets = []     # 已在涨+费率负 = 正在squeeze
    squeeze_targets = []  # 费率极负+还没大涨 = 潜在squeeze
    
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        
        px_chg = float(t["priceChangePercent"])
        vol = float(t["quoteVolume"])
        fr = funding_map.get(sym, 0)
        coin = sym.replace("USDT", "")
        price = float(t["lastPrice"])
        
        item = {
            "coin": coin, "symbol": sym,
            "px_chg": px_chg, "funding": fr,
            "vol": vol, "price": price,
        }
        
        # 正在squeeze: 涨>5% + 费率负 + Vol>$5M
        if px_chg > 5 and fr < -0.0003 and vol > 5_000_000:
            item["fuel_score"] = abs(fr) * 10000 * px_chg
            fuel_targets.append(item)
        
        # 潜在squeeze: 费率很负 + 还没大涨(<10%) + Vol>$2M
        elif fr < -0.0005 and px_chg < 10 and vol > 2_000_000:
            item["fuel_score"] = abs(fr) * 10000
            squeeze_targets.append(item)
    
    fuel_targets.sort(key=lambda x: x["fuel_score"], reverse=True)
    squeeze_targets.sort(key=lambda x: x["fuel_score"], reverse=True)
    
    print(f"  ✅ 正在squeeze: {len(fuel_targets)}个, 潜在squeeze: {len(squeeze_targets)}个")
    return fuel_targets, squeeze_targets


def build_fuel_report(fuel_targets, squeeze_targets):
    """生成空头燃料报告"""
    if not fuel_targets and not squeeze_targets:
        return ""
    
    now = datetime.now(timezone(timedelta(hours=8)))
    lines = [
        f"🔥 **空头燃料扫描**",
        f"⏰ {now.strftime('%Y-%m-%d %H:%M')} CST",
        f"━━━━━━━━━━━━━━━━━━",
        f"逻辑：费率负=大量做空，庄家拉盘爆空单+收资金费",
        "",
    ]
    
    if fuel_targets:
        lines.append(f"🚀 **正在Squeeze** ({len(fuel_targets)}个) — 涨了+空头还在扛")
        for t in fuel_targets[:8]:
            fr_pct = t["funding"] * 100
            flag = "🎯极度!" if fr_pct < -0.1 else "⚠️"
            lines.append(
                f"  {flag} **{t['coin']}** | 涨{t['px_chg']:+.1f}% | "
                f"费率🧊{fr_pct:.4f}% | Vol {format_usd(t['vol'])}"
            )
        lines.append("")
    
    if squeeze_targets:
        lines.append(f"🎯 **潜在Squeeze** ({len(squeeze_targets)}个) — 费率极负+还没大涨")
        for t in squeeze_targets[:8]:
            fr_pct = t["funding"] * 100
            lines.append(
                f"  🧊 {t['coin']} | 价格{t['px_chg']:+.1f}% | "
                f"费率{fr_pct:.4f}% | Vol {format_usd(t['vol'])}"
            )
    
    return "\n".join(lines)

def run_oi_hourly_radar(conn: sqlite3.Connection, *, notify: bool = True) -> Dict[str, Any]:
    """综合扫描：OI + 费率 + 收筹。notify=False 时不推 Telegram，供 HTTP 刷新写快照。"""
    # === 综合扫描：OI + 费率 + 收筹 三维合一 ===
    watchlist = load_watchlist_symbols(conn)

    if not watchlist:
        print("⚠️ 标的池为空，先运行 pool 模式")
        return {"ok": False, "error": "watchlist_empty", "message": "标的池为空，先运行 pool 模式"}
    
    # 1. 拿全市场费率+行情
    tickers_raw = api_get("/fapi/v1/ticker/24hr")
    premiums_raw = api_get("/fapi/v1/premiumIndex")
    
    if not tickers_raw or not premiums_raw:
        print("❌ API失败")
        return {"ok": False, "error": "ticker_api", "message": "币安全市场行情 API 失败"}
    
    ticker_map = {}
    for t in tickers_raw:
        if t["symbol"].endswith("USDT"):
            ticker_map[t["symbol"]] = {
                "px_chg": float(t["priceChangePercent"]),
                "vol": float(t["quoteVolume"]),
                "price": float(t["lastPrice"]),
            }
    
    funding_map = {}
    for p in premiums_raw:
        if p["symbol"].endswith("USDT"):
            funding_map[p["symbol"]] = float(p["lastFundingRate"])
    
    # 1.5 拉真实流通市值（币安现货API，一次全量）
    mcap_map = {}  # coin名 -> marketCap
    try:
        import requests as _req
        _r = _req.get("https://www.binance.com/bapi/composite/v1/public/marketing/symbol/list", timeout=10)
        if _r.status_code == 200:
            for item in _r.json().get("data", []):
                name = item.get("name", "")
                mc = item.get("marketCap", 0)
                if name and mc:
                    mcap_map[name] = float(mc)
            print(f"✅ 拉到 {len(mcap_map)} 个币的真实市值")
    except Exception as e:
        print(f"⚠️ 市值API失败，走fallback: {e}")
    
    # 2. 拉热度数据（CoinGecko Trending + 成交量暴增）
    heat_map = {}  # coin名 -> heat_score (0-100)
    cg_trending = set()
    try:
        import requests as _req
        _r = _req.get("https://api.coingecko.com/api/v3/search/trending", timeout=10)
        if _r.status_code == 200:
            for item in _r.json().get("coins", []):
                sym = item["item"]["symbol"].upper()
                rank = item["item"].get("score", 99)
                cg_trending.add(sym)
                heat_map[sym] = heat_map.get(sym, 0) + max(50 - rank * 3, 10)  # top1=50分, top10=20分
            print(f"🔥 CoinGecko Trending: {len(cg_trending)}个币")
    except Exception as e:
        print(f"⚠️ CG Trending失败: {e}")
    
    # 成交量暴增检测（24hVol vs 5日均Vol）
    vol_surge_coins = set()
    for sym, tk in ticker_map.items():
        coin = sym.replace("USDT", "")
        vol_24h = tk["vol"]
        # 快速拿5天均量（用ticker的数据粗估，精确版在后面OI扫描时补充）
        # 这里先标记vol > $20M的为候选
        if vol_24h > 20_000_000:
            kl = api_get("/fapi/v1/klines", {"symbol": sym, "interval": "1d", "limit": 6})
            if kl and len(kl) >= 5:
                avg_5d = sum(float(k[7]) for k in kl[:-1]) / (len(kl)-1)
                if avg_5d > 0:
                    ratio = vol_24h / avg_5d
                    if ratio >= 2.5:  # 成交量放大2.5倍以上
                        vol_surge_coins.add(coin)
                        heat_map[coin] = heat_map.get(coin, 0) + min(ratio * 10, 50)  # 最高50分
                import time; time.sleep(0.05)
    
    print(f"📈 成交量暴增(≥2.5x): {len(vol_surge_coins)}个币")
    # 双重热度
    dual_heat = cg_trending & vol_surge_coins
    if dual_heat:
        for coin in dual_heat:
            heat_map[coin] = heat_map.get(coin, 0) + 20  # 双重信号bonus
        print(f"🔥🔥 双重热度: {dual_heat}")
    
    # 3. 从DB读收筹数据（含横盘高低点，供热度+收筹方案C多因子区间）
    c2 = conn.cursor()
    c2.execute(
        "SELECT symbol, score, sideways_days, range_pct, avg_vol, status, low_price, high_price FROM watchlist"
    )
    pool_map = {}
    for row in c2.fetchall():
        pool_map[row[0]] = {
            "pool_score": row[1],
            "sideways_days": row[2],
            "range_pct": row[3],
            "avg_vol": row[4],
            "status": row[5],
            "low_price": row[6],
            "high_price": row[7],
        }
    
    # 3. 扫OI（标的池中放量的 + Top100）
    scan_syms = set()
    for sym, pd in pool_map.items():
        if "放量" in pd.get("status", "") or "开始" in pd.get("status", ""):
            scan_syms.add(sym)
    top_by_vol = sorted(ticker_map.items(), key=lambda x: x[1]["vol"], reverse=True)[:100]
    for sym, _ in top_by_vol:
        scan_syms.add(sym)
    
    oi_map = {}
    for i, sym in enumerate(scan_syms):
        oi_hist = api_get("/futures/data/openInterestHist", {"symbol": sym, "period": "1h", "limit": 6})
        if oi_hist and len(oi_hist) >= 2:
            curr = float(oi_hist[-1]["sumOpenInterestValue"])
            prev_1h = float(oi_hist[-2]["sumOpenInterestValue"])
            prev_6h = float(oi_hist[0]["sumOpenInterestValue"])
            d1h = ((curr - prev_1h) / prev_1h * 100) if prev_1h > 0 else 0
            d6h = ((curr - prev_6h) / prev_6h * 100) if prev_6h > 0 else 0
            circ_supply = float(oi_hist[-1].get("CMCCirculatingSupply", 0))
            oi_map[sym] = {"oi_usd": curr, "d1h": d1h, "d6h": d6h, "circ_supply": circ_supply}
        if (i+1) % 10 == 0:
            import time; time.sleep(0.5)
    
    # 4. 三策略独立评分
    
    # 共用数据预处理
    all_syms = set(list(pool_map.keys()) + list(oi_map.keys()))
    coin_data = {}
    for sym in all_syms:
        tk = ticker_map.get(sym, {})
        if not tk: continue
        pool = pool_map.get(sym, {})
        oi = oi_map.get(sym, {})
        fr = funding_map.get(sym, 0)
        coin = sym.replace("USDT", "")
        
        d6h = oi.get("d6h", 0)
        fr_pct = fr * 100
        oi_usd = oi.get("oi_usd", 0)
        # 真实流通市值：优先现货API，fallback合约OI接口的CMC数据，最后粗估
        if coin in mcap_map:
            est_mcap = mcap_map[coin]
        else:
            circ_supply = oi.get("circ_supply", 0)
            price = tk.get("price", 0) if isinstance(tk, dict) else 0
            if circ_supply > 0 and price > 0:
                est_mcap = circ_supply * price
            else:
                est_mcap = max(tk["vol"] * 0.3, oi_usd * 2) if oi_usd > 0 else tk["vol"] * 0.3
        sw_days = pool.get("sideways_days", 0) if pool else 0
        pool_sc = pool.get("pool_score", 0) if pool else 0
        
        heat = heat_map.get(coin, 0)
        
        coin_data[sym] = {
            "coin": coin, "sym": sym,
            "px_chg": tk["px_chg"], "vol": tk["vol"],
            "price": tk["price"],
            "fr_pct": fr_pct, "d6h": d6h,
            "oi_usd": oi_usd, "est_mcap": est_mcap,
            "sw_days": sw_days, "pool_sc": pool_sc,
            "in_pool": bool(pool), "heat": heat,
            "in_cg": coin in cg_trending,
            "vol_surge": coin in vol_surge_coins,
            "low_price": float(pool.get("low_price") or 0) if pool else 0.0,
            "high_price": float(pool.get("high_price") or 0) if pool else 0.0,
        }
    
    # ═══════════════════════════════════════
    # 策略1: 追多 — 纯费率排名
    # ═══════════════════════════════════════
    chase = []
    for sym, d in coin_data.items():
        if d["px_chg"] > 3 and d["fr_pct"] < -0.005 and d["vol"] > 1_000_000:
            # 查费率趋势
            fr_hist = api_get("/fapi/v1/fundingRate", {"symbol": sym, "limit": 5})
            fr_rates = [float(f["fundingRate"]) * 100 for f in fr_hist] if fr_hist else [d["fr_pct"]]
            fr_prev = fr_rates[-2] if len(fr_rates) >= 2 else d["fr_pct"]
            fr_delta = d["fr_pct"] - fr_prev
            
            trend = "🔥加速" if fr_delta < -0.05 else "⬇️变负" if fr_delta < -0.01 else "➡️" if abs(fr_delta) < 0.01 else "⬆️回升"
            
            chase.append({**d, "fr_delta": fr_delta, "trend": trend,
                          "rates": " → ".join([f"{x:.3f}" for x in fr_rates[-3:]])})
            import time; time.sleep(0.2)
    
    # 纯按费率绝对值排序（越负越前）
    chase.sort(key=lambda x: x["fr_pct"])
    
    # ═══════════════════════════════════════
    # 策略2: 综合 — 各维度均衡(各25分)
    # ═══════════════════════════════════════
    combined = []
    for sym, d in coin_data.items():
        # 费率分(25) — 越负越好
        fr = d["fr_pct"]
        if fr < -0.5: f_sc = 25
        elif fr < -0.1: f_sc = 22
        elif fr < -0.05: f_sc = 18
        elif fr < -0.03: f_sc = 14
        elif fr < -0.01: f_sc = 10
        elif fr < 0: f_sc = 5
        else: f_sc = 0
        
        # 市值分(25) — 用真实流通市值
        mc = d["est_mcap"]
        if mc > 0 and mc < 50e6: m_sc = 25
        elif mc < 100e6: m_sc = 22
        elif mc < 200e6: m_sc = 20
        elif mc < 300e6: m_sc = 17
        elif mc < 500e6: m_sc = 12
        elif mc < 1e9: m_sc = 7
        else: m_sc = 0
        
        # 横盘分(25)
        sw = d["sw_days"]
        if sw >= 120: s_sc = 25
        elif sw >= 90: s_sc = 22
        elif sw >= 75: s_sc = 18
        elif sw >= 60: s_sc = 14
        elif sw >= 45: s_sc = 10
        else: s_sc = 0
        
        # OI分(25)
        abs6 = abs(d["d6h"])
        if abs6 >= 15: o_sc = 25
        elif abs6 >= 8: o_sc = 22
        elif abs6 >= 5: o_sc = 18
        elif abs6 >= 3: o_sc = 14
        elif abs6 >= 2: o_sc = 10
        else: o_sc = 0
        
        total = f_sc + m_sc + s_sc + o_sc
        if total < 25: continue
        
        combined.append({**d, "total": total,
                        "f_sc": f_sc, "m_sc": m_sc, "s_sc": s_sc, "o_sc": o_sc})
    
    combined.sort(key=lambda x: x["total"], reverse=True)
    
    # ═══════════════════════════════════════
    # 策略3: 埋伏 — 市值>OI>横盘>费率
    # ═══════════════════════════════════════
    ambush = []
    for sym, d in coin_data.items():
        if not d["in_pool"]: continue  # 必须在收筹池
        if d["px_chg"] > 50: continue  # 已经暴涨的排除
        
        # 1.市值(35分) — 核心！越低越好（真实流通市值）
        mc = d["est_mcap"]
        if mc > 0 and mc < 50e6: m_sc = 35
        elif mc < 100e6: m_sc = 32
        elif mc < 150e6: m_sc = 28
        elif mc < 200e6: m_sc = 25
        elif mc < 300e6: m_sc = 20
        elif mc < 500e6: m_sc = 12
        elif mc < 1e9: m_sc = 5
        else: m_sc = 0
        
        # 2.OI异动(30分) — OI涨+市值低=极好
        abs6 = abs(d["d6h"])
        if abs6 >= 10: o_sc = 30
        elif abs6 >= 5: o_sc = 25
        elif abs6 >= 3: o_sc = 20
        elif abs6 >= 2: o_sc = 14
        elif abs6 >= 1: o_sc = 8
        else: o_sc = 0
        # 暗流加分：OI涨但价格平
        if d["d6h"] > 2 and abs(d["px_chg"]) < 5:
            o_sc = min(o_sc + 5, 30)
        
        # 3.横盘(20分)
        sw = d["sw_days"]
        if sw >= 120: s_sc = 20
        elif sw >= 90: s_sc = 17
        elif sw >= 75: s_sc = 14
        elif sw >= 60: s_sc = 10
        elif sw >= 45: s_sc = 6
        else: s_sc = 0
        
        # 4.负费率(15分) — 有负费率是bonus
        fr = d["fr_pct"]
        if fr < -0.1: f_sc = 15
        elif fr < -0.05: f_sc = 12
        elif fr < -0.03: f_sc = 9
        elif fr < -0.01: f_sc = 6
        elif fr < 0: f_sc = 3
        else: f_sc = 0
        
        total = m_sc + o_sc + s_sc + f_sc
        if total < 20: continue
        
        ambush.append({**d, "total": total,
                      "m_sc": m_sc, "o_sc": o_sc, "s_sc": s_sc, "f_sc": f_sc})
    
    ambush.sort(key=lambda x: x["total"], reverse=True)
    
    # ═══════════════════════════════════════
    # 5. 生成推送 + 值得关注提醒
    # ═══════════════════════════════════════
    def mcap_str(v):
        if v >= 1e6: return f"${v/1e6:.0f}M"
        if v >= 1e3: return f"${v/1e3:.0f}K"
        return f"${v:.0f}"
    
    now = datetime.now(timezone(timedelta(hours=8)))
    lines = [
        f"🏦 **庄家雷达** 三策略+热度",
        f"⏰ {now.strftime('%Y-%m-%d %H:%M')} CST",
    ]
    
    # 表0: 热度榜（最重要，放最前面）
    hot_coins = sorted(
        [d for d in coin_data.values() if d["heat"] > 0],
        key=lambda x: x["heat"], reverse=True
    )
    if hot_coins:
        lines.append(f"\n🔥 **热度榜** (CG趋势+成交量暴增)")
        for s in hot_coins[:8]:
            tags = []
            if s["in_cg"]: tags.append("🌐CG热搜")
            if s["vol_surge"]: tags.append("📈放量")
            oi_tag = f"OI{s['d6h']:+.0f}%" if abs(s["d6h"]) >= 3 else ""
            if oi_tag: tags.append(f"⚡{oi_tag}")
            if s["in_pool"]: tags.append(f"💤池{s['sw_days']}天")
            fr_tag = f"🧊{s['fr_pct']:.2f}%" if s["fr_pct"] < -0.03 else ""
            if fr_tag: tags.append(fr_tag)
            lines.append(
                f"  {s['coin']:<8} ~{mcap_str(s['est_mcap'])} 涨{s['px_chg']:+.0f}% | {' '.join(tags)}"
            )
    
    # 表1: 追多
    lines.append(f"\n🔥 **追多** (按费率排名)")
    if chase:
        for s in chase[:8]:
            lines.append(
                f"  {s['coin']:<7} 费率{s['fr_pct']:+.3f}% {s['trend']}"
                f" | 涨{s['px_chg']:+.0f}% | ~{mcap_str(s['est_mcap'])}"
            )
    else:
        lines.append("  暂无（需涨>3%+费率负）")
    
    # 表2: 综合
    lines.append(f"\n📊 **综合** (费率+市值+横盘+OI 各25)")
    for s in combined[:8]:
        dims = []
        if s["f_sc"] >= 10: dims.append(f"🧊{s['fr_pct']:.2f}%")
        if s["m_sc"] >= 12: dims.append(f"💎{mcap_str(s['est_mcap'])}")
        if s["s_sc"] >= 10: dims.append(f"💤{s['sw_days']}天")
        if s["o_sc"] >= 10: dims.append(f"⚡OI{s['d6h']:+.0f}%")
        lines.append(
            f"  {s['coin']:<7} {s['total']}分 | {' '.join(dims)}"
        )
    
    # 表3: 埋伏
    lines.append(f"\n🎯 **埋伏** (市值35+OI30+横盘20+费率15)")
    for s in ambush[:8]:
        tags = [f"~{mcap_str(s['est_mcap'])}"]
        if abs(s["d6h"]) >= 2: tags.append(f"OI{s['d6h']:+.0f}%")
        if s["d6h"] > 2 and abs(s["px_chg"]) < 5: tags.append("🎯暗流")
        if s["sw_days"] >= 45: tags.append(f"横盘{s['sw_days']}天")
        if s["fr_pct"] < -0.01: tags.append(f"费率{s['fr_pct']:.2f}%")
        lines.append(
            f"  {s['coin']:<7} {s['total']}分 | {' '.join(tags)}"
        )
    
    # ═══ 值得关注提醒 ═══
    highlights = []
    hot_pool_signals: List[Dict[str, Any]] = []
    
    # 热度+收筹池重叠 = 最强信号（放最前面！热度领先OI）— 取前3名。
    hot_pool = [d for d in coin_data.values() if d["heat"] > 0 and d["in_pool"]]
    for s in sorted(hot_pool, key=lambda x: x["heat"], reverse=True)[:3]:
        tags = []
        if s["in_cg"]: tags.append("CG热搜")
        if s["vol_surge"]: tags.append("放量")
        base = f"🔥💤 {s['coin']} 热度({'+'.join(tags)})+收筹{s['sw_days']}天=OI将涨"
        hot_pool_signals.append({
            "coin": s["coin"],
            "symbol": s["sym"],
            "heat": s["heat"],
            "tags": list(tags),
            "sideways_days": s["sw_days"],
            "low_price": s["low_price"],
            "high_price": s["high_price"],
            "price": s["price"],
        })
        highlights.append(base)
    
    # 热度+OI已经在涨 = 正在发生
    hot_oi = [d for d in coin_data.values() if d["heat"] > 0 and d["d6h"] > 5]
    for s in sorted(hot_oi, key=lambda x: x["d6h"], reverse=True)[:2]:
        if s["coin"] not in " ".join(highlights):
            highlights.append(f"🔥⚡ {s['coin']} 热度+OI{s['d6h']:+.0f}%双涨！")
    
    # 追多里费率加速恶化的前2
    chase_fire = [s for s in chase[:5] if "加速" in s.get("trend", "")]
    for s in chase_fire[:2]:
        highlights.append(f"🔥 {s['coin']} 费率{s['fr_pct']:.3f}%加速恶化，空头涌入中")
    
    # 三个表都出现的币
    chase_coins = set(s["coin"] for s in chase[:10])
    combined_coins = set(s["coin"] for s in combined[:10])
    ambush_coins = set(s["coin"] for s in ambush[:10])
    
    # 追多+综合都出现
    overlap_2 = chase_coins & combined_coins
    if overlap_2:
        for c in list(overlap_2)[:2]:
            highlights.append(f"⭐ {c} 追多+综合双榜上榜")
    
    # 埋伏里OI暗流 — 全埋伏榜（total 从高到低）中命中条件的取分数最高的前 N 名
    ambush_dark = [
        s
        for s in ambush
        if s["d6h"] > 2 and abs(s["px_chg"]) < 5
    ][:AMBUSH_WATCH_TOP_N]
    for s in ambush_dark:
        highlights.append(f"🎯 {s['coin']} 暗流！OI{s['d6h']:+.0f}%但价格没动，市值仅{mcap_str(s['est_mcap'])}")
    
    # 埋伏里市值极低+OI异动 — 同上，全榜按 total 序取前 N 名
    ambush_gem = [
        s
        for s in ambush
        if s["est_mcap"] < 100e6 and abs(s["d6h"]) >= 3
    ][:AMBUSH_WATCH_TOP_N]
    for s in ambush_gem:
        if s["coin"] not in [h.split(" ")[1] for h in highlights]:
            highlights.append(f"💎 {s['coin']} 低市值{mcap_str(s['est_mcap'])}+OI{s['d6h']:+.0f}%，埋伏首选")

    if highlights:
        lines.append(f"\n💡 **值得关注**")
        for h in highlights[:15]:
            lines.append(f"  {h}")
    
    # 图例说明
    lines.append(f"\n📖 **图例**")
    lines.append("  🔥热度=CG热搜+成交量暴增(OI领先指标)")
    lines.append("  费率负=空头燃料 | 💎市值 | 💤横盘(收筹)")
    lines.append("  🔥💤热度+收筹=最强预判 | 🔥⚡热度+OI=正在发生")
    
    report = "\n".join(lines)
    ambush_watchlist = merge_and_persist_ambush_watchlist(
        conn, ambush_dark, ambush_gem, now, mcap_str
    )
    heat_accum_watchlist = merge_and_persist_heat_accum_watchlist(conn, hot_pool_signals, now)
    payload = {
        "ok": True,
        "generated_at_cst": now.strftime("%Y-%m-%d %H:%M") + " CST",
        "highlights": highlights[:15],
        "hot_pool_signals": hot_pool_signals,
        "heat_accum_watchlist": heat_accum_watchlist,
        "ambush_watchlist": ambush_watchlist,
        "report_markdown": report,
        "hot_coins": hot_coins[:16],
        "chase": chase[:16],
        "combined": combined[:16],
        "ambush": ambush[:16],
        "coin_data": list(coin_data.values()),
    }
    _persist_oi_radar_snapshot(payload)
    if notify:
        send_telegram(report)
    return payload


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    
    print(f"🏦 庄家收筹雷达 v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   模式: {mode}\n")
    
    conn = init_db()
    
    if mode in ("full", "pool"):
        # === 模块A: 更新收筹标的池 ===
        results = scan_accumulation_pool()
        
        if results:
            save_watchlist(conn, results)
            report = build_pool_report(results)
            if report:
                send_telegram(report)
    
    if mode in ("full", "oi"):
        run_oi_hourly_radar(conn, notify=True)
    
    conn.close()
    print("\n✅ 完成")


if __name__ == "__main__":
    main()

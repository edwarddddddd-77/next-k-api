#!/usr/bin/env python3
"""
每日（或可定时）执行：近 N 天 walk-forward → 触轨池筛选 → 写入 accumulation.db。

表：
- **zct_vwap_touch_pool**：每轮先清空再写入当前入选标的（symbol PRIMARY KEY）。
- **zct_vwap_touch_pool_runs**：每轮追加一条审计（含完整 pool JSON）。

运行：
  python zct_vwap_asset_pool_daily_job.py --once --zct-default-22
  python zct_vwap_asset_pool_daily_job.py --daemon --tz Asia/Shanghai --cron-hour 8

环境变量：`DATA_DIR`、`ZCT_TOUCH_POOL_TABLE`、`ZCT_TOUCH_POOL_RUNS_TABLE`、
`ZCT_TOUCH_POOL_CRON_HOUR`、`ZCT_TOUCH_POOL_TZ`、`ZCT_TOUCH_POOL_SLEEP_SYMBOLS`（默认 0.25）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from accumulation_radar import DB_PATH, init_db
from zct_vwap_asset_pool import (
    _default_symbol_list,
    run_asset_pool_scan,
    zct_default_22_symbols,
)

import zct_vwap_signal_scanner as z

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
except ImportError:  # pragma: no cover
    BlockingScheduler = None  # type: ignore
    CronTrigger = None  # type: ignore

logger = logging.getLogger(__name__)

_DEFAULT_POOL = "zct_vwap_touch_pool"
_DEFAULT_RUNS = "zct_vwap_touch_pool_runs"


def _pool_table() -> str:
    t = os.getenv("ZCT_TOUCH_POOL_TABLE", _DEFAULT_POOL).strip()
    return t if all(c.isalnum() or c == "_" for c in t) else _DEFAULT_POOL


def _runs_table() -> str:
    t = os.getenv("ZCT_TOUCH_POOL_RUNS_TABLE", _DEFAULT_RUNS).strip()
    return t if all(c.isalnum() or c == "_" for c in t) else _DEFAULT_RUNS


def ensure_schema(conn: sqlite3.Connection) -> None:
    pt, rt = _pool_table(), _runs_table()
    c = conn.cursor()
    c.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {pt} (
            symbol TEXT PRIMARY KEY,
            updated_at_ms INTEGER NOT NULL,
            days REAL NOT NULL,
            signal_interval TEXT NOT NULL,
            win INTEGER NOT NULL,
            loss INTEGER NOT NULL,
            win_plus_loss INTEGER NOT NULL,
            win_rate_touch_sl_tp REAL,
            expired INTEGER NOT NULL,
            unresolved INTEGER NOT NULL,
            user_start_open_ms INTEGER,
            hist_end_open_ms INTEGER,
            trades_emitted INTEGER,
            criteria_json TEXT NOT NULL
        )
        """
    )
    c.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {rt} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at_ms INTEGER NOT NULL,
            matched_count INTEGER NOT NULL,
            scanned_count INTEGER NOT NULL,
            criteria_json TEXT NOT NULL,
            pool_json TEXT NOT NULL
        )
        """
    )
    conn.commit()


def write_db(conn: sqlite3.Connection, out: Dict[str, Any]) -> int:
    pt, rt = _pool_table(), _runs_table()
    crit = json.dumps(out.get("criteria") or {}, ensure_ascii=False)
    run_ms = int(out.get("generated_at_ms") or int(time.time() * 1000))
    matched: List[Dict[str, Any]] = list(out.get("matched") or [])
    meta = out.get("backtest_meta") or {}
    scanned = len(out.get("symbols_scanned") or [])
    days = float((out.get("criteria") or {}).get("days") or 0)
    sig_iv = str((out.get("criteria") or {}).get("signal_interval") or "1m")

    cur = conn.cursor()
    cur.execute(f"DELETE FROM {pt}")
    rows: List[tuple] = []
    for m in matched:
        sym = str(m.get("symbol", "")).strip().upper()
        if not sym:
            continue
        rows.append(
            (
                sym,
                run_ms,
                days,
                sig_iv,
                int(m.get("win", 0) or 0),
                int(m.get("loss", 0) or 0),
                int(m.get("win_plus_loss", 0) or 0),
                m.get("win_rate_touch_sl_tp"),
                int(m.get("expired", 0) or 0),
                int(m.get("unresolved", 0) or 0),
                meta.get("user_start_open_ms"),
                meta.get("hist_end_open_ms"),
                meta.get("trades_emitted"),
                crit,
            )
        )
    if rows:
        cur.executemany(
            f"""
            INSERT INTO {pt} (
                symbol, updated_at_ms, days, signal_interval,
                win, loss, win_plus_loss, win_rate_touch_sl_tp,
                expired, unresolved,
                user_start_open_ms, hist_end_open_ms, trades_emitted,
                criteria_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
    cur.execute(
        f"""
        INSERT INTO {rt} (run_at_ms, matched_count, scanned_count, criteria_json, pool_json)
        VALUES (?,?,?,?,?)
        """,
        (
            run_ms,
            len(rows),
            scanned,
            crit,
            json.dumps(out, ensure_ascii=False),
        ),
    )
    conn.commit()
    return len(rows)


def resolve_symbols(ns: argparse.Namespace) -> List[str]:
    if ns.zct_default_22:
        return zct_default_22_symbols()
    if ns.use_env_symbols:
        return z._symbols_from_env()
    if ns.symbols.strip():
        return [x.strip().upper() for x in ns.symbols.split(",") if x.strip()]
    return _default_symbol_list()


def run_once(ns: argparse.Namespace) -> Dict[str, Any]:
    syms = resolve_symbols(ns)
    if not syms:
        raise SystemExit("[touch_pool] no symbols")
    if ns.ignore_db_cooldown and ns.use_db_cooldown:
        raise SystemExit("--ignore-db-cooldown 与 --use-db-cooldown 互斥")
    ignore = not bool(ns.use_db_cooldown)

    if ns.sleep_between_symbols is None:
        try:
            ns.sleep_between_symbols = float(
                os.getenv("ZCT_TOUCH_POOL_SLEEP_SYMBOLS", "0.25").strip() or "0.25"
            )
        except ValueError:
            ns.sleep_between_symbols = 0.25

    out, _ = run_asset_pool_scan(
        days=float(ns.days),
        symbols=syms,
        ignore_db_cooldown=ignore,
        sleep_between_symbols=max(0.0, float(ns.sleep_between_symbols)),
        signal_interval=str(ns.signal_interval),
        min_touch_trades=int(ns.min_touch_trades),
        strict_greater_touch=bool(ns.strict_greater_touch),
        min_touch_win_rate=float(ns.min_touch_win_rate),
        strict_greater_rate=bool(ns.strict_greater_rate),
        quiet=True,
    )

    conn = init_db()
    try:
        ensure_schema(conn)
        n = write_db(conn, out)
        logger.info(
            "touch_pool db=%s table=%s rows=%d symbols=%s",
            DB_PATH,
            _pool_table(),
            n,
            out.get("matched_symbols"),
        )
    finally:
        conn.close()

    if not ns.no_json_out:
        p = Path(
            ns.json_out.strip()
            or str(Path(__file__).resolve().parent / "zct_vwap_asset_pool.json")
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[touch_pool] json -> {p.resolve()}", flush=True)

    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ZCT 触轨资产池每日入库")
    ap.add_argument("--once", action="store_true", help="跑一轮退出（给计划任务用）")
    ap.add_argument("--daemon", action="store_true", help="APScheduler 常驻按 Cron 跑")
    ap.add_argument("--cron-hour", type=int, default=None, metavar="H")
    ap.add_argument("--cron-minute", type=int, default=0, metavar="M")
    ap.add_argument("--tz", type=str, default="", help="默认同环境 ZCT_TOUCH_POOL_TZ 或 UTC")
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--zct-default-22", action="store_true")
    ap.add_argument("--use-env-symbols", action="store_true")
    ap.add_argument("--ignore-db-cooldown", action="store_true")
    ap.add_argument("--use-db-cooldown", action="store_true")
    ap.add_argument("--min-touch-trades", type=int, default=130)
    ap.add_argument("--strict-greater-touch", action="store_true")
    ap.add_argument("--min-touch-win-rate", type=float, default=0.8)
    ap.add_argument("--strict-greater-rate", action="store_true")
    ap.add_argument("--signal-interval", type=str, default="1m", choices=["1m", "5m"])
    ap.add_argument("--sleep-between-symbols", type=float, default=None)
    ap.add_argument("--json-out", type=str, default="")
    ap.add_argument("--no-json-out", action="store_true")
    return ap


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    ap = build_parser()
    ns = ap.parse_args()

    if ns.once and ns.daemon:
        ap.error("--once 与 --daemon 互斥")
    if not ns.once and not ns.daemon:
        ns.once = True

    if ns.daemon:
        if BlockingScheduler is None:
            print("需要: pip install apscheduler", file=sys.stderr)
            sys.exit(1)
        h = ns.cron_hour
        if h is None:
            try:
                h = int(os.getenv("ZCT_TOUCH_POOL_CRON_HOUR", "2").strip() or "2")
            except ValueError:
                h = 2
        h = max(0, min(23, h))
        tz_name = (ns.tz or os.getenv("ZCT_TOUCH_POOL_TZ", "UTC")).strip() or "UTC"

        def job() -> None:
            try:
                run_once(ns)
            except Exception:
                logger.exception("touch_pool job failed")

        try:
            from zoneinfo import ZoneInfo

            tzobj: Any = ZoneInfo(tz_name)
        except Exception:
            try:
                import pytz

                tzobj = pytz.timezone(tz_name)
            except Exception:
                logger.warning("bad tz %r, use UTC", tz_name)
                from zoneinfo import ZoneInfo

                tzobj = ZoneInfo("UTC")

        sched = BlockingScheduler(timezone=tzobj)
        sched.add_job(
            job,
            CronTrigger(hour=h, minute=int(ns.cron_minute)),
            id="zct_touch_pool",
            replace_existing=True,
        )
        logger.info("touch_pool daemon tz=%s %02d:%02d", tz_name, h, int(ns.cron_minute))
        sched.start()
        return

    run_once(ns)
    print("[touch_pool] done", flush=True)


if __name__ == "__main__":
    main()

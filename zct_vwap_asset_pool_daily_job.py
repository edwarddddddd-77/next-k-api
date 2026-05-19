#!/usr/bin/env python3
"""
每日（或可定时）执行：近 N 天 walk-forward（默认 1.5 天≈36h）→ 触轨池筛选 → 写入 accumulation.db。

表：
- **zct_vwap_touch_pool**：每轮在单事务内 **先 DELETE 全表** 再写入当前入选标的（symbol PRIMARY KEY）。
- **zct_vwap_touch_pool_runs**：每轮追加一条审计（含完整 pool JSON）。

运行：
  python zct_vwap_asset_pool_daily_job.py --once --hot-oi-plus-default-22
  python zct_vwap_asset_pool_daily_job.py --once --zct-default-22
  python zct_vwap_asset_pool_daily_job.py --daemon
  （`--daemon` 默认：**Asia/Shanghai 每天 08:05**；可用 `--tz` / `--cron-hour` / `--cron-minute` 或环境变量覆盖）

环境变量：`DATA_DIR`、`ZCT_TOUCH_POOL_TABLE`、`ZCT_TOUCH_POOL_RUNS_TABLE`、
`ZCT_TOUCH_POOL_CRON_HOUR`（默认 8）、`ZCT_TOUCH_POOL_CRON_MINUTE`（默认 5）、
`ZCT_TOUCH_POOL_TZ`（默认 Asia/Shanghai）、`ZCT_TOUCH_POOL_SLEEP_SYMBOLS`（默认 0.25）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from accumulation_radar import DB_PATH, init_db
from zct_vwap_asset_pool import (
    _default_symbol_list,
    notify_touch_pool_empty_if_needed,
    run_asset_pool_scan,
    touch_pool_symbols_hot_oi_plus_default_22,
    touch_pool_symbols_worth_watch_plus_default,
    zct_default_22_symbols,
)
from zct_vwap_touch_pool_db import (
    touch_pool_ensure_schema,
    touch_pool_physical_table_names,
    touch_pool_write_db,
)

import zct_vwap_signal_scanner as z

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
except ImportError:  # pragma: no cover
    BlockingScheduler = None  # type: ignore
    CronTrigger = None  # type: ignore

logger = logging.getLogger(__name__)


def resolve_symbols(ns: argparse.Namespace) -> List[str]:
    if ns.worth_watch_plus_default_22:
        return touch_pool_symbols_worth_watch_plus_default()
    if ns.hot_oi_plus_default_22:
        return touch_pool_symbols_hot_oi_plus_default_22()
    if ns.zct_default_22:
        return zct_default_22_symbols()
    if ns.use_env_symbols:
        return z._symbols_from_env()
    if ns.symbols.strip():
        return [x.strip().upper() for x in ns.symbols.split(",") if x.strip()]
    return touch_pool_symbols_worth_watch_plus_default()


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

    if ns.worth_watch_plus_default_22:
        sym_src = "worth_watch_plus_default_22"
    elif ns.hot_oi_plus_default_22:
        sym_src = "hot_oi_plus_default_22"
    elif ns.zct_default_22:
        sym_src = "zct_default_22"
    elif ns.use_env_symbols:
        sym_src = "env_symbols"
    elif ns.symbols.strip():
        sym_src = "cli_symbols"
    else:
        sym_src = "worth_watch_plus_default_22"

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
        min_total_trades=int(ns.min_total_trades),
        max_expired_ratio=float(ns.max_expired_ratio),
        min_win_loss_abs=int(ns.min_win_loss_abs),
        min_touch_share=float(ns.min_touch_share),
        min_profit_factor=float(ns.min_profit_factor),
        max_consecutive_losses_at_end=int(ns.max_consecutive_losses_at_end),
        quiet=True,
        symbols_source=sym_src,
    )

    conn = init_db()
    try:
        touch_pool_ensure_schema(conn)
        n = touch_pool_write_db(conn, out)
        notify_touch_pool_empty_if_needed(n, criteria=out.get("criteria") or {})
        pool_tbl, _ = touch_pool_physical_table_names()
        logger.info(
            "touch_pool db=%s table=%s rows=%d symbols=%s",
            DB_PATH,
            pool_tbl,
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
    ap.add_argument(
        "--cron-minute",
        type=int,
        default=None,
        metavar="M",
        help="默认同环境 ZCT_TOUCH_POOL_CRON_MINUTE 或 30",
    )
    ap.add_argument(
        "--tz",
        type=str,
        default="",
        help="默认同环境 ZCT_TOUCH_POOL_TZ 或 Asia/Shanghai",
    )
    ap.add_argument("--days", type=float, default=1.0, help="主筛默认 1.0=严格 24h")
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--zct-default-22", action="store_true")
    ap.add_argument(
        "--hot-oi-plus-default-22",
        action="store_true",
        help="worth_watch_hot_oi ∪ 内置默认永续（旧候选，较窄）",
    )
    ap.add_argument(
        "--worth-watch-plus-default-22",
        action="store_true",
        help="值得关注七类 worth_watch_* ∪ 内置默认永续（稳档默认；CLI 名 default22 为历史遗留）",
    )
    ap.add_argument("--use-env-symbols", action="store_true")
    ap.add_argument("--ignore-db-cooldown", action="store_true")
    ap.add_argument("--use-db-cooldown", action="store_true")
    ap.add_argument("--min-touch-trades", type=int, default=1)
    ap.add_argument("--strict-greater-touch", action="store_true")
    ap.add_argument("--min-touch-win-rate", type=float, default=0.72)
    ap.add_argument("--strict-greater-rate", action="store_true")
    ap.add_argument(
        "--min-total-trades",
        type=int,
        default=20,
        help="walk 内 n_trades 须 ≥ 该值（主筛默认 20）",
    )
    ap.add_argument(
        "--max-expired-ratio",
        type=float,
        default=1.0,
        help="expired/n_trades 须 < 该值（主筛默认 1=关闭）",
    )
    ap.add_argument(
        "--min-win-loss-abs",
        type=int,
        default=0,
        help="win+loss 须 ≥ 该值（0=关闭）",
    )
    ap.add_argument(
        "--min-touch-share",
        type=float,
        default=0.0,
        help="(win+loss)/n_trades 须 ≥ 该值（0=关闭）",
    )
    ap.add_argument("--min-profit-factor", type=float, default=1.25)
    ap.add_argument(
        "--max-consecutive-losses-at-end",
        type=int,
        default=2,
        help="周期末连续亏损上限（默认 2 即 <3）",
    )
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

    mode_n = sum(
        1
        for x in (
            ns.worth_watch_plus_default_22,
            ns.hot_oi_plus_default_22,
            ns.zct_default_22,
            ns.use_env_symbols,
            bool(ns.symbols.strip()),
        )
        if x
    )
    if mode_n > 1:
        ap.error(
            "标的来源请只选一种：--worth-watch-plus-default-22 / --hot-oi-plus-default-22 / "
            "--zct-default-22 / --use-env-symbols / --symbols"
        )

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
                h = int(os.getenv("ZCT_TOUCH_POOL_CRON_HOUR", "8").strip() or "8")
            except ValueError:
                h = 8
        h = max(0, min(23, h))

        m = ns.cron_minute
        if m is None:
            try:
                m = int(os.getenv("ZCT_TOUCH_POOL_CRON_MINUTE", "5").strip() or "5")
            except ValueError:
                m = 0
        m = max(0, min(59, m))

        tz_name = (ns.tz or os.getenv("ZCT_TOUCH_POOL_TZ", "Asia/Shanghai")).strip() or "Asia/Shanghai"

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
            CronTrigger(hour=h, minute=m),
            id="zct_touch_pool",
            replace_existing=True,
        )
        logger.info("touch_pool daemon tz=%s %02d:%02d", tz_name, h, m)
        sched.start()
        return

    run_once(ns)
    print("[touch_pool] done", flush=True)


if __name__ == "__main__":
    main()

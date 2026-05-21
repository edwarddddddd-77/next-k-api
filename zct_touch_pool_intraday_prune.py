#!/usr/bin/env python3
"""
已废弃的「池内滚动清洗」入口。

逻辑已合并为每 4h 全宇宙 walk（6h 窗口 + 统一硬阈值），见 `zct_vwap_asset_pool_daily_job.py`。
本脚本保留为兼容 shim：直接执行与 daily job 相同的 `--once` 全量刷新。
"""

from __future__ import annotations

import logging
import sys

from zct_vwap_asset_pool_daily_job import build_parser, run_once

logger = logging.getLogger(__name__)


def rolling_clean_enabled() -> bool:
    """滚动清洗已关闭；调度器应只注册 4h 全量任务。"""
    return False


def run_rolling_pool_clean(**_kwargs):  # type: ignore[no-untyped-def]
    """兼容旧 import：改为全量 4h 刷新；返回值同 run_once（pool payload），非旧版 removed/kept。"""
    ap = build_parser()
    ns = ap.parse_args(["--once", "--worth-watch-plus-default-22"])
    return run_once(ns)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.warning(
        "zct_touch_pool_intraday_prune 已合并入 4h 全量刷新，本入口将执行 --once 全宇宙扫描"
    )
    ap = build_parser()
    ns = ap.parse_args(["--once", "--worth-watch-plus-default-22"])
    run_once(ns)
    print("[touch_pool] 4h full refresh (legacy prune shim) done", flush=True)


if __name__ == "__main__":
    main()

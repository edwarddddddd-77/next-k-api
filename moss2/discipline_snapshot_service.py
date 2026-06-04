"""周度 discipline 快照：对启用 Profile 跑短窗回测并写入 moss2_discipline_snapshots。"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict

from moss2 import config as cfg
from moss2.backtest_service import run_profile_backtest
from moss2.db import insert_discipline_snapshot, list_profiles
from moss2.versioning import effective_version

logger = logging.getLogger(__name__)


def run_weekly_discipline_snapshots(conn: sqlite3.Connection) -> Dict[str, Any]:
    profiles = list_profiles(conn, enabled_only=True)
    saved = 0
    skipped = 0
    errors: list = []

    for prof in profiles:
        pid = int(prof["id"])
        if not cfg.is_ops_variant(prof.get("variant")):
            skipped += 1
            continue
        try:
            out = run_profile_backtest(
                prof, capital=prof.get("virtual_equity_usdt"), limit_bars=4500
            )
        except Exception as e:
            logger.warning("[moss2] discipline snapshot %s: %s", pid, e)
            errors.append({"profile_id": pid, "error": str(e)})
            skipped += 1
            continue
        discipline = out.get("discipline")
        if not discipline:
            skipped += 1
            continue
        insert_discipline_snapshot(
            conn,
            profile_id=pid,
            symbol=str(prof["symbol"]),
            variant=str(prof.get("variant") or cfg.MOSS2_DEFAULT_VARIANT),
            template=str(prof.get("template") or cfg.MOSS2_DEFAULT_TEMPLATE),
            params_version=effective_version(prof),
            data_csv=out.get("data_csv"),
            discipline=discipline,
        )
        saved += 1

    return {
        "profiles": len(profiles),
        "saved": saved,
        "skipped": skipped,
        "errors": errors,
    }

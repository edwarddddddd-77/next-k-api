#!/usr/bin/env python3
"""
Phase 2：日内滚动清洗（Rolling Pool Backtest）

时间（上海，与 scheduler 一致）：12:05 / 16:05 / 20:05 / 00:05 / 04:05

动作：仅对当前 touch_pool 内标的，重跑过去 24h walk-forward（与主筛同源回测引擎）。

淘汰（任一即 DELETE，池子只减不增）：
- 触轨胜率 < 70%（ZCT_TOUCH_POOL_ROLLING_MIN_WIN_RATE）
- 扣摩擦 PF < 1.15（ZCT_TOUCH_POOL_ROLLING_MIN_PF）
- 周期末连续亏损 >= 3（ZCT_TOUCH_POOL_ROLLING_MAX_CONSEC_LOSSES）

主筛（08:05 全市场大选）见 zct_vwap_asset_pool_daily_job.py。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

from accumulation_radar import init_db
from zct_vwap_asset_pool import (
    rolling_clean_config,
    rolling_evict_reason,
    run_walkforward_enriched,
)
from zct_vwap_touch_pool_db import (
    touch_pool_apply_rolling_clean,
    touch_pool_list_symbols,
)

logger = logging.getLogger(__name__)


def rolling_clean_enabled() -> bool:
    raw = os.getenv("ZCT_TOUCH_POOL_ROLLING_ENABLED", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _symbol_row(per: Dict[str, Any], sym: str) -> Optional[Dict[str, Any]]:
    su = str(sym).strip().upper()
    row = per.get(su)
    if row is not None:
        return row
    for k, v in per.items():
        if str(k).strip().upper() == su:
            return v
    return None


def run_rolling_pool_clean(
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    池内标的 24h 滚动回测 → 不达标则从 touch_pool DELETE，并 prune signals。
    回测期间不持有 DB 连接；写库单事务。
    """
    cfg = dict(config or rolling_clean_config())
    pool_syms = touch_pool_list_symbols()
    if not pool_syms:
        return {
            "removed": [],
            "kept": [],
            "checked": 0,
            "criteria": cfg,
            "details": [],
            "signals_pruned": 0,
            "generated_at_ms": int(time.time() * 1000),
        }

    bt = run_walkforward_enriched(
        days=float(cfg["days"]),
        symbols=pool_syms,
        sleep_between_symbols=float(cfg["sleep_between_symbols"]),
        signal_interval=str(cfg["signal_interval"]),
        quiet=True,
        scan_phase="rolling_clean",
        rolling_cfg=cfg,
    )
    per = bt.get("per_symbol") or {}
    criteria = {**cfg, **(bt.get("criteria") or {})}
    meta = {
        "user_start_open_ms": (bt.get("summary") or {}).get("user_start_open_ms"),
        "hist_end_open_ms": (bt.get("summary") or {}).get("hist_end_open_ms"),
        "trades_emitted": (bt.get("summary") or {}).get("trades_emitted"),
    }

    min_wr = float(cfg["min_touch_win_rate"])
    min_pf = float(cfg["min_profit_factor"])
    max_consec = int(cfg["max_consecutive_losses_evict"])
    min_touch = int(cfg.get("min_win_loss_abs", 0))

    to_remove: List[str] = []
    kept: List[str] = []
    kept_rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    for sym in pool_syms:
        row = _symbol_row(per, sym)
        if not row:
            kept.append(sym)
            details.append(
                {
                    "symbol": sym,
                    "evict_reason": None,
                    "note": "no_per_symbol_stats_keep",
                }
            )
            logger.warning(
                "[rolling_clean] keep %s: no per_symbol stats (transient backtest miss)",
                sym,
            )
            continue

        snap = {
            "symbol": sym,
            "win_rate_touch_sl_tp": row.get("win_rate_touch_sl_tp"),
            "profit_factor_net": row.get("profit_factor_net"),
            "profit_factor_net_display": row.get("profit_factor_net_display"),
            "consecutive_losses_at_end": row.get("consecutive_losses_at_end"),
            "win": row.get("win"),
            "loss": row.get("loss"),
            "n_trades": row.get("n_trades"),
            "expired": row.get("expired"),
            "unresolved": row.get("unresolved"),
        }
        reason = rolling_evict_reason(
            row,
            min_touch_win_rate=min_wr,
            min_profit_factor=min_pf,
            max_consecutive_losses_evict=max_consec,
            min_win_loss_abs=min_touch,
        )
        if reason:
            to_remove.append(sym)
            details.append({**snap, "evict_reason": reason})
        else:
            kept.append(sym)
            kept_rows.append(snap)
            details.append({**snap, "evict_reason": None})

    generated_at_ms = int(time.time() * 1000)
    audit: Dict[str, Any] = {
        "generated_at_ms": generated_at_ms,
        "removed": to_remove,
        "kept": kept,
        "checked": len(pool_syms),
        "criteria": criteria,
        "details": details,
        "backtest_meta": meta,
    }

    removed: List[str] = []
    signals_pruned = 0
    pool_deleted = 0
    kept_updated = 0

    conn = init_db()
    try:
        stats = touch_pool_apply_rolling_clean(
            conn,
            to_remove=to_remove,
            kept_rows=kept_rows,
            remaining_symbols=kept,
            audit=audit,
            days=float(cfg["days"]),
            signal_interval=str(cfg["signal_interval"]),
        )
        pool_deleted = int(stats.get("pool_deleted") or 0)
        signals_pruned = int(stats.get("signals_pruned") or 0)
        kept_updated = int(stats.get("kept_updated") or 0)
        removed = list(to_remove)
        logger.info(
            "rolling_clean removed=%s pool_deleted=%s kept_updated=%s signals_pruned=%s",
            removed,
            pool_deleted,
            kept_updated,
            signals_pruned,
        )
    finally:
        conn.close()

    return {
        "removed": removed,
        "kept": kept,
        "checked": len(pool_syms),
        "criteria": criteria,
        "signals_pruned": signals_pruned,
        "pool_deleted": pool_deleted,
        "kept_updated": kept_updated,
        "details": details,
        "backtest_meta": meta,
        "generated_at_ms": generated_at_ms,
    }


# 兼容旧名
run_intraday_prune = run_rolling_pool_clean


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if not rolling_clean_enabled():
        print("[rolling_clean] disabled (ZCT_TOUCH_POOL_ROLLING_ENABLED=0)", flush=True)
        return
    ap = argparse.ArgumentParser(description="ZCT 触轨池日内滚动清洗（24h walk-forward）")
    ap.add_argument("--days", type=float, default=None, help="回测窗口天数，默认 1")
    ap.add_argument(
        "--min-touch-win-rate",
        type=float,
        default=None,
        help="低于该触轨胜率则出池，默认 0.70",
    )
    ap.add_argument(
        "--min-profit-factor",
        type=float,
        default=None,
        help="低于该扣摩擦 PF 则出池，默认 1.15",
    )
    ap.add_argument(
        "--max-consecutive-losses",
        type=int,
        default=None,
        help="末段连亏达到该值则出池，默认 3",
    )
    args = ap.parse_args()
    cfg = rolling_clean_config()
    if args.days is not None:
        cfg["days"] = float(args.days)
    if args.min_touch_win_rate is not None:
        cfg["min_touch_win_rate"] = float(args.min_touch_win_rate)
    if args.min_profit_factor is not None:
        cfg["min_profit_factor"] = float(args.min_profit_factor)
    if args.max_consecutive_losses is not None:
        cfg["max_consecutive_losses_evict"] = int(args.max_consecutive_losses)

    out = run_rolling_pool_clean(config=cfg)
    if out.get("removed"):
        print(f"[rolling_clean] removed={out['removed']}", flush=True)
    else:
        print(
            f"[rolling_clean] ok checked={out.get('checked')} kept={len(out.get('kept') or [])}",
            flush=True,
        )


if __name__ == "__main__":
    main()

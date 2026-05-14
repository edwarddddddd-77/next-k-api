#!/usr/bin/env python3
"""
ZCT VWAP 触轨资产池：walk-forward 近 N 天 → 按触轨胜率与 win+loss 筛选。

口径（与 `zct_vwap_walkforward_backtest` 的 `per_symbol` 一致）：
- **触轨胜率** = win / (win + loss)，即 `win_rate_touch_sl_tp`
- **触轨样本** = win + loss

默认：**触轨胜率 >= 80%** 且 **win+loss >= 130**。严格 **>** 用 `--strict-greater-rate` / `--strict-greater-touch`。

用法：
  cd next-k-api
  python zct_vwap_asset_pool.py --days 3 --zct-default-22

定时 + 写入 accumulation.db 见 **`zct_vwap_asset_pool_daily_job.py`**（`--once` / `--daemon`）。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from zct_vwap_walkforward_backtest import (
    _default_symbol_list,
    run_backtest,
    zct_default_22_symbols,
)

import zct_vwap_signal_scanner as z


def _filter_pool(
    summary: Dict[str, Any],
    *,
    min_touch_trades: int,
    strict_greater_touch: bool,
    min_touch_win_rate: float,
    strict_greater_rate: bool,
) -> Dict[str, Any]:
    per = summary.get("per_symbol") or {}
    matched: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for sym, row in per.items():
        su = str(sym).strip().upper()
        w = int(row.get("win", 0) or 0)
        l_ = int(row.get("loss", 0) or 0)
        touch = w + l_
        wr = row.get("win_rate_touch_sl_tp")
        wr_f = float(wr) if wr is not None else None

        ok_touch = (
            (touch > min_touch_trades)
            if strict_greater_touch
            else (touch >= min_touch_trades)
        )
        ok_wr = False
        if wr_f is not None:
            ok_wr = (
                (wr_f > min_touch_win_rate)
                if strict_greater_rate
                else (wr_f >= min_touch_win_rate)
            )

        rec: Dict[str, Any] = {
            "symbol": su,
            "win": w,
            "loss": l_,
            "win_plus_loss": touch,
            "expired": int(row.get("expired", 0) or 0),
            "unresolved": int(row.get("unresolved", 0) or 0),
            "win_rate_touch_sl_tp": wr_f,
        }
        if ok_touch and ok_wr:
            matched.append(rec)
        else:
            rec["reject_reason"] = []
            if not ok_touch:
                rec["reject_reason"].append(
                    "win_plus_loss"
                    + ("<" if not strict_greater_touch else "<=")
                    + str(min_touch_trades)
                )
            if not ok_wr:
                rec["reject_reason"].append("touch_win_rate_below_threshold")
            rejected.append(rec)

    matched.sort(
        key=lambda r: (-(r["win_rate_touch_sl_tp"] or 0.0), -r["win_plus_loss"])
    )
    return {"matched": matched, "rejected": rejected}


def run_asset_pool_scan(
    *,
    days: float = 3.0,
    symbols: List[str],
    ignore_db_cooldown: bool = True,
    sleep_between_symbols: float = 0.0,
    signal_interval: str = "1m",
    min_touch_trades: int = 130,
    strict_greater_touch: bool = False,
    min_touch_win_rate: float = 0.8,
    strict_greater_rate: bool = False,
    quiet: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """跑 walk-forward 并筛选；返回 (pool_payload, raw_backtest_summary)。"""
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx:
        summary = run_backtest(
            days=float(days),
            symbols=symbols,
            ignore_db_cooldown=ignore_db_cooldown,
            csv_path=None,
            sleep_between_symbols=max(0.0, float(sleep_between_symbols)),
            json_summary_path=None,
            signal_interval=str(signal_interval),
        )

    filt = _filter_pool(
        summary,
        min_touch_trades=int(min_touch_trades),
        strict_greater_touch=bool(strict_greater_touch),
        min_touch_win_rate=float(min_touch_win_rate),
        strict_greater_rate=bool(strict_greater_rate),
    )

    out: Dict[str, Any] = {
        "generated_at_ms": int(time.time() * 1000),
        "criteria": {
            "days": float(days),
            "signal_interval": str(signal_interval),
            "min_touch_win_rate": float(min_touch_win_rate),
            "touch_rate_comparison": ">" if strict_greater_rate else ">=",
            "min_win_plus_loss": int(min_touch_trades),
            "win_plus_loss_comparison": ">" if strict_greater_touch else ">=",
        },
        "symbols_scanned": [str(s).strip().upper() for s in symbols],
        "matched_symbols": [m["symbol"] for m in filt["matched"]],
        "matched": filt["matched"],
        "rejected": filt["rejected"],
        "backtest_meta": {
            "user_start_open_ms": summary.get("user_start_open_ms"),
            "hist_end_open_ms": summary.get("hist_end_open_ms"),
            "trades_emitted": summary.get("trades_emitted"),
        },
    }
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="ZCT VWAP 触轨资产池筛选")
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
    ap.add_argument(
        "--json-out",
        type=str,
        default=str(Path(__file__).resolve().parent / "zct_vwap_asset_pool.json"),
    )
    ap.add_argument("--no-json-out", action="store_true")
    ap.add_argument("--sleep-between-symbols", type=float, default=0.0)
    args = ap.parse_args()

    if args.zct_default_22:
        symbols = zct_default_22_symbols()
    elif args.use_env_symbols:
        symbols = z._symbols_from_env()
    elif args.symbols.strip():
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    else:
        symbols = _default_symbol_list()

    if not symbols:
        print("[pool] no symbols", file=sys.stderr)
        sys.exit(1)
    if args.ignore_db_cooldown and args.use_db_cooldown:
        ap.error("--ignore-db-cooldown 与 --use-db-cooldown 互斥")
    ignore_db_cooldown = not bool(args.use_db_cooldown)

    out, _ = run_asset_pool_scan(
        days=float(args.days),
        symbols=symbols,
        ignore_db_cooldown=ignore_db_cooldown,
        sleep_between_symbols=max(0.0, float(args.sleep_between_symbols)),
        signal_interval=str(args.signal_interval),
        min_touch_trades=int(args.min_touch_trades),
        strict_greater_touch=bool(args.strict_greater_touch),
        min_touch_win_rate=float(args.min_touch_win_rate),
        strict_greater_rate=bool(args.strict_greater_rate),
        quiet=True,
    )

    if not args.no_json_out:
        p = Path(args.json_out.strip() or "zct_vwap_asset_pool.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[pool] json -> {p.resolve()}", flush=True)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    ct = ">" if args.strict_greater_touch else ">="
    cr = ">" if args.strict_greater_rate else ">="
    print(
        f"\n[pool] matched {len(out['matched_symbols'])}/{len(symbols)}  "
        f"(win+loss {ct} {args.min_touch_trades}, touch_win_rate {cr} {args.min_touch_win_rate})",
        flush=True,
    )


if __name__ == "__main__":
    main()

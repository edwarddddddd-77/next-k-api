#!/usr/bin/env python3
"""
ZCT VWAP 触轨资产池：walk-forward 近 N 天（默认 1.5 天 ≈ 36 小时）→ 按触轨胜率与 win+loss 筛选。

口径（与 `zct_vwap_walkforward_backtest` 的 `per_symbol` 一致）：
- **触轨胜率** = 整个 walk 窗口内 win / (win + loss)，即 `win_rate_touch_sl_tp`（不按 UTC 日历日拆分）
- **触轨样本** = win + loss

默认：**触轨胜率 >= 75%** 且 **win+loss >= 50**。严格 **>** 用 `--strict-greater-rate` / `--strict-greater-touch`。

用法：
  cd next-k-api
  python zct_vwap_asset_pool.py --days 1.5 --zct-default-22
  python zct_vwap_asset_pool.py --days 1.5 --hot-oi-plus-default-22

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
from typing import Any, Dict, List, Optional, Tuple

from zct_vwap_walkforward_backtest import (
    _default_symbol_list,
    run_backtest,
    zct_default_22_symbols,
)

import zct_vwap_signal_scanner as z


def touch_pool_symbols_hot_oi_plus_default_22() -> List[str]:
    """
    worth_watch_hot_oi（🔥⚡ 热度+OI）∪ 扫描器内置默认永续列表；顺序为内置默认在前，再追加 hot 表独有标的。
    """
    base = zct_default_22_symbols()
    hot = z.hot_oi_watchlist_symbols()
    seen = set(base)
    out = list(base)
    for s in hot:
        su = str(s).strip().upper()
        if su and su not in seen:
            seen.add(su)
            out.append(su)
    return out


def _window_touch_rows(summary: Dict[str, Any], symbols: List[str]) -> List[Dict[str, Any]]:
    """walk 全窗口（如近 36h）内各标的触轨统计：来自 summary['per_symbol']，不按日历日分桶。"""
    per = summary.get("per_symbol") or {}
    out: List[Dict[str, Any]] = []
    for raw in symbols:
        su = str(raw).strip().upper()
        if not su:
            continue
        row = per.get(su) or {}
        w = int(row.get("win") or 0)
        l_ = int(row.get("loss") or 0)
        wr = row.get("win_rate_touch_sl_tp")
        wr_f = float(wr) if wr is not None else None
        out.append(
            {
                "symbol": su,
                "win_rate_touch_sl_tp": wr_f,
                "win": w,
                "loss": l_,
                "win_plus_loss": w + l_,
                "expired": int(row.get("expired") or 0),
                "unresolved": int(row.get("unresolved") or 0),
            }
        )
    return out


def _format_window_touch_line(rows: List[Dict[str, Any]], days: float) -> str:
    hrs = int(round(float(days) * 24.0))
    parts: List[str] = []
    for r in rows:
        su = str(r.get("symbol") or "")
        wr = r.get("win_rate_touch_sl_tp")
        w, l_ = int(r.get("win") or 0), int(r.get("loss") or 0)
        if wr is None:
            parts.append(f"{su} n/a(w/L={w}/{l_})")
        else:
            parts.append(f"{su} {100.0 * float(wr):.2f}%({w}/{l_})")
    return (
        f"[pool] walk 近{float(days):g}天(≈{hrs}h) 触轨胜率 win/(w+L): "
        + "; ".join(parts)
    )


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
    days: float = 1.5,
    symbols: List[str],
    ignore_db_cooldown: bool = True,
    sleep_between_symbols: float = 0.0,
    signal_interval: str = "1m",
    min_touch_trades: int = 50,
    strict_greater_touch: bool = False,
    min_touch_win_rate: float = 0.75,
    strict_greater_rate: bool = False,
    quiet: bool = True,
    symbols_source: Optional[str] = None,
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
            emit_text_report=False,
        )

    touch_rows = _window_touch_rows(summary, symbols)
    print(_format_window_touch_line(touch_rows, float(days)), flush=True)
    filt = _filter_pool(
        summary,
        min_touch_trades=int(min_touch_trades),
        strict_greater_touch=bool(strict_greater_touch),
        min_touch_win_rate=float(min_touch_win_rate),
        strict_greater_rate=bool(strict_greater_rate),
    )

    crit: Dict[str, Any] = {
        "days": float(days),
        "signal_interval": str(signal_interval),
        "min_touch_win_rate": float(min_touch_win_rate),
        "touch_rate_comparison": ">" if strict_greater_rate else ">=",
        "min_win_plus_loss": int(min_touch_trades),
        "win_plus_loss_comparison": ">" if strict_greater_touch else ">=",
    }
    if symbols_source:
        crit["symbols_source"] = str(symbols_source)

    out: Dict[str, Any] = {
        "generated_at_ms": int(time.time() * 1000),
        "criteria": crit,
        "symbols_scanned": [str(s).strip().upper() for s in symbols],
        "matched_symbols": [m["symbol"] for m in filt["matched"]],
        "matched": filt["matched"],
        "rejected": filt["rejected"],
        "touch_in_window": touch_rows,
        "backtest_meta": {
            "user_start_open_ms": summary.get("user_start_open_ms"),
            "hist_end_open_ms": summary.get("hist_end_open_ms"),
            "trades_emitted": summary.get("trades_emitted"),
        },
    }
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="ZCT VWAP 触轨资产池筛选")
    ap.add_argument("--days", type=float, default=1.5)
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--zct-default-22", action="store_true")
    ap.add_argument(
        "--hot-oi-plus-default-22",
        action="store_true",
        help="worth_watch_hot_oi ∪ 扫描器内置默认永续（内置列表顺序在前）",
    )
    ap.add_argument("--use-env-symbols", action="store_true")
    ap.add_argument("--ignore-db-cooldown", action="store_true")
    ap.add_argument("--use-db-cooldown", action="store_true")
    ap.add_argument("--min-touch-trades", type=int, default=50)
    ap.add_argument("--strict-greater-touch", action="store_true")
    ap.add_argument("--min-touch-win-rate", type=float, default=0.75)
    ap.add_argument("--strict-greater-rate", action="store_true")
    ap.add_argument("--signal-interval", type=str, default="1m", choices=["1m", "5m"])
    ap.add_argument(
        "--json-out",
        type=str,
        default=str(Path(__file__).resolve().parent / "zct_vwap_asset_pool.json"),
    )
    ap.add_argument("--no-json-out", action="store_true")
    ap.add_argument(
        "--write-db",
        action="store_true",
        help="walk-forward 完成后：先清空 zct_vwap_touch_pool 再写入 accumulation.db，并追加 runs 审计",
    )
    ap.add_argument("--sleep-between-symbols", type=float, default=0.0)
    args = ap.parse_args()

    mode_n = sum(
        1
        for x in (
            args.hot_oi_plus_default_22,
            args.zct_default_22,
            args.use_env_symbols,
            bool(args.symbols.strip()),
        )
        if x
    )
    if mode_n > 1:
        ap.error(
            "标的来源请只选一种：--hot-oi-plus-default-22 / --zct-default-22 / "
            "--use-env-symbols / --symbols"
        )

    sym_src: Optional[str] = None
    if args.hot_oi_plus_default_22:
        symbols = touch_pool_symbols_hot_oi_plus_default_22()
        sym_src = "hot_oi_plus_default_22"
    elif args.zct_default_22:
        symbols = zct_default_22_symbols()
        sym_src = "zct_default_22"
    elif args.use_env_symbols:
        symbols = z._symbols_from_env()
        sym_src = "env_symbols"
    elif args.symbols.strip():
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
        sym_src = "cli_symbols"
    else:
        symbols = _default_symbol_list()
        sym_src = "default_symbol_list"

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
        symbols_source=sym_src,
    )

    if args.write_db:
        from accumulation_radar import init_db
        from zct_vwap_touch_pool_db import touch_pool_ensure_schema, touch_pool_write_db

        conn = init_db()
        try:
            touch_pool_ensure_schema(conn)
            n = touch_pool_write_db(conn, out)
        finally:
            conn.close()
        print(f"[pool] db cleared+written touch_pool_rows={n}", flush=True)

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

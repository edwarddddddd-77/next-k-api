#!/usr/bin/env python3
"""
ZCT VWAP 触轨资产池：walk-forward 近 N 天（默认 1.5 天 ≈ 36 小时）→ 按触轨胜率与 win+loss 筛选。

口径（与 `zct_vwap_walkforward_backtest` 的 `per_symbol` 一致）：
- **触轨胜率** = 整个 walk 窗口内 win / (win + loss)，即 `win_rate_touch_sl_tp`（不按 UTC 日历日拆分）
- **触轨样本** = win + loss

默认：**触轨胜率 >= 70%**、**win+loss >= 1**（`--min-touch-trades`）、**win+loss >= 20**（`--min-win-loss-abs`，≤0 关闭）、**(win+loss)/n_trades >= 35%**（`--min-touch-share`，≤0 关闭）、walk 窗口内 **总笔数 n_trades >= 30**、**过期占比 expired/n_trades < 50%**。  
触轨样本与胜率仍按 win/(win+loss)；过期占比为 walk 内该标的全部回测笔。严格 **>** 用 `--strict-greater-rate` / `--strict-greater-touch`。

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
        nt = int(row.get("n_trades") or 0)
        ex = int(row.get("expired") or 0)
        wr = row.get("win_rate_touch_sl_tp")
        wr_f = float(wr) if wr is not None else None
        exp_ratio = (float(ex) / float(nt)) if nt > 0 else None
        out.append(
            {
                "symbol": su,
                "win_rate_touch_sl_tp": wr_f,
                "win": w,
                "loss": l_,
                "win_plus_loss": w + l_,
                "n_trades": nt,
                "expired": ex,
                "expired_ratio": round(exp_ratio, 6) if exp_ratio is not None else None,
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


def _format_pool_scan_aggregate_lines(
    summary: Dict[str, Any],
    touch_rows: List[Dict[str, Any]],
    matched: List[Dict[str, Any]],
) -> str:
    """
    walk 全局汇总 + 本批扫描标的合计 + 本轮筛选入库列表（与 JSON matched_symbols 一致）。
    """
    te = int(summary.get("trades_emitted") or 0)
    glob_ex = int(summary.get("expired") or 0)
    sum_nt = sum(int(r.get("n_trades") or 0) for r in touch_rows)
    sum_ex = sum(int(r.get("expired") or 0) for r in touch_rows)
    syms_m = [str(m.get("symbol") or "").strip().upper() for m in matched if m.get("symbol")]
    matched_str = ",".join(syms_m) if syms_m else "(无)"
    return (
        f"[pool] walk 汇总: 总发单笔 trades_emitted={te}; 全局超时(expired 结案)={glob_ex}\n"
        f"[pool] 本批扫描标的 Σn_trades={sum_nt}; Σexpired(按标的汇总)={sum_ex}\n"
        f"[pool] 本轮入库 matched={len(syms_m)}: {matched_str}"
    )


def _filter_pool(
    summary: Dict[str, Any],
    *,
    min_touch_trades: int,
    strict_greater_touch: bool,
    min_touch_win_rate: float,
    strict_greater_rate: bool,
    min_total_trades: int,
    max_expired_ratio: float,
    min_win_loss_abs: int = 20,
    min_touch_share: float = 0.35,
) -> Dict[str, Any]:
    per = summary.get("per_symbol") or {}
    matched: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for sym, row in per.items():
        su = str(sym).strip().upper()
        w = int(row.get("win", 0) or 0)
        l_ = int(row.get("loss", 0) or 0)
        touch = w + l_
        n_tr = int(row.get("n_trades", 0) or 0)
        ex = int(row.get("expired", 0) or 0)
        exp_ratio = (float(ex) / float(n_tr)) if n_tr > 0 else float("inf")
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
        ok_nt = n_tr >= int(min_total_trades)
        ok_exp = exp_ratio < float(max_expired_ratio)
        floor_abs = int(min_win_loss_abs)
        ok_wl_abs = True if floor_abs <= 0 else (touch >= floor_abs)
        share_min = float(min_touch_share)
        if share_min <= 0.0:
            ok_share = True
        else:
            ok_share = n_tr > 0 and (float(touch) / float(n_tr)) >= share_min

        rec: Dict[str, Any] = {
            "symbol": su,
            "win": w,
            "loss": l_,
            "win_plus_loss": touch,
            "n_trades": n_tr,
            "expired": ex,
            "expired_ratio": round(exp_ratio, 6) if n_tr > 0 else None,
            "unresolved": int(row.get("unresolved", 0) or 0),
            "win_rate_touch_sl_tp": wr_f,
        }
        if ok_touch and ok_wr and ok_nt and ok_exp and ok_wl_abs and ok_share:
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
            if not ok_nt:
                rec["reject_reason"].append(
                    f"n_trades_lt_{int(min_total_trades)}"
                )
            if not ok_exp:
                if n_tr <= 0:
                    rec["reject_reason"].append("expired_ratio_undefined_n_trades_0")
                elif exp_ratio >= float(max_expired_ratio):
                    rec["reject_reason"].append("expired_ratio_above_or_equal_max")
                else:
                    rec["reject_reason"].append("expired_ratio_invalid")
            if not ok_wl_abs and floor_abs > 0:
                rec["reject_reason"].append("touch_trades_below_min")
            if not ok_share and share_min > 0.0:
                rec["reject_reason"].append("touch_share_below_min")
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
    min_touch_trades: int = 1,
    strict_greater_touch: bool = False,
    min_touch_win_rate: float = 0.7,
    strict_greater_rate: bool = False,
    min_total_trades: int = 30,
    max_expired_ratio: float = 0.5,
    min_win_loss_abs: int = 20,
    min_touch_share: float = 0.35,
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
        min_total_trades=int(min_total_trades),
        max_expired_ratio=float(max_expired_ratio),
        min_win_loss_abs=int(min_win_loss_abs),
        min_touch_share=float(min_touch_share),
    )
    print(
        _format_pool_scan_aggregate_lines(summary, touch_rows, filt["matched"]),
        flush=True,
    )

    crit: Dict[str, Any] = {
        "days": float(days),
        "signal_interval": str(signal_interval),
        "min_touch_win_rate": float(min_touch_win_rate),
        "touch_rate_comparison": ">" if strict_greater_rate else ">=",
        "min_win_plus_loss": int(min_touch_trades),
        "win_plus_loss_comparison": ">" if strict_greater_touch else ">=",
        "min_total_trades": int(min_total_trades),
        "n_trades_rule": f"n_trades >= {int(min_total_trades)}",
        "max_expired_ratio_exclusive": float(max_expired_ratio),
        "expired_ratio_rule": f"expired / n_trades < {float(max_expired_ratio)}",
        "min_win_loss_abs": int(min_win_loss_abs),
        "min_win_loss_abs_rule": (
            "off"
            if int(min_win_loss_abs) <= 0
            else f"win+loss >= {int(min_win_loss_abs)}"
        ),
        "min_touch_share": float(min_touch_share),
        "min_touch_share_rule": (
            "off"
            if float(min_touch_share) <= 0.0
            else f"(win+loss)/n_trades >= {float(min_touch_share)}"
        ),
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
    ap.add_argument("--min-touch-trades", type=int, default=1)
    ap.add_argument("--strict-greater-touch", action="store_true")
    ap.add_argument("--min-touch-win-rate", type=float, default=0.7)
    ap.add_argument("--strict-greater-rate", action="store_true")
    ap.add_argument(
        "--min-total-trades",
        type=int,
        default=30,
        help="walk 窗口内该标的总回测笔数 n_trades 须 **≥** 本值（默认 30）",
    )
    ap.add_argument(
        "--max-expired-ratio",
        type=float,
        default=0.5,
        help="过期占比 expired/n_trades 须 **严格小于** 本值（默认 0.5 即 <50%%）",
    )
    ap.add_argument(
        "--min-win-loss-abs",
        type=int,
        default=20,
        help="触轨绝对样本 win+loss 须 ≥ 本值（默认 20；0=关闭）",
    )
    ap.add_argument(
        "--min-touch-share",
        type=float,
        default=0.35,
        help="触轨占比 (win+loss)/n_trades 须 ≥ 本值（默认 0.35；0=关闭）",
    )
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
        min_total_trades=int(args.min_total_trades),
        max_expired_ratio=float(args.max_expired_ratio),
        min_win_loss_abs=int(args.min_win_loss_abs),
        min_touch_share=float(args.min_touch_share),
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
        f"(win+loss {ct} {args.min_touch_trades}, touch_win_rate {cr} {args.min_touch_win_rate}, "
        f"n_trades >= {args.min_total_trades}, expired/n_trades < {args.max_expired_ratio}, "
        f"win+loss_abs>={args.min_win_loss_abs} (0=off), touch_share>={args.min_touch_share} (0=off)",
        flush=True,
    )


if __name__ == "__main__":
    main()

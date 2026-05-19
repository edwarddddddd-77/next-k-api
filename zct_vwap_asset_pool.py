#!/usr/bin/env python3
"""
ZCT VWAP 触轨资产池：walk-forward（默认近 24h）→ 按触轨胜率、扣摩擦 PF 等筛选。

口径（与 `zct_vwap_walkforward_backtest` 的 `per_symbol` 一致）：
- **触轨胜率** = 整个 walk 窗口内 win / (win + loss)，即 `win_rate_touch_sl_tp`（不按 UTC 日历日拆分）
- **触轨样本** = win + loss

**每日主筛（Phase 1，默认）**：上海 **08:05** 跑 **严格 24h** walk-forward（`days=1`），入库须同时满足：

- **n_trades >= 20**（1m 活跃度）
- **触轨胜率 >= 72%**（win/(win+loss)）
- **profit_factor > 1.25**（扣双边 Taker 4bps + 滑点 1.5bps 后）
- **周期末连续亏损 < 3**（`consecutive_losses_at_end`）

可选稳档附加：`max_expired_ratio`、`min_touch_share`（默认主筛关闭，见 CLI）。

候选宇宙（生产默认）：**值得关注七类 worth_watch_* ∪ 内置默认永续**。

**日内滚动清洗（Phase 2）**：见 `zct_touch_pool_intraday_prune.py`（12:05/16:05/20:05/00:05/04:05 上海；
仅池内标的重跑 24h walk-forward，触轨胜率<70% / PF<1.15 / 末段连亏>=3 则 DELETE，只减不增）。

用法：
  cd next-k-api
  python zct_vwap_asset_pool.py --days 1 --worth-watch-plus-default-22
  python zct_vwap_asset_pool.py --days 1 --hot-oi-plus-default-22

定时 + 写入 accumulation.db 见 **`zct_vwap_asset_pool_daily_job.py`**（`--once` / `--daemon`）。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from zct_touch_pool_metrics import (
    enrich_per_symbol_stats,
    friction_bps_per_side,
    slippage_bps_per_side,
    taker_bps_per_side,
)
from zct_vwap_walkforward_backtest import (
    _default_symbol_list,
    run_backtest,
    zct_default_22_symbols,
)

import zct_vwap_signal_scanner as z


def touch_pool_symbols_hot_oi_plus_default_22() -> List[str]:
    """
    worth_watch_hot_oi ∪ 内置默认永续（`_DEFAULT_ZCT_SYMBOLS`，非「仅 22 个」）。
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


def touch_pool_symbols_worth_watch_plus_default() -> List[str]:
    """
    值得关注七类 worth_watch_* ∪ 内置默认永续（`_DEFAULT_ZCT_SYMBOLS`，当前 33 个）。
    顺序：内置默认在前，再按七类表顺序追加独有标的。
    """
    base = zct_default_22_symbols()
    ww = z.worth_watch_all_category_symbols()
    seen = set(base)
    out = list(base)
    for s in ww:
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


def _profit_factor_value(row: Dict[str, Any]) -> float:
    raw = row.get("profit_factor_net")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    if row.get("profit_factor_net_display") == "inf":
        return float("inf")
    return 0.0


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
    min_profit_factor: float = 1.25,
    max_consecutive_losses_at_end: int = 2,
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

        pf_f = _profit_factor_value(row)
        ok_pf = pf_f > float(min_profit_factor)
        consec = int(row.get("consecutive_losses_at_end") or 0)
        ok_consec = consec <= int(max_consecutive_losses_at_end)

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
            "profit_factor_net": row.get("profit_factor_net"),
            "profit_factor_net_display": row.get("profit_factor_net_display"),
            "consecutive_losses_at_end": consec,
            "gross_profit_net_usdt": row.get("gross_profit_net_usdt"),
            "gross_loss_net_usdt": row.get("gross_loss_net_usdt"),
        }
        if (
            ok_touch
            and ok_wr
            and ok_nt
            and ok_exp
            and ok_wl_abs
            and ok_share
            and ok_pf
            and ok_consec
        ):
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
            if not ok_pf:
                rec["reject_reason"].append("profit_factor_below_threshold")
            if not ok_consec:
                rec["reject_reason"].append("consecutive_losses_at_end_too_high")
            rejected.append(rec)

    matched.sort(
        key=lambda r: (
            -(r["win_rate_touch_sl_tp"] or 0.0),
            -(_profit_factor_value(r) if _profit_factor_value(r) != float("inf") else 999.0),
            -r["win_plus_loss"],
        )
    )
    return {"matched": matched, "rejected": rejected}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def rolling_clean_config() -> Dict[str, Any]:
    """Phase 2 滚动清洗阈值（环境变量可覆盖）。"""
    return {
        "days": _float_env("ZCT_TOUCH_POOL_ROLLING_DAYS", 1.0),
        "min_touch_win_rate": _float_env("ZCT_TOUCH_POOL_ROLLING_MIN_WIN_RATE", 0.70),
        "min_profit_factor": _float_env("ZCT_TOUCH_POOL_ROLLING_MIN_PF", 1.15),
        "max_consecutive_losses_evict": _int_env(
            "ZCT_TOUCH_POOL_ROLLING_MAX_CONSEC_LOSSES", 3
        ),
        "min_win_loss_abs": _int_env("ZCT_TOUCH_POOL_ROLLING_MIN_TOUCH_TRADES", 5),
        "sleep_between_symbols": _float_env("ZCT_TOUCH_POOL_SLEEP_SYMBOLS", 0.25),
        "signal_interval": os.getenv("ZCT_TOUCH_POOL_ROLLING_INTERVAL", "1m").strip()
        or "1m",
    }


def rolling_evict_reason(
    row: Dict[str, Any],
    *,
    min_touch_win_rate: float = 0.70,
    min_profit_factor: float = 1.15,
    max_consecutive_losses_evict: int = 3,
    min_win_loss_abs: int = 5,
) -> Optional[str]:
    """
    滚动 24h 窗口内是否应从触轨池剔除；None=保留。
    任一触发：触轨胜率 < min_wr；扣摩擦 PF < min_pf；末段连亏 >= max_consec。
    触轨样本 win+loss < min_win_loss_abs 时不按胜率/PF 淘汰（防小样本颠簸）。
    """
    consec = int(row.get("consecutive_losses_at_end") or 0)
    if consec >= max(1, int(max_consecutive_losses_evict)):
        return "consecutive_losses_at_end_veto"

    w = int(row.get("win", 0) or 0)
    l_ = int(row.get("loss", 0) or 0)
    touch = w + l_
    floor = int(min_win_loss_abs)
    if floor > 0 and touch < floor:
        return None

    wr = row.get("win_rate_touch_sl_tp")
    if wr is None:
        return "touch_win_rate_unavailable"
    try:
        wr_f = float(wr)
    except (TypeError, ValueError):
        return "touch_win_rate_invalid"
    if wr_f < float(min_touch_win_rate):
        return "touch_win_rate_below_rolling_min"

    pf_f = _profit_factor_value(row)
    if pf_f != float("inf") and pf_f < float(min_profit_factor):
        return "profit_factor_below_rolling_min"

    return None


def run_walkforward_enriched(
    *,
    days: float,
    symbols: List[str],
    ignore_db_cooldown: bool = True,
    sleep_between_symbols: float = 0.25,
    signal_interval: str = "1m",
    quiet: bool = True,
    scan_phase: str = "rolling_clean",
    rolling_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """对给定标的列表跑 walk-forward 并 enrich per_symbol（不执行主筛/入库）。"""
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

    trades = list(summary.get("trades") or [])
    per = dict(summary.get("per_symbol") or {})
    summary["per_symbol"] = enrich_per_symbol_stats(
        per,
        trades,
        default_notional=float(z.VIRTUAL_NOTIONAL_USDT),
    )
    cfg = dict(rolling_cfg or rolling_clean_config())
    crit: Dict[str, Any] = {
        "days": float(days),
        "signal_interval": str(signal_interval),
        "scan_phase": str(scan_phase),
        "rolling_min_touch_win_rate": float(cfg["min_touch_win_rate"]),
        "rolling_touch_win_rate_rule": f"evict if win_rate_touch < {cfg['min_touch_win_rate']:.0%}",
        "rolling_min_profit_factor": float(cfg["min_profit_factor"]),
        "rolling_profit_factor_rule": f"evict if PF_after_friction < {cfg['min_profit_factor']}",
        "rolling_max_consecutive_losses_evict": int(cfg["max_consecutive_losses_evict"]),
        "rolling_consecutive_rule": (
            f"evict if end_streak >= {int(cfg['max_consecutive_losses_evict'])}"
        ),
        "rolling_min_win_loss_abs": int(cfg.get("min_win_loss_abs", 0)),
        "rolling_min_win_loss_rule": (
            f"skip wr/pf evict if win+loss < {int(cfg.get('min_win_loss_abs', 0))}"
            if int(cfg.get("min_win_loss_abs", 0)) > 0
            else "off"
        ),
        "friction_taker_bps_per_side": taker_bps_per_side(),
        "friction_slippage_bps_per_side": slippage_bps_per_side(),
        "play03_tp_mode": os.getenv("ZCT_PLAY03_TP_MODE", "vwap").strip().lower(),
    }
    return {
        "summary": summary,
        "per_symbol": summary["per_symbol"],
        "criteria": crit,
        "symbols_scanned": [str(s).strip().upper() for s in symbols],
    }


def notify_touch_pool_empty_if_needed(matched_count: int, *, criteria: Dict[str, Any]) -> None:
    """主筛零入选时 TG 观望播报（不抛异常）。"""
    if matched_count > 0:
        return
    off = os.getenv("ZCT_TOUCH_POOL_EMPTY_TG", "1").strip().lower()
    if off in ("0", "false", "no", "off"):
        return
    try:
        wr = criteria.get("min_touch_win_rate", 0.72)
        msg = (
            "【ZCT 触轨池】今日无标的满足主筛条件，系统进入观望模式。\n"
            f"条件：24h 回测 | 触轨胜率>={wr:.0%} | PF>1.25(扣摩擦) | 末段连亏<3 | n_trades>=20"
        )
        z.send_telegram(msg)
    except Exception as e:
        print(f"[pool] empty-pool TG notify failed: {e}", flush=True)


def run_asset_pool_scan(
    *,
    days: float = 1.0,
    symbols: List[str],
    ignore_db_cooldown: bool = True,
    sleep_between_symbols: float = 0.0,
    signal_interval: str = "1m",
    min_touch_trades: int = 1,
    strict_greater_touch: bool = False,
    min_touch_win_rate: float = 0.72,
    strict_greater_rate: bool = False,
    min_total_trades: int = 20,
    max_expired_ratio: float = 1.0,
    min_win_loss_abs: int = 0,
    min_touch_share: float = 0.0,
    min_profit_factor: float = 1.25,
    max_consecutive_losses_at_end: int = 2,
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

    trades = list(summary.get("trades") or [])
    per = dict(summary.get("per_symbol") or {})
    summary["per_symbol"] = enrich_per_symbol_stats(
        per,
        trades,
        default_notional=float(z.VIRTUAL_NOTIONAL_USDT),
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
        min_profit_factor=float(min_profit_factor),
        max_consecutive_losses_at_end=int(max_consecutive_losses_at_end),
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
        "min_profit_factor_exclusive": float(min_profit_factor),
        "profit_factor_rule": f"PF_after_friction > {float(min_profit_factor)}",
        "friction_taker_bps_per_side": taker_bps_per_side(),
        "friction_slippage_bps_per_side": slippage_bps_per_side(),
        "friction_round_trip_bps": friction_bps_per_side() * 2.0,
        "friction_note": "PF = sum(net_win) / sum(|net_loss|); net = raw_pnl - notional×2×(taker+slip)/1e4",
        "play03_tp_mode": os.getenv("ZCT_PLAY03_TP_MODE", "vwap").strip().lower(),
        "tp_mode_note": "PLAY01/02=1R；PLAY03 默认 vwap（ZCT_PLAY03_TP_MODE）",
        "max_consecutive_losses_at_end": int(max_consecutive_losses_at_end),
        "consecutive_losses_rule": f"end_streak <= {int(max_consecutive_losses_at_end)} (<3)",
        "scan_phase": "daily_master",
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
    ap.add_argument("--days", type=float, default=1.0, help="回测窗口天数；主筛默认 1=严格 24h")
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--zct-default-22", action="store_true")
    ap.add_argument(
        "--hot-oi-plus-default-22",
        action="store_true",
        help="worth_watch_hot_oi ∪ 扫描器内置默认永续（旧候选池，较窄）",
    )
    ap.add_argument(
        "--worth-watch-plus-default-22",
        action="store_true",
        help="值得关注七类 worth_watch_* ∪ 内置默认永续 _DEFAULT_ZCT_SYMBOLS（稳档默认大候选池；CLI 名 default22 为历史遗留）",
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
        help="walk 窗口内 n_trades 须 ≥ 本值（主筛默认 20）",
    )
    ap.add_argument(
        "--max-expired-ratio",
        type=float,
        default=1.0,
        help="expired/n_trades 须 < 本值（主筛默认 1=关闭；稳档可设 0.4）",
    )
    ap.add_argument(
        "--min-win-loss-abs",
        type=int,
        default=0,
        help="win+loss 须 ≥ 本值（0=关闭）",
    )
    ap.add_argument(
        "--min-touch-share",
        type=float,
        default=0.0,
        help="(win+loss)/n_trades 须 ≥ 本值（0=关闭）",
    )
    ap.add_argument(
        "--min-profit-factor",
        type=float,
        default=1.25,
        help="扣摩擦后 profit factor 须严格大于该值",
    )
    ap.add_argument(
        "--max-consecutive-losses-at-end",
        type=int,
        default=2,
        help="周期末连续亏损笔数上限（默认 2 即 <3）",
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
            args.worth_watch_plus_default_22,
            args.hot_oi_plus_default_22,
            args.zct_default_22,
            args.use_env_symbols,
            bool(args.symbols.strip()),
        )
        if x
    )
    if mode_n > 1:
        ap.error(
            "标的来源请只选一种：--worth-watch-plus-default-22 / --hot-oi-plus-default-22 / "
            "--zct-default-22 / --use-env-symbols / --symbols"
        )

    sym_src: Optional[str] = None
    if args.worth_watch_plus_default_22:
        symbols = touch_pool_symbols_worth_watch_plus_default()
        sym_src = "worth_watch_plus_default_22"
    elif args.hot_oi_plus_default_22:
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
        symbols = touch_pool_symbols_worth_watch_plus_default()
        sym_src = "worth_watch_plus_default_22"

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
        min_profit_factor=float(args.min_profit_factor),
        max_consecutive_losses_at_end=int(args.max_consecutive_losses_at_end),
        quiet=True,
        symbols_source=sym_src,
    )
    notify_touch_pool_empty_if_needed(
        len(out.get("matched_symbols") or []),
        criteria=out.get("criteria") or {},
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

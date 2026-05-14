#!/usr/bin/env python3
"""
ZCT VWAP 信号扫描器 — walk-forward 回测（不经 DB、不推 TG）。

**与 `run_scan` / `analyze_symbol` 默认尽量一致**，仅以下两项在回测中**固定关闭**（不跟环境）：
- **流动性 / OI**：恒关（不拉 `openInterestHist`），与当前扫描器默认 `LIQUIDITY_OI_FILTER_ENABLED=False` 一致；
  若你将来在环境打开 OI，本回测仍**强制**不关 OI（与「价量回测」一致）。
- **BTC 宏观红绿灯**：回测全程将 `z.BTC_MACRO_FILTER_ENABLED=False`（恢复运行前原值），
  不先跑 BTC 刷新状态、标的顺序也不按「宏观开启」重排。

**与实盘扫描的差异（回测默认）**：
- **冷却**：**默认不读** `zct_symbol_cooldown`（`z._cooldown_blocks` 恒为假），避免历史 walk-forward 依赖「当前 DB 里的冷却行」；与实盘完全一致时请加 **`--use-db-cooldown`**。`--ignore-db-cooldown` 仍保留，与默认等价，勿与 `--use-db-cooldown` 同开。
- **日损熔断**：walk-forward / portfolio 回测**不启用**（`halt_daily_circuit` 恒为 `False`），不重放「运行当日 UTC 结算触发的 P1 熔断」对历史逐根 classify 的影响；与 `run_scan` 实盘路径不同。

**不修改** `zct_vwap_signal_scanner.py`：`analyze_symbol_pit` 复制 `analyze_symbol` 在 classify 之后的闸门链；回测中 `halt_daily_circuit` 恒为 `False`（不走日损分支）。

Play03 动态 ATR：在 `classify_and_signal` 期间对 `z.fetch_klines` 做 monkeypatch，仅 `SPIKE_ATR_INTERVAL` 加 `endTime=asof`。

**数据从哪来、结果存哪：**
- K 线：运行时向币安 U 本位 `fapi.binance.com` 拉取（`fetch_klines_forward` / `api_get`），**不落盘原始 OHLCV**；只保留内存里的 DataFrame 直到进程结束。
- 结果：**stdout** 仍会打印完整 JSON + per-symbol 表；默认另写 **`next-k-api/zct_vwap_walkforward_last.json`**（与脚本同目录，每跑覆盖）。
  改路径用 `--json-out PATH`；只要打印不要文件用 `--no-json-out`。
- 逐笔明细：加 `--csv PATH` 才会写 CSV。
- **短窗口**：信号 K 线会向前扩到 **UTC 当日 0 点** 再拉线，保证会话 VWAP ≥30 根；统计仍只在你设的 `--days` 对应时间窗内逐根推进（见 JSON 里 `user_start_open_ms` / `kline_fetch_start_ms`）。`--signal-interval 5m` 时全链路按 **5m 步长**（拉线、walk、resolve、组合仿真、仓位约束相邻间距与「下一根开仓」均与 1m 模式同构）。
- **多标的末根时间**：逐根 walk 与 `resolve_forward` **按标的** cap 在该标的 K 线最后一根 `open_time`（不再取全市场 min 以免短序列拖短其它标的）；JSON 字段 **`hist_end_open_ms`** 取各标的末根的 **max**，供 `per_symbol_daily` / 仓位约束统计与日志对齐「最长覆盖」。
- **仓位约束口径**（JSON `position_constraints`，与「每根都下单」的原始 `trades` 并列；步长随 `--signal-interval`）：
  - `dedupe_adjacent_actionable`：同一标的 **连续同周期 actionable** 只保留每链 **第一笔**；
  - `adjacent_stack_chains_only`：**仅连打叠仓**——同标的相邻一根信号 K 且链长≥2 时，链内**每笔**都参与胜率；
  - `one_open_per_symbol`：同标的 **未平仓前不接新单**（下一笔 `entry_bar` 须 ≥ 上一笔 `exit_bar+1 根信号 K`；若上一笔未结则该标的后继全弃）。
- **JSON `per_symbol_daily`**：`utc_dates` 仅为 walk 窗口内的 **完整 UTC 自然日**（首尾半日剔除）；`by_symbol` 按发单 `signal_open_ms`；`by_symbol_exit_day` 按已决 `exit_bar_open_ms`（结案日，无未结）。
- **`--portfolio-sim`**：逐根状态机（同向忽略、反向先平再开、满 N 小时未触轨强平、窗口末强平），
  固定 `margin×leverage` 名义用 `_pnl_usdt`；明细见 `--portfolio-csv`。

用法：
  cd next-k-api
  python zct_vwap_walkforward_backtest.py --days 14 --zct-default-22
  python zct_vwap_walkforward_backtest.py --days 14 --zct-default-22 --use-db-cooldown
  python zct_vwap_walkforward_backtest.py --portfolio-sim --symbols ZECUSDT --days 3 --margin-usdt 100 --leverage 10 --force-flat-hours 6 --portfolio-csv zec_pf.csv
  python zct_vwap_walkforward_backtest.py --days 14 --zct-default-22 --signal-interval 5m
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import zct_vwap_signal_scanner as z

_DEFAULT_SUMMARY_JSON = str(Path(__file__).resolve().parent / "zct_vwap_walkforward_last.json")

_INTERVAL_MS = {
    "1d": 86_400_000,
    "4h": 14_400_000,
    "1h": 3_600_000,
    "15m": 900_000,
}


def _signal_interval_binance(s: str) -> str:
    iv = str(s).strip().lower()
    if iv in ("5m", "5"):
        return "5m"
    if iv in ("1m", "1", ""):
        return "1m"
    raise ValueError(f"unsupported --signal-interval {s!r} (use 1m or 5m)")


def _bar_step_ms(interval: str) -> int:
    return 300_000 if interval == "5m" else 60_000


def _resolve_max_bars_effective(bar_step_ms: int) -> int:
    """使「根数上限」与 1m 下 RESOLVE_MAX_BARS 的墙上时间大致等价（5m 根数更少）。"""
    base = int(z.RESOLVE_MAX_BARS)
    if bar_step_ms <= 60_000:
        return base
    return max(1, int(round(base * 60_000 / float(bar_step_ms))))


# --- monkeypatch：仅 classify 内的 SPIKE ATR 拉线带 endTime（不改扫描器源码）---
_WF_ATR_END_MS: Optional[int] = None
_WF_PATCH_ATR = False
_ORIG_FETCH_KLINES: Any = None


def _wf_fetch_klines(symbol: str, interval: str, limit: int) -> List[Any]:
    if (
        _WF_PATCH_ATR
        and _WF_ATR_END_MS is not None
        and interval == z.SPIKE_ATR_INTERVAL
    ):
        data = z.api_get(
            "/fapi/v1/klines",
            {
                "symbol": str(symbol).strip().upper(),
                "interval": interval,
                "limit": int(limit),
                "endTime": int(_WF_ATR_END_MS),
            },
        )
        return data if isinstance(data, list) else []
    assert _ORIG_FETCH_KLINES is not None
    return _ORIG_FETCH_KLINES(symbol, interval, limit)


def _install_atr_klines_patch() -> None:
    global _ORIG_FETCH_KLINES
    if _ORIG_FETCH_KLINES is None:
        _ORIG_FETCH_KLINES = z.fetch_klines
        z.fetch_klines = _wf_fetch_klines  # type: ignore[method-assign]


def _restore_atr_klines_patch() -> None:
    global _ORIG_FETCH_KLINES
    if _ORIG_FETCH_KLINES is not None:
        z.fetch_klines = _ORIG_FETCH_KLINES  # type: ignore[method-assign]
        _ORIG_FETCH_KLINES = None


def analyze_symbol_pit(
    symbol: str,
    session_1m_raw: pd.DataFrame,
    levels: Dict[str, float],
    *,
    asof_open_ms: int,
    halt_daily_circuit: bool = False,
) -> Optional[z.SignalResult]:
    """
    对齐 `analyze_symbol` 在 classify 之后的闸门链；回测固定不拉 OI。
    `halt_daily_circuit` 与 `run_scan` 传入 `analyze_symbol` 的语义一致。
    `session_1m_raw`：UTC 当日起至 asof 的信号周期 K 线（与 `--signal-interval` 一致，列含 open_time/ts/OHLC）。
    """
    sdf0 = session_1m_raw.copy()
    if sdf0.empty or len(sdf0) < 30:
        return None
    sdf = z.compute_vwap_bands_session(sdf0, z.BAND_SIGMA)
    # 回测不含 OI / 流动性分支（不拉 openInterestHist；与扫描器 analyze_symbol 的 liq 段 intentionally 省略）
    global _WF_ATR_END_MS, _WF_PATCH_ATR
    _WF_ATR_END_MS = int(asof_open_ms)
    _WF_PATCH_ATR = True
    try:
        res = z.classify_and_signal(symbol, sdf, levels)
    finally:
        _WF_ATR_END_MS = None
        _WF_PATCH_ATR = False
    entry_ms = int(sdf.iloc[-1]["open_time"])
    sl, tp, ru = z.compute_sl_tp(res, sdf)
    if res.side in ("LONG", "SHORT"):
        res = replace(
            res,
            entry_bar_open_ms=entry_ms,
            sl_price=sl,
            tp_price=tp,
            r_unit=ru,
        )
    else:
        res = replace(
            res,
            entry_bar_open_ms=None,
            sl_price=None,
            tp_price=None,
            r_unit=None,
        )
    if (
        z.ENFORCE_SETUP_LEVEL
        and res.side in ("LONG", "SHORT")
        and res.setup_level < z.MIN_SETUP_LEVEL_FOR_SIDE
    ):
        res = replace(
            res,
            side="FLAT",
            play="NO_TRADE",
            confidence="low",
            reasons=res.reasons
            + [
                f"已应用 ZCT_ENFORCE_SETUP_LEVEL：setup_level={res.setup_level} < {z.MIN_SETUP_LEVEL_FOR_SIDE}"
                f"{'（海报 level 3+ 档）' if z.MIN_SETUP_LEVEL_FOR_SIDE >= 3 else ''}，方向单已抑制",
            ],
            sl_price=None,
            tp_price=None,
            r_unit=None,
            entry_bar_open_ms=None,
            paper_notional_usdt=None,
        )
    if (
        res.side in ("LONG", "SHORT")
        and z.MAX_BAND_WIDTH_PCT > 0
        and res.band_width_pct > z.MAX_BAND_WIDTH_PCT
    ):
        res = replace(
            res,
            side="FLAT",
            play="NO_TRADE",
            confidence="low",
            reasons=res.reasons
            + [
                f"P2 波动过滤：band_width_pct={res.band_width_pct:.4f} > MAX_BAND_WIDTH_PCT={z.MAX_BAND_WIDTH_PCT}",
            ],
            sl_price=None,
            tp_price=None,
            r_unit=None,
            entry_bar_open_ms=None,
            paper_notional_usdt=None,
        )
    if res.side in ("LONG", "SHORT") and z._cooldown_blocks(symbol):
        res = replace(
            res,
            side="FLAT",
            play="NO_TRADE",
            confidence="low",
            reasons=res.reasons
            + ["P2 止损冷却：该标的仍在冷却窗口内，跳过新开方向单"],
            sl_price=None,
            tp_price=None,
            r_unit=None,
            entry_bar_open_ms=None,
            paper_notional_usdt=None,
        )
    if halt_daily_circuit and res.side in ("LONG", "SHORT"):
        res = replace(
            res,
            side="FLAT",
            play="NO_TRADE",
            confidence="low",
            reasons=res.reasons
            + [
                f"P1 日损熔断：当日已实现盈亏已达 -{z.MAX_DAILY_LOSS_PCT:.1%}×权益 上限，暂停新开仓",
            ],
            sl_price=None,
            tp_price=None,
            r_unit=None,
            entry_bar_open_ms=None,
            paper_notional_usdt=None,
        )
    if z.BTC_MACRO_FILTER_ENABLED:
        if symbol == "BTCUSDT":
            z._BTC_MACRO_STATE["slope_bps"] = float(res.slope_bps or 0.0)
            z._BTC_MACRO_STATE["chop"] = str(res.chop_score or "high") or "high"
        elif res.side in ("LONG", "SHORT"):
            ok, mreason = z.check_btc_macro_permission(
                float(res.slope_bps or 0.0),
                float(z._BTC_MACRO_STATE["slope_bps"]),
                str(z._BTC_MACRO_STATE.get("chop") or "high"),
                res.side,
                slope_threshold=z.BTC_MACRO_SLOPE_THRESHOLD_BPS,
                rs_min_ratio=z.BTC_MACRO_RS_MIN_RATIO,
                long_fuse_slope_bps=z.BTC_MACRO_LONG_FUSE_SLOPE_BPS,
            )
            if not ok:
                res = replace(
                    res,
                    side="FLAT",
                    play="NO_TRADE",
                    confidence="low",
                    reasons=res.reasons + [mreason],
                    sl_price=None,
                    tp_price=None,
                    r_unit=None,
                    entry_bar_open_ms=None,
                    paper_notional_usdt=None,
                    suggested_limit_entry=None,
                )
    if res.side in ("LONG", "SHORT"):
        lim_hint: Optional[float] = None
        if res.play == "PLAY01_BREAKOUT_LONG" and len(sdf) >= 2:
            lim_hint = float(sdf.iloc[-2]["vwap_upper"])
        elif res.play == "PLAY02_BREAKDOWN_SHORT" and len(sdf) >= 2:
            lim_hint = float(sdf.iloc[-2]["vwap_lower"])
        new_reasons = list(res.reasons)
        if lim_hint is not None:
            new_reasons.append(
                f"执行提示（Koroush Breakout）：第二根确认后可于参考轨挂限价回踩 ≈ {lim_hint:g}"
            )
        res = replace(
            res,
            paper_notional_usdt=z._paper_notional_for_signal(res),
            suggested_limit_entry=lim_hint,
            reasons=new_reasons,
        )
    else:
        res = replace(res, paper_notional_usdt=None, suggested_limit_entry=None)
    return res


def _default_symbol_list() -> List[str]:
    if os.getenv("ZCT_TOUCH_POOL_UNIVERSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return z._symbols_from_env()
    raw = os.getenv("ZCT_VWAP_SYMBOLS", "").strip()
    if raw:
        return [x.strip().upper() for x in raw.split(",") if x.strip()]
    return [x.strip() for x in z._DEFAULT_ZCT_SYMBOLS.split(",") if x.strip()]


def zct_default_22_symbols() -> List[str]:
    """扫描器 `_DEFAULT_ZCT_SYMBOLS` 中的默认 22 个标的（不受 ZCT_VWAP_SYMBOLS 影响）。"""
    return [x.strip() for x in z._DEFAULT_ZCT_SYMBOLS.split(",") if x.strip()]


def _per_symbol_win_stats(
    trades: List[Dict[str, Any]], symbols_ordered: List[str]
) -> Dict[str, Any]:
    """按标的统计：触轨胜率 win/(win+loss)；含 expired / unresolved 计数。"""
    by_sym: Dict[str, Dict[str, Any]] = {}
    tot_w = tot_l = tot_ex = 0
    for sym in symbols_ordered:
        su = str(sym).strip().upper()
        rows = [r for r in trades if str(r.get("symbol", "")).strip().upper() == su]
        w = sum(1 for r in rows if r.get("outcome") == "win")
        l_ = sum(1 for r in rows if r.get("outcome") == "loss")
        ex = sum(1 for r in rows if r.get("outcome") == "expired")
        unr = sum(1 for r in rows if r.get("outcome") is None)
        touch = w + l_
        resolved = w + l_ + ex
        by_sym[su] = {
            "n_trades": len(rows),
            "win": w,
            "loss": l_,
            "expired": ex,
            "unresolved": unr,
            "win_rate_touch_sl_tp": round(w / touch, 6) if touch else None,
            "win_rate_vs_all_resolved": round(w / resolved, 6) if resolved else None,
        }
        tot_w += w
        tot_l += l_
        tot_ex += ex
    touch_all = tot_w + tot_l
    res_all = tot_w + tot_l + tot_ex
    return {
        "by_symbol": by_sym,
        "aggregate_touch_win_rate": round(tot_w / touch_all, 6) if touch_all else None,
        "aggregate_resolved_win_rate": round(tot_w / res_all, 6) if res_all else None,
    }


def _complete_utc_dates_in_walk_window(
    start_ms: int, hist_end_ms: int, bar_step_ms: int
) -> List[str]:
    """walk 窗口 [start_ms, hist_end_ms] 内、整段落入区间的 **完整 UTC 自然日**（从日首根到末日末根 open_time）。"""
    step = int(bar_step_ms)
    out: List[str] = []
    lo = int(start_ms)
    hi = int(hist_end_ms)
    day = pd.Timestamp(lo, unit="ms", tz="UTC").normalize()
    end_cap = pd.Timestamp(hi, unit="ms", tz="UTC") + pd.Timedelta(days=1)
    while day < end_cap:
        ds = int(day.value // 1_000_000)
        day_end_last_open = ds + 86_400_000 - step
        if ds >= lo and day_end_last_open <= hi:
            out.append(day.strftime("%Y-%m-%d"))
        day = day + pd.Timedelta(days=1)
    return out


def _daily_per_symbol_win_stats(
    trades: List[Dict[str, Any]],
    symbols_ordered: List[str],
    *,
    start_ms: int,
    hist_end_ms: int,
    bar_step_ms: int,
) -> Dict[str, Any]:
    """完整 UTC 日：`signal_open_ms` 分桶（发单日）+ 已决按 `exit_bar_open_ms`（结案日）；与 `per_symbol` 同胜率口径。"""
    complete_days = _complete_utc_dates_in_walk_window(
        int(start_ms), int(hist_end_ms), int(bar_step_ms)
    )
    day_set = set(complete_days)

    by_sym_sig: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    by_sym_ex: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for sym in symbols_ordered:
        su = str(sym).strip().upper()
        by_sym_sig[su] = {}
        by_sym_ex[su] = {}
        for r in trades:
            if str(r.get("symbol", "")).strip().upper() != su:
                continue
            ts = r.get("signal_open_ms")
            if ts is not None:
                d = pd.Timestamp(int(ts), unit="ms", tz="UTC").strftime("%Y-%m-%d")
                if d in day_set:
                    by_sym_sig[su].setdefault(d, []).append(r)
            exb = r.get("exit_bar_open_ms")
            oco = r.get("outcome")
            if exb is not None and oco is not None:
                de = pd.Timestamp(int(exb), unit="ms", tz="UTC").strftime("%Y-%m-%d")
                if de in day_set:
                    by_sym_ex[su].setdefault(de, []).append(r)

    def _agg(day_map: Dict[str, List[Dict[str, Any]]], day: str) -> Dict[str, Any]:
        day_tr = day_map.get(day, [])
        w = sum(1 for x in day_tr if x.get("outcome") == "win")
        l_ = sum(1 for x in day_tr if x.get("outcome") == "loss")
        ex = sum(1 for x in day_tr if x.get("outcome") == "expired")
        unr = sum(1 for x in day_tr if x.get("outcome") is None)
        touch = w + l_
        resolved = touch + ex
        return {
            "utc_date": day,
            "n_trades": len(day_tr),
            "win": w,
            "loss": l_,
            "expired": ex,
            "unresolved": unr,
            "win_rate_touch_sl_tp": round(w / touch, 6) if touch else None,
            "win_rate_vs_all_resolved": round(w / resolved, 6) if resolved else None,
        }

    by_symbol_signal: Dict[str, List[Dict[str, Any]]] = {}
    by_symbol_exit: Dict[str, List[Dict[str, Any]]] = {}
    for sym in symbols_ordered:
        su = str(sym).strip().upper()
        by_symbol_signal[su] = [
            _agg(by_sym_sig.get(su, {}), d) for d in complete_days
        ]
        by_symbol_exit[su] = []
        for d in complete_days:
            day_tr = by_sym_ex.get(su, {}).get(d, [])
            w = sum(1 for x in day_tr if x.get("outcome") == "win")
            l_ = sum(1 for x in day_tr if x.get("outcome") == "loss")
            ex = sum(1 for x in day_tr if x.get("outcome") == "expired")
            touch = w + l_
            resolved = touch + ex
            by_symbol_exit[su].append(
                {
                    "utc_date": d,
                    "n_resolved": len(day_tr),
                    "win": w,
                    "loss": l_,
                    "expired": ex,
                    "win_rate_touch_sl_tp": round(w / touch, 6) if touch else None,
                    "win_rate_vs_all_resolved": round(w / resolved, 6)
                    if resolved
                    else None,
                }
            )

    return {
        "utc_dates": complete_days,
        "by_symbol": by_symbol_signal,
        "by_symbol_exit_day": by_symbol_exit,
        "note": (
            "utc_dates=walk 窗口内完整 UTC 自然日（首尾半日已剔除）。"
            "by_symbol=按 signal_open_ms 分桶（发单日，含未结）。"
            "by_symbol_exit_day=仅已决，按 exit_bar_open_ms 分桶（结案日，无 unres）。"
        ),
    }


def _dedupe_adjacent_actionable_trades(
    trades: List[Dict[str, Any]],
    *,
    bar_step_ms: int,
) -> List[Dict[str, Any]]:
    """同一标的：signal_open_ms 连续每 bar_step 的一串 actionable 只保留每串第一笔。"""
    by_sym: Dict[str, List[Dict[str, Any]]] = {}
    for tr in trades:
        s = str(tr.get("symbol", "")).strip().upper()
        by_sym.setdefault(s, []).append(tr)
    out: List[Dict[str, Any]] = []
    step = int(bar_step_ms)
    for sym in sorted(by_sym.keys()):
        rows = sorted(by_sym[sym], key=lambda r: int(r.get("signal_open_ms", 0)))
        i = 0
        while i < len(rows):
            out.append(rows[i])
            j = i + 1
            while j < len(rows) and int(rows[j]["signal_open_ms"]) == int(
                rows[j - 1]["signal_open_ms"]
            ) + step:
                j += 1
            i = j
    return out


def _adjacent_stack_chain_trades(
    trades: List[Dict[str, Any]],
    *,
    min_chain_len: int = 2,
    bar_step_ms: int,
) -> List[Dict[str, Any]]:
    """同一标的：signal_open_ms 相邻每 bar_step 的 actionable 串；仅当串长 ≥ min_chain_len 时，串内**全部**笔计入（连打叠仓）。"""
    by_sym: Dict[str, List[Dict[str, Any]]] = {}
    for tr in trades:
        s = str(tr.get("symbol", "")).strip().upper()
        by_sym.setdefault(s, []).append(tr)
    out: List[Dict[str, Any]] = []
    need = max(2, int(min_chain_len))
    step = int(bar_step_ms)
    for sym in sorted(by_sym.keys()):
        rows = sorted(by_sym[sym], key=lambda r: int(r.get("signal_open_ms", 0)))
        i = 0
        while i < len(rows):
            j = i + 1
            while j < len(rows) and int(rows[j]["signal_open_ms"]) == int(
                rows[j - 1]["signal_open_ms"]
            ) + step:
                j += 1
            chain = rows[i:j]
            if len(chain) >= need:
                out.extend(chain)
            i = j
    return out


def _one_open_until_flat_trades(
    trades: List[Dict[str, Any]],
    *,
    hist_end_open_ms: int,
    bar_step_ms: int,
) -> List[Dict[str, Any]]:
    """每标的单笔持仓：上一笔有 exit_bar 则下一笔 entry_bar 须 >= exit_bar+bar_step；未结则本标的后续全跳过。"""
    by_sym: Dict[str, List[Dict[str, Any]]] = {}
    for tr in trades:
        s = str(tr.get("symbol", "")).strip().upper()
        by_sym.setdefault(s, []).append(tr)
    out: List[Dict[str, Any]] = []
    block_rest = 10**18  # 哨兵：本窗口内不再接受该标的后续 entry
    step = int(bar_step_ms)
    for sym in sorted(by_sym.keys()):
        rows = sorted(
            by_sym[sym],
            key=lambda r: (
                int(r.get("entry_bar_open_ms") or r.get("signal_open_ms") or 0),
                int(r.get("signal_open_ms", 0)),
            ),
        )
        next_allowed_entry = 0
        for tr in rows:
            eb = int(tr.get("entry_bar_open_ms") or tr.get("signal_open_ms") or 0)
            if eb < next_allowed_entry:
                continue
            out.append(tr)
            oco = tr.get("outcome")
            exb = tr.get("exit_bar_open_ms")
            if oco is None or exb is None:
                next_allowed_entry = block_rest
                break
            next_allowed_entry = int(exb) + step
    return out


def _constrained_stats_block(
    trades: List[Dict[str, Any]],
    symbols_ordered: List[str],
    *,
    hist_end_open_ms: int,
    bar_step_ms: int,
    signal_interval: str,
) -> Dict[str, Any]:
    ded = _dedupe_adjacent_actionable_trades(trades, bar_step_ms=bar_step_ms)
    one = _one_open_until_flat_trades(
        trades, hist_end_open_ms=hist_end_open_ms, bar_step_ms=bar_step_ms
    )
    stack = _adjacent_stack_chain_trades(
        trades, min_chain_len=2, bar_step_ms=bar_step_ms
    )
    ps_ded = _per_symbol_win_stats(ded, symbols_ordered)
    ps_one = _per_symbol_win_stats(one, symbols_ordered)
    ps_st = _per_symbol_win_stats(stack, symbols_ordered)
    iv = str(signal_interval).strip().lower()
    return {
        "dedupe_adjacent_actionable": {
            "note": f"同标的连续 {iv} actionable 只保留每链首笔；resolve 结果沿用原笔",
            "trades": len(ded),
            "per_symbol": ps_ded["by_symbol"],
            "aggregate_touch_win_rate": ps_ded["aggregate_touch_win_rate"],
            "aggregate_resolved_win_rate": ps_ded["aggregate_resolved_win_rate"],
        },
        "adjacent_stack_chains_only": {
            "note": f"仅统计连打叠仓：同标的相邻 {iv} actionable 且链长≥2 时，链内每笔都计入",
            "min_chain_len": 2,
            "trades": len(stack),
            "per_symbol": ps_st["by_symbol"],
            "aggregate_touch_win_rate": ps_st["aggregate_touch_win_rate"],
            "aggregate_resolved_win_rate": ps_st["aggregate_resolved_win_rate"],
        },
        "one_open_per_symbol": {
            "note": f"同标未平不开新仓；平仓后下一笔 entry_bar>=exit_bar+{iv}；未结则该标的后继全弃",
            "trades": len(one),
            "per_symbol": ps_one["by_symbol"],
            "aggregate_touch_win_rate": ps_one["aggregate_touch_win_rate"],
            "aggregate_resolved_win_rate": ps_one["aggregate_resolved_win_rate"],
        },
    }


def session_slice_utc_day(full_kline: pd.DataFrame, asof_open_ms: int) -> pd.DataFrame:
    """UTC 日历日 00:00 起至 asof 根（含）的信号 K 线子集（对齐会话 VWAP 定义）。"""
    if full_kline.empty:
        return full_kline
    t = pd.Timestamp(int(asof_open_ms), unit="ms", tz="UTC")
    day0 = t.normalize()
    return full_kline[
        (full_kline["open_time"] <= int(asof_open_ms)) & (full_kline["ts"] >= day0)
    ].copy()


def load_kline_range(
    symbol: str, interval: str, start_ms: int, end_ms: int
) -> pd.DataFrame:
    rows = z.fetch_klines_forward(symbol, interval, start_ms, end_ms)
    df = z.klines_to_df(rows)
    if df.empty:
        return df
    return (
        df.drop_duplicates(subset=["open_time"], keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def _preload_iv(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = z.fetch_klines_forward(symbol, interval, start_ms, end_ms)
    return z.klines_to_df(rows)


class RefLevelResolver:
    """预拉多周期 K，在本地按 ref_levels 语义给出 asof 时刻的关键位（无未来函数）。"""

    def __init__(self, symbol: str, start_ms: int, end_ms: int) -> None:
        self.symbol = str(symbol).strip().upper()
        pad = 120 * 86_400_000
        s0 = int(start_ms) - pad
        e0 = int(end_ms)
        self._d1 = _preload_iv(self.symbol, "1d", s0, e0)
        self._h4 = _preload_iv(self.symbol, "4h", s0, e0)
        self._h1 = _preload_iv(self.symbol, "1h", s0, e0)
        self._m15 = _preload_iv(self.symbol, "15m", s0, e0)

    def levels(self, asof_open_ms: int) -> Dict[str, float]:
        out: Dict[str, float] = {}
        t = int(asof_open_ms)

        def _tail_closed(df: pd.DataFrame, dur_ms: int, n: int) -> pd.DataFrame:
            if df is None or df.empty or "open_time" not in df.columns:
                return pd.DataFrame()
            ot = df["open_time"].astype("int64")
            closed = df.loc[ot + int(dur_ms) <= t]
            return closed.tail(int(n))

        d1t = _tail_closed(self._d1, _INTERVAL_MS["1d"], 3)
        if len(d1t) >= 2:
            prev = d1t.iloc[-2]
            out["pdh"] = float(prev["high"])
            out["pdl"] = float(prev["low"])
        for iv_df, dur_ms, pfx in (
            (self._h4, _INTERVAL_MS["4h"], "h4"),
            (self._h1, _INTERVAL_MS["1h"], "h1"),
            (self._m15, _INTERVAL_MS["15m"], "m15"),
        ):
            tt = _tail_closed(iv_df, dur_ms, 4)
            if len(tt) >= 2:
                prev_bar = tt.iloc[-2]
                out[f"{pfx}_high"] = float(prev_bar["high"])
                out[f"{pfx}_low"] = float(prev_bar["low"])
        return out


def resolve_forward(
    df_kline: pd.DataFrame,
    *,
    entry: float,
    entry_bar_open_ms: int,
    side: str,
    sl: float,
    tp: float,
    hist_end_ms: int,
    bar_step_ms: int,
) -> Tuple[Optional[str], float, str, int, Optional[int]]:
    """对齐 resolve_open_signals_from_db 的逐根推进；首根为 entry 的下一根；返回 exit_bar_open_ms 为触轨/过期所在根 open_time。"""
    step = int(bar_step_ms)
    start_ms = int(entry_bar_open_ms) + step
    max_bars = _resolve_max_bars_effective(step)
    if start_ms > hist_end_ms:
        return None, float(entry), "start_after_hist_end", 0, None
    deadline_ms = (
        int(entry_bar_open_ms) + int(z.RESOLVE_MAX_HOLD_MS)
        if z.RESOLVE_MAX_HOLD_MS > 0
        else None
    )
    outcome: Optional[str] = None
    exit_px = float(entry)
    note = "resolved:auto"
    bars_seen = 0
    exit_bar_open_ms: Optional[int] = None
    sub = df_kline[df_kline["open_time"] >= start_ms].sort_values("open_time")
    for _, row in sub.iterrows():
        bo = int(row["open_time"])
        if bo > hist_end_ms:
            break
        o = float(row["open"])
        h = float(row["high"])
        low = float(row["low"])
        c = float(row["close"])
        bars_seen += 1
        if side == "LONG":
            tag, px = z._bar_hit_long(o, h, low, sl, tp)
        else:
            tag, px = z._bar_hit_short(o, h, low, sl, tp)
        if tag == "win":
            outcome = "win"
            exit_px = float(px)
            exit_bar_open_ms = bo
            break
        if tag == "loss":
            outcome = "loss"
            exit_px = float(px)
            exit_bar_open_ms = bo
            break
        if deadline_ms is not None and bo >= deadline_ms:
            outcome = "expired"
            exit_px = c
            note = "resolved:auto_expired_wall_clock"
            exit_bar_open_ms = bo
            break
        if bars_seen >= max_bars:
            outcome = "expired"
            exit_px = c
            note = f"resolved:auto_expired_after_{max_bars}_bars"
            exit_bar_open_ms = bo
            break
    if outcome is None:
        return None, float(entry), "no_sl_tp_touch_within_window", bars_seen, None
    return outcome, exit_px, note, bars_seen, exit_bar_open_ms


def _portfolio_closed_row(
    pos: Dict[str, Any],
    *,
    exit_reason: str,
    outcome: str,
    exit_bar_open_ms: int,
    exit_price: float,
    notional_usdt: float,
) -> Dict[str, Any]:
    """单行平仓明细（含固定杠杆名义下的 USDT 盈亏）。"""
    side = str(pos["side"])
    entry = float(pos["entry"])
    sl = float(pos["sl"])
    tp = float(pos["tp"])
    ex = float(exit_price)
    pnl_r = z._pnl_r(side, entry, ex, sl, tp)
    pnl_u = z._pnl_usdt(side, entry, ex, notional_usdt)
    return {
        "symbol": pos["symbol"],
        "side": side,
        "play": pos.get("play"),
        "signal_open_ms": int(pos["signal_open_ms"]),
        "entry_bar_open_ms": int(pos["entry_bar_open_ms"]),
        "entry_price": entry,
        "sl": sl,
        "tp": tp,
        "exit_bar_open_ms": int(exit_bar_open_ms),
        "exit_price": round(ex, 8),
        "outcome": outcome,
        "exit_reason": exit_reason,
        "hold_bars": int(pos.get("bars_held", 0)),
        "margin_usdt": float(pos["margin_usdt"]),
        "leverage": float(pos["leverage"]),
        "notional_usdt": float(pos["notional_usdt"]),
        "pnl_r": round(float(pnl_r), 6),
        "pnl_usdt": round(float(pnl_u), 4),
    }


def _portfolio_advance(
    df_kline: pd.DataFrame,
    pos: Dict[str, Any],
    until_bo_inclusive: int,
    hist_end_ms: int,
    time_stop_ms: int,
    notional_usdt: float,
    *,
    bar_step_ms: int,
    resolve_max_bars_effective: int,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    从 `resume_from_open_ms` 起逐根推进到 `until_bo_inclusive`（含）：
    先 SL/TP（同根规则与 `resolve_forward` 一致），再 6h 时间强平，再 max_bars（按信号周期缩放根数上限）。
    未触发则更新 `resume_from_open_ms = until_bo_inclusive + bar_step_ms`。
    """
    fills: List[Dict[str, Any]] = []
    cur = int(pos["resume_from_open_ms"])
    entry_bo = int(pos["entry_bar_open_ms"])
    deadline_bo = entry_bo + int(time_stop_ms)
    until_bo = min(int(until_bo_inclusive), int(hist_end_ms))
    step = int(bar_step_ms)
    max_bars = int(resolve_max_bars_effective)

    sub = df_kline[
        (df_kline["open_time"] >= cur)
        & (df_kline["open_time"] <= until_bo)
    ].sort_values("open_time")

    for _, row in sub.iterrows():
        bo = int(row["open_time"])
        o = float(row["open"])
        h = float(row["high"])
        low = float(row["low"])
        c = float(row["close"])
        pos["bars_held"] = int(pos.get("bars_held", 0)) + 1
        side = str(pos["side"])
        sl = float(pos["sl"])
        tp = float(pos["tp"])
        if side == "LONG":
            tag, px = z._bar_hit_long(o, h, low, sl, tp)
        else:
            tag, px = z._bar_hit_short(o, h, low, sl, tp)
        if tag == "win":
            fills.append(
                _portfolio_closed_row(
                    pos,
                    exit_reason="tp_sl_touch",
                    outcome="win",
                    exit_bar_open_ms=bo,
                    exit_price=float(px),
                    notional_usdt=notional_usdt,
                )
            )
            return None, fills
        if tag == "loss":
            fills.append(
                _portfolio_closed_row(
                    pos,
                    exit_reason="tp_sl_touch",
                    outcome="loss",
                    exit_bar_open_ms=bo,
                    exit_price=float(px),
                    notional_usdt=notional_usdt,
                )
            )
            return None, fills
        if bo >= deadline_bo:
            fills.append(
                _portfolio_closed_row(
                    pos,
                    exit_reason="force_time_6h",
                    outcome="force_time",
                    exit_bar_open_ms=bo,
                    exit_price=c,
                    notional_usdt=notional_usdt,
                )
            )
            return None, fills
        if int(pos["bars_held"]) >= max_bars:
            fills.append(
                _portfolio_closed_row(
                    pos,
                    exit_reason="max_bars",
                    outcome="expired",
                    exit_bar_open_ms=bo,
                    exit_price=c,
                    notional_usdt=notional_usdt,
                )
            )
            return None, fills

    pos["resume_from_open_ms"] = int(until_bo) + step
    return pos, fills


def _portfolio_open_from_signal(
    sym: str,
    res: Any,
    *,
    signal_open_ms: int,
    margin_usdt: float,
    leverage: float,
    notional_usdt: float,
    bar_step_ms: int,
) -> Dict[str, Any]:
    eb = int(res.entry_bar_open_ms or signal_open_ms)
    step = int(bar_step_ms)
    return {
        "symbol": sym,
        "side": str(res.side),
        "play": res.play,
        "signal_open_ms": int(signal_open_ms),
        "entry_bar_open_ms": eb,
        "entry": float(res.price),
        "sl": float(res.sl_price or 0.0),
        "tp": float(res.tp_price or 0.0),
        "resume_from_open_ms": eb + step,
        "bars_held": 0,
        "margin_usdt": float(margin_usdt),
        "leverage": float(leverage),
        "notional_usdt": float(notional_usdt),
    }


def run_portfolio_backtest(
    *,
    days: float,
    symbols: List[str],
    ignore_db_cooldown: bool,
    sleep_between_symbols: float,
    json_summary_path: Optional[str],
    portfolio_csv: str,
    margin_usdt: float,
    leverage: float,
    force_flat_hours: float,
    signal_interval: str = "1m",
) -> Dict[str, Any]:
    """
    单账户规则（每标的独立状态机）：
    - 已有持仓且新信号同向 → 忽略；
    - 已有持仓且新信号反向 → 当前根收盘价强平，并同价开新仓；
    - 持仓满 `force_flat_hours`（自 entry_bar_open_ms 起算，按信号 K 线根的 open_time）仍未触轨 → 该根收盘价强平；
    - 固定保证金×杠杆 → 线性 PnL 名义 `notional = margin * leverage`（与 `_pnl_usdt` 一致）。
    """
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(float(days) * 86_400_000)
    iv = _signal_interval_binance(signal_interval)
    bar_step_ms = _bar_step_ms(iv)
    resolve_max_bars_effective = _resolve_max_bars_effective(bar_step_ms)
    fetch_start_ms = _padded_1m_fetch_start_ms(start_ms, end_ms)
    time_stop_ms = int(float(force_flat_hours) * 3_600_000)
    notional_usdt = float(margin_usdt) * float(leverage)

    _orig_btc_macro = z.BTC_MACRO_FILTER_ENABLED
    _orig_cooldown = z._cooldown_blocks
    try:
        z.BTC_MACRO_FILTER_ENABLED = False
        if ignore_db_cooldown:
            z._cooldown_blocks = lambda _sym: False  # type: ignore[method-assign]
        _install_atr_klines_patch()

        dfs: Dict[str, pd.DataFrame] = {}
        resolvers: Dict[str, RefLevelResolver] = {}
        syms = [s.strip().upper() for s in symbols if s.strip()]

        print(
            f"[pf] range symbols={len(syms)} signal_interval={iv} user_start_ms={start_ms} end_ms={end_ms} "
            f"kline_fetch_start_ms={fetch_start_ms} bar_step_ms={bar_step_ms} | margin={margin_usdt} lev={leverage} "
            f"notional={notional_usdt} force_flat_h={force_flat_hours}",
            flush=True,
        )
        for sym in syms:
            if sleep_between_symbols > 0:
                time.sleep(sleep_between_symbols)
            dfs[sym] = load_kline_range(sym, iv, fetch_start_ms, end_ms)
            resolvers[sym] = RefLevelResolver(sym, fetch_start_ms, end_ms)
            print(f"[pf] loaded {sym} {iv} rows={len(dfs[sym])}", flush=True)

        last_bars = [
            int(dfs[s]["open_time"].iloc[-1])
            for s in syms
            if dfs.get(s) is not None and not dfs[s].empty
        ]
        hist_end_open_ms_max = max(last_bars) if last_bars else int(end_ms)
        if last_bars:
            print(
                "[pf] kline_last_open_ms "
                f"min={min(last_bars)} max={max(last_bars)} "
                "(per-symbol advance cap; JSON hist_end_open_ms=max)",
                flush=True,
            )

        all_fills: List[Dict[str, Any]] = []
        meta_by_sym: Dict[str, Any] = {}

        for sym in syms:
            fills_before_sym = len(all_fills)
            df_full = dfs.get(sym)
            if df_full is None or df_full.empty:
                continue
            df_loop = df_full[df_full["open_time"] >= int(start_ms)].reset_index(
                drop=True
            )
            if df_loop.empty:
                continue
            rr = resolvers[sym]
            hist_end_sym = int(df_full["open_time"].iloc[-1])
            pos: Optional[Dict[str, Any]] = None
            ignored_same = 0
            signals_action = 0
            signals_flat = 0

            if sleep_between_symbols > 0:
                time.sleep(sleep_between_symbols)

            for i in range(len(df_loop)):
                t = int(df_loop.iloc[i]["open_time"])
                if t > hist_end_sym:
                    break

                row_t = df_loop.iloc[i]
                if pos is not None:
                    pos, adv_fills = _portfolio_advance(
                        df_full,
                        pos,
                        until_bo_inclusive=t,
                        hist_end_ms=hist_end_sym,
                        time_stop_ms=time_stop_ms,
                        notional_usdt=notional_usdt,
                        bar_step_ms=bar_step_ms,
                        resolve_max_bars_effective=resolve_max_bars_effective,
                    )
                    all_fills.extend(adv_fills)

                alt_sess = session_slice_utc_day(df_full, t)
                if len(alt_sess) < 30:
                    continue
                lv = rr.levels(t)
                res = analyze_symbol_pit(
                    sym,
                    alt_sess,
                    lv,
                    asof_open_ms=t,
                    halt_daily_circuit=False,
                )
                if res is None:
                    continue
                if (
                    res.side in ("LONG", "SHORT")
                    and res.sl_price is not None
                    and res.tp_price is not None
                ):
                    signals_action += 1
                else:
                    signals_flat += 1

                if (
                    res.side not in ("LONG", "SHORT")
                    or res.sl_price is None
                    or res.tp_price is None
                    or res.entry_bar_open_ms is None
                ):
                    continue

                if pos is None:
                    pos = _portfolio_open_from_signal(
                        sym,
                        res,
                        signal_open_ms=t,
                        margin_usdt=margin_usdt,
                        leverage=leverage,
                        notional_usdt=notional_usdt,
                        bar_step_ms=bar_step_ms,
                    )
                elif str(pos["side"]) == str(res.side):
                    ignored_same += 1
                else:
                    ex_px = float(row_t["close"])
                    all_fills.append(
                        _portfolio_closed_row(
                            pos,
                            exit_reason="reverse_signal_flat",
                            outcome="reverse_flat",
                            exit_bar_open_ms=t,
                            exit_price=ex_px,
                            notional_usdt=notional_usdt,
                        )
                    )
                    pos = _portfolio_open_from_signal(
                        sym,
                        res,
                        signal_open_ms=t,
                        margin_usdt=margin_usdt,
                        leverage=leverage,
                        notional_usdt=notional_usdt,
                        bar_step_ms=bar_step_ms,
                    )

            if pos is not None:
                last_bo = int(
                    df_full[df_full["open_time"] <= hist_end_sym]["open_time"].max()
                )
                tail = df_full[df_full["open_time"] == last_bo]
                last_c = float(tail.iloc[-1]["close"]) if not tail.empty else float(
                    pos["entry"]
                )
                pos, tail_fills = _portfolio_advance(
                    df_full,
                    pos,
                    until_bo_inclusive=last_bo,
                    hist_end_ms=hist_end_sym,
                    time_stop_ms=time_stop_ms,
                    notional_usdt=notional_usdt,
                    bar_step_ms=bar_step_ms,
                    resolve_max_bars_effective=resolve_max_bars_effective,
                )
                all_fills.extend(tail_fills)
                if pos is not None:
                    all_fills.append(
                        _portfolio_closed_row(
                            pos,
                            exit_reason="hist_end_window",
                            outcome="hist_end",
                            exit_bar_open_ms=last_bo,
                            exit_price=last_c,
                            notional_usdt=notional_usdt,
                        )
                    )
                    pos = None

            meta_by_sym[sym] = {
                "closed_trades": len(all_fills) - int(fills_before_sym),
                "signals_actionable": signals_action,
                "signals_rows": signals_action + signals_flat,
                "ignored_same_side_while_open": ignored_same,
            }

        pnl_sum = sum(float(x.get("pnl_usdt") or 0.0) for x in all_fills)
        wins = sum(1 for x in all_fills if x.get("outcome") == "win")
        losses = sum(1 for x in all_fills if x.get("outcome") == "loss")
        outcome_breakdown: Dict[str, int] = {}
        for x in all_fills:
            k = str(x.get("outcome") or "")
            outcome_breakdown[k] = int(outcome_breakdown.get(k, 0)) + 1

        summary: Dict[str, Any] = {
            "mode": "portfolio_sim",
            "days": days,
            "signal_interval": iv,
            "bar_step_ms": int(bar_step_ms),
            "user_start_open_ms": int(start_ms),
            "kline_fetch_start_ms": int(fetch_start_ms),
            "hist_end_open_ms": int(hist_end_open_ms_max),
            "symbols": syms,
            "margin_usdt": float(margin_usdt),
            "leverage": float(leverage),
            "notional_usdt": float(notional_usdt),
            "force_flat_hours": float(force_flat_hours),
            "force_flat_ms": int(time_stop_ms),
            "closed_trades": len(all_fills),
            "touch_win": wins,
            "touch_loss": losses,
            "outcome_breakdown": outcome_breakdown,
            "sum_pnl_usdt": round(pnl_sum, 4),
            "same_bar_rule": z.SAME_BAR_RULE,
            "resolve_max_bars": z.RESOLVE_MAX_BARS,
            "resolve_max_bars_effective": int(resolve_max_bars_effective),
            "per_symbol_meta": meta_by_sym,
            "portfolio_csv": portfolio_csv,
            "backtest_assumptions": {
                "liquidity_oi_filter": False,
                "btc_macro_forced_off": True,
                "cooldown_uses_db": not ignore_db_cooldown,
                "daily_loss_halt_per_run": False,
                "daily_loss_halt_note": "walk-forward/portfolio 回测不启用日损熔断（halt_daily_circuit 恒为 False）",
            },
        }

        cols = [
            "trade_seq",
            "symbol",
            "side",
            "play",
            "signal_open_ms",
            "entry_bar_open_ms",
            "entry_price",
            "sl",
            "tp",
            "exit_bar_open_ms",
            "exit_price",
            "outcome",
            "exit_reason",
            "hold_bars",
            "margin_usdt",
            "leverage",
            "notional_usdt",
            "pnl_r",
            "pnl_usdt",
        ]
        outp_csv = Path(portfolio_csv)
        outp_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(outp_csv, "w", newline="", encoding="utf-8") as fc:
            wr = csv.DictWriter(fc, fieldnames=cols)
            wr.writeheader()
            for seq, row in enumerate(all_fills, start=1):
                r2 = dict(row)
                r2["trade_seq"] = seq
                wr.writerow({k: r2.get(k) for k in cols})

        print(f"[pf] trade detail csv -> {outp_csv.resolve()}", flush=True)

        if json_summary_path:
            outp = Path(json_summary_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", encoding="utf-8") as fj:
                json.dump(summary, fj, ensure_ascii=False, indent=2)
            print(f"[pf] summary json -> {outp.resolve()}", flush=True)

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        _restore_atr_klines_patch()
        z.BTC_MACRO_FILTER_ENABLED = _orig_btc_macro
        z._cooldown_blocks = _orig_cooldown  # type: ignore[method-assign]


def _utc_day_floor_ms(ms: int) -> int:
    t = pd.Timestamp(int(ms), unit="ms", tz="UTC").floor("D")
    return int(t.value // 1_000_000)


def _padded_1m_fetch_start_ms(start_ms: int, end_ms: int) -> int:
    """会话 VWAP 需当日从 UTC 0 点起的 K 线；只拉 [start_ms,end] 时若 start 在日中会导致 session 根数恒 <30。"""
    lo = _utc_day_floor_ms(min(int(start_ms), int(end_ms)))
    return min(int(start_ms), lo)


def run_backtest(
    *,
    days: float,
    symbols: List[str],
    ignore_db_cooldown: bool,
    csv_path: Optional[str],
    sleep_between_symbols: float,
    json_summary_path: Optional[str],
    signal_interval: str = "1m",
    emit_text_report: bool = True,
) -> Dict[str, Any]:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(float(days) * 86_400_000)
    iv = _signal_interval_binance(signal_interval)
    bar_step_ms = _bar_step_ms(iv)
    resolve_max_bars_effective = _resolve_max_bars_effective(bar_step_ms)
    fetch_start_ms = _padded_1m_fetch_start_ms(start_ms, end_ms)

    _orig_btc_macro = z.BTC_MACRO_FILTER_ENABLED
    _orig_cooldown = z._cooldown_blocks
    try:
        z.BTC_MACRO_FILTER_ENABLED = False
        if ignore_db_cooldown:
            z._cooldown_blocks = lambda _sym: False  # type: ignore[method-assign]
        _install_atr_klines_patch()

        dfs: Dict[str, pd.DataFrame] = {}
        resolvers: Dict[str, RefLevelResolver] = {}
        syms = [s.strip().upper() for s in symbols if s.strip()]

        if emit_text_report:
            print(
                f"[bt] range symbols={len(syms)} signal_interval={iv} user_start_ms={start_ms} end_ms={end_ms} "
                f"kline_fetch_start_ms={fetch_start_ms} bar_step_ms={bar_step_ms} (UTC day floor for session VWAP)",
                flush=True,
            )
            print(
                "[bt] policy: liquidity_oi=off | btc_macro=forced_off (env was "
                f"{_orig_btc_macro}) | cooldown_db="
                f"{'off' if ignore_db_cooldown else 'on'} | daily_halt=off "
                "(walk-forward: 不启用日损熔断)",
                flush=True,
            )
        for sym in syms:
            if sleep_between_symbols > 0:
                time.sleep(sleep_between_symbols)
            dfs[sym] = load_kline_range(sym, iv, fetch_start_ms, end_ms)
            resolvers[sym] = RefLevelResolver(sym, fetch_start_ms, end_ms)
            if emit_text_report:
                print(f"[bt] loaded {sym} {iv} rows={len(dfs[sym])}", flush=True)

        last_bars = [
            int(dfs[s]["open_time"].iloc[-1])
            for s in syms
            if dfs.get(s) is not None and not dfs[s].empty
        ]
        hist_end_open_ms_max = max(last_bars) if last_bars else int(end_ms)
        if emit_text_report and last_bars:
            print(
                "[bt] kline_last_open_ms "
                f"min={min(last_bars)} max={max(last_bars)} "
                "(walk/resolve 每标的 cap 于自身末根；JSON hist_end_open_ms=max)",
                flush=True,
            )
        trades: List[Dict[str, Any]] = []
        signals_flat = 0
        signals_action = 0
        unresolved = 0

        for sym in syms:
            df_full = dfs.get(sym)
            if df_full is None or df_full.empty:
                continue
            df_loop = df_full[df_full["open_time"] >= int(start_ms)].reset_index(
                drop=True
            )
            if df_loop.empty:
                continue
            rr = resolvers[sym]
            hist_end_sym = int(df_full["open_time"].iloc[-1])
            if sleep_between_symbols > 0:
                time.sleep(sleep_between_symbols)
            for i in range(len(df_loop)):
                t = int(df_loop.iloc[i]["open_time"])
                if t > hist_end_sym:
                    break
                alt_sess = session_slice_utc_day(df_full, t)
                if len(alt_sess) < 30:
                    continue
                lv = rr.levels(t)
                res = analyze_symbol_pit(
                    sym,
                    alt_sess,
                    lv,
                    asof_open_ms=t,
                    halt_daily_circuit=False,
                )

                if res is None:
                    continue
                if (
                    res.side in ("LONG", "SHORT")
                    and res.sl_price is not None
                    and res.tp_price is not None
                ):
                    signals_action += 1
                else:
                    signals_flat += 1

                if (
                    res.side not in ("LONG", "SHORT")
                    or res.sl_price is None
                    or res.tp_price is None
                    or res.entry_bar_open_ms is None
                ):
                    continue

                out, ex_px, note, bars_seen, exit_bo = resolve_forward(
                    df_full,
                    entry=float(res.price),
                    entry_bar_open_ms=int(res.entry_bar_open_ms),
                    side=str(res.side),
                    sl=float(res.sl_price),
                    tp=float(res.tp_price),
                    hist_end_ms=hist_end_sym,
                    bar_step_ms=bar_step_ms,
                )
                notion = float(res.paper_notional_usdt or z.VIRTUAL_NOTIONAL_USDT)
                eb_ms = int(res.entry_bar_open_ms)
                if out is None:
                    unresolved += 1
                    trades.append(
                        {
                            "symbol": sym,
                            "signal_open_ms": t,
                            "entry_bar_open_ms": eb_ms,
                            "exit_bar_open_ms": None,
                            "side": res.side,
                            "play": res.play,
                            "entry": res.price,
                            "sl": res.sl_price,
                            "tp": res.tp_price,
                            "outcome": None,
                            "exit_price": None,
                            "pnl_r": None,
                            "pnl_usdt": None,
                            "resolve_note": note,
                            "bars_seen": bars_seen,
                        }
                    )
                    continue
                pnl_r = z._pnl_r(
                    res.side,
                    float(res.price),
                    ex_px,
                    float(res.sl_price),
                    float(res.tp_price),
                )
                pnl_u = z._pnl_usdt(res.side, float(res.price), ex_px, notion)
                trades.append(
                    {
                        "symbol": sym,
                        "signal_open_ms": t,
                        "entry_bar_open_ms": eb_ms,
                        "exit_bar_open_ms": exit_bo,
                        "side": res.side,
                        "play": res.play,
                        "entry": res.price,
                        "sl": res.sl_price,
                        "tp": res.tp_price,
                        "outcome": out,
                        "exit_price": ex_px,
                        "pnl_r": round(pnl_r, 6),
                        "pnl_usdt": round(pnl_u, 4),
                        "resolve_note": note,
                        "bars_seen": bars_seen,
                    }
                )

        resolved = [x for x in trades if x.get("outcome")]
        wins = sum(1 for x in resolved if x["outcome"] == "win")
        losses = sum(1 for x in resolved if x["outcome"] == "loss")
        exp = sum(1 for x in resolved if x["outcome"] == "expired")
        pnl_sum = sum(float(x["pnl_usdt"] or 0) for x in resolved)

        per_sym = _per_symbol_win_stats(trades, syms)
        daily_sym = _daily_per_symbol_win_stats(
            trades,
            syms,
            start_ms=int(start_ms),
            hist_end_ms=int(hist_end_open_ms_max),
            bar_step_ms=int(bar_step_ms),
        )
        pos_con = _constrained_stats_block(
            trades,
            syms,
            hist_end_open_ms=int(hist_end_open_ms_max),
            bar_step_ms=bar_step_ms,
            signal_interval=iv,
        )

        summary: Dict[str, Any] = {
            "days": days,
            "signal_interval": iv,
            "bar_step_ms": int(bar_step_ms),
            "user_start_open_ms": int(start_ms),
            "kline_fetch_start_ms": int(fetch_start_ms),
            "hist_end_open_ms": int(hist_end_open_ms_max),
            "symbols": syms,
            "signals_actionable": signals_action,
            "signals_rows": signals_action + signals_flat,
            "trades_emitted": len(trades),
            "resolved_count": len(resolved),
            "unresolved_still_open": unresolved,
            "win": wins,
            "loss": losses,
            "expired": exp,
            "sum_pnl_usdt": round(pnl_sum, 4),
            "same_bar_rule": z.SAME_BAR_RULE,
            "resolve_max_bars": z.RESOLVE_MAX_BARS,
            "resolve_max_bars_effective": int(resolve_max_bars_effective),
            "resolve_max_hold_ms": z.RESOLVE_MAX_HOLD_MS,
            "btc_macro_filter_effective": False,
            "backtest_assumptions": {
                "liquidity_oi_filter": False,
                "btc_macro_forced_off": True,
                "btc_macro_env_before_run": _orig_btc_macro,
                "cooldown_uses_db": not ignore_db_cooldown,
                "daily_loss_halt_per_run": False,
                "daily_loss_halt_note": "walk-forward：不启用日损熔断（halt_daily_circuit 恒为 False）",
            },
            "per_symbol": per_sym["by_symbol"],
            "per_symbol_daily": daily_sym,
            "aggregate_touch_win_rate": per_sym["aggregate_touch_win_rate"],
            "aggregate_resolved_win_rate": per_sym["aggregate_resolved_win_rate"],
            "position_constraints": pos_con,
        }

        if csv_path:
            cols = [
                "symbol",
                "signal_open_ms",
                "entry_bar_open_ms",
                "exit_bar_open_ms",
                "side",
                "play",
                "entry",
                "sl",
                "tp",
                "outcome",
                "exit_price",
                "pnl_r",
                "pnl_usdt",
                "resolve_note",
                "bars_seen",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for row in trades:
                    w.writerow({k: row.get(k) for k in cols})
            summary["csv"] = csv_path

        if json_summary_path:
            outp = Path(json_summary_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", encoding="utf-8") as fj:
                json.dump(summary, fj, ensure_ascii=False, indent=2)
            if emit_text_report:
                print(f"[bt] summary json -> {outp.resolve()}", flush=True)

        if emit_text_report:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(
                "\n[bt] per-symbol 触轨胜率 win/(win+loss)（SL/TP 先触发；同根规则见 ZCT_SAME_BAR_RULE）",
                flush=True,
            )
            for sym in syms:
                row = per_sym["by_symbol"].get(str(sym).strip().upper(), {})
                wr = row.get("win_rate_touch_sl_tp")
                wrs = f"{100.0 * float(wr):.2f}%" if wr is not None else "n/a"
                print(
                    f"  {sym:16s} {wrs:>8s}  "
                    f"w={row.get('win', 0)} L={row.get('loss', 0)} "
                    f"exp={row.get('expired', 0)} unres={row.get('unresolved', 0)}",
                    flush=True,
                )
            aw = per_sym["aggregate_touch_win_rate"]
            ar = per_sym["aggregate_resolved_win_rate"]
            touch_s = f"{100.0 * float(aw):.2f}%" if aw is not None else "n/a"
            res_s = f"{100.0 * float(ar):.2f}%" if ar is not None else "n/a"
            print(
                f"  {'ALL(agg)':16s}  touch_win_rate={touch_s}  win_vs_all_resolved={res_s}",
                flush=True,
            )
            uds = daily_sym.get("utc_dates") or []
            if uds:
                print(
                    "\n[bt] 完整 UTC 日 | 发单日 signal_open_ms（by_symbol；含未结）",
                    flush=True,
                )
                print(f"  ({daily_sym.get('note', '')})", flush=True)
                for di, day in enumerate(uds):
                    print(f"  --- {day} (UTC) ---", flush=True)
                    for sym in syms:
                        su = str(sym).strip().upper()
                        row = daily_sym["by_symbol"].get(su, [])[di]
                        tw = row.get("win_rate_touch_sl_tp")
                        vr = row.get("win_rate_vs_all_resolved")
                        tws = f"{100.0 * float(tw):.2f}%" if tw is not None else "n/a"
                        vrs = f"{100.0 * float(vr):.2f}%" if vr is not None else "n/a"
                        print(
                            f"    {sym:16s} n={row.get('n_trades', 0)}  "
                            f"w/L/exp/unr={row.get('win')}/{row.get('loss')}/"
                            f"{row.get('expired')}/{row.get('unresolved')}  "
                            f"touch={tws}  vs_resolved={vrs}",
                            flush=True,
                        )
                exmap = daily_sym.get("by_symbol_exit_day") or {}
                if exmap:
                    print(
                        "\n[bt] 完整 UTC 日 | 结案日 exit_bar_open_ms（仅已决；by_symbol_exit_day）",
                        flush=True,
                    )
                    for di, day in enumerate(uds):
                        print(f"  --- {day} (UTC) ---", flush=True)
                        for sym in syms:
                            su = str(sym).strip().upper()
                            er = exmap.get(su, [])[di]
                            tw = er.get("win_rate_touch_sl_tp")
                            vr = er.get("win_rate_vs_all_resolved")
                            tws = f"{100.0 * float(tw):.2f}%" if tw is not None else "n/a"
                            vrs = f"{100.0 * float(vr):.2f}%" if vr is not None else "n/a"
                            print(
                                f"    {sym:16s} n_resolved={er.get('n_resolved', 0)}  "
                                f"w/L/exp={er.get('win')}/{er.get('loss')}/{er.get('expired')}  "
                                f"touch={tws}  vs_resolved={vrs}",
                                flush=True,
                            )
            print(
                "\n[bt] position_constraints（在「原始每笔」上再筛；详见 JSON position_constraints）",
                flush=True,
            )
            da = pos_con["dedupe_adjacent_actionable"]
            st = pos_con["adjacent_stack_chains_only"]
            oo = pos_con["one_open_per_symbol"]

            def _pct(x: Any) -> str:
                return f"{100.0 * float(x):.2f}%" if x is not None else "n/a"

            print(
                f"  dedupe_adjacent: trades={da['trades']}  "
                f"touch={_pct(da['aggregate_touch_win_rate'])}  "
                f"vs_resolved={_pct(da['aggregate_resolved_win_rate'])}",
                flush=True,
            )
            print(
                f"  stack_chains_only: trades={st['trades']}  "
                f"touch={_pct(st['aggregate_touch_win_rate'])}  "
                f"vs_resolved={_pct(st['aggregate_resolved_win_rate'])}",
                flush=True,
            )
            print(
                f"  one_open_per_sym: trades={oo['trades']}  "
                f"touch={_pct(oo['aggregate_touch_win_rate'])}  "
                f"vs_resolved={_pct(oo['aggregate_resolved_win_rate'])}",
                flush=True,
            )
        return summary
    finally:
        _restore_atr_klines_patch()
        z.BTC_MACRO_FILTER_ENABLED = _orig_btc_macro
        z._cooldown_blocks = _orig_cooldown  # type: ignore[method-assign]


def main() -> None:
    ap = argparse.ArgumentParser(description="ZCT VWAP walk-forward backtest")
    ap.add_argument("--days", type=float, default=7.0, help="回溯自然日（可小数）")
    ap.add_argument(
        "--symbols",
        type=str,
        default="",
        help="逗号分隔标的；默认用与扫描器相同的默认/环境标的",
    )
    ap.add_argument(
        "--zct-default-22",
        action="store_true",
        help="固定使用扫描器 _DEFAULT_ZCT_SYMBOLS 的 22 个标的（忽略 ZCT_VWAP_SYMBOLS）",
    )
    ap.add_argument(
        "--use-env-symbols",
        action="store_true",
        help="标的列表完全等同 z._symbols_from_env()（含触轨池 TOUCH_POOL）",
    )
    ap.add_argument(
        "--ignore-db-cooldown",
        action="store_true",
        help="[默认已忽略 DB 冷却] 显式关闭读 zct_symbol_cooldown；勿与 --use-db-cooldown 同用",
    )
    ap.add_argument(
        "--use-db-cooldown",
        action="store_true",
        help="回测沿用真实 z._cooldown_blocks（读 zct_symbol_cooldown），对齐实盘扫描",
    )
    ap.add_argument("--csv", type=str, default="", help="可选：逐笔 CSV 路径")
    ap.add_argument(
        "--json-out",
        type=str,
        default=_DEFAULT_SUMMARY_JSON,
        metavar="PATH",
        help=f"汇总 JSON（含 per_symbol）；默认 {_DEFAULT_SUMMARY_JSON}",
    )
    ap.add_argument(
        "--no-json-out",
        action="store_true",
        help="不写 JSON 文件，仅 stdout",
    )
    ap.add_argument(
        "--sleep-between-symbols",
        type=float,
        default=0.0,
        help="拉各周期/标的之间的休眠秒数，防限频",
    )
    ap.add_argument(
        "--signal-interval",
        type=str,
        default="1m",
        choices=["1m", "5m"],
        help="信号与 walk-forward 使用的币安 K 线周期；5m 时 resolve / 组合 / 仓位约束步长均为 5 分钟",
    )
    ap.add_argument(
        "--portfolio-sim",
        action="store_true",
        help="组合仿真：同向持仓忽略新信号；反向先平再开；满 N 小时未结强平；"
        "保证金×杠杆=名义；写 --portfolio-csv",
    )
    ap.add_argument(
        "--portfolio-csv",
        type=str,
        default="",
        metavar="PATH",
        help="--portfolio-sim 时交易明细 CSV（默认 next-k-api/zct_portfolio_trades.csv）",
    )
    ap.add_argument("--margin-usdt", type=float, default=100.0, help="portfolio 单笔保证金 USDT")
    ap.add_argument("--leverage", type=float, default=10.0, help="portfolio 杠杆倍数")
    ap.add_argument(
        "--force-flat-hours",
        type=float,
        default=6.0,
        help="portfolio：自 entry_bar_open_ms 起满该小时数未触轨则市价强平（按信号 K 线根的 open_time）",
    )
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
        print("[bt] no symbols, exit", file=sys.stderr)
        sys.exit(1)

    if args.ignore_db_cooldown and args.use_db_cooldown:
        ap.error("--ignore-db-cooldown 与 --use-db-cooldown 不能同时指定")
    ignore_db_cooldown = not bool(args.use_db_cooldown)

    if args.portfolio_sim:
        pcsv = (args.portfolio_csv or "").strip() or str(
            Path(__file__).resolve().parent / "zct_portfolio_trades.csv"
        )
        run_portfolio_backtest(
            days=args.days,
            symbols=symbols,
            ignore_db_cooldown=ignore_db_cooldown,
            sleep_between_symbols=max(0.0, float(args.sleep_between_symbols)),
            json_summary_path=None
            if args.no_json_out
            else (args.json_out.strip() or _DEFAULT_SUMMARY_JSON),
            portfolio_csv=pcsv,
            margin_usdt=float(args.margin_usdt),
            leverage=float(args.leverage),
            force_flat_hours=float(args.force_flat_hours),
            signal_interval=str(args.signal_interval),
        )
    else:
        run_backtest(
            days=args.days,
            symbols=symbols,
            ignore_db_cooldown=ignore_db_cooldown,
            csv_path=args.csv.strip() or None,
            sleep_between_symbols=max(0.0, float(args.sleep_between_symbols)),
            json_summary_path=None
            if args.no_json_out
            else (args.json_out.strip() or _DEFAULT_SUMMARY_JSON),
            signal_interval=str(args.signal_interval),
        )


if __name__ == "__main__":
    main()

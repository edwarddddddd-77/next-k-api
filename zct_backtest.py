#!/usr/bin/env python3
"""
P0：ZCT VWAP 规则离线回测（带手续费/滑点近似 + 分层统计）

  python zct_backtest.py --symbol BTCUSDT --start 2024-11-01 --end 2024-12-01
  python zct_backtest.py --symbol ETHUSDT --start 2024-01-01 --end 2024-06-01 --step-bars 15 --fee-bps 4

不调用 Telegram / 不写 accumulation.db。从币安拉 1m K 线，按步长重放会话内 VWAP 与分类逻辑。
参考 `zct_vwap_signal_scanner` 的 classify + SL/TP，用 1m 前向判定平仓（与 resolve 一致）。

参数与 `ZCT_*` 环境变量（斜率/带宽等）在 import 时已读入；可 export 后重跑。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np

# 与扫描器同目录 .env.oi
from pathlib import Path

_env_oi = Path(__file__).resolve().parent / ".env.oi"
if _env_oi.is_file():
    with open(_env_oi, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from zct_vwap_signal_scanner import (  # noqa: E402
    BAND_SIGMA,
    RESOLVE_MAX_BARS,
    VIRTUAL_NOTIONAL_USDT,
    USE_RISK_SIZED_NOTIONAL,
    _bar_hit_long,
    _bar_hit_short,
    _paper_notional_for_signal,
    classify_and_signal,
    compute_sl_tp,
    compute_vwap_bands_session,
    klines_to_df,
    replace,
    session_cut_utc,
    fetch_klines_forward,
)


@dataclass
class SimTrade:
    entry_i: int
    exit_i: int
    side: str
    play: str
    regime: str
    setup_level: int
    vwap_cross_bucket: str
    pnl_r: float
    pnl_usdt_gross: float
    pnl_usdt_net: float
    outcome: str
    entry_price: float
    exit_price: float
    notional: float


def _parse_utc_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _fee_slip_cost_usdt(notional: float, fee_bps: float, slip_bps: float) -> float:
    """双边手续费 + 单边滑点（按名义近似）。"""
    return notional * (2 * fee_bps + slip_bps) / 10000.0


def _simulate_one_trade(
    df,
    entry_idx: int,
    res,
    fee_bps: float,
    slip_bps: float,
) -> Optional[SimTrade]:
    """从 entry_idx 当根收盘入场，前向 1m 判定 SL/TP（与 resolve 同源）。"""
    if res.side not in ("LONG", "SHORT") or res.sl_price is None or res.tp_price is None:
        return None
    entry_px = float(df.iloc[entry_idx]["close"])
    sl = float(res.sl_price)
    tp = float(res.tp_price)
    start_ms = int(df.iloc[entry_idx]["open_time"]) + 60_000
    end_ms = int(df.iloc[-1]["open_time"]) + 60_000
    # 在内存 df 中找起始索引
    j0 = entry_idx + 1
    outcome = None
    exit_px = entry_px
    exit_j = entry_idx
    bars_seen = 0
    for j in range(j0, len(df)):
        if int(df.iloc[j]["open_time"]) < start_ms:
            continue
        bars_seen += 1
        row = df.iloc[j]
        o, h, low, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        if res.side == "LONG":
            tag, px = _bar_hit_long(o, h, low, sl, tp)
        else:
            tag, px = _bar_hit_short(o, h, low, sl, tp)
        if tag == "win":
            outcome = "win"
            exit_px = px
            exit_j = j
            break
        if tag == "loss":
            outcome = "loss"
            exit_px = px
            exit_j = j
            break
        if bars_seen >= RESOLVE_MAX_BARS:
            outcome = "expired"
            exit_px = c
            exit_j = j
            break
    if outcome is None:
        return None

    res_px = replace(res, price=entry_px)
    notion = _paper_notional_for_signal(res_px) if USE_RISK_SIZED_NOTIONAL else float(VIRTUAL_NOTIONAL_USDT)
    if res.side == "LONG":
        pnl_u_g = notion * (exit_px - entry_px) / entry_px
        risk = entry_px - sl
        pnl_rv = (exit_px - entry_px) / risk if risk > 0 else 0.0
    else:
        pnl_u_g = notion * (entry_px - exit_px) / entry_px
        risk = sl - entry_px
        pnl_rv = (entry_px - exit_px) / risk if risk > 0 else 0.0

    costs = _fee_slip_cost_usdt(notion, fee_bps, slip_bps)
    pnl_net = pnl_u_g - costs

    return SimTrade(
        entry_i=entry_idx,
        exit_i=exit_j,
        side=res.side,
        play=res.play,
        regime=res.regime,
        setup_level=res.setup_level,
        vwap_cross_bucket=res.vwap_cross_bucket,
        pnl_r=float(pnl_rv),
        pnl_usdt_gross=float(pnl_u_g),
        pnl_usdt_net=float(pnl_net),
        outcome=outcome,
        entry_price=entry_px,
        exit_price=float(exit_px),
        notional=float(notion),
    )


def run_backtest(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    step_bars: int,
    fee_bps: float,
    slip_bps: float,
    initial_equity: float,
) -> Dict[str, Any]:
    kl = fetch_klines_forward(symbol, "1m", start_ms, end_ms)
    if not kl:
        return {"ok": False, "error": "no_klines"}
    df = klines_to_df(kl)
    if len(df) < 80:
        return {"ok": False, "error": "too_few_bars", "n": len(df)}

    trades: List[SimTrade] = []
    equity = initial_equity
    equity_by_day: Dict[str, float] = {}

    next_i = 40
    while next_i < len(df) - 2:
        i = next_i
        sub = df.iloc[: i + 1].copy()
        sdf = session_cut_utc(sub)
        if len(sdf) < 30:
            next_i = i + step_bars
            continue
        sdf = compute_vwap_bands_session(sdf, BAND_SIGMA)
        res = classify_and_signal(symbol, sdf, {})
        sl, tp, _ru = compute_sl_tp(res, sdf)
        res = replace(res, sl_price=sl, tp_price=tp, r_unit=_ru, price=float(sdf.iloc[-1]["close"]))

        tr_done: Optional[SimTrade] = None
        if res.side in ("LONG", "SHORT") and sl is not None and tp is not None:
            tr_done = _simulate_one_trade(df, i, res, fee_bps, slip_bps)
            if tr_done:
                trades.append(tr_done)
                equity += tr_done.pnl_usdt_net
                day_key = datetime.fromtimestamp(
                    df.iloc[tr_done.exit_i]["open_time"] / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d")
                equity_by_day[day_key] = equity

        if tr_done:
            next_i = max(tr_done.exit_i + 1, i + step_bars)
        else:
            next_i = i + step_bars

    # 指标
    wins = sum(1 for t in trades if t.outcome == "win")
    losses = sum(1 for t in trades if t.outcome == "loss")

    # 日收益 Sharpe（近似）
    days_sorted = sorted(equity_by_day.keys())
    daily_rets: List[float] = []
    prev_e = initial_equity
    for d in days_sorted:
        e = equity_by_day[d]
        if prev_e > 0:
            daily_rets.append((e - prev_e) / prev_e)
        prev_e = e
    if len(daily_rets) > 1:
        dr = np.array(daily_rets)
        sharpe = float(np.mean(dr) / (np.std(dr) + 1e-12) * np.sqrt(252))
    else:
        sharpe = float("nan")

    peak = initial_equity
    max_dd = 0.0
    run_e = initial_equity
    for t in trades:
        run_e += t.pnl_usdt_net
        peak = max(peak, run_e)
        if peak > 0:
            max_dd = max(max_dd, (peak - run_e) / peak)

    by_regime: DefaultDict[str, List[float]] = defaultdict(list)
    by_bucket: DefaultDict[str, List[float]] = defaultdict(list)
    by_level: DefaultDict[str, List[float]] = defaultdict(list)
    for t in trades:
        by_regime[t.regime].append(t.pnl_usdt_net)
        by_bucket[t.vwap_cross_bucket].append(t.pnl_usdt_net)
        by_level[str(t.setup_level)].append(t.pnl_usdt_net)

    def _sum_mean(xs: List[float]) -> Tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        return float(sum(xs)), float(sum(xs) / len(xs))

    layered = {
        "by_regime": {k: {"n": len(v), "total_net": _sum_mean(v)[0], "mean_net": _sum_mean(v)[1]} for k, v in by_regime.items()},
        "by_vwap_cross_bucket": {k: {"n": len(v), "total_net": _sum_mean(v)[0]} for k, v in by_bucket.items()},
        "by_setup_level": {k: {"n": len(v), "total_net": _sum_mean(v)[0]} for k, v in by_level.items()},
    }

    expired_n = sum(1 for t in trades if t.outcome == "expired")
    decisive = wins + losses
    win_rate_vs_sl = (wins / decisive) if decisive > 0 else None
    win_rate_all = (wins / len(trades)) if trades else None

    per_symbol: Dict[str, Any] = {
        symbol: {
            "trades": len(trades),
            "wins": wins,
            "losses": losses,
            "expired": expired_n,
            "win_rate_vs_sl": round(win_rate_vs_sl, 4) if win_rate_vs_sl is not None else None,
            "win_rate_all_trades": round(win_rate_all, 4) if win_rate_all is not None else None,
            "note": "win_rate_vs_sl = wins/(wins+losses)；不含 expired。win_rate_all_trades = wins/全部笔数。",
        }
    }

    return {
        "ok": True,
        "symbol": symbol,
        "bars": len(df),
        "step_bars": step_bars,
        "fee_bps": fee_bps,
        "slippage_bps": slip_bps,
        "initial_equity": initial_equity,
        "final_equity": equity,
        "total_return_pct": (equity / initial_equity - 1.0) * 100.0 if initial_equity > 0 else 0.0,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "expired": expired_n,
        "win_rate_vs_sl": round(win_rate_vs_sl, 4) if win_rate_vs_sl is not None else None,
        "win_rate_all_trades": round(win_rate_all, 4) if win_rate_all is not None else None,
        "sharpe_daily_approx": sharpe,
        "max_drawdown_pct": max_dd * 100.0,
        "layered_pnl_usdt": layered,
        "per_symbol": per_symbol,
        "trade_list": [asdict(t) for t in trades],
    }


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description="ZCT VWAP offline backtest (P0)")
    ap.add_argument("--symbol", default="BTCUSDT", help="U 本位永续 symbol")
    ap.add_argument("--start", required=True, help="UTC 起始日 YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="UTC 结束日 YYYY-MM-DD（不含当日结束可用次日）")
    ap.add_argument("--step-bars", type=int, default=15, help="重放步长（分钟），默认 15 对齐定时扫描")
    ap.add_argument("--fee-bps", type=float, default=4.0, help="单边费率 bp，双边合计 2×")
    ap.add_argument("--slippage-bps", type=float, default=2.0, help="成交滑点近似 bp（计入成本）")
    ap.add_argument("--initial-equity", type=float, default=10000.0)
    ap.add_argument("--json-out", help="可选：写入 JSON 文件路径")
    args = ap.parse_args()

    sdt = _parse_utc_date(args.start)
    edt = _parse_utc_date(args.end)
    start_ms = int(sdt.timestamp() * 1000)
    end_ms = int(edt.timestamp() * 1000)
    if end_ms <= start_ms:
        print("end must be after start", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching {args.symbol} 1m ...")
    out = run_backtest(
        args.symbol.upper(),
        start_ms,
        end_ms,
        step_bars=max(1, args.step_bars),
        fee_bps=args.fee_bps,
        slip_bps=args.slippage_bps,
        initial_equity=args.initial_equity,
    )
    if not out.get("ok"):
        print(json.dumps(out, indent=2))
        sys.exit(2)

    summary = _json_safe({k: v for k, v in out.items() if k != "trade_list"})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(_json_safe(out), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote full report → {args.json_out}")


if __name__ == "__main__":
    main()

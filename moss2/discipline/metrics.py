"""从成交记录计算 EV / 存活指标。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def trade_stats_from_rows(trades: List[dict]) -> Dict[str, Any]:
    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "ev_per_trade_pct": 0.0,
            "profit_factor": 0.0,
            "max_consecutive_losses": 0,
        }
    pnls: List[float] = []
    for t in trades:
        v = t.get("pnl_pct")
        if v is None and t.get("pnl_usdt") is not None:
            pnls.append(float(t["pnl_usdt"]))
        elif v is not None:
            pnls.append(float(v))
    if not pnls:
        return {"trade_count": len(trades), "win_rate": 0.0, "ev_per_trade_pct": 0.0}

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n = len(pnls)
    p = len(wins) / n if n else 0.0
    w = sum(wins) / len(wins) if wins else 0.0
    l = abs(sum(losses) / len(losses)) if losses else 0.0
    q = 1.0 - p
    ev = p * w - q * l
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    max_streak = 0
    streak = 0
    for pnl in pnls:
        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "trade_count": n,
        "win_rate": round(p, 4),
        "avg_win_pct": round(w, 6),
        "avg_loss_pct": round(l, 6),
        "ev_per_trade_pct": round(ev, 6),
        "profit_factor": round(min(pf, 999.0), 4),
        "max_consecutive_losses": max_streak,
    }


def kelly_fraction(
    win_rate: float, avg_win: float, avg_loss: float, *, half: bool = True
) -> float:
    if avg_loss <= 0 or win_rate <= 0:
        return 0.0
    b = avg_win / avg_loss if avg_loss else 1.0
    p = win_rate
    q = 1.0 - p
    f = (b * p - q) / b if b > 0 else 0.0
    f = max(0.0, f)
    if half:
        f *= 0.5
    return round(f, 4)


def summary_from_backtest(summary: dict, trades: List[dict]) -> Dict[str, Any]:
    ts = trade_stats_from_rows(trades)
    kf = kelly_fraction(ts["win_rate"], ts["avg_win_pct"], ts["avg_loss_pct"])
    return {
        "ev": ts,
        "survival": {
            "max_consecutive_losses": ts["max_consecutive_losses"],
            "max_drawdown": summary.get("max_drawdown"),
            "blowup_count": summary.get("blowup_count", 0),
        },
        "kelly": {
            "half_kelly_fraction": kf,
            "cap": None,
        },
        "backtest": {
            "total_return": summary.get("total_return"),
            "sharpe": summary.get("sharpe"),
            "win_rate": summary.get("win_rate"),
            "total_trades": summary.get("total_trades"),
        },
    }

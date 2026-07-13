#!/usr/bin/env python3
"""Compare anchor drift: weekend_only vs all sessions."""
from __future__ import annotations

from quant.anchor_drift.backtest import BacktestParams, run_backtest
from quant.anchor_drift.config import AnchorDriftConfig


def summarize(label: str, *, weekend_only: bool) -> None:
    cfg = AnchorDriftConfig.from_env()
    params = BacktestParams(
        symbols=["MSTR", "COIN", "HOOD"],
        days=60,
        interval="5m",
        equity_usdt=14.0,
        compound=False,
        cfg=cfg,
        weekend_only=weekend_only,
    )
    out = run_backtest(params)
    sm = out["summary"]
    trades = [t for sym in out["symbols"] for t in sym["trades"]]
    by_reason: dict[str, dict] = {}
    for t in trades:
        bucket = by_reason.setdefault(t["exit_reason"], {"n": 0, "pnl": 0.0, "w": 0})
        bucket["n"] += 1
        bucket["pnl"] += t["pnl_net_usdt"]
        if t["pnl_net_usdt"] > 0:
            bucket["w"] += 1

    print(f"=== {label} ===")
    print(
        f"trades={sm['total_trades']}  pnl={sm['total_pnl_net']:+.4f} USDT  "
        f"win_rate={sm['win_rate'] * 100:.1f}%"
    )
    print(
        f"  weekend entries: {sm['weekend']['trades']}t  pnl={sm['weekend']['pnl_net']:+.4f}"
    )
    print(
        f"  overnight entries: {sm['overnight']['trades']}t  pnl={sm['overnight']['pnl_net']:+.4f}"
    )
    for k, v in sorted(by_reason.items(), key=lambda x: -x[1]["n"]):
        wr = v["w"] / v["n"] * 100 if v["n"] else 0.0
        print(f"  {k}: n={v['n']}  win={wr:.0f}%  pnl={v['pnl']:+.4f}")
    print()


def main() -> None:
    summarize("ALL sessions (Mon–Thu overnight + weekend)", weekend_only=False)
    summarize("WEEKEND ONLY (Fri 16:00 → Mon 9:25 anchor)", weekend_only=True)


if __name__ == "__main__":
    main()

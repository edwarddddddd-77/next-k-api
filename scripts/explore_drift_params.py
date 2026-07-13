#!/usr/bin/env python3
"""Parameter sweep: anchor drift weekend / threshold / adverse stop."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from quant.anchor_drift.backtest import BacktestParams, run_backtest
from quant.anchor_drift.config import AnchorDriftConfig


def run_case(
    *,
    label: str,
    symbols: list[str],
    days: int,
    weekend_only: bool,
    sat_sun_entry_only: bool,
    disable_adverse_stop: bool,
    signal_threshold: float,
) -> dict[str, Any]:
    cfg = replace(
        AnchorDriftConfig.from_env(),
        signal_threshold=float(signal_threshold),
    )
    params = BacktestParams(
        symbols=symbols,
        days=days,
        interval="5m",
        equity_usdt=14.0,
        compound=False,
        cfg=cfg,
        weekend_only=weekend_only,
        sat_sun_entry_only=sat_sun_entry_only,
        disable_adverse_stop=disable_adverse_stop,
    )
    out = run_backtest(params)
    trades = [t for sym in out["symbols"] for t in sym["trades"]]
    sm = out["summary"]
    by_reason: dict[str, dict[str, float]] = {}
    for t in trades:
        r = t["exit_reason"]
        b = by_reason.setdefault(r, {"n": 0, "pnl": 0.0})
        b["n"] += 1
        b["pnl"] += t["pnl_net_usdt"]
    return {
        "label": label,
        "trades": sm["total_trades"],
        "pnl": sm["total_pnl_net"],
        "win_rate": sm["win_rate"],
        "weekend_pnl": sm["weekend"]["pnl_net"],
        "weekend_n": sm["weekend"]["trades"],
        "adverse_n": int(by_reason.get("adverse_drift", {}).get("n", 0)),
        "adverse_pnl": by_reason.get("adverse_drift", {}).get("pnl", 0.0),
        "converged_n": int(by_reason.get("converged", {}).get("n", 0)),
        "converged_pnl": by_reason.get("converged", {}).get("pnl", 0.0),
        "preopen_n": int(by_reason.get("preopen_flat", {}).get("n", 0)),
        "preopen_pnl": by_reason.get("preopen_flat", {}).get("pnl", 0.0),
    }


def main() -> None:
    symbols_3 = ["MSTR", "COIN", "HOOD"]
    symbols_6 = ["MSTR", "COIN", "HOOD", "CRCL", "SOXL", "SNDK"]
    days = 60
    thresholds = (0.015, 0.02, 0.025)

    cases: list[dict[str, Any]] = []

    def add(pool: list[str], **kw: Any) -> None:
        cases.append({**kw, "symbols": pool})

    for th in thresholds:
        th_pct = f"{th * 100:.1f}%"
        add(
            symbols_3,
            label=f"ALL thr={th_pct}",
            weekend_only=False,
            sat_sun_entry_only=False,
            disable_adverse_stop=False,
            signal_threshold=th,
        )
        add(
            symbols_3,
            label=f"ALL thr={th_pct} no-adverse",
            weekend_only=False,
            sat_sun_entry_only=False,
            disable_adverse_stop=True,
            signal_threshold=th,
        )

    for th in thresholds:
        th_pct = f"{th * 100:.1f}%"
        add(
            symbols_3,
            label=f"wknd thr={th_pct}",
            weekend_only=True,
            sat_sun_entry_only=False,
            disable_adverse_stop=False,
            signal_threshold=th,
        )
        add(
            symbols_3,
            label=f"wknd thr={th_pct} no-adverse",
            weekend_only=True,
            sat_sun_entry_only=False,
            disable_adverse_stop=True,
            signal_threshold=th,
        )
        add(
            symbols_3,
            label=f"wknd+SatSun thr={th_pct}",
            weekend_only=True,
            sat_sun_entry_only=True,
            disable_adverse_stop=False,
            signal_threshold=th,
        )
        add(
            symbols_3,
            label=f"wknd+SatSun thr={th_pct} no-adverse",
            weekend_only=True,
            sat_sun_entry_only=True,
            disable_adverse_stop=True,
            signal_threshold=th,
        )

    # Best candidates on full 6-symbol pool
    for th in (0.02, 0.025):
        th_pct = f"{th * 100:.1f}%"
        add(
            symbols_6,
            label=f"[6sym] wknd+SatSun thr={th_pct} no-adverse",
            weekend_only=True,
            sat_sun_entry_only=True,
            disable_adverse_stop=True,
            signal_threshold=th,
        )

    print(f"Running {len(cases)} backtests ({days}d / 5m)...")
    rows: list[dict[str, Any]] = []
    for i, c in enumerate(cases, 1):
        sym = c.pop("symbols")
        print(f"  [{i}/{len(cases)}] {c['label']} ({len(sym)} symbols)")
        rows.append(run_case(**c, symbols=sym, days=days))

    rows.sort(key=lambda r: r["pnl"], reverse=True)

    print()
    print("=== Ranked by net PnL (USDT) ===")
    hdr = (
        f"{'label':<38} {'trades':>6} {'pnl':>8} {'win%':>6} "
        f"{'conv':>12} {'adverse':>12} {'preopen':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        conv = f"{r['converged_n']:>2}t {r['converged_pnl']:+.3f}"
        adv = f"{r['adverse_n']:>2}t {r['adverse_pnl']:+.3f}"
        pre = f"{r['preopen_n']:>2}t {r['preopen_pnl']:+.3f}"
        print(
            f"{r['label']:<38} {r['trades']:>6} {r['pnl']:>+8.4f} "
            f"{r['win_rate'] * 100:>5.1f}% {conv:>12} {adv:>12} {pre:>12}"
        )

    profitable = [r for r in rows if r["pnl"] > 0]
    print()
    if profitable:
        best = profitable[0]
        print(f"Best profitable: {best['label']}  pnl={best['pnl']:+.4f}  trades={best['trades']}")
    else:
        best = rows[0]
        print(f"No profitable combo. Least bad: {best['label']}  pnl={best['pnl']:+.4f}")


if __name__ == "__main__":
    main()

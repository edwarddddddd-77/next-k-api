#!/usr/bin/env python3
"""ORB 回测复利收益曲线 → SVG + CSV（本地打开即可）。"""

from __future__ import annotations

import argparse
import html
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.backtest import run_backtest  # noqa: E402
from orb.config import OrbConfig  # noqa: E402
from tools.print_pltr_backtest_detail import days_since_onboard, symbol_onboard_ms  # noqa: E402


def _resolved_trades(raw: dict) -> List[Dict[str, Any]]:
    return [
        t
        for t in (raw.get("trades") or [])
        if t.get("outcome") and t["outcome"] != "supersede"
    ]


def equity_series(trades: List[Dict[str, Any]], *, init: float) -> List[Tuple[str, float]]:
    """(session_date, wallet_after) 按日期排序；首点为开仓前本金。"""
    if not trades:
        return [("start", init)]

    rows: List[Tuple[str, float, float]] = []
    for t in trades:
        pu = float(t.get("pnl_usdt") or 0)
        after = float(t.get("wallet_after") or 0)
        before = round(after - pu, 4)
        rows.append((str(t.get("session_date") or ""), before, after))

    rows.sort(key=lambda x: x[0])
    out: List[Tuple[str, float]] = []
    wallet = init
    for day, before, after in rows:
        if not out:
            out.append((day, init))
        elif abs(before - wallet) > 0.02:
            out.append((day, wallet))
        out.append((day, after))
        wallet = after
    return out


def _svg_line_chart(
    points: List[Tuple[str, float]],
    *,
    title: str,
    width: int = 960,
    height: int = 420,
) -> str:
    if len(points) < 2:
        raise ValueError("need at least 2 points for chart")

    labels = [p[0] for p in points]
    vals = [p[1] for p in points]
    y_min = min(vals)
    y_max = max(vals)
    pad_y = max((y_max - y_min) * 0.08, y_max * 0.02, 1.0)
    y0, y1 = y_min - pad_y, y_max + pad_y

    margin = dict(l=72, r=24, t=48, b=64)
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]

    def x_px(i: int) -> float:
        if len(points) == 1:
            return margin["l"] + plot_w / 2
        return margin["l"] + i * plot_w / (len(points) - 1)

    def y_px(v: float) -> float:
        if y1 <= y0:
            return margin["t"] + plot_h / 2
        return margin["t"] + (y1 - v) / (y1 - y0) * plot_h

    poly = " ".join(f"{x_px(i):.1f},{y_px(v):.1f}" for i, v in enumerate(vals))
    area = (
        f"{x_px(0):.1f},{y_px(y0):.1f} "
        + poly
        + f" {x_px(len(vals)-1):.1f},{y_px(y0):.1f}"
    )

    # y-axis ticks (5)
    ticks: List[str] = []
    for k in range(5):
        v = y0 + (y1 - y0) * k / 4
        y = y_px(v)
        ticks.append(
            f'<line x1="{margin["l"]}" y1="{y:.1f}" x2="{width-margin["r"]}" y2="{y:.1f}" '
            f'stroke="#e8ecf1" stroke-width="1"/>'
        )
        ticks.append(
            f'<text x="{margin["l"]-8}" y="{y+4:.1f}" text-anchor="end" '
            f'font-family="Segoe UI,sans-serif" font-size="11" fill="#64748b">'
            f"{v:,.0f}</text>"
        )

    # x labels: first, mid, last
    x_labels: List[str] = []
    for idx in {0, len(labels) // 2, len(labels) - 1}:
        x_labels.append(
            f'<text x="{x_px(idx):.1f}" y="{height-28}" text-anchor="middle" '
            f'font-family="Segoe UI,sans-serif" font-size="11" fill="#64748b">'
            f"{html.escape(labels[idx])}</text>"
        )

    init_v = vals[0]
    final_v = vals[-1]
    ret_pct = (final_v / init_v - 1) * 100 if init_v > 0 else 0.0
    subtitle = f"init {init_v:,.0f} U → {final_v:,.0f} U ({ret_pct:+.1f}%)"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fafbfc"/>
  <text x="{margin['l']}" y="28" font-family="Segoe UI,sans-serif" font-size="16" font-weight="600" fill="#0f172a">{html.escape(title)}</text>
  <text x="{margin['l']}" y="44" font-family="Segoe UI,sans-serif" font-size="12" fill="#64748b">{html.escape(subtitle)}</text>
  {''.join(ticks)}
  <polygon points="{area}" fill="#2563eb" fill-opacity="0.12"/>
  <polyline points="{poly}" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
  {''.join(x_labels)}
  <circle cx="{x_px(len(vals)-1):.1f}" cy="{y_px(final_v):.1f}" r="4" fill="#2563eb"/>
</svg>
"""


def _write_csv(path: Path, points: List[Tuple[str, float]]) -> None:
    lines = ["date,wallet_usdt,return_pct"]
    base = points[0][1]
    for day, w in points:
        ret = (w / base - 1) * 100 if base > 0 else 0.0
        lines.append(f"{day},{w:.4f},{ret:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    load_env_oi()
    ap = argparse.ArgumentParser(description="ORB backtest equity curve → SVG + CSV")
    ap.add_argument("--symbol", default="COINUSDT")
    ap.add_argument("--days", type=float, default=None)
    ap.add_argument("--since-listing", action="store_true", help="from Binance onboardDate")
    ap.add_argument(
        "--no-premarket",
        action="store_true",
        help="baseline without premarket filter (A/B second line)",
    )
    ap.add_argument("--out-dir", default=str(ROOT / "output"))
    args = ap.parse_args()

    sym = str(args.symbol).strip().upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    base = sym[:-4]

    cfg = OrbConfig.for_backtest()
    if args.no_premarket:
        cfg = replace(cfg, premarket_filter=False)

    if args.since_listing or args.days is None:
        days = days_since_onboard(sym)
        onboard = symbol_onboard_ms(sym)
        import pandas as pd

        listed = pd.Timestamp(onboard, unit="ms", tz=cfg.session_tz).strftime("%Y-%m-%d")
        window = f"since_listing ({listed}, {days:.0f}d)"
    else:
        days = float(args.days)
        window = f"{int(days)}d"

    init = cfg.per_symbol_bot_equity()
    raw = run_backtest(days=days, symbols=[sym], cfg=cfg, json_path=None, csv_path=None)
    trades = _resolved_trades(raw)
    points = equity_series(trades, init=init)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "baseline" if args.no_premarket else "premarket"
    stem = f"orb_{base.lower()}_equity_{tag}"
    svg_path = out_dir / f"{stem}.svg"
    csv_path = out_dir / f"{stem}.csv"

    pm = "off" if args.no_premarket else f"on ({cfg.premarket_mode})"
    title = f"{base} ORB compound equity | {window} | premarket {pm}"
    svg_path.write_text(_svg_line_chart(points, title=title), encoding="utf-8")
    _write_csv(csv_path, points)

    wins = sum(1 for t in trades if float(t.get("pnl_usdt") or 0) > 0)
    losses = len(trades) - wins
    final = points[-1][1]
    ret = (final / init - 1) * 100 if init > 0 else 0.0

    print(f"symbol={sym} | trades={len(trades)} | win/loss={wins}/{losses}")
    print(f"final_wallet={final:,.2f} U | return={ret:+.2f}%")
    print(f"SVG: {svg_path.resolve()}")
    print(f"CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()

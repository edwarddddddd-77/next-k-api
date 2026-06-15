#!/usr/bin/env python3
"""ORB V2 单标的 live_gate 回测（本地 K 线，scan-by-scan）。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.core.config import OrbConfig  # noqa: E402
from orb.core.kline_cache import has_kline_cache, kline_path, load_klines, norm_symbol, symbol_cache_dir  # noqa: E402
from orb.core.session import session_day_str  # noqa: E402
from orb.core.us_equity_calendar import is_us_equity_market, is_us_equity_trading_day  # noqa: E402
from orb.ml.gate import LiveGateConfig  # noqa: E402
from orb.ml.ranker import BreakoutRanker  # noqa: E402
from orb.v2.paths import resolve_gate_config_path, resolve_gbm_path, resolve_profiles_path  # noqa: E402

from tools.orb.ml.eval_live_gate import (  # noqa: E402
    _ml_cfg,
    init_symbol_wallets,
    simulate_live_gate_day,
)


def session_dates_from_cache(symbol: str, cfg: OrbConfig) -> List[str]:
    """K 线缓存中的 session_date；美股市场仅含 NYSE 交易日。"""
    sym = norm_symbol(symbol)
    df = load_klines(sym, cfg.signal_interval)
    if df.empty:
        return []
    tz = cfg.session_tz
    open_time = cfg.session_open_time
    dates = {
        session_day_str(int(t), tz=tz, session_open_time=open_time)
        for t in df["open_time"].astype("int64")
    }
    out = sorted(d for d in dates if d)
    if is_us_equity_market(cfg.market):
        out = [d for d in out if is_us_equity_trading_day(d)]
    return out


def _day_pnl(day: dict) -> float:
    return round(sum(float(r.get("pnl_usdt") or 0) for r in day.get("opened") or []), 4)


def main() -> int:
    load_env_oi()
    ap = argparse.ArgumentParser(description="ORB V2 single-symbol live_gate backtest")
    ap.add_argument("--symbol", default="COINUSDT", help="如 COIN 或 COINUSDT")
    ap.add_argument("--from-date", default="", help="YYYY-MM-DD，默认缓存首日")
    ap.add_argument("--to-date", default="", help="YYYY-MM-DD，默认缓存末日")
    ap.add_argument("--gate-config", default=str(resolve_gate_config_path()))
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--fixed-notional", type=float, default=0.0, help="0=方案A复利")
    ap.add_argument(
        "--no-live-filters",
        action="store_true",
        help="关闭 env 宏观过滤（旧回测口径）",
    )
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    sym = norm_symbol(args.symbol)
    label = sym.replace("USDT", "")
    if not has_kline_cache(sym, "5m"):
        print(f"No kline cache for {sym}. Run: python tools/orb/v2/refresh_klines.py --symbols {label}")
        return 1

    compound = float(args.fixed_notional) <= 0
    cfg = _ml_cfg(
        compound_per_symbol=compound,
        fixed_notional=float(args.fixed_notional),
        respect_env_filters=not bool(args.no_live_filters),
    )
    gate = LiveGateConfig.from_json(Path(args.gate_config))
    if args.min_p is not None:
        gate.min_p_true = float(args.min_p)

    ranker = BreakoutRanker.load(
        gbm_path=resolve_gbm_path(),
        profiles_path=resolve_profiles_path(),
    )
    if ranker.gbm is None and ranker.logistic is None:
        print("ML model missing — run tools/orb/v2/monthly_train.py --bootstrap-only")
        return 1

    all_dates = session_dates_from_cache(sym, cfg)
    if not all_dates:
        print("No session dates in cache")
        return 1

    d0 = args.from_date.strip() or all_dates[0]
    d1 = args.to_date.strip() or all_dates[-1]
    dates = [d for d in all_dates if d0 <= d <= d1]
    if not dates:
        print(f"No dates in range {d0} .. {d1}")
        return 1

    print(
        f"[v2 backtest] {sym} | {dates[0]} .. {dates[-1]} | {len(dates)} sessions | "
        f"gate p>={gate.min_p_true} max={gate.max_opens_per_day} | ranker={ranker.kind}",
        flush=True,
    )
    print("note: 单标的 sync=0，早突破陷阱(sync 3-14)不会触发", flush=True)

    wallets = init_symbol_wallets([sym], cfg) if compound else None
    t0 = time.time()
    days = []
    for i, d in enumerate(dates, 1):
        print(f"[{i}/{len(dates)}] {d} ...", flush=True)
        days.append(simulate_live_gate_day(d, [sym], cfg, ranker, gate, wallets=wallets))

    pnls = [_day_pnl(d) for d in days]
    total_pnl = round(sum(pnls), 2)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    flats = sum(1 for p in pnls if p == 0)
    opens_total = sum(d.get("opens") or 0 for d in days)
    true_total = sum(d.get("true_opens") or 0 for d in days)
    gate_skips = sum(len(d.get("skipped_sample") or []) for d in days)

    out = {
        "kind": "orb_v2_symbol_backtest",
        "symbol": sym,
        "listing_note": "Binance COINUSDT 上线 2026-02-09 UTC；缓存首日即上市日",
        "single_symbol_caveat": "sync=0：早突破陷阱规则不生效",
        "date_range": {"from": dates[0], "to": dates[-1], "sessions": len(dates)},
        "gate": gate.__dict__,
        "ranker": ranker.kind,
        "sizing": "per_symbol_compound" if compound else "fixed_notional",
        "summary": {
            "total_pnl_usdt": total_pnl,
            "avg_daily_pnl_usdt": round(total_pnl / len(days), 2) if days else 0,
            "win_days": wins,
            "loss_days": losses,
            "flat_days": flats,
            "total_opens": opens_total,
            "total_true_opens": true_total,
            "avg_opens_per_day": round(opens_total / len(days), 2) if days else 0,
            "goal_min_hit_days": sum(1 for d in days if d.get("goal_met_min")),
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "days": days,
    }
    if compound and wallets is not None:
        out["final_wallet_usdt"] = round(wallets.get(sym, 0), 2)
        out["initial_wallet_usdt"] = float(cfg.per_symbol_bot_equity())

    json_out = Path(args.json_out) if args.json_out.strip() else ROOT / "output" / "orb" / "v2" / "eval" / f"{label.lower()}_v2_backtest.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    s = out["summary"]
    print(json.dumps(s, indent=2, ensure_ascii=False))
    print(f"\nfull -> {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

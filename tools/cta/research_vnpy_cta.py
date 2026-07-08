#!/usr/bin/env python3
"""vnpy_ctastrategy 官方示例策略回测 — 默认 BacktestingEngine；--legacy 为旧 1m 触价引擎。"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402

load_env_oi()

import pandas as pd  # noqa: E402

from binance_fapi import fetch_klines_forward, klines_to_df  # noqa: E402
from orb.core.config import OrbConfig  # noqa: E402
from orb.core.kline_cache import load_klines, norm_symbol, session_dates_from_cache  # noqa: E402
from orb.core.symbols import parse_symbol_list  # noqa: E402
from orb.cta.registry import CTA_STRATEGIES, list_strategies  # noqa: E402
from orb.cta.vnpy.backtest import (  # noqa: E402
    CtaVnpyBacktestConfig,
    klines_df_to_bars,
    range_engine_dates,
    range_replay_end,
    run_vnpy_cta_backtest,
    trades_to_rows,
)
from orb.cta.vnpy.registry import VNPY_CTA_STRATEGIES, list_vnpy_strategies  # noqa: E402
from orb.core.symbols_path import resolve_symbols_path  # noqa: E402
from orb.vnpy.binance_gateway import vnpy_vt_symbol  # noqa: E402
from tools.cta.research_vnpy_cta_legacy import _session_slice  # noqa: E402 — 兼容旧 import

__all__ = ["_session_slice"]


def _fetch_range(sym: str, from_date: str, to_date: str, cfg: OrbConfig) -> pd.DataFrame:
    tz = cfg.session_tz
    lo = pd.Timestamp(from_date.strip(), tz=tz)
    hi = pd.Timestamp(to_date.strip(), tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    rows = fetch_klines_forward(sym, "1m", int(lo.value // 1_000_000), int(hi.value // 1_000_000))
    df = klines_to_df(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time").reset_index(drop=True)


def _load_symbol_df(sym: str, fetch_from: str, hi: str, cfg: OrbConfig) -> pd.DataFrame:
    tz = cfg.session_tz
    lo_ts = pd.Timestamp(fetch_from.strip(), tz=tz)
    hi_ts = pd.Timestamp(hi.strip(), tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    lo_ms, hi_ms = int(lo_ts.value // 1_000_000), int(hi_ts.value // 1_000_000)
    df = load_klines(sym, "1m")
    if df is not None and not df.empty:
        sl = df[(df["open_time"] >= lo_ms) & (df["open_time"] <= hi_ms)].copy()
        if not sl.empty:
            return sl.sort_values("open_time").reset_index(drop=True)
    return _fetch_range(sym, fetch_from, hi, cfg)


def count_opens(trades: list) -> int:
    from vnpy.trader.constant import Offset

    n = 0
    for t in trades:
        off = getattr(t, "offset", None)
        if off == Offset.OPEN:
            n += 1
    return n


def run_one_vnpy(
    strategy_key: str,
    symbol: str,
    *,
    cfg: OrbConfig,
    bt_cfg: CtaVnpyBacktestConfig,
    lo: str,
    hi: str,
    fetch_start: str,
    engine_start_s: str,
    replay_start,
    replay_end,
) -> Dict[str, Any]:
    sym = norm_symbol(symbol)
    label = sym.replace("USDT", "")
    meta = VNPY_CTA_STRATEGIES[strategy_key]
    df = _load_symbol_df(sym, fetch_start, hi, cfg)
    if df.empty:
        return {
            "symbol": label,
            "strategy": strategy_key,
            "summary": {"net_pnl_usdt": 0, "opens": 0, "equity_end": bt_cfg.equity_usdt},
        }
    px = float(df.iloc[-1]["close"])
    bars = klines_df_to_bars(df, sym, vt_symbol=vnpy_vt_symbol(sym))
    engine_start = pd.Timestamp(engine_start_s, tz=cfg.session_tz).to_pydatetime().replace(tzinfo=timezone.utc)
    out = run_vnpy_cta_backtest(
        sym,
        bars,
        strategy_key=strategy_key,
        bt_cfg=bt_cfg,
        start=engine_start,
        end=replay_end,
        price=px,
        quiet=True,
        replay_start=replay_start,
        replay_end=replay_end,
        orb_cfg=cfg,
    )
    if out.get("error"):
        return {
            "symbol": label,
            "strategy": strategy_key,
            "summary": {"net_pnl_usdt": 0, "opens": 0, "equity_end": bt_cfg.equity_usdt, "error": out["error"]},
        }
    fills = trades_to_rows(out.get("trades") or [])
    tz = cfg.session_tz
    lo_ms = int(pd.Timestamp(f"{lo} 00:00:00", tz=tz).value // 1_000_000)
    hi_ms = int(pd.Timestamp(f"{hi} 23:59:59", tz=tz).value // 1_000_000)
    in_range_trades = [
        t for t in (out.get("trades") or []) if lo_ms <= int(t.datetime.timestamp() * 1000) <= hi_ms
    ]
    opens = count_opens(in_range_trades)
    in_range = [f for f in fills if lo_ms <= int(f["ms"]) <= hi_ms]
    end_w = float(out.get("end_wallet") or bt_cfg.equity_usdt)
    net = end_w - float(bt_cfg.equity_usdt)
    return {
        "symbol": label,
        "strategy": strategy_key,
        "title": meta["title"],
        "summary": {
            "net_pnl_usdt": round(net, 2),
            "opens": opens,
            "fills": len(in_range),
            "equity_end": round(end_w, 2),
            "engine": "vnpy",
        },
    }


def run_one_legacy(
    strategy_key: str,
    symbol: str,
    dates: List[str],
    *,
    cfg: OrbConfig,
    equity: float,
    risk_pct: float,
) -> Dict[str, Any]:
    from orb.cta.engine import run_cta_backtest  # noqa: E402
    from orb.cta.registry import cta_config_for_strategy  # noqa: E402
    from tools.cta.research_vnpy_cta_legacy import _session_slice  # noqa: E402

    meta = CTA_STRATEGIES[strategy_key]
    sym = norm_symbol(symbol)
    df1 = load_klines(sym, "1m")
    if df1.empty:
        return {"symbol": sym.replace("USDT", ""), "strategy": strategy_key, "summary": {"net_pnl_usdt": 0, "opens": 0}}
    chunks = [_session_slice(df1, d, cfg) for d in dates if d]
    df = pd.concat([c for c in chunks if not c.empty], ignore_index=True)
    if df.empty:
        return {"symbol": sym.replace("USDT", ""), "strategy": strategy_key, "summary": {"net_pnl_usdt": 0, "opens": 0}}
    out = run_cta_backtest(
        df,
        strategy_fn=meta["fn"],
        orb_cfg=cfg,
        cta_cfg=cta_config_for_strategy(
            strategy_key,
            equity_usdt=float(equity),
            risk_pct=float(risk_pct),
            compound=True,
            eod_flat=bool(meta.get("eod_flat")),
        ),
        warmup=int(meta.get("warmup") or 30),
    )
    s = out["summary"]
    return {
        "symbol": sym.replace("USDT", ""),
        "strategy": strategy_key,
        "title": meta["title"],
        "summary": {
            "net_pnl_usdt": float(s.get("net_pnl_usdt") or 0),
            "opens": int(s.get("opens") or 0),
            "equity_end": float(s.get("equity_end") or equity),
            "engine": "legacy_1m_touch",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Research vnpy CTA strategies (official BacktestingEngine)")
    ap.add_argument("--symbol", default="")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--symbols-file", default=str(resolve_symbols_path()))
    ap.add_argument("--strategy", default="all")
    ap.add_argument("--from-date", default="2026-02-01")
    ap.add_argument("--to-date", default="2026-06-30")
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--risk-pct", type=float, default=0.01)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--legacy", action="store_true", help="deprecated 1m touch engine")
    args = ap.parse_args()

    cfg = OrbConfig.from_env()
    lo, hi = args.from_date.strip(), args.to_date.strip()
    if (args.symbols or "").strip():
        symbols = [norm_symbol(s.strip()) for s in args.symbols.split(",") if s.strip()]
    elif (args.symbol or "").strip():
        symbols = [norm_symbol(args.symbol.strip())]
    else:
        symbols = parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8"))

    keys = list_vnpy_strategies() if args.strategy == "all" else [args.strategy.strip()]
    for k in keys:
        if k not in VNPY_CTA_STRATEGIES:
            print(f"Unknown strategy: {k}")
            return 1

    if args.legacy:
        warnings.warn("research_vnpy_cta --legacy 使用 1m 触价引擎，显著乐观", stacklevel=1)
        ref = symbols[0]
        dates = [d for d in session_dates_from_cache(ref, cfg) if lo <= d <= hi]
        engine_tag = "legacy_1m_touch"
    else:
        fetch_start, engine_start_s, replay_start = range_engine_dates(lo, cfg)
        replay_end = range_replay_end(hi, cfg)
        dates = []
        engine_tag = "vnpy"

    bt_cfg = CtaVnpyBacktestConfig(equity_usdt=float(args.equity), risk_pct=float(args.risk_pct), compound=True)
    print(
        f"[cta research] {lo}..{hi} | {len(symbols)} sym | eq={args.equity} | engine={engine_tag}",
        flush=True,
    )
    t0 = time.time()
    results: List[Dict[str, Any]] = []
    for k in keys:
        total_net = 0.0
        total_opens = 0
        per_sym = []
        for sym in symbols:
            if args.legacy:
                r = run_one_legacy(k, sym, dates, cfg=cfg, equity=float(args.equity), risk_pct=float(args.risk_pct))
            else:
                r = run_one_vnpy(
                    k,
                    sym,
                    cfg=cfg,
                    bt_cfg=bt_cfg,
                    lo=lo,
                    hi=hi,
                    fetch_start=fetch_start,
                    engine_start_s=engine_start_s,
                    replay_start=replay_start,
                    replay_end=replay_end,
                )
            s = r["summary"]
            total_net += float(s.get("net_pnl_usdt") or 0)
            total_opens += int(s.get("opens") or 0)
            per_sym.append(
                {
                    "symbol": r["symbol"],
                    "net_pnl_usdt": s.get("net_pnl_usdt"),
                    "opens": s.get("opens"),
                    "fills": s.get("fills"),
                    "equity_end": s.get("equity_end"),
                    "engine": s.get("engine"),
                }
            )
        row = {
            "strategy": k,
            "title": VNPY_CTA_STRATEGIES[k]["title"],
            "net_pnl_usdt": round(total_net, 2),
            "opens": total_opens,
            "engine": engine_tag,
            "per_symbol": per_sym,
        }
        results.append(row)
        print(
            f"  {k:14s} {VNPY_CTA_STRATEGIES[k]['title']:22s} net={total_net:+.2f}U opens={total_opens}",
            flush=True,
        )

    out_dir = ROOT / "output" / "orb" / "cta"
    out_dir.mkdir(parents=True, exist_ok=True)
    sym_tag = symbols[0].replace("USDT", "") if len(symbols) == 1 else f"pool{len(symbols)}"
    out_path = (
        Path(args.json_out)
        if args.json_out
        else out_dir / f"vnpy_cta_{sym_tag}_eq{int(args.equity)}_{lo}_{hi}.json"
    )
    payload = {
        "date_range": {"from": lo, "to": hi},
        "equity_usdt": float(args.equity),
        "risk_pct": float(args.risk_pct),
        "engine": engine_tag,
        "symbols": [norm_symbol(s) for s in symbols],
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\njson -> {out_path} ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

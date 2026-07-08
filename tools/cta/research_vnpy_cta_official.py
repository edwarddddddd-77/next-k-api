#!/usr/bin/env python3
"""严格按 vnpy 官方 backtesting_demo.ipynb 流程回测 CTA 示例策略。

与 orb/cta/vnpy/backtest.py 的区别：
- 不猴子补丁 load_data
- 不过滤 RTH
- 不用复利子类 / KK 会话规则
- add_strategy(Class, {}) 空参数，走策略类默认值
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Type

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402

load_env_oi()

import pandas as pd  # noqa: E402

from binance_fapi import fetch_klines_forward, klines_to_df  # noqa: E402
from orb.core.kline_cache import load_klines, norm_symbol  # noqa: E402
from orb.core.symbols import parse_symbol_list  # noqa: E402
from orb.core.symbols_path import resolve_symbols_path  # noqa: E402
from orb.vnpy.bootstrap import ensure_vnpy_path  # noqa: E402

ensure_vnpy_path()

from vnpy.trader.constant import Exchange, Interval  # noqa: E402
from vnpy.trader.database import get_database  # noqa: E402
from vnpy.trader.object import BarData  # noqa: E402
from vnpy.trader.setting import SETTINGS  # noqa: E402
from vnpy_ctastrategy import CtaTemplate  # noqa: E402
from vnpy_ctastrategy.backtesting import BacktestingEngine, load_bar_data  # noqa: E402
from vnpy_ctastrategy.strategies.atr_rsi_strategy import AtrRsiStrategy  # noqa: E402
from vnpy_ctastrategy.strategies.boll_channel_strategy import BollChannelStrategy  # noqa: E402
from vnpy_ctastrategy.strategies.double_ma_strategy import DoubleMaStrategy  # noqa: E402
from vnpy_ctastrategy.strategies.dual_thrust_strategy import DualThrustStrategy  # noqa: E402
from vnpy_ctastrategy.strategies.king_keltner_strategy import KingKeltnerStrategy  # noqa: E402
from vnpy_ctastrategy.strategies.turtle_signal_strategy import TurtleSignalStrategy  # noqa: E402

OFFICIAL_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "double_ma": {"title": "双均线金叉死叉", "class": DoubleMaStrategy},
    "atr_rsi": {"title": "ATR放大 + RSI极端", "class": AtrRsiStrategy},
    "boll_channel": {"title": "布林通道 + CCI + ATR止损", "class": BollChannelStrategy},
    "king_keltner": {"title": "肯特纳通道突破", "class": KingKeltnerStrategy},
    "dual_thrust": {"title": "Dual Thrust 日内突破", "class": DualThrustStrategy},
    "turtle": {"title": "海龟唐奇安通道", "class": TurtleSignalStrategy},
}


def _vt_symbol(symbol: str) -> str:
    sym = norm_symbol(symbol)
    return f"{sym}_SWAP_BINANCE.GLOBAL"


def _bar_symbol(vt_symbol: str) -> str:
    return str(vt_symbol).split(".", 1)[0]


def _pricetick(price: float) -> float:
    px = float(price or 1.0)
    if px >= 1000:
        return 0.001
    if px >= 100:
        return 0.01
    if px >= 10:
        return 0.1
    return 0.01


def _configure_sqlite_db() -> Path:
    fd, name = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_path = Path(name)
    SETTINGS["database.name"] = "sqlite"
    SETTINGS["database.database"] = str(db_path)
    return db_path


def _fetch_range(sym: str, from_date: str, to_date: str) -> pd.DataFrame:
    lo = pd.Timestamp(from_date.strip(), tz="America/New_York")
    hi = pd.Timestamp(to_date.strip(), tz="America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    rows = fetch_klines_forward(sym, "1m", int(lo.value // 1_000_000), int(hi.value // 1_000_000))
    df = klines_to_df(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time").reset_index(drop=True)


def _load_symbol_df(sym: str, from_date: str, to_date: str) -> pd.DataFrame:
    lo = pd.Timestamp(from_date.strip(), tz="America/New_York")
    hi = pd.Timestamp(to_date.strip(), tz="America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    lo_ms, hi_ms = int(lo.value // 1_000_000), int(hi.value // 1_000_000)
    df = load_klines(sym, "1m")
    if df is not None and not df.empty:
        sl = df[(df["open_time"] >= lo_ms) & (df["open_time"] <= hi_ms)].copy()
        if not sl.empty:
            return sl.sort_values("open_time").reset_index(drop=True)
    return _fetch_range(sym, from_date, to_date)


def _df_to_bars(df: pd.DataFrame, vt_symbol: str) -> List[BarData]:
    sym = _bar_symbol(vt_symbol)
    bars: List[BarData] = []
    for _, row in df.iterrows():
        ms = int(row["open_time"])
        dt = datetime.fromtimestamp(ms / 1000.0)
        bars.append(
            BarData(
                symbol=sym,
                exchange=Exchange.GLOBAL,
                datetime=dt,
                interval=Interval.MINUTE,
                open_price=float(row["open"]),
                high_price=float(row["high"]),
                low_price=float(row["low"]),
                close_price=float(row["close"]),
                volume=float(row.get("volume") or 0.0),
                gateway_name="BACKTESTING",
            )
        )
    return bars


def _count_opens(trades: dict) -> int:
    from vnpy.trader.constant import Offset

    return sum(1 for t in trades.values() if t.offset == Offset.OPEN)


def run_official_backtest(
    strategy_key: str,
    symbol: str,
    *,
    df: pd.DataFrame,
    start: datetime,
    end: datetime,
    capital: int,
    quiet: bool = True,
) -> Dict[str, Any]:
    meta = OFFICIAL_STRATEGIES[strategy_key]
    strat_cls: Type[CtaTemplate] = meta["class"]
    vt_symbol = _vt_symbol(symbol)
    label = norm_symbol(symbol).replace("USDT", "")

    if df.empty:
        return {
            "symbol": label,
            "strategy": strategy_key,
            "error": "no_data",
            "summary": {"net_pnl": 0, "end_balance": capital, "total_trade_count": 0},
        }

    px = float(df.iloc[-1]["close"])
    bars = _df_to_bars(df, vt_symbol)
    db_path = _configure_sqlite_db()
    try:
        load_bar_data.cache_clear()
        get_database().save_bar_data(bars)

        engine = BacktestingEngine()
        if quiet:
            engine.output = lambda msg: None  # type: ignore[method-assign]

        # 与官方 demo 相同调用方式（参数结构一致；vt_symbol/合约乘数适配 Binance 股票代币）
        engine.set_parameters(
            vt_symbol=vt_symbol,
            interval=Interval.MINUTE,
            start=start,
            end=end,
            rate=0.3 / 10_000,
            slippage=0.2,
            size=1,
            pricetick=_pricetick(px),
            capital=capital,
        )
        engine.add_strategy(strat_cls, {})
        engine.load_data()
        if not engine.history_data:
            return {
                "symbol": label,
                "strategy": strategy_key,
                "error": "no_history_data",
                "summary": {"net_pnl": 0, "end_balance": capital, "total_trade_count": 0},
            }
        engine.run_backtesting()
        engine.calculate_result()
        stats = engine.calculate_statistics(output=False)
        net = float(stats.get("total_net_pnl") or 0)
        end_bal = float(stats.get("end_balance") or capital)
        opens = _count_opens(engine.trades)
        return {
            "symbol": label,
            "strategy": strategy_key,
            "title": meta["title"],
            "summary": {
                "net_pnl": round(net, 2),
                "end_balance": round(end_bal, 2),
                "total_trade_count": int(stats.get("total_trade_count") or 0),
                "opens": opens,
                "total_return": stats.get("total_return"),
                "sharpe_ratio": stats.get("sharpe_ratio"),
                "max_ddpercent": stats.get("max_ddpercent"),
            },
            "engine": "vnpy_official",
        }
    finally:
        try:
            db_path.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Strict vnpy official BacktestingEngine (demo flow)")
    ap.add_argument("--symbol", default="")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--symbols-file", default=str(resolve_symbols_path()))
    ap.add_argument("--strategy", default="all")
    ap.add_argument("--from-date", default="2026-02-01")
    ap.add_argument("--to-date", default="2026-06-30")
    ap.add_argument("--capital", type=int, default=1_000_000, help="same default as official demo")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    lo, hi = args.from_date.strip(), args.to_date.strip()
    start = datetime.strptime(lo, "%Y-%m-%d")
    end = datetime.strptime(hi, "%Y-%m-%d")

    if (args.symbols or "").strip():
        symbols = [norm_symbol(s.strip()) for s in args.symbols.split(",") if s.strip()]
    elif (args.symbol or "").strip():
        symbols = [norm_symbol(args.symbol.strip())]
    else:
        symbols = parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8"))

    keys = list(OFFICIAL_STRATEGIES) if args.strategy == "all" else [args.strategy.strip()]
    for k in keys:
        if k not in OFFICIAL_STRATEGIES:
            print(f"Unknown strategy: {k}")
            return 1

    print(
        f"[official vnpy] {lo}..{hi} | {len(symbols)} sym | capital={args.capital} | setting={{}}",
        flush=True,
    )
    t0 = time.time()
    data_cache: Dict[str, pd.DataFrame] = {}
    results: List[Dict[str, Any]] = []

    for k in keys:
        total_net = 0.0
        total_opens = 0
        per_sym = []
        for sym in symbols:
            if sym not in data_cache:
                data_cache[sym] = _load_symbol_df(sym, lo, hi)
            r = run_official_backtest(
                k,
                sym,
                df=data_cache[sym],
                start=start,
                end=end,
                capital=int(args.capital),
                quiet=True,
            )
            s = r.get("summary") or {}
            total_net += float(s.get("net_pnl") or 0)
            total_opens += int(s.get("opens") or 0)
            per_sym.append(
                {
                    "symbol": r["symbol"],
                    "net_pnl": s.get("net_pnl"),
                    "end_balance": s.get("end_balance"),
                    "opens": s.get("opens"),
                    "total_trade_count": s.get("total_trade_count"),
                    "error": r.get("error"),
                }
            )
        row = {
            "strategy": k,
            "title": OFFICIAL_STRATEGIES[k]["title"],
            "net_pnl": round(total_net, 2),
            "opens": total_opens,
            "engine": "vnpy_official",
            "per_symbol": per_sym,
        }
        results.append(row)
        print(
            f"  {k:14s} {OFFICIAL_STRATEGIES[k]['title']:22s} net={total_net:+.2f} opens={total_opens}",
            flush=True,
        )

    out_dir = ROOT / "output" / "orb" / "cta"
    out_dir.mkdir(parents=True, exist_ok=True)
    sym_tag = symbols[0].replace("USDT", "") if len(symbols) == 1 else f"pool{len(symbols)}"
    out_path = (
        Path(args.json_out)
        if args.json_out
        else out_dir / f"official_vnpy_{sym_tag}_cap{int(args.capital)}_{lo}_{hi}.json"
    )
    payload = {
        "date_range": {"from": lo, "to": hi},
        "capital": int(args.capital),
        "engine": "vnpy_official",
        "setting": {},
        "engine_params": {
            "rate": 0.3 / 10_000,
            "slippage": 0.2,
            "size": 1,
            "note": "vt_symbol/pricetick adapted for Binance stock tokens; flow matches backtesting_demo.ipynb",
        },
        "symbols": [norm_symbol(s) for s in symbols],
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\njson -> {out_path} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

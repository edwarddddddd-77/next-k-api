"""GTL vnpy backtest without RTH filter (24h structural research)."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Type

import pandas as pd

from orb.core.kline_cache import norm_symbol
from orb.cta.vnpy.registry import VNPY_CTA_STRATEGIES, get_vnpy_strategy_class
from orb.vnpy.bootstrap import ensure_vnpy_path
from orb.vnpy.binance_gateway import vnpy_vt_symbol

ensure_vnpy_path()

from vnpy.trader.constant import Exchange, Interval, Offset  # noqa: E402
from vnpy.trader.database import get_database  # noqa: E402
from vnpy.trader.object import BarData  # noqa: E402
from vnpy.trader.setting import SETTINGS  # noqa: E402
from vnpy_ctastrategy import CtaTemplate  # noqa: E402
from vnpy_ctastrategy.backtesting import BacktestingEngine, load_bar_data  # noqa: E402


def _force_flat_at_close(engine: BacktestingEngine) -> None:
    """Close open position at last bar close (limit orders need one more cross)."""
    strat = engine.strategy
    if not getattr(strat, "force_flat_on_stop", False) or not strat.pos:
        return
    bar = getattr(strat, "_last_bar", None)
    if bar is None or not engine.history_data:
        return
    strat.cancel_all()
    px = float(bar.close_price)
    if strat.pos > 0:
        strat.sell(px, abs(strat.pos))
    elif strat.pos < 0:
        strat.cover(px, abs(strat.pos))
    engine.cross_limit_order()


def _vt_symbol(symbol: str) -> str:
    return vnpy_vt_symbol(norm_symbol(symbol))


def _pricetick(price: float) -> float:
    px = float(price or 1.0)
    if px >= 1000:
        return 0.001
    if px >= 100:
        return 0.01
    if px >= 10:
        return 0.1
    return 0.01


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _configure_sqlite_db() -> Path:
    fd, name = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_path = Path(name)
    SETTINGS["database.name"] = "sqlite"
    SETTINGS["database.database"] = str(db_path)
    return db_path


def _df_to_bars(df: pd.DataFrame, vt_symbol: str) -> List[BarData]:
    sym = vt_symbol.split(".", 1)[0]
    bars: List[BarData] = []
    for _, row in df.iterrows():
        ms = int(row["open_time"])
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
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
    return sum(1 for t in trades.values() if t.offset == Offset.OPEN)


def _round_trip_stats(trades: dict) -> dict[str, float | int]:
    """Realized round-trip PnL excluding any still-open leg at backtest end."""
    from vnpy.trader.constant import Direction

    ordered = sorted(trades.values(), key=lambda t: t.datetime or datetime.min.replace(tzinfo=timezone.utc))
    entry = None
    realized: list[float] = []
    for t in ordered:
        px = float(t.price)
        if t.offset == Offset.OPEN:
            entry = {"side": t.direction, "px": px}
        elif t.offset == Offset.CLOSE and entry:
            if entry["side"] == Direction.LONG and t.direction == Direction.SHORT:
                realized.append(px - entry["px"])
            elif entry["side"] == Direction.SHORT and t.direction == Direction.LONG:
                realized.append(entry["px"] - px)
            entry = None
    if not realized:
        return {"realized_round_trips": 0, "realized_pnl": 0.0, "realized_win_rate": 0.0}
    wins = sum(1 for p in realized if p > 0)
    return {
        "realized_round_trips": len(realized),
        "realized_pnl": round(float(sum(realized)), 2),
        "realized_win_rate": round(wins / len(realized), 3),
    }


def _buy_hold_move(df: pd.DataFrame, start: datetime, end: datetime) -> float:
    ms_start = int(_as_utc(start).timestamp() * 1000)
    ms_end = int(_as_utc(end).timestamp() * 1000)
    sl = df[(df["open_time"] >= ms_start) & (df["open_time"] <= ms_end)]
    if len(sl) < 2:
        return 0.0
    return round(float(sl.iloc[-1]["close"]) - float(sl.iloc[0]["close"]), 2)


def run_gtl_vnpy_backtest(
    strategy_key: str,
    symbol: str,
    *,
    df: pd.DataFrame,
    start: datetime,
    end: datetime,
    capital: float = 1000.0,
    quiet: bool = True,
    strategy_overrides: dict | None = None,
) -> Dict[str, Any]:
    meta = VNPY_CTA_STRATEGIES[strategy_key]
    strat_cls: Type[CtaTemplate] = get_vnpy_strategy_class(strategy_key)
    sym = norm_symbol(symbol)
    vt_symbol = _vt_symbol(sym)
    label = sym.replace("USDT", "")

    if df.empty:
        return {"symbol": label, "strategy": strategy_key, "error": "no_data", "summary": {}}

    setting = dict(meta.get("default_setting") or {})
    setting["fixed_size"] = 1
    if strategy_overrides:
        setting.update(strategy_overrides)
    px = float(df.iloc[-1]["close"])
    bars = _df_to_bars(df, vt_symbol)
    db_path = _configure_sqlite_db()
    try:
        load_bar_data.cache_clear()
        get_database().save_bar_data(bars)

        engine = BacktestingEngine()
        if quiet:
            engine.output = lambda msg: None  # type: ignore[method-assign]

        engine.set_parameters(
            vt_symbol=vt_symbol,
            interval=Interval.MINUTE,
            start=_as_utc(start),
            end=_as_utc(end),
            rate=0.3 / 10_000,
            slippage=0.2,
            size=1,
            pricetick=_pricetick(px),
            capital=int(capital),
        )
        engine.add_strategy(strat_cls, setting)
        engine.load_data()
        if not engine.history_data:
            return {"symbol": label, "strategy": strategy_key, "error": "no_history_data", "summary": {}}
        engine.run_backtesting()
        _force_flat_at_close(engine)
        engine.calculate_result()
        stats = engine.calculate_statistics(output=False)
        net = float(stats.get("total_net_pnl") or 0)
        rt = _round_trip_stats(engine.trades)
        bh = _buy_hold_move(df, start, end)
        return {
            "symbol": label,
            "strategy": strategy_key,
            "title": meta["title"],
            "summary": {
                "net_pnl": round(net, 2),
                "realized_pnl": rt["realized_pnl"],
                "realized_round_trips": rt["realized_round_trips"],
                "realized_win_rate": rt["realized_win_rate"],
                "buy_hold_move": bh,
                "opens": _count_opens(engine.trades),
                "end_balance": round(float(stats.get("end_balance") or capital), 2),
                "total_trade_count": int(stats.get("total_trade_count") or 0),
                "sharpe_ratio": stats.get("sharpe_ratio"),
                "max_ddpercent": stats.get("max_ddpercent"),
            },
            "engine": "vnpy_gtl",
        }
    finally:
        try:
            db_path.unlink(missing_ok=True)
        except OSError:
            pass

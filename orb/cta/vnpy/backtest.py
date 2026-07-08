"""vnpy 官方 BacktestingEngine 通用封装（CTA 示例策略）。"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pandas as pd

from orb.core.config import OrbConfig
from orb.core.kline_cache import norm_symbol
from orb.core.session import (
    effective_session_close_time,
    is_trading_session,
    session_anchor_ms,
    session_close_ms,
)
from orb.cta.vnpy.compound import make_compound_backtest_class
from orb.cta.vnpy.registry import VNPY_CTA_STRATEGIES, get_vnpy_strategy_class
from orb.cta.vnpy.sizing import fixed_size_for_equity
from orb.vnpy.bootstrap import ensure_vnpy_path
from orb.vnpy.binance_gateway import vnpy_vt_symbol

ensure_vnpy_path()

from vnpy.trader.constant import Exchange, Interval  # noqa: E402
from vnpy.trader.database import get_database  # noqa: E402
from vnpy.trader.object import BarData, TradeData  # noqa: E402
from vnpy.trader.setting import SETTINGS  # noqa: E402
from vnpy_ctastrategy import CtaTemplate  # noqa: E402
from vnpy_ctastrategy.backtesting import BacktestingEngine  # noqa: E402


@dataclass
class CtaVnpyBacktestConfig:
    equity_usdt: float = 1000.0
    risk_pct: float = 0.01
    compound: bool = True
    fee_taker_bps: float = 4.0
    slip_bps_entry: float = 5.0
    trail_risk_frac: float = 0.008
    max_notional_usdt: float = 0.0


def session_bounds_for_date(session_date: str, cfg: OrbConfig) -> tuple[datetime, datetime, int, int, str]:
    tz = cfg.session_tz
    ts = pd.Timestamp(f"{session_date.strip()} 12:00:00", tz=tz)
    anchor_ms = session_anchor_ms(
        int(ts.value // 1_000_000),
        tz=tz,
        session_open_time=cfg.session_open_time,
    )
    close_time = effective_session_close_time(
        int(anchor_ms),
        tz=tz,
        session_open_time=cfg.session_open_time,
        session_close_time=cfg.session_close_time,
        market=cfg.market,
    )
    close_ms = session_close_ms(anchor_ms, tz=tz, session_close_time=close_time)
    if close_ms is None:
        close_ms = anchor_ms + 6 * 60 * 60 * 1000
    start = datetime.fromtimestamp(anchor_ms / 1000.0, tz=timezone.utc)
    end = datetime.fromtimestamp(close_ms / 1000.0, tz=timezone.utc)
    return start, end, int(anchor_ms), int(close_ms), close_time


def filter_rth_bars(bars: List[BarData], cfg: OrbConfig) -> List[BarData]:
    out: List[BarData] = []
    for b in bars:
        ms = _bar_ms(b.datetime)
        if is_trading_session(
            ms,
            tz=cfg.session_tz,
            session_open_time=cfg.session_open_time,
            session_close_time=cfg.session_close_time,
            market=cfg.market,
        ):
            out.append(b)
    return out


def _configure_sqlite_db(path: Optional[Path] = None) -> Path:
    if path is None:
        fd, name = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db_path = Path(name)
    else:
        db_path = path
    SETTINGS["database.name"] = "sqlite"
    SETTINGS["database.database"] = str(db_path)
    return db_path


def bar_symbol_from_vt(vt_symbol: str) -> str:
    return str(vt_symbol or "").split(".", 1)[0]


def _bar_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def klines_df_to_bars(df: pd.DataFrame, symbol: str, *, vt_symbol: Optional[str] = None) -> List[BarData]:
    sym = bar_symbol_from_vt(vt_symbol) if vt_symbol else norm_symbol(symbol)
    bars: List[BarData] = []
    for _, row in df.iterrows():
        ms = int(row["open_time"])
        dt = _as_utc(datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc))
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


def pricetick_for(price: float) -> float:
    px = float(price or 1.0)
    if px >= 1000:
        return 0.001
    if px >= 100:
        return 0.01
    if px >= 10:
        return 0.1
    return 0.01


def backtest_engine_params(cfg: CtaVnpyBacktestConfig, *, price: float) -> Dict[str, Any]:
    slip_bps = float(cfg.slip_bps_entry or 5.0)
    px = max(1e-9, float(price))
    tick = pricetick_for(px)
    slippage = max(tick, px * slip_bps / 10_000.0)
    rate = float(cfg.fee_taker_bps or 4.0) / 10_000.0
    return {
        "rate": rate,
        "slippage": slippage,
        "size": 1.0,
        "pricetick": tick,
        "capital": int(max(1.0, float(cfg.equity_usdt))),
    }


def _prepare_bars(bars: List[BarData], *, gateway_name: str = "BACKTESTING") -> List[BarData]:
    out: List[BarData] = []
    for b in bars:
        gw = getattr(b, "gateway_name", None) or gateway_name
        out.append(
            BarData(
                symbol=b.symbol,
                exchange=b.exchange,
                datetime=_as_utc(b.datetime),
                interval=b.interval or Interval.MINUTE,
                open_price=float(b.open_price),
                high_price=float(b.high_price),
                low_price=float(b.low_price),
                close_price=float(b.close_price),
                volume=float(b.volume or 0.0),
                turnover=float(getattr(b, "turnover", 0.0) or 0.0),
                open_interest=float(getattr(b, "open_interest", 0.0) or 0.0),
                gateway_name=gw,
            )
        )
    return out


def save_bars(bars: List[BarData]) -> None:
    if not bars:
        return
    db = get_database()
    db.save_bar_data(_prepare_bars(bars))


def _sizing_fn(bt_cfg: CtaVnpyBacktestConfig, orb_cfg: OrbConfig):
    safety = float(getattr(orb_cfg, "position_safety_pct", 0.0) or 0.0)

    def _fn(price: float, equity: float) -> float:
        return fixed_size_for_equity(
            price,
            equity_usdt=equity,
            risk_pct=bt_cfg.risk_pct,
            trail_frac=bt_cfg.trail_risk_frac,
            safety_pct=safety,
            max_notional_usdt=bt_cfg.max_notional_usdt,
        )

    return _fn


def resolve_strategy_class(
    strategy_key: str,
    *,
    bt_cfg: CtaVnpyBacktestConfig,
    orb_cfg: OrbConfig,
) -> Type[CtaTemplate]:
    meta = VNPY_CTA_STRATEGIES[strategy_key]
    base = get_vnpy_strategy_class(strategy_key)
    use_compound = bt_cfg.compound and meta.get("compound", True)
    if not use_compound:
        return base
    fee_rate = float(bt_cfg.fee_taker_bps or 4.0) / 10_000.0
    return make_compound_backtest_class(base, sizing_fn=_sizing_fn(bt_cfg, orb_cfg), fee_rate=fee_rate)


def build_strategy_setting(
    strategy_key: str,
    symbol: str,
    *,
    bt_cfg: CtaVnpyBacktestConfig,
    price: float,
    orb_cfg: OrbConfig,
) -> dict:
    sym = norm_symbol(symbol)
    setting: dict = {}
    meta = VNPY_CTA_STRATEGIES[strategy_key]
    if meta.get("default_setting"):
        setting.update(dict(meta["default_setting"]))

    sizing = _sizing_fn(bt_cfg, orb_cfg)
    vol = sizing(price, bt_cfg.equity_usdt)
    if meta.get("uses_fixed_size") or not meta.get("uses_fixed_size"):
        setting["fixed_size"] = vol
    if bt_cfg.compound and meta.get("compound", True):
        setting["bt_wallet"] = float(bt_cfg.equity_usdt)
    return setting


def run_vnpy_cta_backtest(
    symbol: str,
    bars: List[BarData],
    *,
    strategy_key: str,
    bt_cfg: CtaVnpyBacktestConfig,
    start: datetime,
    end: datetime,
    price: float = 100.0,
    db_path: Optional[Path] = None,
    quiet: bool = False,
    replay_start: Optional[datetime] = None,
    replay_end: Optional[datetime] = None,
    orb_cfg: Optional[OrbConfig] = None,
) -> Dict[str, Any]:
    sym = norm_symbol(symbol)
    vt_symbol = vnpy_vt_symbol(sym)
    norm_bars = _prepare_bars(bars if bars else [])
    if bars and bars[0].symbol != bar_symbol_from_vt(vt_symbol):
        norm_bars = _prepare_bars(
            klines_df_to_bars(
                pd.DataFrame(
                    [
                        {
                            "open_time": int(b.datetime.timestamp() * 1000),
                            "open": b.open_price,
                            "high": b.high_price,
                            "low": b.low_price,
                            "close": b.close_price,
                            "volume": b.volume,
                        }
                        for b in bars
                    ]
                ),
                sym,
                vt_symbol=vt_symbol,
            )
        )
    if db_path is None:
        db_path = _configure_sqlite_db()
    else:
        _configure_sqlite_db(db_path)

    from vnpy_ctastrategy.backtesting import load_bar_data as vnpy_load_bar_data

    vnpy_load_bar_data.cache_clear()
    save_bars(norm_bars)

    ocfg = orb_cfg or OrbConfig.from_env()
    params = backtest_engine_params(bt_cfg, price=price)
    setting = build_strategy_setting(strategy_key, sym, bt_cfg=bt_cfg, price=price, orb_cfg=ocfg)
    strat_cls = resolve_strategy_class(strategy_key, bt_cfg=bt_cfg, orb_cfg=ocfg)

    lo = _as_utc(replay_start or start)
    hi = _as_utc(replay_end or end)
    lo_ms, hi_ms = _bar_ms(lo), _bar_ms(hi)
    replay_bars = [b for b in norm_bars if lo_ms <= _bar_ms(b.datetime) <= hi_ms]
    replay_bars = filter_rth_bars(replay_bars, ocfg)

    engine = BacktestingEngine()
    if quiet:
        engine.output = lambda msg: None  # type: ignore[method-assign]

    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=Interval.MINUTE,
        start=_as_utc(start),
        end=_as_utc(end),
        **params,
    )
    engine.add_strategy(strat_cls, setting)

    def _load_replay_data() -> None:
        engine.history_data = list(replay_bars)
        if not quiet:
            engine.output(f"历史数据加载完成，数据量：{len(replay_bars)}")

    engine.load_data = _load_replay_data  # type: ignore[method-assign]
    engine.load_data()
    if not engine.history_data:
        return {
            "symbol": sym,
            "strategy": strategy_key,
            "trades": [],
            "statistics": {},
            "daily_df": pd.DataFrame(),
            "logs": [],
            "error": "no_history_data",
        }
    engine.run_backtesting()
    err = None
    for line in engine.logs:
        if "触发异常" in str(line) or "Traceback" in str(line):
            err = "backtest_aborted"
            break
    daily_df = engine.calculate_result()
    stats = engine.calculate_statistics(output=not quiet)

    end_wallet = float(bt_cfg.equity_usdt)
    strat = engine.strategy
    stats_end = float(stats.get("end_balance") or 0)
    if meta := VNPY_CTA_STRATEGIES.get(strategy_key):
        if meta.get("compound", True) and hasattr(strat, "bt_wallet"):
            end_wallet = float(getattr(strat, "bt_wallet", end_wallet))
        elif stats_end > 0:
            end_wallet = stats_end
    elif stats_end > 0:
        end_wallet = stats_end

    return {
        "symbol": sym,
        "strategy": strategy_key,
        "trades": list(engine.trades.values()),
        "statistics": stats,
        "daily_df": daily_df,
        "logs": list(engine.logs),
        "engine": engine,
        "error": err,
        "end_wallet": end_wallet,
        "net_pnl_usdt": end_wallet - float(bt_cfg.equity_usdt),
    }


def trades_to_rows(trades: List[TradeData]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t in sorted(trades, key=lambda x: x.datetime or datetime.min.replace(tzinfo=timezone.utc)):
        rows.append(
            {
                "datetime": t.datetime,
                "ms": int(t.datetime.timestamp() * 1000) if t.datetime else 0,
                "direction": t.direction.value,
                "offset": t.offset.value,
                "price": float(t.price),
                "volume": float(t.volume),
            }
        )
    return rows


def range_engine_dates(lo: str, cfg: OrbConfig) -> tuple[str, str, datetime]:
    tz = cfg.session_tz
    engine_start = (pd.Timestamp(lo.strip(), tz=tz) - pd.Timedelta(days=12)).strftime("%Y-%m-%d")
    fetch_start = (pd.Timestamp(engine_start, tz=tz) - pd.Timedelta(days=12)).strftime("%Y-%m-%d")
    replay_start, _, _, _, _ = session_bounds_for_date(lo, cfg)
    return fetch_start, engine_start, replay_start


def range_replay_end(hi: str, cfg: OrbConfig) -> datetime:
    _, replay_end, _, _, _ = session_bounds_for_date(hi, cfg)
    return replay_end

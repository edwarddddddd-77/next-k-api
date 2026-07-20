"""ICT Fractal T-Spot / CISD event-driven backtest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from quant.common.fees import fee_taker_bps_from_env, trade_fee_usdt
from quant.common.kline_cache import load_klines, norm_symbol, save_klines
from quant.common.resample import resample_ohlcv
from quant.fractal_ict.config import FractalIctConfig, _interval_minutes
from quant.fractal_ict.core import (
    ActiveSetup,
    BarOhlc,
    TSpotSetup,
    bar_hits_stop_tp,
    detect_tspot,
    entry_signal_c3,
    entry_signal_cisd,
    position_qty,
    stop_tp_from_setup,
)
from quant.market import fetch_klines_forward, klines_to_df


@dataclass
class FractalTrade:
    symbol: str
    side: str
    pattern: str
    entry_mode: str
    entry_ms: int
    exit_ms: int
    entry_price: float
    exit_price: float
    stop_price: float
    tp_price: float
    exit_reason: str
    qty: float
    pnl_gross_usdt: float
    fee_usdt: float
    pnl_net_usdt: float


@dataclass
class SymbolBacktestResult:
    symbol: str
    bars: int
    htf_interval: str
    trades: List[FractalTrade] = field(default_factory=list)
    total_pnl_net: float = 0.0
    win_rate: float = 0.0
    max_drawdown_usdt: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["trades"] = [asdict(t) for t in self.trades]
        return d


@dataclass
class BacktestParams:
    symbols: List[str]
    days: int = 60
    exchange_id: str = "binance"
    interval: str = "5m"
    refresh: bool = False
    cfg: Optional[FractalIctConfig] = None

    def resolved_cfg(self) -> FractalIctConfig:
        c = self.cfg or FractalIctConfig.from_env()
        if not c.ltf_interval:
            c.ltf_interval = self.interval
        return c


def _apply_slippage(price: float, *, side: int, is_entry: bool, bps: float) -> float:
    slip = max(0.0, bps) / 10_000.0
    px = float(price)
    if is_entry:
        return px * (1.0 + slip) if side > 0 else px * (1.0 - slip)
    return px * (1.0 - slip) if side > 0 else px * (1.0 + slip)


def fetch_bars(
    symbol: str,
    interval: str,
    *,
    days: int,
    exchange_id: str = "binance",
    refresh: bool = False,
) -> pd.DataFrame:
    sym = norm_symbol(symbol)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - max(1, days) * 86_400_000
    cached = load_klines(sym, interval, start_ms=start_ms, end_ms=end_ms)
    span = 0.0
    if not cached.empty:
        span = (cached["open_time"].max() - cached["open_time"].min()) / 86_400_000
    if refresh or cached.empty or span < days * 0.85:
        for ex in (exchange_id, "bitget", "binance"):
            print(f"[fetch] {sym} {interval} {days}d from {ex} ...")
            rows = fetch_klines_forward(sym, interval, start_ms, end_ms, exchange_id=ex)
            fresh = klines_to_df(rows)
            if not fresh.empty:
                merged = fresh if cached.empty else (
                    pd.concat([cached, fresh]).drop_duplicates("open_time").sort_values("open_time")
                )
                save_klines(sym, interval, merged)
                cached = merged
                break
    return cached.sort_values("open_time").reset_index(drop=True)


def _df_to_bars(df: pd.DataFrame) -> list[BarOhlc]:
    return [
        (int(r.open_time), float(r.open), float(r.high), float(r.low), float(r.close))
        for r in df.itertuples(index=False)
    ]


def _htf_period_ms(htf_interval: str) -> int:
    return _interval_minutes(htf_interval) * 60_000


def _assign_htf_keys(ltf_df: pd.DataFrame, htf_interval: str) -> pd.Series:
    period_ms = _htf_period_ms(htf_interval)
    return (ltf_df["open_time"] // period_ms) * period_ms


def _find_extreme_bar_idx(
    ltf_bars: Sequence[BarOhlc],
    ltf_indices: Sequence[int],
    *,
    extreme: float,
    use_high: bool,
) -> int:
    for idx in ltf_indices:
        _, _o, h, l, _c = ltf_bars[idx]
        val = h if use_high else l
        if abs(val - extreme) < 1e-9 * max(abs(extreme), 1.0):
            return idx
    return ltf_indices[-1] if ltf_indices else 0


def _build_htf_context(
    ltf_df: pd.DataFrame,
    ltf_idx: int,
    htf_df: pd.DataFrame,
    htf_interval: str,
    ltf_htf_key: pd.Series,
) -> list[BarOhlc] | None:
    """Build [prev_prev, prev, last_closed(C2), current(C3)] at HTF roll."""
    period_ms = _htf_period_ms(htf_interval)
    cur_key = int(ltf_htf_key.iloc[ltf_idx])
    prev_key = cur_key - period_ms
    prior = htf_df[htf_df["open_time"] < prev_key].tail(2)
    last_closed_row = htf_df[htf_df["open_time"] == prev_key]
    if len(prior) < 2 or last_closed_row.empty:
        return None
    forming_mask = (ltf_htf_key == cur_key) & (ltf_df.index <= ltf_idx)
    forming = ltf_df.loc[forming_mask]
    if forming.empty:
        return None
    forming_bar: BarOhlc = (
        int(cur_key),
        float(forming["open"].iloc[0]),
        float(forming["high"].max()),
        float(forming["low"].min()),
        float(forming["close"].iloc[-1]),
    )
    return _df_to_bars(prior) + _df_to_bars(last_closed_row) + [forming_bar]


def _session_utc(hour: int) -> str:
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "ny"
    return "late"


def _is_range_market(
    ltf_df: pd.DataFrame,
    ltf_idx: int,
    *,
    max_pct: float = 0.015,
    lookback_bars: int = 288,
) -> bool:
    sub = ltf_df.iloc[: ltf_idx + 1].tail(lookback_bars + 1)
    if len(sub) < 20:
        return False
    ret = abs(sub["close"].iloc[-1] - sub["close"].iloc[0]) / sub["close"].iloc[0]
    return ret <= max_pct


def _passes_entry_filters(
    cfg: FractalIctConfig,
    ltf_df: pd.DataFrame,
    ltf_idx: int,
    bar_ms: int,
    setup: TSpotSetup,
) -> bool:
    if cfg.allowed_patterns and setup.pattern not in cfg.allowed_patterns:
        return False
    if cfg.range_only and not _is_range_market(
        ltf_df, ltf_idx, max_pct=cfg.range_max_pct
    ):
        return False
    if cfg.allowed_sessions is not None:
        hour = datetime.fromtimestamp(bar_ms / 1000, tz=timezone.utc).hour
        if _session_utc(hour) not in cfg.allowed_sessions:
            return False
    return True


def _map_setup_to_ltf(
    setup: TSpotSetup,
    ltf_bars: Sequence[BarOhlc],
    ltf_df: pd.DataFrame,
    htf_interval: str,
    htf_open_ms: int,
) -> TSpotSetup:
    """Replace HTF c2_bar_idx with matching LTF bar index."""
    period_ms = _htf_period_ms(htf_interval)
    c2_htf_key = htf_open_ms - period_ms
    keys = _assign_htf_keys(ltf_df, htf_interval)
    mask = keys == c2_htf_key
    ltf_indices = [i for i, ok in enumerate(mask.tolist()) if ok]
    if not ltf_indices:
        return setup
    c2_ltf_idx = _find_extreme_bar_idx(
        ltf_bars,
        ltf_indices,
        extreme=setup.c2_level,
        use_high=setup.side < 0,
    )
    return TSpotSetup(
        side=setup.side,
        c2_level=setup.c2_level,
        c2_bar_idx=c2_ltf_idx,
        sweep_mid=setup.sweep_mid,
        close_level=setup.close_level,
        htf_open_ms=setup.htf_open_ms,
        pattern=setup.pattern,
    )


def simulate_bars(
    df: pd.DataFrame,
    *,
    symbol: str,
    params: BacktestParams,
) -> SymbolBacktestResult:
    cfg = params.resolved_cfg()
    htf = cfg.resolved_htf()
    taker_bps = fee_taker_bps_from_env()
    slippage = cfg.slippage_bps

    if df.empty:
        return SymbolBacktestResult(symbol=symbol, bars=0, htf_interval=htf)

    ltf_df = df.sort_values("open_time").reset_index(drop=True)
    ltf_bars = _df_to_bars(ltf_df)
    htf_df = resample_ohlcv(ltf_df, htf)
    ltf_htf_key = _assign_htf_keys(ltf_df, htf)

    trades: List[FractalTrade] = []
    equity = float(cfg.equity_usdt)
    peak = equity
    max_dd = 0.0

    active: ActiveSetup | None = None
    pos_side = 0
    entry_ms = 0
    entry_px = 0.0
    entry_qty = 0.0
    stop_px = 0.0
    tp_px = 0.0
    entry_mode = ""
    entry_pattern = ""
    bars_since_exit = 10_000

    prev_htf_key: int | None = None
    ltf_period_ms = _interval_minutes(cfg.ltf_interval) * 60_000
    htf_period_ms = _htf_period_ms(htf)
    bars_per_htf = max(1, htf_period_ms // ltf_period_ms)

    for i, row in enumerate(ltf_df.itertuples(index=False)):
        bar_ms = int(row.open_time)
        o, h, l, c = float(row.open), float(row.high), float(row.low), float(row.close)
        htf_key = int(ltf_htf_key.iloc[i])

        # HTF roll: first LTF bar of a new HTF period
        if prev_htf_key is not None and htf_key != prev_htf_key:
            ctx = _build_htf_context(ltf_df, i, htf_df, htf, ltf_htf_key)
            if ctx is not None:
                setup = detect_tspot(
                    ctx,
                    bias=cfg.bias,
                    use_body=cfg.use_body_for_confirmation,
                )
                if setup is not None and pos_side == 0 and bars_since_exit >= cfg.cooldown_bars:
                    if cfg.allowed_patterns and setup.pattern not in cfg.allowed_patterns:
                        setup = None
                if setup is not None and pos_side == 0 and bars_since_exit >= cfg.cooldown_bars:
                    mapped = _map_setup_to_ltf(setup, ltf_bars, ltf_df, htf, htf_key)
                    expire = i + (cfg.max_setup_bars if cfg.max_setup_bars > 0 else bars_per_htf)
                    active = ActiveSetup(
                        setup=mapped,
                        start_ltf_idx=i,
                        expire_ltf_idx=expire,
                    )
        prev_htf_key = htf_key

        # Expire stale setup
        if active is not None and i > active.expire_ltf_idx and pos_side == 0:
            active = None

        # Manage open position
        if pos_side != 0:
            hit = bar_hits_stop_tp(side=pos_side, high=h, low=l, stop=stop_px, tp=tp_px)
            if hit:
                raw_exit = stop_px if hit == "stop" else tp_px
                exit_px = _apply_slippage(raw_exit, side=pos_side, is_entry=False, bps=slippage)
                gross = (exit_px - entry_px) * entry_qty if pos_side > 0 else (entry_px - exit_px) * entry_qty
                fee = trade_fee_usdt((entry_px + exit_px) * entry_qty / 2.0, taker_bps=taker_bps)
                net = gross - fee
                equity += net
                peak = max(peak, equity)
                max_dd = max(max_dd, peak - equity)
                trades.append(
                    FractalTrade(
                        symbol=symbol,
                        side="LONG" if pos_side > 0 else "SHORT",
                        pattern=entry_pattern,
                        entry_mode=entry_mode,
                        entry_ms=entry_ms,
                        exit_ms=bar_ms,
                        entry_price=round(entry_px, 6),
                        exit_price=round(exit_px, 6),
                        stop_price=round(stop_px, 6),
                        tp_price=round(tp_px, 6),
                        exit_reason=hit,
                        qty=round(entry_qty, 6),
                        pnl_gross_usdt=round(gross, 4),
                        fee_usdt=round(fee, 4),
                        pnl_net_usdt=round(net, 4),
                    )
                )
                pos_side = 0
                bars_since_exit = 0
                active = None
            continue

        bars_since_exit += 1

        # Entry logic
        if active is None:
            continue

        entry_idx: int | None = None
        trigger_level: float | None = None

        if cfg.entry_mode == "c3_touch":
            entry_idx = entry_signal_c3(
                ltf_bars,
                active,
                i,
                use_body=cfg.use_body_for_confirmation,
            )
            entry_mode = "c3_touch"
        else:
            entry_idx = entry_signal_cisd(
                ltf_bars,
                active,
                i,
                use_body=cfg.use_body_for_confirmation,
                require_touch=cfg.require_fractal_touch,
            )
            entry_mode = "cisd_c2"
            if entry_idx is not None:
                trigger_level = active.cisd_trigger_level

        if entry_idx is None or entry_idx != i:
            continue

        if not _passes_entry_filters(cfg, ltf_df, i, bar_ms, active.setup):
            continue

        side = active.setup.side
        raw_entry = c
        entry_px = _apply_slippage(raw_entry, side=side, is_entry=True, bps=slippage)
        stop_px, tp_px = stop_tp_from_setup(
            active.setup,
            entry_px,
            rr_ratio=cfg.rr_ratio,
            trigger_level=trigger_level,
        )
        qty = position_qty(entry_px, stop_px, equity=equity, risk_pct=cfg.risk_pct)
        if qty <= 0:
            active = None
            continue

        pos_side = side
        entry_ms = bar_ms
        entry_qty = qty
        entry_pattern = active.setup.pattern

    wins = sum(1 for t in trades if t.pnl_net_usdt > 0)
    wr = (wins / len(trades) * 100.0) if trades else 0.0
    return SymbolBacktestResult(
        symbol=symbol,
        bars=len(ltf_df),
        htf_interval=htf,
        trades=trades,
        total_pnl_net=round(sum(t.pnl_net_usdt for t in trades), 4),
        win_rate=round(wr, 2),
        max_drawdown_usdt=round(max_dd, 4),
    )


def run_backtest(params: BacktestParams) -> List[SymbolBacktestResult]:
    cfg = params.resolved_cfg()
    results: List[SymbolBacktestResult] = []
    for sym in params.symbols:
        df = fetch_bars(
            sym,
            cfg.ltf_interval,
            days=params.days,
            exchange_id=params.exchange_id,
            refresh=params.refresh,
        )
        print(f"[sim] {sym} bars={len(df)} ltf={cfg.ltf_interval} htf={cfg.resolved_htf()}")
        results.append(simulate_bars(df, symbol=sym, params=params))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ICT Fractal T-Spot / CISD backtest")
    p.add_argument("--symbols", default="BTCUSDT", help="comma-separated symbols")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--interval", default="5m", help="LTF interval")
    p.add_argument("--htf", default="", help="HTF interval (auto if empty)")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--entry-mode", choices=["cisd_c2", "c3_touch"], default="cisd_c2")
    p.add_argument("--rr", type=float, default=2.0, help="risk-reward ratio")
    p.add_argument("--bias", choices=["none", "bullish", "bearish"], default="none")
    p.add_argument("--require-touch", action="store_true")
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--risk-pct", type=float, default=1.0)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    cfg = FractalIctConfig(
        ltf_interval=args.interval,
        htf_interval=args.htf,
        bias=args.bias,
        entry_mode=args.entry_mode,
        rr_ratio=args.rr,
        require_fractal_touch=args.require_touch,
        equity_usdt=args.equity,
        risk_pct=args.risk_pct,
    )
    params = BacktestParams(
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        days=args.days,
        exchange_id=args.exchange,
        interval=args.interval,
        refresh=args.refresh,
        cfg=cfg,
    )
    results = run_backtest(params)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0

    for r in results:
        print(f"\n=== {r.symbol} ({r.htf_interval} HTF) ===")
        print(f"bars={r.bars} trades={len(r.trades)} pnl={r.total_pnl_net} win%={r.win_rate} maxDD={r.max_drawdown_usdt}")
        for t in r.trades[:20]:
            print(
                f"  {t.side:5} {t.entry_mode:8} {t.pattern:8} "
                f"entry={t.entry_price} exit={t.exit_price} {t.exit_reason} net={t.pnl_net_usdt}"
            )
        if len(r.trades) > 20:
            print(f"  ... +{len(r.trades) - 20} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())

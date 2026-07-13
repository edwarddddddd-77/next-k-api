"""Anchor Drift 事件驱动回测（永续 K 线 + BQuant 阈值）。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from quant.anchor_drift.config import AnchorDriftConfig
from quant.anchor_drift.core import (
    adverse_drift_stop,
    calculate_drift,
    generate_signal,
)
from quant.anchor_drift.session import (
    in_drift_trading_window,
    in_preopen_flat_window,
    is_weekend_anchor_session,
)
from quant.common.session import session_day_str
from quant.anchor_drift.sizing import size_for_drift
from quant.common.fees import fee_taker_bps_from_env, trade_fee_usdt
from quant.common.kline_cache import load_klines, norm_symbol, save_klines
from quant.common.session_paper import in_regular_session
from quant.market import fetch_klines_forward, klines_to_df


@dataclass
class DriftTrade:
    symbol: str
    side: str
    entry_ms: int
    exit_ms: int
    anchor: float
    entry_price: float
    exit_price: float
    drift_at_entry_pct: float
    exit_reason: str
    period: str
    qty: float
    pnl_gross_usdt: float
    fee_usdt: float
    pnl_net_usdt: float


@dataclass
class SymbolBacktestResult:
    symbol: str
    bars: int
    trades: List[DriftTrade] = field(default_factory=list)
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
    days: int = 90
    exchange_id: str = "binance"
    equity_usdt: float = 14.0
    compound: bool = True
    slippage_bps: float = 10.0
    taker_fee_bps: Optional[float] = None
    interval: str = "5m"
    refresh: bool = False
    one_trade_per_anchor: bool = True
    weekend_only: bool = False
    sat_sun_entry_only: bool = False
    disable_adverse_stop: bool = False
    cfg: Optional[AnchorDriftConfig] = None

    def resolved_cfg(self) -> AnchorDriftConfig:
        return self.cfg or AnchorDriftConfig.from_env()


def _apply_slippage(price: float, *, side: int, is_entry: bool, slippage_bps: float) -> float:
    """side: 1 long, -1 short；slippage 对交易者不利。"""
    bps = max(0.0, float(slippage_bps)) / 10_000.0
    px = float(price)
    if is_entry:
        return px * (1.0 + bps) if side > 0 else px * (1.0 - bps)
    return px * (1.0 - bps) if side > 0 else px * (1.0 + bps)


def _period_label(entry_ms: int, sess_tz: str) -> str:
    ts = pd.Timestamp(int(entry_ms), unit="ms", tz=sess_tz)
    if int(ts.weekday()) >= 5:
        return "weekend"
    return "overnight"


def _close_position(
    *,
    symbol: str,
    side: int,
    qty: float,
    entry_ms: int,
    entry_px: float,
    anchor: float,
    drift_at_entry: float,
    exit_ms: int,
    exit_px_raw: float,
    reason: str,
    slippage_bps: float,
    taker_bps: float,
    sess_tz: str,
) -> DriftTrade:
    exit_px = _apply_slippage(exit_px_raw, side=side, is_entry=False, slippage_bps=slippage_bps)
    qty = abs(float(qty))
    if side > 0:
        gross = (exit_px - entry_px) * qty
    else:
        gross = (entry_px - exit_px) * qty
    notional = (entry_px + exit_px) * qty / 2.0
    fee = trade_fee_usdt(notional, taker_bps=taker_bps)
    net = gross - fee
    return DriftTrade(
        symbol=symbol,
        side="LONG" if side > 0 else "SHORT",
        entry_ms=int(entry_ms),
        exit_ms=int(exit_ms),
        anchor=float(anchor),
        entry_price=float(entry_px),
        exit_price=float(exit_px),
        drift_at_entry_pct=float(drift_at_entry) * 100.0,
        exit_reason=reason,
        period=_period_label(entry_ms, sess_tz),
        qty=qty,
        pnl_gross_usdt=round(gross, 4),
        fee_usdt=round(fee, 4),
        pnl_net_usdt=round(net, 4),
    )


def simulate_bars(
    df: pd.DataFrame,
    *,
    symbol: str,
    params: BacktestParams,
) -> SymbolBacktestResult:
    cfg = params.resolved_cfg()
    sess = cfg.session_cfg()
    taker_bps = float(params.taker_fee_bps if params.taker_fee_bps is not None else fee_taker_bps_from_env())
    slippage = float(params.slippage_bps)
    tick_ms = max(5_000, int(float(cfg.tick_interval_sec) * 1000))

    if df.empty:
        return SymbolBacktestResult(symbol=symbol, bars=0)

    work = df.sort_values("open_time").reset_index(drop=True)
    trades: List[DriftTrade] = []
    equity = float(params.equity_usdt)
    peak = equity
    max_dd = 0.0

    anchor = 0.0
    last_rth_close = 0.0
    was_in_rth = False
    pos_side = 0
    entry_ms = 0
    entry_px = 0.0
    entry_qty = 0.0
    drift_at_entry = 0.0
    last_eval_ms = 0
    traded_this_anchor = False
    anchor_session = ""
    anchor_is_weekend = False

    for row in work.itertuples(index=False):
        bar_ms = int(row.open_time)
        px = float(row.close)
        if px <= 0:
            continue

        in_rth = bool(in_regular_session(sess, now_ms=bar_ms))

        if in_rth:
            last_rth_close = px
            was_in_rth = True
            continue

        if was_in_rth and last_rth_close > 0:
            if anchor != last_rth_close:
                traded_this_anchor = False
            anchor = last_rth_close
            anchor_session = session_day_str(
                bar_ms,
                tz=sess.session_tz,
                session_open_time=sess.session_open_time,
            )
            anchor_is_weekend = is_weekend_anchor_session(anchor_session)
        was_in_rth = False

        if anchor <= 0:
            continue

        drift = calculate_drift(anchor_price=anchor, current_price=px)
        if drift is None:
            continue

        if in_preopen_flat_window(bar_ms, sess, flat_minutes=int(cfg.preopen_flat_minutes)):
            if pos_side != 0:
                tr = _close_position(
                    symbol=symbol,
                    side=pos_side,
                    qty=entry_qty,
                    entry_ms=entry_ms,
                    entry_px=entry_px,
                    anchor=anchor,
                    drift_at_entry=drift_at_entry,
                    exit_ms=bar_ms,
                    exit_px_raw=px,
                    reason="preopen_flat",
                    slippage_bps=slippage,
                    taker_bps=taker_bps,
                    sess_tz=sess.session_tz,
                )
                trades.append(tr)
                if params.compound:
                    equity += tr.pnl_net_usdt
                peak = max(peak, equity)
                max_dd = max(max_dd, peak - equity)
                pos_side = 0
                traded_this_anchor = True
            continue

        if not in_drift_trading_window(bar_ms, sess, flat_minutes=int(cfg.preopen_flat_minutes)):
            continue

        if params.weekend_only and not anchor_is_weekend:
            continue

        sig = generate_signal(
            drift,
            signal_threshold=float(cfg.signal_threshold),
            converge_threshold=float(cfg.converge_threshold),
        )

        if pos_side != 0:
            hit_adverse = (
                False
                if params.disable_adverse_stop
                else adverse_drift_stop(
                    drift,
                    side=pos_side,
                    signal_threshold=float(cfg.signal_threshold),
                    max_adverse_extension=float(cfg.max_adverse_extension),
                )
            )
            if sig.signal == "CONVERGED" or hit_adverse:
                reason = "converged" if sig.signal == "CONVERGED" else "adverse_drift"
                tr = _close_position(
                    symbol=symbol,
                    side=pos_side,
                    qty=entry_qty,
                    entry_ms=entry_ms,
                    entry_px=entry_px,
                    anchor=anchor,
                    drift_at_entry=drift_at_entry,
                    exit_ms=bar_ms,
                    exit_px_raw=px,
                    reason=reason,
                    slippage_bps=slippage,
                    taker_bps=taker_bps,
                    sess_tz=sess.session_tz,
                )
                trades.append(tr)
                if params.compound:
                    equity += tr.pnl_net_usdt
                peak = max(peak, equity)
                max_dd = max(max_dd, peak - equity)
                pos_side = 0
                traded_this_anchor = True
            continue

        if sig.signal not in ("LONG", "SHORT"):
            continue
        if params.one_trade_per_anchor and traded_this_anchor:
            continue
        if bar_ms - last_eval_ms < tick_ms:
            continue
        if params.sat_sun_entry_only:
            bar_dow = int(
                pd.Timestamp(int(bar_ms), unit="ms", tz=sess.session_tz).weekday()
            )
            if bar_dow < 5:
                continue
        last_eval_ms = bar_ms

        side = 1 if sig.signal == "LONG" else -1
        eq = equity if params.compound else float(params.equity_usdt)
        qty = size_for_drift(cfg, px, anchor_price=anchor, equity_usdt=eq)
        if qty <= 0:
            continue
        entry_px = _apply_slippage(px, side=side, is_entry=True, slippage_bps=slippage)
        pos_side = side
        entry_ms = bar_ms
        entry_qty = qty
        drift_at_entry = drift

    wins = sum(1 for t in trades if t.pnl_net_usdt > 0)
    total = sum(t.pnl_net_usdt for t in trades)
    wr = (wins / len(trades)) if trades else 0.0
    return SymbolBacktestResult(
        symbol=symbol,
        bars=len(work),
        trades=trades,
        total_pnl_net=round(total, 4),
        win_rate=round(wr, 4),
        max_drawdown_usdt=round(max_dd, 4),
    )


def fetch_bars(
    symbol: str,
    *,
    days: int,
    interval: str,
    exchange_id: str,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    sym = norm_symbol(symbol)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - max(1, int(days)) * 86_400_000
    cached = pd.DataFrame()
    if use_cache and not refresh:
        cached = load_klines(sym, interval, start_ms=start_ms, end_ms=end_ms)
        if not cached.empty:
            span_days = (cached["open_time"].max() - cached["open_time"].min()) / 86_400_000
            if span_days >= max(1, int(days)) * 0.85:
                print(f"[backtest] {sym} {interval} cache hit ({len(cached)} bars)", file=sys.stderr)
                return cached

    print(
        f"[backtest] fetching {sym} {interval} {days}d from {exchange_id}...",
        file=sys.stderr,
    )
    rows = fetch_klines_forward(sym, interval, start_ms, end_ms, exchange_id=exchange_id)
    df = klines_to_df(rows)
    if df.empty:
        return cached if not cached.empty else df
    if use_cache:
        merged = pd.concat([cached, df], ignore_index=True) if not cached.empty else df
        merged = merged.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time")
        save_klines(sym, interval, merged)
        print(f"[backtest] cached {len(merged)} bars for {sym}", file=sys.stderr)
        return merged.reset_index(drop=True)
    return df


def run_backtest(params: BacktestParams) -> Dict[str, Any]:
    cfg = params.resolved_cfg()
    symbols = params.symbols or cfg.symbol_list()
    if not symbols:
        raise ValueError("no symbols for backtest")

    results: List[SymbolBacktestResult] = []
    for raw in symbols:
        sym = norm_symbol(raw)
        df = fetch_bars(
            sym,
            days=int(params.days),
            interval=str(params.interval),
            exchange_id=str(params.exchange_id),
            refresh=bool(getattr(params, "refresh", False)),
        )
        results.append(simulate_bars(df, symbol=sym, params=params))

    all_trades = [t for r in results for t in r.trades]
    total_pnl = sum(r.total_pnl_net for r in results)
    wins = sum(1 for t in all_trades if t.pnl_net_usdt > 0)
    weekend = [t for t in all_trades if t.period == "weekend"]
    overnight = [t for t in all_trades if t.period == "overnight"]

    def _seg_stats(trades: List[DriftTrade]) -> Dict[str, Any]:
        if not trades:
            return {"trades": 0, "pnl_net": 0.0, "win_rate": 0.0}
        w = sum(1 for t in trades if t.pnl_net_usdt > 0)
        return {
            "trades": len(trades),
            "pnl_net": round(sum(t.pnl_net_usdt for t in trades), 4),
            "win_rate": round(w / len(trades), 4),
        }

    return {
        "ok": True,
        "params": {
            "symbols": [norm_symbol(s) for s in symbols],
            "days": int(params.days),
            "exchange_id": params.exchange_id,
            "interval": params.interval,
            "equity_usdt": float(params.equity_usdt),
            "compound": bool(params.compound),
            "slippage_bps": float(params.slippage_bps),
            "signal_threshold": float(cfg.signal_threshold),
            "converge_threshold": float(cfg.converge_threshold),
        },
        "summary": {
            "total_pnl_net": round(total_pnl, 4),
            "total_trades": len(all_trades),
            "win_rate": round(wins / len(all_trades), 4) if all_trades else 0.0,
            "weekend": _seg_stats(weekend),
            "overnight": _seg_stats(overnight),
        },
        "symbols": [r.to_dict() for r in results],
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anchor Drift backtest on equity perp 1m klines")
    p.add_argument("--symbols", default="", help="Comma-separated, default from config pool")
    p.add_argument("--days", type=int, default=90, help="Lookback days (max ~since Feb 2026 for equity perps)")
    p.add_argument("--exchange", default="binance", choices=("binance", "bitget"))
    p.add_argument("--interval", default="5m", help="1m 更细但更慢；试跑建议 5m")
    p.add_argument("--equity", type=float, default=14.0, help="USDT per symbol")
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--no-compound", action="store_true")
    p.add_argument("--refresh", action="store_true", help="忽略本地 K 线缓存")
    p.add_argument(
        "--multi-entry",
        action="store_true",
        help="同一 anchor 周期允许多次进出（默认每周期只做一次）",
    )
    p.add_argument(
        "--weekend-only",
        action="store_true",
        help="仅 Fri 16:00 → Mon 9:25 周末 anchor 周期交易",
    )
    p.add_argument("--json", action="store_true", help="Print full JSON")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = AnchorDriftConfig.from_env()
    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = cfg.symbol_list()
    params = BacktestParams(
        symbols=symbols,
        days=int(args.days),
        exchange_id=str(args.exchange),
        equity_usdt=float(args.equity),
        compound=not bool(args.no_compound),
        slippage_bps=float(args.slippage_bps),
        interval=str(args.interval),
        refresh=bool(args.refresh),
        one_trade_per_anchor=not bool(args.multi_entry),
        weekend_only=bool(args.weekend_only),
        cfg=cfg,
    )
    out = run_backtest(params)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    sm = out["summary"]
    print(f"Anchor Drift backtest — {params.days}d — {params.exchange_id}")
    print(
        f"  trades={sm['total_trades']}  pnl_net={sm['total_pnl_net']:+.4f} USDT  "
        f"win_rate={sm['win_rate'] * 100:.1f}%"
    )
    print(
        f"  weekend: {sm['weekend']['trades']} trades  pnl={sm['weekend']['pnl_net']:+.4f}  "
        f"win={sm['weekend']['win_rate'] * 100:.1f}%"
    )
    print(
        f"  overnight: {sm['overnight']['trades']} trades  pnl={sm['overnight']['pnl_net']:+.4f}  "
        f"win={sm['overnight']['win_rate'] * 100:.1f}%"
    )
    for row in out["symbols"]:
        print(
            f"  {row['symbol']}: bars={row['bars']} trades={len(row['trades'])} "
            f"pnl={row['total_pnl_net']:+.4f} max_dd={row['max_drawdown_usdt']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

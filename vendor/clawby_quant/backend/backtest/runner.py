"""Replay main loop: one strategy x one param set over the window.

Per-minute sequence:
  1. clock=t, prices = this minute's opens
  2. settle funding crossed in (t-60, t]
  3. scan if the strategy's interval is due (minute-level strategies)
  4. second-level sub-loop for symbols whose S11 window covers this minute
  5. bar-based exit management for the [t, t+60) bar (skipping window symbols)
"""
import asyncio
import bisect
import copy
import math
import statistics

import yaml

from .. import config
from . import data
from .broker import SimBroker
from .sim import Sim

_INTERVALS = {"1s": 1, "5s": 5, "15s": 15, "30s": 30, "1m": 60, "5m": 300,
              "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400}

# strategies whose replay needs a relaxed depth source (spec 放宽项)
_DEPTH_MODE = {"S02_LIQ_REBOUND": "pass", "S06_CVD_FADE": "taker_proxy"}

# S04 needs a 7d z-window; Binance LSR history can't warm up before T0
_DELAYED_START = {"S04_SMART_DUMB_DIV": data.T0 + 7 * 86400}


def interval_sec(v):
    if isinstance(v, (int, float)):
        return max(1, int(v))
    return _INTERVALS.get(str(v), 900)


def base_cfg(sid):
    cfg = yaml.safe_load(config.STRATEGIES_PATH.read_text(encoding="utf-8"))
    scfg = copy.deepcopy(cfg["strategies"][sid])
    universe = list(cfg["global"]["universe"])
    return scfg, universe


class Result:
    def __init__(self, sid, params, trades, equity, t0, t1, broker):
        self.sid = sid
        self.params = params
        self.trades = trades
        self.equity = equity
        self.t0, self.t1 = t0, t1
        self.notional = broker.notional
        self.fees_total = broker.fees_total
        self.funding_total = broker.funding_total

    def metrics(self):
        n = len(self.trades)
        pnl = sum(t["pnl"] for t in self.trades)
        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        gp = sum(t["pnl"] for t in wins)
        gl = -sum(t["pnl"] for t in losses)
        # max drawdown over the equity curve (includes unrealized)
        peak, mdd = 0.0, 0.0
        for _ts, eq in self.equity:
            peak = max(peak, eq)
            mdd = max(mdd, peak - eq)
        # hourly-resampled sharpe (annualized, 24/7 market)
        sharpe = 0.0
        if self.equity:
            hourly, last_h, last_eq = [], None, 0.0
            for ts, eq in self.equity:
                h = ts // 3600
                if last_h is None:
                    last_h, last_eq = h, eq
                elif h != last_h:
                    hourly.append(eq - last_eq)
                    last_h, last_eq = h, eq
            if len(hourly) > 24 and statistics.pstdev(hourly) > 0:
                sharpe = (statistics.fmean(hourly) / statistics.pstdev(hourly)
                          * math.sqrt(24 * 365))
        return {
            "trades": n,
            "net_usd": round(pnl, 2),
            "net_return_pct": round(pnl / self.notional * 100, 2),
            "max_dd_usd": round(mdd, 2),
            "max_dd_pct": round(mdd / self.notional * 100, 2),
            "win_rate": round(len(wins) / n * 100, 1) if n else 0.0,
            "profit_factor": round(gp / gl, 2) if gl > 0 else (999.0 if gp > 0 else 0.0),
            "sharpe": round(sharpe, 2),
            "avg_hold_h": round(statistics.fmean(t["hold_sec"] for t in self.trades)
                                / 3600, 2) if n else 0.0,
            "fees_usd": round(self.fees_total, 2),
            "funding_usd": round(self.funding_total, 2),
        }


async def run_backtest(sid, params_override, ds, t0=None, t1=None,
                       notional=1000.0, record_equity=True):
    import backend.strategies as S

    scfg, universe = base_cfg(sid)
    scfg["params"] = {**scfg.get("params", {}), **(params_override or {})}
    universe = [s for s in universe if s in ds.k1m]

    t0 = max(t0 if t0 is not None else data.T0, data.T0)
    t0 = max(t0, _DELAYED_START.get(sid, t0))
    t1 = min(t1 if t1 is not None else data.T1, ds.t_end)

    sim = Sim(ds, t0)
    base = scfg.get("base") or sid
    sim.binance.depth_mode = _DEPTH_MODE.get(base, "neutral")
    broker = SimBroker(sim, notional=notional, record_equity=record_equity)
    sim.install()
    try:
        cls = S.REGISTRY[base]
        strat = cls({**scfg, "_iid": sid})
        strategies = {sid: strat}
        # backtests replay the strategy's DESIGN cadence — the runtime yaml
        # cadence is a live-ops knob and must not change backtest semantics
        step = interval_sec(getattr(cls, "DEFAULT_INTERVAL",
                                    scfg.get("scan_interval", "15m")))
        second_level = step < 60
        next_scan = t0

        # pre-index S11 windows per symbol for the run range
        win_by_min = {}
        if second_level:
            for sym in (strat.symbols or universe):
                for lo, hi in ds.windows.get(sym, []):
                    if hi < t0 or lo > t1:
                        continue
                    for m in range(int(lo) // 60 * 60, int(hi) + 60, 60):
                        win_by_min.setdefault(m, []).append((sym, lo, hi))

        t = t0
        while t < t1:
            sim.clock.set(t)
            for sym in universe:
                sim.ws.price[sym] = sim._minute_open(sym, t)
            broker.settle_funding(t - 60, t)

            ctx = S.Ctx(universe)
            if not second_level and t >= next_scan:
                sigs = await strat.evaluate(ctx)
                for sig in sigs:
                    await broker.on_signal(sig)
                next_scan = t + step

            window_syms = set()
            if second_level and t in win_by_min:
                for sym, lo, hi in win_by_min[t]:
                    window_syms.add(sym)
                    ticks = ds.ticks.get((sym, int(lo)))
                    if not ticks:
                        continue
                    sim.ws.enter_window(sym, lo, ticks)
                    tick_ts = [a for a, _ in ticks]
                    sec_end = min(t + 60, hi, t1)
                    last_price = sim.ws.price.get(sym) or 0.0
                    for sec in range(int(max(t, lo)), int(sec_end)):
                        sim.clock.set(sec)
                        i = bisect.bisect_right(tick_ts, sec)
                        if i:
                            last_price = ticks[i - 1][1]
                        if not last_price:
                            continue
                        sim.ws.price[sym] = last_price
                        await broker.manage_second(sym, last_price, ctx,
                                                   strategies, sec)
                        sigs = await strat.evaluate(ctx)
                        for sig in sigs:
                            await broker.on_signal(sig)
                    if sec_end >= hi:
                        sim.ws.exit_window(sym)
                    sim.clock.set(t)

            await broker.manage_tick(t, ctx, strategies,
                                     skip_symbols=window_syms)
            t += 60

        # force-close whatever is still open at the end of the window
        sim.clock.set(t1)
        for pos in list(sim.db.open_positions()):
            price = sim.ws.live_price(pos["symbol"]) or pos["entry_price"]
            broker._close(pos, price, "回测窗口结束强制平仓", t1)
    finally:
        sim.uninstall()

    return Result(sid, dict(scfg["params"]), broker.trades, broker.equity,
                  t0, t1, broker)


def run(sid, params_override=None, ds=None, symbols=None, **kw):
    """Sync convenience wrapper."""
    from . import data as d
    if ds is None:
        ds = DatasetCache.get(symbols)
    return asyncio.run(run_backtest(sid, params_override, ds, **kw))


class DatasetCache:
    _ds = None
    _syms = None

    @classmethod
    def get(cls, symbols=None):
        from .sim import Dataset
        symbols = symbols or data.valid_symbols()
        key = tuple(sorted(symbols))
        if cls._ds is None or cls._syms != key:
            cls._ds = Dataset(symbols)
            cls._syms = key
        return cls._ds

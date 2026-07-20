"""Replay layer: drive the LIVE strategy classes over historical data.

Sim.install() swaps the module attributes that backend.strategies (and
backend.ws) resolve at call time — time, db, factors, binance, ws functions —
for replay implementations bound to a virtual clock. Strategy code and the
live Ctx class run unmodified; Sim.uninstall() restores everything.

Anti-lookahead rules:
- factor series contain only buckets whose END <= virtual now (t)
- klines: complete bars strictly before t, plus a partial bar aggregated from
  completed 1m bars (mirrors the live "current unfinished candle")
- price at t = open of the 1m bar containing t (or the aggTrades tick price
  inside S11 second-level windows)
"""
import bisect
import calendar
import time as _realtime

from .. import factors as live_factors
from . import data

_INTERVAL_SEC = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}

# factors with no historical source -> None (live strategy code degrades
# gracefully on each of these; see spec "放宽项")
_NO_HISTORY = {"liq_orders", "ob_wall", "exch_chain_tx", "whale_tx",
               "mkt_price_chg", "netflow_xsec", "unlock", "alt_season",
               "fear_greed", "funding_xsec", "liq_map", "mark_all"}


def _ts_sec(row):
    for k in ("time", "timestamp", "fundingTime", "publish_timestamp", "ts"):
        v = row.get(k)
        if v is not None:
            v = float(v)
            return v / 1000 if v > 4e10 else v
    return None


class SimClock:
    """Duck-typed stand-in for the `time` module inside strategy code."""
    timezone = 0
    altzone = 0

    def __init__(self, t):
        self.t = float(t)

    def set(self, t):
        self.t = float(t)

    def time(self):
        return self.t

    def monotonic(self):
        return self.t

    def gmtime(self, secs=None):
        return _realtime.gmtime(self.t if secs is None else secs)

    def mktime(self, tup):
        return float(calendar.timegm(tup))

    def sleep(self, _secs):
        pass


class Dataset:
    """All cached history for a symbol set, parsed + indexed once."""

    def __init__(self, symbols):
        self.symbols = list(symbols)
        self.k1m, self.k1m_ts = {}, {}
        self.agg = {}            # sym -> {300: (ts_list, bars), 3600: (...)}
        self.series = {}         # (kind, sym) -> (ts_list, rows_parsed)
        self.funding = {}        # sym -> (ts_list, rates)
        self.premium = {}        # sym -> (ts_list, closes)
        self.windows = {}        # sym -> [[lo, hi], ...]
        self.ticks = {}          # (sym, lo) -> [(sec, price), ...]
        self.econ_events = []
        self.etf = []            # [(avail_ts, flow_usd)]  avail = data day + 1d
        self.t_end = data.T1

        for sym in self.symbols:
            ks = data.load("klines_1m", sym)
            if not ks:
                raise RuntimeError(f"klines_1m missing for {sym}")
            self.k1m[sym] = ks
            self.k1m_ts[sym] = [k["ts"] for k in ks]
            self.t_end = min(self.t_end, ks[-1]["ts"] + 60)
            self.agg[sym] = {s: self._aggregate(ks, s) for s in (300, 3600)}

            fr = data.load("funding_rate", sym) or []
            pairs = sorted((_ts_sec(r), float(r["fundingRate"])) for r in fr)
            self.funding[sym] = ([p[0] for p in pairs], [p[1] for p in pairs])

            pm = data.load("premium_1h", sym) or []
            self.premium[sym] = ([r["ts"] for r in pm], [r["close"] for r in pm])

            self._load_series("lsr_global_1h", sym, lambda r: float(r["longShortRatio"]))
            self._load_series("lsr_top_1h", sym, lambda r: float(r["longShortRatio"]))
            self._load_series("taker_5m", sym, lambda r: float(r["buySellRatio"]))
            self._load_series("oi_5m", sym, lambda r: float(r["sumOpenInterestValue"]))
            self._load_series("clawby_funding_agg", sym, lambda r: float(r.get("close") or 0))
            self._load_series("clawby_oi_agg", sym, lambda r: float(r.get("close") or 0))
            self._load_series("clawby_cvd", sym,
                              lambda r: float(r.get("cum_vol_delta") or 0))
            self._load_series("clawby_liq_agg", sym,
                              lambda r: (float(r.get("aggregated_long_liquidation_usd") or 0),
                                         float(r.get("aggregated_short_liquidation_usd") or 0)))
            aw = data.load("aggtrades_windows", sym)
            if aw and isinstance(aw, dict):
                self.windows[sym] = aw.get("windows") or []
                for lo_str, tk in (aw.get("ticks") or {}).items():
                    self.ticks[(sym, int(lo_str))] = [(int(a), float(b)) for a, b in tk]
            else:
                self.windows[sym] = []

        self._load_series("clawby_coinbase_prem", "",
                          lambda r: float(r.get("premium_rate") or 0))
        for r in (data.load("clawby_econ_cal") or []):
            ts = _ts_sec(r)
            if ts:
                self.econ_events.append(
                    {"ts": int(ts), "title": str(r.get("calendar_name") or "")[:80],
                     "importance": str(r.get("importance_level") or "")})
        self.econ_events.sort(key=lambda e: e["ts"])
        etf_rows = []
        for r in (data.load("clawby_etf_flow") or []):
            ts = _ts_sec(r)
            if ts:
                try:
                    etf_rows.append((ts + 86400, float(r.get("flow_usd") or 0)))
                except (TypeError, ValueError):
                    continue
        self.etf = sorted(etf_rows)

    @classmethod
    def synthetic(cls, k1m_by_sym, series=None, funding=None, windows=None,
                  ticks=None, econ=None, etf=None):
        """Build a Dataset from in-memory pieces (tests / debugging)."""
        ds = cls.__new__(cls)
        ds.symbols = list(k1m_by_sym)
        ds.k1m = dict(k1m_by_sym)
        ds.k1m_ts = {s: [k["ts"] for k in v] for s, v in ds.k1m.items()}
        ds.agg = {s: {sec: cls._aggregate(v, sec) for sec in (300, 3600)}
                  for s, v in ds.k1m.items()}
        ds.series = {}
        for key, pairs in (series or {}).items():
            pairs = sorted(pairs)
            ds.series[key] = ([p[0] for p in pairs], [p[1] for p in pairs])
        ds.funding = {}
        for s, pairs in (funding or {}).items():
            pairs = sorted(pairs)
            ds.funding[s] = ([p[0] for p in pairs], [p[1] for p in pairs])
        for s in ds.symbols:
            ds.funding.setdefault(s, ([], []))
        ds.premium = {s: ([], []) for s in ds.symbols}
        ds.windows = {s: (windows or {}).get(s, []) for s in ds.symbols}
        ds.ticks = dict(ticks or {})
        ds.econ_events = list(econ or [])
        ds.etf = list(etf or [])
        ds.t_end = max(v[-1]["ts"] + 60 for v in ds.k1m.values())
        return ds

    def _load_series(self, kind, sym, parse):
        rows = data.load(kind, sym) or []
        out = []
        for r in rows:
            ts = _ts_sec(r)
            if ts is None:
                continue
            try:
                out.append((ts, parse(r)))
            except (TypeError, ValueError, KeyError):
                continue
        out.sort(key=lambda p: p[0])
        self.series[(kind, sym)] = ([p[0] for p in out], [p[1] for p in out])

    @staticmethod
    def _aggregate(k1m, bucket_sec):
        ts_list, bars = [], []
        cur_start, cur = None, None
        for k in k1m:
            b = k["ts"] // bucket_sec * bucket_sec
            if b != cur_start:
                if cur is not None:
                    ts_list.append(cur_start)
                    bars.append(cur)
                cur_start = b
                cur = dict(ts=b, open=k["open"], high=k["high"],
                           low=k["low"], close=k["close"], volume=k["volume"])
            else:
                cur["high"] = max(cur["high"], k["high"])
                cur["low"] = min(cur["low"], k["low"])
                cur["close"] = k["close"]
                cur["volume"] += k["volume"]
        if cur is not None:
            ts_list.append(cur_start)
            bars.append(cur)
        return ts_list, bars


class SimWs:
    """Second-level price feed inside S11 candidate windows; per-minute
    open price everywhere else (live_price only)."""

    def __init__(self, clock):
        self.clock = clock
        self.price = {}
        self._win = {}           # sym -> (win_lo, ticks, tick_ts_list)

    def enter_window(self, sym, win_lo, ticks):
        self._win[sym] = (win_lo, ticks, [a for a, _ in ticks])

    def exit_window(self, sym):
        self._win.pop(sym, None)

    def live_price(self, symbol, max_age_sec=10):
        return self.price.get(symbol, 0.0)

    def price_ago(self, symbol, secs):
        w = self._win.get(symbol)
        if not w:
            return None
        _lo, ticks, ts_list = w
        now = self.clock.t
        n = bisect.bisect_right(ts_list, now)     # ticks visible so far
        if not n:
            return None
        cutoff = now - secs
        i = bisect.bisect_right(ts_list, cutoff, 0, n)
        if i:
            return ticks[i - 1][1]
        first_ts, first_p = ticks[0]
        if now - first_ts >= secs * 0.5:           # mirrors live ws.price_ago
            return first_p
        return None

    def change_pct(self, symbol, secs):
        old = self.price_ago(symbol, secs)
        cur = self.live_price(symbol)
        if old and cur:
            return (cur - old) / old * 100
        return None

    def buffer_span(self, symbol):
        w = self._win.get(symbol)
        if not w:
            return 0
        _lo, ticks, ts_list = w
        n = bisect.bisect_right(ts_list, self.clock.t)
        if not n:
            return 0
        return self.clock.t - ticks[0][0]


class SimDb:
    """In-memory stand-in for backend.db, factor reads routed to the store."""

    def __init__(self, clock, store):
        self.clock, self.store = clock, store
        self.meta, self.signals = {}, []
        self.positions = {}
        self._next = 1

    # factors
    def get_factor(self, factor, symbol="", max_age_sec=None):
        return self.store.factor_at(factor, symbol, self.clock.t)

    def set_factor(self, *a, **k):
        pass

    def get_factor_ts(self, factor, symbol=""):
        return int(self.clock.t)

    # meta
    def get_meta(self, key, default=""):
        return self.meta.get(key, default)

    def set_meta(self, key, value):
        self.meta[key] = str(value)

    # positions
    def open_position(self, **kw):
        pid = self._next
        self._next += 1
        pos = dict(id=pid, status="open", opened_at=int(self.clock.t),
                   closed_at=None, close_price=None, pnl=None, close_reason=None,
                   extreme=kw.get("entry_price"), mfe_pct=0.0, mae_pct=0.0,
                   stop_price=None, take_profit=None, trail_atr=None,
                   time_exit_at=None, entry_factors=None, reason="")
        pos.update(kw)
        self.positions[pid] = pos
        return pid

    def get_position(self, pid):
        return self.positions.get(pid)

    def open_positions(self, strategy=None, symbol=None, mode="auto"):
        # replay has a single mode; the arg exists for signature parity
        return [p for p in self.positions.values() if p["status"] == "open"
                and (not strategy or p["strategy"] == strategy)
                and (not symbol or p["symbol"] == symbol)]

    def update_position(self, pid, **fields):
        self.positions[pid].update(fields)

    def close_position(self, pid, price, pnl, reason):
        self.positions[pid].update(status="closed", closed_at=int(self.clock.t),
                                   close_price=price, pnl=pnl, close_reason=reason)

    def closed_positions_today(self, day_start_ts):
        return [p for p in self.positions.values()
                if p["status"] == "closed" and (p["closed_at"] or 0) >= day_start_ts]

    # logging (kept lightweight)
    def log_signal(self, strategy, symbol, side, reason, acted, skip_reason=""):
        self.signals.append({"ts": int(self.clock.t), "strategy": strategy,
                             "symbol": symbol, "side": side, "acted": int(acted)})

    def signal_count_since(self, strategy, symbol, since_ts):
        return sum(1 for s in self.signals
                   if s["strategy"] == strategy and s["symbol"] == symbol
                   and s["ts"] >= since_ts and s["acted"])

    def log(self, level, msg):
        pass

    def log_trade(self, *a, **k):
        pass

    def record_equity(self, *a, **k):
        pass


class SimFactorStore:
    """factor_at(name, symbol, t) -> same dict shapes the live f_* produce,
    built from completed buckets only. Cached per (name, sym, bucket)."""

    def __init__(self, ds):
        self.ds = ds
        self._cache = {}
        self._oi_hourly = {}

    def _completed(self, kind, sym, t, bucket_sec):
        ts_list, vals = self.ds.series.get((kind, sym), ([], []))
        n = bisect.bisect_right(ts_list, t - bucket_sec)
        return vals[:n], n

    def _oi_hourly_of(self, sym):
        """Precomputed (hour_ts, last 5m value of that hour) — built once."""
        hit = self._oi_hourly.get(sym)
        if hit is None:
            ts_list, vals = self.ds.series.get(("oi_5m", sym), ([], []))
            hts, hvals = [], []
            for ts, v in zip(ts_list, vals):
                h = int(ts) // 3600 * 3600
                if hts and hts[-1] == h:
                    hvals[-1] = v
                else:
                    hts.append(h)
                    hvals.append(v)
            hit = (hts, hvals)
            self._oi_hourly[sym] = hit
        return hit

    def factor_at(self, name, symbol, t):
        if name in _NO_HISTORY:
            return None
        bucket = int(t) // 300
        key = (name, symbol, bucket)
        hit = self._cache.get(key)
        if hit is not None:
            return hit[0]
        val = self._compute(name, symbol, t)
        self._cache[key] = (val,)
        if len(self._cache) > 200_000:
            self._cache.clear()
        return val

    def _compute(self, name, symbol, t):
        ds = self.ds
        if name == "funding_agg":
            vals, n = self._completed("clawby_funding_agg", symbol, t, 3600)
            if not n:
                return None
            s = vals[-24:]
            return {"latest": s[-1], "series": s}
        if name == "lsr_global":
            vals, n = self._completed("lsr_global_1h", symbol, t, 3600)
            if not n:
                return None
            s = vals[-168:]
            return {"latest": s[-1], "series": s}
        if name == "lsr_top_pos":
            vals, n = self._completed("lsr_top_1h", symbol, t, 3600)
            if not n:
                return None
            s = vals[-168:]
            return {"latest": s[-1], "series": s}
        if name == "taker_flow":
            vals, n = self._completed("taker_5m", symbol, t, 300)
            if not n:
                return None
            s = vals[-48:]
            return {"latest": s[-1], "series": s}
        if name == "oi_binance":
            ts_list, vals = ds.series.get(("oi_5m", symbol), ([], []))
            n = bisect.bisect_right(ts_list, t - 300)
            if not n:
                return None
            hts, hvals = self._oi_hourly_of(symbol)
            cur_hour = int(t) // 3600 * 3600
            k = bisect.bisect_right(hts, cur_hour - 3600)  # hours before now
            s = hvals[max(0, k - 47):k]
            last_ts, last_val = ts_list[n - 1], vals[n - 1]
            if last_ts >= cur_hour or not s:
                s = s + [last_val]         # current hour's freshest sample
            chg = (s[-1] / s[-2] - 1) * 100 if len(s) >= 2 and s[-2] else 0
            return {"latest": s[-1], "chg_1h_pct": chg, "series": s}
        if name == "oi_agg":
            vals, n = self._completed("clawby_oi_agg", symbol, t, 3600)
            if not n:
                return None
            s = vals[-24:]
            slope = (s[-1] / s[-4] - 1) * 100 if len(s) >= 4 and s[-4] else 0
            return {"latest": s[-1], "slope_3h_pct": slope}
        if name == "liq_agg":
            vals, n = self._completed("clawby_liq_agg", symbol, t, 3600)
            if not n:
                return None
            pairs = vals[-168:]
            longs = [p[0] for p in pairs]
            shorts = [p[1] for p in pairs]
            avg_l = sum(longs[:-1]) / max(len(longs) - 1, 1)
            avg_s = sum(shorts[:-1]) / max(len(shorts) - 1, 1)
            return {"long_1h": longs[-1], "short_1h": shorts[-1],
                    "long_mult": longs[-1] / avg_l if avg_l else 0,
                    "short_mult": shorts[-1] / avg_s if avg_s else 0}
        if name == "cvd":
            vals, n = self._completed("clawby_cvd", symbol, t, 3600)
            if not n:
                return None
            s = vals[-48:]
            return {"latest": s[-1], "series": s}
        if name == "coinbase_prem":
            vals, n = self._completed("clawby_coinbase_prem", "", t, 3600)
            if not n:
                return None
            s = vals[-48:]
            return {"latest": s[-1], "series": s}
        if name == "etf_flow_btc":
            ts_list = [p[0] for p in self.ds.etf]
            i = bisect.bisect_right(ts_list, t)
            return {"last_day_flow_usd": self.ds.etf[i - 1][1]} if i else None
        if name == "econ_cal":
            return {"events": self.ds.econ_events}
        return None


class FactorsFacade:
    """Swapped in for the `factors` module name inside backend.strategies."""

    def __init__(self, sim):
        self._sim = sim
        self.atr_from_klines = live_factors.atr_from_klines

    async def get_klines(self, symbol, interval="1h", limit=100):
        return self._sim.klines_at(symbol, interval, limit)

    def mark_price_of(self, symbol):
        return self._sim.ws.live_price(symbol)

    def basis_pct_of(self, symbol):
        ts_list, closes = self._sim.ds.premium.get(symbol, ([], []))
        i = bisect.bisect_right(ts_list, self._sim.clock.t - 3600)
        return closes[i - 1] * 100 if i else 0.0

    def snapshot_for(self, symbol):
        return {}


class SimBinance:
    """Only Ctx.depth_ratio reaches binance in replay. Modes:
    neutral (1.0) / pass (S02: condition waived) / taker_proxy (S06)."""

    def __init__(self, sim):
        self._sim = sim
        self.depth_mode = "neutral"

    async def depth(self, symbol, limit=50):
        if self.depth_mode == "pass":
            return {"bids": [["1", "999"]], "asks": [["1", "1"]]}
        if self.depth_mode == "taker_proxy":
            tk = self._sim.store.factor_at("taker_flow", symbol, self._sim.clock.t)
            r = (tk or {}).get("latest") or 1.0
            return {"bids": [["1", str(r)]], "asks": [["1", "1"]]}
        return {"bids": [["1", "1"]], "asks": [["1", "1"]]}


class Sim:
    def __init__(self, ds, t_start=None):
        self.ds = ds
        self.clock = SimClock(t_start if t_start is not None else data.T0)
        self.ws = SimWs(self.clock)
        self.store = SimFactorStore(ds)
        self.db = SimDb(self.clock, self.store)
        self.facade = FactorsFacade(self)
        self.binance = SimBinance(self)
        self._kcache = {}
        self._saved = None

    # -- klines / price ------------------------------------------------------

    def klines_at(self, symbol, interval="1h", limit=100):
        t = self.clock.t
        sec = _INTERVAL_SEC[interval]
        key = (symbol, interval, limit, int(t) // 60)
        hit = self._kcache.get(key)
        if hit is not None:
            return hit
        if len(self._kcache) > 50_000:
            self._kcache.clear()
        if sec == 60:
            ts_list = self.ds.k1m_ts[symbol]
            hi = bisect.bisect_right(ts_list, t - 60)
            out = self.ds.k1m[symbol][max(0, hi - limit):hi]
        else:
            fts, fbars = self.ds.agg[symbol][sec]
            hi = bisect.bisect_right(fts, t - sec)
            out = list(fbars[max(0, hi - (limit - 1)):hi])
            partial = self._partial_bar(symbol, sec, t)
            if partial:
                out.append(partial)
            out = out[-limit:]
        self._kcache[key] = out
        return out

    def _partial_bar(self, symbol, bucket_sec, t):
        start = int(t) // bucket_sec * bucket_sec
        ts_list = self.ds.k1m_ts[symbol]
        lo = bisect.bisect_left(ts_list, start)
        hi = bisect.bisect_right(ts_list, t - 60)
        if hi <= lo:
            p = self.ws.live_price(symbol) or self._minute_open(symbol, t)
            if not p:
                return None
            return dict(ts=start, open=p, high=p, low=p, close=p, volume=0.0)
        ks = self.ds.k1m[symbol][lo:hi]
        return dict(ts=start, open=ks[0]["open"],
                    high=max(k["high"] for k in ks),
                    low=min(k["low"] for k in ks),
                    close=ks[-1]["close"],
                    volume=sum(k["volume"] for k in ks))

    def _minute_open(self, symbol, t):
        ts_list = self.ds.k1m_ts[symbol]
        i = bisect.bisect_right(ts_list, t) - 1
        if i < 0:
            return 0.0
        bar = self.ds.k1m[symbol][i]
        return bar["open"] if t < bar["ts"] + 60 else bar["close"]

    def minute_bar(self, symbol, t):
        """The 1m bar starting exactly at t (or None)."""
        ts_list = self.ds.k1m_ts[symbol]
        i = bisect.bisect_left(ts_list, int(t))
        if i < len(ts_list) and ts_list[i] == int(t):
            return self.ds.k1m[symbol][i]
        return None

    # -- funding -------------------------------------------------------------

    def funding_events(self, symbol, t_from, t_to):
        """[(ts, rate)] settlements in (t_from, t_to]."""
        ts_list, rates = self.ds.funding.get(symbol, ([], []))
        lo = bisect.bisect_right(ts_list, t_from)
        hi = bisect.bisect_right(ts_list, t_to)
        return [(ts_list[i], rates[i]) for i in range(lo, hi)]

    # -- module patching -----------------------------------------------------

    def install(self):
        import backend.strategies as S
        import backend.ws as W
        self._saved = {"S_time": S.time, "S_db": S.db, "S_factors": S.factors,
                       "S_binance": S.binance, "W_live": W.live_price,
                       "W_ago": W.price_ago, "W_chg": W.change_pct,
                       "W_span": W.buffer_span}
        S.time = self.clock
        S.db = self.db
        S.factors = self.facade
        S.binance = self.binance
        W.live_price = self.ws.live_price
        W.price_ago = self.ws.price_ago
        W.change_pct = self.ws.change_pct
        W.buffer_span = self.ws.buffer_span

    def uninstall(self):
        if not self._saved:
            return
        import backend.strategies as S
        import backend.ws as W
        S.time = self._saved["S_time"]
        S.db = self._saved["S_db"]
        S.factors = self._saved["S_factors"]
        S.binance = self._saved["S_binance"]
        W.live_price = self._saved["W_live"]
        W.price_ago = self._saved["W_ago"]
        W.change_pct = self._saved["W_chg"]
        W.buffer_span = self._saved["W_span"]
        self._saved = None

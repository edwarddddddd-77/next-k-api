"""Backtest historical data: download (Binance REST + Clawby relay) + local cache.

Window: 2026-06-18 00:00 UTC -> 2026-07-18 00:00 UTC (30 days).
Cache layout: backtest/data/<kind>/<symbol or _>.json
    {"meta": {"t0":..., "t1":..., "downloaded_at":...}, "rows": [...]}

Kinds:
  klines_1m         [{ts,open,high,low,close,volume}]        (Binance, permanent)
  funding_rate      [{fundingTime, fundingRate}]             (Binance, permanent)
  oi_5m             raw openInterestHist rows                (Binance, 30d only!)
  lsr_global_1h     raw globalLongShortAccountRatio rows     (Binance, 30d only!)
  lsr_top_1h        raw topLongShortPositionRatio rows       (Binance, 30d only!)
  taker_5m          raw takerlongshortRatio rows             (Binance, 30d only!)
  clawby_funding_agg / clawby_liq_agg / clawby_cvd / clawby_oi_agg
                    raw Clawby history rows (1h)             (per symbol)
  clawby_coinbase_prem / clawby_etf_flow / clawby_econ_cal   (global, symbol="_")
  aggtrades_windows {"windows": [[t_from,t_to],...], "ticks": {"<t_from>": [[sec,price],...]}}
"""
import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

from .. import clawby, config

log = logging.getLogger("bt.data")

T0 = 1781740800          # 2026-06-18 00:00 UTC
T1 = 1784332800          # 2026-07-18 00:00 UTC
IS_END = T0 + 21 * 86400  # 2026-07-09 00:00 UTC (train/validation split)
WARMUP = 7 * 86400        # Clawby factors need history before T0 (168h baselines,
                          # 7d z-windows); Binance futures/data can't backfill
                          # past 30d so those series start thin at T0.

DATA_DIR = config.ROOT / "backtest" / "data"
FUT = "https://fapi.binance.com"

# S11 candidate-window scan, sized to the grid floor drop_pct=1.5%/30-45s.
# A qualifying move that splits across two 1m bars leaves >=0.75% range on
# each, so the adjacent-range-sum test (>=1.45, with float margin) keeps the
# scan mathematically lossless while cutting window hours ~2x vs a flat 0.75%.
WIN_RANGE1 = 1.0     # single-bar intra-range
WIN_CHG2 = 1.2       # 2-minute cumulative move (directional slides)
WIN_JOINT = 1.45     # adjacent two-bar range sum (V-shaped wicks)
PAD_BEFORE = 120     # S11 needs >=20s buffer + 45s window before the trigger
PAD_AFTER = 420      # second-level exit precision for the bounce phase

BINANCE_UNIVERSE = list(config.KNOWN_SYMBOLS)  # 30 symbols == strategies.yaml universe


# -- cache ------------------------------------------------------------------

def _path(kind, symbol=""):
    return DATA_DIR / kind / f"{symbol or '_'}.json"


def save(kind, symbol, rows, extra_meta=None):
    p = _path(kind, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = {"t0": T0, "t1": T1, "downloaded_at": int(time.time())}
    meta.update(extra_meta or {})
    p.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False))


def load(kind, symbol=""):
    p = _path(kind, symbol)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())["rows"]
    except (json.JSONDecodeError, KeyError):
        return None


def is_cached(kind, symbol="", expected_t0=None):
    p = _path(kind, symbol)
    if not p.exists():
        return False
    if expected_t0 is None:
        return True
    try:
        return json.loads(p.read_text())["meta"].get("t0") == expected_t0
    except (json.JSONDecodeError, KeyError):
        return False


# -- throttled Binance GET ---------------------------------------------------

_WEIGHTS = {"/fapi/v1/klines": 10, "/fapi/v1/aggTrades": 20, "/fapi/v1/fundingRate": 1}


async def _get(client, path, params, budget_per_min=1500):
    weight = _WEIGHTS.get(path, 1)
    for attempt in range(5):
        try:
            resp = await client.get(FUT + path, params=params)
            if resp.status_code in (418, 429):
                wait = int(resp.headers.get("Retry-After", 30))
                log.warning("rate limited on %s — sleeping %ds", path, wait)
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            await asyncio.sleep(weight / budget_per_min * 60)
            return resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500:     # bad symbol etc. — no retry
                raise
            if attempt == 4:
                raise
            log.warning("%s 5xx (%s) — retry %d", path, exc, attempt + 1)
            await asyncio.sleep(2 ** attempt)
        except httpx.TransportError as exc:
            if attempt == 4:
                raise
            log.warning("%s failed (%s) — retry %d", path, exc, attempt + 1)
            await asyncio.sleep(2 ** attempt)
    return None


# -- Binance downloads -------------------------------------------------------

async def dl_klines_1m(client, symbol):
    rows, start = [], T0 * 1000
    while start < T1 * 1000:
        raw = await _get(client, "/fapi/v1/klines",
                         {"symbol": symbol, "interval": "1m", "startTime": start,
                          "endTime": T1 * 1000 - 1, "limit": 1500})
        if not raw:
            break
        rows += [{"ts": int(k[0] // 1000), "open": float(k[1]), "high": float(k[2]),
                  "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])}
                 for k in raw]
        start = raw[-1][0] + 60_000
        if len(raw) < 1500:
            break
    return rows


async def dl_premium_1h(client, symbol):
    """premiumIndexKlines 1h — historical basis %, feeds factors.basis_pct_of."""
    rows, start = [], T0 * 1000
    while start < T1 * 1000:
        raw = await _get(client, "/fapi/v1/premiumIndexKlines",
                         {"symbol": symbol, "interval": "1h", "startTime": start,
                          "endTime": T1 * 1000 - 1, "limit": 1000})
        if not raw:
            break
        rows += [{"ts": int(k[0] // 1000), "close": float(k[4])} for k in raw]
        start = raw[-1][0] + 3_600_000
        if len(raw) < 1000:
            break
    return rows


async def dl_funding_rate(client, symbol):
    rows, start = [], T0 * 1000
    while start < T1 * 1000:
        raw = await _get(client, "/fapi/v1/fundingRate",
                         {"symbol": symbol, "startTime": start,
                          "endTime": T1 * 1000 - 1, "limit": 1000})
        if not raw:
            break
        rows += raw
        start = int(raw[-1]["fundingTime"]) + 1
        if len(raw) < 1000:
            break
    return rows


async def dl_futures_data(client, path, symbol, period, step_sec):
    """Paged download for /futures/data/* (limit<=500, last-30d retention)."""
    rows, start = [], T0 * 1000
    while start < T1 * 1000:
        end = min(start + 500 * step_sec * 1000 - 1, T1 * 1000 - 1)
        raw = await _get(client, path,
                         {"symbol": symbol, "period": period,
                          "startTime": start, "endTime": end, "limit": 500})
        if isinstance(raw, list):
            rows += raw
        start = end + 1
    seen, out = set(), []
    for r in rows:
        ts = int(r.get("timestamp") or 0)
        if ts not in seen:
            seen.add(ts)
            out.append(r)
    out.sort(key=lambda r: int(r.get("timestamp") or 0))
    return out


# -- S11 candidate windows + aggTrades --------------------------------------

def scan_windows(klines):
    """Candidate 1m bars for S11 second-level replay (see thresholds above).
    Returns merged padded [t_from, t_to] windows."""
    hits = []
    prev_rng = 0.0
    for i, k in enumerate(klines):
        if not k["open"]:
            continue
        rng1 = (k["high"] - k["low"]) / k["open"] * 100
        chg2 = 0
        if i > 0 and klines[i - 1]["open"]:
            chg2 = abs(k["close"] / klines[i - 1]["open"] - 1) * 100
        if rng1 >= WIN_RANGE1 or chg2 >= WIN_CHG2 or rng1 + prev_rng >= WIN_JOINT:
            hits.append(k["ts"])
        prev_rng = rng1
    windows = []
    for ts in hits:
        lo, hi = ts - PAD_BEFORE, ts + 60 + PAD_AFTER
        if windows and lo <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], hi)
        else:
            windows.append([lo, hi])
    return windows


async def dl_aggtrades_window(client, symbol, t_from, t_to):
    """Fetch aggTrades for one window, downsampled to per-second last price."""
    per_sec = {}
    start = t_from * 1000
    while start < t_to * 1000:
        raw = await _get(client, "/fapi/v1/aggTrades",
                         {"symbol": symbol, "startTime": start,
                          "endTime": t_to * 1000 - 1, "limit": 1000})
        if not raw:
            break
        for r in raw:
            per_sec[int(r["T"] // 1000)] = float(r["p"])
        nxt = int(raw[-1]["T"]) + 1
        if nxt <= start:
            nxt = start + 1000
        start = nxt
        if len(raw) < 1000:
            break
    return sorted(per_sec.items())


async def dl_aggtrades_windows(client, symbol):
    ks = load("klines_1m", symbol)
    if not ks:
        raise RuntimeError(f"klines_1m missing for {symbol} — download klines first")
    windows = scan_windows(ks)
    ticks = {}
    for i, (lo, hi) in enumerate(windows):
        ticks[str(lo)] = await dl_aggtrades_window(client, symbol, lo, hi)
        if (i + 1) % 10 == 0:
            log.info("%s aggTrades windows %d/%d", symbol, i + 1, len(windows))
    return {"windows": windows, "ticks": ticks}


# -- Clawby downloads --------------------------------------------------------

def _coin(symbol):
    # mirrors backend.factors._coin exactly (incl. the 1000-prefix fix)
    s = symbol.replace("USDT", "").replace("BUSD", "")
    return s[4:] if s.startswith("1000") and len(s) > 4 else s


async def dl_clawby_symbol(kind, symbol):
    """Per-symbol Clawby history over the window (1h interval, with warmup)."""
    p = {"interval": "1h", "limit": 1000,
         "start_time": (T0 - WARMUP) * 1000, "end_time": T1 * 1000}
    if kind == "clawby_funding_agg":
        data = await clawby.relay_safe("futures_funding_rate_oi_weight_history",
                                       {**p, "symbol": _coin(symbol)})
    elif kind == "clawby_liq_agg":
        data = await clawby.relay_safe("futures_liquidation_aggregated_history",
                                       {**p, "symbol": _coin(symbol),
                                        "exchange_list": "Binance,OKX,Bybit"})
    elif kind == "clawby_cvd":
        data = await clawby.relay_safe("futures_cvd_history",
                                       {**p, "symbol": symbol, "exchange": "Binance"})
    elif kind == "clawby_oi_agg":
        data = await clawby.relay_safe("futures_open_interest_aggregated_history",
                                       {**p, "symbol": _coin(symbol)})
    else:
        raise KeyError(kind)
    return data if isinstance(data, list) else None


async def dl_clawby_global(kind):
    if kind == "clawby_coinbase_prem":
        data = await clawby.relay_safe("coinbase_premium_index",
                                       {"interval": "1h", "limit": 1000,
                                        "start_time": (T0 - WARMUP) * 1000,
                                        "end_time": T1 * 1000})
    elif kind == "clawby_etf_flow":
        data = await clawby.relay_safe("etf_bitcoin_flow_history")
    elif kind == "clawby_econ_cal":
        data = await clawby.relay_safe("calendar_economic_data",
                                       {"start_time": T0 * 1000, "end_time": T1 * 1000,
                                        "language": "zh-CN"})
    else:
        raise KeyError(kind)
    return data if isinstance(data, list) else ([data] if data else None)


# -- orchestration -----------------------------------------------------------

FUTURES_DATA = [
    ("oi_5m", "/futures/data/openInterestHist", "5m", 300),
    ("lsr_global_1h", "/futures/data/globalLongShortAccountRatio", "1h", 3600),
    ("lsr_top_1h", "/futures/data/topLongShortPositionRatio", "1h", 3600),
    ("taker_5m", "/futures/data/takerlongshortRatio", "5m", 300),
]
CLAWBY_SYMBOL_KINDS = ["clawby_funding_agg", "clawby_liq_agg", "clawby_cvd",
                       "clawby_oi_agg"]
CLAWBY_GLOBAL_KINDS = ["clawby_coinbase_prem", "clawby_etf_flow", "clawby_econ_cal"]


async def _guarded(coro, kind, sym):
    """Run one download; on 4xx mark the (kind, sym) invalid so we never
    retry a symbol Binance doesn't list (e.g. PEPEUSDT vs 1000PEPEUSDT)."""
    try:
        return await coro
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code < 500:
            log.error("%s %s: HTTP %s — marked invalid, skipping",
                      kind, sym, exc.response.status_code)
            save(kind, sym, [], {"invalid": True})
        else:
            log.error("%s %s failed: %s", kind, sym, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        log.error("%s %s failed: %s", kind, sym, exc)
        return None


async def download_all(symbols=None, force=False, skip_aggtrades=False):
    symbols = symbols or BINANCE_UNIVERSE
    async with httpx.AsyncClient(timeout=30) as client:
        # 1) time-critical: /futures/data/* keeps only ~30 days
        for kind, path, period, step in FUTURES_DATA:
            for sym in symbols:
                if not force and is_cached(kind, sym):
                    continue
                rows = await _guarded(
                    dl_futures_data(client, path, sym, period, step), kind, sym)
                if rows is not None:
                    save(kind, sym, rows)
                    log.info("%s %s: %d rows", kind, sym, len(rows))
        # 2) klines + funding + premium
        for sym in symbols:
            for kind, fn in (("klines_1m", dl_klines_1m),
                             ("funding_rate", dl_funding_rate),
                             ("premium_1h", dl_premium_1h)):
                if not force and is_cached(kind, sym):
                    continue
                rows = await _guarded(fn(client, sym), kind, sym)
                if rows is not None:
                    save(kind, sym, rows)
                    log.info("%s %s: %d rows", kind, sym, len(rows))
        # 3) Clawby (with 7d warmup; stale caches without warmup get refetched)
        warm_t0 = T0 - WARMUP
        for kind in CLAWBY_SYMBOL_KINDS:
            for sym in symbols:
                if not force and is_cached(kind, sym, expected_t0=warm_t0):
                    continue
                rows = await dl_clawby_symbol(kind, sym)
                save(kind, sym, rows or [], {"empty": not rows, "t0": warm_t0})
                log.info("%s %s: %d rows", kind, sym, len(rows or []))
        for kind in CLAWBY_GLOBAL_KINDS:
            expected = warm_t0 if kind == "clawby_coinbase_prem" else None
            if force or not is_cached(kind, expected_t0=expected):
                rows = await dl_clawby_global(kind)
                save(kind, "", rows or [],
                     {"empty": not rows,
                      **({"t0": warm_t0} if kind == "clawby_coinbase_prem" else {})})
                log.info("%s: %d rows", kind, len(rows or []))
        # 4) S11 aggTrades candidate windows (slowest, least urgent)
        if not skip_aggtrades:
            for sym in symbols:
                if not force and is_cached("aggtrades_windows", sym):
                    continue
                if not load("klines_1m", sym):
                    save("aggtrades_windows", sym, {"windows": [], "ticks": {}},
                         {"invalid": True})
                    continue
                obj = await _guarded(dl_aggtrades_windows(client, sym),
                                     "aggtrades_windows", sym)
                if obj is not None:
                    save("aggtrades_windows", sym, obj)
                    log.info("aggtrades_windows %s: %d windows",
                             sym, len(obj["windows"]))


def valid_symbols():
    """Symbols with real kline history in cache (invalid/absent excluded)."""
    return [s for s in BINANCE_UNIVERSE if load("klines_1m", s)]


def coverage_report():
    """Per kind: how many symbols cached + row-count range. For `download` CLI."""
    out = {}
    per_symbol = ["klines_1m", "funding_rate", "premium_1h"] \
        + [k for k, *_ in FUTURES_DATA] + CLAWBY_SYMBOL_KINDS + ["aggtrades_windows"]
    for kind in per_symbol:
        counts = []
        for sym in BINANCE_UNIVERSE:
            rows = load(kind, sym)
            if rows is None:
                continue
            counts.append(len(rows.get("ticks", {})) if kind == "aggtrades_windows"
                          and isinstance(rows, dict) else len(rows))
        out[kind] = {"symbols": len(counts),
                     "rows_min": min(counts) if counts else 0,
                     "rows_max": max(counts) if counts else 0}
    for kind in CLAWBY_GLOBAL_KINDS:
        rows = load(kind)
        out[kind] = {"symbols": 1 if rows is not None else 0,
                     "rows_min": len(rows or []), "rows_max": len(rows or [])}
    return out

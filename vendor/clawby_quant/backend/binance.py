"""Binance REST client: public futures/spot market data + signed trading.

Uses the official REST API directly (same surface the Clawby binance-cli
connector wraps); keys stay local, nothing routes through Clawby.
"""
import hashlib
import hmac
import logging
import time
import urllib.parse

import httpx

from . import config

log = logging.getLogger("binance")

SPOT = "https://api.binance.com"
FUT = "https://fapi.binance.com"


async def _get(base, path, params=None, timeout=15):
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(base + path, params=params or {})
        resp.raise_for_status()
        return resp.json()


async def _signed(method, base, path, params=None, timeout=15):
    params = dict(params or {})
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 10000
    qs = urllib.parse.urlencode(params)
    sig = hmac.new(config.BINANCE_SECRET_KEY.encode(), qs.encode(), hashlib.sha256).hexdigest()
    url = f"{base}{path}?{qs}&signature={sig}"
    headers = {"X-MBX-APIKEY": config.BINANCE_API_KEY}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.request(method, url, headers=headers)
        resp.raise_for_status()
        return resp.json()


# -- public market data (futures) -------------------------------------------

async def klines(symbol, interval="1m", limit=200):
    """Futures klines -> list of dicts (ts, open, high, low, close, volume)."""
    raw = await _get(FUT, "/fapi/v1/klines",
                     {"symbol": symbol, "interval": interval, "limit": limit})
    return [{"ts": int(k[0] // 1000), "open": float(k[1]), "high": float(k[2]),
             "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])}
            for k in raw]


async def mark_price(symbol=None):
    params = {"symbol": symbol} if symbol else {}
    return await _get(FUT, "/fapi/v1/premiumIndex", params)


async def depth(symbol, limit=50):
    return await _get(FUT, "/fapi/v1/depth", {"symbol": symbol, "limit": limit})


async def open_interest(symbol):
    return await _get(FUT, "/fapi/v1/openInterest", {"symbol": symbol})


async def open_interest_hist(symbol, period="1h", limit=48):
    return await _get(FUT, "/futures/data/openInterestHist",
                      {"symbol": symbol, "period": period, "limit": limit})


async def long_short_global(symbol, period="1h", limit=48):
    return await _get(FUT, "/futures/data/globalLongShortAccountRatio",
                      {"symbol": symbol, "period": period, "limit": limit})


async def long_short_top_positions(symbol, period="1h", limit=168):
    return await _get(FUT, "/futures/data/topLongShortPositionRatio",
                      {"symbol": symbol, "period": period, "limit": limit})


async def taker_ratio(symbol, period="5m", limit=48):
    return await _get(FUT, "/futures/data/takerlongshortRatio",
                      {"symbol": symbol, "period": period, "limit": limit})


async def funding_history(symbol, limit=30):
    return await _get(FUT, "/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit})


async def ticker_24h_all():
    return await _get(FUT, "/fapi/v1/ticker/24hr")


# -- signed (trading + account, futures) ------------------------------------

async def account():
    return await _signed("GET", FUT, "/fapi/v2/account")


async def place_market_order(symbol, side, qty, reduce_only=False):
    params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty}
    if reduce_only:
        params["reduceOnly"] = "true"
    return await _signed("POST", FUT, "/fapi/v1/order", params)


async def set_leverage(symbol, leverage):
    """Best-effort set initial leverage before opening (live only)."""
    try:
        return await _signed("POST", FUT, "/fapi/v1/leverage",
                             {"symbol": symbol, "leverage": int(leverage)})
    except Exception as exc:  # noqa: BLE001
        log.warning("set_leverage %s x%s failed: %s", symbol, leverage, exc)
        return None


async def exchange_info():
    return await _get(FUT, "/fapi/v1/exchangeInfo")


_qty_steps = {}


async def qty_step(symbol):
    """LOT_SIZE stepSize for rounding order quantities."""
    global _qty_steps
    if not _qty_steps:
        info = await exchange_info()
        for s in info.get("symbols", []):
            for f in s.get("filters", []):
                if f.get("filterType") == "LOT_SIZE":
                    _qty_steps[s["symbol"]] = float(f["stepSize"])
    return _qty_steps.get(symbol, 0.001)


def round_step(qty, step):
    if step <= 0:
        return qty
    precision = max(0, len(str(step).rstrip("0").split(".")[-1])) if "." in str(step) else 0
    return round((qty // step) * step, precision)

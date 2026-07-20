"""Binance futures WebSocket: real-time price feed + tick-level price buffer.

<symbol>@bookTicker gives tick-level best bid/ask; mid = (bid+ask)/2 feeds the
0.5s position loop AND a rolling per-symbol price history so the high-frequency
crash/spike strategy (S11) can measure sub-second velocity — the true
second-level signal that minute-level liquidation data cannot provide.
Reconnects on drop and when the universe changes.
"""
import asyncio
import json
import logging
import time
from collections import deque

import websockets

from . import config

log = logging.getLogger("ws")

PRICES = {}       # symbol -> {mark, bid, ask, ts}
PRICE_HIST = {}   # symbol -> deque[(ts, mid)]  (~last 10 min of ticks)
_HIST_MAX = 4000


def _record(sym, mid):
    dq = PRICE_HIST.get(sym)
    if dq is None:
        dq = deque(maxlen=_HIST_MAX)
        PRICE_HIST[sym] = dq
    dq.append((time.time(), mid))


def live_price(symbol, max_age_sec=10):
    p = PRICES.get(symbol)
    if p and time.time() - p["ts"] <= max_age_sec:
        return p["mark"]
    return 0.0


def price_ago(symbol, secs):
    """Price roughly `secs` seconds ago (last tick at/older than the cutoff)."""
    dq = PRICE_HIST.get(symbol)
    if not dq:
        return None
    cutoff = time.time() - secs
    best = None
    for ts, p in dq:
        if ts <= cutoff:
            best = p
        else:
            break
    if best is None:  # buffer younger than the window — use earliest if it spans enough
        first_ts, first_p = dq[0]
        if time.time() - first_ts >= secs * 0.5:
            best = first_p
    return best


def change_pct(symbol, secs):
    """Percent change of current price vs `secs` ago (negative = drop)."""
    old = price_ago(symbol, secs)
    cur = live_price(symbol)
    if old and cur:
        return (cur - old) / old * 100
    return None


def buffer_span(symbol):
    dq = PRICE_HIST.get(symbol)
    if not dq:
        return 0
    return time.time() - dq[0][0]


def _stream_url(symbols):
    streams = "/".join(f"{s.lower()}@bookTicker" for s in symbols)
    return f"wss://fstream.binance.com/stream?streams={streams}"


async def ws_loop():
    backoff = 1
    while True:
        subscribed = list(config.UNIVERSE)
        try:
            async with websockets.connect(_stream_url(subscribed), ping_interval=20,
                                          close_timeout=5) as conn:
                log.info("bookTicker stream connected (%d symbols)", len(subscribed))
                backoff = 1
                async for raw in conn:
                    msg = json.loads(raw)
                    d = msg.get("data") or {}
                    sym = d.get("s")
                    if sym:
                        bid = float(d.get("b") or 0)
                        ask = float(d.get("a") or 0)
                        if bid and ask:
                            mid = (bid + ask) / 2
                            PRICES[sym] = {"mark": mid, "bid": bid, "ask": ask, "ts": time.time()}
                            _record(sym, mid)
                    # universe changed -> reconnect with the new subscription
                    if set(config.UNIVERSE) != set(subscribed):
                        log.info("universe changed — reconnecting stream")
                        break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("stream dropped: %s — reconnecting in %ds", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

"""Clawby relay client: async, throttled, tolerant of the two-layer envelope.

Outer envelope: {source, data, credits}; derivatives interfaces nest another
{code, data} where code=="0" means success.
"""
import asyncio
import logging
import time

import httpx

from . import config

log = logging.getLogger("clawby")

_MIN_GAP = 0.25          # admin plan: 360/min; stay well under
_last_call = 0.0
_gap_lock = asyncio.Lock()


async def relay(name, params=None, timeout=30):
    global _last_call
    async with _gap_lock:
        wait = _MIN_GAP - (time.monotonic() - _last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{config.CLAWBY_BASE}/api/relay",
            headers={"X-API-Key": config.CLAWBY_API_KEY,
                     "Content-Type": "application/json"},
            json={"name": name, "params": params or {}},
        )
        resp.raise_for_status()
        body = resp.json()
    data = body.get("data")
    # unwrap the derivatives-style inner envelope
    if isinstance(data, dict) and "code" in data:
        if str(data.get("code")) != "0":
            raise RuntimeError(f"{name}: upstream code={data.get('code')} msg={data.get('msg')}")
        data = data.get("data")
    return data


async def relay_safe(name, params=None, timeout=30):
    """Never raises; returns None on any failure (factor collectors use this)."""
    try:
        return await relay(name, params, timeout)
    except Exception as exc:  # noqa: BLE001
        log.warning("relay %s failed: %s", name, exc)
        return None

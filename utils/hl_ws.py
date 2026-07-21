"""Hyperliquid WebSocket client (userEvents / orderUpdates)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

MessageHandler = Callable[[str, dict], Awaitable[None] | None]


class HyperliquidWsClient:
    """Minimal HL WS: subscribe userEvents per address, auto-reconnect + ping."""

    def __init__(self, ws_url: str = "wss://api.hyperliquid.xyz/ws") -> None:
        self.ws_url = ws_url
        self._ws: Any = None
        self._running = False
        self._subs: dict[str, dict] = {}
        self._handlers: dict[str, MessageHandler] = {}
        self._task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self.reconnect_delay = 5
        self.ping_interval = 50

    @property
    def running(self) -> bool:
        return self._running

    def set_handler(self, address: str, handler: MessageHandler) -> None:
        addr = address.lower()
        self._handlers[addr] = handler
        self._subs[addr] = {
            "method": "subscribe",
            "subscription": {"type": "userEvents", "user": address},
        }

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="hl-ws-loop")

    async def stop(self) -> None:
        self._running = False
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _run_loop(self) -> None:
        while self._running:
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=None,
                    close_timeout=5,
                    max_size=8 * 1024 * 1024,
                ) as ws:
                    self._ws = ws
                    logger.info("HL WS connected %s", self.ws_url)
                    for sub in self._subs.values():
                        await ws.send(json.dumps(sub))
                    self._ping_task = asyncio.create_task(self._ping_loop(ws))
                    async for raw in ws:
                        await self._on_raw(raw)
            except asyncio.CancelledError:
                break
            except ConnectionClosed as exc:
                logger.warning("HL WS closed: %s", exc)
            except Exception as exc:
                logger.warning("HL WS error: %s", exc)
            finally:
                self._ws = None
                if self._ping_task and not self._ping_task.done():
                    self._ping_task.cancel()
            if self._running:
                await asyncio.sleep(self.reconnect_delay)

    async def _ping_loop(self, ws: Any) -> None:
        try:
            while self._running:
                await asyncio.sleep(self.ping_interval)
                await ws.send(json.dumps({"method": "ping"}))
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug("HL WS ping stopped: %s", exc)

    async def _on_raw(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        channel = str(msg.get("channel") or "")
        if channel == "pong":
            return
        if channel == "subscriptionResponse":
            logger.debug("HL WS subscribed: %s", msg.get("data"))
            return
        data = msg.get("data")
        if not isinstance(data, dict):
            return
        # userEvents payload may include user field
        user = str(data.get("user") or "").lower()
        if user and user in self._handlers:
            await self._dispatch(user, channel, data)
            return
        # Single-handler client: deliver to the only subscription
        if len(self._handlers) == 1:
            addr = next(iter(self._handlers))
            await self._dispatch(addr, channel, data)
            return
        logger.debug("HL WS event dropped (ambiguous user): channel=%s keys=%s", channel, list(data.keys()))

    async def _dispatch(self, addr: str, channel: str, data: dict) -> None:
        handler = self._handlers.get(addr)
        if not handler:
            return
        res = handler(channel, data)
        if asyncio.iscoroutine(res):
            await res

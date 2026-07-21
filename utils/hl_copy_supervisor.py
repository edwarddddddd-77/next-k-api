"""Background supervisor: HL WebSocket → paper copy for watchlist wallets."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool) -> bool:
    raw = (os.getenv(key) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


class HlCopySupervisor:
    """One WS client per watchlist address; paper-copies fills. No private keys."""

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._clients: list[Any] = []
        self._status: dict[str, Any] = {
            "running": False,
            "mode": "paper",
            "started_at": None,
            "wallets": [],
            "last_fill_at": None,
            "fills_seen": 0,
            "error": None,
        }
        self._lock = threading.Lock()

    @property
    def status(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self._status)
        try:
            from utils.hl_paper_copy import load_paper, paper_config

            out["paper"] = {
                "equity": load_paper().get("equity"),
                "balance": load_paper().get("balance"),
                "realized_pnl": load_paper().get("realized_pnl"),
                "open_positions": len(load_paper().get("positions") or {}),
            }
            out["config"] = paper_config()
        except Exception as exc:
            out["paper_error"] = str(exc)
        return out

    def should_start(self) -> bool:
        return _env_bool("HL_COPY_ENABLED", True)

    def start(self) -> None:
        if not self.should_start():
            logger.info("HL copy supervisor disabled (HL_COPY_ENABLED=0)")
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._thread_main,
            name="hl-copy-supervisor",
            daemon=True,
        )
        self._thread.start()
        logger.info("HL copy supervisor thread started (paper mode)")

    def stop(self) -> None:
        loop = self._loop
        if loop and loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._async_stop(), loop)
            try:
                fut.result(timeout=10)
            except Exception as exc:
                logger.warning("HL copy stop: %s", exc)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        with self._lock:
            self._status["running"] = False

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._async_main())
        except Exception as exc:
            logger.exception("HL copy supervisor crashed: %s", exc)
            with self._lock:
                self._status["running"] = False
                self._status["error"] = str(exc)

    async def _async_main(self) -> None:
        self._loop = asyncio.get_running_loop()
        from utils.hl_short_term import load_watchlist
        from utils.hl_ws import HyperliquidWsClient
        from utils.hl_paper_copy import ingest_user_event

        wallets = load_watchlist()
        if not wallets:
            logger.warning("HL copy: empty watchlist")
            with self._lock:
                self._status["error"] = "empty watchlist"
            return

        with self._lock:
            self._status.update(
                {
                    "running": True,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "wallets": [
                        {"id": w.get("id"), "address": w.get("address")} for w in wallets
                    ],
                    "error": None,
                }
            )

        clients: list[HyperliquidWsClient] = []
        for w in wallets:
            addr = str(w.get("address") or "")
            if not addr:
                continue
            wid = w.get("id") or addr[:10]
            client = HyperliquidWsClient()

            async def _handler(channel: str, data: dict, *, _addr=addr, _wid=wid) -> None:
                fills = data.get("fills")
                if not isinstance(fills, list) or not fills:
                    return
                applied = await asyncio.to_thread(ingest_user_event, _addr, data)
                with self._lock:
                    self._status["fills_seen"] = int(self._status.get("fills_seen") or 0) + len(
                        fills
                    )
                    self._status["last_fill_at"] = datetime.now(timezone.utc).isoformat()
                    self._status["last_wallet"] = _wid
                    self._status["last_applied"] = len(applied)
                logger.info(
                    "HL WS fills wallet=%s n=%s applied=%s",
                    _wid,
                    len(fills),
                    len(applied),
                )

            client.set_handler(addr, _handler)
            clients.append(client)
            await client.start()
            logger.info("HL WS subscribed %s (%s)", wid, addr[:10])

        self._clients = clients
        # Keep alive until cancelled
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await self._async_stop()

    async def _async_stop(self) -> None:
        for c in self._clients:
            try:
                await c.stop()
            except Exception:
                pass
        self._clients = []
        with self._lock:
            self._status["running"] = False


hl_copy_supervisor = HlCopySupervisor()

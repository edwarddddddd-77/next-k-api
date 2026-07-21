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
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._clients: dict[str, Any] = {}  # address.lower() -> HyperliquidWsClient
        self._shutdown = False
        self._status: dict[str, Any] = {
            "running": False,
            "mode": "paper",
            "started_at": None,
            "wallets": [],
            "last_fill_at": None,
            "fills_seen": 0,
            "error": None,
            "watchlist_path": None,
        }
        self._lock = threading.Lock()

    @property
    def status(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self._status)
            out["subscribed"] = sorted(self._clients.keys())
        try:
            from utils.hl_paper_copy import load_paper, paper_config

            book = load_paper()
            out["paper"] = {
                "equity": book.get("equity"),
                "balance": book.get("balance"),
                "realized_pnl": book.get("realized_pnl"),
                "open_positions": len(book.get("positions") or {}),
                "bot_count": book.get("bot_count"),
                "bots": [
                    {
                        "id": b.get("id"),
                        "address": b.get("address"),
                        "equity": b.get("equity"),
                        "balance": b.get("balance"),
                        "realized_pnl": b.get("realized_pnl"),
                        "copy_ratio": b.get("copy_ratio"),
                        "target_av": b.get("target_av"),
                        "risk_halted": b.get("risk_halted"),
                        "positions": len(b.get("positions") or {}),
                    }
                    for b in (book.get("bots") or {}).values()
                ],
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
        self._shutdown = False
        self._thread = threading.Thread(
            target=self._thread_main,
            name="hl-copy-supervisor",
            daemon=True,
        )
        self._thread.start()
        logger.info("HL copy supervisor thread started (paper mode)")

    def stop(self) -> None:
        self._shutdown = True
        loop = self._loop
        if loop and loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._async_stop(), loop)
            try:
                fut.result(timeout=10)
            except Exception as exc:
                logger.warning("HL copy stop: %s", exc)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=8)
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

    async def _sleep_interruptible(self, seconds: float) -> None:
        end = asyncio.get_running_loop().time() + max(0.0, seconds)
        while not self._shutdown:
            remaining = end - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(1.0, remaining))

    async def _ensure_wallet_clients(self, wallets: list[dict]) -> None:
        from utils.hl_ws import HyperliquidWsClient
        from utils.hl_paper_copy import ingest_user_event

        wanted: dict[str, dict] = {}
        for w in wallets:
            addr = str(w.get("address") or "").strip()
            if not addr:
                continue
            wanted[addr.lower()] = w

        # Drop removed
        for addr in list(self._clients.keys()):
            if addr not in wanted:
                client = self._clients.pop(addr)
                try:
                    await client.stop()
                except Exception:
                    pass
                logger.info("HL WS unsubscribed %s", addr[:10])

        # Add new
        for addr, w in wanted.items():
            if addr in self._clients:
                continue
            wid = w.get("id") or addr[:10]
            client = HyperliquidWsClient()

            async def _handler(
                channel: str, data: dict, *, _addr=addr, _wid=wid
            ) -> None:
                fills = data.get("fills")
                if not isinstance(fills, list) or not fills:
                    return
                is_snap = bool(data.get("isSnapshot"))
                try:
                    applied = await asyncio.to_thread(ingest_user_event, _addr, data)
                except Exception as exc:
                    logger.exception(
                        "HL ingest failed wallet=%s channel=%s: %s", _wid, channel, exc
                    )
                    with self._lock:
                        self._status["last_error"] = str(exc)
                        self._status["last_wallet"] = _wid
                    return
                with self._lock:
                    self._status["fills_seen"] = int(self._status.get("fills_seen") or 0) + len(
                        fills
                    )
                    self._status["last_fill_at"] = datetime.now(timezone.utc).isoformat()
                    self._status["last_wallet"] = _wid
                    self._status["last_applied"] = len(applied)
                    self._status["last_channel"] = channel
                    self._status["last_snapshot"] = is_snap
                    self._status.pop("last_error", None)
                logger.info(
                    "HL WS fills wallet=%s channel=%s n=%s snap=%s applied=%s",
                    _wid,
                    channel,
                    len(fills),
                    is_snap,
                    len(applied),
                )

            client.set_handler(addr, _handler)
            await client.start()
            self._clients[addr] = client
            logger.info("HL WS subscribed %s (%s)", wid, addr[:10])

        with self._lock:
            self._status["wallets"] = [
                {"id": w.get("id"), "address": w.get("address")} for w in wanted.values()
            ]
            self._status["subscribed"] = sorted(self._clients.keys())

    async def _async_main(self) -> None:
        self._loop = asyncio.get_running_loop()
        from utils.hl_short_term import load_watchlist, _watchlist_path
        from utils.hl_paper_copy import load_paper, refresh_marks

        wl_path = str(_watchlist_path())
        with self._lock:
            self._status.update(
                {
                    "running": True,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "error": None,
                    "watchlist_path": wl_path,
                }
            )
        logger.info("HL copy using watchlist %s", wl_path)

        # Ensure paper bots exist for every watchlist wallet
        try:
            load_paper()
        except Exception as exc:
            logger.warning("paper ensure on start: %s", exc)

        mark_every = float(os.getenv("HL_PAPER_MARK_LOOP_SEC", "60") or 60)
        reload_every = float(os.getenv("HL_WATCHLIST_RELOAD_SEC", "120") or 120)
        last_reload = 0.0

        try:
            while not self._shutdown:
                now = asyncio.get_running_loop().time()
                if now - last_reload >= max(15.0, reload_every) or not self._clients:
                    last_reload = now
                    wallets = load_watchlist()
                    if not wallets:
                        logger.warning("HL copy: empty watchlist — retrying")
                        with self._lock:
                            self._status["error"] = "empty watchlist"
                            self._status["wallets"] = []
                        await self._sleep_interruptible(30)
                        continue
                    with self._lock:
                        self._status["error"] = None
                    try:
                        load_paper()  # pick up new bot slots
                    except Exception:
                        pass
                    await self._ensure_wallet_clients(wallets)

                try:
                    await asyncio.to_thread(refresh_marks)
                except Exception as exc:
                    logger.debug("paper mark refresh: %s", exc)

                await self._sleep_interruptible(max(15.0, mark_every))
        except asyncio.CancelledError:
            pass
        finally:
            await self._async_stop()

    async def _async_stop(self) -> None:
        self._shutdown = True
        for addr, client in list(self._clients.items()):
            try:
                await client.stop()
            except Exception:
                pass
            self._clients.pop(addr, None)
        with self._lock:
            self._status["running"] = False
            self._status["subscribed"] = []


hl_copy_supervisor = HlCopySupervisor()

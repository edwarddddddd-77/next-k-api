"""进程内自动拉起 ICT 2022 vnpy。"""

from __future__ import annotations

import logging
import os
import threading
from threading import Event
from typing import Any, Dict, Optional

from orb.ict.config import IctVnpyConfig

logger = logging.getLogger(__name__)


def _autostart_enabled() -> bool:
    raw = (os.getenv("ICT_VNPY_AUTO_START") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


class IctVnpySupervisor:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = Event()
        self._last_status: Dict[str, Any] = {}
        self._restart_count = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_status(self) -> Dict[str, Any]:
        return dict(self._last_status)

    def should_start(self) -> bool:
        if not _autostart_enabled():
            return False
        if os.getenv("ICT_VNPY_STANDALONE", "").strip().lower() in ("1", "true", "yes", "on"):
            return False
        ict = IctVnpyConfig.from_env()
        return bool(ict.enabled and ict.is_vnpy_engine())

    def start(self) -> None:
        if not self.should_start():
            logger.info("[ict-vnpy] supervisor skipped (ICT_VNPY_ENABLED=%s)", os.getenv("ICT_VNPY_ENABLED", "0"))
            return
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ict-vnpy-supervisor", daemon=True)
        self._thread.start()
        logger.info("[ict-vnpy] supervisor thread started")

    def stop(self, *, join_timeout: float = 45.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            self._thread = None

    def _run(self) -> None:
        restart_delay = max(5.0, float(os.getenv("ICT_VNPY_RESTART_SEC") or 30))
        while not self._stop.is_set():
            engine = None
            try:
                from orb.ict.vnpy.runner import IctVnpyEngine

                init_wait = float(os.getenv("ICT_VNPY_INIT_WAIT_SEC") or 60)
                engine = IctVnpyEngine()
                status = engine.bootstrap(init_wait_sec=init_wait)
                self._last_status = {**status, "restart_count": self._restart_count}
                if not status.get("ok"):
                    logger.error("[ict-vnpy] bootstrap failed: %s", status)
                    return
                if status.get("skipped"):
                    logger.info("[ict-vnpy] bootstrap skipped: %s", status.get("reason"))
                    return
                engine.run_until(self._stop)
            except ImportError as exc:
                logger.error("[ict-vnpy] vnpy import failed: %s", exc)
                return
            except Exception as exc:
                logger.exception("[ict-vnpy] supervisor crashed: %s", exc)
                self._last_status = {"ok": False, "reason": "crash", "error": str(exc)}
            finally:
                if engine is not None:
                    try:
                        engine.shutdown()
                    except Exception as exc:
                        logger.warning("[ict-vnpy] shutdown: %s", exc)
            if self._stop.is_set():
                break
            self._restart_count += 1
            self._stop.wait(restart_delay)


ict_vnpy_supervisor = IctVnpySupervisor()

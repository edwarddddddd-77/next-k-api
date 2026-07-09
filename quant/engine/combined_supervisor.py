"""多 lane 联合 vnpy supervisor（执行层）。"""

from __future__ import annotations

import logging
import os
import threading
from threading import Event
from typing import Any, Dict, Optional

from quant.engine.lane import get_enabled_vnpy_lanes

logger = logging.getLogger(__name__)


def _autostart_enabled() -> bool:
    raw = (os.getenv("VNPY_AUTO_START") or os.getenv("ORB_VNPY_AUTO_START") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


class CombinedVnpySupervisor:
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
        if os.getenv("VNPY_STANDALONE", "").strip().lower() in ("1", "true", "yes", "on"):
            return False
        return bool(get_enabled_vnpy_lanes())

    def start(self) -> None:
        if not self.should_start():
            logger.info("[combined-vnpy] supervisor skipped (no lanes enabled)")
            return
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="combined-vnpy-supervisor", daemon=True)
        self._thread.start()
        logger.info("[combined-vnpy] supervisor started lanes=%s", [n for n, _ in get_enabled_vnpy_lanes()])

    def stop(self, *, join_timeout: float = 45.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            self._thread = None

    def _run(self) -> None:
        restart_delay = max(5.0, float(os.getenv("VNPY_RESTART_SEC") or os.getenv("ORB_VNPY_RESTART_SEC") or 30))
        while not self._stop.is_set():
            engine = None
            try:
                from quant.engine.combined_runner import CombinedVnpyEngine

                init_wait = float(os.getenv("VNPY_INIT_WAIT_SEC") or os.getenv("ORB_VNPY_INIT_WAIT_SEC") or 60)
                engine = CombinedVnpyEngine()
                status = engine.bootstrap(init_wait_sec=init_wait)
                self._last_status = {**status, "restart_count": self._restart_count}
                if not status.get("ok"):
                    logger.error("[combined-vnpy] bootstrap failed: %s", status)
                    return
                if status.get("skipped"):
                    logger.info("[combined-vnpy] skipped: %s", status.get("reason"))
                    return
                engine.run_until(self._stop)
            except ImportError as exc:
                logger.error("[combined-vnpy] import failed: %s", exc)
                return
            except Exception as exc:
                logger.exception("[combined-vnpy] crashed: %s", exc)
                self._last_status = {"ok": False, "reason": "crash", "error": str(exc)}
            finally:
                if engine is not None:
                    try:
                        engine.shutdown()
                    except Exception as exc:
                        logger.warning("[combined-vnpy] shutdown: %s", exc)
            if self._stop.is_set():
                break
            self._restart_count += 1
            self._stop.wait(restart_delay)


combined_vnpy_supervisor = CombinedVnpySupervisor()

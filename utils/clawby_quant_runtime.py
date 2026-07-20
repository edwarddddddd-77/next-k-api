"""Manage clawby-quant sidecar process (uvicorn) for Next K embed."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_proc: subprocess.Popen | None = None
_lock = threading.Lock()


def vendor_root() -> Path:
    raw = (os.getenv("CLAWBY_QUANT_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "vendor" / "clawby_quant").resolve()


def clawby_base_url() -> str:
    return (os.getenv("CLAWBY_QUANT_URL") or "http://127.0.0.1:8899").rstrip("/")


def clawby_port() -> int:
    return int(os.getenv("CLAWBY_QUANT_PORT", "8899") or "8899")


def embed_enabled() -> bool:
    raw = (os.getenv("NEXT_K_CLAWBY_EMBED", "1") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def status() -> dict[str, Any]:
    alive = _proc is not None and _proc.poll() is None
    return {
        "ok": True,
        "embed_enabled": embed_enabled(),
        "running": alive,
        "pid": _proc.pid if alive and _proc else None,
        "url": clawby_base_url(),
        "vendor_root": str(vendor_root()),
        "port": clawby_port(),
    }


def start_sidecar() -> dict[str, Any]:
    """Start uvicorn backend.main:app in vendor root (no-op if already running)."""
    global _proc
    if not embed_enabled():
        return {**status(), "started": False, "reason": "embed_disabled"}

    root = vendor_root()
    if not (root / "backend" / "main.py").is_file():
        logger.error("clawby-quant vendor missing at %s", root)
        return {**status(), "started": False, "reason": "vendor_missing", "root": str(root)}

    with _lock:
        if _proc is not None and _proc.poll() is None:
            return {**status(), "started": False, "reason": "already_running"}

        port = clawby_port()
        env = os.environ.copy()
        env.setdefault("CLAWBY_API_PREFIX", "/api/clawby-quant")
        # Keep DB under Next K data dir when available
        data_dir = (os.getenv("DATA_DIR") or "").strip()
        if data_dir:
            env.setdefault("QB_DB_PATH", str(Path(data_dir) / "clawby_quantbot.db"))

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "info",
        ]
        logger.info("Starting clawby-quant sidecar: %s (cwd=%s)", " ".join(cmd), root)
        try:
            _proc = subprocess.Popen(
                cmd,
                cwd=str(root),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.exception("clawby-quant start failed: %s", e)
            return {**status(), "started": False, "reason": str(e)}

        # brief wait for bind
        time.sleep(1.2)
        alive = _proc.poll() is None
        if not alive:
            return {**status(), "started": False, "reason": "exited_immediately"}
        return {**status(), "started": True}


def stop_sidecar() -> None:
    global _proc
    with _lock:
        if _proc is None:
            return
        if _proc.poll() is None:
            logger.info("Stopping clawby-quant sidecar pid=%s", _proc.pid)
            try:
                _proc.terminate()
                _proc.wait(timeout=8)
            except Exception:
                try:
                    _proc.kill()
                except Exception:
                    pass
        _proc = None

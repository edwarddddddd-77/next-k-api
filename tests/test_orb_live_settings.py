#!/usr/bin/env python3
"""ORB 实盘通知由 ORB_LIVE_ENABLED 环境变量控制。"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_API_ROOT = Path(__file__).resolve().parent.parent
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

from fastapi.testclient import TestClient


class OrbLiveNotifyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._env = patch.dict(
            os.environ,
            {
                "DATA_DIR": self._tmpdir.name,
                "ORB_LIVE_ENABLED": "0",
                "PROTOCOL_API_URL": "http://127.0.0.1:8001",
            },
            clear=False,
        )
        self._env.start()
        import accumulation_radar as ar

        self._old_db = ar.DB_PATH
        ar.DB_PATH = Path(self._tmpdir.name) / "test.db"
        import importlib
        import routers.orb as orb_router

        importlib.reload(orb_router)
        import main

        importlib.reload(main)
        self.client = TestClient(main.app)

    def tearDown(self) -> None:
        import accumulation_radar as ar

        ar.DB_PATH = self._old_db
        self._env.stop()
        self._tmpdir.cleanup()

    def test_env_off(self) -> None:
        r = self.client.get("/api/orb/live")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertFalse(body["orb_live_enabled"])
        self.assertFalse(body["live_notify_binance"])

    def test_env_on_reflected_after_reload(self) -> None:
        os.environ["ORB_LIVE_ENABLED"] = "1"
        import importlib
        import orb.core.live_settings as ls

        importlib.reload(ls)
        r = self.client.get("/api/orb/live")
        body = r.json()
        self.assertTrue(body["orb_live_enabled"])
        self.assertTrue(body["live_notify_binance"])


if __name__ == "__main__":
    unittest.main()

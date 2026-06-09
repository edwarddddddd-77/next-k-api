"""调度开关默认行为。"""

from __future__ import annotations

import os
import unittest
from unittest import mock


class TestEmbedSchedulerDefault(unittest.TestCase):
    def test_embed_scheduler_default_on_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEXT_K_EMBED_SCHEDULER", None)
            import importlib

            import scheduler_config as sc

            importlib.reload(sc)
            self.assertTrue(sc.embed_scheduler_enabled())

    def test_embed_scheduler_off_when_zero(self):
        with mock.patch.dict(os.environ, {"NEXT_K_EMBED_SCHEDULER": "0"}):
            import importlib

            import scheduler_config as sc

            importlib.reload(sc)
            self.assertFalse(sc.embed_scheduler_enabled())


class TestOrbScanCron(unittest.TestCase):
    def test_cron_kwargs_5m_utc(self):
        import importlib

        import scheduler_config as sc

        importlib.reload(sc)
        kw = sc.orb_scan_cron_kwargs(5, second=5)
        self.assertIsNotNone(kw)
        self.assertEqual(kw["minute"], "*/5")
        self.assertEqual(kw["second"], 5)
        self.assertEqual(kw["timezone"], sc.ORB_SCAN_CRON_TZ)

    def test_cron_kwargs_1m(self):
        import scheduler_config as sc

        kw = sc.orb_scan_cron_kwargs(1, second=3)
        self.assertEqual(kw, {"minute": "*", "second": 3, "timezone": sc.ORB_SCAN_CRON_TZ})

    def test_cron_kwargs_invalid_interval(self):
        import scheduler_config as sc

        self.assertIsNone(sc.orb_scan_cron_kwargs(7))


if __name__ == "__main__":
    unittest.main()

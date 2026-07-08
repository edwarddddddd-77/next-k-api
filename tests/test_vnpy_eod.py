"""EOD 强平时刻测试。"""

from __future__ import annotations

import unittest

import pandas as pd

from orb.core.config import OrbConfig
from orb.vnpy.eod import effective_eod_hm, should_eod_flat_bar


class TestEod(unittest.TestCase):
    def test_effective_eod_hm_caps_at_session_close(self):
        cfg = OrbConfig.from_env()
        bar_ms = int(pd.Timestamp("2026-02-03 12:00", tz=cfg.session_tz).timestamp() * 1000)
        eh, em = effective_eod_hm(
            bar_ms=bar_ms,
            session_tz=cfg.session_tz,
            session_open_time=cfg.session_open_time,
            session_close_time=cfg.session_close_time,
            market=cfg.market,
            exit_hour=16,
            exit_minute=0,
        )
        self.assertLessEqual(eh * 60 + em, 16 * 60)

    def test_should_eod_flat_bar_after_exit(self):
        cfg = OrbConfig.from_env()
        bar_ms = int(pd.Timestamp("2026-02-03 15:55", tz=cfg.session_tz).timestamp() * 1000)
        ts = pd.Timestamp(bar_ms, unit="ms", tz=cfg.session_tz)
        self.assertTrue(
            should_eod_flat_bar(
                bar_ms=bar_ms,
                ts=ts,
                cfg=cfg,
                exit_hour=15,
                exit_minute=50,
            )
        )


if __name__ == "__main__":
    unittest.main()

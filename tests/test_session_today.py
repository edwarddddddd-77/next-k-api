"""当日会话提示 API 测试。"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import pandas as pd

from orb.core.config import OrbConfig
from orb.core.macro_calendar import clear_macro_calendar_cache
from orb.core.session_today import build_session_today


class TestSessionToday(unittest.TestCase):
    def setUp(self):
        clear_macro_calendar_cache()
        os.environ["ORB_MACRO_CALENDAR_FETCH"] = "0"

    def tearDown(self):
        clear_macro_calendar_cache()

    def _cfg(self, **kw) -> OrbConfig:
        base = dict(
            market="us_equity",
            session_tz="America/New_York",
            session_open_time="09:30",
            session_close_time="16:00",
            macro_filter=False,
        )
        base.update(kw)
        return OrbConfig(**base)

    def test_weekend_alert(self):
        # 2026-06-06 Saturday
        ms = int(pd.Timestamp("2026-06-06 12:00", tz="America/New_York").value // 1_000_000)
        out = build_session_today(self._cfg(), now_ms=ms)
        self.assertFalse(out["is_trading_day"])
        self.assertEqual(out["non_trading_reason"], "weekend")
        kinds = [a["kind"] for a in out["alerts"]]
        self.assertIn("weekend", kinds)

    def test_holiday_alert(self):
        ms = int(pd.Timestamp("2026-01-01 12:00", tz="America/New_York").value // 1_000_000)
        out = build_session_today(self._cfg(), now_ms=ms)
        self.assertFalse(out["is_trading_day"])
        self.assertEqual(out["non_trading_reason"], "exchange_holiday")

    def test_cpi_day_info_when_filter_off(self):
        ms = int(pd.Timestamp("2026-05-12 10:00", tz="America/New_York").value // 1_000_000)
        out = build_session_today(self._cfg(macro_filter=False), now_ms=ms)
        self.assertTrue(out["is_trading_day"])
        self.assertIn("cpi", out["macro_events"])
        cpi = next(a for a in out["alerts"] if a["kind"] == "cpi")
        self.assertEqual(cpi["severity"], "info")
        self.assertFalse(out["skip_new_entries"])

    def test_cpi_day_warn_when_filter_on(self):
        ms = int(pd.Timestamp("2026-05-12 10:00", tz="America/New_York").value // 1_000_000)
        out = build_session_today(self._cfg(macro_filter=True), now_ms=ms)
        cpi = next(a for a in out["alerts"] if a["kind"] == "cpi")
        self.assertEqual(cpi["severity"], "warn")
        self.assertTrue(out["skip_new_entries"])

    def test_fomc_alert(self):
        ms = int(pd.Timestamp("2026-04-29 10:00", tz="America/New_York").value // 1_000_000)
        out = build_session_today(self._cfg(macro_filter=True), now_ms=ms)
        self.assertIn("fomc", out["macro_events"])
        self.assertTrue(any(a["kind"] == "fomc" for a in out["alerts"]))

    def test_early_close_alert(self):
        ms = int(pd.Timestamp("2026-07-03 10:00", tz="America/New_York").value // 1_000_000)
        out = build_session_today(self._cfg(), now_ms=ms)
        self.assertTrue(out["early_close"])
        self.assertEqual(out["session_close_time"], "13:00")
        self.assertTrue(any(a["kind"] == "early_close" for a in out["alerts"]))

    def test_normal_trading_day_no_alerts(self):
        ms = int(pd.Timestamp("2026-06-04 10:00", tz="America/New_York").value // 1_000_000)
        out = build_session_today(self._cfg(), now_ms=ms)
        self.assertTrue(out["is_trading_day"])
        self.assertEqual(out["macro_events"], [])
        self.assertFalse(out["has_alerts"])


if __name__ == "__main__":
    unittest.main()

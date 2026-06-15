"""宏观日历在线刷新测试。"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from orb.core.macro_calendar import (
    clear_macro_calendar_cache,
    fetch_cpi_skip_dates,
    fetch_fomc_skip_dates,
    get_macro_skip_dates,
    is_macro_skip_day,
    refresh_macro_calendar,
)

_FOMC_SNIPPET = """
<a>2026 FOMC Meetings</a>
<div class="fomc-meeting">
  <div class="fomc-meeting__month"><strong>January</strong></div>
  <div class="fomc-meeting__date">27-28</div>
</div>
<div class="fomc-meeting">
  <div class="fomc-meeting__month"><strong>March</strong></div>
  <div class="fomc-meeting__date">17-18*</div>
</div>
<a>2025 FOMC Meetings</a>
<div class="fomc-meeting">
  <div class="fomc-meeting__month"><strong>December</strong></div>
  <div class="fomc-meeting__date">9-10</div>
</div>
"""

_CPI_SNIPPET = """
Schedule of Releases for the Consumer Price Index
| April 2026 | May 12, 2026 | 08:30 AM |
| May 2026 | Jun. 10, 2026 | 08:30 AM |
"""


class TestMacroCalendar(unittest.TestCase):
    def setUp(self):
        clear_macro_calendar_cache()
        self._old_fetch = os.environ.get("ORB_MACRO_CALENDAR_FETCH")
        os.environ["ORB_MACRO_CALENDAR_FETCH"] = "0"

    def tearDown(self):
        clear_macro_calendar_cache()
        if self._old_fetch is None:
            os.environ.pop("ORB_MACRO_CALENDAR_FETCH", None)
        else:
            os.environ["ORB_MACRO_CALENDAR_FETCH"] = self._old_fetch

    def test_builtin_2026_cpi(self):
        self.assertTrue(is_macro_skip_day("2026-05-12"))
        self.assertFalse(is_macro_skip_day("2026-05-13"))

    @patch("orb.core.macro_calendar.requests.get")
    def test_fetch_fomc_parses_announcement_days(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = _FOMC_SNIPPET
        mock_get.return_value.raise_for_status = lambda: None
        dates, ok = fetch_fomc_skip_dates()
        self.assertTrue(ok)
        self.assertIn("2026-01-28", dates)
        self.assertIn("2026-03-18", dates)

    @patch("orb.core.macro_calendar.requests.get")
    def test_fetch_cpi_parses_release_dates(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = _CPI_SNIPPET
        dates, ok = fetch_cpi_skip_dates()
        self.assertTrue(ok)
        self.assertIn("2026-05-12", dates)
        self.assertIn("2026-06-10", dates)

    @patch("orb.core.macro_calendar.fetch_cpi_skip_dates", return_value=({"2099-01-01"}, True))
    @patch("orb.core.macro_calendar.fetch_fomc_skip_dates", return_value=({"2099-02-02"}, True))
    def test_refresh_merges_live_and_builtin(self, _fomc, _cpi):
        os.environ["ORB_MACRO_CALENDAR_FETCH"] = "1"
        cache = refresh_macro_calendar(force=True)
        self.assertIn("2099-01-01", cache.dates)
        self.assertIn("2099-02-02", cache.dates)
        self.assertIn("2026-05-12", cache.dates)
        self.assertTrue(cache.fomc_ok)
        self.assertTrue(cache.cpi_ok)

    @patch("orb.core.macro_calendar.fetch_cpi_skip_dates", return_value=(set(), False))
    @patch("orb.core.macro_calendar.fetch_fomc_skip_dates", return_value=(set(), False))
    def test_refresh_fallback_to_builtin_on_fetch_fail(self, _fomc, _cpi):
        os.environ["ORB_MACRO_CALENDAR_FETCH"] = "1"
        dates = get_macro_skip_dates(force_refresh=True)
        self.assertIn("2026-05-12", dates)

    def test_macro_skip_logs_once(self):
        import logging

        with self.assertLogs("orb.core.macro_calendar", level="INFO") as cm:
            self.assertTrue(is_macro_skip_day("2026-05-12"))
            self.assertTrue(is_macro_skip_day("2026-05-12"))
        skip_logs = [m for m in cm.output if "macro skip day" in m]
        self.assertEqual(len(skip_logs), 1)

    def test_macro_calendar_status_fields(self):
        os.environ["ORB_MACRO_CALENDAR_FETCH"] = "0"
        from orb.core.macro_calendar import macro_calendar_status

        st = macro_calendar_status()
        self.assertIn("total_dates", st)
        self.assertIn("builtin_dates", st)
        self.assertIn("cache_age_seconds", st)
        self.assertGreater(st["total_dates"], 0)
        self.assertFalse(st["fetch_enabled"])
        self.assertFalse(st["live_fetch_ok"])


if __name__ == "__main__":
    unittest.main()

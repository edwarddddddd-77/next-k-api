"""ORB live_exec ingest result helpers."""

from __future__ import annotations

import unittest

from orb.core.live_exec import live_ingest_succeeded


class TestLiveIngestSucceeded(unittest.TestCase):
    def test_none_or_skipped_is_success(self) -> None:
        self.assertTrue(live_ingest_succeeded(None))
        self.assertTrue(live_ingest_succeeded({"skipped": True, "reason": "live_disabled"}))

    def test_traded_is_success(self) -> None:
        self.assertTrue(live_ingest_succeeded({"traded": 1, "errors": 0, "details": []}))

    def test_error_or_failed_detail_is_failure(self) -> None:
        self.assertFalse(live_ingest_succeeded({"error": "timeout"}))
        self.assertFalse(live_ingest_succeeded({"traded": 0, "errors": 1, "details": []}))
        self.assertFalse(
            live_ingest_succeeded(
                {
                    "traded": 0,
                    "errors": 1,
                    "details": [{"action": "error", "error": "execute_trade returned False"}],
                }
            )
        )


if __name__ == "__main__":
    unittest.main()

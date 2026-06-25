"""ORB live_exec ingest result helpers."""

from __future__ import annotations

import unittest

from orb.core.live_exec import (
    _close_signal_id,
    build_close_payload,
    live_ingest_succeeded,
    live_open_is_pending,
)


class TestLiveIngestSucceeded(unittest.TestCase):
    def test_none_or_skipped_is_success(self) -> None:
        self.assertTrue(live_ingest_succeeded(None))
        self.assertTrue(live_ingest_succeeded({"skipped": True, "reason": "live_disabled"}))

    def test_traded_is_success(self) -> None:
        self.assertTrue(live_ingest_succeeded({"traded": 1, "errors": 0, "details": []}))

    def test_submitted_is_success(self) -> None:
        payload = {"traded": 0, "errors": 0, "details": [{"action": "submitted"}]}
        self.assertTrue(live_ingest_succeeded(payload))
        self.assertTrue(live_open_is_pending(payload))

    def test_duplicate_is_failure(self) -> None:
        self.assertFalse(
            live_ingest_succeeded(
                {"traded": 0, "errors": 0, "details": [{"action": "duplicate"}]}
            )
        )

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

    def test_close_signal_id_is_deterministic(self) -> None:
        a = _close_signal_id("COINUSDT", signal_id=42, tag="loss")
        b = _close_signal_id("COINUSDT", signal_id=42, tag="loss")
        self.assertEqual(a, b)
        self.assertEqual(
            build_close_payload("COIN", "LONG", tag="loss", signal_id=42)["api_signal_id"],
            "orb:close:COINUSDT:42:loss",
        )


if __name__ == "__main__":
    unittest.main()

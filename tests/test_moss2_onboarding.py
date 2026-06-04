"""Moss2 onboarding 建议（不依赖外部寻优目录）。"""

from __future__ import annotations

import unittest


class TestMoss2Onboarding(unittest.TestCase):
    def test_seed_bases_25(self):
        from moss2.config import MOSS2_SEED_BASES

        self.assertEqual(len(MOSS2_SEED_BASES), 25)
        self.assertIn("BTC", MOSS2_SEED_BASES)
        self.assertIn("HYPE", MOSS2_SEED_BASES)

    def test_regime_hint_trend(self):
        from moss2.onboarding import _regime_template_hint

        self.assertEqual(
            _regime_template_hint({"BULL": 0.6, "SIDEWAYS": 0.3, "BEAR": 0.1}),
            "trend",
        )

    def test_regime_hint_mean_revert(self):
        from moss2.onboarding import _regime_template_hint

        self.assertEqual(
            _regime_template_hint({"SIDEWAYS": 0.55, "BULL": 0.25, "BEAR": 0.2}),
            "mean_revert",
        )

    def test_suggest_btc_structure(self):
        from moss2.dataset import resolve_csv_path
        from moss2.onboarding import suggest_profile

        if not resolve_csv_path("BTCUSDT", "en"):
            self.skipTest("BTC EN CSV missing")
        out = suggest_profile("BTC", backtest_bars=800)
        self.assertTrue(out.get("ok"))
        self.assertEqual(out["symbol"], "BTCUSDT")
        self.assertIn(out["recommended_template"], ("balanced", "momentum", "trend", "mean_revert"))
        self.assertIn("regime_recent", out)


if __name__ == "__main__":
    unittest.main()

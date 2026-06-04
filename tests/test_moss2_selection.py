"""Moss2 选优闸门与淘汰逻辑。"""

from __future__ import annotations

import unittest
from unittest.mock import patch


class TestMoss2Selection(unittest.TestCase):
    def test_passes_backtest_gates(self):
        from moss2.selection import passes_backtest_gates

        summ = {"total_trades": 10, "sharpe": 0.5, "max_drawdown": 0.2}
        disc = {"ev": {"ev_per_trade_pct": 0.1}}
        self.assertTrue(passes_backtest_gates(summ, disc))
        self.assertFalse(
            passes_backtest_gates(
                {"total_trades": 3, "sharpe": 0.5, "max_drawdown": 0.2},
                disc,
            )
        )
        self.assertFalse(
            passes_backtest_gates(
                {"total_trades": 10, "sharpe": 0.5, "max_drawdown": 0.5},
                disc,
            )
        )

    def test_composite_score_penalizes_mdd(self):
        from moss2.selection import composite_score

        low_mdd = composite_score(
            {"sharpe": 1.0, "total_return": 0.1, "max_drawdown": 0.1},
            {"ev": {"ev_per_trade_pct": 0.1}},
        )
        high_mdd = composite_score(
            {"sharpe": 1.0, "total_return": 0.1, "max_drawdown": 0.5},
            {"ev": {"ev_per_trade_pct": 0.1}},
        )
        self.assertGreater(low_mdd, high_mdd)

    def test_compete_keeps_coarse_when_refined_not_better(self):
        from moss2 import selection as sel

        coarse = {
            "template": "balanced",
            "score": 2.0,
            "passes_gates": True,
            "params": {},
            "summary": {},
            "discipline": {"ev": {"ev_per_trade_pct": 0.1}},
        }
        refined = {**coarse, "score": 1.0}
        with (
            patch.object(sel, "_backtest_row", return_value=coarse),
            patch.object(sel, "optimize_template_params", return_value=refined),
            patch.object(sel, "list_templates", return_value=["balanced"]),
            patch.object(sel.cfg, "MOSS2_SELECTION_TACTICAL_NARROW", True),
        ):
            out = sel.compete_templates("BTCUSDT", optimize_tactical=True)
        self.assertEqual(out["best"]["score"], 2.0)
        self.assertFalse(out["tactical_refined"])


if __name__ == "__main__":
    unittest.main()

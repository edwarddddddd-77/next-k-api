"""ORB V2 paper 扫描参数与无 ML 开单路径。"""

from __future__ import annotations

import os
import unittest

from orb.core.config import OrbConfig
from orb.ml.gate import LiveGateConfig, LiveGateDayState, evaluate_open_decision_without_ml, gate_with_ml_bypass
from orb.v2.config import OrbV2Config
from orb.v2.paper import _scan_params_v2


class TestOrbV2Paper(unittest.TestCase):
    def test_scan_params_v2_without_model(self):
        cfg = OrbConfig.from_env()
        gate = gate_with_ml_bypass(LiveGateConfig())
        params = _scan_params_v2(
            cfg,
            gate=gate,
            model=None,
            ml_enabled=False,
            shadow=False,
            use_robots=True,
            robot_bound=True,
            robot_count=8,
            robot_equity=1000.0,
        )
        self.assertEqual(params["ml_ranker"], "disabled")
        self.assertFalse(params["ml_enabled"])
        self.assertEqual(params["sizing"], "robot_bound")

    def test_gate_ml_disabled_when_env_zero(self):
        saved = os.environ.get("ORB_V2_GATE_ML")
        try:
            os.environ["ORB_V2_GATE_ML"] = "0"
            v2 = OrbV2Config.from_env()
            self.assertFalse(v2.gate_ml_enabled())
            gate = v2.load_gate()
            self.assertEqual(gate.min_p_true, 0.0)
            self.assertEqual(gate.min_breakout_score, 0.0)
        finally:
            if saved is None:
                os.environ.pop("ORB_V2_GATE_ML", None)
            else:
                os.environ["ORB_V2_GATE_ML"] = saved

    def test_gate_ml_from_live_gate_when_env_unset(self):
        saved = os.environ.pop("ORB_V2_GATE_ML", None)
        try:
            v2 = OrbV2Config.from_env()
            self.assertTrue(v2.gate_ml_enabled())
            gate = v2.load_gate()
            self.assertEqual(gate.min_p_true, 0.35)
            self.assertEqual(gate.min_breakout_score, 45.0)
        finally:
            if saved is None:
                os.environ.pop("ORB_V2_GATE_ML", None)
            else:
                os.environ["ORB_V2_GATE_ML"] = saved

    def test_evaluate_open_decision_without_ml(self):
        gate = gate_with_ml_bypass(LiveGateConfig(max_opens_per_day=8, robot_reuse_after_exit=True))
        state = LiveGateDayState()
        feat = {"minutes_after_or": 30.0}
        row = evaluate_open_decision_without_ml(
            symbol="COINUSDT",
            feat=feat,
            sync=0,
            state=state,
            gate=gate,
        )
        self.assertTrue(row["opened"])
        self.assertEqual(row["reason"], "open_ok")
        self.assertEqual(state.opens, 1)


if __name__ == "__main__":
    unittest.main()

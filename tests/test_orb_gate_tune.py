"""Gate 重放与 pick 逻辑测试。"""

from __future__ import annotations

from orb.ml.gate import LiveGateConfig
from orb.ml.gate_replay import replay_day, replay_sessions, sweep_min_p_grid
from orb.ml.model.auto_config import MlAutoConfig
from orb.ml.model.gate_tune import _pick_candidate


def _day(pnl: float, p: float) -> dict:
    return {
        "timeline": [
            {
                "p_true": p,
                "sync_same_side": 0,
                "minutes_after_or": 30.0,
                "pnl_usdt": pnl,
                "true_breakout": pnl > 0,
            }
        ]
    }


def test_replay_respects_min_p():
    gate = LiveGateConfig(min_p_true=0.35, max_opens_per_day=8, day_abort_enabled=False)
    day = _day(100.0, 0.40)
    hi = replay_day(day["timeline"], min_p=0.38, max_opens=8, gate=gate)
    lo = replay_day(day["timeline"], min_p=0.42, max_opens=8, gate=gate)
    assert hi["opens"] == 1
    assert lo["opens"] == 0


def test_sweep_orders_by_pnl():
    gate = LiveGateConfig(max_opens_per_day=8, day_abort_enabled=False)
    days = [_day(50.0, 0.40), _day(80.0, 0.36)]
    rows = sweep_min_p_grid(days, gate=gate, grid=[0.35, 0.40])
    assert rows[0]["total_pnl"] >= rows[-1]["total_pnl"]


def test_pick_candidate_respects_delta():
    cfg = MlAutoConfig.from_env()
    gate = LiveGateConfig(max_opens_per_day=8, day_abort_enabled=False)
    days = [_day(100.0, 0.40)]
    current = replay_sessions(days, min_p=0.35, gate=gate)
    rows = sweep_min_p_grid(days, gate=gate, grid=[0.30, 0.35, 0.39])
    best, reasons = _pick_candidate(
        rows=rows,
        current_min_p=0.35,
        current_replay=current,
        cfg=cfg,
        tighten_only=False,
    )
    assert best is None or abs(float(best["min_p"]) - 0.35) <= cfg.gate_min_p_delta + 1e-9
    if best is None:
        assert reasons

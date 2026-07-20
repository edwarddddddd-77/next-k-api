"""Fill/fee/funding arithmetic + pessimistic bar-exit rules."""
import asyncio

from backend.backtest import data
from backend.backtest.broker import FEE_RATE, SimBroker
from backend.backtest.sim import Dataset, Sim
from backend.strategies import Signal

T0 = data.T0


def bars_from(specs, start=T0):
    return [dict(ts=start + i * 60, open=o, high=h, low=lo, close=c, volume=10.0)
            for i, (o, h, lo, c) in enumerate(specs)]


def make(specs, funding=None):
    ds = Dataset.synthetic({"TESTUSDT": bars_from(specs)}, funding=funding)
    sim = Sim(ds, T0)
    sim.ws.price["TESTUSDT"] = specs[0][0]
    return sim, SimBroker(sim, notional=1000.0)


def sig(side="long", stop=0.0, tp=0.0, trail=0.0, hold=86400):
    return Signal("S01_FUNDING_FADE", "TESTUSDT", side, "test",
                  stop_price=stop, take_profit=tp, trail_dist=trail,
                  max_hold_sec=hold)


def test_entry_fill_slippage_and_fee():
    sim, br = make([(100.0, 101, 99, 100)] * 5)
    pid = asyncio.run(br.on_signal(sig("long")))
    pos = sim.db.positions[pid]
    assert abs(pos["entry_price"] - 100.0 * 1.00025) < 1e-9   # minor-coin 2.5bp
    assert abs(pos["qty"] - 1000.0 / pos["entry_price"]) < 1e-9
    assert abs(pos["open_fee"] - pos["entry_price"] * pos["qty"] * FEE_RATE) < 1e-9


def test_pessimistic_stop_beats_tp_in_same_bar():
    sim, br = make([(100.0, 101, 99, 100), (100.0, 103.0, 97.0, 102.0)])
    asyncio.run(br.on_signal(sig("long", stop=98.0, tp=102.5)))
    asyncio.run(br.manage_tick(T0 + 60, None, {}))
    assert len(br.trades) == 1
    tr = br.trades[0]
    assert tr["reason_exit"] == "止损触发"
    assert abs(tr["exit_price"] - 98.0 * (1 - 0.00025)) < 1e-9


def test_trailing_checks_old_extreme_before_update():
    sim, br = make([(100.0, 100, 100, 100),
                    (100.0, 103.0, 99.0, 102.0),     # rally: extreme -> 103
                    (102.0, 102.5, 100.9, 101.0)])   # 103-100.9 >= 2 -> exit
    asyncio.run(br.on_signal(sig("long", trail=2.0)))
    asyncio.run(br.manage_tick(T0 + 60, None, {}))
    pos = next(iter(sim.db.positions.values()))
    assert pos["status"] == "open" and pos["extreme"] == 103.0
    asyncio.run(br.manage_tick(T0 + 120, None, {}))
    assert br.trades and br.trades[0]["reason_exit"] == "移动止损触发"
    assert abs(br.trades[0]["exit_price"] - (103.0 - 2.0) * (1 - 0.00025)) < 1e-9


def test_max_hold_exit_at_bar_open():
    sim, br = make([(100.0, 101, 99, 100)] * 10)
    asyncio.run(br.on_signal(sig("long", hold=120)))
    asyncio.run(br.manage_tick(T0 + 60, None, {}))
    assert not br.trades
    asyncio.run(br.manage_tick(T0 + 120, None, {}))
    assert br.trades and br.trades[0]["reason_exit"] == "达到最长持仓时限"


def test_funding_long_pays_positive_rate():
    settle = T0 + 8 * 3600
    sim, br = make([(100.0, 101, 99, 100)] * (8 * 60 + 5),
                   funding={"TESTUSDT": [(settle, 0.0001)]})
    asyncio.run(br.on_signal(sig("long")))
    br.settle_funding(settle - 60, settle)
    pos = next(iter(sim.db.positions.values()))
    assert abs(pos["funding_cost"] - 0.0001 * pos["qty"] * 100.0) < 1e-6
    sim.clock.set(settle)
    br._close(pos, 100.0, "test", settle)
    tr = br.trades[0]
    assert abs(tr["funding"] - pos["funding_cost"]) < 1e-9
    assert tr["pnl"] < tr["gross"] - tr["funding"]  # fees also deducted


def test_short_receives_positive_rate():
    settle = T0 + 8 * 3600
    sim, br = make([(100.0, 101, 99, 100)] * (8 * 60 + 5),
                   funding={"TESTUSDT": [(settle, 0.0001)]})
    asyncio.run(br.on_signal(sig("short")))
    br.settle_funding(settle - 60, settle)
    pos = next(iter(sim.db.positions.values()))
    assert pos["funding_cost"] < 0          # negative cost == received

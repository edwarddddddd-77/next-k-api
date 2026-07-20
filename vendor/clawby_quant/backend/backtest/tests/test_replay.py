"""End-to-end synthetic replays through runner.run_backtest:
S11 catches a fabricated crash and takes profit on the bounce;
S01 shorts extreme funding and exits when funding normalizes.
"""
import asyncio

from backend.backtest import data
from backend.backtest.runner import run_backtest
from backend.backtest.sim import Dataset

T0 = data.T0


def flat(n, price=100.0, start=T0, wick=1.0):
    return [dict(ts=start + i * 60, open=price, high=price + wick,
                 low=price - wick, close=price, volume=10.0) for i in range(n)]


def test_s11_catches_crash_and_takes_profit():
    t_evt = T0 + 3600
    bars = flat(180, wick=0.2)
    i_evt = (t_evt - T0) // 60
    bars[i_evt].update(high=100.0, low=98.55, close=98.6)
    bars[i_evt + 1].update(open=98.6, high=99.2, low=98.55, close=99.1)

    lo = t_evt - 600
    ticks = [(lo + s, 100.0) for s in range(600)]
    for s in range(31):                                  # 30s slide -1.45%
        ticks.append((t_evt + s, 100.0 - 1.45 / 30 * s * 100 / 100))
    for s in range(31, 37):                              # stall (deceleration)
        ticks.append((t_evt + s, 98.55))
    for s in range(37, 121):                             # bounce to 99.6
        ticks.append((t_evt + s, 98.55 + (99.6 - 98.55) * (s - 36) / 84))

    ds = Dataset.synthetic({"BTCUSDT": bars},
                           windows={"BTCUSDT": [[lo, t_evt + 660]]},
                           ticks={("BTCUSDT", lo): ticks})
    res = asyncio.run(run_backtest("S11_CRASH_SCALP", {}, ds))
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr["side"] == "long"
    assert tr["reason_exit"] == "止盈触发"
    assert tr["pnl"] > 0
    assert tr["hold_sec"] < 300
    assert T0 + 3600 <= tr["entry_ts"] <= T0 + 3640


def test_s11_no_trade_on_quiet_data():
    ds = Dataset.synthetic({"BTCUSDT": flat(180, wick=0.2)})
    res = asyncio.run(run_backtest("S11_CRASH_SCALP", {}, ds))
    assert res.trades == []


def test_s02_catches_liq_cascade_and_takes_profit():
    # flat until T0+16h (ATR ready after 15 complete 1h bars), then a 30m
    # slide 100->94 with a 9x long-liquidation spike bucket visible at entry;
    # bounce through tp = 94 + 6*0.5 = 97 closes the trade
    evt = T0 + 16 * 3600
    bars = flat(20 * 60, wick=1.0)
    i_evt = (evt - T0) // 60
    for j in range(30):                       # slide, 0.2/min
        px = 100.0 - 6.0 * (j + 1) / 30
        b = bars[i_evt - 30 + j]
        b.update(open=px + 0.2, close=px, high=px + 0.4, low=px - 0.2)
    for j in range(40):                       # bounce 94 -> 97.6
        px = 94.0 + 3.6 * (j + 1) / 40
        b = bars[i_evt + j]
        b.update(open=px - 0.09, close=px, high=px + 0.1, low=px - 0.2)
    liq = [(T0 - 86400 + h * 3600, (100.0, 50.0))
           for h in range(24 + 15)]           # calm baseline
    liq.append((evt - 3600, (900.0, 50.0)))   # spike bucket, complete at evt
    ds = Dataset.synthetic({"BTCUSDT": bars},
                           series={("clawby_liq_agg", "BTCUSDT"): liq})
    res = asyncio.run(run_backtest("S02_LIQ_REBOUND", {}, ds))
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr["side"] == "long"
    assert tr["reason_exit"] == "止盈触发"
    assert tr["pnl"] > 0
    m = res.metrics()
    assert m["trades"] == 1 and m["fees_usd"] > 0


def test_metrics_shape():
    ds = Dataset.synthetic({"BTCUSDT": flat(120)})
    res = asyncio.run(run_backtest("S02_LIQ_REBOUND", {}, ds))
    m = res.metrics()
    for key in ("trades", "net_usd", "net_return_pct", "max_dd_pct", "win_rate",
                "profit_factor", "sharpe", "avg_hold_h", "fees_usd", "funding_usd"):
        assert key in m
    assert m["trades"] == 0

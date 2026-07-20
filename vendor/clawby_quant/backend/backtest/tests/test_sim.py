"""Anti-lookahead + structural-parity tests for the replay layer."""
import time as realtime

from backend.backtest import data
from backend.backtest.sim import Dataset, Sim, SimClock

T0 = data.T0


def flat_bars(n, price=100.0, start=T0):
    return [dict(ts=start + i * 60, open=price, high=price + 1,
                 low=price - 1, close=price, volume=10.0) for i in range(n)]


def make_sim(series=None, bars=None, **kw):
    ds = Dataset.synthetic({"BTCUSDT": bars or flat_bars(240)},
                           series=series, **kw)
    return Sim(ds, T0)


# -- clock -------------------------------------------------------------------

def test_clock_virtual_time_and_us_session():
    clk = SimClock(T0)
    assert clk.time() == T0
    # 2026-06-18 is a Thursday; 14:00 UTC is inside the US session window
    clk.set(T0 + 14 * 3600)
    g = clk.gmtime()
    assert (g.tm_year, g.tm_mon, g.tm_mday, g.tm_hour) == (2026, 6, 18, 14)
    assert clk.mktime((2026, 6, 18, 21, 0, 0, 0, 0, 0)) == T0 + 21 * 3600
    assert clk.timezone == 0


# -- factors: completed buckets only ----------------------------------------

def test_factor_bucket_not_visible_until_complete():
    series = {("lsr_global_1h", "BTCUSDT"): [(T0, 1.0), (T0 + 3600, 2.0)]}
    sim = make_sim(series=series)
    st = sim.store
    assert st.factor_at("lsr_global", "BTCUSDT", T0 + 3599) is None
    assert st.factor_at("lsr_global", "BTCUSDT", T0 + 3600)["latest"] == 1.0
    assert st.factor_at("lsr_global", "BTCUSDT", T0 + 7200)["latest"] == 2.0
    assert st.factor_at("lsr_global", "BTCUSDT", T0 + 7200)["series"] == [1.0, 2.0]


def test_liq_agg_mirrors_live_shape():
    pairs = [(T0 + i * 3600, (100.0, 50.0)) for i in range(10)]
    pairs.append((T0 + 10 * 3600, (900.0, 50.0)))       # spike bucket
    sim = make_sim(series={("clawby_liq_agg", "BTCUSDT"): pairs})
    f = sim.store.factor_at("liq_agg", "BTCUSDT", T0 + 11 * 3600)
    assert set(f) == {"long_1h", "short_1h", "long_mult", "short_mult"}
    assert f["long_1h"] == 900.0
    assert abs(f["long_mult"] - 9.0) < 1e-9      # 900 / avg(10x100)


def test_no_history_factors_return_none():
    sim = make_sim()
    for name in ("liq_orders", "ob_wall", "exch_chain_tx", "netflow_xsec",
                 "unlock", "fear_greed"):
        assert sim.store.factor_at(name, "BTCUSDT", T0 + 3600) is None


# -- klines: strict truncation ----------------------------------------------

def test_1m_klines_exclude_current_minute():
    sim = make_sim(bars=flat_bars(120))
    ks = sim.klines_at("BTCUSDT", "1m", 30)
    sim.clock.set(T0 + 300)
    ks = sim.klines_at("BTCUSDT", "1m", 30)
    assert ks[-1]["ts"] == T0 + 240          # bar [T0+240, T0+300) is complete
    sim.clock.set(T0 + 301)
    ks = sim.klines_at("BTCUSDT", "1m", 30)
    assert ks[-1]["ts"] == T0 + 240          # bar starting T0+300 still open


def test_hourly_partial_bar_only_uses_past_minutes():
    bars = flat_bars(180)
    bars[90]["high"] = 999.0                  # spike at T0+90m (future)
    sim = make_sim(bars=bars)
    sim.clock.set(T0 + 75 * 60)               # inside hour 2, before the spike
    ks = sim.klines_at("BTCUSDT", "1h", 10)
    partial = ks[-1]
    assert partial["ts"] == T0 + 3600
    assert partial["high"] < 999.0            # future spike not leaked
    assert partial["volume"] == 10.0 * 15     # minutes 60..74 completed
    sim.clock.set(T0 + 92 * 60)               # after the spike minute completes
    ks = sim.klines_at("BTCUSDT", "1h", 10)
    assert ks[-1]["high"] == 999.0


def test_full_hour_bars_are_complete_hours_only():
    sim = make_sim(bars=flat_bars(200))
    sim.clock.set(T0 + 3600)
    ks = sim.klines_at("BTCUSDT", "1h", 10)
    full = [k for k in ks if k["ts"] == T0]
    assert len(full) == 1 and full[0]["volume"] == 600.0   # 60 bars x 10


# -- ws second-level window ---------------------------------------------------

def test_ws_change_pct_and_span():
    ticks = [(T0 + i, 100.0 - i * 0.05) for i in range(61)]  # steady slide
    sim = make_sim()
    sim.ws.enter_window("BTCUSDT", T0, ticks)
    sim.clock.set(T0 + 60)
    sim.ws.price["BTCUSDT"] = ticks[-1][1]
    chg = sim.ws.change_pct("BTCUSDT", 30)
    expect = (97.0 - 98.5) / 98.5 * 100
    assert abs(chg - expect) < 0.02
    assert sim.ws.buffer_span("BTCUSDT") == 60
    sim.ws.exit_window("BTCUSDT")
    assert sim.ws.buffer_span("BTCUSDT") == 0
    assert sim.ws.change_pct("BTCUSDT", 30) is None


def test_ws_no_future_ticks_visible():
    ticks = [(T0 + i, 100.0) for i in range(30)] + [(T0 + 40, 50.0)]
    sim = make_sim()
    sim.ws.enter_window("BTCUSDT", T0, ticks)
    sim.clock.set(T0 + 29)
    sim.ws.price["BTCUSDT"] = 100.0
    assert sim.ws.price_ago("BTCUSDT", 5) == 100.0   # the 50.0 tick is future


# -- ctx structural parity ----------------------------------------------------

def test_live_ctx_runs_against_sim(event_loop=None):
    import asyncio

    import backend.strategies as S
    series = {("lsr_global_1h", "BTCUSDT"): [(T0 + i * 3600, 1.5) for i in range(5)],
              ("taker_5m", "BTCUSDT"): [(T0 + i * 300, 1.2) for i in range(60)]}
    sim = make_sim(series=series, bars=flat_bars(20 * 60))
    sim.clock.set(T0 + 18 * 3600)     # >=15 complete 1h bars -> ATR computable
    sim.ws.price["BTCUSDT"] = 100.0
    sim.install()
    try:
        ctx = S.Ctx(["BTCUSDT"])
        assert ctx.factor("lsr_global", "BTCUSDT")["latest"] == 1.5
        assert ctx.price("BTCUSDT") == 100.0
        assert ctx.has_position("BTCUSDT") is False
        ks = asyncio.run(ctx.klines("BTCUSDT", "1h", 10))
        assert set(ks[0]) == {"ts", "open", "high", "low", "close", "volume"}
        atr = asyncio.run(ctx.atr("BTCUSDT"))
        assert atr > 0
        ratio = asyncio.run(ctx.depth_ratio("BTCUSDT"))
        assert ratio == 1.0
    finally:
        sim.uninstall()


def test_install_uninstall_restores_modules():
    import backend.strategies as S
    orig_time, orig_db = S.time, S.db
    sim = make_sim()
    sim.install()
    assert S.time is sim.clock
    sim.uninstall()
    assert S.time is orig_time and S.db is orig_db
    assert S.time.time() - realtime.time() < 5

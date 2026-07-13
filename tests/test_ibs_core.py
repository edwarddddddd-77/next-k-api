"""IBS 核心信号测试。"""

from __future__ import annotations

import unittest

from quant.ibs.core import (
    SessionDailyBar,
    compute_ibs,
    ema_last,
    evaluate_signal,
    evaluate_signal_context,
    select_signal_context,
    stop_loss_hit,
)


class TestIbsCore(unittest.TestCase):
    def test_compute_ibs_mid(self):
        self.assertAlmostEqual(compute_ibs(110.0, 100.0, 105.0), 0.5)

    def test_compute_ibs_low(self):
        self.assertAlmostEqual(compute_ibs(110.0, 100.0, 100.0), 0.0)

    def test_conservative_buy_on_weak_close(self):
        bar = SessionDailyBar("2026-01-02", 0, 100.0, 110.0, 100.0, 101.0)
        self.assertEqual(
            evaluate_signal(
                prev_bar=bar,
                in_position=False,
                entry_threshold=0.20,
                exit_threshold=0.50,
                daily_closes=[100.0] * 200,
                sma_period=200,
            ),
            "BUY",
        )

    def test_conservative_sma_blocks_buy(self):
        bar = SessionDailyBar("2026-01-02", 0, 100.0, 110.0, 100.0, 101.0)
        closes = [120.0] * 200
        self.assertEqual(
            evaluate_signal(
                prev_bar=bar,
                in_position=False,
                entry_threshold=0.20,
                exit_threshold=0.50,
                daily_closes=closes,
                sma_period=200,
            ),
            "HOLD",
        )

    def test_conservative_sma_uses_prev_close_not_current(self):
        bar = SessionDailyBar("2026-01-02", 0, 100.0, 110.0, 100.0, 101.0)
        closes = [120.0] * 200
        self.assertEqual(
            evaluate_signal(
                prev_bar=bar,
                in_position=False,
                entry_threshold=0.20,
                exit_threshold=0.50,
                daily_closes=closes,
                sma_period=200,
                trend_price=None,
            ),
            "HOLD",
        )
        self.assertEqual(
            evaluate_signal(
                prev_bar=bar,
                in_position=False,
                entry_threshold=0.20,
                exit_threshold=0.50,
                daily_closes=closes,
                sma_period=200,
                trend_price=130.0,
            ),
            "BUY",
        )

    def test_select_signal_context_tv(self):
        daily = [
            SessionDailyBar("2026-01-01", 0, 100.0, 110.0, 100.0, 105.0),
            SessionDailyBar("2026-01-02", 0, 105.0, 115.0, 104.0, 112.0),
        ]
        ctx = select_signal_context(daily, trend_price_mode="current", current_price=113.0)
        self.assertIsNotNone(ctx)
        assert ctx is not None
        self.assertEqual(ctx.prev_bar.session_day, "2026-01-01")
        self.assertEqual(len(ctx.ibs_closes), 1)
        self.assertEqual(len(ctx.ma_closes), 2)
        self.assertAlmostEqual(float(ctx.trend_price), 113.0)

    def test_select_signal_context_tv_excludes_partial_last_for_ma(self):
        daily = [
            SessionDailyBar("2026-01-01", 0, 100.0, 110.0, 100.0, 105.0),
            SessionDailyBar("2026-01-02", 0, 105.0, 115.0, 104.0, 112.0),
        ]
        ctx = select_signal_context(
            daily,
            trend_price_mode="current",
            current_price=113.0,
            ma_excludes_last_bar=True,
        )
        assert ctx is not None
        self.assertEqual(len(ctx.ma_closes), 1)

    def test_select_signal_context_conservative(self):
        daily = [
            SessionDailyBar("2026-01-01", 0, 100.0, 110.0, 100.0, 105.0),
            SessionDailyBar("2026-01-02", 0, 105.0, 115.0, 104.0, 112.0),
        ]
        ctx = select_signal_context(daily, trend_price_mode="prev_close", current_price=113.0)
        assert ctx is not None
        self.assertEqual(len(ctx.ma_closes), 1)
        self.assertIsNone(ctx.trend_price)

    def test_evaluate_signal_context_tv(self):
        daily = [
            SessionDailyBar("2026-01-01", 0, 100.0, 110.0, 100.0, 101.0),
            SessionDailyBar("2026-01-02", 0, 101.0, 111.0, 100.0, 110.0),
        ]
        ctx = select_signal_context(daily, trend_price_mode="current", current_price=105.0)
        assert ctx is not None
        self.assertEqual(
            evaluate_signal_context(
                ctx,
                in_position=False,
                entry_threshold=0.20,
                exit_threshold=0.90,
                trend_ma_type="ema",
                trend_ma_period=252,
            ),
            "BUY",
        )

    def test_aggressive_exit_at_high_ibs(self):
        bar = SessionDailyBar("2026-01-03", 0, 100.0, 110.0, 100.0, 109.6)
        self.assertEqual(
            evaluate_signal(
                prev_bar=bar,
                in_position=True,
                entry_threshold=0.19,
                exit_threshold=0.95,
                daily_closes=[100.0],
                sma_period=0,
            ),
            "SELL",
        )

    def test_stop_loss_hit(self):
        self.assertTrue(stop_loss_hit(side=1, entry_price=100.0, close=94.0, stop_loss_pct=0.05))
        self.assertFalse(stop_loss_hit(side=1, entry_price=100.0, close=96.0, stop_loss_pct=0.05))

    def test_tv_ema252_blocks_buy(self):
        bar = SessionDailyBar("2026-01-02", 0, 100.0, 110.0, 100.0, 101.0)
        closes = [100.0] * 300
        self.assertEqual(
            evaluate_signal(
                prev_bar=bar,
                in_position=False,
                entry_threshold=0.20,
                exit_threshold=0.90,
                daily_closes=closes,
                trend_ma_type="ema",
                trend_ma_period=252,
                trend_price=99.0,
                ma_closes=closes + [99.0],
            ),
            "HOLD",
        )

    def test_tv_buy_when_above_ema(self):
        bar = SessionDailyBar("2026-01-02", 0, 100.0, 110.0, 100.0, 101.0)
        closes = [100.0] * 300
        self.assertEqual(
            evaluate_signal(
                prev_bar=bar,
                in_position=False,
                entry_threshold=0.20,
                exit_threshold=0.90,
                daily_closes=closes,
                trend_ma_type="ema",
                trend_ma_period=252,
                trend_price=105.0,
                ma_closes=closes + [105.0],
            ),
            "BUY",
        )

    def test_tv_exit_at_0_9(self):
        bar = SessionDailyBar("2026-01-03", 0, 100.0, 110.0, 100.0, 109.1)
        self.assertEqual(
            evaluate_signal(
                prev_bar=bar,
                in_position=True,
                entry_threshold=0.20,
                exit_threshold=0.90,
                daily_closes=[100.0] * 300,
                trend_ma_type="ema",
                trend_ma_period=252,
            ),
            "SELL",
        )

    def test_tv_max_trade_duration(self):
        bar = SessionDailyBar("2026-01-10", 0, 100.0, 110.0, 100.0, 105.0)
        self.assertEqual(
            evaluate_signal(
                prev_bar=bar,
                in_position=True,
                entry_threshold=0.20,
                exit_threshold=0.90,
                daily_closes=[100.0] * 300,
                holding_days=30,
                max_trade_duration_days=30,
            ),
            "SELL",
        )

    def test_tv_short_on_high_ibs_below_ema(self):
        bar = SessionDailyBar("2026-01-02", 0, 100.0, 110.0, 100.0, 109.5)
        closes = [100.0] * 300
        self.assertEqual(
            evaluate_signal(
                prev_bar=bar,
                position_side=0,
                trade_type="long_short",
                entry_threshold=0.20,
                exit_threshold=0.90,
                daily_closes=closes,
                trend_ma_type="ema",
                trend_ma_period=252,
                trend_price=95.0,
                ma_closes=closes + [95.0],
            ),
            "SHORT",
        )

    def test_tv_cover_short_on_low_ibs(self):
        bar = SessionDailyBar("2026-01-03", 0, 100.0, 110.0, 100.0, 101.0)
        self.assertEqual(
            evaluate_signal(
                prev_bar=bar,
                position_side=-1,
                trade_type="long_short",
                entry_threshold=0.20,
                exit_threshold=0.90,
                daily_closes=[100.0] * 300,
                trend_ma_type="ema",
                trend_ma_period=252,
            ),
            "COVER",
        )

    def test_stop_loss_hit_short(self):
        self.assertTrue(stop_loss_hit(side=-1, entry_price=100.0, close=106.0, stop_loss_pct=0.05))

    def test_ema_last(self):
        closes = [float(i) for i in range(1, 301)]
        ema = ema_last(closes, 252)
        self.assertIsNotNone(ema)
        self.assertGreater(float(ema), 150.0)


if __name__ == "__main__":
    unittest.main()

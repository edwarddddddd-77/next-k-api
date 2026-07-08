"""为 vnpy 官方策略类生成复利回测子类。"""

from __future__ import annotations

from typing import Callable, Type

from orb.vnpy.bootstrap import ensure_vnpy_path

ensure_vnpy_path()

from vnpy_ctastrategy import CtaTemplate  # noqa: E402
from vnpy_ctastrategy.strategies.double_ma_strategy import DoubleMaStrategy  # noqa: E402


def make_compound_backtest_class(
    base_cls: Type[CtaTemplate],
    *,
    sizing_fn: Callable[[float, float], float],
    fee_rate: float,
) -> Type[CtaTemplate]:
    """平仓后刷新 fixed_size / bt_wallet（官方策略 + 复利钱包）。"""

    if base_cls is DoubleMaStrategy:

        class DoubleMaCompoundBacktest(DoubleMaStrategy):
            fixed_size: float = 1.0
            bt_wallet: float = 1000.0

            parameters = list(DoubleMaStrategy.parameters) + ["fixed_size", "bt_wallet"]
            variables = list(DoubleMaStrategy.variables)

            def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
                super().__init__(cta_engine, strategy_name, vt_symbol, setting)
                self._bt_entry_px: float | None = None
                self._bt_entry_side: int = 0
                self._bt_last_px: float = 0.0

            def _refresh_size(self) -> None:
                px = float(self._bt_last_px or 100.0)
                eq = max(0.01, float(self.bt_wallet))
                vol = sizing_fn(px, eq)
                if vol > 0:
                    self.fixed_size = vol

            def _apply_close_pnl(self, trade) -> None:
                if self._bt_entry_px is None:
                    return
                px = float(trade.price)
                vol = float(trade.volume)
                side = int(self._bt_entry_side)
                gross = (px - self._bt_entry_px) * vol if side > 0 else (self._bt_entry_px - px) * vol
                fee = (self._bt_entry_px * vol + px * vol) * float(fee_rate)
                self.bt_wallet = max(0.01, float(self.bt_wallet) + gross - fee)
                self._bt_entry_px = None
                self._bt_entry_side = 0

            def on_bar(self, bar) -> None:
                self.cancel_all()
                am = self.am
                am.update_bar(bar)
                if not am.inited:
                    return
                import numpy as np

                fast_ma = am.sma(self.fast_window, array=True)
                self.fast_ma0 = fast_ma[-1]
                self.fast_ma1 = fast_ma[-2]
                slow_ma = am.sma(self.slow_window, array=True)
                self.slow_ma0 = slow_ma[-1]
                self.slow_ma1 = slow_ma[-2]
                cross_over = self.fast_ma0 > self.slow_ma0 and self.fast_ma1 < self.slow_ma1
                cross_below = self.fast_ma0 < self.slow_ma0 and self.fast_ma1 > self.slow_ma1
                vol = float(self.fixed_size)
                if cross_over:
                    if self.pos == 0:
                        self.buy(bar.close_price, vol)
                    elif self.pos < 0:
                        self.cover(bar.close_price, vol)
                        self.buy(bar.close_price, vol)
                elif cross_below:
                    if self.pos == 0:
                        self.short(bar.close_price, vol)
                    elif self.pos > 0:
                        self.sell(bar.close_price, vol)
                        self.short(bar.close_price, vol)
                self.put_event()

            def on_trade(self, trade) -> None:
                DoubleMaStrategy.on_trade(self, trade)
                self._bt_last_px = float(trade.price)
                if self.pos != 0:
                    if self._bt_entry_px is None:
                        self._bt_entry_px = float(trade.price)
                        self._bt_entry_side = 1 if self.pos > 0 else -1
                else:
                    self._apply_close_pnl(trade)
                    self._refresh_size()

        DoubleMaCompoundBacktest.__name__ = f"{base_cls.__name__}CompoundBacktest"
        return DoubleMaCompoundBacktest

    class CompoundBacktestStrategy(base_cls):
        bt_wallet: float = 1000.0

        parameters = list(getattr(base_cls, "parameters", [])) + ["bt_wallet"]
        variables = list(getattr(base_cls, "variables", []))

        def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
            super().__init__(cta_engine, strategy_name, vt_symbol, setting)
            self._bt_entry_px: float | None = None
            self._bt_entry_side: int = 0
            self._bt_last_px: float = 0.0

        def _refresh_size(self) -> None:
            if not hasattr(self, "fixed_size"):
                return
            px = float(self._bt_last_px or 100.0)
            eq = max(0.01, float(self.bt_wallet))
            vol = sizing_fn(px, eq)
            if vol > 0:
                self.fixed_size = vol

        def _apply_close_pnl(self, trade) -> None:
            if self._bt_entry_px is None:
                return
            px = float(trade.price)
            vol = float(trade.volume)
            side = int(self._bt_entry_side)
            gross = (px - self._bt_entry_px) * vol if side > 0 else (self._bt_entry_px - px) * vol
            fee = (self._bt_entry_px * vol + px * vol) * float(fee_rate)
            self.bt_wallet = max(0.01, float(self.bt_wallet) + gross - fee)
            self._bt_entry_px = None
            self._bt_entry_side = 0

        def on_trade(self, trade) -> None:
            super().on_trade(trade)
            self._bt_last_px = float(trade.price)
            if self.pos != 0:
                if self._bt_entry_px is None:
                    self._bt_entry_px = float(trade.price)
                    self._bt_entry_side = 1 if self.pos > 0 else -1
            else:
                self._apply_close_pnl(trade)
                self._refresh_size()

    CompoundBacktestStrategy.__name__ = f"{base_cls.__name__}CompoundBacktest"
    return CompoundBacktestStrategy

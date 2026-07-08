"""GTL strategy aligned with article semantics (birth forecast + break entry)."""

from __future__ import annotations

from orb.gtl.engine import GtlEngine
from orb.vnpy.bootstrap import ensure_vnpy_path

ensure_vnpy_path()

from vnpy.trader.constant import Direction  # noqa: E402
from vnpy_ctastrategy import (  # noqa: E402
    ArrayManager,
    BarData,
    BarGenerator,
    CtaTemplate,
    OrderData,
    StopOrder,
    TickData,
    TradeData,
)


class GtlBreakoutStrategy(CtaTemplate):
    """GTL vnpy strategy.

    trade_mode (article-aligned first):
      - birth_break: enter only when break direction matches frozen birth forecast (Log P Birth)
      - break: raw HH/LL close break (baseline)
      - signal: legacy per-bar live signal (not recommended)
      - signal_break: legacy break + live signal same bar
    """

    author = "next-k-api"

    lookback: int = 23
    vol_window: int = 500
    trade_mode: str = "birth_break"
    atr_window: int = 20
    atr_stop_mult: float = 2.0
    fixed_size: int = 1
    force_flat_on_stop: bool = True
    max_hold_bars: int = 0
    exit_on_opposite_break: bool = False

    frozen_hh: float = 0.0
    frozen_ll: float = 0.0
    theta_ceiling: float = 0.0
    theta_floor: float = 0.0
    prob_up: float = 0.5
    display_prob_up: float = 0.5
    verified_up: float = 0.5
    forecast_confidence: str = "low"
    gtl_signal: str = ""

    parameters = [
        "lookback",
        "vol_window",
        "trade_mode",
        "atr_window",
        "atr_stop_mult",
        "fixed_size",
        "force_flat_on_stop",
        "max_hold_bars",
        "exit_on_opposite_break",
    ]
    variables = [
        "frozen_hh",
        "frozen_ll",
        "theta_ceiling",
        "theta_floor",
        "prob_up",
        "display_prob_up",
        "verified_up",
        "forecast_confidence",
        "gtl_signal",
    ]

    def on_init(self) -> None:
        self.write_log("GTL strategy init")
        self.bg = BarGenerator(self.on_bar)
        warmup = max(self.vol_window, self.lookback, self.atr_window) + 5
        self.am = ArrayManager(size=warmup)
        self.gtl = GtlEngine(lookback=self.lookback, vol_window=self.vol_window)
        self.long_stop = 0.0
        self.short_stop = 0.0
        self.atr_value = 0.0
        self.entry_box_hh = 0.0
        self.entry_box_ll = 0.0
        self._last_reading = None
        self._last_bar: BarData | None = None
        self.bars_in_trade = 0
        self.load_bar(warmup)

    def on_start(self) -> None:
        self.write_log("GTL strategy start")

    def on_stop(self) -> None:
        if not self.force_flat_on_stop or self.pos == 0 or self._last_bar is None:
            self.write_log("GTL strategy stop")
            return
        px = float(self._last_bar.close_price)
        if self.pos > 0:
            self.sell(px, abs(self.pos))
        elif self.pos < 0:
            self.cover(px, abs(self.pos))
        self.write_log(f"GTL force flat on stop @ {px}")

    def on_tick(self, tick: TickData) -> None:
        self.bg.update_tick(tick)

    def _sync_reading(self, bar: BarData) -> None:
        r = self.gtl.update(bar.open_price, bar.high_price, bar.low_price, bar.close_price)
        self.frozen_hh = r.frozen_hh
        self.frozen_ll = r.frozen_ll
        self.theta_ceiling = r.theta_ceiling_display
        self.theta_floor = r.theta_floor_display
        self.prob_up = r.prob_up
        self.display_prob_up = r.display_prob_up
        self.verified_up = r.verified_up
        self.forecast_confidence = r.forecast_confidence
        if r.break_aligns_birth:
            self.gtl_signal = "birth_break"
        elif r.birth_gates_ok and r.is_birth_bar:
            self.gtl_signal = "birth_up" if r.birth_signal_up else "birth_down" if r.birth_signal_down else "birth"
        elif r.forecast_up:
            self.gtl_signal = f"up_{int(round(r.display_prob_up * 100))}"
        elif r.forecast_down:
            self.gtl_signal = f"down_{int(round(r.display_prob_down * 100))}"
        elif r.signal_up:
            self.gtl_signal = "up"
        elif r.signal_down:
            self.gtl_signal = "down"
        elif r.trade_abstain_reason:
            self.gtl_signal = r.trade_abstain_reason
        elif r.abstain_reason:
            self.gtl_signal = r.abstain_reason
        else:
            self.gtl_signal = ""
        self._last_reading = r

    def _allow_long(self, r) -> bool:
        mode = str(self.trade_mode).lower()
        if mode == "birth_break":
            return bool(r.break_aligns_birth and r.break_dir > 0)
        if mode == "break":
            return r.break_dir > 0
        if mode == "signal":
            return r.signal_up
        if mode == "signal_break":
            return r.break_dir > 0 and r.signal_up
        return False

    def _allow_short(self, r) -> bool:
        mode = str(self.trade_mode).lower()
        if mode == "birth_break":
            return bool(r.break_aligns_birth and r.break_dir < 0)
        if mode == "break":
            return r.break_dir < 0
        if mode == "signal":
            return r.signal_down
        if mode == "signal_break":
            return r.break_dir < 0 and r.signal_down
        return False

    def _gtl_ready(self, r) -> bool:
        return bool(r.structure_active) and r.abstain_reason != "warmup"

    def on_bar(self, bar: BarData) -> None:
        self._last_bar = bar
        self.cancel_all()
        self.am.update_bar(bar)
        self._sync_reading(bar)
        r = self._last_reading
        if r is None or not self.am.inited or not self._gtl_ready(r):
            return

        self.atr_value = float(self.am.atr(self.atr_window))

        if self.pos == 0:
            self.bars_in_trade = 0
            if self._allow_long(r):
                self.entry_box_hh = r.broken_hh or r.frozen_hh
                self.entry_box_ll = r.broken_ll or r.frozen_ll
                self.long_stop = bar.close_price - self.atr_stop_mult * self.atr_value
                self.buy(bar.close_price, self.fixed_size)
            elif self._allow_short(r):
                self.entry_box_hh = r.broken_hh or r.frozen_hh
                self.entry_box_ll = r.broken_ll or r.frozen_ll
                self.short_stop = bar.close_price + self.atr_stop_mult * self.atr_value
                self.short(bar.close_price, self.fixed_size)
        else:
            self.bars_in_trade += 1
            time_exit = self.max_hold_bars > 0 and self.bars_in_trade >= self.max_hold_bars
            opp_exit = (
                self.exit_on_opposite_break
                and r.break_aligns_birth
                and ((self.pos > 0 and r.break_dir < 0) or (self.pos < 0 and r.break_dir > 0))
            )
            if time_exit or opp_exit:
                px = float(bar.close_price)
                if self.pos > 0:
                    self.sell(px, abs(self.pos))
                else:
                    self.cover(px, abs(self.pos))
            elif self.pos > 0:
                box_stop = self.entry_box_ll if self.entry_box_ll > 0 else self.frozen_ll
                stop = max(self.long_stop, box_stop)
                self.sell(stop, abs(self.pos), True)
            elif self.pos < 0:
                box_stop = self.entry_box_hh if self.entry_box_hh > 0 else self.frozen_hh
                stop = min(self.short_stop, box_stop)
                self.cover(stop, abs(self.pos), True)

        self.put_event()

    def on_trade(self, trade: TradeData) -> None:
        if trade.direction == Direction.LONG:
            self.long_stop = trade.price - self.atr_stop_mult * self.atr_value
        else:
            self.short_stop = trade.price + self.atr_stop_mult * self.atr_value

    def on_order(self, order: OrderData) -> None:
        pass

    def on_stop_order(self, stop_order: StopOrder) -> None:
        pass

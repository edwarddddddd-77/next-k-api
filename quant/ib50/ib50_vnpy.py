"""IB50 vnpy 策略 — Initial Balance 50% 机械延续。"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from quant.common.macro_calendar import is_macro_skip_day
from quant.common.session import session_anchor_ms, session_day_str
from quant.engine.bootstrap import ensure_vnpy_path
from quant.engine.eod import should_eod_flat_bar
from quant.engine.exchanges.registry import symbol_from_vt
from quant.ib50.config import Ib50Config
from quant.ib50.core import (
    build_ib50_setup,
    finalize_initial_balance,
    ib_complete_at_bar,
    in_ib_window,
    update_ib_range,
    weekday_allowed,
)
from quant.ib50.sizing import fixed_size_for_ib50, risk_budget_usdt

ensure_vnpy_path()

from vnpy.trader.constant import Direction, Interval, Offset, OrderType  # noqa: E402
from vnpy.trader.utility import round_to  # noqa: E402
from vnpy_ctastrategy import (  # noqa: E402
    BarData,
    BarGenerator,
    CtaTemplate,
    OrderData,
    StopOrder,
    TickData,
    TradeData,
)


class Ib50VnpyStrategy(CtaTemplate):
    """Initial Balance 50% — 60m IB，先极值定方向，IB 结束后市价入场。"""

    author = "next-k-api"

    ib_minutes: int = 60
    direction_mode: str = "continuation"
    entry_end_hour: int = 15
    entry_end_minute: int = 0
    ib50_rth_only: bool = True
    ib50_eod_flat: bool = True
    ib50_exit_hour: int = 15
    ib50_exit_minute: int = 50
    fixed_size: float = 1.0

    parameters = [
        "ib_minutes",
        "direction_mode",
        "entry_end_hour",
        "entry_end_minute",
        "ib50_rth_only",
        "ib50_eod_flat",
        "ib50_exit_hour",
        "ib50_exit_minute",
        "fixed_size",
    ]
    variables = [
        "session_date",
        "ib_high",
        "ib_low",
        "first_extreme",
        "ib_ready",
        "traded_today",
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.session_date: str = ""
        self.ib_high: float = 0.0
        self.ib_low: float = 0.0
        self.first_extreme: str = ""
        self.ib_ready: bool = False
        self.traded_today: bool = False
        self.entry_price: float = 0.0
        self.stop_price: float = 0.0
        self.target_price: float = 0.0
        self._entry_pending: bool = False
        self._exit_pending: bool = False
        self._restore_entry_px: float = 0.0
        self._last_bar: Optional[BarData] = None
        self._prev_bar_high: float = 0.0
        self._prev_bar_low: float = 0.0

    @classmethod
    def from_ib50_config(cls, cfg: Ib50Config) -> dict:
        return {
            "ib_minutes": int(cfg.ib_minutes),
            "direction_mode": str(cfg.direction_mode),
            "entry_end_hour": int(cfg.entry_end_hour),
            "entry_end_minute": int(cfg.entry_end_minute),
            "ib50_rth_only": bool(cfg.rth_only),
            "ib50_eod_flat": bool(cfg.eod_flat),
            "ib50_exit_hour": int(cfg.exit_hour),
            "ib50_exit_minute": int(cfg.exit_minute),
        }

    def _cfg(self) -> Ib50Config:
        return Ib50Config.from_env()

    def _session_cfg(self):
        return self._cfg().session_cfg()

    def _bar_ms(self, bar: BarData) -> int:
        return int(bar.datetime.timestamp() * 1000)

    def _bar_session_ts(self, bar: BarData) -> pd.Timestamp:
        cfg = self._session_cfg()
        return pd.Timestamp(self._bar_ms(bar), unit="ms", tz=cfg.session_tz)

    def _in_rth(self, bar: BarData) -> bool:
        if not self.ib50_rth_only:
            return True
        from quant.common.session_paper import in_regular_session

        return bool(in_regular_session(self._session_cfg(), now_ms=self._bar_ms(bar)))

    def _is_eod_bar(self, bar: BarData) -> bool:
        if not self.ib50_eod_flat:
            return False
        cfg = self._session_cfg()
        ms = self._bar_ms(bar)
        ts = self._bar_session_ts(bar)
        return should_eod_flat_bar(
            bar_ms=ms,
            ts=ts,
            cfg=cfg,
            exit_hour=int(self.ib50_exit_hour),
            exit_minute=int(self.ib50_exit_minute),
        )

    def _should_flatten_eod(self, bar: BarData) -> bool:
        if not self.ib50_eod_flat or self.pos == 0:
            return False
        return self._is_eod_bar(bar) or not self._in_rth(bar)

    def _hm_to_min(self, hour: int, minute: int) -> int:
        return int(hour) * 60 + int(minute)

    def _in_entry_window(self, bar: BarData) -> bool:
        cfg = self._session_cfg()
        ms = self._bar_ms(bar)
        anchor = session_anchor_ms(ms, tz=cfg.session_tz, session_open_time=cfg.session_open_time)
        ib_end = anchor + int(self.ib_minutes) * 60_000
        if ms < ib_end:
            return False
        ts = self._bar_session_ts(bar)
        t = ts.hour * 60 + ts.minute
        end = self._hm_to_min(self.entry_end_hour, self.entry_end_minute)
        ib_end_ts = pd.Timestamp(ib_end, unit="ms", tz=cfg.session_tz)
        start = ib_end_ts.hour * 60 + ib_end_ts.minute
        return start <= t <= end

    def _reset_session_state(self) -> None:
        self.ib_high = 0.0
        self.ib_low = 0.0
        self.first_extreme = ""
        self.ib_ready = False
        self.traded_today = False
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.target_price = 0.0
        self._entry_pending = False
        self._exit_pending = False
        self._restore_entry_px = 0.0
        self._prev_bar_high = 0.0
        self._prev_bar_low = 0.0

    def _send_market(self, direction: Direction, offset: Offset, volume: float) -> List[str]:
        if not self.trading or not self.cta_engine:
            return []
        vol = float(volume or 0.0)
        if vol <= 0:
            return []
        contract = self.cta_engine.main_engine.get_contract(self.vt_symbol)
        if contract is None:
            return []
        vol = round_to(vol, float(contract.min_volume or 0.001))
        if vol <= 0:
            return []
        return self.cta_engine.send_server_order(
            self,
            contract,
            direction,
            offset,
            0.0,
            vol,
            OrderType.MARKET,
            False,
            False,
        )

    def _flatten_at_bar(self, bar: BarData) -> None:
        if self.pos == 0 or self._exit_pending:
            return
        self.cancel_all()
        vol = abs(self.pos)
        side = "LONG" if self.pos > 0 else "SHORT"
        self.write_log(f"EOD flatten {self.vt_symbol} {side} vol={vol} market")
        oids: List[str] = []
        if self.pos > 0:
            oids = self._send_market(Direction.SHORT, Offset.CLOSE, vol)
        elif self.pos < 0:
            oids = self._send_market(Direction.LONG, Offset.CLOSE, vol)
        if oids:
            self._exit_pending = True

    def _apply_levels(self, setup) -> None:
        self.entry_price = float(setup.entry_price)
        self.stop_price = float(setup.stop)
        self.target_price = float(setup.target)

    def restore_synced_position(self, *, entry_px: float, pos: float) -> None:
        if pos == 0 or entry_px <= 0:
            return
        if self.stop_price > 0 and self.entry_price > 0:
            return
        self._entry_pending = False
        cfg = self._cfg()
        if cfg.one_trade_per_session:
            self.traded_today = True
        ib = finalize_initial_balance(
            ib_high=self.ib_high,
            ib_low=self.ib_low,
            first_extreme=self.first_extreme or None,
        )
        if ib is None:
            self._restore_entry_px = float(entry_px)
            return
        setup = build_ib50_setup(ib, entry_px, direction_mode=self.direction_mode)
        self._apply_levels(setup)
        self._restore_entry_px = 0.0
        self.write_log(
            f"restored levels {self.vt_symbol} entry={entry_px:.4f} "
            f"sl={self.stop_price:.4f} tp={self.target_price:.4f}"
        )

    def _try_deferred_restore(self) -> None:
        if self._restore_entry_px <= 0 or self.pos == 0 or not self.ib_ready:
            return
        self.restore_synced_position(entry_px=self._restore_entry_px, pos=float(self.pos))

    def _sync_session(self, bar: BarData) -> None:
        cfg = self._session_cfg()
        ms = self._bar_ms(bar)
        day = session_day_str(ms, tz=cfg.session_tz, session_open_time=cfg.session_open_time)
        if day != self.session_date:
            self.session_date = day
            self._reset_session_state()

    def _update_initial_balance(self, bar: BarData) -> None:
        cfg = self._session_cfg()
        ms = self._bar_ms(bar)
        anchor = session_anchor_ms(ms, tz=cfg.session_tz, session_open_time=cfg.session_open_time)
        if not in_ib_window(ms, anchor_ms=anchor, ib_minutes=int(self.ib_minutes)):
            if ib_complete_at_bar(ms, anchor_ms=anchor, ib_minutes=int(self.ib_minutes)):
                if self.ib_high > 0 and self.ib_low > 0 and not self.ib_ready:
                    self.ib_ready = True
            return
        first = self.first_extreme or None
        self.ib_high, self.ib_low, first = update_ib_range(
            ib_high=self.ib_high,
            ib_low=self.ib_low,
            first_extreme=first,
            open_=float(bar.open_price),
            high=float(bar.high_price),
            low=float(bar.low_price),
        )
        if first:
            self.first_extreme = first

    def _check_exit_on_bar(self, bar: BarData) -> bool:
        if self.pos == 0 or self._exit_pending or self.stop_price <= 0:
            return False
        hi = float(bar.high_price)
        lo = float(bar.low_price)
        vol = abs(self.pos)
        reason = ""
        if self.pos > 0:
            if lo <= self.stop_price:
                reason = "stop_loss"
            elif self.target_price > 0 and hi >= self.target_price:
                reason = "target_hit"
            else:
                return False
            self.cancel_all()
            self.write_log(
                f"EXIT {self.vt_symbol} LONG {reason} stop={self.stop_price:.4f} "
                f"target={self.target_price:.4f} market"
            )
            oids = self._send_market(Direction.SHORT, Offset.CLOSE, vol)
            if oids:
                self._exit_pending = True
            return bool(oids)
        if hi >= self.stop_price:
            reason = "stop_loss"
        elif self.target_price > 0 and lo <= self.target_price:
            reason = "target_hit"
        else:
            return False
        self.cancel_all()
        self.write_log(
            f"EXIT {self.vt_symbol} SHORT {reason} stop={self.stop_price:.4f} "
            f"target={self.target_price:.4f} market"
        )
        oids = self._send_market(Direction.LONG, Offset.CLOSE, vol)
        if oids:
            self._exit_pending = True
        return bool(oids)

    def _open_market(self, side: int, vol: float) -> None:
        if side > 0:
            oids = self._send_market(Direction.LONG, Offset.OPEN, vol)
        else:
            oids = self._send_market(Direction.SHORT, Offset.OPEN, vol)
        cfg = self._cfg()
        if oids:
            self._entry_pending = True
            if cfg.one_trade_per_session:
                self.traded_today = True
            return
        if cfg.shadow or not cfg.live_enabled:
            if cfg.one_trade_per_session:
                self.traded_today = True
            self.write_log(
                f"signal-only {self.vt_symbol} (shadow={cfg.shadow} live={cfg.live_enabled})"
            )
            return
        self.write_log(f"live entry rejected {self.vt_symbol}")

    def _try_entry(self, bar: BarData) -> None:
        cfg = self._cfg()
        if self.pos != 0 or self._entry_pending:
            return
        if cfg.one_trade_per_session and self.traded_today:
            return
        if cfg.macro_filter and is_macro_skip_day(self.session_date):
            return
        if not self.ib_ready or self.ib_high <= self.ib_low:
            return
        if not self._in_entry_window(bar):
            return
        ts = self._bar_session_ts(bar)
        if not weekday_allowed(int(ts.weekday()), cfg.weekday_filter()):
            return

        ib = finalize_initial_balance(
            ib_high=self.ib_high,
            ib_low=self.ib_low,
            first_extreme=self.first_extreme or None,
        )
        if ib is None:
            return

        close = float(bar.close_price)
        setup = build_ib50_setup(ib, close, direction_mode=self.direction_mode)
        stop_dist = abs(close - setup.stop)
        if stop_dist <= 0:
            return
        self._apply_levels(setup)

        eq = float(cfg.equity_usdt or 50.0)
        if cfg.compound:
            try:
                from accumulation_radar import init_db
                from quant.common.vnpy_wallet import lane_equity_usdt, migrate_vnpy_lane_tables

                conn = init_db()
                try:
                    cur = conn.cursor()
                    migrate_vnpy_lane_tables(cur)
                    sym = symbol_from_vt(self.vt_symbol)
                    eq = lane_equity_usdt(cfg, sym, cur=cur)
                finally:
                    conn.close()
            except Exception:
                pass

        vol = fixed_size_for_ib50(
            cfg,
            symbol_from_vt(self.vt_symbol),
            close,
            stop_distance=stop_dist,
            equity_usdt=eq,
        )
        if vol <= 0:
            return
        self.fixed_size = vol
        risk_usd = risk_budget_usdt(cfg, equity_usdt=eq)
        self.write_log(
            f"IB50 signal {self.vt_symbol} side={'LONG' if setup.side > 0 else 'SHORT'} "
            f"px={close:.4f} ib=[{ib.low:.4f},{ib.high:.4f}] first={ib.first_extreme} "
            f"risk=${risk_usd:.2f} vol={vol}"
        )
        try:
            from quant.engine.strategy_signals import LANE_IB50, record_strategy_open_signal

            record_strategy_open_signal(
                lane=LANE_IB50,
                symbol=symbol_from_vt(self.vt_symbol),
                side="LONG" if setup.side > 0 else "SHORT",
                entry_price=close,
                sl_price=float(self.stop_price) if self.stop_price else None,
                tp_price=float(self.target_price) if self.target_price else None,
                status="shadow" if (cfg.shadow or not cfg.live_enabled) else "emitted",
                bar_ms=self._bar_ms(bar),
                detail={
                    "first_extreme": ib.first_extreme,
                    "ib_high": ib.high,
                    "ib_low": ib.low,
                    "direction_mode": self.direction_mode,
                    "vol": vol,
                    "risk_usd": risk_usd,
                },
            )
        except Exception as exc:
            self.write_log(f"strategy signal persist failed: {exc}")
        self._open_market(setup.side, vol)

    def on_init(self) -> None:
        self.write_log("IB50 strategy init")
        self.bg = BarGenerator(self.on_bar, 5, self.on_5min_bar, Interval.MINUTE)
        self._reset_session_state()

    def on_start(self) -> None:
        self.write_log("IB50 strategy start")

    def on_stop(self) -> None:
        if self.pos != 0 and self._last_bar is not None:
            self._flatten_at_bar(self._last_bar)
        self.write_log("IB50 strategy stop")

    def on_tick(self, tick: TickData) -> None:
        extra = getattr(tick, "extra", None) or {}
        bar = extra.get("bar")
        if bar is not None:
            self._on_1min_bar(bar)
            return
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        self._on_1min_bar(bar)

    def _on_1min_bar(self, bar: BarData) -> None:
        if self.trading:
            self._sync_session(bar)
            self._update_initial_balance(bar)
            self._try_deferred_restore()
        self.bg.update_bar(bar)

    def on_5min_bar(self, bar: BarData) -> None:
        self._last_bar = bar
        if self._should_flatten_eod(bar):
            self._flatten_at_bar(bar)
            self.put_event()
            return
        if not self._in_rth(bar):
            cfg = self._cfg()
            if cfg.vnpy_idle_outside_rth:
                self.cancel_all()
            self.put_event()
            return

        if self.pos != 0:
            if self._check_exit_on_bar(bar):
                self.put_event()
                return
            self.put_event()
            return

        self._try_entry(bar)
        self.put_event()

    def on_trade(self, trade: TradeData) -> None:
        if trade.offset == Offset.OPEN and self.pos != 0:
            ib = finalize_initial_balance(
                ib_high=self.ib_high,
                ib_low=self.ib_low,
                first_extreme=self.first_extreme or None,
            )
            if ib is not None:
                setup = build_ib50_setup(
                    ib,
                    float(trade.price),
                    direction_mode=self.direction_mode,
                )
                self._apply_levels(setup)
            cfg = self._cfg()
            if cfg.one_trade_per_session:
                self.traded_today = True
            self._entry_pending = False
        if self.pos == 0:
            self.cancel_all()
            self._entry_pending = False
            self._exit_pending = False
            self.entry_price = 0.0
            self.stop_price = 0.0
            self.target_price = 0.0

    def on_order(self, order: OrderData) -> None:
        pass

    def on_stop_order(self, stop_order: StopOrder) -> None:
        pass

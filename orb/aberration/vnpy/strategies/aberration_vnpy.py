"""Aberration 布林带突破 CTA — vnpy 策略（币安 USDT 永续）。

逻辑对齐 FMZ #126612 社区开源版：
  · 空仓：收盘价突破上轨做多 / 跌破下轨做空
  · 持仓：回到中轨平仓
  · 默认 1h K 线（bar_hours=1），可改为 4h / 日线等
"""

from __future__ import annotations

from datetime import time
from typing import List, Optional

from orb.aberration.config import AberrationVnpyConfig
from orb.aberration.core import aberration_action, aberration_bands
from orb.aberration.vnpy.bootstrap import ensure_vnpy_path
from orb.vnpy.binance_gateway import symbol_from_vt

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


class AberrationVnpyStrategy(CtaTemplate):
    author = "next-k-api"

    n_period: int = 35
    k_up: float = 2.0
    k_down: float = 2.0
    bar_hours: int = 1
    position_pct: float = 1.0
    leverage: float = 2.0
    fixed_size: float = 0.01

    up_track: float = 0.0
    mid_track: float = 0.0
    down_track: float = 0.0
    last_signal: str = ""

    parameters = [
        "n_period",
        "k_up",
        "k_down",
        "bar_hours",
        "position_pct",
        "leverage",
        "fixed_size",
    ]
    variables = ["up_track", "mid_track", "down_track", "last_signal"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self._closes: List[float] = []
        self._shadow_pos: float = 0.0
        self._last_bar: Optional[BarData] = None

    @classmethod
    def from_aberration_config(cls, cfg: AberrationVnpyConfig) -> dict:
        return {
            "n_period": int(cfg.n_period),
            "k_up": float(cfg.k_up),
            "k_down": float(cfg.k_down),
            "bar_hours": int(cfg.bar_hours),
            "position_pct": float(cfg.position_pct),
            "leverage": float(cfg.leverage),
        }

    def _cfg(self) -> AberrationVnpyConfig:
        return AberrationVnpyConfig.from_env()

    def _live_or_shadow(self) -> bool:
        cfg = self._cfg()
        return bool(cfg.shadow or not cfg.live_enabled)

    def _effective_pos(self) -> float:
        if self._live_or_shadow():
            return float(self._shadow_pos)
        return float(self.pos)

    def _equity_usdt(self) -> float:
        cfg = self._cfg()
        eq = float(cfg.equity_usdt)
        if not cfg.compound:
            return eq
        try:
            from accumulation_radar import init_db
            from orb.aberration.equity import symbol_equity_usdt
            from orb.trading_orb.db import migrate_orb_vnpy_tables

            conn = init_db()
            try:
                cur = conn.cursor()
                migrate_orb_vnpy_tables(cur)
                sym = symbol_from_vt(self.vt_symbol)
                eq = symbol_equity_usdt(cfg, sym, cur=cur)
            finally:
                conn.close()
        except Exception:
            pass
        return max(1.0, eq)

    def _order_volume(self, price: float) -> float:
        from orb.aberration.vnpy.sizing import fixed_size_for_aberration

        cfg = self._cfg()
        vol = fixed_size_for_aberration(cfg, price, equity_usdt=self._equity_usdt())
        if vol > 0:
            self.fixed_size = vol
        return vol

    def _send_market(self, direction: Direction, offset: Offset, volume: float) -> bool:
        if not self.trading or not self.cta_engine:
            return False
        vol = float(volume or 0.0)
        if vol <= 0:
            return False
        contract = self.cta_engine.main_engine.get_contract(self.vt_symbol)
        if contract is None:
            return False
        vol = round_to(vol, float(contract.min_volume or 0.001))
        if vol <= 0:
            return False
        oids = self.cta_engine.send_server_order(
            self, contract, direction, offset, 0.0, vol, OrderType.MARKET, False, False
        )
        return bool(oids)

    def _record_open_signal(
        self,
        *,
        side: str,
        price: float,
        bar_ms: int,
        detail: Optional[dict] = None,
    ) -> None:
        try:
            from orb.vnpy.strategy_signals import LANE_ABERRATION, record_strategy_open_signal

            status = "shadow" if self._live_or_shadow() else "emitted"
            record_strategy_open_signal(
                lane=LANE_ABERRATION,
                symbol=symbol_from_vt(self.vt_symbol),
                side=side,
                entry_price=price,
                status=status,
                bar_ms=int(bar_ms or 0),
                detail={
                    "upper": self.up_track,
                    "middle": self.mid_track,
                    "lower": self.down_track,
                    **(detail or {}),
                },
            )
        except Exception as exc:
            self.write_log(f"strategy signal persist failed: {exc}")

    def _make_bar_generator(self) -> BarGenerator:
        hours = max(1, int(self.bar_hours))
        if hours >= 24:
            return BarGenerator(
                self.on_bar,
                1,
                self.on_signal_bar,
                Interval.DAILY,
                daily_end=time(23, 59),
            )
        return BarGenerator(self.on_bar, hours, self.on_signal_bar, Interval.HOUR)

    def on_init(self) -> None:
        self.write_log("Aberration strategy init")
        self.bg = self._make_bar_generator()
        days = max(7, int(self._cfg().init_bar_days))
        self.load_bar(days)
        self.write_log(
            f"Aberration init done closes={len(self._closes)} "
            f"bands=({self.up_track:.4f},{self.mid_track:.4f},{self.down_track:.4f})"
        )

    def on_start(self) -> None:
        self.write_log("Aberration strategy start")

    def on_stop(self) -> None:
        if self._live_or_shadow() and self._shadow_pos != 0:
            self._shadow_pos = 0.0
        self.write_log("Aberration strategy stop")

    def on_tick(self, tick: TickData) -> None:
        extra = getattr(tick, "extra", None) or {}
        bar = extra.get("bar")
        if bar is not None:
            self.on_bar(bar)
            return
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        self.bg.update_bar(bar)

    def _append_close(self, bar: BarData) -> None:
        self._closes.append(float(bar.close_price))
        max_len = max(500, int(self.n_period) * 4)
        if len(self._closes) > max_len:
            self._closes = self._closes[-max_len:]

    def restore_synced_position(self, *, entry_px: float, pos: float) -> None:
        if pos == 0 or entry_px <= 0:
            return
        self.write_log(f"restored Aberration {self.vt_symbol} entry={entry_px:.4f} pos={pos}")

    def on_signal_bar(self, bar: BarData) -> None:
        self._last_bar = bar
        prior = list(self._closes)
        self._append_close(bar)
        bands = aberration_bands(
            prior,
            n_period=int(self.n_period),
            k_up=float(self.k_up),
            k_down=float(self.k_down),
        )
        if bands is None:
            self.put_event()
            return
        upper, middle, lower = bands
        self.up_track = round(upper, 8)
        self.mid_track = round(middle, 8)
        self.down_track = round(lower, 8)

        if not self.trading:
            self.put_event()
            return

        close = float(bar.close_price)
        pos = self._effective_pos()
        act = aberration_action(pos, close, upper, middle, lower)
        if not act:
            self.put_event()
            return

        bar_ms = int(bar.datetime.timestamp() * 1000)
        vol = self._order_volume(close)
        if vol <= 0:
            self.put_event()
            return

        self.last_signal = act
        shadow = self._live_or_shadow()

        if act == "long":
            self._record_open_signal(side="LONG", price=close, bar_ms=bar_ms)
            if shadow:
                self._shadow_pos = vol
                self.write_log(f"SHADOW LONG @ {close:.4f} upper={upper:.4f}")
            elif self.pos <= 0:
                if self.pos < 0:
                    self._send_market(Direction.LONG, Offset.CLOSE, abs(self.pos))
                self._send_market(Direction.LONG, Offset.OPEN, vol)
                self.write_log(f"LONG @ {close:.4f} upper={upper:.4f}")

        elif act == "short":
            self._record_open_signal(side="SHORT", price=close, bar_ms=bar_ms)
            if shadow:
                self._shadow_pos = -vol
                self.write_log(f"SHADOW SHORT @ {close:.4f} lower={lower:.4f}")
            elif self.pos >= 0:
                if self.pos > 0:
                    self._send_market(Direction.SHORT, Offset.CLOSE, abs(self.pos))
                self._send_market(Direction.SHORT, Offset.OPEN, vol)
                self.write_log(f"SHORT @ {close:.4f} lower={lower:.4f}")

        elif act == "close_long":
            if shadow:
                self._shadow_pos = 0.0
                self.write_log(f"SHADOW CLOSE LONG @ {close:.4f} mid={middle:.4f}")
            elif self.pos > 0:
                self._send_market(Direction.SHORT, Offset.CLOSE, abs(self.pos))
                self.write_log(f"CLOSE LONG @ {close:.4f} mid={middle:.4f}")

        elif act == "close_short":
            if shadow:
                self._shadow_pos = 0.0
                self.write_log(f"SHADOW CLOSE SHORT @ {close:.4f} mid={middle:.4f}")
            elif self.pos < 0:
                self._send_market(Direction.LONG, Offset.CLOSE, abs(self.pos))
                self.write_log(f"CLOSE SHORT @ {close:.4f} mid={middle:.4f}")

        self.put_event()

    def on_order(self, order: OrderData) -> None:
        pass

    def on_trade(self, trade: TradeData) -> None:
        if self.pos == 0:
            self._shadow_pos = 0.0
            self.put_event()

    def on_stop_order(self, stop_order: StopOrder) -> None:
        pass

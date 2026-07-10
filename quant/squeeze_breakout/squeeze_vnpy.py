"""Smart Breakout Targets vnpy 策略（WillyAlgoTrader v1.5）。"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Optional

from quant.engine.bootstrap import ensure_vnpy_path
from quant.engine.exchanges.registry import symbol_from_vt
from quant.market import fetch_klines_forward, klines_to_df, resolve_market_data_exchange_id
from quant.squeeze_breakout.config import SqueezeBreakoutConfig
from quant.squeeze_breakout.core import (
    BarOhlcv,
    BreakoutEngineState,
    BreakoutSignal,
    bar_exit_reason,
    bar_tp_milestones,
    breakout_levels,
    process_signal_bar,
    replay_engine_state,
)
from quant.squeeze_breakout.sizing import risk_budget_usdt, size_for_breakout

ensure_vnpy_path()

from vnpy.trader.constant import Direction, Interval, Offset, OrderType  # noqa: E402
from vnpy.trader.object import BarData, OrderData, TickData, TradeData  # noqa: E402
from vnpy.trader.utility import round_to  # noqa: E402
from vnpy_ctastrategy import BarGenerator, CtaTemplate  # noqa: E402


class SqueezeBreakoutVnpyStrategy(CtaTemplate):
    """双引擎 squeeze → 自适应 Donchian 区间 → 冲动 K 突破 → 1R/2R/3R 目标。"""

    author = "next-k-api"

    squeeze_length: int = 20
    bb_mult: float = 2.0
    squeeze_threshold: float = 0.6
    atr_compress_ratio: float = 0.75
    min_squeeze_bars: int = 5
    impulse_mult: float = 0.8
    sl_atr_buffer: float = 0.5
    tp1_rr: float = 1.0
    tp2_rr: float = 2.0
    tp3_rr: float = 3.0
    signal_minutes: int = 15
    fixed_size: float = 1.0
    orb_rth_only: bool = False

    parameters = [
        "squeeze_length",
        "bb_mult",
        "squeeze_threshold",
        "atr_compress_ratio",
        "min_squeeze_bars",
        "impulse_mult",
        "sl_atr_buffer",
        "tp1_rr",
        "tp2_rr",
        "tp3_rr",
        "signal_minutes",
        "fixed_size",
        "orb_rth_only",
    ]
    variables = ["stop_price", "target_price", "entry_price", "tp1_price", "tp2_price"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.entry_price: float = 0.0
        self.stop_price: float = 0.0
        self.tp1_price: float = 0.0
        self.tp2_price: float = 0.0
        self.target_price: float = 0.0
        self._signal_bars: List[BarOhlcv] = []
        self._engine_state = BreakoutEngineState()
        self._prev_bar_high: float = 0.0
        self._prev_bar_low: float = 0.0
        self._last_signal_bar: Optional[BarData] = None
        self._exit_pending: bool = False
        self._entry_pending: bool = False
        self._pending_signal: BreakoutSignal | None = None
        self._range_top: float = 0.0
        self._range_bottom: float = 0.0
        self._last_atr: float = 0.0
        self._tp1_hit: bool = False
        self._tp2_hit: bool = False
        self.bg: BarGenerator | None = None

    @classmethod
    def from_breakout_config(cls, cfg: SqueezeBreakoutConfig) -> dict:
        return {
            "squeeze_length": int(cfg.squeeze_length),
            "bb_mult": float(cfg.bb_mult),
            "squeeze_threshold": float(cfg.squeeze_threshold),
            "atr_compress_ratio": float(cfg.atr_compress_ratio),
            "min_squeeze_bars": int(cfg.min_squeeze_bars),
            "impulse_mult": float(cfg.impulse_mult),
            "sl_atr_buffer": float(cfg.sl_atr_buffer),
            "tp1_rr": float(cfg.tp1_rr),
            "tp2_rr": float(cfg.tp2_rr),
            "tp3_rr": float(cfg.tp3_rr),
            "signal_minutes": int(cfg.signal_minutes),
            "orb_rth_only": False,
        }

    def _cfg(self) -> SqueezeBreakoutConfig:
        return SqueezeBreakoutConfig.from_env()

    def _append_signal_bar(self, bar: BarData) -> None:
        ts = int(bar.datetime.timestamp() * 1000)
        row: BarOhlcv = (
            ts,
            float(bar.open_price),
            float(bar.high_price),
            float(bar.low_price),
            float(bar.close_price),
            float(bar.volume or 0.0),
        )
        if self._signal_bars and self._signal_bars[-1][0] == ts:
            self._signal_bars[-1] = row
        else:
            self._signal_bars.append(row)
        max_bars = (24 * 4) * max(40, self._cfg().init_bar_days)
        if len(self._signal_bars) > max_bars:
            self._signal_bars = self._signal_bars[-max_bars:]
            self._rebuild_engine_state()

    def _engine_kwargs(self, cfg: SqueezeBreakoutConfig) -> dict:
        return {
            "squeeze_length": int(self.squeeze_length),
            "bb_mult": float(self.bb_mult),
            "squeeze_threshold": float(self.squeeze_threshold),
            "atr_compress_ratio": float(self.atr_compress_ratio),
            "min_squeeze_bars": int(self.min_squeeze_bars),
            "impulse_mult": float(self.impulse_mult),
            "sl_atr_buffer": float(self.sl_atr_buffer),
            "tp1_rr": float(self.tp1_rr),
            "tp2_rr": float(self.tp2_rr),
            "tp3_rr": float(self.tp3_rr),
            "prevent_overlap": bool(cfg.prevent_overlap),
            "volume_filter": bool(cfg.volume_filter),
            "volume_mult": float(cfg.volume_mult),
        }

    def _rebuild_engine_state(self) -> None:
        cfg = self._cfg()
        self._engine_state = replay_engine_state(self._signal_bars, **self._engine_kwargs(cfg))

    def _levels_from_range(self, entry: float, side: int, top: float, bottom: float, atr: float) -> None:
        stop, tp1, tp2, tp3, _ = breakout_levels(
            float(entry),
            side,
            float(top),
            float(bottom),
            float(atr),
            sl_atr_buffer=float(self.sl_atr_buffer),
            tp1_rr=float(self.tp1_rr),
            tp2_rr=float(self.tp2_rr),
            tp3_rr=float(self.tp3_rr),
        )
        self.entry_price = float(entry)
        self.stop_price = float(stop)
        self.tp1_price = float(tp1)
        self.tp2_price = float(tp2)
        self.target_price = float(tp3)

    def _apply_levels(self, sig: BreakoutSignal) -> None:
        self._range_top = float(sig.range_top)
        self._range_bottom = float(sig.range_bottom)
        self._last_atr = float(sig.atr)
        self._levels_from_range(sig.entry, sig.side, sig.range_top, sig.range_bottom, sig.atr)

    def restore_synced_position(self, *, entry_px: float, pos: float) -> None:
        if abs(float(pos or 0.0)) < 1e-12:
            return
        side = 1 if float(pos) > 0 else -1
        px = float(entry_px)
        width = max(px * 0.02, 1.0)
        top = px + width / 2
        bottom = px - width / 2
        atr_proxy = width / 4
        self._range_top = top
        self._range_bottom = bottom
        self._last_atr = atr_proxy
        self._levels_from_range(px, side, top, bottom, atr_proxy)
        self.write_log(
            f"restored synced position entry={px:.4f} stop={self.stop_price:.4f} tp3={self.target_price:.4f}"
        )

    def _equity_usdt(self, cfg: SqueezeBreakoutConfig) -> float:
        base = float(cfg.equity_usdt or 100.0)
        if not cfg.compound:
            return base
        try:
            from accumulation_radar import init_db
            from quant.common.vnpy_wallet import lane_equity_usdt, migrate_vnpy_lane_tables

            conn = init_db()
            try:
                cur = conn.cursor()
                migrate_vnpy_lane_tables(cur)
                return lane_equity_usdt(
                    cfg,
                    symbol_from_vt(self.vt_symbol),
                    cur=cur,
                )
            finally:
                conn.close()
        except Exception:
            return base

    def _refresh_compound_size(self) -> None:
        cfg = self._cfg()
        if not cfg.compound:
            return
        from quant.market import fetch_mark_price, resolve_market_data_exchange_id

        sym = symbol_from_vt(self.vt_symbol)
        md = resolve_market_data_exchange_id(cfg.market_data_exchange)
        px = fetch_mark_price(sym, exchange_id=md) or 100.0
        eq = self._equity_usdt(cfg)
        stop_dist = abs(float(self.entry_price or px) - float(self.stop_price or px * 0.985))
        if stop_dist <= 0:
            stop_dist = px * 0.015
        vol = size_for_breakout(cfg, px, stop_distance=stop_dist, equity_usdt=eq)
        if vol > 0 and abs(float(self.fixed_size) - vol) >= 1e-6:
            self.fixed_size = vol

    def on_init(self) -> None:
        self.write_log("Smart Breakout init")
        mins = max(1, int(self.signal_minutes))
        self.bg = BarGenerator(self.on_bar, mins, self.on_signal_bar, Interval.MINUTE)
        self._preload_signal_bars()

    def _preload_signal_bars(self) -> None:
        cfg = self._cfg()
        sym = symbol_from_vt(self.vt_symbol)
        days = max(7, int(cfg.init_bar_days))
        interval = f"{max(1, int(cfg.signal_minutes))}m"
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 86_400_000
        md = resolve_market_data_exchange_id(cfg.market_data_exchange)
        try:
            rows = fetch_klines_forward(sym, interval, start_ms, end_ms, exchange_id=md)
        except Exception as exc:
            self.write_log(f"preload {interval} klines failed: {exc}")
            return
        df = klines_to_df(rows)
        if df.empty:
            self.write_log(f"preload {interval} klines empty")
            return
        for _, row in df.iterrows():
            dt = datetime.fromtimestamp(int(row["open_time"]) / 1000, tz=timezone.utc)
            bar = BarData(
                symbol=sym,
                exchange=None,
                datetime=dt,
                interval=Interval.MINUTE,
                open_price=float(row["open"]),
                high_price=float(row["high"]),
                low_price=float(row["low"]),
                close_price=float(row["close"]),
                volume=float(row["volume"]),
                gateway_name="",
            )
            self._append_signal_bar(bar)
        self._rebuild_engine_state()
        self.write_log(f"preloaded {len(self._signal_bars)} x {interval} bars")

    def on_start(self) -> None:
        self.write_log("Smart Breakout start")

    def on_stop(self) -> None:
        if self.pos != 0:
            self._flatten_market()
        self.write_log("Smart Breakout stop")

    def on_tick(self, tick: TickData) -> None:
        extra = getattr(tick, "extra", None) or {}
        bar = extra.get("bar")
        if bar is not None:
            self.on_bar(bar)
            return
        if self.bg is not None:
            self.bg.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        if self.bg is not None:
            self.bg.update_bar(bar)

    def on_signal_bar(self, bar: BarData) -> None:
        self._last_signal_bar = bar
        self._append_signal_bar(bar)
        cfg = self._cfg()
        self._engine_state, signal = process_signal_bar(
            self._signal_bars,
            self._engine_state,
            **self._engine_kwargs(cfg),
        )
        if not self.trading:
            self._prev_bar_high = float(bar.high_price)
            self._prev_bar_low = float(bar.low_price)
            return

        high = float(bar.high_price)
        low = float(bar.low_price)

        if self.pos != 0:
            side = 1 if self.pos > 0 else -1
            if signal is not None:
                self.write_log("early exit: new breakout signal while trade open")
                self._pending_signal = signal
                self._flatten_market()
                self.put_event()
                self._prev_bar_high = high
                self._prev_bar_low = low
                return
            hit = bar_exit_reason(
                side=side,
                high=high,
                low=low,
                stop=float(self.stop_price),
                tp1=float(self.tp1_price),
                tp2=float(self.tp2_price),
                tp3=float(self.target_price),
                prev_high=self._prev_bar_high,
                prev_low=self._prev_bar_low,
                tp1_hit=self._tp1_hit,
                tp2_hit=self._tp2_hit,
            )
            if hit is None:
                for milestone in bar_tp_milestones(
                    side=side,
                    high=high,
                    low=low,
                    tp1=float(self.tp1_price),
                    tp2=float(self.tp2_price),
                    prev_high=self._prev_bar_high,
                    prev_low=self._prev_bar_low,
                    tp1_hit=self._tp1_hit,
                    tp2_hit=self._tp2_hit,
                ):
                    if milestone == "tp1":
                        self._tp1_hit = True
                    elif milestone == "tp2":
                        self._tp2_hit = True
                    self.write_log(f"milestone {milestone} px={bar.close_price:.4f}")
            if hit:
                self.write_log(f"exit {hit} px={bar.close_price:.4f}")
                self._flatten_market()
                self.put_event()
                self._prev_bar_high = high
                self._prev_bar_low = low
                return
            self.put_event()
            self._prev_bar_high = high
            self._prev_bar_low = low
            return

        if signal is None or self._entry_pending:
            self._prev_bar_high = high
            self._prev_bar_low = low
            self.put_event()
            return

        stop_dist = abs(signal.entry - signal.stop)
        eq = self._equity_usdt(cfg)
        vol = size_for_breakout(cfg, signal.entry, stop_distance=stop_dist, equity_usdt=eq)
        if vol <= 0:
            self._prev_bar_high = high
            self._prev_bar_low = low
            return
        self.fixed_size = vol
        self._apply_levels(signal)
        risk_usd = risk_budget_usdt(cfg, equity_usdt=eq)
        side_s = "LONG" if signal.side > 0 else "SHORT"
        self.write_log(
            f"Breakout {self.vt_symbol} {side_s} px={signal.entry:.4f} "
            f"range=[{signal.range_bottom:.4f},{signal.range_top:.4f}] "
            f"strength={signal.strength} risk=${risk_usd:.2f} vol={vol}"
        )
        try:
            from quant.engine.strategy_signals import LANE_SQUEEZE_BREAKOUT, record_strategy_open_signal

            record_strategy_open_signal(
                lane=LANE_SQUEEZE_BREAKOUT,
                symbol=symbol_from_vt(self.vt_symbol),
                side=side_s,
                entry_price=signal.entry,
                sl_price=signal.stop,
                tp_price=signal.tp3,
                status="shadow" if (cfg.shadow or not cfg.live_enabled) else "emitted",
                bar_ms=int(bar.datetime.timestamp() * 1000),
                detail={
                    "vol": vol,
                    "risk_usd": risk_usd,
                    "strength": signal.strength,
                    "tp1": signal.tp1,
                    "tp2": signal.tp2,
                    "tp3": signal.tp3,
                    "squeeze_bars": signal.squeeze_bars,
                },
            )
        except Exception as exc:
            self.write_log(f"strategy signal persist failed: {exc}")
        self._open_market(signal.side, vol)
        self._prev_bar_high = high
        self._prev_bar_low = low
        self.put_event()

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

    def _open_market(self, side: int, vol: float) -> None:
        if side > 0:
            oids = self._send_market(Direction.LONG, Offset.OPEN, vol)
        else:
            oids = self._send_market(Direction.SHORT, Offset.OPEN, vol)
        if oids:
            self._entry_pending = True
            return
        cfg = self._cfg()
        if cfg.shadow or not cfg.live_enabled:
            self.write_log(f"signal-only {self.vt_symbol} (shadow={cfg.shadow} live={cfg.live_enabled})")
            return
        self.write_log(f"live entry rejected {self.vt_symbol}")

    def _flatten_market(self) -> None:
        if self.pos == 0 or self._exit_pending:
            return
        self.cancel_all()
        vol = abs(float(self.pos))
        oids: List[str] = []
        if self.pos > 0:
            oids = self._send_market(Direction.SHORT, Offset.CLOSE, vol)
        elif self.pos < 0:
            oids = self._send_market(Direction.LONG, Offset.CLOSE, vol)
        if oids:
            self._exit_pending = True

    def on_trade(self, trade: TradeData) -> None:
        if trade.offset == Offset.OPEN and self.pos != 0:
            side = 1 if self.pos > 0 else -1
            if self._range_top > self._range_bottom and self._last_atr > 0:
                self._levels_from_range(
                    float(trade.price),
                    side,
                    self._range_top,
                    self._range_bottom,
                    self._last_atr,
                )
            else:
                self.entry_price = float(trade.price)
            self._entry_pending = False
            self._tp1_hit = False
            self._tp2_hit = False
            self._prev_bar_high = float(trade.price)
            self._prev_bar_low = float(trade.price)
        if self.pos == 0:
            self.cancel_all()
            self._entry_pending = False
            self._exit_pending = False
            self.entry_price = 0.0
            self.stop_price = 0.0
            self.tp1_price = 0.0
            self.tp2_price = 0.0
            self.target_price = 0.0
            self._range_top = 0.0
            self._range_bottom = 0.0
            self._last_atr = 0.0
            self._tp1_hit = False
            self._tp2_hit = False
            self._refresh_compound_size()
            pending = self._pending_signal
            self._pending_signal = None
            if pending is not None and self.trading:
                cfg = self._cfg()
                stop_dist = abs(pending.entry - pending.stop)
                eq = self._equity_usdt(cfg)
                vol = size_for_breakout(cfg, pending.entry, stop_distance=stop_dist, equity_usdt=eq)
                if vol > 0:
                    self.fixed_size = vol
                    self._apply_levels(pending)
                    self._open_market(pending.side, vol)
        self.put_event()

    def on_order(self, order: OrderData) -> None:
        from vnpy.trader.constant import Status

        if order.status not in (Status.CANCELLED, Status.REJECTED):
            return
        if order.offset == Offset.OPEN and self.pos == 0:
            self._entry_pending = False
        if order.offset == Offset.CLOSE:
            self._exit_pending = False
        self.put_event()

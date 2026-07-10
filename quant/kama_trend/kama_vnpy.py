"""KAMA Trend Following vnpy 策略（Jesse kama-trendfollowing）。"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Optional

from quant.engine.bootstrap import ensure_vnpy_path
from quant.engine.exchanges.registry import symbol_from_vt
from quant.common.tf_klines import fetch_tf_closes
from quant.kama_trend.config import KamaTrendConfig
from quant.kama_trend.core import (
    BarOhlc,
    bar_hits_stop_tp,
    compute_snapshot,
    entry_signal,
    stop_tp_prices,
)
from quant.kama_trend.sizing import risk_budget_usdt, size_for_kama
from quant.market import fetch_klines_forward, klines_to_df, resolve_market_data_exchange_id

ensure_vnpy_path()

from vnpy.trader.constant import Direction, Interval, Offset, OrderType  # noqa: E402
from vnpy.trader.object import BarData, OrderData, TickData, TradeData  # noqa: E402
from vnpy.trader.utility import round_to  # noqa: E402
from vnpy_ctastrategy import BarGenerator, CtaTemplate  # noqa: E402


class KamaTrendVnpyStrategy(CtaTemplate):
    """KAMA + ADX + CHOP + BB 宽度过滤；15m 信号，4h KAMA 趋势确认。"""

    author = "next-k-api"

    kama_period: int = 14
    adx_period: int = 14
    chop_period: int = 14
    bb_period: int = 20
    adx_min: float = 50.0
    chop_max: float = 50.0
    bb_width_max_pct: float = 7.0
    cooldown_bars: int = 10
    stop_atr: float = 2.5
    tp_atr: float = 2.5
    signal_minutes: int = 15
    fixed_size: float = 1.0
    orb_rth_only: bool = False

    parameters = [
        "kama_period",
        "adx_period",
        "chop_period",
        "bb_period",
        "adx_min",
        "chop_max",
        "bb_width_max_pct",
        "cooldown_bars",
        "stop_atr",
        "tp_atr",
        "signal_minutes",
        "fixed_size",
        "orb_rth_only",
    ]
    variables = ["stop_price", "target_price", "entry_price"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.entry_price: float = 0.0
        self.stop_price: float = 0.0
        self.target_price: float = 0.0
        self._signal_bars: List[BarOhlc] = []
        self._bar_index: int = 0
        self._last_trade_bar_index: int = -10_000
        self._last_signal_bar: Optional[BarData] = None
        self._exit_pending: bool = False
        self._entry_pending: bool = False
        self._prev_bar_high: float = 0.0
        self._prev_bar_low: float = 0.0
        self._long_tf_closes: List[float] = []
        self.bg: BarGenerator | None = None

    @classmethod
    def from_kama_config(cls, cfg: KamaTrendConfig) -> dict:
        return {
            "kama_period": int(cfg.kama_period),
            "adx_period": int(cfg.adx_period),
            "chop_period": int(cfg.chop_period),
            "bb_period": int(cfg.bb_period),
            "adx_min": float(cfg.adx_min),
            "chop_max": float(cfg.chop_max),
            "bb_width_max_pct": float(cfg.bb_width_max_pct),
            "cooldown_bars": int(cfg.cooldown_bars),
            "stop_atr": float(cfg.stop_atr),
            "tp_atr": float(cfg.tp_atr),
            "signal_minutes": int(cfg.signal_minutes),
            "orb_rth_only": False,
        }

    def _cfg(self) -> KamaTrendConfig:
        return KamaTrendConfig.from_env()

    def _append_signal_bar(self, bar: BarData) -> None:
        ts = int(bar.datetime.timestamp() * 1000)
        row: BarOhlc = (
            ts,
            float(bar.open_price),
            float(bar.high_price),
            float(bar.low_price),
            float(bar.close_price),
        )
        if self._signal_bars and self._signal_bars[-1][0] == ts:
            self._signal_bars[-1] = row
        else:
            self._signal_bars.append(row)
            self._bar_index += 1
        max_bars = (24 * 4) * max(40, self._cfg().init_bar_days)
        if len(self._signal_bars) > max_bars:
            trim = len(self._signal_bars) - max_bars
            self._signal_bars = self._signal_bars[trim:]
            self._bar_index = max(0, self._bar_index - trim)
            self._last_trade_bar_index = max(-10_000, self._last_trade_bar_index - trim)

    def _snapshot(self):
        return compute_snapshot(
            self._signal_bars,
            long_tf_closes=self._long_tf_closes,
            kama_period=int(self.kama_period),
            adx_period=int(self.adx_period),
            chop_period=int(self.chop_period),
            bb_period=int(self.bb_period),
        )

    def _refresh_long_tf_closes(self) -> None:
        cfg = self._cfg()
        sym = symbol_from_vt(self.vt_symbol)
        md = resolve_market_data_exchange_id(cfg.market_data_exchange)
        days = max(30, int(cfg.init_bar_days) * 2)
        self._long_tf_closes = fetch_tf_closes(sym, "4h", days=days, exchange_id=md)

    def _apply_stop_target(self, entry: float, side: int) -> None:
        snap = self._snapshot()
        if snap is None:
            return
        stop, tp = stop_tp_prices(
            entry,
            side,
            snap.atr,
            stop_atr=float(self.stop_atr),
            tp_atr=float(self.tp_atr),
        )
        self.entry_price = float(entry)
        self.stop_price = float(stop)
        self.target_price = float(tp)

    def restore_synced_position(self, *, entry_px: float, pos: float) -> None:
        if abs(float(pos or 0.0)) < 1e-12:
            return
        side = 1 if float(pos) > 0 else -1
        self._apply_stop_target(float(entry_px), side)
        self.write_log(
            f"restored synced position entry={entry_px:.4f} stop={self.stop_price:.4f} tp={self.target_price:.4f}"
        )

    def _equity_usdt(self, cfg: KamaTrendConfig) -> float:
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
                return lane_equity_usdt(cfg, symbol_from_vt(self.vt_symbol), cur=cur)
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
        snap = self._snapshot()
        if snap is None or snap.atr <= 0:
            return
        stop, _ = stop_tp_prices(
            px,
            1,
            snap.atr,
            stop_atr=float(self.stop_atr),
            tp_atr=float(self.tp_atr),
        )
        stop_dist = abs(px - stop)
        vol = size_for_kama(cfg, px, stop_distance=stop_dist, equity_usdt=eq)
        if vol > 0 and abs(float(self.fixed_size) - vol) >= 1e-6:
            self.fixed_size = vol

    def on_init(self) -> None:
        self.write_log("KAMA Trend init")
        mins = max(1, int(self.signal_minutes))
        self.bg = BarGenerator(self.on_bar, mins, self.on_signal_bar, Interval.MINUTE)
        self._refresh_long_tf_closes()
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
        self.write_log(f"preloaded {len(self._signal_bars)} x {interval} bars")

    def on_start(self) -> None:
        self.write_log("KAMA Trend start")

    def on_stop(self) -> None:
        if self.pos != 0:
            self._flatten_market()
        self.write_log("KAMA Trend stop")

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
        self._refresh_long_tf_closes()
        high = float(bar.high_price)
        low = float(bar.low_price)
        if not self.trading:
            self._prev_bar_high = high
            self._prev_bar_low = low
            return

        snap = self._snapshot()
        if snap is None:
            self._prev_bar_high = high
            self._prev_bar_low = low
            return

        if self.pos != 0:
            side = 1 if self.pos > 0 else -1
            hit = bar_hits_stop_tp(
                side=side,
                high=high,
                low=low,
                stop=float(self.stop_price),
                tp=float(self.target_price),
                prev_high=self._prev_bar_high,
                prev_low=self._prev_bar_low,
            )
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

        bars_since = self._bar_index - self._last_trade_bar_index
        sig = entry_signal(
            float(bar.close_price),
            snap,
            adx_min=float(self.adx_min),
            chop_max=float(self.chop_max),
            bb_width_max_pct=float(self.bb_width_max_pct),
            bars_since_trade=bars_since,
            cooldown_bars=int(self.cooldown_bars),
        )
        if sig == 0 or self._entry_pending:
            self._prev_bar_high = high
            self._prev_bar_low = low
            self.put_event()
            return

        cfg = self._cfg()
        close = float(bar.close_price)
        stop, tp = stop_tp_prices(close, sig, snap.atr, stop_atr=float(self.stop_atr), tp_atr=float(self.tp_atr))
        stop_dist = abs(close - stop)
        eq = self._equity_usdt(cfg)
        vol = size_for_kama(cfg, close, stop_distance=stop_dist, equity_usdt=eq)
        if vol <= 0:
            self._prev_bar_high = high
            self._prev_bar_low = low
            return
        self.fixed_size = vol
        self.entry_price = close
        self.stop_price = stop
        self.target_price = tp
        risk_usd = risk_budget_usdt(cfg, equity_usdt=eq)
        side_s = "LONG" if sig > 0 else "SHORT"
        self.write_log(
            f"KAMA signal {self.vt_symbol} {side_s} px={close:.4f} adx={snap.adx:.1f} "
            f"chop={snap.chop:.1f} bbw={snap.bb_width_pct:.2f}% risk=${risk_usd:.2f} vol={vol}"
        )
        try:
            from quant.engine.strategy_signals import LANE_KAMA_TREND, record_strategy_open_signal

            record_strategy_open_signal(
                lane=LANE_KAMA_TREND,
                symbol=symbol_from_vt(self.vt_symbol),
                side=side_s,
                entry_price=close,
                sl_price=stop,
                tp_price=tp,
                status="shadow" if (cfg.shadow or not cfg.live_enabled) else "emitted",
                bar_ms=int(bar.datetime.timestamp() * 1000),
                detail={
                    "vol": vol,
                    "risk_usd": risk_usd,
                    "adx": snap.adx,
                    "chop": snap.chop,
                    "bb_width_pct": snap.bb_width_pct,
                },
            )
        except Exception as exc:
            self.write_log(f"strategy signal persist failed: {exc}")
        self._open_market(sig, vol)
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
            self._apply_stop_target(float(trade.price), side)
            self._entry_pending = False
            self._prev_bar_high = float(trade.price)
            self._prev_bar_low = float(trade.price)
        if self.pos == 0:
            self.cancel_all()
            self._entry_pending = False
            self._exit_pending = False
            self._last_trade_bar_index = self._bar_index
            self.entry_price = 0.0
            self.stop_price = 0.0
            self.target_price = 0.0
            self._refresh_compound_size()
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

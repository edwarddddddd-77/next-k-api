"""MtfMomo2xA vnpy 策略（Jesse MtfMomo2xA 移植）。"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Optional

from quant.engine.bootstrap import ensure_vnpy_path
from quant.engine.exchanges.registry import symbol_from_vt
from quant.market import fetch_klines_forward, klines_to_df, resolve_market_data_exchange_id
from quant.common.tf_klines import fetch_tf_closes
from quant.mtfmomo.config import MtfMomoConfig
from quant.mtfmomo.core import (
    HourOhlc,
    bar_hits_stop_tp,
    compute_levels,
    entry_signal,
    should_ema_exit,
    stop_tp_prices,
)
from quant.mtfmomo.sizing import risk_budget_usdt, size_for_momo

ensure_vnpy_path()

from vnpy.trader.constant import Direction, Interval, Offset, OrderType  # noqa: E402
from vnpy.trader.object import BarData, OrderData, TickData, TradeData  # noqa: E402
from vnpy.trader.utility import round_to  # noqa: E402
from vnpy_ctastrategy import BarGenerator, CtaTemplate  # noqa: E402


class MtfMomoVnpyStrategy(CtaTemplate):
    """Dual-anchor (4h+1D) 1h Donchian momentum — Jesse mtfmomo2xa."""

    author = "next-k-api"

    entry_lb: int = 26
    ema_exit: int = 35
    ema_4h: int = 21
    ema_1d: int = 16
    stop_atr: float = 3.295829874337854
    tp_atr: float = 8.681332636811806
    fixed_size: float = 1.0
    orb_rth_only: bool = False

    parameters = [
        "entry_lb",
        "ema_exit",
        "ema_4h",
        "ema_1d",
        "stop_atr",
        "tp_atr",
        "fixed_size",
        "orb_rth_only",
    ]
    variables = ["stop_price", "target_price", "entry_price"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.entry_price: float = 0.0
        self.stop_price: float = 0.0
        self.target_price: float = 0.0
        self._hour_bars: List[HourOhlc] = []
        self._anchor_4h_closes: List[float] = []
        self._anchor_1d_closes: List[float] = []
        self._last_hour_bar: Optional[BarData] = None
        self._exit_pending: bool = False
        self._entry_pending: bool = False
        self.bg: BarGenerator | None = None

    @classmethod
    def from_mtfmomo_config(cls, cfg: MtfMomoConfig) -> dict:
        return {
            "entry_lb": int(cfg.entry_lb),
            "ema_exit": int(cfg.ema_exit),
            "ema_4h": int(cfg.ema_4h),
            "ema_1d": int(cfg.ema_1d),
            "stop_atr": float(cfg.stop_atr),
            "tp_atr": float(cfg.tp_atr),
            "orb_rth_only": False,
        }

    def _cfg(self) -> MtfMomoConfig:
        return MtfMomoConfig.from_env()

    def _append_hour(self, bar: BarData) -> None:
        ts = int(bar.datetime.timestamp() * 1000)
        row: HourOhlc = (
            ts,
            float(bar.open_price),
            float(bar.high_price),
            float(bar.low_price),
            float(bar.close_price),
        )
        if self._hour_bars and self._hour_bars[-1][0] == ts:
            self._hour_bars[-1] = row
            return
        self._hour_bars.append(row)
        max_bars = 24 * max(40, self._cfg().init_bar_days)
        if len(self._hour_bars) > max_bars:
            self._hour_bars = self._hour_bars[-max_bars:]

    def _refresh_anchor_closes(self) -> None:
        cfg = self._cfg()
        sym = symbol_from_vt(self.vt_symbol)
        md = resolve_market_data_exchange_id(cfg.market_data_exchange)
        days = max(60, int(cfg.init_bar_days) * 3)
        self._anchor_4h_closes = fetch_tf_closes(sym, "4h", days=days, exchange_id=md)
        self._anchor_1d_closes = fetch_tf_closes(sym, "1d", days=days, exchange_id=md)

    def _levels(self):
        return compute_levels(
            self._hour_bars,
            entry_lb=int(self.entry_lb),
            ema_exit=int(self.ema_exit),
            ema_4h=int(self.ema_4h),
            ema_1d=int(self.ema_1d),
            anchor_4h_closes=self._anchor_4h_closes,
            anchor_1d_closes=self._anchor_1d_closes,
        )

    def _apply_stop_target(self, entry: float, side: int) -> None:
        levels = self._levels()
        if levels is None:
            return
        stop, tp = stop_tp_prices(
            entry,
            side,
            levels,
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

    def _equity_usdt(self, cfg: MtfMomoConfig) -> float:
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
        levels = self._levels()
        if levels is None:
            return
        stop, _ = stop_tp_prices(
            px,
            1,
            levels,
            stop_atr=float(self.stop_atr),
            tp_atr=float(self.tp_atr),
        )
        stop_dist = abs(px - stop)
        vol = size_for_momo(cfg, px, stop_distance=stop_dist, equity_usdt=eq)
        if vol > 0 and abs(float(self.fixed_size) - vol) >= 1e-6:
            self.fixed_size = vol

    def on_init(self) -> None:
        self.write_log("MtfMomo2xA init")
        self.bg = BarGenerator(self.on_bar, 1, self.on_hour_bar, Interval.HOUR)
        self._refresh_anchor_closes()
        self._preload_hour_bars()

    def _preload_hour_bars(self) -> None:
        cfg = self._cfg()
        sym = symbol_from_vt(self.vt_symbol)
        days = max(7, int(cfg.init_bar_days))
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 86_400_000
        md = resolve_market_data_exchange_id(cfg.market_data_exchange)
        try:
            rows = fetch_klines_forward(sym, "1h", start_ms, end_ms, exchange_id=md)
        except Exception as exc:
            self.write_log(f"preload 1h klines failed: {exc}")
            return
        df = klines_to_df(rows)
        if df.empty:
            self.write_log("preload 1h klines empty")
            return
        for _, row in df.iterrows():
            dt = datetime.fromtimestamp(int(row["open_time"]) / 1000, tz=timezone.utc)
            bar = BarData(
                symbol=sym,
                exchange=None,
                datetime=dt,
                interval=Interval.HOUR,
                open_price=float(row["open"]),
                high_price=float(row["high"]),
                low_price=float(row["low"]),
                close_price=float(row["close"]),
                volume=float(row["volume"]),
                gateway_name="",
            )
            self._append_hour(bar)
        self.write_log(f"preloaded {len(self._hour_bars)} x 1h bars")

    def on_start(self) -> None:
        self.write_log("MtfMomo2xA start")

    def on_stop(self) -> None:
        if self.pos != 0 and self._last_hour_bar is not None:
            self._flatten_market()
        self.write_log("MtfMomo2xA stop")

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

    def on_hour_bar(self, bar: BarData) -> None:
        self._last_hour_bar = bar
        self._append_hour(bar)
        self._refresh_anchor_closes()
        if not self.trading:
            return

        levels = self._levels()
        if levels is None:
            return

        if self.pos != 0:
            side = 1 if self.pos > 0 else -1
            hit = bar_hits_stop_tp(
                side=side,
                high=float(bar.high_price),
                low=float(bar.low_price),
                stop=float(self.stop_price),
                tp=float(self.target_price),
            )
            if hit:
                self.write_log(f"exit {hit} px={bar.close_price:.4f}")
                self._flatten_market()
                self.put_event()
                return
            if should_ema_exit(float(bar.close_price), side, levels.ema_exit):
                self.write_log(f"exit ema px={bar.close_price:.4f} ema={levels.ema_exit:.4f}")
                self._flatten_market()
                self.put_event()
                return
            self.put_event()
            return

        sig = entry_signal(float(bar.close_price), levels)
        if sig == 0 or self._entry_pending:
            self.put_event()
            return

        cfg = self._cfg()
        close = float(bar.close_price)
        stop, tp = stop_tp_prices(close, sig, levels, stop_atr=float(self.stop_atr), tp_atr=float(self.tp_atr))
        stop_dist = abs(close - stop)
        eq = self._equity_usdt(cfg)
        vol = size_for_momo(cfg, close, stop_distance=stop_dist, equity_usdt=eq)
        if vol <= 0:
            return
        self.fixed_size = vol
        self.entry_price = close
        self.stop_price = stop
        self.target_price = tp
        risk_usd = risk_budget_usdt(cfg, equity_usdt=eq)
        side_s = "LONG" if sig > 0 else "SHORT"
        self.write_log(
            f"Momo signal {self.vt_symbol} {side_s} px={close:.4f} risk=${risk_usd:.2f} vol={vol} "
            f"stop={stop:.4f} tp={tp:.4f}"
        )
        try:
            from quant.engine.strategy_signals import LANE_MTFMOMO, record_strategy_open_signal

            record_strategy_open_signal(
                lane=LANE_MTFMOMO,
                symbol=symbol_from_vt(self.vt_symbol),
                side=side_s,
                entry_price=close,
                sl_price=stop,
                tp_price=tp,
                status="shadow" if (cfg.shadow or not cfg.live_enabled) else "emitted",
                bar_ms=int(bar.datetime.timestamp() * 1000),
                detail={"vol": vol, "risk_usd": risk_usd, "agree": levels.trend_agree},
            )
        except Exception as exc:
            self.write_log(f"strategy signal persist failed: {exc}")
        self._open_market(sig, vol)
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
        if self.pos == 0:
            self.cancel_all()
            self._entry_pending = False
            self._exit_pending = False
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

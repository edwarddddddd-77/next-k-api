"""Donchian breakout vnpy strategy (breakoutscanner-aligned)."""

from __future__ import annotations

from quant.breakout_donchian.bars import BarRow, drop_incomplete_bars, fetch_bars
from quant.breakout_donchian.config import BreakoutDonchianConfig
from quant.breakout_donchian.core import bar_exit_reason, detect_donchian_signal
from quant.breakout_donchian.resonance import evaluate_resonance, preload_days_for_interval, weekly_bars_from_daily
from quant.breakout_donchian.sizing import risk_budget_usdt, size_for_donchian
from quant.engine.bootstrap import ensure_vnpy_path
from quant.engine.exchanges.registry import symbol_from_vt
from quant.market import resolve_market_data_exchange_id
from typing import List, Optional

ensure_vnpy_path()

from vnpy.trader.constant import Direction, Interval, Offset, OrderType  # noqa: E402
from vnpy.trader.object import BarData, OrderData, TickData, TradeData  # noqa: E402
from vnpy.trader.utility import round_to  # noqa: E402
from vnpy_ctastrategy import BarGenerator, CtaTemplate  # noqa: E402


class BreakoutDonchianVnpyStrategy(CtaTemplate):
    """1D calendar bar execute + 1W W-SUN confirm + 1H bonus sizing."""

    author = "next-k-api"

    lookback: int = 20
    vol_lookback: int = 20
    vol_mult: float = 1.30
    strong_close_pct: float = 0.60
    strict_vol_mult: float = 1.6
    strict_atr_mult: float = 1.3
    atr_period: int = 14
    breakout_mode: str = "strict"
    tp1_rr: float = 2.0
    tp2_rr: float = 3.5
    tp3_rr: float = 3.5
    long_only: bool = True
    fixed_size: float = 1.0

    parameters = [
        "lookback",
        "vol_lookback",
        "vol_mult",
        "strong_close_pct",
        "strict_vol_mult",
        "strict_atr_mult",
        "atr_period",
        "breakout_mode",
        "tp1_rr",
        "tp2_rr",
        "tp3_rr",
        "long_only",
        "fixed_size",
    ]
    variables = ["stop_price", "target_price", "entry_price", "tp1_price", "tp2_price"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.entry_price: float = 0.0
        self.stop_price: float = 0.0
        self.tp1_price: float = 0.0
        self.tp2_price: float = 0.0
        self.target_price: float = 0.0
        self._signal_bars: List[BarRow] = []
        self._weekly_bars: List[BarRow] = []
        self._hourly_bars: List[BarRow] = []
        self._prev_bar_high: float = 0.0
        self._prev_bar_low: float = 0.0
        self._entry_pending: bool = False
        self._exit_pending: bool = False
        self._pending_side: int = 0
        self._last_signal_bar: Optional[BarData] = None
        self.bg: BarGenerator | None = None

    @classmethod
    def from_donchian_config(cls, cfg: BreakoutDonchianConfig) -> dict:
        return {
            "lookback": int(cfg.lookback),
            "vol_lookback": int(cfg.vol_lookback),
            "vol_mult": float(cfg.vol_mult),
            "strong_close_pct": float(cfg.strong_close_pct),
            "strict_vol_mult": float(cfg.strict_vol_mult),
            "strict_atr_mult": float(cfg.strict_atr_mult),
            "atr_period": int(cfg.atr_period),
            "breakout_mode": str(cfg.breakout_mode),
            "tp1_rr": float(cfg.tp1_rr),
            "tp2_rr": float(cfg.tp2_rr),
            "tp3_rr": float(cfg.tp3_rr),
            "long_only": bool(cfg.long_only),
        }

    def _cfg(self) -> BreakoutDonchianConfig:
        return BreakoutDonchianConfig.from_env()

    def _bar_row(self, bar: BarData) -> BarRow:
        return (
            int(bar.datetime.timestamp() * 1000),
            float(bar.open_price),
            float(bar.high_price),
            float(bar.low_price),
            float(bar.close_price),
            float(bar.volume or 0.0),
        )

    def _append_signal_bar(self, bar: BarData) -> None:
        row = self._bar_row(bar)
        cfg = self._cfg()
        max_bars = max(200, int(cfg.init_bar_days) + self.lookback + 30)
        if self._signal_bars and self._signal_bars[-1][0] == row[0]:
            self._signal_bars[-1] = row
        else:
            self._signal_bars.append(row)
        if len(self._signal_bars) > max_bars:
            self._signal_bars = self._signal_bars[-max_bars:]

    def _detect_signal(self, cfg: BreakoutDonchianConfig):
        direction = "bullish" if cfg.long_only else None
        mode = "strict" if str(cfg.breakout_mode).lower() == "strict" else "standard"
        bars = drop_incomplete_bars(self._signal_bars, "1d")
        return detect_donchian_signal(
            bars,
            lookback=int(self.lookback),
            vol_lookback=int(self.vol_lookback),
            vol_mult=float(self.vol_mult),
            strong_close_pct=float(self.strong_close_pct),
            mode=mode,
            strict_vol_mult=float(self.strict_vol_mult),
            strict_atr_mult=float(self.strict_atr_mult),
            atr_period=int(self.atr_period),
            direction_filter=direction,
            tp1_rr=float(self.tp1_rr),
            tp2_rr=float(self.tp2_rr),
            tp3_rr=float(self.tp3_rr),
            sl_atr_mult=float(cfg.sl_atr_mult),
            sl_level_buffer=float(cfg.sl_level_buffer),
        )

    def _refresh_market_bars(self, cfg: BreakoutDonchianConfig) -> None:
        sym = symbol_from_vt(self.vt_symbol)
        md = resolve_market_data_exchange_id(cfg.market_data_exchange)
        days = max(30, int(cfg.init_bar_days))
        try:
            daily = fetch_bars(sym, "1d", days=days, exchange_id=md)
            if daily:
                self._signal_bars = daily[-max(200, days + 30) :]
            self._weekly_bars = weekly_bars_from_daily(self._signal_bars)
            if cfg.check_1h_bonus:
                hdays = preload_days_for_interval("1h", min_bars=cfg.hourly_lookback + cfg.hourly_vol_lookback + 20)
                self._hourly_bars = fetch_bars(sym, "1h", days=hdays, exchange_id=md)
        except Exception as exc:
            self.write_log(f"refresh market bars failed: {exc}")

    def _apply_levels(self, entry: float, side: int, stop: float, tp1: float, tp2: float, tp3: float) -> None:
        self.entry_price = float(entry)
        self.stop_price = float(stop)
        self.tp1_price = float(tp1)
        self.tp2_price = float(tp2)
        self.target_price = float(tp3)

    def restore_synced_position(self, *, entry_px: float, pos: float) -> None:
        if abs(float(pos or 0.0)) < 1e-12:
            return
        side = 1 if float(pos) > 0 else -1
        px = float(entry_px)
        risk = max(px * 0.02, 1.0)
        stop = px - risk if side > 0 else px + risk
        tp1 = px + risk * float(self.tp1_rr) if side > 0 else px - risk * float(self.tp1_rr)
        tp2 = px + risk * float(self.tp2_rr) if side > 0 else px - risk * float(self.tp2_rr)
        tp3 = px + risk * float(self.tp3_rr) if side > 0 else px - risk * float(self.tp3_rr)
        self._apply_levels(px, side, stop, tp1, tp2, tp3)
        self.write_log(f"restored synced position entry={px:.4f} stop={stop:.4f} tp3={tp3:.4f}")

    def _equity_usdt(self, cfg: BreakoutDonchianConfig) -> float:
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

    def on_init(self) -> None:
        self.write_log("Donchian Breakout init (1D exec + 1W confirm)")
        self.bg = BarGenerator(self.on_bar, 1, self.on_signal_bar, Interval.DAILY)
        cfg = self._cfg()
        self._refresh_market_bars(cfg)
        self.write_log(
            f"preloaded daily={len(self._signal_bars)} weekly={len(self._weekly_bars)} hourly={len(self._hourly_bars)}"
        )

    def on_start(self) -> None:
        self.write_log("Donchian Breakout start")

    def on_stop(self) -> None:
        if self.pos != 0:
            self._flatten_market()
        self.write_log("Donchian Breakout stop")

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

    def _try_enter(self, cfg: BreakoutDonchianConfig, signal, bar: BarData) -> None:
        sym = symbol_from_vt(self.vt_symbol)
        resonance = evaluate_resonance(
            cfg,
            weekly_bars=self._weekly_bars,
            hourly_bars=self._hourly_bars if cfg.check_1h_bonus else None,
        )
        if cfg.require_weekly_confirm and not resonance.weekly_ok:
            self.write_log(f"skip {sym}: 1D signal but 1W confirm missing")
            return

        risk_mult = resonance.risk_mult if resonance.risk_mult > 0 else cfg.risk_mult_base
        stop_dist = abs(signal.entry - signal.stop)
        eq = self._equity_usdt(cfg)
        vol = size_for_donchian(cfg, signal.entry, stop_distance=stop_dist, equity_usdt=eq, risk_mult=risk_mult, symbol=sym)
        if vol <= 0:
            return

        self.fixed_size = vol
        self._apply_levels(signal.entry, signal.side, signal.stop, signal.tp1, signal.tp2, signal.tp3)
        risk_usd = risk_budget_usdt(cfg, equity_usdt=eq, risk_mult=risk_mult)
        side_s = "LONG" if signal.side > 0 else "SHORT"
        self.write_log(
            f"Donchian {self.vt_symbol} {side_s} px={signal.entry:.4f} "
            f"tier={resonance.tier} 1W={resonance.weekly_ok} 1H_bonus={resonance.hourly_bonus} "
            f"risk_mult={risk_mult:.2f} risk=${risk_usd:.2f} vol={vol}"
        )
        try:
            from quant.engine.strategy_signals import LANE_BREAKOUT_DONCHIAN, record_strategy_open_signal

            record_strategy_open_signal(
                lane=LANE_BREAKOUT_DONCHIAN,
                symbol=sym,
                side=side_s,
                entry_price=signal.entry,
                sl_price=signal.stop,
                tp_price=signal.tp3,
                status="shadow" if (cfg.shadow or not cfg.live_enabled) else "emitted",
                bar_ms=int(bar.datetime.timestamp() * 1000),
                detail={
                    "vol": vol,
                    "risk_usd": risk_usd,
                    "risk_mult": risk_mult,
                    "tier": resonance.tier,
                    "weekly_ok": resonance.weekly_ok,
                    "hourly_bonus": resonance.hourly_bonus,
                    "strength": signal.strength,
                    "tp1": signal.tp1,
                    "tp2": signal.tp2,
                    "tp3": signal.tp3,
                    "volume_ratio": signal.volume_ratio,
                    "mode": signal.mode,
                },
            )
        except Exception as exc:
            self.write_log(f"strategy signal persist failed: {exc}")
        self._open_market(signal.side, vol)

    def on_signal_bar(self, bar: BarData) -> None:
        cfg = self._cfg()
        self._last_signal_bar = bar
        self._append_signal_bar(bar)
        self._refresh_market_bars(cfg)
        signal = self._detect_signal(cfg)
        high = float(bar.high_price)
        low = float(bar.low_price)

        if not self.trading:
            self._prev_bar_high = high
            self._prev_bar_low = low
            return

        if self.pos != 0:
            side = 1 if self.pos > 0 else -1
            flip = (
                signal is not None
                and cfg.signal_flip_exit
                and int(signal.side) != int(side)
            )
            if flip:
                self.write_log("early exit: opposite Donchian signal while trade open")
                self._pending_side = signal.side
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
                exit_target=str(cfg.exit_target or "tp1"),
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

        if signal is None or self._entry_pending:
            self._prev_bar_high = high
            self._prev_bar_low = low
            self.put_event()
            return

        self._try_enter(cfg, signal, bar)
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
            self.entry_price = float(trade.price)
            self._entry_pending = False
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
            pending_side = self._pending_side
            self._pending_side = 0
            if pending_side and self.trading:
                cfg = self._cfg()
                signal = self._detect_signal(cfg)
                if signal is not None and signal.side == pending_side and self._last_signal_bar is not None:
                    self._try_enter(cfg, signal, self._last_signal_bar)
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

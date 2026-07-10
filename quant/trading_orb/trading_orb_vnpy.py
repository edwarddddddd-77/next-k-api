"""Trading ORB vnpy 策略（sam-bateman/trading-orb 规则）。"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from quant.common.macro_calendar import is_macro_skip_day
from quant.common.session import session_anchor_ms, session_day_str
from quant.engine.eod import should_eod_flat_bar
from quant.engine.exchanges.registry import symbol_from_vt
from quant.market import resolve_market_data_exchange_id
from quant.engine.bootstrap import ensure_vnpy_path
from quant.trading_orb.config import OrbVnpyConfig
from quant.trading_orb.rel_volume import load_volume_baselines
from quant.trading_orb.sizing import fixed_size_for_orb, risk_budget_usdt

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


class TradingOrbVnpyStrategy(CtaTemplate):
    """Opening Range Breakout — 20m OR, rel vol, 10:00–11:30 entry, OR-fraction SL/TP.

    执行对齐 trading-orb 原版 paper_trade_orb：市价入场，SL/TP/EOD 由 5m 高低价
    软件判断后市价平仓（不向交易所挂 bracket stop/limit）。
    """

    author = "next-k-api"

    or_minutes: int = 20
    vol_thresh: float = 1.2
    stop_or_mult: float = 0.5
    target_or_mult: float = 0.75
    breakeven_or_mult: float = 1.0
    entry_start_hour: int = 10
    entry_start_minute: int = 0
    entry_end_hour: int = 11
    entry_end_minute: int = 30
    orb_rth_only: bool = True
    orb_eod_flat: bool = True
    orb_exit_hour: int = 15
    orb_exit_minute: int = 50
    fixed_size: float = 1.0

    parameters = [
        "or_minutes",
        "vol_thresh",
        "stop_or_mult",
        "target_or_mult",
        "breakeven_or_mult",
        "entry_start_hour",
        "entry_start_minute",
        "entry_end_hour",
        "entry_end_minute",
        "orb_rth_only",
        "orb_eod_flat",
        "orb_exit_hour",
        "orb_exit_minute",
        "fixed_size",
    ]
    variables = [
        "session_date",
        "or_high",
        "or_low",
        "traded_today",
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.session_date: str = ""
        self.or_high: float = 0.0
        self.or_low: float = 0.0
        self.or_range: float = 0.0
        self.traded_today: bool = False
        self.entry_price: float = 0.0
        self.stop_price: float = 0.0
        self.target_price: float = 0.0
        self.or_range_at_entry: float = 0.0
        self.breakeven_active: bool = False
        self._entry_pending: bool = False
        self._exit_pending: bool = False
        self._restore_entry_px: float = 0.0
        self._vol_baselines: Dict[str, float] = {}
        self._last_bar: Optional[BarData] = None

    @classmethod
    def from_orb_config(cls, cfg: OrbVnpyConfig) -> dict:
        return {
            "or_minutes": int(cfg.or_minutes),
            "vol_thresh": float(cfg.vol_thresh),
            "stop_or_mult": float(cfg.stop_or_mult),
            "target_or_mult": float(cfg.target_or_mult),
            "breakeven_or_mult": float(cfg.breakeven_or_mult),
            "entry_start_hour": int(cfg.entry_start_hour),
            "entry_start_minute": int(cfg.entry_start_minute),
            "entry_end_hour": int(cfg.entry_end_hour),
            "entry_end_minute": int(cfg.entry_end_minute),
            "orb_rth_only": bool(cfg.rth_only),
            "orb_eod_flat": bool(cfg.eod_flat),
            "orb_exit_hour": int(cfg.exit_hour),
            "orb_exit_minute": int(cfg.exit_minute),
        }

    def _orb_cfg(self) -> OrbVnpyConfig:
        return OrbVnpyConfig.from_env()

    def _session_cfg(self):
        return self._orb_cfg().orb_session_cfg()

    def _bar_ms(self, bar: BarData) -> int:
        return int(bar.datetime.timestamp() * 1000)

    def _bar_session_ts(self, bar: BarData) -> pd.Timestamp:
        cfg = self._session_cfg()
        return pd.Timestamp(self._bar_ms(bar), unit="ms", tz=cfg.session_tz)

    def _in_rth(self, bar: BarData) -> bool:
        if not self.orb_rth_only:
            return True
        from quant.common.session_paper import in_regular_session

        return bool(in_regular_session(self._session_cfg(), now_ms=self._bar_ms(bar)))

    def _is_eod_bar(self, bar: BarData) -> bool:
        if not self.orb_eod_flat:
            return False
        cfg = self._session_cfg()
        ms = self._bar_ms(bar)
        ts = self._bar_session_ts(bar)
        return should_eod_flat_bar(
            bar_ms=ms,
            ts=ts,
            cfg=cfg,
            exit_hour=int(self.orb_exit_hour),
            exit_minute=int(self.orb_exit_minute),
        )

    def _should_flatten_eod(self, bar: BarData) -> bool:
        if not self.orb_eod_flat or self.pos == 0:
            return False
        return self._is_eod_bar(bar) or not self._in_rth(bar)

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

    def _send_market(self, direction: Direction, offset: Offset, volume: float) -> List[str]:
        """币安市价单（trading-orb 原版 Alpaca MarketOrder 对齐）。"""
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

    def _hm_to_min(self, hour: int, minute: int) -> int:
        return int(hour) * 60 + int(minute)

    def _in_entry_window(self, ts: pd.Timestamp) -> bool:
        t = ts.hour * 60 + ts.minute
        start = self._hm_to_min(self.entry_start_hour, self.entry_start_minute)
        end = self._hm_to_min(self.entry_end_hour, self.entry_end_minute)
        return start <= t <= end

    def _reset_session_state(self) -> None:
        self.or_high = 0.0
        self.or_low = 0.0
        self.or_range = 0.0
        self.traded_today = False
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.target_price = 0.0
        self.or_range_at_entry = 0.0
        self.breakeven_active = False
        self._entry_pending = False
        self._exit_pending = False
        self._restore_entry_px = 0.0

    def _price_tick(self) -> float:
        if not self.cta_engine:
            return 0.01
        contract = self.cta_engine.main_engine.get_contract(self.vt_symbol)
        if contract is None:
            return 0.01
        return max(1e-9, float(contract.pricetick or 0.01))

    def _apply_levels_for_entry(self, entry_px: float, side: int) -> None:
        or_r = self.or_range_at_entry if self.or_range_at_entry > 0 else self.or_range
        stop_dist = float(self.stop_or_mult) * or_r
        if stop_dist <= 0 or or_r <= 0:
            return
        px = float(entry_px)
        self.entry_price = px
        if side > 0:
            self.stop_price = px - stop_dist
            self.target_price = px + float(self.target_or_mult) * or_r
        else:
            self.stop_price = px + stop_dist
            self.target_price = px - float(self.target_or_mult) * or_r

    def restore_synced_position(self, *, entry_px: float, pos: float) -> None:
        """重启后从交易所持仓恢复 SL/TP（需当日 OR 已计算）。"""
        if pos == 0 or entry_px <= 0:
            return
        if self.stop_price > 0 and self.entry_price > 0:
            return
        self._entry_pending = False
        if self._orb_cfg().one_trade_per_session:
            self.traded_today = True
        if self.or_range <= 0:
            self._restore_entry_px = float(entry_px)
            return
        if self.or_range_at_entry <= 0:
            self.or_range_at_entry = self.or_range
        side = 1 if pos > 0 else -1
        self._apply_levels_for_entry(entry_px, side)
        self._restore_entry_px = 0.0
        self.write_log(
            f"restored levels {self.vt_symbol} entry={entry_px:.4f} "
            f"sl={self.stop_price:.4f} tp={self.target_price:.4f}"
        )

    def _try_deferred_restore(self) -> None:
        if self._restore_entry_px <= 0 or self.pos == 0 or self.or_range <= 0:
            return
        self.restore_synced_position(entry_px=self._restore_entry_px, pos=float(self.pos))

    def _sync_session(self, bar: BarData) -> None:
        cfg = self._session_cfg()
        ms = self._bar_ms(bar)
        day = session_day_str(ms, tz=cfg.session_tz, session_open_time=cfg.session_open_time)
        if day != self.session_date:
            self.session_date = day
            self._reset_session_state()
            sym = symbol_from_vt(self.vt_symbol)
            orb = self._orb_cfg()
            self._vol_baselines = load_volume_baselines(
                sym,
                cfg=cfg,
                lookback_days=int(orb.vol_lookback_days),
                market_data_exchange=resolve_market_data_exchange_id(orb.market_data_exchange),
            )

    def _update_opening_range(self, bar: BarData) -> None:
        cfg = self._session_cfg()
        ms = self._bar_ms(bar)
        anchor = session_anchor_ms(
            ms, tz=cfg.session_tz, session_open_time=cfg.session_open_time
        )
        or_end_ms = anchor + int(self.or_minutes) * 60_000
        if ms < anchor:
            return
        if ms >= or_end_ms:
            if self.or_high > 0 and self.or_low > 0:
                self.or_range = self.or_high - self.or_low
            return
        hi = float(bar.high_price)
        lo = float(bar.low_price)
        if self.or_high <= 0:
            self.or_high = hi
            self.or_low = lo
        else:
            self.or_high = max(self.or_high, hi)
            self.or_low = min(self.or_low, lo)
        self.or_range = self.or_high - self.or_low

    def _rel_volume(self, bar: BarData) -> float:
        key = self._bar_session_ts(bar).strftime("%H:%M")
        base = float(self._vol_baselines.get(key, 0.0) or 0.0)
        if base <= 0:
            return 0.0
        return float(bar.volume) / base

    def _check_exit_on_bar(self, bar: BarData) -> bool:
        """5m 高低价触发止损/止盈 → 市价平仓（原版 paper_trade_orb.check_exits）。"""
        if self.pos == 0 or self._exit_pending:
            return False
        if self.stop_price <= 0:
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

    def _manage_position(self, bar: BarData) -> None:
        if self.pos == 0:
            return
        if not self.breakeven_active and self.or_range_at_entry > 0:
            fav = (float(bar.close_price) - self.entry_price) * (1 if self.pos > 0 else -1)
            if fav >= self.breakeven_or_mult * self.or_range_at_entry:
                tick = self._price_tick()
                if self.pos > 0:
                    self.stop_price = max(self.stop_price, self.entry_price + tick)
                else:
                    self.stop_price = min(self.stop_price, self.entry_price - tick)
                self.breakeven_active = True

    def _open_market(self, side: int, vol: float) -> None:
        if side > 0:
            oids = self._send_market(Direction.LONG, Offset.OPEN, vol)
        else:
            oids = self._send_market(Direction.SHORT, Offset.OPEN, vol)
        if oids:
            self._entry_pending = True
            orb = self._orb_cfg()
            if orb.one_trade_per_session:
                self.traded_today = True
            return
        orb = self._orb_cfg()
        if orb.shadow or not orb.live_enabled:
            if orb.one_trade_per_session:
                self.traded_today = True
            self.write_log(
                f"signal-only {self.vt_symbol} (shadow={orb.shadow} live={orb.live_enabled})"
            )
            return
        self.write_log(f"live entry rejected {self.vt_symbol}")

    def _try_entry(self, bar: BarData) -> None:
        orb_cfg = self._orb_cfg()
        if self.pos != 0 or self._entry_pending:
            return
        if orb_cfg.one_trade_per_session and self.traded_today:
            return
        if orb_cfg.macro_filter and is_macro_skip_day(self.session_date):
            return
        if self.or_range <= 0 or self.or_high <= self.or_low:
            return
        ts = self._bar_session_ts(bar)
        if not self._in_entry_window(ts):
            return
        rel_vol = self._rel_volume(bar)
        if rel_vol < float(self.vol_thresh):
            return
        close = float(bar.close_price)
        side = 0
        if close > self.or_high:
            side = 1
        elif close < self.or_low:
            side = -1
        else:
            return

        stop_dist = float(self.stop_or_mult) * self.or_range
        if stop_dist <= 0:
            return
        self.or_range_at_entry = self.or_range
        self._apply_levels_for_entry(close, side)

        eq = float(orb_cfg.equity_usdt or 14.0)
        if orb_cfg.compound:
            try:
                from accumulation_radar import init_db
                from quant.trading_orb.db import migrate_orb_vnpy_tables
                from quant.trading_orb.equity import symbol_equity_usdt

                conn = init_db()
                try:
                    cur = conn.cursor()
                    migrate_orb_vnpy_tables(cur)
                    sym = symbol_from_vt(self.vt_symbol)
                    eq = symbol_equity_usdt(orb_cfg, sym, cur=cur)
                finally:
                    conn.close()
            except Exception:
                pass

        vol = fixed_size_for_orb(
            orb_cfg,
            symbol_from_vt(self.vt_symbol),
            close,
            stop_distance=stop_dist,
            equity_usdt=eq,
        )
        if vol <= 0:
            return
        self.fixed_size = vol
        risk_usd = risk_budget_usdt(orb_cfg, equity_usdt=eq)
        self.write_log(
            f"ORB signal {self.vt_symbol} side={'LONG' if side > 0 else 'SHORT'} "
            f"px={close:.4f} relVol={rel_vol:.2f} risk=${risk_usd:.2f} vol={vol}"
        )
        try:
            from quant.engine.strategy_signals import LANE_TRADING_ORB, record_strategy_open_signal

            record_strategy_open_signal(
                lane=LANE_TRADING_ORB,
                symbol=symbol_from_vt(self.vt_symbol),
                side="LONG" if side > 0 else "SHORT",
                entry_price=close,
                sl_price=float(self.stop_price) if self.stop_price else None,
                tp_price=float(self.target_price) if self.target_price else None,
                status="shadow" if (orb_cfg.shadow or not orb_cfg.live_enabled) else "emitted",
                bar_ms=self._bar_ms(bar),
                detail={"rel_vol": rel_vol, "vol": vol, "risk_usd": risk_usd},
            )
        except Exception as exc:
            self.write_log(f"strategy signal persist failed: {exc}")
        self._open_market(side, vol)

    def on_init(self) -> None:
        self.write_log("Trading ORB strategy init")
        self.bg = BarGenerator(self.on_bar, 5, self.on_5min_bar, Interval.MINUTE)
        sym = symbol_from_vt(self.vt_symbol)
        cfg = self._session_cfg()
        orb = self._orb_cfg()
        self._vol_baselines = load_volume_baselines(
            sym,
            cfg=cfg,
            lookback_days=int(orb.vol_lookback_days),
            market_data_exchange=resolve_market_data_exchange_id(orb.market_data_exchange),
        )
        self._reset_session_state()
        self.write_log(f"vol baselines={len(self._vol_baselines)} (live kline, skip load_bar)")

    def on_start(self) -> None:
        self.write_log("Trading ORB strategy start")

    def on_stop(self) -> None:
        if self.pos != 0 and self._last_bar is not None:
            self._flatten_at_bar(self._last_bar)
        self.write_log("Trading ORB strategy stop")

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
        """1m K 线：OR 必须用分钟级（实盘 20m 区间）。"""
        if self.trading:
            self._sync_session(bar)
            self._update_opening_range(bar)
            self._try_deferred_restore()
        self.bg.update_bar(bar)

    def on_5min_bar(self, bar: BarData) -> None:
        self._last_bar = bar
        if self._should_flatten_eod(bar):
            self._flatten_at_bar(bar)
            self.put_event()
            return
        if not self._in_rth(bar):
            orb = self._orb_cfg()
            if orb.vnpy_idle_outside_rth:
                self.cancel_all()
            self.put_event()
            return

        if self.pos != 0:
            if self._check_exit_on_bar(bar):
                self.put_event()
                return
            self._manage_position(bar)
            self.put_event()
            return

        self._try_entry(bar)
        self.put_event()

    def on_trade(self, trade: TradeData) -> None:
        if trade.offset == Offset.OPEN and self.pos != 0:
            side = 1 if self.pos > 0 else -1
            self._apply_levels_for_entry(float(trade.price), side)
            orb = self._orb_cfg()
            if orb.one_trade_per_session:
                self.traded_today = True
            self._entry_pending = False
        if self.pos == 0:
            self.cancel_all()
            self._entry_pending = False
            self._exit_pending = False
            self._refresh_compound_size()
        self.put_event()

    def on_order(self, order: OrderData) -> None:
        from vnpy.trader.constant import Status

        if order.status not in (Status.CANCELLED, Status.REJECTED):
            return
        if order.offset == Offset.OPEN and self.pos == 0:
            self._entry_pending = False
            orb = self._orb_cfg()
            if orb.one_trade_per_session:
                self.traded_today = False
        if order.offset == Offset.CLOSE:
            self._exit_pending = False
        self.put_event()

    def _refresh_compound_size(self) -> None:
        orb = self._orb_cfg()
        if not orb.compound:
            return
        from quant.market import fetch_mark_price, resolve_market_data_exchange_id
        from quant.trading_orb.equity import symbol_equity_usdt

        sym = symbol_from_vt(self.vt_symbol)
        md_exchange = resolve_market_data_exchange_id(orb.market_data_exchange)
        px = fetch_mark_price(sym, exchange_id=md_exchange) or 100.0
        eq = float(orb.equity_usdt or 14.0)
        try:
            from accumulation_radar import init_db
            from quant.trading_orb.db import migrate_orb_vnpy_tables

            conn = init_db()
            try:
                cur = conn.cursor()
                migrate_orb_vnpy_tables(cur)
                eq = symbol_equity_usdt(orb, sym, cur=cur)
            finally:
                conn.close()
        except Exception:
            pass
        stop_dist = max(1e-9, float(self.stop_or_mult) * max(self.or_range_at_entry, self.or_range))
        vol = fixed_size_for_orb(orb, sym, px, stop_distance=stop_dist, equity_usdt=eq)
        if vol > 0 and abs(float(self.fixed_size) - vol) >= 1e-6:
            self.fixed_size = vol
            if self.cta_engine:
                setting = {**TradingOrbVnpyStrategy.from_orb_config(orb), "fixed_size": vol}
                self.cta_engine.update_strategy_setting(self.strategy_name, setting)

    def on_stop_order(self, stop_order: StopOrder) -> None:
        pass

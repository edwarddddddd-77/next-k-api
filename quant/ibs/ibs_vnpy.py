"""IBS vnpy 策略 — 美股 session 日线 IBS（CazSyd / Pagonidis）。"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List

from quant.common.config import OrbConfig
from quant.common.session import (
    is_regular_session,
    session_anchor_ms,
    session_close_ms,
    session_day_str,
)
from quant.common.us_equity_calendar import is_us_equity_trading_day
from quant.engine.bootstrap import ensure_vnpy_path
from quant.engine.exchanges.registry import symbol_from_vt
from quant.ibs.config import IbsLaneConfig
from quant.ibs.core import (
    IbsSignalContext,
    evaluate_signal_context,
    select_signal_context,
    stop_loss_hit,
)
from quant.ibs.session_daily import fetch_daily_bars
from quant.ibs.sizing import size_for_ibs
from quant.market import fetch_klines_forward, klines_to_df, resolve_market_data_exchange_id

ensure_vnpy_path()

from vnpy.trader.constant import Direction, Interval, Offset, OrderType  # noqa: E402
from vnpy.trader.object import BarData, OrderData, TickData, TradeData  # noqa: E402
from vnpy.trader.utility import round_to  # noqa: E402
from vnpy_ctastrategy import BarGenerator, CtaTemplate  # noqa: E402


def _load_lane_config(lane: str) -> IbsLaneConfig:
    if lane == "ibs_aggressive":
        from quant.ibs_aggressive.config import IbsAggressiveConfig

        return IbsAggressiveConfig.from_env()
    if lane == "ibs_tv":
        from quant.ibs_tv.config import IbsTvConfig

        return IbsTvConfig.from_env()
    from quant.ibs_conservative.config import IbsConservativeConfig

    return IbsConservativeConfig.from_env()


class IbsVnpyStrategy(CtaTemplate):
    """Long-only IBS：前一日 IBS < entry 做多，> exit 平仓。"""

    author = "next-k-api"

    lane: str = "ibs_conservative"
    entry_threshold: float = 0.20
    exit_threshold: float = 0.50
    position_pct: float = 0.10
    stop_loss_pct: float = 0.0
    sma_period: int = 200
    trend_ma_type: str = "sma"
    trend_ma_period: int = 200
    min_entry_distance_pct: float = 0.0
    max_trade_duration_days: int = 0
    eval_after_close_minutes: int = 5
    exec_after_open_minutes: int = 1
    execute_at_next_open: bool = True
    signal_minutes: int = 5
    trade_type: str = "long_only"
    fixed_size: float = 1.0

    parameters = [
        "lane",
        "entry_threshold",
        "exit_threshold",
        "position_pct",
        "stop_loss_pct",
        "sma_period",
        "trend_ma_type",
        "trend_ma_period",
        "min_entry_distance_pct",
        "max_trade_duration_days",
        "eval_after_close_minutes",
        "exec_after_open_minutes",
        "execute_at_next_open",
        "signal_minutes",
        "trade_type",
        "fixed_size",
    ]
    variables = ["entry_price", "last_eval_session_day", "last_ibs"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.entry_price: float = 0.0
        self.last_eval_session_day: str = ""
        self.last_ibs: float = 0.0
        self._intraday_rows: List[dict] = []
        self._entry_pending: bool = False
        self._exit_pending: bool = False
        self._holding_days: int = 0
        self._last_closed_entry_price: float = 0.0
        self._pending_action: str = ""
        self._pending_signal_day: str = ""
        self._pending_ctx: IbsSignalContext | None = None
        self._last_open_exec_day: str = ""
        self.bg: BarGenerator | None = None

    @classmethod
    def from_ibs_config(cls, cfg: IbsLaneConfig) -> dict:
        return {
            "lane": cfg.lane,
            "entry_threshold": float(cfg.entry_threshold),
            "exit_threshold": float(cfg.exit_threshold),
            "position_pct": float(cfg.position_pct),
            "stop_loss_pct": float(cfg.stop_loss_pct),
            "sma_period": int(cfg.sma_period),
            "trend_ma_type": str(cfg.trend_ma_type),
            "trend_ma_period": int(cfg.trend_ma_period),
            "min_entry_distance_pct": float(cfg.min_entry_distance_pct),
            "max_trade_duration_days": int(cfg.max_trade_duration_days),
            "eval_after_close_minutes": int(cfg.eval_after_close_minutes),
            "exec_after_open_minutes": int(cfg.exec_after_open_minutes),
            "execute_at_next_open": bool(cfg.execute_at_next_open),
            "signal_minutes": int(cfg.signal_minutes),
            "trade_type": str(cfg.trade_type),
        }

    def _cfg(self) -> IbsLaneConfig:
        return _load_lane_config(str(self.lane))

    def _sess(self) -> OrbConfig:
        return self._cfg().orb_session_cfg()

    def _session_daily(self) -> list:
        cfg = self._cfg()
        sess = self._sess()
        sym = symbol_from_vt(self.vt_symbol)
        md = resolve_market_data_exchange_id(cfg.market_data_exchange)
        intraday_df = None
        if self._intraday_rows:
            import pandas as pd

            intraday_df = pd.DataFrame(self._intraday_rows)
        return fetch_daily_bars(
            sym,
            days=max(30, int(cfg.init_bar_days)),
            exchange_id=md,
            sess=sess,
            source=str(cfg.daily_bar_source),
            intraday_df=intraday_df,
            interval=f"{max(1, int(cfg.signal_minutes))}m",
        )

    def _append_intraday(self, bar: BarData) -> None:
        ms = int(bar.datetime.timestamp() * 1000)
        row = {
            "open_time": ms,
            "open": float(bar.open_price),
            "high": float(bar.high_price),
            "low": float(bar.low_price),
            "close": float(bar.close_price),
            "volume": float(bar.volume or 0.0),
        }
        if self._intraday_rows and int(self._intraday_rows[-1]["open_time"]) == ms:
            self._intraday_rows[-1] = row
        else:
            self._intraday_rows.append(row)
        max_rows = 24 * 12 * max(30, int(self._cfg().init_bar_days))
        if len(self._intraday_rows) > max_rows:
            self._intraday_rows = self._intraday_rows[-max_rows:]

    def _eval_ready(self, bar_ms: int) -> bool:
        sess = self._sess()
        tz = sess.session_tz
        open_time = sess.session_open_time
        close_time = sess.session_close_time
        day = session_day_str(bar_ms, tz=tz, session_open_time=open_time)
        anchor_ms = session_anchor_ms(bar_ms, tz=tz, session_open_time=open_time)
        close_ms = session_close_ms(anchor_ms, tz=tz, session_close_time=close_time)
        if close_ms is None:
            return False
        delay_ms = max(1, int(self.eval_after_close_minutes)) * 60_000
        if int(bar_ms) < int(close_ms) + delay_ms:
            return False
        if day == self.last_eval_session_day:
            return False
        if not is_us_equity_trading_day(day):
            return False
        return True

    def _open_ready(self, bar_ms: int) -> bool:
        sess = self._sess()
        tz = sess.session_tz
        open_time = sess.session_open_time
        close_time = sess.session_close_time
        day = session_day_str(bar_ms, tz=tz, session_open_time=open_time)
        if not is_us_equity_trading_day(day):
            return False
        if not is_regular_session(
            int(bar_ms),
            tz=tz,
            session_open_time=open_time,
            session_close_time=close_time,
        ):
            return False
        anchor_ms = session_anchor_ms(bar_ms, tz=tz, session_open_time=open_time)
        delay_ms = max(0, int(self.exec_after_open_minutes)) * 60_000
        if int(bar_ms) < int(anchor_ms) + delay_ms:
            return False
        if int(bar_ms) >= int(anchor_ms) + 60 * 60_000:
            return False
        if day == self._last_open_exec_day:
            return False
        if self._pending_signal_day and day <= self._pending_signal_day:
            return False
        return True

    def _equity_usdt(self, cfg: IbsLaneConfig) -> float:
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

    def restore_synced_position(self, *, entry_px: float, pos: float) -> None:
        if abs(float(pos or 0.0)) < 1e-12:
            return
        self.entry_price = float(entry_px)
        self.write_log(f"restored synced position entry={entry_px:.4f}")

    def on_init(self) -> None:
        self.write_log(f"IBS init lane={self.lane}")
        mins = max(1, int(self.signal_minutes))
        self.bg = BarGenerator(self.on_bar, mins, self.on_signal_bar, Interval.MINUTE)
        self._preload_intraday()

    def _preload_intraday(self) -> None:
        cfg = self._cfg()
        sym = symbol_from_vt(self.vt_symbol)
        days = max(30, int(cfg.init_bar_days))
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
            self._append_intraday(bar)
        self.write_log(f"preloaded {len(self._intraday_rows)} x {interval} bars")

    def on_start(self) -> None:
        self.write_log(f"IBS start lane={self.lane}")

    def _position_side(self) -> int:
        pos = float(self.pos or 0.0)
        if pos > 0:
            return 1
        if pos < 0:
            return -1
        return 0

    def on_stop(self) -> None:
        if self.pos != 0:
            self._flatten_market()
        self.write_log("IBS stop")

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
        self._append_intraday(bar)
        if not self.trading:
            return

        close = float(bar.close_price)
        bar_ms = int(bar.datetime.timestamp() * 1000)
        cfg = self._cfg()

        if self._pending_action and self._open_ready(bar_ms):
            if self._execute_pending_at_open(bar, bar_ms, close, cfg):
                self.put_event()
                return

        if not self._eval_ready(bar_ms):
            self.put_event()
            return

        daily = self._session_daily()
        ctx = select_signal_context(
            daily,
            trend_price_mode=str(cfg.trend_price_mode),
            current_price=close,
        )
        if ctx is None:
            self.put_event()
            return

        pos_side = self._position_side()
        if pos_side != 0 and stop_loss_hit(
            side=pos_side,
            entry_price=float(self.entry_price),
            close=close,
            stop_loss_pct=float(self.stop_loss_pct),
        ):
            self.write_log(f"IBS stop loss px={close:.4f}")
            self._flatten_market()
            self.put_event()
            return

        sess = self._sess()
        eval_day = session_day_str(
            bar_ms,
            tz=sess.session_tz,
            session_open_time=sess.session_open_time,
        )
        action = evaluate_signal_context(
            ctx,
            position_side=pos_side,
            trade_type=str(cfg.trade_type),
            entry_threshold=float(self.entry_threshold),
            exit_threshold=float(self.exit_threshold),
            sma_period=int(self.sma_period),
            trend_ma_type=str(self.trend_ma_type),
            trend_ma_period=int(self.trend_ma_period),
            holding_days=int(self._holding_days),
            max_trade_duration_days=int(self.max_trade_duration_days),
            last_entry_price=float(self._last_closed_entry_price),
            min_entry_distance_pct=float(self.min_entry_distance_pct),
        )
        if pos_side != 0:
            self._holding_days += 1
        else:
            self._holding_days = 0
        self.last_ibs = float(ctx.prev_bar.ibs)
        self.last_eval_session_day = eval_day
        self.write_log(
            f"IBS eval day={eval_day} prev={ctx.prev_bar.session_day} "
            f"ibs={ctx.prev_bar.ibs:.4f} action={action} px={close:.4f}"
        )

        if action in ("BUY", "SELL", "SHORT", "COVER") and bool(self.execute_at_next_open):
            self._pending_action = action
            self._pending_signal_day = eval_day
            self._pending_ctx = ctx
            self.write_log(f"IBS pending {action} for next session open")
            self.put_event()
            return

        self._apply_action(action, bar, bar_ms, close, cfg, ctx)
        self.put_event()

    def _execute_pending_at_open(
        self,
        bar: BarData,
        bar_ms: int,
        close: float,
        cfg: IbsLaneConfig,
    ) -> bool:
        sess = self._sess()
        exec_day = session_day_str(
            bar_ms,
            tz=sess.session_tz,
            session_open_time=sess.session_open_time,
        )
        action = self._pending_action
        ctx = self._pending_ctx
        signal_day = self._pending_signal_day
        if signal_day and exec_day <= signal_day:
            return False

        self._pending_action = ""
        self._pending_signal_day = ""
        self._pending_ctx = None
        self._last_open_exec_day = exec_day

        if ctx is None:
            return False

        if action == "BUY":
            open_ctx = select_signal_context(
                self._session_daily(),
                trend_price_mode=str(cfg.trend_price_mode),
                current_price=close,
                ma_excludes_last_bar=True,
            )
            if open_ctx is None:
                self.write_log("IBS pending BUY skipped: insufficient daily bars")
                return True
            recheck = evaluate_signal_context(
                open_ctx,
                position_side=0,
                trade_type=str(cfg.trade_type),
                entry_threshold=float(self.entry_threshold),
                exit_threshold=float(self.exit_threshold),
                sma_period=int(self.sma_period),
                trend_ma_type=str(self.trend_ma_type),
                trend_ma_period=int(self.trend_ma_period),
                holding_days=0,
                max_trade_duration_days=int(self.max_trade_duration_days),
                last_entry_price=float(self._last_closed_entry_price),
                min_entry_distance_pct=float(self.min_entry_distance_pct),
            )
            if recheck != "BUY":
                self.write_log(f"IBS pending BUY cancelled at open recheck={recheck}")
                return True
            ctx = open_ctx

        if action == "SHORT":
            open_ctx = select_signal_context(
                self._session_daily(),
                trend_price_mode=str(cfg.trend_price_mode),
                current_price=close,
                ma_excludes_last_bar=True,
            )
            if open_ctx is None:
                self.write_log("IBS pending SHORT skipped: insufficient daily bars")
                return True
            recheck = evaluate_signal_context(
                open_ctx,
                position_side=0,
                trade_type=str(cfg.trade_type),
                entry_threshold=float(self.entry_threshold),
                exit_threshold=float(self.exit_threshold),
                sma_period=int(self.sma_period),
                trend_ma_type=str(self.trend_ma_type),
                trend_ma_period=int(self.trend_ma_period),
                holding_days=0,
                max_trade_duration_days=int(self.max_trade_duration_days),
                last_entry_price=float(self._last_closed_entry_price),
                min_entry_distance_pct=float(self.min_entry_distance_pct),
            )
            if recheck != "SHORT":
                self.write_log(f"IBS pending SHORT cancelled at open recheck={recheck}")
                return True
            ctx = open_ctx

        if action == "SELL" and self.pos <= 0:
            return True
        if action == "COVER" and self.pos >= 0:
            return True
        if action == "BUY" and self.pos > 0:
            return True
        if action == "SHORT" and self.pos < 0:
            return True

        self.write_log(f"IBS execute pending {action} at open day={exec_day} px={close:.4f}")
        self._apply_action(action, bar, bar_ms, close, cfg, ctx)
        return True

    def _apply_action(
        self,
        action: str,
        bar: BarData,
        bar_ms: int,
        close: float,
        cfg: IbsLaneConfig,
        ctx: IbsSignalContext,
    ) -> None:
        if action == "SELL" and self.pos > 0:
            self._flatten_market()
            return
        if action == "COVER" and self.pos < 0:
            self._flatten_market()
            return

        if action == "BUY" and self.pos == 0 and not self._entry_pending:
            eq = self._equity_usdt(cfg)
            vol = size_for_ibs(cfg, close, equity_usdt=eq)
            if vol <= 0:
                return
            self.fixed_size = vol
            self.entry_price = close
            sl = close * (1.0 - float(self.stop_loss_pct)) if float(self.stop_loss_pct) > 0 else None
            self.write_log(
                f"IBS BUY {self.vt_symbol} ibs={ctx.prev_bar.ibs:.4f} px={close:.4f} vol={vol} eq={eq:.2f}"
            )
            try:
                from quant.engine.strategy_signals import record_strategy_open_signal

                record_strategy_open_signal(
                    lane=str(self.lane),
                    symbol=symbol_from_vt(self.vt_symbol),
                    side="LONG",
                    entry_price=close,
                    sl_price=sl,
                    tp_price=None,
                    status="shadow" if (cfg.shadow or not cfg.live_enabled) else "emitted",
                    bar_ms=bar_ms,
                    detail={"ibs": ctx.prev_bar.ibs, "vol": vol, "profile": cfg.profile},
                )
            except Exception as exc:
                self.write_log(f"strategy signal persist failed: {exc}")
            self._open_long(vol)
            return

        if action == "SHORT" and self.pos == 0 and not self._entry_pending:
            eq = self._equity_usdt(cfg)
            vol = size_for_ibs(cfg, close, equity_usdt=eq)
            if vol <= 0:
                return
            self.fixed_size = vol
            self.entry_price = close
            sl = close * (1.0 + float(self.stop_loss_pct)) if float(self.stop_loss_pct) > 0 else None
            self.write_log(
                f"IBS SHORT {self.vt_symbol} ibs={ctx.prev_bar.ibs:.4f} px={close:.4f} vol={vol} eq={eq:.2f}"
            )
            try:
                from quant.engine.strategy_signals import record_strategy_open_signal

                record_strategy_open_signal(
                    lane=str(self.lane),
                    symbol=symbol_from_vt(self.vt_symbol),
                    side="SHORT",
                    entry_price=close,
                    sl_price=sl,
                    tp_price=None,
                    status="shadow" if (cfg.shadow or not cfg.live_enabled) else "emitted",
                    bar_ms=bar_ms,
                    detail={"ibs": ctx.prev_bar.ibs, "vol": vol, "profile": cfg.profile},
                )
            except Exception as exc:
                self.write_log(f"strategy signal persist failed: {exc}")
            self._open_short(vol)

    def _open_short(self, vol: float) -> None:
        oids = self._send_market(Direction.SHORT, Offset.OPEN, vol)
        if oids:
            self._entry_pending = True
            return
        cfg = self._cfg()
        if cfg.shadow or not cfg.live_enabled:
            self.write_log(f"signal-only {self.vt_symbol} (shadow={cfg.shadow} live={cfg.live_enabled})")
            return
        self.write_log(f"live short entry rejected {self.vt_symbol}")

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

    def _open_long(self, vol: float) -> None:
        oids = self._send_market(Direction.LONG, Offset.OPEN, vol)
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
        if self.pos > 0:
            oids = self._send_market(Direction.SHORT, Offset.CLOSE, vol)
        else:
            oids = self._send_market(Direction.LONG, Offset.CLOSE, vol)
        if oids:
            self._exit_pending = True

    def on_trade(self, trade: TradeData) -> None:
        if trade.offset == Offset.OPEN and self.pos != 0:
            self.entry_price = float(trade.price)
            self._entry_pending = False
            self._holding_days = 1
        if self.pos == 0:
            self.cancel_all()
            self._entry_pending = False
            self._exit_pending = False
            if float(self.entry_price or 0.0) > 0:
                self._last_closed_entry_price = float(self.entry_price)
            self.entry_price = 0.0
            self._holding_days = 0
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

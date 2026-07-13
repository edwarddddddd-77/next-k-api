"""Anchor Drift vnpy 策略 — RTH 收盘锚定 + 非 RTH 均值回归。"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List

from quant.anchor_drift.config import AnchorDriftConfig
from quant.anchor_drift.core import (
    adverse_drift_stop,
    calculate_drift,
    generate_signal,
)
from quant.anchor_drift.db import load_anchor, save_anchor
from quant.anchor_drift.session import in_drift_trading_window, in_preopen_flat_window
from quant.anchor_drift.sizing import risk_budget_usdt, size_for_drift
from quant.common.session import session_day_str
from quant.common.session_paper import in_regular_session
from quant.engine.bootstrap import ensure_vnpy_path
from quant.engine.exchanges.registry import symbol_from_vt

ensure_vnpy_path()

from vnpy.trader.constant import Direction, Interval, Offset, OrderType, Status  # noqa: E402
from vnpy.trader.object import BarData, OrderData, TickData, TradeData  # noqa: E402
from vnpy.trader.utility import round_to  # noqa: E402
from vnpy_ctastrategy import CtaTemplate  # noqa: E402


class AnchorDriftVnpyStrategy(CtaTemplate):
    """闭市/周末 token 价相对 RTH 锚定价 drift 均值回归（BQuant 阈值）。"""

    author = "next-k-api"

    signal_threshold: float = 0.015
    converge_threshold: float = 0.003
    max_adverse_extension: float = 0.025
    preopen_flat_minutes: int = 5
    tick_interval_sec: float = 30.0
    fixed_size: float = 1.0
    orb_rth_only: bool = False

    parameters = [
        "signal_threshold",
        "converge_threshold",
        "max_adverse_extension",
        "preopen_flat_minutes",
        "tick_interval_sec",
        "fixed_size",
        "orb_rth_only",
    ]
    variables = [
        "anchor_session",
        "anchor_price",
        "last_drift_pct",
        "last_signal",
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.anchor_session: str = ""
        self.anchor_price: float = 0.0
        self.last_drift_pct: float = 0.0
        self.last_signal: str = "FLAT"
        self._last_rth_price: float = 0.0
        self._was_in_rth: bool = False
        self._last_tick_ts: float = 0.0
        self._entry_pending: bool = False
        self._exit_pending: bool = False
        self._entry_side: int = 0
        self._traded_this_anchor: bool = False

    @classmethod
    def from_drift_config(cls, cfg: AnchorDriftConfig) -> dict:
        return {
            "signal_threshold": float(cfg.signal_threshold),
            "converge_threshold": float(cfg.converge_threshold),
            "max_adverse_extension": float(cfg.max_adverse_extension),
            "preopen_flat_minutes": int(cfg.preopen_flat_minutes),
            "tick_interval_sec": float(cfg.tick_interval_sec),
            "orb_rth_only": False,
        }

    def _cfg(self) -> AnchorDriftConfig:
        return AnchorDriftConfig.from_env()

    def _session(self):
        return self._cfg().session_cfg()

    def _tick_ms(self, tick: TickData) -> int:
        dt = getattr(tick, "datetime", None)
        if dt is not None:
            return int(dt.timestamp() * 1000)
        return int(time.time() * 1000)

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _persist_anchor(self, price: float, now_ms: int, session_date: str) -> None:
        self.anchor_price = float(price)
        self.anchor_session = session_date
        try:
            from accumulation_radar import init_db

            conn = init_db()
            try:
                cur = conn.cursor()
                from quant.anchor_drift.db import migrate_anchor_drift_tables

                migrate_anchor_drift_tables(cur)
                save_anchor(
                    cur,
                    symbol=symbol_from_vt(self.vt_symbol),
                    anchor_session=session_date,
                    anchor_price=float(price),
                    anchor_ms=int(now_ms),
                    now_utc=self._utc_now(),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            self.write_log(f"anchor persist failed: {exc}")

    def _load_persisted_anchor(self) -> None:
        try:
            from accumulation_radar import init_db

            conn = init_db()
            try:
                cur = conn.cursor()
                from quant.anchor_drift.db import migrate_anchor_drift_tables

                migrate_anchor_drift_tables(cur)
                row = load_anchor(cur, symbol_from_vt(self.vt_symbol))
                if row is None:
                    return
                sess, price, _ms = row
                self.anchor_session = sess
                self.anchor_price = float(price)
            finally:
                conn.close()
        except Exception as exc:
            self.write_log(f"anchor load failed: {exc}")

    def _equity_usdt(self, cfg: AnchorDriftConfig) -> float:
        base = float(cfg.equity_usdt or 14.0)
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
        self.write_log("Anchor Drift init")
        self._load_persisted_anchor()
        if self.anchor_price > 0:
            self.write_log(
                f"loaded anchor session={self.anchor_session} px={self.anchor_price:.4f}"
            )

    def on_start(self) -> None:
        self.write_log("Anchor Drift start")

    def on_stop(self) -> None:
        if self.pos != 0:
            self._flatten_market("stop")
        self.write_log("Anchor Drift stop")

    def on_tick(self, tick: TickData) -> None:
        extra = getattr(tick, "extra", None) or {}
        bar = extra.get("bar")
        if bar is not None:
            self.on_bar(bar)
            return

        now = time.time()
        if now - self._last_tick_ts < float(self.tick_interval_sec):
            return
        self._last_tick_ts = now

        px = float(getattr(tick, "last_price", 0.0) or 0.0)
        if px <= 0:
            return

        cfg = self._cfg()
        sess = self._session()
        now_ms = self._tick_ms(tick)
        in_rth = bool(in_regular_session(sess, now_ms=now_ms))

        if in_rth:
            self._last_rth_price = px
            if self._was_in_rth:
                self.put_event()
                return
            self._was_in_rth = True
            self.put_event()
            return

        if self._was_in_rth and self._last_rth_price > 0:
            if self.anchor_price != self._last_rth_price:
                self._traded_this_anchor = False
            anchor_day = session_day_str(
                now_ms,
                tz=sess.session_tz,
                session_open_time=sess.session_open_time,
            )
            self._persist_anchor(self._last_rth_price, now_ms, anchor_day)
            self.write_log(
                f"anchor set session={anchor_day} px={self._last_rth_price:.4f} drift_window open"
            )
        self._was_in_rth = False

        if in_preopen_flat_window(now_ms, sess, flat_minutes=int(self.preopen_flat_minutes)):
            if self.pos != 0:
                self._flatten_market("preopen_flat")
                self._traded_this_anchor = True
            self.put_event()
            return

        if not in_drift_trading_window(
            now_ms,
            sess,
            flat_minutes=int(self.preopen_flat_minutes),
        ):
            self.put_event()
            return

        if self.anchor_price <= 0:
            self.put_event()
            return

        drift = calculate_drift(anchor_price=self.anchor_price, current_price=px)
        if drift is None:
            self.put_event()
            return

        self.last_drift_pct = float(drift)
        sig = generate_signal(
            drift,
            signal_threshold=float(self.signal_threshold),
            converge_threshold=float(self.converge_threshold),
        )
        self.last_signal = sig.signal

        if self.pos != 0:
            side = 1 if self.pos > 0 else -1
            if sig.signal == "CONVERGED":
                self._flatten_market("converged")
                self._traded_this_anchor = True
            elif adverse_drift_stop(
                drift,
                side=side,
                signal_threshold=float(self.signal_threshold),
                max_adverse_extension=float(self.max_adverse_extension),
            ):
                self._flatten_market("adverse_drift")
                self._traded_this_anchor = True
            self.put_event()
            return

        if sig.signal not in ("LONG", "SHORT") or self._entry_pending:
            self.put_event()
            return
        if self._traded_this_anchor:
            self.put_event()
            return

        eq = self._equity_usdt(cfg)
        vol = size_for_drift(cfg, px, anchor_price=self.anchor_price, equity_usdt=eq)
        if vol <= 0:
            self.put_event()
            return
        self.fixed_size = vol
        side = 1 if sig.signal == "LONG" else -1
        self._entry_side = side
        risk_usd = risk_budget_usdt(cfg, equity_usdt=eq)
        self.write_log(
            f"drift signal {self.vt_symbol} {sig.signal} px={px:.4f} "
            f"anchor={self.anchor_price:.4f} drift={drift * 100:+.2f}% "
            f"conf={sig.confidence} risk=${risk_usd:.2f} vol={vol}"
        )
        try:
            from quant.engine.strategy_signals import LANE_ANCHOR_DRIFT, record_strategy_open_signal

            record_strategy_open_signal(
                lane=LANE_ANCHOR_DRIFT,
                symbol=symbol_from_vt(self.vt_symbol),
                side=sig.signal,
                entry_price=px,
                sl_price=0.0,
                tp_price=self.anchor_price,
                status="shadow" if (cfg.shadow or not cfg.live_enabled) else "emitted",
                bar_ms=now_ms,
                detail={
                    "vol": vol,
                    "anchor_price": self.anchor_price,
                    "drift_pct": drift,
                    "confidence": sig.confidence,
                    "reasoning": sig.reasoning,
                },
            )
        except Exception as exc:
            self.write_log(f"strategy signal persist failed: {exc}")
        self._open_market(side, vol)
        self.put_event()

    def on_bar(self, bar: BarData) -> None:
        tick = TickData(
            symbol=bar.symbol,
            exchange=bar.exchange,
            datetime=bar.datetime,
            name=bar.symbol,
            last_price=float(bar.close_price),
            volume=float(bar.volume),
            gateway_name=bar.gateway_name,
        )
        self.on_tick(tick)

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

    def _flatten_market(self, reason: str) -> None:
        if self.pos == 0 or self._exit_pending:
            return
        self.write_log(f"flatten {self.vt_symbol} reason={reason}")
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
            self._entry_pending = False
        if self.pos == 0:
            self.cancel_all()
            self._entry_pending = False
            self._exit_pending = False
            self._entry_side = 0
        self.put_event()

    def on_order(self, order: OrderData) -> None:
        if order.status not in (Status.CANCELLED, Status.REJECTED):
            return
        if order.offset == Offset.OPEN and self.pos == 0:
            self._entry_pending = False
        if order.offset == Offset.CLOSE:
            self._exit_pending = False
        self.put_event()

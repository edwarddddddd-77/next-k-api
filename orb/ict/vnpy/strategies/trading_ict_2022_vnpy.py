"""ICT 2022 Model + HMM 方案 A — vnpy 策略（币安永续 5m）。

- 入场：扫荡 → 突破 → FVG CE 限价单
- 过滤：HMM 5m official == RANGE（可关）
- 出场：5m 高低价触发 SL/TP → 市价平仓（对齐回测软件撮合）
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from orb.ict.config import IctVnpyConfig
from orb.ict.vnpy.bootstrap import ensure_vnpy_path
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

from tools.cta.hmm_regime_filter import HMMConfig, compute_hmm_regime  # noqa: E402
from tools.cta.ict_tradingiq_core import (  # noqa: E402
    FVG,
    M2022_ENTRY_PCT,
    detect_fvgs,
    detect_sweeps,
    nearest_fvg_to_level,
)

RANGE = 1
MS_1H = 3_600_000


class TradingIct2022VnpyStrategy(CtaTemplate):
    author = "next-k-api"

    position_pct: float = 1.0
    leverage: float = 2.0
    rr: float = 1.5
    hmm_filter: bool = True
    hmm_stick: float = 0.97
    hmm_confirm: int = 3
    fixed_size: float = 0.01

    parameters = [
        "position_pct",
        "leverage",
        "rr",
        "hmm_filter",
        "hmm_stick",
        "hmm_confirm",
        "fixed_size",
    ]
    variables = [
        "hmm_regime",
        "active_setups",
        "entry_price",
        "stop_price",
        "target_price",
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.hmm_regime: int = -1
        self.active_setups: int = 0
        self.entry_price: float = 0.0
        self.stop_price: float = 0.0
        self.target_price: float = 0.0

        self._bars: List[dict] = []
        self._setups: List[dict] = []
        self._all_fvgs: List[FVG] = []
        self._hmm_df: Optional[pd.DataFrame] = None
        self._pending_limit_px: float = 0.0
        self._pending_sl: float = 0.0
        self._pending_tp: float = 0.0
        self._pending_side: str = ""
        self._pending_born_ms: int = 0
        self._entry_pending: bool = False
        self._exit_pending: bool = False
        self._last_bar: Optional[BarData] = None
        self._shadow_pos: float = 0.0
        self._used_fvg_keys: set[tuple] = set()

    @classmethod
    def from_ict_config(cls, cfg: IctVnpyConfig) -> dict:
        return {
            "position_pct": float(cfg.position_pct),
            "leverage": float(cfg.leverage),
            "rr": float(cfg.rr),
            "hmm_filter": bool(cfg.hmm_filter),
            "hmm_stick": float(cfg.hmm_stick),
            "hmm_confirm": int(cfg.hmm_confirm),
        }

    def _ict_cfg(self) -> IctVnpyConfig:
        return IctVnpyConfig.from_env()

    def _bar_ms(self, bar: BarData) -> int:
        return int(bar.datetime.timestamp() * 1000)

    def _price_tick(self) -> float:
        if not self.cta_engine:
            return 0.01
        contract = self.cta_engine.main_engine.get_contract(self.vt_symbol)
        if contract is None:
            return 0.01
        return max(1e-9, float(contract.pricetick or 0.01))

    def _limit_fillable(self, px: float, hi: float, lo: float, side: str) -> bool:
        tick = self._price_tick() * 2
        return lo <= px + tick if side == "long" else hi >= px - tick

    def _tp_from_rr(self, entry: float, sl: float, side: str) -> float:
        risk = abs(entry - sl)
        if risk <= 0:
            return entry
        return entry + self.rr * risk if side == "long" else entry - self.rr * risk

    def _fvg_key(self, g: FVG) -> tuple:
        return (int(g.born_ms), int(g.direction), round(float(g.top), 8), round(float(g.bot), 8))

    def _equity_usdt(self) -> float:
        cfg = self._ict_cfg()
        eq = float(cfg.equity_usdt)
        if not cfg.compound:
            return eq
        try:
            from accumulation_radar import init_db
            from orb.ict.equity import symbol_equity_usdt
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
        from orb.ict.vnpy.sizing import fixed_size_for_ict

        cfg = self._ict_cfg()
        vol = fixed_size_for_ict(cfg, price, equity_usdt=self._equity_usdt())
        if vol > 0:
            self.fixed_size = vol
        return vol

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
            self, contract, direction, offset, 0.0, vol, OrderType.MARKET, False, False
        )

    def _send_limit(self, direction: Direction, price: float, volume: float) -> List[str]:
        if not self.trading or not self.cta_engine:
            return []
        contract = self.cta_engine.main_engine.get_contract(self.vt_symbol)
        if contract is None:
            return []
        px = round_to(float(price), float(contract.pricetick or 0.01))
        vol = round_to(float(volume), float(contract.min_volume or 0.001))
        if vol <= 0 or px <= 0:
            return []
        return self.cta_engine.send_server_order(
            self, contract, direction, Offset.OPEN, px, vol, OrderType.LIMIT, False, False
        )

    def _append_bar(self, bar: BarData) -> None:
        self._bars.append(
            {
                "open_time": self._bar_ms(bar),
                "open": float(bar.open_price),
                "high": float(bar.high_price),
                "low": float(bar.low_price),
                "close": float(bar.close_price),
                "volume": float(bar.volume),
            }
        )
        max_bars = 3000
        if len(self._bars) > max_bars:
            self._bars = self._bars[-max_bars:]

    def _df(self) -> pd.DataFrame:
        if not self._bars:
            return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
        return pd.DataFrame(self._bars)

    def _refresh_indicators(self) -> None:
        df = self._df()
        if len(df) < 30:
            return
        self._all_fvgs = detect_fvgs(df)
        if not self.hmm_filter:
            return
        self._hmm_df = compute_hmm_regime(
            df,
            HMMConfig(stick=float(self.hmm_stick), confirm_bars=int(self.hmm_confirm)),
        )
        if not self._hmm_df.empty:
            self.hmm_regime = int(self._hmm_df.iloc[-1]["hmm_official"])

    def _hmm_allows_entry(self) -> bool:
        if not self.hmm_filter:
            return True
        return self.hmm_regime == RANGE

    def _cancel_pending_entry(self, reason: str) -> None:
        if self._entry_pending or self._pending_limit_px > 0:
            self.cancel_all()
        self._entry_pending = False
        self._pending_limit_px = 0.0
        self._pending_sl = 0.0
        self._pending_tp = 0.0
        self._pending_side = ""
        self._pending_born_ms = 0
        if reason:
            self.write_log(f"cancel pending entry: {reason}")

    def _place_limit_entry(self, *, side: str, entry: float, sl: float, tp: float, t_ms: int) -> None:
        vol = self._order_volume(entry)
        if vol <= 0:
            return
        cfg = self._ict_cfg()
        if cfg.shadow or not cfg.live_enabled:
            self._pending_limit_px = entry
            self._pending_sl = sl
            self._pending_tp = tp
            self._pending_side = side
            self._pending_born_ms = t_ms
            self._entry_pending = True
            self.write_log(
                f"signal-only LIMIT {side} entry={entry:.4f} sl={sl:.4f} tp={tp:.4f} "
                f"hmm={self.hmm_regime} vol={vol}"
            )
            return
        direction = Direction.LONG if side == "long" else Direction.SHORT
        oids = self._send_limit(direction, entry, vol)
        if oids:
            self._pending_limit_px = entry
            self._pending_sl = sl
            self._pending_tp = tp
            self._pending_side = side
            self._pending_born_ms = t_ms
            self._entry_pending = True
            self.write_log(
                f"LIMIT {side} entry={entry:.4f} sl={sl:.4f} tp={tp:.4f} "
                f"hmm={self.hmm_regime} vol={vol}"
            )

    def _check_pending_limit(self, bar: BarData) -> bool:
        if not self._entry_pending or self.pos != 0:
            return False
        cfg = self._ict_cfg()
        t = self._bar_ms(bar)
        hi = float(bar.high_price)
        lo = float(bar.low_price)
        side = self._pending_side
        if not side:
            return False
        ttl = int(cfg.limit_ttl_hours * MS_1H)
        if t - self._pending_born_ms > ttl:
            self._cancel_pending_entry("limit_ttl")
            return True
        if (side == "long" and lo <= self._pending_sl) or (side == "short" and hi >= self._pending_sl):
            self._cancel_pending_entry("sl_before_fill")
            return True
        if self._limit_fillable(self._pending_limit_px, hi, lo, side):
            if cfg.shadow or not cfg.live_enabled:
                self._on_entry_filled(self._pending_limit_px, side)
                return True
        return False

    def _effective_pos(self) -> float:
        cfg = self._ict_cfg()
        if cfg.shadow or not cfg.live_enabled:
            return float(self._shadow_pos)
        return float(self.pos)

    def restore_synced_position(self, *, entry_px: float, pos: float) -> None:
        """重启后从交易所持仓恢复；SL/TP 优先用 vnpy 持久化变量。"""
        if pos == 0 or entry_px <= 0:
            return
        self._entry_pending = False
        if self.entry_price <= 0:
            self.entry_price = float(entry_px)
        if self.stop_price > 0 and self.target_price > 0:
            self.write_log(
                f"restored ICT {self.vt_symbol} entry={self.entry_price:.4f} "
                f"sl={self.stop_price:.4f} tp={self.target_price:.4f}"
            )
            return
        self.write_log(
            f"WARNING: ICT open position without SL/TP entry={entry_px:.4f} — will flatten on start"
        )

    def _on_entry_filled(self, px: float, side: str) -> None:
        self.entry_price = float(px)
        self.stop_price = float(self._pending_sl)
        self.target_price = float(self._pending_tp)
        self._entry_pending = False
        self._pending_limit_px = 0.0
        cfg = self._ict_cfg()
        if (cfg.shadow or not cfg.live_enabled) and self.pos == 0:
            vol = self._order_volume(px)
            self._shadow_pos = vol if side == "long" else -vol
        self.write_log(
            f"ENTRY {side} px={self.entry_price:.4f} sl={self.stop_price:.4f} tp={self.target_price:.4f}"
        )

    def _check_exit_on_bar(self, bar: BarData) -> bool:
        pos = self._effective_pos()
        if pos == 0 or self._exit_pending or self.stop_price <= 0:
            return False
        cfg = self._ict_cfg()
        shadow = cfg.shadow or not cfg.live_enabled
        hi = float(bar.high_price)
        lo = float(bar.low_price)
        vol = abs(pos)
        if pos > 0:
            if lo <= self.stop_price:
                reason = "stop_loss"
            elif self.target_price > 0 and hi >= self.target_price:
                reason = "take_profit"
            else:
                return False
            if shadow:
                self.write_log(f"EXIT LONG {reason} (shadow)")
                self._shadow_pos = 0.0
                self.entry_price = self.stop_price = self.target_price = 0.0
                return True
            self.cancel_all()
            self.write_log(f"EXIT LONG {reason} market")
            oids = self._send_market(Direction.SHORT, Offset.CLOSE, vol)
            if oids:
                self._exit_pending = True
            return bool(oids)
        if hi >= self.stop_price:
            reason = "stop_loss"
        elif self.target_price > 0 and lo <= self.target_price:
            reason = "take_profit"
        else:
            return False
        if shadow:
            self.write_log(f"EXIT SHORT {reason} (shadow)")
            self._shadow_pos = 0.0
            self.entry_price = self.stop_price = self.target_price = 0.0
            return True
        self.cancel_all()
        self.write_log(f"EXIT SHORT {reason} market")
        oids = self._send_market(Direction.LONG, Offset.CLOSE, vol)
        if oids:
            self._exit_pending = True
        return bool(oids)

    def _process_setups(self, bar: BarData) -> None:
        if self.pos != 0 or self._entry_pending:
            return
        if not self._hmm_allows_entry():
            return

        df = self._df()
        if len(df) < 25:
            return
        cfg = self._ict_cfg()
        i = len(df) - 1
        row = df.iloc[i]
        t = int(row["open_time"])
        hi = float(row["high"])
        lo = float(row["low"])
        cl = float(row["close"])
        op = float(row["open"])
        sweep_low, sweep_high = detect_sweeps(df)
        live_fvgs = [g for g in self._all_fvgs if not g.invalidated(cl)]

        if bool(sweep_low.iloc[i]):
            self._setups.append(
                {
                    "side": "long",
                    "range_lo": lo,
                    "sweep_ms": t,
                    "pre_break_hi": hi,
                    "range_hi": hi,
                    "breakout": False,
                    "active": True,
                }
            )
        if bool(sweep_high.iloc[i]):
            self._setups.append(
                {
                    "side": "short",
                    "range_hi": hi,
                    "sweep_ms": t,
                    "pre_break_lo": lo,
                    "range_lo": lo,
                    "breakout": False,
                    "active": True,
                }
            )
        self._setups = self._setups[-20:]
        self.active_setups = sum(1 for s in self._setups if s.get("active"))

        setup_ttl = int(cfg.setup_ttl_hours * MS_1H)
        breakout_ttl = int(cfg.breakout_ttl_hours * MS_1H)

        for setup in self._setups:
            if not setup.get("active") or t - setup["sweep_ms"] > setup_ttl:
                setup["active"] = False
                continue
            if setup["side"] == "long":
                if not setup["breakout"]:
                    if cl > setup["pre_break_hi"] and cl > op:
                        setup["breakout"] = True
                        setup["range_hi"] = hi
                        setup["breakout_ms"] = t
                    setup["pre_break_hi"] = max(setup["pre_break_hi"], hi)
                    continue
                if t - setup.get("breakout_ms", t) > breakout_ttl:
                    setup["active"] = False
                    continue
                level_50 = setup["range_lo"] + M2022_ENTRY_PCT * (setup["range_hi"] - setup["range_lo"])
                pool = [
                    g
                    for g in live_fvgs
                    if g.born_ms >= setup["breakout_ms"]
                    and g.direction == 1
                    and self._fvg_key(g) not in self._used_fvg_keys
                ]
                g = nearest_fvg_to_level(pool, level_50, 1)
                if g is None:
                    continue
                entry = g.mid
                sl = setup["range_lo"]
                tp = self._tp_from_rr(entry, sl, "long")
                self._used_fvg_keys.add(self._fvg_key(g))
                setup["active"] = False
                self._place_limit_entry(side="long", entry=entry, sl=sl, tp=tp, t_ms=t)
                return
            if not setup["breakout"]:
                if cl < setup["pre_break_lo"] and cl < op:
                    setup["breakout"] = True
                    setup["range_lo"] = lo
                    setup["breakout_ms"] = t
                setup["pre_break_lo"] = min(setup["pre_break_lo"], lo)
                continue
            if t - setup.get("breakout_ms", t) > breakout_ttl:
                setup["active"] = False
                continue
            level_50 = setup["range_hi"] - M2022_ENTRY_PCT * (setup["range_hi"] - setup["range_lo"])
            pool = [
                g
                for g in live_fvgs
                if g.born_ms >= setup["breakout_ms"]
                and g.direction == -1
                and self._fvg_key(g) not in self._used_fvg_keys
            ]
            g = nearest_fvg_to_level(pool, level_50, -1)
            if g is None:
                continue
            entry = g.mid
            sl = setup["range_hi"]
            tp = self._tp_from_rr(entry, sl, "short")
            self._used_fvg_keys.add(self._fvg_key(g))
            setup["active"] = False
            self._place_limit_entry(side="short", entry=entry, sl=sl, tp=tp, t_ms=t)
            return

    def on_init(self) -> None:
        self.write_log("ICT 2022 + HMM RANGE strategy init")
        self.bg = BarGenerator(self.on_bar, 5, self.on_5min_bar, Interval.MINUTE)
        days = int(self._ict_cfg().init_bar_days)
        self.write_log(f"loading {days}d history (indicators deferred until replay done)")
        self.load_bar(days)
        self._refresh_indicators()
        self.write_log(
            f"init done bars={len(self._bars)} fvgs={len(self._all_fvgs)} hmm={self.hmm_regime}"
        )

    def on_start(self) -> None:
        self.write_log("ICT 2022 strategy start")
        if self.pos != 0 and self.stop_price <= 0 and not (self._ict_cfg().shadow or not self._ict_cfg().live_enabled):
            self.write_log("flattening unprotected synced position")
            self.cancel_all()
            vol = abs(self.pos)
            if self.pos > 0:
                self._send_market(Direction.SHORT, Offset.CLOSE, vol)
            else:
                self._send_market(Direction.LONG, Offset.CLOSE, vol)

    def on_stop(self) -> None:
        cfg = self._ict_cfg()
        shadow = cfg.shadow or not cfg.live_enabled
        if shadow and self._shadow_pos != 0:
            self._shadow_pos = 0.0
            self.entry_price = self.stop_price = self.target_price = 0.0
        elif self.pos != 0 and self._last_bar is not None:
            self.cancel_all()
            vol = abs(self.pos)
            if self.pos > 0:
                self._send_market(Direction.SHORT, Offset.CLOSE, vol)
            else:
                self._send_market(Direction.LONG, Offset.CLOSE, vol)
        self.write_log("ICT 2022 strategy stop")

    def on_tick(self, tick: TickData) -> None:
        extra = getattr(tick, "extra", None) or {}
        bar = extra.get("bar")
        if bar is not None:
            self.on_bar(bar)
            return
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        self.bg.update_bar(bar)

    def on_5min_bar(self, bar: BarData) -> None:
        self._last_bar = bar
        self._append_bar(bar)
        if not self.trading:
            return
        self._refresh_indicators()

        if self._check_pending_limit(bar):
            self.put_event()
            return

        if self._effective_pos() != 0:
            if self._check_exit_on_bar(bar):
                self.put_event()
                return
            self.put_event()
            return

        self._process_setups(bar)
        self.put_event()

    def on_trade(self, trade: TradeData) -> None:
        if trade.offset == Offset.OPEN and self.pos != 0:
            side = "long" if self.pos > 0 else "short"
            if self._pending_sl > 0:
                self._on_entry_filled(float(trade.price), side)
            else:
                self.entry_price = float(trade.price)
                self.write_log("WARNING: OPEN fill without pending SL metadata")
            self._entry_pending = False
        if self.pos == 0:
            self.cancel_all()
            self._entry_pending = False
            self._exit_pending = False
            self._shadow_pos = 0.0
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
            self._pending_limit_px = 0.0
        if order.offset == Offset.CLOSE:
            self._exit_pending = False
        self.put_event()

    def _refresh_compound_size(self) -> None:
        cfg = self._ict_cfg()
        if not cfg.compound:
            return
        from binance_fapi import fetch_mark_price

        sym = symbol_from_vt(self.vt_symbol)
        px = fetch_mark_price(sym) or 100.0
        vol = self._order_volume(px)
        if vol > 0 and self.cta_engine:
            setting = {**TradingIct2022VnpyStrategy.from_ict_config(cfg), "fixed_size": vol}
            self.cta_engine.update_strategy_setting(self.strategy_name, setting)

    def on_stop_order(self, stop_order: StopOrder) -> None:
        pass

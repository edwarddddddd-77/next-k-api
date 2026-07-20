"""Simulated broker: fills with taker fee + slippage, funding settlement, and
exit management that mirrors engine.manage_positions.

Bar-based exits (1m OHLC) use the pessimistic rule: when both stop and target
are reachable inside one bar, the stop fires. Trailing checks the OLD extreme
against the bar low/high first, then updates the extreme (also pessimistic).
Inside S11 second-level windows exits run per second on tick prices instead.
"""
import logging

log = logging.getLogger("bt.broker")

MAJOR = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"}
FEE_RATE = 0.0005


def slip_of(symbol, sid=""):
    s = 0.0001 if symbol in MAJOR else 0.00025
    return s * 2 if sid == "S11_CRASH_SCALP" else s


class SimBroker:
    def __init__(self, sim, notional=1000.0, record_equity=True):
        self.sim = sim
        self.db = sim.db
        self.notional = notional
        self.trades = []
        self.equity = []          # [(ts, cum_realized + unrealized)]
        self.realized = 0.0
        self.fees_total = 0.0
        self.funding_total = 0.0
        self.record_equity = record_equity

    # -- entries -------------------------------------------------------------

    async def on_signal(self, sig):
        price = self.sim.ws.live_price(sig.symbol)
        if not price or price <= 0:
            return None
        slip = slip_of(sig.symbol, sig.strategy)
        fill = price * (1 + slip) if sig.side == "long" else price * (1 - slip)
        qty = self.notional * (sig.size_mult or 1.0) / fill
        fee = fill * qty * FEE_RATE
        pid = self.db.open_position(
            strategy=sig.strategy, symbol=sig.symbol, side=sig.side, qty=qty,
            entry_price=fill, stop_price=sig.stop_price or None,
            take_profit=sig.take_profit or None, trail_atr=sig.trail_dist or None,
            max_hold_sec=sig.max_hold_sec, time_exit_at=sig.time_exit_at or None,
            reason=sig.reason)
        pos = self.db.positions[pid]
        pos["open_fee"] = fee
        pos["funding_cost"] = 0.0
        self.db.log_signal(sig.strategy, sig.symbol, sig.side, sig.reason, True)
        return pid

    # -- funding -------------------------------------------------------------

    def settle_funding(self, t_from, t_to):
        for pos in self.db.open_positions():
            events = self.sim.funding_events(pos["symbol"], t_from, t_to)
            for ts, rate in events:
                if pos["opened_at"] >= ts:
                    continue
                price = self.sim._minute_open(pos["symbol"], ts) or pos["entry_price"]
                sign = 1 if pos["side"] == "long" else -1
                cost = rate * pos["qty"] * price * sign
                pos["funding_cost"] = pos.get("funding_cost", 0.0) + cost

    # -- exits ---------------------------------------------------------------

    def _close(self, pos, raw_price, reason, ts):
        slip = slip_of(pos["symbol"], pos["strategy"])
        side = pos["side"]
        fill = raw_price * (1 - slip) if side == "long" else raw_price * (1 + slip)
        fee = fill * pos["qty"] * FEE_RATE
        sign = 1 if side == "long" else -1
        gross = (fill - pos["entry_price"]) * pos["qty"] * sign
        fees = fee + pos.get("open_fee", 0.0)
        funding = pos.get("funding_cost", 0.0)
        net = gross - fees - funding
        self.db.close_position(pos["id"], fill, net, reason)
        self.realized += net
        self.fees_total += fees
        self.funding_total += funding
        self.trades.append({
            "sid": pos["strategy"], "symbol": pos["symbol"], "side": side,
            "qty": pos["qty"], "entry_ts": pos["opened_at"],
            "entry_price": pos["entry_price"], "exit_ts": int(ts),
            "exit_price": fill, "gross": gross, "fees": fees,
            "funding": funding, "pnl": net,
            "reason_entry": pos.get("reason", ""), "reason_exit": reason,
            "mfe_pct": pos.get("mfe_pct", 0.0), "mae_pct": pos.get("mae_pct", 0.0),
            "hold_sec": int(ts) - pos["opened_at"]})

    async def _strategy_exit(self, pos, strategies, ctx):
        strat = strategies.get(pos["strategy"])
        if not strat:
            return None
        try:
            return await strat.check_exit(ctx, pos)
        except Exception as exc:  # noqa: BLE001
            log.warning("check_exit %s failed: %s", pos["strategy"], exc)
            return None

    async def manage_tick(self, t, ctx, strategies, skip_symbols=()):
        """Exit checks against the [t, t+60) 1m bar of each held symbol."""
        for pos in list(self.db.open_positions()):
            sym = pos["symbol"]
            if sym in skip_symbols:
                continue
            bar = self.sim.minute_bar(sym, t)
            if bar is None:
                continue
            side, sign = pos["side"], 1 if pos["side"] == "long" else -1
            hi, lo_, op = bar["high"], bar["low"], bar["open"]
            # MFE/MAE from bar extremes
            best = hi if side == "long" else lo_
            worst = lo_ if side == "long" else hi
            mfe = (best - pos["entry_price"]) / pos["entry_price"] * 100 * sign
            mae = (worst - pos["entry_price"]) / pos["entry_price"] * 100 * sign
            if mfe > pos.get("mfe_pct", 0):
                pos["mfe_pct"] = mfe
            if mae < pos.get("mae_pct", 0):
                pos["mae_pct"] = mae

            if pos["opened_at"] + pos["max_hold_sec"] < t + 60:
                self._close(pos, op, "达到最长持仓时限", t)
                continue
            if pos["time_exit_at"] and pos["time_exit_at"] < t + 60:
                self._close(pos, op, "策略时间出场", t)
                continue
            if pos["trail_atr"]:
                extreme = pos["extreme"] or pos["entry_price"]
                trail = pos["trail_atr"]
                if side == "long" and extreme - lo_ >= trail:
                    self._close(pos, extreme - trail, "移动止损触发", t)
                    continue
                if side == "short" and hi - extreme >= trail:
                    self._close(pos, extreme + trail, "移动止损触发", t)
                    continue
                pos["extreme"] = max(extreme, hi) if side == "long" else min(extreme, lo_)
            if pos["stop_price"]:
                if (side == "long" and lo_ <= pos["stop_price"]) or \
                   (side == "short" and hi >= pos["stop_price"]):
                    self._close(pos, pos["stop_price"], "止损触发", t)
                    continue
            if pos["take_profit"]:
                if (side == "long" and hi >= pos["take_profit"]) or \
                   (side == "short" and lo_ <= pos["take_profit"]):
                    self._close(pos, pos["take_profit"], "止盈触发", t)
                    continue
            # strategy check_exit reads factor buckets that refresh at 5m/1h;
            # calling it on 5m edges loses nothing vs the live 0.5s loop but
            # avoids re-deriving long factor series every simulated minute
            # (S11's second-level channel calls it every second regardless)
            if t % 300 == 0:
                reason = await self._strategy_exit(pos, strategies, ctx)
                if reason:
                    self._close(pos, bar["close"], reason, t + 59)

        if self.record_equity:
            upnl = 0.0
            for p in self.db.open_positions():
                price = self.sim.ws.live_price(p["symbol"])
                if price:
                    s = 1 if p["side"] == "long" else -1
                    upnl += (price - p["entry_price"]) * p["qty"] * s
            self.equity.append((t, self.realized + upnl))

    async def manage_second(self, sym, price, ctx, strategies, t):
        """Second-level exit management inside an S11 window (exact prices)."""
        for pos in list(self.db.open_positions(symbol=sym)):
            side, sign = pos["side"], 1 if pos["side"] == "long" else -1
            exc = (price - pos["entry_price"]) / pos["entry_price"] * 100 * sign
            if exc > pos.get("mfe_pct", 0):
                pos["mfe_pct"] = exc
            if exc < pos.get("mae_pct", 0):
                pos["mae_pct"] = exc
            if t - pos["opened_at"] >= pos["max_hold_sec"]:
                self._close(pos, price, "达到最长持仓时限", t)
                continue
            if pos["time_exit_at"] and t >= pos["time_exit_at"]:
                self._close(pos, price, "策略时间出场", t)
                continue
            if pos["trail_atr"]:
                extreme = pos["extreme"] or pos["entry_price"]
                if side == "long":
                    if price > extreme:
                        pos["extreme"] = extreme = price
                    if extreme - price >= pos["trail_atr"]:
                        self._close(pos, price, "移动止损触发", t)
                        continue
                else:
                    if price < extreme:
                        pos["extreme"] = extreme = price
                    if price - extreme >= pos["trail_atr"]:
                        self._close(pos, price, "移动止损触发", t)
                        continue
            if pos["stop_price"]:
                if (side == "long" and price <= pos["stop_price"]) or \
                   (side == "short" and price >= pos["stop_price"]):
                    self._close(pos, price, "止损触发", t)
                    continue
            if pos["take_profit"]:
                if (side == "long" and price >= pos["take_profit"]) or \
                   (side == "short" and price <= pos["take_profit"]):
                    self._close(pos, price, "止盈触发", t)
                    continue
            reason = await self._strategy_exit(pos, strategies, ctx)
            if reason:
                self._close(pos, price, reason, t)

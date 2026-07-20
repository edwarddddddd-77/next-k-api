"""Order execution: paper (simulated fills, virtual balance) or live (Binance
USDⓈ-M market orders). Mode stored in DB meta; default paper. Every close is
appended to the trade journal (journal/) with entry/exit factor snapshots.
"""
import json
import logging

from . import binance, config, db, exchanges, factors, journal

log = logging.getLogger("executor")


def mode():
    return db.get_meta("mode", "paper")


def set_mode(m):
    db.set_meta("mode", "paper" if m != "live" else "live")


def exchange():
    """Live execution venue: binance (default) | bitget | okx."""
    v = db.get_meta("executor_exchange", "binance")
    return v if v in exchanges.SUPPORTED else "binance"


def set_exchange(venue):
    if venue not in exchanges.SUPPORTED:
        raise ValueError(f"unsupported exchange {venue}")
    if not exchanges.has_credentials(venue):
        raise ValueError(f"{venue} credentials not configured")
    db.set_meta("executor_exchange", venue)


def paper_balance():
    v = db.get_meta("paper_balance", "")
    if not v:
        db.set_meta("paper_balance", config.PAPER_START_BALANCE)
        return config.PAPER_START_BALANCE
    return float(v)


def _adjust_paper_balance(delta):
    db.set_meta("paper_balance", paper_balance() + delta)


async def equity():
    """(equity, balance, unrealized) in USDT for the active mode."""
    if mode() == "live":
        try:
            venue = exchange()
            if venue == "binance":
                acc = await binance.account()
                bal = float(acc.get("totalWalletBalance") or 0)
                upnl = float(acc.get("totalUnrealizedProfit") or 0)
                return bal + upnl, bal, upnl
            total, _avail = await exchanges.equity(venue)
            return total, total, 0.0
        except Exception as exc:  # noqa: BLE001
            log.warning("live account fetch failed: %s", exc)
    bal = paper_balance()
    upnl = 0.0
    for p in db.open_positions():
        price = factors.mark_price_of(p["symbol"])
        if price:
            sign = 1 if p["side"] == "long" else -1
            upnl += (price - p["entry_price"]) * p["qty"] * sign
    return bal + upnl, bal, upnl


async def open_position(sig, qty, price, leverage=None):
    """Fill an entry. Returns position id or None."""
    if qty <= 0 or price <= 0:
        return None
    if mode() == "live":
        venue = exchange()
        side = "BUY" if sig.side == "long" else "SELL"
        try:
            if venue == "binance":
                step = await binance.qty_step(sig.symbol)
                qty = binance.round_step(qty, step)
                if qty <= 0:
                    return None
                if leverage:
                    await binance.set_leverage(sig.symbol, leverage)
                await binance.place_market_order(sig.symbol, side, qty)
            else:
                if leverage:
                    await exchanges.set_leverage(venue, sig.symbol, leverage)
                await exchanges.place_market(venue, sig.symbol, side, qty)
        except Exception as exc:  # noqa: BLE001
            db.log("error", f"live[{venue}] 开仓失败 {sig.symbol} {sig.side}: {exc}")
            return None
        fill = price
        fee = fill * qty * 0.0005
    else:
        slip = 1 + config.PAPER_SLIPPAGE if sig.side == "long" else 1 - config.PAPER_SLIPPAGE
        fill = price * slip
        fee = fill * qty * config.PAPER_FEE_RATE
        _adjust_paper_balance(-fee)
    pid = db.open_position(
        strategy=sig.strategy, symbol=sig.symbol, side=sig.side, qty=qty,
        entry_price=fill, stop_price=sig.stop_price or None,
        take_profit=sig.take_profit or None,
        trail_atr=sig.trail_dist or None,
        max_hold_sec=sig.max_hold_sec,
        time_exit_at=sig.time_exit_at or None,
        entry_factors=json.dumps(factors.snapshot_for(sig.symbol), ensure_ascii=False),
        reason=sig.reason)
    db.log_trade(sig.strategy, sig.symbol, sig.side, qty, fill, fee, "open", pid, None, mode())
    db.log("info", f"开仓 [{sig.strategy}] {sig.symbol} {sig.side} qty={qty:.6g} @ {fill:.6g} — {sig.reason}")
    return pid


async def close_position(pos, price, reason):
    qty, side = pos["qty"], pos["side"]
    if mode() == "live":
        venue = exchange()
        order_side = "SELL" if side == "long" else "BUY"
        try:
            if venue == "binance":
                step = await binance.qty_step(pos["symbol"])
                await binance.place_market_order(pos["symbol"], order_side,
                                                 binance.round_step(qty, step),
                                                 reduce_only=True)
            else:
                await exchanges.place_market(venue, pos["symbol"], order_side,
                                             qty, reduce_only=True)
        except Exception as exc:  # noqa: BLE001
            db.log("error", f"live[{venue}] 平仓失败 {pos['symbol']}: {exc}")
            return False
        fill = price
        fee = fill * qty * 0.0005
    else:
        slip = 1 - config.PAPER_SLIPPAGE if side == "long" else 1 + config.PAPER_SLIPPAGE
        fill = price * slip
        fee = fill * qty * config.PAPER_FEE_RATE
    sign = 1 if side == "long" else -1
    pnl = (fill - pos["entry_price"]) * qty * sign - fee
    if mode() != "live":
        _adjust_paper_balance(pnl)
    db.close_position(pos["id"], fill, pnl, reason)
    db.log_trade(pos["strategy"], pos["symbol"],
                 "sell" if side == "long" else "buy", qty, fill, fee,
                 "close", pos["id"], pnl, mode())
    journal.record_close(pos, fill, pnl, fee, reason,
                         factors.snapshot_for(pos["symbol"]), mode())
    db.log("info", f"平仓 [{pos['strategy']}] {pos['symbol']} {side} pnl={pnl:+.2f} — {reason}")
    return True

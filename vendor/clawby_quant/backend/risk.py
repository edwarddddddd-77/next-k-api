"""Global risk layer — every entry signal passes through here.

Rules (from STRATEGIES.md): event quiet window, fear&greed sizing, per-coin
cap, gross leverage cap, daily loss halt, max concurrent coins, no hedged
same-coin positions.
"""
import time

from . import db, factors

QUIET_EXEMPT = {"S09_EVENT_BREAKOUT"}


def _day_start():
    return int(time.time()) // 86400 * 86400


def daily_realized_pnl():
    return sum(p["pnl"] or 0 for p in db.closed_positions_today(_day_start()))


def halted():
    until = float(db.get_meta("halt_until", "0") or 0)
    return time.time() < until


def check_daily_halt(equity, cfg_global):
    pnl = daily_realized_pnl()
    if equity > 0 and pnl < 0 and abs(pnl) / equity * 100 >= cfg_global["daily_loss_halt_pct"]:
        tomorrow = _day_start() + 86400
        db.set_meta("halt_until", tomorrow)
        db.log("warn", f"日内熔断触发:已实现亏损 {pnl:.2f} ({abs(pnl)/equity*100:.1f}%),停止开仓至次日 UTC0")
        return True
    return False


def event_quiet(cfg_global):
    """True while inside the pre-event quiet window of a high-importance event."""
    cal = (db.get_factor("econ_cal") or {}).get("events", [])
    now = time.time()
    window = cfg_global["event_quiet_minutes"] * 60
    for e in cal:
        imp = str(e.get("importance", ""))
        if any(x in imp for x in ("3", "高", "high", "High")) and 0 < e["ts"] - now <= window:
            return e.get("title", "event")
    return ""


def size_multiplier(side, cfg_global):
    fg = db.get_factor("fear_greed") or {}
    v = fg.get("latest")
    ext = cfg_global["fear_greed_extreme"]
    if v is None:
        return 1.0
    if side == "long" and v >= ext["greed"]:
        return ext["size_mult"]
    if side == "short" and v <= ext["fear"]:
        return ext["size_mult"]
    return 1.0


def _strategy_capacity(sig, scfg, equity, cfg_global):
    """(capital_base, leverage, max_position_usd, used_notional) for a strategy."""
    risk = scfg.get("risk") or {}
    capital = float(risk.get("capital_usd") or 0) or equity
    leverage = int(risk.get("leverage") or cfg_global.get("max_gross_leverage", 3))
    max_pos = float(risk.get("max_position_usd") or 0) or capital * leverage
    used = sum(abs(p["qty"] * factors.mark_price_of(p["symbol"]))
               for p in db.open_positions(strategy=sig.strategy))
    return capital, leverage, max_pos, used


def allow_entry(sig, scfg, equity, cfg_global):
    """Returns (allowed: bool, reason: str, size_mult: float)."""
    if halted():
        return False, "日内熔断中", 0
    if sig.strategy not in QUIET_EXEMPT:
        ev = event_quiet(cfg_global)
        if ev:
            return False, f"事件静默({ev[:30]})", 0
    open_pos = db.open_positions()
    coins = {p["symbol"] for p in open_pos}
    if sig.symbol in coins:
        return False, "同币已有仓位(禁止对锁/加仓)", 0
    if len(coins) >= cfg_global["max_concurrent_coins"]:
        return False, "并发币数已达上限", 0
    gross = sum(abs(p["qty"] * factors.mark_price_of(p["symbol"])) for p in open_pos)
    if equity > 0 and gross / equity >= cfg_global["max_gross_leverage"]:
        return False, "总敞口已达杠杆上限", 0
    capital, leverage, _max_pos, used = _strategy_capacity(sig, scfg, equity, cfg_global)
    if used >= capital * leverage:
        return False, "策略分配资金已用满", 0
    return True, "", size_multiplier(sig.side, cfg_global)


def position_qty(sig, scfg, equity, cfg_global, price):
    """Per-strategy risk-based sizing:
    qty = (capital_base × risk_per_trade%) / stop_distance,
    then capped by the strategy's max_position_usd, its remaining
    capital×leverage capacity, and the global per-coin exposure limit."""
    if price <= 0:
        return 0
    stop_dist = abs(price - sig.stop_price) if sig.stop_price else price * 0.01
    if stop_dist <= 0:
        return 0
    risk = scfg.get("risk") or {}
    capital, leverage, max_pos, used = _strategy_capacity(sig, scfg, equity, cfg_global)
    rpt = float(risk.get("risk_per_trade_pct") or 0) or cfg_global["risk_per_trade_pct"]
    risk_usd = capital * rpt / 100 * sig.size_mult
    qty = risk_usd / stop_dist
    notional = qty * price
    notional = min(notional, max_pos)                       # per-position cap
    notional = min(notional, max(capital * leverage - used, 0))  # strategy capacity left
    notional = min(notional, equity * cfg_global["max_position_pct_per_coin"] / 100
                   * leverage)                              # global per-coin cap (×lev)
    return notional / price

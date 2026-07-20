"""Engine: two independent 0.5s loops.

- position_loop: every 0.5s manage open positions (stops / take-profit /
  trailing / time exits / 24h cap) against real-time WebSocket marks.
- scan_loop: every 0.5s refresh due factors and fire strategies whose own
  scan_interval has elapsed (data granularity still gates each strategy;
  trigger latency is now sub-second instead of a 30s tick).
Bookkeeping (equity snapshot, halt check, meta) is throttled to ~10s.
"""
import asyncio
import logging
import time

from . import config, db, executor, factors, risk
from .strategies import REGISTRY, Ctx

log = logging.getLogger("engine")

LOOP_SEC = 0.5

_INTERVALS = {"1s": 1, "5s": 5, "15s": 15, "30s": 30, "1m": 60, "5m": 300,
              "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400}
_last_scan = {}


def _interval_sec(v):
    if isinstance(v, (int, float)):
        return max(1, int(v))
    return _INTERVALS.get(str(v), 900)


def _due(sid, scan_interval):
    if time.time() - _last_scan.get(sid, 0) >= _interval_sec(scan_interval) - 0.25:
        _last_scan[sid] = time.time()
        return True
    return False


def _strategies(cfg):
    out = {}
    for iid, scfg in cfg.get("strategies", {}).items():
        cls = REGISTRY.get(scfg.get("base") or iid)
        if cls:
            out[iid] = cls({**scfg, "_iid": iid})
    return out


async def manage_positions(ctx, strategies):
    for pos in db.open_positions():
        price = factors.mark_price_of(pos["symbol"])
        if not price:
            continue
        side, now = pos["side"], time.time()
        # MFE/MAE excursion tracking (monotonic -> sparse writes) for the journal
        sign = 1 if side == "long" else -1
        exc_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100 * sign
        if exc_pct > (pos.get("mfe_pct") or 0):
            db.update_position(pos["id"], mfe_pct=exc_pct)
            pos["mfe_pct"] = exc_pct
        if exc_pct < (pos.get("mae_pct") or 0):
            db.update_position(pos["id"], mae_pct=exc_pct)
            pos["mae_pct"] = exc_pct
        reason = None
        if now - pos["opened_at"] >= pos["max_hold_sec"]:
            reason = "达到最长持仓时限"
        elif pos["time_exit_at"] and now >= pos["time_exit_at"]:
            reason = "策略时间出场"
        if not reason and pos["trail_atr"]:
            extreme = pos["extreme"] or pos["entry_price"]
            if side == "long":
                if price > extreme:
                    db.update_position(pos["id"], extreme=price)
                    extreme = price
                if extreme - price >= pos["trail_atr"]:
                    reason = "移动止损触发"
            else:
                if price < extreme:
                    db.update_position(pos["id"], extreme=price)
                    extreme = price
                if price - extreme >= pos["trail_atr"]:
                    reason = "移动止损触发"
        if not reason and pos["stop_price"]:
            if (side == "long" and price <= pos["stop_price"]) or \
               (side == "short" and price >= pos["stop_price"]):
                reason = "止损触发"
        if not reason and pos["take_profit"]:
            if (side == "long" and price >= pos["take_profit"]) or \
               (side == "short" and price <= pos["take_profit"]):
                reason = "止盈触发"
        if not reason:
            strat = strategies.get(pos["strategy"])
            if strat:
                try:
                    reason = await strat.check_exit(ctx, pos)
                except Exception as exc:  # noqa: BLE001
                    log.warning("check_exit %s failed: %s", pos["strategy"], exc)
        if reason:
            await executor.close_position(pos, price, reason)


async def position_loop():
    log.info("position loop started (%.1fs, ws-priced)", LOOP_SEC)
    while True:
        started = time.monotonic()
        try:
            cfg = config.load_strategies()
            ctx = Ctx(config.UNIVERSE)
            await manage_positions(ctx, _strategies(cfg))
            db.set_meta("last_manage_ts_ms", int(time.time() * 1000))
        except Exception as exc:  # noqa: BLE001
            log.exception("position loop tick failed")
            db.set_meta("last_error", f"{int(time.time())}: manage: {exc}")
        await asyncio.sleep(max(LOOP_SEC - (time.monotonic() - started), 0.05))


async def scan_loop():
    log.info("scan loop started (%.1fs scheduler)", LOOP_SEC)
    db.set_meta("started_at", int(time.time()))
    last_book = 0.0
    equity_cache = config.PAPER_START_BALANCE
    while True:
        started = time.monotonic()
        try:
            cfg = config.load_strategies()
            cfg_global = cfg["global"]

            n = await factors.collect_due()
            if n:
                log.info("factors refreshed: %d", n)

            # bookkeeping ~10s: equity snapshot + halt check
            if time.time() - last_book >= 10:
                eq, bal, upnl = await executor.equity()
                equity_cache = eq
                db.record_equity(eq, bal, upnl)
                risk.check_daily_halt(eq, cfg_global)
                last_book = time.time()

            strategies = _strategies(cfg)
            ctx = Ctx(config.UNIVERSE)
            for sid, scfg in cfg.get("strategies", {}).items():
                if not scfg.get("enabled") or sid not in strategies:
                    continue
                if not _due(sid, scfg.get("scan_interval", "15m")):
                    continue
                try:
                    signals = await strategies[sid].evaluate(ctx)
                except Exception as exc:  # noqa: BLE001
                    log.exception("strategy %s evaluate failed", sid)
                    db.log("error", f"{sid} 评估异常: {exc}")
                    continue
                for sig in signals:
                    allowed, why, mult = risk.allow_entry(sig, scfg, equity_cache, cfg_global)
                    if not allowed:
                        db.log_signal(sig.strategy, sig.symbol, sig.side, sig.reason, False, why)
                        continue
                    sig.size_mult *= mult
                    price = factors.mark_price_of(sig.symbol)
                    qty = risk.position_qty(sig, scfg, equity_cache, cfg_global, price)
                    leverage = int((scfg.get("risk") or {}).get("leverage")
                                   or cfg_global.get("max_gross_leverage", 3))
                    pid = await executor.open_position(sig, qty, price, leverage)
                    db.log_signal(sig.strategy, sig.symbol, sig.side, sig.reason,
                                  bool(pid), "" if pid else "执行失败/资金不足")
            db.set_meta("last_tick_ts", int(time.time()))
            db.set_meta("last_error", "")
        except Exception as exc:  # noqa: BLE001
            log.exception("scan loop tick failed")
            db.set_meta("last_error", f"{int(time.time())}: {exc}")
        await asyncio.sleep(max(LOOP_SEC - (time.monotonic() - started), 0.05))

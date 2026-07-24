"""7d desk paper backtest — final portfolio risk scheme.

Rules (aligned with utils/hl_paper_copy.py):
  - Desk return = sum of ALL bots (halted still count)
  - Soft TP +5.5%: cut open size to 50% once/cycle; copy_scale=0.5 until hard
  - Hard TP +8% / Hard SL −3% / ≥3 bots halted: flatten ALL, compound rebase,
    copy_scale=1, clear soft flag + bot halts
  - Per-bot −20% vs cycle anchor: flatten that bot + halt; unlock on hard
    rebase OR 6h cooldown
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.hl_paper_copy import (  # noqa: E402
    is_hl_spot_coin,
    paper_config,
    target_empty_av,
    _adjusted_leverage,
)
from utils.hl_short_term import load_watchlist  # noqa: E402
from utils.hl_wr_screen import _fetch_fills_7d, _hl_info  # noqa: E402

BJ = timezone(timedelta(hours=8))
OUT = ROOT / "hl_desk_portfolio_risk_7d.json"


def _ts_ms(x: Any) -> int:
    try:
        return int(x or 0)
    except (TypeError, ValueError):
        return 0


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _parse_signed(fill: dict) -> tuple[str, float, float] | None:
    coin = str(fill.get("coin") or "").strip().upper()
    if not coin or is_hl_spot_coin(coin):
        return None
    px = _f(fill.get("px"))
    sz = abs(_f(fill.get("sz")))
    if px <= 0 or sz <= 0:
        return None
    side = str(fill.get("side") or "").strip().upper()
    if side in ("B", "BUY"):
        signed = sz
    elif side in ("A", "SELL"):
        signed = -sz
    else:
        direction = str(fill.get("dir") or "").strip().lower()
        if "open long" in direction or "close short" in direction:
            signed = sz
        elif "open short" in direction or "close long" in direction:
            signed = -sz
        else:
            return None
    return coin, signed, px


def _av_series(addr: str) -> list[tuple[int, float]]:
    port = _hl_info({"type": "portfolio", "user": addr})
    series: list[tuple[int, float]] = []
    if not isinstance(port, list):
        return series
    for item in port:
        if not isinstance(item, list) or len(item) < 2:
            continue
        if item[0] != "week":
            continue
        hist = (item[1] or {}).get("accountValueHistory") or []
        for row in hist:
            if not isinstance(row, list) or len(row) < 2:
                continue
            series.append((_ts_ms(row[0]), _f(row[1])))
    series.sort(key=lambda x: x[0])
    return series


def _av_at(series: list[tuple[int, float]], t_ms: int) -> float:
    if not series:
        return 0.0
    best = series[0][1]
    for ts, av in series:
        if ts > t_ms:
            break
        best = av
    return max(0.0, best)


@dataclass
class BotState:
    id: str
    address: str
    balance: float
    init_balance: float = 0.0
    positions: dict[str, dict[str, float]] = field(default_factory=dict)
    realized: float = 0.0
    marks: dict[str, float] = field(default_factory=dict)
    risk_halted: bool = False
    risk_halted_at_ms: int = 0
    risk_anchor: float = 0.0

    def equity(self) -> float:
        upnl = 0.0
        for coin, pos in self.positions.items():
            sz = pos["sz"]
            entry = pos["entry"]
            mid = self.marks.get(coin) or entry
            if sz > 0:
                upnl += (mid - entry) * sz
            else:
                upnl += (entry - mid) * abs(sz)
        return self.balance + upnl


def _realize(bot: BotState, coin: str, exit_px: float, close_sz: float) -> float:
    pos = bot.positions.get(coin)
    if not pos:
        return 0.0
    entry = pos["entry"]
    signed = pos["sz"]
    qty = min(abs(signed), abs(close_sz))
    if qty <= 1e-16:
        return 0.0
    if signed > 0:
        pnl = (exit_px - entry) * qty
    else:
        pnl = (entry - exit_px) * qty
    bot.balance += pnl
    bot.realized += pnl
    remain = abs(signed) - qty
    if remain <= 1e-12:
        bot.positions.pop(coin, None)
    else:
        pos["sz"] = math.copysign(remain, signed)
    return pnl


def _apply_delta(bot: BotState, coin: str, delta: float, px: float, max_n: float) -> None:
    old = bot.positions.get(coin)
    old_sz = float(old["sz"]) if old else 0.0
    raw_new = old_sz + delta
    if abs(raw_new) * px > max_n + 1e-9:
        cap_sz = math.copysign(max_n / px, raw_new)
        if old_sz * raw_new < 0:
            _realize(bot, coin, px, abs(old_sz))
            raw_new = cap_sz
            old_sz = 0.0
        elif abs(raw_new) > abs(old_sz):
            raw_new = math.copysign(min(abs(raw_new), abs(cap_sz)), raw_new)
            if abs(raw_new - old_sz) < 1e-12:
                return
    if abs(old_sz) > 1e-16 and old_sz * raw_new < 0:
        _realize(bot, coin, px, abs(old_sz))
        old_sz = 0.0
        old = None
    if abs(raw_new) < 1e-16:
        if old:
            _realize(bot, coin, px, abs(old_sz))
        return
    if abs(raw_new) < abs(old_sz) - 1e-12:
        _realize(bot, coin, px, abs(old_sz) - abs(raw_new))
        if coin in bot.positions:
            bot.positions[coin]["sz"] = raw_new
        return
    add = abs(raw_new) - abs(old_sz)
    if old and abs(old_sz) > 1e-16 and old_sz * raw_new > 0 and add > 1e-16:
        entry = (old["entry"] * abs(old_sz) + px * add) / (abs(old_sz) + add)
        bot.positions[coin] = {"sz": raw_new, "entry": entry}
    else:
        bot.positions[coin] = {"sz": raw_new, "entry": px}
    bot.marks[coin] = px


def _flatten_bot(bot: BotState, px_by_coin: dict[str, float] | None = None) -> None:
    for coin in list(bot.positions.keys()):
        px = (px_by_coin or {}).get(coin) or bot.marks.get(coin) or bot.positions[coin]["entry"]
        _realize(bot, coin, px, abs(bot.positions[coin]["sz"]))


def _reduce_bot(bot: BotState, keep_frac: float) -> None:
    keep = max(0.0, min(1.0, keep_frac))
    for coin in list(bot.positions.keys()):
        pos = bot.positions[coin]
        sz = pos["sz"]
        target = sz * keep
        cut = abs(sz) - abs(target)
        if cut <= 1e-16:
            continue
        px = bot.marks.get(coin) or pos["entry"]
        _realize(bot, coin, px, cut)
        if coin in bot.positions:
            bot.positions[coin]["sz"] = target


def _lev_for_coin(coin: str) -> int:
    """Match live caps (BTC/ETH up to 50). Fills lack leverage — use asset max."""
    return _adjusted_leverage(50.0, 1.0, coin)


def _mark_all(bots: dict[str, BotState], coin: str, px: float) -> None:
    """Live marks via all_mids every cycle; replay any print across the desk."""
    for b in bots.values():
        if coin in b.positions or coin in b.marks:
            b.marks[coin] = px


def _check_bot_halts(
    bots: dict[str, BotState],
    *,
    bot_sl: float,
    t_ms: int,
    trips: list[dict[str, Any]],
    trigger_bot: str,
    trigger_coin: str,
) -> None:
    if bot_sl <= 0:
        return
    for bid, b in bots.items():
        if b.risk_halted or b.risk_anchor <= 1e-9:
            continue
        bot_ret = (b.equity() - b.risk_anchor) / b.risk_anchor
        if bot_ret > -bot_sl:
            continue
        eq_before = b.equity()
        _flatten_bot(b)
        b.risk_halted = True
        b.risk_halted_at_ms = t_ms
        trips.append(
            {
                "t": datetime.fromtimestamp(t_ms / 1000, BJ).isoformat(),
                "reason": "bot_sl_halt",
                "ret_pct": round(bot_ret * 100, 3),
                "equity_before": round(eq_before, 2),
                "anchor_before": round(b.risk_anchor, 2),
                "trigger_bot": bid,
                "via_fill_bot": trigger_bot,
                "trigger_coin": trigger_coin,
            }
        )


def _halted_count(bots: dict[str, BotState]) -> int:
    return sum(1 for b in bots.values() if b.risk_halted)


def _desk_equity(bots: dict[str, BotState]) -> float:
    return sum(b.equity() for b in bots.values())


def _reset_bot_risk(bot: BotState) -> None:
    bot.risk_halted = False
    bot.risk_halted_at_ms = 0
    bot.risk_anchor = bot.equity()


def _hard_rebase(
    bots: dict[str, BotState],
    *,
    reason: str,
    ret: float,
    anchor: float,
    equity: float,
    t_ms: int,
    trips: list[dict[str, Any]],
    eq_curve: list[dict[str, Any]],
    trigger_bot: str = "",
    trigger_coin: str = "",
) -> tuple[float, float, bool]:
    """Flatten all, clear halts/soft, return (new_anchor, copy_scale=1, soft_taken=False)."""
    for b in bots.values():
        if b.positions:
            _flatten_bot(b)
        _reset_bot_risk(b)
    new_eq = _desk_equity(bots)
    trips.append(
        {
            "t": datetime.fromtimestamp(t_ms / 1000, BJ).isoformat(),
            "reason": reason,
            "ret_pct": round(ret * 100, 3),
            "equity_before": round(equity, 2),
            "anchor_before": round(anchor, 2),
            "anchor_after": round(new_eq, 2),
            "trigger_bot": trigger_bot,
            "trigger_coin": trigger_coin,
            "halted_count": _halted_count(bots),
        }
    )
    eq_curve.append(
        {
            "t": datetime.fromtimestamp(t_ms / 1000, BJ).isoformat(),
            "equity": round(new_eq, 2),
            "anchor": round(new_eq, 2),
            "ret_pct": 0.0,
            "trip": reason,
            "copy_scale": 1.0,
        }
    )
    return new_eq, 1.0, False


def _load_market(
    wallets: list[dict[str, Any]], start_ms: int
) -> tuple[dict[str, list[tuple[int, float]]], list[tuple[int, str, dict]]]:
    av_map: dict[str, list[tuple[int, float]]] = {}
    events: list[tuple[int, str, dict]] = []
    for w in wallets:
        bid = str(w["id"])
        addr = str(w["address"])
        print(f"fetch {bid} …", flush=True)
        time.sleep(0.35)
        av_map[bid] = _av_series(addr)
        time.sleep(0.35)
        fills = _fetch_fills_7d(addr, start_ms)
        n_perp = 0
        for f in fills:
            parsed = _parse_signed(f)
            if not parsed:
                continue
            n_perp += 1
            events.append((_ts_ms(f.get("time")), bid, f))
        print(f"  fills={len(fills)} perp={n_perp} av_pts={len(av_map[bid])}", flush=True)
    events.sort(key=lambda x: (x[0], x[1]))
    return av_map, events


def _cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    raw = cfg.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _cfg_int(cfg: dict[str, Any], key: str, default: int) -> int:
    return int(_cfg_float(cfg, key, float(default)))


def run(
    *,
    av_map: dict[str, list[tuple[int, float]]],
    events: list[tuple[int, str, dict]],
    wallets: list[dict[str, Any]],
    start_ms: int,
    now_ms: int,
    use_risk: bool,
) -> dict[str, Any]:
    cfg = paper_config()
    empty_thr = target_empty_av()

    soft_tp = _cfg_float(cfg, "portfolio_tp_pct", 0.0)
    hard_tp = _cfg_float(cfg, "portfolio_tp_hard_pct", 0.0)
    sl = _cfg_float(cfg, "portfolio_sl_pct", 0.0)
    soft_keep = _cfg_float(cfg, "portfolio_soft_reduce", 0.5)
    bot_sl = _cfg_float(cfg, "daily_loss_pct", 0.20)
    halt_trigger = _cfg_int(cfg, "portfolio_halt_count_trigger", 0)
    cool_ms = int(_cfg_float(cfg, "bot_halt_cooldown_sec", 6 * 3600) * 1000)

    bots: dict[str, BotState] = {}
    for w in wallets:
        bid = str(w["id"])
        bal = float(w.get("paper_balance") or cfg["bot_balance"] or 1000)
        bots[bid] = BotState(
            id=bid,
            address=str(w["address"]),
            balance=bal,
            init_balance=bal,
            risk_anchor=bal,
        )

    init_eq = _desk_equity(bots)
    anchor = init_eq
    copy_scale = 1.0
    soft_taken = False
    trips: list[dict[str, Any]] = []
    eq_curve: list[dict[str, Any]] = []
    last_curve_t = 0
    min_eq: dict[str, float] = {bid: bal for bid, bal in ((b.id, b.balance) for b in bots.values())}

    for t_ms, bid, fill in events:
        bot = bots[bid]
        parsed = _parse_signed(fill)
        if not parsed:
            continue
        coin, signed, px = parsed
        # Mark every bot holding/touched this coin (live uses all_mids).
        _mark_all(bots, coin, px)

        # Cooldown unlock
        if use_risk and cool_ms > 0:
            for b in bots.values():
                if (
                    b.risk_halted
                    and b.risk_halted_at_ms > 0
                    and t_ms - b.risk_halted_at_ms >= cool_ms
                ):
                    _reset_bot_risk(b)

        av = _av_at(av_map[bid], t_ms)
        if av < empty_thr:
            if bot.positions:
                _flatten_bot(bot, {coin: px})
            if use_risk:
                _check_bot_halts(
                    bots,
                    bot_sl=bot_sl,
                    t_ms=t_ms,
                    trips=trips,
                    trigger_bot=bid,
                    trigger_coin=coin,
                )
            continue

        if not (use_risk and bot.risk_halted):
            eq = bot.equity()
            ratio = (eq / av) * copy_scale if av > 1e-9 and eq > 0 else 0.0
            if ratio > 0:
                our_delta = signed * ratio
                lev = _lev_for_coin(coin)
                max_n = eq * float(lev)
                _apply_delta(bot, coin, our_delta, px, max_n)

        for b in bots.values():
            min_eq[b.id] = min(min_eq[b.id], b.equity())

        desk_eq = _desk_equity(bots)
        if t_ms - last_curve_t >= 3600_000 or not eq_curve:
            eq_curve.append(
                {
                    "t": datetime.fromtimestamp(t_ms / 1000, BJ).isoformat(),
                    "equity": round(desk_eq, 2),
                    "anchor": round(anchor, 2),
                    "ret_pct": round((desk_eq - anchor) / anchor * 100, 3) if anchor else 0.0,
                    "copy_scale": copy_scale,
                    "halted": _halted_count(bots),
                }
            )
            last_curve_t = t_ms

        if not use_risk or anchor <= 0:
            continue

        # Per-bot −20% on ALL bots after MTM (not only the fill bot)
        _check_bot_halts(
            bots,
            bot_sl=bot_sl,
            t_ms=t_ms,
            trips=trips,
            trigger_bot=bid,
            trigger_coin=coin,
        )

        desk_eq = _desk_equity(bots)
        n_halt = _halted_count(bots)
        active_left = any(not b.risk_halted for b in bots.values())

        # Multi-halt → hard (disabled when halt_trigger ≤ 0)
        if halt_trigger > 0 and (
            n_halt >= halt_trigger or (n_halt > 0 and not active_left)
        ):
            ret = (desk_eq - anchor) / anchor if anchor > 1e-9 else -1.0
            anchor, copy_scale, soft_taken = _hard_rebase(
                bots,
                reason="portfolio_multi_halt",
                ret=ret,
                anchor=anchor,
                equity=desk_eq,
                t_ms=t_ms,
                trips=trips,
                eq_curve=eq_curve,
                trigger_bot=bid,
                trigger_coin=coin,
            )
            continue

        ret = (desk_eq - anchor) / anchor
        if sl > 0 and ret <= -sl:
            anchor, copy_scale, soft_taken = _hard_rebase(
                bots,
                reason="portfolio_sl",
                ret=ret,
                anchor=anchor,
                equity=desk_eq,
                t_ms=t_ms,
                trips=trips,
                eq_curve=eq_curve,
                trigger_bot=bid,
                trigger_coin=coin,
            )
            continue
        if hard_tp > 0 and ret >= hard_tp:
            anchor, copy_scale, soft_taken = _hard_rebase(
                bots,
                reason="portfolio_tp_hard",
                ret=ret,
                anchor=anchor,
                equity=desk_eq,
                t_ms=t_ms,
                trips=trips,
                eq_curve=eq_curve,
                trigger_bot=bid,
                trigger_coin=coin,
            )
            continue
        if soft_tp > 0 and (not soft_taken) and ret >= soft_tp:
            for b in bots.values():
                if b.risk_halted or not b.positions:
                    continue
                _reduce_bot(b, soft_keep)
            soft_taken = True
            copy_scale = soft_keep
            new_eq = _desk_equity(bots)
            trips.append(
                {
                    "t": datetime.fromtimestamp(t_ms / 1000, BJ).isoformat(),
                    "reason": "portfolio_tp_soft",
                    "ret_pct": round(ret * 100, 3),
                    "equity_before": round(desk_eq, 2),
                    "equity_after": round(new_eq, 2),
                    "anchor_before": round(anchor, 2),
                    "copy_scale": copy_scale,
                    "trigger_bot": bid,
                    "trigger_coin": coin,
                }
            )
            eq_curve.append(
                {
                    "t": datetime.fromtimestamp(t_ms / 1000, BJ).isoformat(),
                    "equity": round(new_eq, 2),
                    "anchor": round(anchor, 2),
                    "ret_pct": round((new_eq - anchor) / anchor * 100, 3) if anchor else 0.0,
                    "trip": "portfolio_tp_soft",
                    "copy_scale": copy_scale,
                }
            )

    final_eq = _desk_equity(bots)
    open_n = sum(len(b.positions) for b in bots.values())
    per_bot = []
    for bid, b in bots.items():
        per_bot.append(
            {
                "id": bid,
                "equity": round(b.equity(), 2),
                "balance": round(b.balance, 2),
                "realized": round(b.realized, 2),
                "open_positions": len(b.positions),
                "risk_halted": b.risk_halted,
                "min_equity": round(min_eq.get(bid, b.equity()), 2),
                "min_dd_pct": round(
                    (min_eq.get(bid, b.init_balance) - b.init_balance) / b.init_balance * 100, 2
                ),
                "pnl": round(b.equity() - b.init_balance, 2),
            }
        )

    by_reason: dict[str, int] = {}
    for t in trips:
        r = str(t.get("reason") or "")
        by_reason[r] = by_reason.get(r, 0) + 1

    return {
        "window_start": datetime.fromtimestamp(start_ms / 1000, BJ).isoformat(),
        "window_end": datetime.fromtimestamp(now_ms / 1000, BJ).isoformat(),
        "soft_tp_pct": soft_tp,
        "hard_tp_pct": hard_tp,
        "sl_pct": sl,
        "bot_sl_pct": bot_sl,
        "soft_keep": soft_keep,
        "halt_trigger": halt_trigger,
        "cooldown_sec": cool_ms / 1000.0,
        "use_risk": use_risk,
        "init_equity": round(init_eq, 2),
        "final_equity": round(final_eq, 2),
        "return_pct": round((final_eq - init_eq) / init_eq * 100, 3) if init_eq else 0.0,
        "trips": trips,
        "trip_count": len(trips),
        "trips_by_reason": by_reason,
        "open_positions": open_n,
        "events": len(events),
        "per_bot": per_bot,
        "equity_curve": eq_curve[-80:],
    }


def main() -> None:
    cfg = paper_config()
    soft_tp = _cfg_float(cfg, "portfolio_tp_pct", 0.0)
    hard_tp = _cfg_float(cfg, "portfolio_tp_hard_pct", 0.0)
    sl = _cfg_float(cfg, "portfolio_sl_pct", 0.0)
    bot_sl = _cfg_float(cfg, "daily_loss_pct", 0.20)
    wallets = [w for w in load_watchlist() if str(w.get("id") or "").startswith("bot_")]
    wallets.sort(key=lambda w: str(w.get("id") or ""))

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 7 * 86400 * 1000

    print(
        f"=== Desk 7d backtest "
        f"desk soft+{soft_tp*100:.1f}%/hard+{hard_tp*100:.0f}%/−{sl*100:.0f}% "
        f"bot−{bot_sl*100:.0f}% ===",
        flush=True,
    )
    av_map, events = _load_market(wallets, start_ms)
    with_risk = run(
        av_map=av_map,
        events=events,
        wallets=wallets,
        start_ms=start_ms,
        now_ms=now_ms,
        use_risk=True,
    )
    print(
        f"WITH risk: {with_risk['init_equity']} → {with_risk['final_equity']} "
        f"({with_risk['return_pct']}%) trips={with_risk['trip_count']} "
        f"{with_risk.get('trips_by_reason')}",
        flush=True,
    )
    no_risk = run(
        av_map=av_map,
        events=events,
        wallets=wallets,
        start_ms=start_ms,
        now_ms=now_ms,
        use_risk=False,
    )
    print(
        f"NO risk:   {no_risk['init_equity']} → {no_risk['final_equity']} "
        f"({no_risk['return_pct']}%)",
        flush=True,
    )
    out = {
        "generated_at": datetime.now(BJ).isoformat(),
        "note": (
            "Current defaults: desk TP/SL/multi-halt OFF; "
            f"per-bot −{bot_sl*100:.0f}% halt + cooldown only. "
            "NO risk = pure proportional follow."
        ),
        "with_risk": with_risk,
        "no_risk": no_risk,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

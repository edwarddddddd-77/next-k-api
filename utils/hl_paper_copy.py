"""Hyperliquid paper copy — immediate fill-delta market follow (no snapshot seed).

One bot per watchlist address (default 1000U; override via watchlist paper_balance):
  our_delta = fill.sz × (bot_equity / target_AV)
  trade/entry at fill.px (market), not target average entry
  hard cap: |notional| ≤ equity × leverage_cap

WS snapshots are ignored so deploy starts flat. Mark refresh updates uPnL and
runs the daily-loss circuit breaker (same gate as fill ingest).
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.hl_short_term import (
    is_hl_spot_coin,
    load_watchlist,
    resolve_spot_coin,
    snapshot_hyperevm_usdc as hl_snapshot_hyperevm_usdc,
    snapshot_positions as hl_snapshot_positions,
    snapshot_spot as hl_snapshot_spot,
    snapshot_spot_usdc as hl_snapshot_spot_usdc,
)
from utils.rate_limit import MinIntervalGuard

logger = logging.getLogger(__name__)

PAPER_NAME = "hl_paper_copy.json"
_lock = threading.Lock()
_mids_cache: dict[str, float] = {}
_mids_cache_at: float = 0.0
_mark_guard = MinIntervalGuard("HL_PAPER_MARK_COOLDOWN_SEC", 10.0)
_mids_ttl_sec = float(os.getenv("HL_MIDS_CACHE_SEC", "30") or 30)
_av_ttl_sec = float(os.getenv("HL_TARGET_AV_TTL_SEC", "30") or 30)
_health_guard = MinIntervalGuard("HL_TARGET_HEALTH_SEC", 300.0)


def _data_dir() -> Path:
    raw = (os.getenv("DATA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser()
    for candidate in (Path("/app/data"), Path("/data")):
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parents[1]


def _path() -> Path:
    return _data_dir() / PAPER_NAME


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)).strip() or default)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = (os.getenv(key) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def target_empty_av() -> float:
    """Target account value below this → treat as empty."""
    return max(0.0, _env_float("HL_TARGET_EMPTY_AV", 100.0))


def target_inactive_hours() -> float:
    """No target fill for this many hours → inactive."""
    return max(0.5, _env_float("HL_TARGET_INACTIVE_HOURS", 8.0))


def paper_enabled() -> bool:
    return _env_bool("HL_COPY_ENABLED", True)


def paper_config() -> dict[str, Any]:
    return {
        "enabled": paper_enabled(),
        "mode": "fill_delta_market",
        "bot_balance": _env_float("HL_PAPER_BALANCE", 1000.0),
        "copy_scale": 1.0,
        "min_notional": _env_float("HL_MIN_NOTIONAL", 10.0),
        "leverage_adjustment": _env_float("HL_LEVERAGE_ADJUSTMENT", 1.0),
        # Per-bot hard stop vs cycle anchor. Unlock via cooldown (desk rebase if enabled).
        "daily_loss_pct": _env_float("HL_DAILY_LOSS_PCT", 0.20),
        "bot_halt_cooldown_sec": _env_float("HL_BOT_HALT_COOLDOWN_SEC", 6 * 3600),
        # Desk-wide TP/SL off by default (7d backtest: clipped ~47%→~27% with little benefit).
        # Re-enable via env: HL_PORTFOLIO_TP_PCT / _HARD / _SL / _HALT_COUNT_TRIGGER.
        "portfolio_tp_pct": _env_float("HL_PORTFOLIO_TP_PCT", 0.0),
        "portfolio_tp_hard_pct": _env_float("HL_PORTFOLIO_TP_HARD_PCT", 0.0),
        "portfolio_sl_pct": _env_float("HL_PORTFOLIO_SL_PCT", 0.0),
        "portfolio_soft_reduce": _env_float("HL_PORTFOLIO_SOFT_REDUCE", 0.5),
        "portfolio_halt_count_trigger": int(
            _env_float("HL_PORTFOLIO_HALT_COUNT_TRIGGER", 0) or 0
        ),
        "target_empty_av": target_empty_av(),
        "target_inactive_hours": target_inactive_hours(),
        "note": (
            "Fill-delta market follow. Risk: per-bot −20% → flatten+halt; "
            "unlock after cooldown hours. "
            "Desk soft/hard TP/SL and multi-halt rebase are OFF by default "
            "(set HL_PORTFOLIO_* env to re-enable). "
            "Target empty/inactive via target_health."
        ),
    }


def _bot_initial_balance(wallet: dict[str, Any], cfg: dict[str, Any] | None = None) -> float:
    """Initial paper cash for one bot. Priority: env → watchlist paper_balance → default."""
    cfg = cfg or paper_config()
    default = float(cfg.get("bot_balance") or 1000.0)
    bid = str(wallet.get("id") or "").strip()
    if bid:
        env_key = f"HL_PAPER_BALANCE_{bid.upper()}"
        raw = (os.getenv(env_key) or "").strip()
        if raw:
            try:
                return max(0.0, float(raw))
            except (TypeError, ValueError):
                pass
    for key in ("paper_balance", "balance", "initial_balance"):
        if wallet.get(key) is not None and str(wallet.get(key)).strip() != "":
            try:
                return max(0.0, float(wallet.get(key)))
            except (TypeError, ValueError):
                pass
    return default


def _apply_initial_balance(bot: dict[str, Any], want: float, *, default: float) -> None:
    """Set/target paper_balance; top up cash if the configured initial changed."""
    want = max(0.0, float(want))
    prev = bot.get("paper_balance")
    try:
        old = float(prev) if prev is not None else float(default)
    except (TypeError, ValueError):
        old = float(default)
    if abs(want - old) > 1e-9:
        cur = float(bot.get("balance") or old)
        bot["balance"] = round(cur + (want - old), 4)
        if bot.get("day_start_equity") is not None:
            try:
                bot["day_start_equity"] = round(
                    float(bot["day_start_equity"]) + (want - old), 4
                )
            except (TypeError, ValueError):
                bot["day_start_equity"] = want
    bot["paper_balance"] = want
    _recompute_bot(bot)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _beijing_day() -> str:
    """Calendar day in Asia/Shanghai for daily risk reset (matches UI clocks)."""
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    except Exception:
        from datetime import timedelta

        return (datetime.now(timezone.utc) + timedelta(hours=8)).date().isoformat()


def _fill_dedupe_keys(fills: list) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for f in fills:
        if not isinstance(f, dict):
            continue
        tid = str(f.get("tid") or f.get("hash") or "").strip()
        if tid:
            keys.append(("tid", tid))
            continue
        fp = "|".join(
            [
                str(f.get("coin") or ""),
                str(f.get("time") or ""),
                str(f.get("px") or ""),
                str(f.get("sz") or ""),
                str(f.get("side") or ""),
            ]
        )
        if fp.replace("|", ""):
            keys.append(("fp", fp))
    return keys


def _seen_fill_key(existing: list, kind: str, value: str) -> bool:
    for x in existing:
        if not isinstance(x, dict):
            continue
        if kind == "tid":
            if str(x.get("target_tid") or "") == value:
                return True
            tids = x.get("target_tids")
            if isinstance(tids, list) and value in [str(t) for t in tids]:
                return True
        elif kind == "fp" and str(x.get("target_fp") or "") == value:
            return True
    return False


def _empty_bot(wallet: dict[str, Any], balance: float) -> dict[str, Any]:
    return {
        "id": wallet.get("id") or str(wallet.get("address") or "")[:10],
        "address": wallet.get("address"),
        "balance": balance,
        "equity": balance,
        "realized_pnl": 0.0,
        "positions": {},
        "fills": [],
        "copy_ratio": None,
        "target_av": None,
        "target_last_fill_at": None,
        "target_health": None,
        "risk_halted": False,
        "day_key": None,
        "day_start_equity": balance,
        "risk_anchor_equity": balance,
    }


def _ensure_bots(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy single-ledger → multi-bot, ensure every watchlist wallet has a bot."""
    cfg = paper_config()
    default_bal = float(cfg["bot_balance"])
    wallets = load_watchlist()
    bots = data.get("bots")
    if not isinstance(bots, dict):
        bots = {}
        # migrate old flat positions if present
        legacy_pos = data.get("positions") if isinstance(data.get("positions"), dict) else {}
        legacy_fills = data.get("fills") if isinstance(data.get("fills"), list) else []
        legacy_bal = float(data.get("balance") or default_bal)
        for i, w in enumerate(wallets):
            bid = str(w.get("id") or w.get("address") or "")[:32]
            init = _bot_initial_balance(w, cfg)
            bot = _empty_bot(w, init if i > 0 else legacy_bal)
            bot["paper_balance"] = init
            if i == 0 and legacy_pos:
                # keep only positions matching this source
                bot["positions"] = {
                    k: v
                    for k, v in legacy_pos.items()
                    if str(v.get("source") or "") in (bid, "", None)
                    or str(v.get("target_address") or "").lower()
                    == str(w.get("address") or "").lower()
                }
                bot["fills"] = legacy_fills[:200]
                bot["realized_pnl"] = float(data.get("realized_pnl") or 0)
                _apply_initial_balance(bot, init, default=legacy_bal)
            bots[bid] = bot

    want_ids: set[str] = set()
    for w in wallets:
        bid = str(w.get("id") or w.get("address") or "")[:32]
        want_ids.add(bid)
        init = _bot_initial_balance(w, cfg)
        new_addr = str(w.get("address") or "").strip().lower()
        if bid not in bots:
            bots[bid] = _empty_bot(w, init)
            bots[bid]["paper_balance"] = init
        else:
            old_addr = str(bots[bid].get("address") or "").strip().lower()
            # Same bot_* id rebound to a new leader → wipe stale positions/fills
            if old_addr and new_addr and old_addr != new_addr:
                logger.info(
                    "paper bot %s rebound %s → %s; resetting ledger",
                    bid,
                    old_addr[:14],
                    new_addr[:14],
                )
                bots[bid] = _empty_bot(w, init)
                bots[bid]["paper_balance"] = init
            else:
                bots[bid]["id"] = bid
                bots[bid]["address"] = w.get("address")
                bots[bid].setdefault("positions", {})
                bots[bid].setdefault("fills", [])
                bots[bid].setdefault("realized_pnl", 0.0)
                bots[bid].setdefault("risk_halted", False)
                if bots[bid].get("risk_anchor_equity") is None:
                    try:
                        bots[bid]["risk_anchor_equity"] = round(
                            float(bots[bid].get("equity") or bots[bid].get("balance") or 0),
                            4,
                        )
                    except (TypeError, ValueError):
                        bots[bid]["risk_anchor_equity"] = float(
                            bots[bid].get("paper_balance") or default_bal
                        )
                _apply_initial_balance(bots[bid], init, default=default_bal)
        # Keep paper allowlist in sync with watchlist coins (None = all)
        allow = _parse_allow_coins(w.get("coins"))
        if allow is None:
            bots[bid]["allow_coins"] = None
        else:
            bots[bid]["allow_coins"] = sorted(allow)
        # 日内 / 波段 label (watchlist tag or ht_style)
        tag = str(w.get("tag") or "").strip()
        ht = str(w.get("ht_style") or w.get("style") or "").strip().lower()
        if not tag:
            if ht in ("day_trader", "day", "日内") or ht.startswith("day"):
                tag = "日内"
            elif ht in ("swing_trader", "swing", "波段") or ht.startswith("swing"):
                tag = "波段"
        bots[bid]["tag"] = tag or None
        if tag == "日内" or ht in ("day_trader", "day") or ht.startswith("day"):
            bots[bid]["ht_style"] = "day_trader"
        elif tag == "波段" or ht in ("swing_trader", "swing") or ht.startswith("swing"):
            bots[bid]["ht_style"] = "swing_trader"
        else:
            bots[bid]["ht_style"] = ht or None
        # Extra labels e.g. concentrated → 单币集中
        raw_tags = w.get("style_tags")
        style_tags: list[str] = []
        if isinstance(raw_tags, list):
            for t in raw_tags:
                s = str(t or "").strip().lower()
                if s and s not in style_tags:
                    style_tags.append(s)
        bots[bid]["style_tags"] = style_tags or None

    # Drop bots removed from the watchlist (old dig ids clutter the desk)
    if want_ids:
        bots = {k: v for k, v in bots.items() if k in want_ids}

    data["bots"] = bots
    return data


def _recompute_bot(bot: dict[str, Any]) -> None:
    unreal = sum(float(p.get("u_pnl") or 0) for p in (bot.get("positions") or {}).values())
    bal = float(bot.get("balance") or 0)
    bot["equity"] = round(bal + unreal, 4)


def _aggregate(data: dict[str, Any]) -> dict[str, Any]:
    bots = data.get("bots") or {}
    positions: dict[str, Any] = {}
    fills: list[dict] = []
    balance = 0.0
    equity = 0.0
    realized = 0.0
    alerts: list[str] = []
    for bot in bots.values():
        _recompute_bot(bot)
        # Refresh quiet-hours label without hitting HL
        if bot.get("target_av") is not None or bot.get("target_last_fill_at") is not None:
            bot["target_health"] = _compute_target_health(bot)
            h = bot["target_health"]
            if not h.get("ok"):
                alerts.append(f"{bot.get('id')}:{h.get('status')}")
        balance += float(bot.get("balance") or 0)
        equity += float(bot.get("equity") or 0)
        realized += float(bot.get("realized_pnl") or 0)
        for k, p in (bot.get("positions") or {}).items():
            positions[k] = p
        fills.extend(bot.get("fills") or [])
    fills.sort(key=lambda x: str(x.get("ts") or ""), reverse=True)
    data["balance"] = round(balance, 4)
    data["equity"] = round(equity, 4)
    data["realized_pnl"] = round(realized, 4)
    data["positions"] = positions
    data["fills"] = fills[:500]
    data["bot_count"] = len(bots)
    data["target_alerts"] = alerts
    data["ok"] = True
    data["mode"] = "fill_delta_market"
    data["config"] = paper_config()
    # Portfolio risk snapshot — full desk (halted bots included)
    try:
        anchor = data.get("portfolio_anchor_equity")
        anchor_f = float(anchor) if anchor is not None else None
    except (TypeError, ValueError):
        anchor_f = None
    if anchor_f is None or anchor_f <= 0:
        anchor_f = equity if equity > 0 else None
        if anchor_f is not None:
            data["portfolio_anchor_equity"] = round(anchor_f, 4)
    data["portfolio_equity"] = round(equity, 4)
    if anchor_f and anchor_f > 0:
        data["portfolio_return_pct"] = round((equity - anchor_f) / anchor_f, 6)
    else:
        data["portfolio_return_pct"] = None
    data["portfolio_copy_scale"] = float(data.get("portfolio_copy_scale") or 1.0)
    data["portfolio_halted_count"] = sum(
        1 for b in bots.values() if isinstance(b, dict) and b.get("risk_halted")
    )
    pr = data.get("portfolio_risk")
    data["portfolio_risk"] = pr if isinstance(pr, dict) else None
    return data


def load_paper() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        data = {"bots": {}, "updated_at": _now()}
        data = _ensure_bots(data)
        return _aggregate(data)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except Exception as exc:
        logger.warning("hl paper load failed: %s", exc)
        data = {"error": str(exc)}
    data = _ensure_bots(data)
    return _aggregate(data)


def save_paper(data: dict[str, Any]) -> None:
    data = _ensure_bots(data)
    data = _aggregate(data)
    data["updated_at"] = _now()
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_paper() -> dict[str, Any]:
    with _lock:
        data = _ensure_bots({"bots": {}})
        cfg = paper_config()
        wallets = {str(w.get("id") or "")[:32]: w for w in load_watchlist()}
        for bot in data["bots"].values():
            bid = str(bot.get("id") or "")
            w = wallets.get(bid) or {"id": bid}
            bal = _bot_initial_balance(w, cfg)
            bot.update(_empty_bot(bot, bal))
            bot["paper_balance"] = bal
        save_paper(data)
        return load_paper()


def fetch_all_mids(*, force: bool = False) -> dict[str, float]:
    """Cached allMids to avoid HL 429 under UI polling.

    Merges main DEX mids with HIP-3 ``xyz`` stock/commodity mids
    (``allMids`` alone has no ``xyz:SKHX`` etc., so marks would stick at entry).
    """
    global _mids_cache, _mids_cache_at
    now = time.monotonic()
    if (
        not force
        and _mids_cache
        and _mids_ttl_sec > 0
        and (now - _mids_cache_at) < _mids_ttl_sec
    ):
        return dict(_mids_cache)

    from utils.hl_short_term import http_json

    def _ingest(raw: Any, dest: dict[str, float]) -> None:
        if not isinstance(raw, dict):
            return
        payload = raw.get("mids") if "mids" in raw else raw
        if not isinstance(payload, dict):
            return
        for k, v in payload.items():
            try:
                dest[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

    out: dict[str, float] = {}
    try:
        _ingest(http_json({"type": "allMids"}), out)
    except Exception as exc:
        logger.warning("paper allMids failed: %s", exc)
    try:
        # HIP-3 equity/commodity perps (xyz:TSLA, xyz:SKHX, …)
        _ingest(http_json({"type": "allMids", "dex": "xyz"}), out)
    except Exception as exc:
        logger.warning("paper xyz allMids failed: %s", exc)

    if out:
        _mids_cache = out
        _mids_cache_at = now
    return dict(_mids_cache if not out else out)


def _mid_for_coin(mids: dict[str, float], coin: str) -> float:
    """Resolve mid when HL keys differ in case (``xyz:TSLA`` vs ``XYZ:TSLA``)."""
    raw = str(coin or "").strip()
    if not raw or not mids:
        return 0.0
    candidates = [raw, raw.upper(), raw.lower()]
    if ":" in raw:
        pref, rest = raw.split(":", 1)
        candidates.extend(
            [
                f"{pref.lower()}:{rest}",
                f"{pref.lower()}:{rest.upper()}",
                f"{pref.upper()}:{rest.upper()}",
            ]
        )
    base = _coin_base(raw)
    if base:
        candidates.append(base)
    seen: set[str] = set()
    for key in candidates:
        if not key or key in seen:
            continue
        seen.add(key)
        if key not in mids:
            continue
        try:
            val = float(mids[key])
        except (TypeError, ValueError):
            continue
        if val > 0:
            return val
    return 0.0


def refresh_marks(*, force: bool = False) -> dict[str, Any]:
    """Mark-to-market + portfolio / daily risk. Throttled; ratio still updates on fills."""
    if not paper_enabled():
        return load_paper()

    if not force:
        allowed, _wait = _mark_guard.check_allow()
        if not allowed:
            return load_paper()

    try:
        mids = fetch_all_mids(force=force)
    except Exception as exc:
        logger.warning("paper mark mids failed: %s", exc)
        mids = dict(_mids_cache)

    halt_logged: list[dict[str, Any]] = []
    with _lock:
        data = load_paper()
        cfg = paper_config()
        for bot in (data.get("bots") or {}).values():
            _roll_day(bot, cfg)
            for pos in (bot.get("positions") or {}).values():
                coin = str(pos.get("coin") or "")
                mid = _mid_for_coin(mids, coin) or float(pos.get("mark_px") or 0)
                if mid > 0:
                    _mark_one(pos, mid)
            _recompute_bot(bot)
            # Optional per-bot daily loss (off by default).
            if float(cfg.get("daily_loss_pct") or 0) > 0:
                halt_rows = _maybe_risk_halt(bot, mids, cfg)
                if halt_rows:
                    day_start = float(bot.get("day_start_equity") or 0)
                    eq = float(bot.get("equity") or 0)
                    loss_pct = (
                        0.0 if day_start <= 0 else (day_start - eq) / day_start
                    )
                    logger.warning(
                        "HL risk halt on mark bot=%s loss_pct=%.1f%% equity=%.2f day_start=%.2f",
                        bot.get("id"),
                        loss_pct * 100.0,
                        eq,
                        day_start,
                    )
                    halt_logged.extend(halt_rows)
        # Desk-wide compound TP/SL (sum of all bots).
        port_rows = _maybe_portfolio_risk(data, mids, cfg)
        if port_rows:
            halt_logged.extend(port_rows)
        save_paper(data)
        _mark_guard.mark_used()
        out = load_paper()

    if halt_logged:
        try:
            from utils.hl_bitget_executor import maybe_execute_rows_async

            maybe_execute_rows_async(halt_logged)
        except Exception:
            logger.exception("HL Bitget live hook failed (mark halt)")
    return out


def _parse_live_fill(fill: dict) -> dict[str, Any] | None:
    """Extract coin, signed target delta, px, ids from one HL fill."""
    if not isinstance(fill, dict):
        return None
    coin = str(fill.get("coin") or "").strip()
    if not coin:
        return None
    # Spot is monitor-only — never size into paper / Bitget follow.
    if is_hl_spot_coin(coin):
        return None
    try:
        px = float(fill.get("px") or 0)
        sz = abs(float(fill.get("sz") or 0))
    except (TypeError, ValueError):
        return None
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
    tid = str(fill.get("tid") or fill.get("hash") or "").strip()
    fill_time = fill.get("time")
    return {
        "coin": coin.upper(),
        "target_delta": signed,
        "px": px,
        "tid": tid or None,
        "fill_time": fill_time,
        "side": "buy" if signed > 0 else "sell",
        "raw": fill,
    }


def _spot_fill_row(fill: dict) -> dict[str, Any] | None:
    """Normalize one HL spot fill for desk monitoring (not paper copy)."""
    if not isinstance(fill, dict) or not is_hl_spot_coin(fill.get("coin")):
        return None
    raw_coin = str(fill.get("coin") or "")
    side = str(fill.get("side") or "").strip().upper()
    if side in ("B", "BUY"):
        side_l = "buy"
    elif side in ("A", "SELL"):
        side_l = "sell"
    else:
        side_l = side.lower() or None
    try:
        px = float(fill.get("px") or 0)
        sz = float(fill.get("sz") or 0)
    except (TypeError, ValueError):
        px, sz = 0.0, 0.0
    if px <= 0 or abs(sz) <= 0:
        return None
    ft = _fill_time_epoch(fill.get("time"))
    return {
        "coin": resolve_spot_coin(raw_coin),
        "coin_raw": raw_coin,
        "side": side_l,
        "dir": fill.get("dir"),
        "sz": sz,
        "px": px,
        "notional": round(abs(px * sz), 4),
        "time": fill.get("time"),
        "tid": fill.get("tid") or fill.get("hash"),
        "ts": (
            datetime.fromtimestamp(ft, timezone.utc).isoformat()
            if ft is not None
            else datetime.now(timezone.utc).isoformat()
        ),
    }


def _merge_target_spot_fills(bot: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    prev = list(bot.get("target_spot_fills") or [])

    def _fp(x: dict[str, Any]) -> str:
        tid = str(x.get("tid") or "").strip()
        if tid:
            return f"tid:{tid}"
        return "|".join(
            [
                "fp",
                str(x.get("coin_raw") or x.get("coin") or ""),
                str(x.get("time") or ""),
                str(x.get("px") or ""),
                str(x.get("sz") or ""),
                str(x.get("side") or ""),
            ]
        )

    seen = {_fp(x) for x in prev if isinstance(x, dict)}
    merged = list(prev)
    for r in rows:
        if not isinstance(r, dict):
            continue
        key = _fp(r)
        if key in seen:
            continue
        seen.add(key)
        merged.insert(0, r)
    merged.sort(key=lambda x: float((x or {}).get("time") or 0), reverse=True)
    bot["target_spot_fills"] = merged[:40]


def _apply_spot_snapshot(bot: dict[str, Any], spot: dict[str, Any] | None) -> None:
    if not spot:
        return
    try:
        bot["target_spot_usdc"] = round(float(spot.get("usdc") or 0), 4)
    except (TypeError, ValueError):
        bot["target_spot_usdc"] = 0.0
    bot["target_spot_at"] = time.time()
    bot["target_spot_balances"] = list(spot.get("balances") or [])[:40]
    # Merge poll fills with any newer WS-only rows (do not wipe).
    _merge_target_spot_fills(bot, list(spot.get("recent_fills") or []))


def _empty_book_snap(bot: dict[str, Any], snap: dict[str, Any] | None) -> dict[str, Any]:
    """Snap suitable for flattening paper to match a flat/empty target."""
    if snap is not None and _target_snap_flat(snap):
        return {
            "account_value": float(snap.get("account_value") or bot.get("target_av") or 0),
            "positions": [],
        }
    return {
        "account_value": float(bot.get("target_av") or 0),
        "positions": [],
    }


def _should_flatten_paper(
    bot: dict[str, Any],
    snap: dict[str, Any] | None,
    *,
    ratio: float,
) -> bool:
    """Flatten paper leftovers when target book is flat or AV is gone."""
    if not (bot.get("positions") or {}):
        return False
    if _target_snap_flat(snap):
        return True
    # Snap failed / stale but AV already ~0 — still clear zombies.
    if snap is None and ratio <= 0:
        return True
    try:
        av = float(bot.get("target_av") or 0)
    except (TypeError, ValueError):
        av = 0.0
    if ratio <= 0 and av < target_empty_av():
        return True
    return False


def _cache_target_meta(bot: dict[str, Any], snap: dict[str, Any] | None) -> None:
    if not snap:
        return
    try:
        av = float(snap.get("account_value") or 0)
    except (TypeError, ValueError):
        av = 0.0
    # Always record AV (incl. 0) so empty wallets are detectable.
    bot["target_av"] = av
    bot["target_av_at"] = time.time()
    lev_map: dict[str, float] = {}
    for p in snap.get("positions") or []:
        if not isinstance(p, dict):
            continue
        c = str(p.get("coin") or "").strip().upper()
        if not c:
            continue
        try:
            if p.get("lev") is not None:
                lev_map[c] = float(p["lev"])
        except (TypeError, ValueError):
            continue
    if lev_map:
        prev = bot.get("target_lev_by_coin")
        if isinstance(prev, dict):
            prev.update(lev_map)
            bot["target_lev_by_coin"] = prev
        else:
            bot["target_lev_by_coin"] = lev_map


def _fill_time_epoch(fill_time: Any) -> float | None:
    try:
        if fill_time is None:
            return None
        ft = float(fill_time)
        if ft > 1e12:
            ft /= 1000.0
        if ft > 1e9:
            return ft
    except (TypeError, ValueError):
        pass
    return None


def note_target_fill(bot: dict[str, Any], fill_time: Any = None) -> None:
    """Record that the watched wallet just traded (WS live fill)."""
    ts = _fill_time_epoch(fill_time) or time.time()
    prev = bot.get("target_last_fill_at")
    try:
        prev_f = float(prev) if prev is not None else 0.0
    except (TypeError, ValueError):
        prev_f = 0.0
    if ts >= prev_f:
        bot["target_last_fill_at"] = ts


def _compute_target_health(bot: dict[str, Any]) -> dict[str, Any]:
    av_raw = bot.get("target_av")
    try:
        av = float(av_raw) if av_raw is not None else None
    except (TypeError, ValueError):
        av = None
    empty_thr = target_empty_av()
    inactive_h = target_inactive_hours()
    now = time.time()

    last = bot.get("target_last_fill_at")
    try:
        last_f = float(last) if last is not None else None
    except (TypeError, ValueError):
        last_f = None

    quiet_h = None if last_f is None else max(0.0, (now - last_f) / 3600.0)
    empty = av is not None and av < empty_thr
    inactive = quiet_h is not None and quiet_h >= inactive_h
    # No fill ever seen since we started watching, but AV looks funded → still flag after threshold
    # from first health probe timestamp.
    if last_f is None and not empty:
        watched = bot.get("target_watched_at")
        try:
            w = float(watched) if watched is not None else None
        except (TypeError, ValueError):
            w = None
        if w is not None and (now - w) / 3600.0 >= inactive_h:
            inactive = True
            quiet_h = (now - w) / 3600.0

    spot_raw = bot.get("target_spot_usdc")
    try:
        spot = float(spot_raw) if spot_raw is not None else None
    except (TypeError, ValueError):
        spot = None
    evm_raw = bot.get("target_evm_usdc")
    try:
        evm = float(evm_raw) if evm_raw is not None else None
    except (TypeError, ValueError):
        evm = None

    def _empty_label(base: str) -> str:
        # Same address; spot/EVM USDC does not restore copy sizing (needs perp AV).
        bits: list[str] = []
        if spot is not None and spot >= 1.0:
            bits.append(f"Core现货 ${spot:,.0f}")
        elif spot is not None and spot < 1.0:
            bits.append("Core现货空")
        if evm is not None and evm >= 1.0:
            bits.append(f"EVM ${evm:,.0f}")
        elif evm is not None and evm < 1.0:
            bits.append("EVM空")
        if bits:
            return f"{base}（{' · '.join(bits)}）"
        return base

    if empty and inactive:
        status = "empty_inactive"
        # Perp AV of the *watched* wallet — not paper equity.
        label = _empty_label("对方永续已空且不活跃")
    elif empty:
        status = "empty"
        label = _empty_label("对方永续已空")
    elif inactive:
        status = "inactive"
        label = "对方不活跃"
    else:
        status = "ok"
        label = "正常"

    return {
        "status": status,
        "label": label,
        "ok": status == "ok",
        "empty": empty,
        "inactive": inactive,
        "target_av": None if av is None else round(av, 2),
        "target_spot_usdc": None if spot is None else round(spot, 2),
        "target_evm_usdc": None if evm is None else round(evm, 2),
        "target_spot_balances": list(bot.get("target_spot_balances") or [])[:40],
        "target_spot_fills": list(bot.get("target_spot_fills") or [])[:20],
        "empty_below": empty_thr,
        "quiet_hours": None if quiet_h is None else round(quiet_h, 2),
        "inactive_after_hours": inactive_h,
        "last_fill_at": (
            datetime.fromtimestamp(last_f, timezone.utc).isoformat() if last_f else None
        ),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def refresh_target_health(*, force: bool = False) -> dict[str, Any]:
    """Poll target clearinghouse + recent fill age; tag empty/inactive bots."""
    if not paper_enabled():
        return load_paper()
    if not force:
        allowed, _wait = _health_guard.check_allow()
        if not allowed:
            return load_paper()

    from utils.hl_short_term import http_json

    with _lock:
        book = load_paper()
        now = time.time()
        alerts: list[str] = []
        try:
            mids = fetch_all_mids()
        except Exception:
            mids = dict(_mids_cache)
        cfg = paper_config()
        for bot in (book.get("bots") or {}).values():
            addr = str(bot.get("address") or "").strip()
            if not addr:
                continue
            if bot.get("target_watched_at") is None:
                bot["target_watched_at"] = now

            snap = None
            # Refresh perp AV + same-wallet spot USDC
            try:
                snap = hl_snapshot_positions(addr)
                _cache_target_meta(bot, snap)
            except Exception as exc:
                logger.warning("target health AV %s: %s", bot.get("id"), exc)
            try:
                spot = hl_snapshot_spot(addr, fill_limit=20)
                _apply_spot_snapshot(bot, spot)
            except Exception as exc:
                logger.warning("target health spot %s: %s", bot.get("id"), exc)
                try:
                    bot["target_spot_usdc"] = round(hl_snapshot_spot_usdc(addr), 4)
                    bot["target_spot_at"] = now
                except Exception as exc2:
                    logger.warning("target health spot usdc %s: %s", bot.get("id"), exc2)

            # HyperEVM USDC (same address) — monitor only
            try:
                evm = hl_snapshot_hyperevm_usdc(addr)
                if evm is None:
                    bot["target_evm_usdc"] = None
                else:
                    bot["target_evm_usdc"] = round(float(evm), 4)
                bot["target_evm_at"] = now
            except Exception as exc:
                logger.warning("target health evm %s: %s", bot.get("id"), exc)
                bot["target_evm_usdc"] = None

            # Seed last-fill from recent HL *perp* fills if we have never seen one on WS
            if bot.get("target_last_fill_at") is None:
                try:
                    fills = http_json({"type": "userFills", "user": addr})
                    if isinstance(fills, list) and fills:
                        latest = None
                        for f in fills[:80]:
                            if not isinstance(f, dict):
                                continue
                            if is_hl_spot_coin(f.get("coin")):
                                continue
                            ft = _fill_time_epoch(f.get("time"))
                            if ft is not None and (latest is None or ft > latest):
                                latest = ft
                        if latest is not None:
                            bot["target_last_fill_at"] = latest
                except Exception as exc:
                    logger.debug("target health fills %s: %s", bot.get("id"), exc)

            prev = (bot.get("target_health") or {}).get("status") if isinstance(bot.get("target_health"), dict) else None
            health = _compute_target_health(bot)
            bot["target_health"] = health
            # Target flat on perps → close zombie paper (even if AV still funded)
            if _should_flatten_paper(
                bot,
                snap,
                ratio=_copy_ratio(bot, cfg) if (bot.get("positions") or {}) else 1.0,
            ):
                closed = _mirror_target_book(
                    bot, _empty_book_snap(bot, snap), mids, cfg
                )
                if closed:
                    logger.warning(
                        "HL target flat flatten %s: closed %s paper rows",
                        bot.get("id"),
                        len(closed),
                    )
            if not health.get("ok"):
                alerts.append(f"{bot.get('id')}:{health.get('status')}")
                if prev != health.get("status"):
                    logger.warning(
                        "HL target health %s → %s av=%s quiet_h=%s",
                        bot.get("id"),
                        health.get("status"),
                        health.get("target_av"),
                        health.get("quiet_hours"),
                    )
            elif prev and prev != "ok":
                logger.info("HL target health %s recovered → ok", bot.get("id"))

        book["target_alerts"] = alerts
        save_paper(book)
        _health_guard.mark_used()
        return load_paper()


def _need_target_av_refresh(bot: dict[str, Any]) -> bool:
    av = float(bot.get("target_av") or 0)
    if av <= 1e-9:
        return True
    if _av_ttl_sec <= 0:
        return False
    at = float(bot.get("target_av_at") or 0)
    return (time.time() - at) >= _av_ttl_sec


def _target_snap_flat(snap: dict[str, Any] | None) -> bool:
    """True when target clearinghouse has no open positions (AV may still be >0)."""
    if not snap:
        return False
    for p in snap.get("positions") or []:
        if not isinstance(p, dict):
            continue
        try:
            if abs(float(p.get("szi") or 0)) > 1e-16:
                return False
        except (TypeError, ValueError):
            continue
    return True


def _stamp_skipped_fills(
    bot: dict[str, Any],
    fresh: list[dict[str, Any]],
    *,
    reason: str,
    note_activity: bool = False,
) -> None:
    """Record fill tids so zero-ratio / empty-target skips do not reprocess forever."""
    fills = list(bot.get("fills") or [])
    for item in fresh:
        keys = _fill_dedupe_keys([item.get("raw")] if isinstance(item.get("raw"), dict) else [])
        tid = item.get("tid")
        if tid:
            keys = list(keys) + [("tid", str(tid))]
        for kind, value in keys:
            if not value or _seen_fill_key(fills, kind, value):
                continue
            mark: dict[str, Any] = {
                "id": str(uuid.uuid4())[:8],
                "action": "signal",
                "skipped": True,
                "skip_reason": reason,
                "source": bot.get("id"),
                "coin": item.get("coin"),
                "ts": _now(),
            }
            if kind == "tid":
                mark["target_tid"] = value
                mark["target_tids"] = [value]
            else:
                mark["target_fp"] = value
            fills.insert(0, mark)
        if note_activity:
            note_target_fill(bot, item.get("fill_time"))
    bot["fills"] = fills[:300]


def _copy_ratio(
    bot: dict[str, Any],
    cfg: dict[str, Any],
    *,
    size_mult: float = 1.0,
) -> float:
    """equity / target_AV — sizing basis; optional soft-TP size_mult (e.g. 0.5)."""
    _recompute_bot(bot)
    eq = float(bot.get("equity") or bot.get("balance") or cfg["bot_balance"])
    av = float(bot.get("target_av") or 0)
    if av <= 1e-9 or eq <= 0:
        bot["copy_ratio"] = 0.0
        return 0.0
    try:
        mult = float(size_mult)
    except (TypeError, ValueError):
        mult = 1.0
    if mult <= 0:
        bot["copy_ratio"] = 0.0
        return 0.0
    ratio = (eq / av) * mult
    bot["copy_ratio"] = round(ratio, 10)
    bot["copy_size_mult"] = round(mult, 6)
    return ratio


def _book_copy_scale(book: dict[str, Any], cfg: dict[str, Any] | None = None) -> float:
    """After soft TP, follow fills at keep_frac until hard portfolio rebase."""
    cfg = cfg or paper_config()
    if book.get("portfolio_soft_tp_taken"):
        try:
            scale = float(book.get("portfolio_copy_scale"))
        except (TypeError, ValueError):
            scale = float(cfg.get("portfolio_soft_reduce") or 0.5)
        if scale <= 0:
            scale = float(cfg.get("portfolio_soft_reduce") or 0.5)
        return max(0.0, min(1.0, scale))
    try:
        raw = book.get("portfolio_copy_scale")
        if raw is not None:
            return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        pass
    return 1.0


def _lev_for_coin(bot: dict[str, Any], coin: str, cfg: dict[str, Any]) -> int:
    lev_map = bot.get("target_lev_by_coin") if isinstance(bot.get("target_lev_by_coin"), dict) else {}
    raw = None
    for key in _scope_keys_for_coin(coin):
        if key in lev_map:
            raw = lev_map[key]
            break
    pos = (bot.get("positions") or {}).get(f"{bot.get('id')}:{coin}")
    if raw is None and isinstance(pos, dict) and pos.get("leverage") is not None:
        raw = pos.get("leverage")
    if raw is None:
        raw = 10.0
    return _adjusted_leverage(raw, cfg.get("leverage_adjustment", 1.0), coin)


def _max_notional(bot: dict[str, Any], lev: int, cfg: dict[str, Any]) -> float:
    _recompute_bot(bot)
    eq = float(bot.get("equity") or bot.get("balance") or cfg["bot_balance"])
    return max(0.0, eq * float(max(1, lev)))


def _clip_sz_to_notional(sz: float, px: float, max_notional: float) -> float:
    if px <= 0 or max_notional <= 0:
        return 0.0
    cap = max_notional / px
    if abs(sz) <= cap + 1e-12:
        return sz
    return math.copysign(cap, sz)


def _apply_market_fill(
    bot: dict[str, Any],
    *,
    coin: str,
    target_delta: float,
    px: float,
    cfg: dict[str, Any],
    mids: dict[str, float],
    ratio: float,
    lev: int,
    trigger_tid: str | None = None,
    fill_time: Any = None,
) -> list[dict[str, Any]]:
    """Apply one proportional fill at market px; enforce equity×lev notional cap."""
    allow = _bot_allow_coins(bot)
    if not _coin_allowed(coin, allow):
        return []

    our_delta = float(target_delta) * float(ratio)
    if abs(our_delta) < 1e-16:
        return []

    key = f"{bot.get('id')}:{coin}"
    positions = bot.setdefault("positions", {})
    old = positions.get(key)
    old_sz = float(old.get("sz") or 0) if isinstance(old, dict) else 0.0
    max_n = _max_notional(bot, lev, cfg)
    raw_new = old_sz + our_delta

    # Increasing exposure (incl. open / add / flip-to-new-side) must respect margin
    increasing = abs(raw_new) > abs(old_sz) + 1e-12 or (
        abs(old_sz) < 1e-16 and abs(raw_new) > 1e-16
    )
    if old_sz * raw_new < 0:
        # Flip: close old fully, open opposite clipped to cap
        new_sz = _clip_sz_to_notional(raw_new, px, max_n)
    elif increasing:
        new_sz = _clip_sz_to_notional(raw_new, px, max_n)
        if abs(new_sz - old_sz) < 1e-12:
            # Already at cap — cannot add
            row = {
                "id": str(uuid.uuid4())[:8],
                "action": "signal",
                "skipped": True,
                "reason": "margin_cap",
                "source": bot.get("id"),
                "coin": coin,
                "px": px,
                "our_sz": 0,
                "target_delta": target_delta,
                "copy_ratio": round(ratio, 10),
                "leverage": lev,
                "max_notional": round(max_n, 4),
                "target_tid": trigger_tid,
                "fill_time": fill_time,
                "ts": _now(),
            }
            fills = list(bot.get("fills") or [])
            fills.insert(0, row)
            bot["fills"] = fills[:300]
            return []
    else:
        new_sz = raw_new

    # Dust open
    if abs(old_sz) < 1e-16 and abs(new_sz) * px < cfg["min_notional"]:
        return []

    rows: list[dict[str, Any]] = []
    fills = list(bot.get("fills") or [])
    mark = _mid_for_coin(mids, coin) or px

    def _row(
        action: str,
        qty: float,
        trade_px: float,
        realized: float | None,
        side: str,
        *,
        pos_side: str,
    ) -> dict:
        out: dict[str, Any] = {
            "id": str(uuid.uuid4())[:8],
            "action": action,
            "source": bot.get("id"),
            "coin": coin,
            "px": trade_px,
            "our_sz": qty,
            "notional": qty * trade_px,
            "leverage": lev,
            "copy_ratio": round(ratio, 10),
            "target_delta": target_delta,
            "target_tid": trigger_tid,
            "target_tids": [trigger_tid] if trigger_tid else [],
            "target_address": bot.get("address"),
            "fill_time": fill_time,
            "side": side,
            "pos": pos_side,  # long | short — UI 开多/平空
            "ts": _now(),
            "max_notional": round(max_n, 4),
        }
        if realized is not None:
            out["realized_pnl"] = realized
        return out

    # Flatten then reopen on flip
    if old and abs(old_sz) > 1e-16 and old_sz * new_sz < 0:
        pnl = _realize(bot, old, px, abs(old_sz))
        close_side = "sell" if old_sz > 0 else "buy"
        close_pos = "long" if old_sz > 0 else "short"
        close_row = _row("close", abs(old_sz), px, pnl, close_side, pos_side=close_pos)
        rows.append(close_row)
        fills.insert(0, close_row)
        old = None
        old_sz = 0.0

    if abs(new_sz) < 1e-16:
        if old and abs(old_sz) > 1e-16:
            pnl = _realize(bot, old, px, abs(old_sz))
            close_side = "sell" if old_sz > 0 else "buy"
            close_pos = "long" if old_sz > 0 else "short"
            close_row = _row(
                "close", abs(old_sz), px, pnl, close_side, pos_side=close_pos
            )
            rows.append(close_row)
            fills.insert(0, close_row)
            positions.pop(key, None)
        bot["fills"] = fills[:300]
        _recompute_bot(bot)
        return rows

    applied_delta = new_sz - old_sz
    side = "buy" if applied_delta > 0 else "sell"
    pos_side = "long" if new_sz > 0 else "short"

    if not old or abs(old_sz) < 1e-16:
        pos = {
            "key": key,
            "source": bot.get("id"),
            "coin": coin,
            "sz": new_sz,
            "entry_px": px,
            "copy_ratio": round(ratio, 10),
            "leverage": lev,
            "target_address": bot.get("address"),
            "target_av": bot.get("target_av"),
            "opened_at": _now(),
            "u_pnl": 0.0,
            "mark_px": mark,
        }
        _mark_one(pos, mark)
        positions[key] = pos
        open_row = _row("open", abs(new_sz), px, None, side, pos_side=pos_side)
        rows.append(open_row)
        fills.insert(0, open_row)
    elif abs(new_sz) + 1e-12 < abs(old_sz):
        closed = abs(old_sz) - abs(new_sz)
        pnl = _realize(bot, old, px, closed)
        # reduce keeps the side of the remaining (same as old) book
        red_pos = "long" if old_sz > 0 else "short"
        red_row = _row("reduce", closed, px, pnl, side, pos_side=red_pos)
        rows.append(red_row)
        fills.insert(0, red_row)
        old["sz"] = new_sz
        old["leverage"] = lev
        old["copy_ratio"] = round(ratio, 10)
        old["target_av"] = bot.get("target_av")
        _mark_one(old, mark)
        positions[key] = old
    else:
        add = abs(new_sz) - abs(old_sz)
        if add > 1e-16:
            old_entry = float(old.get("entry_px") or px)
            old_abs = abs(old_sz)
            old["entry_px"] = (old_entry * old_abs + px * add) / (old_abs + add)
            inc_row = _row("increase", add, px, None, side, pos_side=pos_side)
            rows.append(inc_row)
            fills.insert(0, inc_row)
        old["sz"] = new_sz
        old["leverage"] = lev
        old["copy_ratio"] = round(ratio, 10)
        old["target_av"] = bot.get("target_av")
        _mark_one(old, mark)
        positions[key] = old

    bot["fills"] = fills[:300]
    _recompute_bot(bot)
    return rows


def ingest_user_event(address: str, data: dict) -> list[dict]:
    """On live target fill(s): immediately market-follow fill deltas (no settle sleep)."""
    if data.get("isSnapshot"):
        return []
    fills = data.get("fills")
    if not isinstance(fills, list) or not fills:
        return []
    if not paper_enabled():
        return []

    spot_rows: list[dict[str, Any]] = []
    parsed: list[dict[str, Any]] = []
    for f in fills:
        spot_row = _spot_fill_row(f) if isinstance(f, dict) else None
        if spot_row:
            spot_rows.append(spot_row)
            continue
        item = _parse_live_fill(f)
        if item:
            parsed.append(item)

    addr = address.lower()
    recv_at = time.time()

    # Spot monitor only — record then continue to perp follow if any.
    if spot_rows and not parsed:
        with _lock:
            book = load_paper()
            for bot in (book.get("bots") or {}).values():
                if str(bot.get("address") or "").lower() != addr:
                    continue
                _merge_target_spot_fills(bot, spot_rows)
                # Do not touch target_last_fill_at — spot must not reset perp inactive.
            save_paper(book)
        return []

    if not parsed:
        return []

    cfg = paper_config()
    logged: list[dict] = []
    with _lock:
        book = load_paper()
        # Re-fetch AV/book under lock so ratio=0 flatten sees post-close state.
        snap: dict[str, Any] | None = None
        try:
            snap = hl_snapshot_positions(address)
        except Exception as exc:
            logger.warning("target AV refresh failed %s: %s", address[:10], exc)
        try:
            mids = fetch_all_mids()
        except Exception:
            mids = dict(_mids_cache)

        for bot in (book.get("bots") or {}).values():
            if str(bot.get("address") or "").lower() != addr:
                continue

            if spot_rows:
                _merge_target_spot_fills(bot, spot_rows)

            existing = list(bot.get("fills") or [])
            fresh: list[dict[str, Any]] = []
            for item in parsed:
                keys = _fill_dedupe_keys([item["raw"]])
                if keys and all(_seen_fill_key(existing, k, v) for k, v in keys):
                    continue
                # also skip if tid already recorded from twin channel
                tid = item.get("tid")
                if tid and _seen_fill_key(existing, "tid", tid):
                    continue
                fresh.append(item)
            if not fresh:
                continue

            if snap is not None:
                _cache_target_meta(bot, snap)
            elif _need_target_av_refresh(bot):
                logger.warning(
                    "HL follow skip %s: no target_av (cannot size)", bot.get("id")
                )
                _stamp_skipped_fills(bot, fresh, reason="no_snap", note_activity=False)
                continue

            halt_rows = _maybe_risk_halt(bot, mids, cfg)
            if halt_rows is not None:
                logged.extend(halt_rows)
                continue

            size_mult = _book_copy_scale(book, cfg)
            ratio = _copy_ratio(bot, cfg, size_mult=size_mult)
            if ratio <= 0:
                # Cannot size opens. Flatten paper when target is flat / AV empty.
                if _should_flatten_paper(bot, snap, ratio=ratio):
                    rows = _mirror_target_book(
                        bot, _empty_book_snap(bot, snap), mids, cfg
                    )
                    logged.extend(rows)
                    logger.warning(
                        "HL follow flatten %s: target flat/AV=0 closed %s paper pos",
                        bot.get("id"),
                        len(rows),
                    )
                    _stamp_skipped_fills(
                        bot, fresh, reason="flattened_empty_target", note_activity=True
                    )
                else:
                    logger.warning(
                        "HL follow skip %s: ratio=0 equity=%s av=%s snap_flat=%s",
                        bot.get("id"),
                        bot.get("equity"),
                        bot.get("target_av"),
                        _target_snap_flat(snap),
                    )
                    _stamp_skipped_fills(
                        bot, fresh, reason="ratio_zero", note_activity=False
                    )
                continue

            for item in fresh:
                coin = item["coin"]
                lev = _lev_for_coin(bot, coin, cfg)
                fill_ts = item.get("fill_time")
                lag_ms = None
                try:
                    if fill_ts is not None:
                        # HL time is usually ms epoch
                        ft = float(fill_ts)
                        if ft > 1e12:
                            ft /= 1000.0
                        lag_ms = int(max(0.0, (recv_at - ft) * 1000))
                except (TypeError, ValueError):
                    lag_ms = None
                logger.info(
                    "HL market-follow bot=%s coin=%s tdelta=%s px=%s ratio=%.6g lev=%s "
                    "av=%s equity=%s fill_time=%s lag_ms=%s",
                    bot.get("id"),
                    coin,
                    item["target_delta"],
                    item["px"],
                    ratio,
                    lev,
                    bot.get("target_av"),
                    bot.get("equity"),
                    fill_ts,
                    lag_ms,
                )
                rows = _apply_market_fill(
                    bot,
                    coin=coin,
                    target_delta=float(item["target_delta"]),
                    px=float(item["px"]),
                    cfg=cfg,
                    mids=mids,
                    ratio=ratio,
                    lev=lev,
                    trigger_tid=item.get("tid"),
                    fill_time=fill_ts,
                )
                note_target_fill(bot, fill_ts)
                logged.extend(rows)
                # keep existing list in sync for multi-fill dedupe in same event
                existing = list(bot.get("fills") or [])

            bot["fills"] = (bot.get("fills") or [])[:300]
        port_rows = _maybe_portfolio_risk(book, mids, cfg)
        if port_rows:
            logged.extend(port_rows)
        save_paper(book)

    if logged:
        try:
            from utils.hl_bitget_executor import maybe_execute_rows_async

            maybe_execute_rows_async(logged)
        except Exception:
            logger.exception("HL Bitget live hook failed")
    return logged


def _mark_one(pos: dict, mid: float) -> float:
    entry = float(pos.get("entry_px") or 0)
    sz = float(pos.get("sz") or 0)
    if entry <= 0 or abs(sz) < 1e-16 or mid <= 0:
        pos["u_pnl"] = 0.0
        return 0.0
    if sz > 0:
        upnl = (mid - entry) * sz
    else:
        upnl = (entry - mid) * abs(sz)
    pos["u_pnl"] = round(upnl, 4)
    pos["mark_px"] = mid
    # leverage vs bot equity recorded separately; keep target lev on pos
    return upnl


def _realize(bot: dict, pos: dict, exit_px: float, close_sz: float) -> float:
    entry = float(pos.get("entry_px") or exit_px)
    signed = float(pos.get("sz") or 0)
    qty = min(abs(signed), abs(close_sz))
    if qty <= 1e-16:
        return 0.0
    if signed > 0:
        pnl = (exit_px - entry) * qty
    else:
        pnl = (entry - exit_px) * qty
    bot["balance"] = round(float(bot.get("balance") or 0) + pnl, 4)
    bot["realized_pnl"] = round(float(bot.get("realized_pnl") or 0) + pnl, 4)
    return round(pnl, 4)


def _roll_day(bot: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Track Beijing calendar day for stats only — does NOT clear risk_halted."""
    day = _beijing_day()
    sizing = float(bot.get("balance") or cfg["bot_balance"])
    if bot.get("day_key") != day:
        bot["day_key"] = day
        if bot.get("day_start_equity") is None:
            bot["day_start_equity"] = float(bot.get("equity") or sizing)


def _bot_risk_anchor(bot: dict[str, Any], cfg: dict[str, Any]) -> float:
    """Equity baseline for per-bot −20% hard stop (last portfolio rebase / unlock)."""
    sizing = float(bot.get("balance") or cfg.get("bot_balance") or 1000)
    for key in ("risk_anchor_equity", "day_start_equity"):
        raw = bot.get(key)
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v > 1e-9:
            return v
    return max(sizing, 1e-9)


def _reset_bot_after_portfolio_rebase(bot: dict[str, Any]) -> None:
    """Clear per-bot halt and rebase its −20% anchor to current equity."""
    _recompute_bot(bot)
    eq = float(bot.get("equity") or bot.get("balance") or 0)
    bot["risk_halted"] = False
    bot.pop("risk_halted_at", None)
    bot["risk_anchor_equity"] = round(eq, 4)
    bot["day_start_equity"] = round(eq, 4)


def _iter_bots(book: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for bot in (book.get("bots") or {}).values():
        if isinstance(bot, dict):
            out.append(bot)
    return out


def _active_bots(book: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in _iter_bots(book) if not b.get("risk_halted")]


def _halted_bots(book: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in _iter_bots(book) if b.get("risk_halted")]


def _portfolio_equity(book: dict[str, Any], *, active_only: bool = False) -> float:
    total = 0.0
    for bot in _iter_bots(book):
        if active_only and bot.get("risk_halted"):
            continue
        _recompute_bot(bot)
        total += float(bot.get("equity") or bot.get("balance") or 0)
    return round(total, 4)


def _portfolio_active_anchor(book: dict[str, Any], cfg: dict[str, Any]) -> float:
    """Sum of per-bot risk anchors for non-halted bots (desk return basis)."""
    total = 0.0
    for bot in _active_bots(book):
        total += _bot_risk_anchor(bot, cfg)
    return round(total, 4)


def _ensure_portfolio_anchor(book: dict[str, Any], cfg: dict[str, Any] | None = None) -> float:
    """Seed desk + per-bot anchors when missing."""
    cfg = cfg or paper_config()
    for bot in _iter_bots(book):
        if bot.get("risk_anchor_equity") is None:
            _recompute_bot(bot)
            eq = float(bot.get("equity") or bot.get("balance") or 0)
            bot["risk_anchor_equity"] = round(eq, 4)
    try:
        anchor = float(book["portfolio_anchor_equity"])
        if anchor > 1e-9:
            return anchor
    except (TypeError, ValueError, KeyError):
        pass
    anchor = _portfolio_equity(book, active_only=False)
    book["portfolio_anchor_equity"] = anchor
    return anchor


def _maybe_release_bot_halt_cooldown(
    book: dict[str, Any], cfg: dict[str, Any]
) -> list[str]:
    """Unlock per-bot halt after cooldown; rebase that bot's −20% anchor."""
    cool = float(cfg.get("bot_halt_cooldown_sec") or 0)
    if cool <= 0:
        return []
    now = time.time()
    released: list[str] = []
    for bot in _halted_bots(book):
        try:
            halted_at = float(bot.get("risk_halted_at") or 0)
        except (TypeError, ValueError):
            halted_at = 0.0
        if halted_at <= 0 or (now - halted_at) < cool:
            continue
        _reset_bot_after_portfolio_rebase(bot)
        released.append(str(bot.get("id") or ""))
        logger.info(
            "HL per-bot halt cooldown release %s after %.0fs",
            bot.get("id"),
            cool,
        )
    return released


def _flatten_bot_positions(
    bot: dict[str, Any],
    mids: dict[str, float],
    *,
    action: str,
    risk_reason: str,
    keep_halted: bool = False,
) -> list[dict[str, Any]]:
    """Realize all paper positions on one bot; return Bitget sync rows."""
    fills = list(bot.get("fills") or [])
    now = _now()
    sync_rows: list[dict[str, Any]] = []
    for pos in list((bot.get("positions") or {}).values()):
        coin = str(pos.get("coin") or "")
        mid = (
            _mid_for_coin(mids, coin)
            or float(pos.get("mark_px") or 0)
            or float(pos.get("entry_px") or 0)
        )
        signed = float(pos.get("sz") or 0)
        qty = abs(signed)
        pnl = _realize(bot, pos, mid, qty) if mid > 0 else 0.0
        side = "sell" if signed > 0 else "buy"
        fills.insert(
            0,
            {
                "id": str(uuid.uuid4())[:8],
                "action": action,
                "source": bot.get("id"),
                "coin": coin,
                "side": side,
                "pos": "long" if signed > 0 else "short",
                "px": mid,
                "our_sz": qty,
                "notional": qty * mid,
                "leverage": pos.get("leverage"),
                "realized_pnl": pnl,
                "risk_reason": risk_reason,
                "ts": now,
            },
        )
        sync_rows.append(
            {
                "id": str(uuid.uuid4())[:8],
                "action": "close",
                "source": bot.get("id"),
                "coin": coin,
                "side": side,
                "px": mid,
                "our_sz": qty,
                "notional": qty * mid,
                "leverage": pos.get("leverage"),
                "skipped": False,
                "risk_halt": True,
                "risk_reason": risk_reason,
                "ts": now,
            }
        )
    bot["positions"] = {}
    bot["fills"] = fills[:300]
    bot["copy_ratio"] = 0.0
    bot["risk_halted"] = bool(keep_halted)
    if keep_halted:
        bot["risk_halted_at"] = time.time()
    else:
        bot.pop("risk_halted_at", None)
    _recompute_bot(bot)
    if not sync_rows:
        sync_rows.append(
            {
                "id": str(uuid.uuid4())[:8],
                "action": "close",
                "source": bot.get("id"),
                "coin": "",
                "our_sz": 0,
                "skipped": False,
                "risk_halt": True,
                "risk_reason": risk_reason,
                "ts": now,
            }
        )
    return sync_rows


def _reduce_bot_positions(
    bot: dict[str, Any],
    mids: dict[str, float],
    *,
    keep_frac: float,
    action: str,
    risk_reason: str,
) -> list[dict[str, Any]]:
    """Cut each position to keep_frac of size (soft take-profit)."""
    keep_frac = min(1.0, max(0.0, float(keep_frac)))
    if keep_frac >= 1.0 - 1e-12:
        return []
    if keep_frac <= 1e-12:
        return _flatten_bot_positions(
            bot, mids, action=action, risk_reason=risk_reason, keep_halted=False
        )

    fills = list(bot.get("fills") or [])
    now = _now()
    sync_rows: list[dict[str, Any]] = []
    for key, pos in list((bot.get("positions") or {}).items()):
        if not isinstance(pos, dict):
            continue
        coin = str(pos.get("coin") or "")
        mid = (
            _mid_for_coin(mids, coin)
            or float(pos.get("mark_px") or 0)
            or float(pos.get("entry_px") or 0)
        )
        signed = float(pos.get("sz") or 0)
        close_qty = abs(signed) * (1.0 - keep_frac)
        if close_qty <= 1e-16 or mid <= 0:
            continue
        pnl = _realize(bot, pos, mid, close_qty)
        new_sz = signed - (close_qty if signed > 0 else -close_qty)
        if abs(new_sz) <= 1e-12:
            (bot.get("positions") or {}).pop(key, None)
        else:
            pos["sz"] = new_sz
            if mid > 0:
                _mark_one(pos, mid)
        side = "sell" if signed > 0 else "buy"
        fills.insert(
            0,
            {
                "id": str(uuid.uuid4())[:8],
                "action": action,
                "source": bot.get("id"),
                "coin": coin,
                "side": side,
                "pos": "long" if signed > 0 else "short",
                "px": mid,
                "our_sz": close_qty,
                "notional": close_qty * mid,
                "leverage": pos.get("leverage"),
                "realized_pnl": pnl,
                "risk_reason": risk_reason,
                "ts": now,
            },
        )
        sync_rows.append(
            {
                "id": str(uuid.uuid4())[:8],
                "action": "reduce",
                "source": bot.get("id"),
                "coin": coin,
                "side": side,
                "px": mid,
                "our_sz": close_qty,
                "notional": close_qty * mid,
                "leverage": pos.get("leverage"),
                "skipped": False,
                "risk_halt": False,
                "risk_reason": risk_reason,
                "ts": now,
            }
        )
    bot["fills"] = fills[:300]
    _recompute_bot(bot)
    return sync_rows


def _hard_portfolio_rebase(
    book: dict[str, Any],
    mids: dict[str, float],
    cfg: dict[str, Any],
    *,
    reason: str,
    ret: float,
    anchor: float,
    equity: float,
) -> list[dict[str, Any]]:
    """Flatten all bots, compound-rebase anchors, clear halts + soft-TP flag."""
    action = (
        "risk_tp_close"
        if reason in ("portfolio_tp_hard", "portfolio_tp")
        else (
            "risk_sl_close"
            if reason == "portfolio_sl"
            else "risk_halt_close"
        )
    )
    sync_rows: list[dict[str, Any]] = []
    for bot in _iter_bots(book):
        if bot.get("positions"):
            sync_rows.extend(
                _flatten_bot_positions(
                    bot, mids, action=action, risk_reason=reason, keep_halted=False
                )
            )
        _reset_bot_after_portfolio_rebase(bot)

    new_eq = _portfolio_equity(book, active_only=False)
    book["portfolio_anchor_equity"] = new_eq
    book["portfolio_return_pct"] = 0.0
    book["portfolio_soft_tp_taken"] = False
    book["portfolio_copy_scale"] = 1.0
    book["portfolio_risk"] = {
        "reason": reason,
        "tripped_at": _now(),
        "anchor_before": round(anchor, 4),
        "equity_before": round(equity, 4),
        "return_pct": round(ret, 6),
        "anchor_after": round(new_eq, 4),
        "tp_soft_pct": float(cfg.get("portfolio_tp_pct") or 0),
        "tp_hard_pct": float(cfg.get("portfolio_tp_hard_pct") or 0),
        "sl_pct": float(cfg.get("portfolio_sl_pct") or 0),
    }
    logger.warning(
        "HL portfolio HARD %s ret=%.2f%% equity=%.2f anchor=%.2f → rebase %.2f "
        "(bots=%s closes=%s)",
        reason,
        ret * 100.0,
        equity,
        anchor,
        new_eq,
        len(_iter_bots(book)),
        len(sync_rows),
    )
    return sync_rows


def _maybe_portfolio_risk(
    book: dict[str, Any],
    mids: dict[str, float],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Full-desk return (all bots). Soft TP cut+scale; hard TP/SL/multi-halt rebase."""
    _maybe_release_bot_halt_cooldown(book, cfg)
    _ensure_portfolio_anchor(book, cfg)

    soft_tp = float(cfg.get("portfolio_tp_pct") or 0)
    hard_tp = float(cfg.get("portfolio_tp_hard_pct") or 0)
    sl = float(cfg.get("portfolio_sl_pct") or 0)
    soft_keep = float(cfg.get("portfolio_soft_reduce") or 0.5)
    halt_trigger = int(cfg.get("portfolio_halt_count_trigger") or 3)

    halted = _halted_bots(book)
    active = _active_bots(book)
    full_eq = _portfolio_equity(book, active_only=False)

    # Desk hard anchor = last compound rebase (all bots).
    try:
        anchor = float(book.get("portfolio_anchor_equity") or 0)
    except (TypeError, ValueError):
        anchor = 0.0
    if anchor <= 1e-9:
        anchor = full_eq
        book["portfolio_anchor_equity"] = anchor

    equity = full_eq
    ret = (equity - anchor) / anchor if anchor > 1e-9 else 0.0
    book["portfolio_return_pct"] = round(ret, 6)

    # ≥N bots halted (or no active left) → hard reset. Disabled when trigger ≤ 0.
    if halt_trigger > 0 and (
        len(halted) >= halt_trigger or (halted and not active)
    ):
        return _hard_portfolio_rebase(
            book,
            mids,
            cfg,
            reason="portfolio_multi_halt",
            ret=ret,
            anchor=anchor,
            equity=full_eq,
        )

    if soft_tp <= 0 and hard_tp <= 0 and sl <= 0:
        return []

    if sl > 0 and ret <= -sl:
        return _hard_portfolio_rebase(
            book, mids, cfg, reason="portfolio_sl", ret=ret, anchor=anchor, equity=equity
        )
    if hard_tp > 0 and ret >= hard_tp:
        return _hard_portfolio_rebase(
            book,
            mids,
            cfg,
            reason="portfolio_tp_hard",
            ret=ret,
            anchor=anchor,
            equity=equity,
        )

    soft_taken = bool(book.get("portfolio_soft_tp_taken"))
    if soft_tp > 0 and (not soft_taken) and ret >= soft_tp:
        sync_rows: list[dict[str, Any]] = []
        # Reduce open size on every bot that still has positions (halted are flat).
        for bot in _iter_bots(book):
            if bot.get("risk_halted") or not bot.get("positions"):
                continue
            sync_rows.extend(
                _reduce_bot_positions(
                    bot,
                    mids,
                    keep_frac=soft_keep,
                    action="risk_tp_reduce",
                    risk_reason="portfolio_tp_soft",
                )
            )
        book["portfolio_soft_tp_taken"] = True
        book["portfolio_copy_scale"] = soft_keep
        new_eq = _portfolio_equity(book, active_only=False)
        book["portfolio_risk"] = {
            "reason": "portfolio_tp_soft",
            "tripped_at": _now(),
            "anchor_before": round(anchor, 4),
            "equity_before": round(equity, 4),
            "equity_after": round(new_eq, 4),
            "return_pct": round(ret, 6),
            "keep_frac": soft_keep,
            "copy_scale": soft_keep,
            "tp_soft_pct": soft_tp,
            "tp_hard_pct": hard_tp,
            "sl_pct": sl,
        }
        book["portfolio_return_pct"] = round(
            (new_eq - anchor) / anchor if anchor > 1e-9 else 0.0, 6
        )
        logger.warning(
            "HL portfolio SOFT TP ret=%.2f%% keep=%.0f%% copy_scale=%.2f "
            "equity %.2f→%.2f (rows=%s)",
            ret * 100.0,
            soft_keep * 100.0,
            soft_keep,
            equity,
            new_eq,
            len(sync_rows),
        )
        return sync_rows

    return []


def _maybe_risk_halt(
    bot: dict[str, Any], mids: dict[str, float], cfg: dict[str, Any]
) -> list[dict[str, Any]] | None:
    """Per-bot hard stop: −daily_loss_pct vs risk_anchor_equity (default 20%).

    Halt until portfolio hard rebase OR bot_halt_cooldown_sec.
    """
    if float(cfg.get("daily_loss_pct") or 0) <= 0:
        return None

    _roll_day(bot, cfg)
    _recompute_bot(bot)
    sizing = float(bot.get("balance") or cfg["bot_balance"])
    anchor = _bot_risk_anchor(bot, cfg)
    if bot.get("risk_anchor_equity") is None:
        bot["risk_anchor_equity"] = round(anchor, 4)
    equity_now = float(bot.get("equity") or sizing)
    loss_pct = 0.0 if anchor <= 0 else (anchor - equity_now) / anchor
    if not (
        bot.get("risk_halted")
        or (cfg["daily_loss_pct"] > 0 and loss_pct >= cfg["daily_loss_pct"])
    ):
        return None

    already = bot.get("risk_halted") and not (bot.get("positions") or {})
    bot["risk_halted"] = True
    if already:
        return []

    rows = _flatten_bot_positions(
        bot,
        mids,
        action="risk_halt_close",
        risk_reason="bot_hard_stop",
        keep_halted=True,
    )
    bot["risk_halted"] = True
    bot["risk_halted_at"] = time.time()
    logger.warning(
        "HL per-bot hard stop %s loss_pct=%.1f%% equity=%.2f risk_anchor=%.2f "
        "(halt until portfolio rebase or cooldown)",
        bot.get("id"),
        loss_pct * 100.0,
        equity_now,
        anchor,
    )
    return rows


def _adjusted_leverage(target_lev: float | None, adjustment: float, symbol: str) -> int:
    """Cap paper leverage by base ticker (xyz:TSLA → TSLA)."""
    try:
        from utils.hl_bitget_symbol_map import hl_base_ticker

        base = hl_base_ticker(symbol) or str(symbol or "").upper()
    except Exception:
        base = str(symbol or "").upper().split(":")[-1]
    max_by_asset = {"BTC": 50, "ETH": 50, "SOL": 20, "HYPE": 10}
    cap = max_by_asset.get(base, 10)
    base_lev = float(target_lev or 1.0) * float(adjustment or 1.0)
    return max(1, min(cap, int(round(base_lev))))


def _parse_allow_coins(raw: Any) -> frozenset[str] | None:
    """None = unrestricted. Watchlist coins like TSLA match xyz:TSLA via hl_base_ticker."""
    if raw is None or raw == [] or raw == "*" or raw == "":
        return None
    if isinstance(raw, str):
        parts = [c.strip().upper() for c in raw.split(",") if c.strip()]
    else:
        parts = [str(c).strip().upper() for c in raw if str(c).strip()]
    if not parts:
        return None
    try:
        from utils.hl_bitget_symbol_map import hl_base_ticker

        return frozenset(hl_base_ticker(c) or c for c in parts)
    except Exception:
        return frozenset(parts)


def _bot_allow_coins(bot: dict[str, Any]) -> frozenset[str] | None:
    if "allow_coins" in bot:
        return _parse_allow_coins(bot.get("allow_coins"))
    bid = str(bot.get("id") or "")
    for w in load_watchlist():
        if str(w.get("id") or "") == bid:
            return _parse_allow_coins(w.get("coins"))
    return None


def _coin_base(coin: str) -> str:
    raw = str(coin or "").strip()
    if not raw:
        return ""
    try:
        from utils.hl_bitget_symbol_map import hl_base_ticker

        return hl_base_ticker(raw) or raw.upper().split(":")[-1]
    except Exception:
        base = raw.upper().split(":")[-1]
        if base.endswith("USDT"):
            base = base[:-4]
        return base


def _coin_allowed(coin: str, allow: frozenset[str] | None) -> bool:
    if allow is None:
        return True
    base = _coin_base(coin)
    return bool(base) and base in allow


def _scope_keys_for_coin(coin: str) -> set[str]:
    """Raw upper + base ticker so xyz:TSLA fills match snap/position keys."""
    raw = str(coin or "").strip()
    if not raw:
        return set()
    out = {raw.upper()}
    base = _coin_base(raw)
    if base:
        out.add(base.upper())
    return out


def _coin_in_scope(coin: str, scope: set[str] | frozenset[str] | None) -> bool:
    if scope is None:
        return True
    return bool(_scope_keys_for_coin(coin) & set(scope))


def _mirror_target_book(
    bot: dict[str, Any],
    snap: dict[str, Any],
    mids: dict[str, float],
    cfg: dict[str, Any],
    *,
    trigger_tids: list[str] | None = None,
    trigger_keys: list[tuple[str, str]] | None = None,
    scope_coins: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Align bot to target. With scope_coins, only touch those coins (fill-driven).

    Disallowed holdings (watchlist coins filter) are always flattened, even when
    out of the current fill scope — otherwise allowlist changes never clear them.
    """
    halt_rows = _maybe_risk_halt(bot, mids, cfg)
    if halt_rows is not None:
        return halt_rows

    _recompute_bot(bot)
    target_av = float(snap.get("account_value") or 0)
    your_eq = float(bot.get("equity") or bot.get("balance") or cfg["bot_balance"])
    ratio = (your_eq / target_av) if target_av > 1e-9 else 0.0
    bot["copy_ratio"] = round(ratio, 10)
    bot["target_av"] = target_av

    old = dict(bot.get("positions") or {})
    allow = _bot_allow_coins(bot)
    scope = {str(c).strip().upper() for c in (scope_coins or []) if str(c).strip()} or None
    desired: dict[str, dict[str, Any]] = {}
    for p in snap.get("positions") or []:
        coin = str(p.get("coin") or "").upper()
        if not coin:
            continue
        if not _coin_allowed(coin, allow):
            continue
        if not _coin_in_scope(coin, scope):
            continue
        try:
            t_sz = float(p.get("szi") or 0)
            entry = float(p.get("entry") or 0)
        except (TypeError, ValueError):
            continue
        if abs(t_sz) < 1e-16:
            continue
        our_sz = t_sz * ratio
        mid = _mid_for_coin(mids, coin) or float(entry or 0)
        px = entry or mid
        if abs(our_sz) * (px or 0) < cfg["min_notional"]:
            continue
        try:
            lev_raw = float(p["lev"]) if p.get("lev") is not None else 1.0
        except (TypeError, ValueError):
            lev_raw = 1.0
        our_lev = _adjusted_leverage(lev_raw, cfg.get("leverage_adjustment", 1.0), coin)
        key = f"{bot.get('id')}:{coin}"
        desired[key] = {
            "key": key,
            "source": bot.get("id"),
            "coin": coin,
            "sz": our_sz,
            "entry_px": px,
            "target_sz": t_sz,
            "target_av": target_av,
            "copy_ratio": round(ratio, 10),
            "leverage": our_lev,
            "target_address": bot.get("address"),
            "opened_at": (old.get(key) or {}).get("opened_at") or _now(),
            "u_pnl": 0.0,
            "mark_px": mid or None,
        }

    rows: list[dict[str, Any]] = []
    fills = list(bot.get("fills") or [])
    new_positions: dict[str, dict] = {}
    if scope is not None:
        for key, pos in old.items():
            coin = str(pos.get("coin") or "").upper()
            # Keep out-of-scope only if still allowlisted
            if coin and not _coin_in_scope(coin, scope) and _coin_allowed(coin, allow):
                new_positions[key] = pos
    tid0 = (trigger_tids or [None])[0]
    all_tids = [v for k, v in (trigger_keys or []) if k == "tid"] or list(trigger_tids or [])
    all_fps = [v for k, v in (trigger_keys or []) if k == "fp"]

    def _row(
        action: str,
        coin: str,
        qty: float,
        px: float,
        lev: Any,
        realized: float | None = None,
        side: str | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": str(uuid.uuid4())[:8],
            "action": action,
            "source": bot.get("id"),
            "coin": coin,
            "px": px,
            "our_sz": qty,
            "notional": qty * px,
            "leverage": lev,
            "copy_ratio": round(ratio, 10),
            "target_tid": tid0,
            "target_tids": all_tids[:20],
            "target_address": bot.get("address"),
            "ts": _now(),
        }
        if all_fps:
            out["target_fp"] = all_fps[0]
            out["target_fps"] = all_fps[:20]
        if side:
            out["side"] = side
        if realized is not None:
            out["realized_pnl"] = realized
        return out

    for key, pos in old.items():
        if key in desired:
            continue
        coin = str(pos.get("coin") or "").upper()
        allowed = _coin_allowed(coin, allow)
        in_scope = _coin_in_scope(coin, scope)
        # Fill-driven: only close in-scope flats; always close disallowed leftovers
        if scope is not None and not in_scope and allowed:
            continue
        mid = (
            _mid_for_coin(mids, coin)
            or float(pos.get("mark_px") or 0)
            or float(pos.get("entry_px") or 0)
        )
        qty = abs(float(pos.get("sz") or 0))
        pnl = _realize(bot, pos, mid, qty) if mid > 0 else 0.0
        row = _row(
            "close",
            coin,
            qty,
            mid,
            pos.get("leverage"),
            pnl,
            "sell" if float(pos.get("sz") or 0) > 0 else "buy",
        )
        rows.append(row)
        fills.insert(0, row)

    for key, want in desired.items():
        coin = want["coin"]
        mid = float(want.get("mark_px") or 0) or _mid_for_coin(mids, coin) or float(
            want["entry_px"] or 0
        )
        px = float(want["entry_px"] or mid)
        new_sz = float(want["sz"])
        old_pos = old.get(key)
        side = "buy" if new_sz > 0 else "sell"

        if not old_pos:
            pos = dict(want)
            if mid > 0:
                _mark_one(pos, mid)
            new_positions[key] = pos
            row = _row("open", coin, abs(new_sz), px, want.get("leverage"), side=side)
            row["target_sz"] = want.get("target_sz")
            rows.append(row)
            fills.insert(0, row)
            continue

        old_sz = float(old_pos.get("sz") or 0)

        if old_sz * new_sz < 0 and abs(old_sz) > 1e-16:
            pnl = _realize(bot, old_pos, mid or px, abs(old_sz)) if (mid or px) > 0 else 0.0
            close_row = _row(
                "close",
                coin,
                abs(old_sz),
                mid or px,
                want.get("leverage"),
                pnl,
                "sell" if old_sz > 0 else "buy",
            )
            rows.append(close_row)
            fills.insert(0, close_row)
            pos = dict(want)
            pos["opened_at"] = _now()
            if mid > 0:
                _mark_one(pos, mid)
            new_positions[key] = pos
            open_row = _row("open", coin, abs(new_sz), px, want.get("leverage"), side=side)
            open_row["target_sz"] = want.get("target_sz")
            rows.append(open_row)
            fills.insert(0, open_row)
            continue

        if abs(new_sz) + 1e-12 < abs(old_sz) and (mid or px) > 0:
            closed = abs(old_sz) - abs(new_sz)
            pnl = _realize(bot, old_pos, mid or px, closed)
            row = _row(
                "reduce",
                coin,
                closed,
                mid or px,
                want.get("leverage"),
                pnl,
                "sell" if old_sz > 0 else "buy",
            )
            rows.append(row)
            fills.insert(0, row)
        elif abs(new_sz) > abs(old_sz) + 1e-12:
            add = abs(new_sz) - abs(old_sz)
            old_entry = float(old_pos.get("entry_px") or px)
            old_abs = abs(old_sz)
            if old_abs + add > 0:
                want["entry_px"] = (old_entry * old_abs + px * add) / (old_abs + add)
            row = _row("increase", coin, add, px, want.get("leverage"), side=side)
            row["target_sz"] = want.get("target_sz")
            rows.append(row)
            fills.insert(0, row)

        pos = dict(old_pos)
        pos["sz"] = new_sz
        pos["target_sz"] = want.get("target_sz")
        pos["copy_ratio"] = want.get("copy_ratio")
        pos["leverage"] = want.get("leverage")
        pos["target_av"] = target_av
        if abs(new_sz) > abs(old_sz) + 1e-12:
            pos["entry_px"] = want["entry_px"]
        if mid > 0:
            _mark_one(pos, mid)
        new_positions[key] = pos

    for pos in new_positions.values():
        coin = str(pos.get("coin") or "")
        mid = _mid_for_coin(mids, coin) or float(pos.get("mark_px") or 0)
        if mid > 0:
            _mark_one(pos, mid)

    for kind, value in trigger_keys or []:
        if not value:
            continue
        if _seen_fill_key(fills, kind, value):
            continue
        mark: dict[str, Any] = {
            "id": str(uuid.uuid4())[:8],
            "action": "signal",
            "skipped": True,
            "source": bot.get("id"),
            "ts": _now(),
        }
        if kind == "tid":
            mark["target_tid"] = value
            mark["target_tids"] = [value]
        else:
            mark["target_fp"] = value
        fills.insert(0, mark)

    bot["positions"] = new_positions
    bot["fills"] = fills[:300]
    _recompute_bot(bot)
    return rows

"""Hyperliquid paper copy — immediate fill-delta market follow (no snapshot seed).

One bot per watchlist address (default 1000U):
  our_delta = fill.sz × (bot_equity / target_AV)
  trade/entry at fill.px (market), not target average entry
  hard cap: |notional| ≤ equity × leverage_cap

WS snapshots are ignored so deploy starts flat. Mark refresh only updates uPnL.
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

from utils.hl_short_term import load_watchlist, snapshot_positions as hl_snapshot_positions
from utils.rate_limit import MinIntervalGuard

logger = logging.getLogger(__name__)

PAPER_NAME = "hl_paper_copy.json"
_lock = threading.Lock()
_mids_cache: dict[str, float] = {}
_mids_cache_at: float = 0.0
_mark_guard = MinIntervalGuard("HL_PAPER_MARK_COOLDOWN_SEC", 45.0)
_mids_ttl_sec = float(os.getenv("HL_MIDS_CACHE_SEC", "30") or 30)
_av_ttl_sec = float(os.getenv("HL_TARGET_AV_TTL_SEC", "30") or 30)


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
        "daily_loss_pct": _env_float("HL_DAILY_LOSS_PCT", 0.20),
        "note": (
            "Fill-delta market follow: ignore WS snapshots; on each live fill "
            "our_delta = fill.sz × equity/target_AV at fill.px; "
            "notional capped by equity × leverage."
        ),
    }


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
        "risk_halted": False,
        "day_key": None,
        "day_start_equity": balance,
    }


def _ensure_bots(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy single-ledger → multi-bot, ensure every watchlist wallet has a bot."""
    cfg = paper_config()
    bal = cfg["bot_balance"]
    wallets = load_watchlist()
    bots = data.get("bots")
    if not isinstance(bots, dict):
        bots = {}
        # migrate old flat positions if present
        legacy_pos = data.get("positions") if isinstance(data.get("positions"), dict) else {}
        legacy_fills = data.get("fills") if isinstance(data.get("fills"), list) else []
        legacy_bal = float(data.get("balance") or bal)
        for i, w in enumerate(wallets):
            bid = str(w.get("id") or w.get("address") or "")[:32]
            bot = _empty_bot(w, bal if i > 0 else legacy_bal)
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
            bots[bid] = bot

    want_ids: set[str] = set()
    for w in wallets:
        bid = str(w.get("id") or w.get("address") or "")[:32]
        want_ids.add(bid)
        if bid not in bots:
            bots[bid] = _empty_bot(w, bal)
        else:
            bots[bid]["id"] = bid
            bots[bid]["address"] = w.get("address")
            bots[bid].setdefault("positions", {})
            bots[bid].setdefault("fills", [])
            bots[bid].setdefault("realized_pnl", 0.0)
            bots[bid].setdefault("risk_halted", False)
        # Keep paper allowlist in sync with watchlist coins (None = all)
        allow = _parse_allow_coins(w.get("coins"))
        if allow is None:
            bots[bid]["allow_coins"] = None
        else:
            bots[bid]["allow_coins"] = sorted(allow)

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
    for bot in bots.values():
        _recompute_bot(bot)
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
    data["ok"] = True
    data["mode"] = "fill_delta_market"
    data["config"] = paper_config()
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
        for bot in data["bots"].values():
            bal = cfg["bot_balance"]
            bot.update(_empty_bot(bot, bal))
        save_paper(data)
        return load_paper()


def fetch_all_mids(*, force: bool = False) -> dict[str, float]:
    """Cached allMids to avoid HL 429 under UI polling."""
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

    raw = http_json({"type": "allMids"})
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        payload = raw.get("mids") if "mids" in raw else raw
        if isinstance(payload, dict):
            for k, v in payload.items():
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
    _mids_cache = out
    _mids_cache_at = now
    return dict(out)


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
    """Mark-to-market only. Throttled; does not hit clearinghouse (ratio updates on fills)."""
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
        save_paper(data)
        _mark_guard.mark_used()
        return load_paper()


def _parse_live_fill(fill: dict) -> dict[str, Any] | None:
    """Extract coin, signed target delta, px, ids from one HL fill."""
    if not isinstance(fill, dict):
        return None
    coin = str(fill.get("coin") or "").strip()
    if not coin:
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


def _cache_target_meta(bot: dict[str, Any], snap: dict[str, Any] | None) -> None:
    if not snap:
        return
    try:
        av = float(snap.get("account_value") or 0)
    except (TypeError, ValueError):
        av = 0.0
    if av > 1e-9:
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


def _need_target_av_refresh(bot: dict[str, Any]) -> bool:
    av = float(bot.get("target_av") or 0)
    if av <= 1e-9:
        return True
    if _av_ttl_sec <= 0:
        return False
    at = float(bot.get("target_av_at") or 0)
    return (time.time() - at) >= _av_ttl_sec


def _copy_ratio(bot: dict[str, Any], cfg: dict[str, Any]) -> float:
    """equity / target_AV — sizing basis; caller must ensure target_av is set."""
    _recompute_bot(bot)
    eq = float(bot.get("equity") or bot.get("balance") or cfg["bot_balance"])
    av = float(bot.get("target_av") or 0)
    if av <= 1e-9 or eq <= 0:
        bot["copy_ratio"] = 0.0
        return 0.0
    ratio = eq / av
    bot["copy_ratio"] = round(ratio, 10)
    return ratio


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

    def _row(action: str, qty: float, trade_px: float, realized: float | None, side: str) -> dict:
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
        close_row = _row("close", abs(old_sz), px, pnl, close_side)
        rows.append(close_row)
        fills.insert(0, close_row)
        old = None
        old_sz = 0.0

    if abs(new_sz) < 1e-16:
        if old and abs(old_sz) > 1e-16:
            pnl = _realize(bot, old, px, abs(old_sz))
            close_side = "sell" if old_sz > 0 else "buy"
            close_row = _row("close", abs(old_sz), px, pnl, close_side)
            rows.append(close_row)
            fills.insert(0, close_row)
            positions.pop(key, None)
        bot["fills"] = fills[:300]
        _recompute_bot(bot)
        return rows

    applied_delta = new_sz - old_sz
    side = "buy" if applied_delta > 0 else "sell"

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
        open_row = _row("open", abs(new_sz), px, None, side)
        rows.append(open_row)
        fills.insert(0, open_row)
    elif abs(new_sz) + 1e-12 < abs(old_sz):
        closed = abs(old_sz) - abs(new_sz)
        pnl = _realize(bot, old, px, closed)
        red_row = _row("reduce", closed, px, pnl, side)
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
            inc_row = _row("increase", add, px, None, side)
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

    parsed: list[dict[str, Any]] = []
    for f in fills:
        item = _parse_live_fill(f)
        if item:
            parsed.append(item)
    if not parsed:
        return []

    addr = address.lower()
    recv_at = time.time()

    # Fresh AV for ratio (no artificial sleep). Size math depends on this.
    snap: dict[str, Any] | None = None
    try:
        snap = hl_snapshot_positions(address)
    except Exception as exc:
        logger.warning("target AV refresh failed %s: %s", address[:10], exc)

    try:
        mids = fetch_all_mids()
    except Exception:
        mids = dict(_mids_cache)

    cfg = paper_config()
    logged: list[dict] = []
    with _lock:
        book = load_paper()
        for bot in (book.get("bots") or {}).values():
            if str(bot.get("address") or "").lower() != addr:
                continue

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
                continue

            halt_rows = _maybe_risk_halt(bot, mids, cfg)
            if halt_rows is not None:
                logged.extend(halt_rows)
                continue

            ratio = _copy_ratio(bot, cfg)
            if ratio <= 0:
                logger.warning(
                    "HL follow skip %s: ratio=0 equity=%s av=%s",
                    bot.get("id"),
                    bot.get("equity"),
                    bot.get("target_av"),
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
                logged.extend(rows)
                # keep existing list in sync for multi-fill dedupe in same event
                existing = list(bot.get("fills") or [])

            bot["fills"] = (bot.get("fills") or [])[:300]
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
    day = _beijing_day()
    sizing = float(bot.get("balance") or cfg["bot_balance"])
    if bot.get("day_key") != day:
        bot["day_key"] = day
        bot["day_start_equity"] = float(bot.get("equity") or sizing)
        bot["risk_halted"] = False


def _maybe_risk_halt(
    bot: dict[str, Any], mids: dict[str, float], cfg: dict[str, Any]
) -> list[dict[str, Any]] | None:
    """If daily loss tripped, flatten paper and return Bitget sync trigger rows.

    Returns None if not halted; empty list if already flat-halted; non-empty rows
    to push Bitget sub-accounts to flat for the closed coins.
    """
    _roll_day(bot, cfg)
    sizing = float(bot.get("balance") or cfg["bot_balance"])
    day_start = float(bot.get("day_start_equity") or sizing)
    _recompute_bot(bot)
    equity_now = float(bot.get("equity") or sizing)
    loss_pct = 0.0 if day_start <= 0 else (day_start - equity_now) / day_start
    if not (
        bot.get("risk_halted")
        or (cfg["daily_loss_pct"] > 0 and loss_pct >= cfg["daily_loss_pct"])
    ):
        return None

    already = bot.get("risk_halted") and not (bot.get("positions") or {})
    bot["risk_halted"] = True
    if already:
        return []

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
        qty = abs(float(pos.get("sz") or 0))
        pnl = _realize(bot, pos, mid, qty) if mid > 0 else 0.0
        side = "sell" if float(pos.get("sz") or 0) > 0 else "buy"
        fills.insert(
            0,
            {
                "id": str(uuid.uuid4())[:8],
                "action": "risk_halt_close",
                "source": bot.get("id"),
                "coin": coin,
                "side": side,
                "px": mid,
                "our_sz": qty,
                "notional": qty * mid,
                "leverage": pos.get("leverage"),
                "realized_pnl": pnl,
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
                "ts": now,
            }
        )
    bot["positions"] = {}
    bot["fills"] = fills[:300]
    bot["copy_ratio"] = 0.0
    _recompute_bot(bot)
    # Even if no coins, return a bot-touch marker so sub sync discovers Bitget book
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
                "ts": now,
            }
        )
    return sync_rows


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

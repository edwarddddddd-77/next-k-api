"""Hyperliquid paper copy — one bot per target wallet (perfect proportional mirror).

Each bot has its own 1000U book bound to one address:
  our_sz = target_sz * (bot_balance / target_account_value)
  leverage = target leverage (recorded; size ratio already preserves exposure)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.hl_short_term import load_watchlist, snapshot as hl_snapshot

logger = logging.getLogger(__name__)

PAPER_NAME = "hl_paper_copy.json"
_lock = threading.Lock()


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
        "mode": "perfect",
        "bot_balance": _env_float("HL_PAPER_BALANCE", 1000.0),
        "copy_scale": 1.0,
        "min_notional": _env_float("HL_MIN_NOTIONAL", 5.0),
        "daily_loss_pct": _env_float("HL_DAILY_LOSS_PCT", 0.20),
        "note": "One bot per address, each with independent balance; perfect proportional size + same leverage.",
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    for w in wallets:
        bid = str(w.get("id") or w.get("address") or "")[:32]
        if bid not in bots:
            bots[bid] = _empty_bot(w, bal)
        else:
            bots[bid]["id"] = bid
            bots[bid]["address"] = w.get("address")
            bots[bid].setdefault("positions", {})
            bots[bid].setdefault("fills", [])
            bots[bid].setdefault("realized_pnl", 0.0)
            bots[bid].setdefault("risk_halted", False)

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
    data["mode"] = "perfect"
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


def fetch_all_mids() -> dict[str, float]:
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
    return out


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


def _sync_one_bot(
    bot: dict[str, Any],
    snap: dict[str, Any],
    mids: dict[str, float],
    cfg: dict[str, Any],
) -> None:
    """Perfect mirror: size by equity ratio, copy target leverage."""
    now = _now()
    day = now[:10]
    sizing = float(bot.get("balance") or cfg["bot_balance"])
    if bot.get("day_key") != day:
        bot["day_key"] = day
        bot["day_start_equity"] = float(bot.get("equity") or sizing)
        bot["risk_halted"] = False

    day_start = float(bot.get("day_start_equity") or sizing)
    _recompute_bot(bot)
    equity_now = float(bot.get("equity") or sizing)
    loss_pct = 0.0 if day_start <= 0 else (day_start - equity_now) / day_start

    old = dict(bot.get("positions") or {})
    fills = list(bot.get("fills") or [])

    if bot.get("risk_halted") or (
        cfg["daily_loss_pct"] > 0 and loss_pct >= cfg["daily_loss_pct"]
    ):
        bot["risk_halted"] = True
        for pos in old.values():
            coin = str(pos.get("coin") or "")
            mid = float(mids.get(coin) or pos.get("mark_px") or pos.get("entry_px") or 0)
            pnl = _realize(bot, pos, mid, abs(float(pos.get("sz") or 0))) if mid > 0 else 0.0
            fills.insert(
                0,
                {
                    "id": str(uuid.uuid4())[:8],
                    "action": "risk_halt_close",
                    "source": bot.get("id"),
                    "coin": coin,
                    "px": mid,
                    "our_sz": abs(float(pos.get("sz") or 0)),
                    "notional": abs(float(pos.get("sz") or 0)) * mid,
                    "leverage": pos.get("leverage"),
                    "realized_pnl": pnl,
                    "ts": now,
                },
            )
        bot["positions"] = {}
        bot["fills"] = fills[:300]
        bot["copy_ratio"] = 0.0
        _recompute_bot(bot)
        return

    target_av = float(snap.get("account_value") or 0)
    ratio = (sizing / target_av) if target_av > 1e-9 else 0.0
    bot["copy_ratio"] = round(ratio, 10)
    bot["target_av"] = target_av

    desired: dict[str, dict] = {}
    for p in snap.get("positions") or []:
        coin = str(p.get("coin") or "").upper()
        if not coin:
            continue
        try:
            t_sz = float(p.get("szi") or 0)
            entry = float(p.get("entry") or 0)
        except (TypeError, ValueError):
            continue
        if abs(t_sz) < 1e-16:
            continue
        our_sz = t_sz * ratio
        mid = float(mids.get(coin) or entry or 0)
        px = entry or mid
        notional = abs(our_sz) * px
        if notional < cfg["min_notional"]:
            continue
        lev = p.get("lev")
        try:
            lev_f = float(lev) if lev is not None else None
        except (TypeError, ValueError):
            lev_f = None
        # If HL doesn't give lev, infer from target notional / target equity
        if lev_f is None and target_av > 0 and px > 0:
            lev_f = round(abs(t_sz) * px / target_av, 2)

        key = f"{bot.get('id')}:{coin}"
        desired[key] = {
            "key": key,
            "source": bot.get("id"),
            "coin": coin,
            "sz": our_sz,
            "entry_px": entry or mid,
            "target_sz": t_sz,
            "target_av": target_av,
            "copy_ratio": round(ratio, 10),
            "leverage": lev_f,
            "target_address": bot.get("address"),
            "opened_at": (old.get(key) or {}).get("opened_at") or now,
            "u_pnl": 0.0,
            "mark_px": mid or None,
        }

    new_positions: dict[str, dict] = {}

    for key, pos in old.items():
        if key in desired:
            continue
        coin = str(pos.get("coin") or "")
        mid = float(mids.get(coin) or pos.get("mark_px") or pos.get("entry_px") or 0)
        pnl = _realize(bot, pos, mid, abs(float(pos.get("sz") or 0))) if mid > 0 else 0.0
        fills.insert(
            0,
            {
                "id": str(uuid.uuid4())[:8],
                "action": "sync_close",
                "source": bot.get("id"),
                "coin": coin,
                "px": mid,
                "our_sz": abs(float(pos.get("sz") or 0)),
                "notional": abs(float(pos.get("sz") or 0)) * mid,
                "leverage": pos.get("leverage"),
                "realized_pnl": pnl,
                "ts": now,
            },
        )

    for key, want in desired.items():
        mid = float(want.get("mark_px") or mids.get(want["coin"]) or want["entry_px"] or 0)
        old_pos = old.get(key)
        if not old_pos:
            pos = dict(want)
            if mid > 0:
                _mark_one(pos, mid)
            new_positions[key] = pos
            fills.insert(
                0,
                {
                    "id": str(uuid.uuid4())[:8],
                    "action": "sync_open",
                    "source": bot.get("id"),
                    "coin": want["coin"],
                    "px": want["entry_px"],
                    "our_sz": abs(want["sz"]),
                    "notional": abs(want["sz"]) * float(want["entry_px"] or mid or 0),
                    "side": "buy" if want["sz"] > 0 else "sell",
                    "leverage": want.get("leverage"),
                    "copy_ratio": want.get("copy_ratio"),
                    "ts": now,
                },
            )
            continue

        old_sz = float(old_pos.get("sz") or 0)
        new_sz = float(want["sz"])
        if old_sz * new_sz < 0 and abs(old_sz) > 1e-16:
            # flip
            pnl = _realize(bot, old_pos, mid, abs(old_sz)) if mid > 0 else 0.0
            fills.insert(
                0,
                {
                    "id": str(uuid.uuid4())[:8],
                    "action": "sync_flip",
                    "source": bot.get("id"),
                    "coin": want["coin"],
                    "px": mid,
                    "our_sz": abs(old_sz),
                    "notional": abs(old_sz) * mid,
                    "leverage": want.get("leverage"),
                    "realized_pnl": pnl,
                    "ts": now,
                },
            )
            pos = dict(want)
            pos["opened_at"] = now
            if mid > 0:
                _mark_one(pos, mid)
            new_positions[key] = pos
            continue

        if abs(new_sz) < abs(old_sz) - 1e-12 and mid > 0:
            pnl = _realize(bot, old_pos, mid, abs(old_sz) - abs(new_sz))
            fills.insert(
                0,
                {
                    "id": str(uuid.uuid4())[:8],
                    "action": "sync_reduce",
                    "source": bot.get("id"),
                    "coin": want["coin"],
                    "px": mid,
                    "our_sz": abs(old_sz) - abs(new_sz),
                    "notional": (abs(old_sz) - abs(new_sz)) * mid,
                    "leverage": want.get("leverage"),
                    "realized_pnl": pnl,
                    "ts": now,
                },
            )

        pos = dict(old_pos)
        pos["sz"] = new_sz
        pos["target_sz"] = want.get("target_sz")
        pos["copy_ratio"] = want.get("copy_ratio")
        pos["leverage"] = want.get("leverage")
        pos["entry_px"] = float(old_pos.get("entry_px") or want["entry_px"])
        if mid > 0:
            _mark_one(pos, mid)
        new_positions[key] = pos

    for pos in new_positions.values():
        coin = str(pos.get("coin") or "")
        mid = float(mids.get(coin) or pos.get("mark_px") or 0)
        if mid > 0:
            _mark_one(pos, mid)

    bot["positions"] = new_positions
    bot["fills"] = fills[:300]
    _recompute_bot(bot)


def sync_proportional(*, snaps: dict[str, dict] | None = None, mids: dict[str, float] | None = None) -> dict[str, Any]:
    """Sync every bot to its bound address — perfect proportional copy."""
    if not paper_enabled():
        return load_paper()

    cfg = paper_config()
    wallets = load_watchlist()
    if snaps is None:
        snaps = {}
        for w in wallets:
            addr = str(w.get("address") or "")
            if not addr:
                continue
            try:
                snaps[addr.lower()] = hl_snapshot(addr)
            except Exception as exc:
                logger.warning("bot sync snapshot failed %s: %s", addr[:10], exc)

    if mids is None:
        try:
            mids = fetch_all_mids()
        except Exception as exc:
            logger.warning("bot sync mids failed: %s", exc)
            mids = {}

    with _lock:
        data = load_paper()
        bots = data.get("bots") or {}
        for w in wallets:
            bid = str(w.get("id") or w.get("address") or "")[:32]
            bot = bots.get(bid)
            if not bot:
                continue
            addr = str(w.get("address") or "").lower()
            snap = snaps.get(addr) or {}
            _sync_one_bot(bot, snap, mids or {}, cfg)
        data["bots"] = bots
        save_paper(data)
        return load_paper()


def refresh_marks() -> dict[str, Any]:
    return sync_proportional()


def ingest_user_event(address: str, data: dict) -> list[dict]:
    """On target fill → resync that bot (and others) to stay perfectly aligned."""
    fills = data.get("fills")
    if not isinstance(fills, list) or not fills:
        return []
    addr = address.lower()
    logged: list[dict] = []
    with _lock:
        book = load_paper()
        for bot in (book.get("bots") or {}).values():
            if str(bot.get("address") or "").lower() != addr:
                continue
            rows = list(bot.get("fills") or [])
            for f in fills:
                if not isinstance(f, dict):
                    continue
                tid = str(f.get("tid") or f.get("hash") or "")
                if tid and any(str(x.get("target_tid") or "") == tid for x in rows):
                    continue
                row = {
                    "id": str(uuid.uuid4())[:8],
                    "action": "signal",
                    "source": bot.get("id"),
                    "target_address": address,
                    "coin": f.get("coin"),
                    "px": f.get("px"),
                    "target_sz": f.get("sz"),
                    "side": "buy" if str(f.get("side") or "").upper() in ("B", "BUY") else "sell",
                    "target_tid": tid or None,
                    "ts": _now(),
                }
                rows.insert(0, row)
                logged.append(row)
            bot["fills"] = rows[:300]
        save_paper(book)
    sync_proportional()
    return logged

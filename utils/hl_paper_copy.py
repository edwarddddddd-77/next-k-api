"""Hyperliquid paper copy — full proportional mirror of the target book.

One bot per watchlist address (default 1000U):
  our_sz = target_szi × (bot_balance / target_account_value)

On each real target fill, re-read their positions and align our book
(open / increase / reduce / close / flip). Mark refresh only updates uPnL.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
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
        "mode": "proportional_mirror",
        "bot_balance": _env_float("HL_PAPER_BALANCE", 1000.0),
        "copy_scale": 1.0,
        "min_notional": _env_float("HL_MIN_NOTIONAL", 10.0),
        "leverage_adjustment": _env_float("HL_LEVERAGE_ADJUSTMENT", 1.0),
        "daily_loss_pct": _env_float("HL_DAILY_LOSS_PCT", 0.20),
        "note": (
            "Full proportional mirror of target book: "
            "our_sz = target_szi × (bot_balance / target_account_value); "
            "open/close/reduce/flip all followed."
        ),
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
    data["mode"] = "proportional_mirror"
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


def _roll_day(bot: dict[str, Any], cfg: dict[str, Any]) -> None:
    now = _now()
    day = now[:10]
    sizing = float(bot.get("balance") or cfg["bot_balance"])
    if bot.get("day_key") != day:
        bot["day_key"] = day
        bot["day_start_equity"] = float(bot.get("equity") or sizing)
        bot["risk_halted"] = False


def _maybe_risk_halt(bot: dict[str, Any], mids: dict[str, float], cfg: dict[str, Any]) -> bool:
    """If daily loss tripped, flatten and log once. Returns True if halted."""
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
        return False

    already = bot.get("risk_halted") and not (bot.get("positions") or {})
    bot["risk_halted"] = True
    if already:
        return True

    fills = list(bot.get("fills") or [])
    now = _now()
    for pos in list((bot.get("positions") or {}).values()):
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
    return True


def _adjusted_leverage(target_lev: float | None, adjustment: float, symbol: str) -> int:
    max_by_asset = {"BTC": 50, "ETH": 50, "SOL": 20, "HYPE": 10}
    cap = max_by_asset.get((symbol or "").upper(), 10)
    base = float(target_lev or 1.0) * float(adjustment or 1.0)
    return max(1, min(cap, int(round(base))))


def _mirror_target_book(
    bot: dict[str, Any],
    snap: dict[str, Any],
    mids: dict[str, float],
    cfg: dict[str, Any],
    *,
    trigger_tids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Align bot to our_sz = target_szi * (balance / target_av)."""
    if _maybe_risk_halt(bot, mids, cfg):
        return []

    target_av = float(snap.get("account_value") or 0)
    your_bal = float(bot.get("balance") or cfg["bot_balance"])
    ratio = (your_bal / target_av) if target_av > 1e-9 else 0.0
    bot["copy_ratio"] = round(ratio, 10)
    bot["target_av"] = target_av

    old = dict(bot.get("positions") or {})
    desired: dict[str, dict[str, Any]] = {}
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
    tid0 = (trigger_tids or [None])[0]

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
            "target_address": bot.get("address"),
            "ts": _now(),
        }
        if side:
            out["side"] = side
        if realized is not None:
            out["realized_pnl"] = realized
        return out

    for key, pos in old.items():
        if key in desired:
            continue
        coin = str(pos.get("coin") or "")
        mid = float(mids.get(coin) or pos.get("mark_px") or pos.get("entry_px") or 0)
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
        mid = float(want.get("mark_px") or mids.get(coin) or want["entry_px"] or 0)
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
            close_row = _row("close", coin, abs(old_sz), mid or px, want.get("leverage"), pnl)
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
            row = _row("reduce", coin, closed, mid or px, want.get("leverage"), pnl)
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
        mid = float(mids.get(coin) or pos.get("mark_px") or 0)
        if mid > 0:
            _mark_one(pos, mid)

    for tid in trigger_tids or []:
        if not tid:
            continue
        if any(str(x.get("target_tid") or "") == tid for x in fills):
            continue
        fills.insert(
            0,
            {
                "id": str(uuid.uuid4())[:8],
                "action": "signal",
                "skipped": True,
                "source": bot.get("id"),
                "target_tid": tid,
                "ts": _now(),
            },
        )

    bot["positions"] = new_positions
    bot["fills"] = fills[:300]
    _recompute_bot(bot)
    return rows


def refresh_marks() -> dict[str, Any]:
    """Mark-to-market only — do not resize (avoids sync noise)."""
    if not paper_enabled():
        return load_paper()
    try:
        mids = fetch_all_mids()
    except Exception as exc:
        logger.warning("paper mark mids failed: %s", exc)
        mids = {}

    with _lock:
        data = load_paper()
        cfg = paper_config()
        for bot in (data.get("bots") or {}).values():
            _roll_day(bot, cfg)
            for pos in (bot.get("positions") or {}).values():
                coin = str(pos.get("coin") or "")
                mid = float(mids.get(coin) or pos.get("mark_px") or 0)
                if mid > 0:
                    _mark_one(pos, mid)
            _recompute_bot(bot)
            addr = str(bot.get("address") or "")
            if addr:
                try:
                    snap = hl_snapshot(addr)
                    tav = float(snap.get("account_value") or 0)
                    bal = float(bot.get("balance") or cfg["bot_balance"])
                    bot["target_av"] = tav
                    bot["copy_ratio"] = round(bal / tav, 10) if tav > 1e-9 else 0.0
                except Exception:
                    pass
        save_paper(data)
        return load_paper()


def ingest_user_event(address: str, data: dict) -> list[dict]:
    """On target fill(s): mirror entire target book at current ratio."""
    fills = data.get("fills")
    if not isinstance(fills, list) or not fills:
        return []
    if not paper_enabled():
        return []

    addr = address.lower()
    tids: list[str] = []
    for f in fills:
        if isinstance(f, dict):
            tid = str(f.get("tid") or f.get("hash") or "")
            if tid:
                tids.append(tid)

    time.sleep(1.0)
    try:
        snap = hl_snapshot(address)
    except Exception as exc:
        logger.warning("mirror snapshot failed %s: %s", address[:10], exc)
        return []
    try:
        mids = fetch_all_mids()
    except Exception:
        mids = {}

    cfg = paper_config()
    logged: list[dict] = []
    with _lock:
        book = load_paper()
        for bot in (book.get("bots") or {}).values():
            if str(bot.get("address") or "").lower() != addr:
                continue
            existing = list(bot.get("fills") or [])
            if tids and all(
                any(str(x.get("target_tid") or "") == t for x in existing) for t in tids
            ):
                continue
            logged.extend(_mirror_target_book(bot, snap, mids, cfg, trigger_tids=tids))
            bot["fills"] = (bot.get("fills") or [])[:300]
        save_paper(book)
    return logged

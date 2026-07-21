"""Hyperliquid paper copy ledger — simulate following watchlist wallets (no keys)."""

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
        "mode": (os.getenv("HL_COPY_MODE") or "ratio").strip().lower() or "ratio",
        "initial_balance": _env_float("HL_PAPER_BALANCE", 1000.0),
        "leverage_adjustment": _env_float("HL_LEVERAGE_ADJUSTMENT", 0.3),
        "min_notional": _env_float("HL_MIN_NOTIONAL", 10.0),
        "copy_all": _env_bool("HL_COPY_ALL", True),
        "note": "ratio = mirror target positions by equity ratio; fills = accumulate from WS fills only.",
    }


def _empty_ledger() -> dict[str, Any]:
    cfg = paper_config()
    bal = cfg["initial_balance"]
    return {
        "ok": True,
        "mode": "paper",
        "balance": bal,
        "equity": bal,
        "realized_pnl": 0.0,
        "positions": {},
        "fills": [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
    }


def load_paper() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        return _empty_ledger()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _empty_ledger()
        data.setdefault("positions", {})
        data.setdefault("fills", [])
        data.setdefault("realized_pnl", 0.0)
        data["ok"] = True
        data["config"] = paper_config()
        return data
    except Exception as exc:
        logger.warning("hl paper load failed: %s", exc)
        out = _empty_ledger()
        out["error"] = str(exc)
        return out


def save_paper(data: dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["ok"] = True
    data["config"] = paper_config()
    # mark-to-market equity = balance + unrealized (entry-based, no live mids here)
    unreal = 0.0
    for pos in (data.get("positions") or {}).values():
        unreal += float(pos.get("u_pnl") or 0)
    data["equity"] = round(float(data.get("balance") or 0) + unreal, 4)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_paper() -> dict[str, Any]:
    with _lock:
        data = _empty_ledger()
        save_paper(data)
        return data


def _wallet_meta(address: str) -> dict[str, Any]:
    addr = address.lower()
    for w in load_watchlist():
        if str(w.get("address") or "").lower() == addr:
            return w
    return {"address": address, "id": address[:10]}


def _is_close_fill(fill: dict) -> bool:
    closed = fill.get("closedPnl")
    try:
        if closed is not None and abs(float(closed)) > 1e-12:
            return True
    except (TypeError, ValueError):
        pass
    direction = str(fill.get("dir") or "").lower()
    return "close" in direction or "reduce" in direction


def _scaled_size(target_sz: float, target_av: float, paper_equity: float, lev_adj: float) -> float:
    if target_av <= 1e-9 or paper_equity <= 0:
        return 0.0
    return abs(float(target_sz)) * (paper_equity / target_av) * lev_adj


def copy_ratio(paper_equity: float, target_av: float, lev_adj: float) -> float:
    """our_size = target_size * ratio."""
    if target_av <= 1e-9 or paper_equity <= 0:
        return 0.0
    return (paper_equity / target_av) * lev_adj


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
    return upnl


def _realize(data: dict, pos: dict, exit_px: float, close_sz: float) -> float:
    entry = float(pos.get("entry_px") or exit_px)
    signed = float(pos.get("sz") or 0)
    qty = min(abs(signed), abs(close_sz))
    if qty <= 1e-16:
        return 0.0
    if signed > 0:
        pnl = (exit_px - entry) * qty
    else:
        pnl = (entry - exit_px) * qty
    data["balance"] = round(float(data.get("balance") or 0) + pnl, 4)
    data["realized_pnl"] = round(float(data.get("realized_pnl") or 0) + pnl, 4)
    return round(pnl, 4)


def sync_proportional(*, snaps: dict[str, dict] | None = None, mids: dict[str, float] | None = None) -> dict[str, Any]:
    """
    Rebalance paper positions to:
      our_sz = target_sz * (paper_balance / target_account_value) * leverage_adjustment

    Uses cash balance (not equity) as the sizing base to avoid feedback from uPnL.
    """
    if not paper_enabled():
        return load_paper()

    cfg = paper_config()
    wallets = load_watchlist()
    if not cfg["copy_all"]:
        wallets = [w for w in wallets if int(w.get("priority") or 99) == 1]

    if snaps is None:
        snaps = {}
        for w in wallets:
            addr = str(w.get("address") or "")
            if not addr:
                continue
            try:
                snaps[addr.lower()] = hl_snapshot(addr)
            except Exception as exc:
                logger.warning("sync snapshot failed %s: %s", addr[:10], exc)

    if mids is None:
        try:
            mids = fetch_all_mids()
        except Exception as exc:
            logger.warning("sync mids failed: %s", exc)
            mids = {}

    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        data = load_paper()
        sizing = float(data.get("balance") or cfg["initial_balance"])
        old_positions = dict(data.get("positions") or {})
        desired: dict[str, dict] = {}
        ratios: dict[str, float] = {}

        for w in wallets:
            addr = str(w.get("address") or "")
            if not addr:
                continue
            snap = snaps.get(addr.lower()) or {}
            target_av = float(snap.get("account_value") or 0)
            ratio = copy_ratio(sizing, target_av, cfg["leverage_adjustment"])
            source = str(w.get("id") or addr[:10])
            ratios[source] = ratio
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
                our_sz = t_sz * ratio  # signed
                notional = abs(our_sz) * (entry or float(mids.get(coin) or 0))
                if notional < cfg["min_notional"]:
                    continue
                key = f"{source}:{coin}"
                mid = float(mids.get(coin) or entry or 0)
                desired[key] = {
                    "key": key,
                    "source": source,
                    "coin": coin,
                    "sz": our_sz,
                    "entry_px": entry or mid,
                    "target_sz": t_sz,
                    "target_av": target_av,
                    "copy_ratio": round(ratio, 8),
                    "target_address": addr,
                    "opened_at": (old_positions.get(key) or {}).get("opened_at") or now,
                    "u_pnl": 0.0,
                    "mark_px": mid or None,
                }

        fills = list(data.get("fills") or [])
        new_positions: dict[str, dict] = {}

        # Close positions no longer desired
        for key, pos in old_positions.items():
            if key in desired:
                continue
            coin = str(pos.get("coin") or "")
            mid = float(mids.get(coin) or pos.get("mark_px") or pos.get("entry_px") or 0)
            pnl = _realize(data, pos, mid, abs(float(pos.get("sz") or 0))) if mid > 0 else 0.0
            fills.insert(
                0,
                {
                    "id": str(uuid.uuid4())[:8],
                    "skipped": False,
                    "action": "sync_close",
                    "source": pos.get("source"),
                    "coin": coin,
                    "px": mid,
                    "our_sz": abs(float(pos.get("sz") or 0)),
                    "notional": abs(float(pos.get("sz") or 0)) * mid,
                    "side": "sell" if float(pos.get("sz") or 0) > 0 else "buy",
                    "realized_pnl": pnl,
                    "copy_ratio": ratios.get(str(pos.get("source") or ""), None),
                    "ts": now,
                },
            )

        for key, want in desired.items():
            mid = float(want.get("mark_px") or mids.get(want["coin"]) or want["entry_px"] or 0)
            old = old_positions.get(key)
            if not old:
                pos = dict(want)
                _mark_one(pos, mid) if mid > 0 else None
                new_positions[key] = pos
                fills.insert(
                    0,
                    {
                        "id": str(uuid.uuid4())[:8],
                        "skipped": False,
                        "action": "sync_open",
                        "source": want["source"],
                        "coin": want["coin"],
                        "px": want["entry_px"],
                        "our_sz": abs(want["sz"]),
                        "notional": abs(want["sz"]) * float(want["entry_px"] or mid or 0),
                        "side": "buy" if want["sz"] > 0 else "sell",
                        "copy_ratio": want.get("copy_ratio"),
                        "target_sz": want.get("target_sz"),
                        "ts": now,
                    },
                )
                continue

            old_sz = float(old.get("sz") or 0)
            new_sz = float(want["sz"])
            # Same direction: resize, keep entry; opposite: realize then reopen
            if old_sz * new_sz > 0 or abs(old_sz) < 1e-16:
                # reduce realizes partial PnL
                if abs(new_sz) < abs(old_sz) - 1e-12 and mid > 0:
                    pnl = _realize(data, old, mid, abs(old_sz) - abs(new_sz))
                    fills.insert(
                        0,
                        {
                            "id": str(uuid.uuid4())[:8],
                            "skipped": False,
                            "action": "sync_reduce",
                            "source": want["source"],
                            "coin": want["coin"],
                            "px": mid,
                            "our_sz": abs(old_sz) - abs(new_sz),
                            "notional": (abs(old_sz) - abs(new_sz)) * mid,
                            "side": "sell" if old_sz > 0 else "buy",
                            "realized_pnl": pnl,
                            "copy_ratio": want.get("copy_ratio"),
                            "ts": now,
                        },
                    )
                pos = dict(old)
                pos["sz"] = new_sz
                pos["target_sz"] = want.get("target_sz")
                pos["target_av"] = want.get("target_av")
                pos["copy_ratio"] = want.get("copy_ratio")
                pos["entry_px"] = float(old.get("entry_px") or want["entry_px"])
                if mid > 0:
                    _mark_one(pos, mid)
                new_positions[key] = pos
            else:
                # flip
                pnl = _realize(data, old, mid, abs(old_sz)) if mid > 0 else 0.0
                fills.insert(
                    0,
                    {
                        "id": str(uuid.uuid4())[:8],
                        "skipped": False,
                        "action": "sync_flip",
                        "source": want["source"],
                        "coin": want["coin"],
                        "px": mid,
                        "our_sz": abs(old_sz),
                        "notional": abs(old_sz) * mid,
                        "side": "sell" if old_sz > 0 else "buy",
                        "realized_pnl": pnl,
                        "copy_ratio": want.get("copy_ratio"),
                        "ts": now,
                    },
                )
                pos = dict(want)
                pos["opened_at"] = now
                if mid > 0:
                    _mark_one(pos, mid)
                new_positions[key] = pos
                fills.insert(
                    0,
                    {
                        "id": str(uuid.uuid4())[:8],
                        "skipped": False,
                        "action": "sync_open",
                        "source": want["source"],
                        "coin": want["coin"],
                        "px": want["entry_px"],
                        "our_sz": abs(new_sz),
                        "notional": abs(new_sz) * float(want["entry_px"] or mid or 0),
                        "side": "buy" if new_sz > 0 else "sell",
                        "copy_ratio": want.get("copy_ratio"),
                        "ts": now,
                    },
                )

        # Mark remaining
        for pos in new_positions.values():
            coin = str(pos.get("coin") or "")
            mid = float(mids.get(coin) or pos.get("mark_px") or 0)
            if mid > 0:
                _mark_one(pos, mid)

        data["positions"] = new_positions
        data["fills"] = fills[:500]
        data["copy_ratios"] = ratios
        data["sizing_balance"] = sizing
        save_paper(data)
        return data


def _target_account_value(address: str) -> float:
    try:
        snap = hl_snapshot(address)
        return float(snap.get("account_value") or 0)
    except Exception as exc:
        logger.warning("target AV fetch failed %s: %s", address[:10], exc)
        return 0.0


def apply_target_fill(address: str, fill: dict) -> dict[str, Any] | None:
    """Mirror one target fill into the paper ledger. Returns trade row or None if skipped."""
    if not paper_enabled():
        return None

    cfg = paper_config()
    meta = _wallet_meta(address)
    if not cfg["copy_all"] and int(meta.get("priority") or 99) != 1:
        return None

    coin = str(fill.get("coin") or "").upper()
    if not coin:
        return None
    try:
        px = float(fill.get("px") or 0)
        sz = float(fill.get("sz") or 0)
    except (TypeError, ValueError):
        return None
    if px <= 0 or sz <= 0:
        return None

    tid = str(fill.get("tid") or fill.get("hash") or "")
    side_raw = str(fill.get("side") or "").upper()
    # HL: B = bid/buy, A = ask/sell
    is_buy = side_raw in ("B", "BUY", "LONG")

    with _lock:
        data = load_paper()
        fills = data.get("fills") or []
        if tid and any(str(x.get("target_tid") or "") == tid for x in fills):
            return None

        target_av = _target_account_value(address)
        equity = float(data.get("equity") or data.get("balance") or cfg["initial_balance"])
        our_sz = _scaled_size(sz, target_av or equity, equity, cfg["leverage_adjustment"])
        notional = our_sz * px
        if notional < cfg["min_notional"]:
            skip = {
                "id": str(uuid.uuid4())[:8],
                "skipped": True,
                "reason": "min_notional",
                "coin": coin,
                "target_address": address,
                "target_sz": sz,
                "px": px,
                "our_sz": our_sz,
                "notional": notional,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            fills.insert(0, skip)
            data["fills"] = fills[:500]
            save_paper(data)
            return skip

        source = str(meta.get("id") or address[:10])
        key = f"{source}:{coin}"
        positions = dict(data.get("positions") or {})
        pos = positions.get(key)
        is_close = _is_close_fill(fill)
        trade: dict[str, Any] = {
            "id": str(uuid.uuid4())[:8],
            "skipped": False,
            "source": source,
            "target_address": address,
            "coin": coin,
            "px": px,
            "target_sz": sz,
            "our_sz": round(our_sz, 8),
            "notional": round(notional, 4),
            "side": "buy" if is_buy else "sell",
            "is_close": is_close,
            "target_tid": tid or None,
            "target_closed_pnl": fill.get("closedPnl"),
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        if is_close and pos:
            # Realize PnL against entry
            entry = float(pos.get("entry_px") or px)
            signed = float(pos.get("sz") or 0)
            close_sz = min(abs(signed), our_sz)
            if signed > 0:
                pnl = (px - entry) * close_sz
            else:
                pnl = (entry - px) * close_sz
            data["balance"] = round(float(data.get("balance") or 0) + pnl, 4)
            data["realized_pnl"] = round(float(data.get("realized_pnl") or 0) + pnl, 4)
            remain = abs(signed) - close_sz
            if remain <= 1e-12:
                positions.pop(key, None)
            else:
                pos["sz"] = remain if signed > 0 else -remain
                positions[key] = pos
            trade["realized_pnl"] = round(pnl, 4)
            trade["action"] = "close"
        else:
            # Open / add
            signed_sz = our_sz if is_buy else -our_sz
            if not pos:
                positions[key] = {
                    "key": key,
                    "source": source,
                    "coin": coin,
                    "sz": signed_sz,
                    "entry_px": px,
                    "opened_at": trade["ts"],
                    "u_pnl": 0.0,
                    "target_address": address,
                }
                trade["action"] = "open"
            else:
                old_sz = float(pos["sz"])
                old_entry = float(pos["entry_px"])
                # same direction → average; flip → close then open (simplified: replace)
                if old_sz * signed_sz > 0:
                    new_sz = old_sz + signed_sz
                    pos["entry_px"] = (abs(old_sz) * old_entry + our_sz * px) / (abs(old_sz) + our_sz)
                    pos["sz"] = new_sz
                    trade["action"] = "add"
                else:
                    # reduce / flip
                    if abs(signed_sz) >= abs(old_sz):
                        # close old
                        if old_sz > 0:
                            pnl = (px - old_entry) * abs(old_sz)
                        else:
                            pnl = (old_entry - px) * abs(old_sz)
                        data["balance"] = round(float(data.get("balance") or 0) + pnl, 4)
                        data["realized_pnl"] = round(float(data.get("realized_pnl") or 0) + pnl, 4)
                        leftover = abs(signed_sz) - abs(old_sz)
                        if leftover > 1e-12:
                            positions[key] = {
                                "key": key,
                                "source": source,
                                "coin": coin,
                                "sz": leftover if signed_sz > 0 else -leftover,
                                "entry_px": px,
                                "opened_at": trade["ts"],
                                "u_pnl": 0.0,
                                "target_address": address,
                            }
                        else:
                            positions.pop(key, None)
                        trade["realized_pnl"] = round(pnl, 4)
                        trade["action"] = "flip"
                    else:
                        if old_sz > 0:
                            pnl = (px - old_entry) * our_sz
                        else:
                            pnl = (old_entry - px) * our_sz
                        data["balance"] = round(float(data.get("balance") or 0) + pnl, 4)
                        data["realized_pnl"] = round(float(data.get("realized_pnl") or 0) + pnl, 4)
                        pos["sz"] = old_sz + signed_sz
                        positions[key] = pos
                        trade["realized_pnl"] = round(pnl, 4)
                        trade["action"] = "reduce"

        data["positions"] = positions
        fills.insert(0, trade)
        data["fills"] = fills[:500]
        save_paper(data)
        logger.info(
            "paper copy %s %s %s sz=%.6f px=%s",
            trade.get("action"),
            source,
            coin,
            our_sz,
            px,
        )
        return trade


def mark_positions_from_mids(mids: dict[str, float]) -> dict[str, Any]:
    """Update u_pnl from mid prices {coin: mid}."""
    with _lock:
        data = load_paper()
        positions = data.get("positions") or {}
        for pos in positions.values():
            coin = str(pos.get("coin") or "")
            mid = mids.get(coin)
            if mid is None:
                # try common aliases
                mid = mids.get(coin.replace("-PERP", ""))
            if mid is None:
                continue
            try:
                mid_f = float(mid)
                entry = float(pos.get("entry_px") or 0)
                sz = float(pos.get("sz") or 0)
            except (TypeError, ValueError):
                continue
            if entry <= 0 or abs(sz) < 1e-16:
                continue
            if sz > 0:
                pos["u_pnl"] = round((mid_f - entry) * sz, 4)
            else:
                pos["u_pnl"] = round((entry - mid_f) * abs(sz), 4)
            pos["mark_px"] = mid_f
        data["positions"] = positions
        save_paper(data)
        return data


def fetch_all_mids() -> dict[str, float]:
    """HL mid prices via info allMids."""
    from utils.hl_short_term import http_json

    raw = http_json({"type": "allMids"})
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        # sometimes wrapped
        payload = raw.get("mids") if "mids" in raw else raw
        if isinstance(payload, dict):
            for k, v in payload.items():
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
    return out


def refresh_marks() -> dict[str, Any]:
    """Refresh paper book: ratio-sync positions (default) then mark to mid."""
    cfg = paper_config()
    if cfg.get("mode") == "ratio":
        return sync_proportional()
    try:
        mids = fetch_all_mids()
    except Exception as exc:
        logger.warning("fetch allMids failed: %s", exc)
        return load_paper()
    if not mids:
        return load_paper()
    return mark_positions_from_mids(mids)


def ingest_user_event(address: str, data: dict) -> list[dict]:
    """Process WS userEvents. ratio mode → resync book; fills mode → apply each fill."""
    cfg = paper_config()
    fills = data.get("fills")
    if not isinstance(fills, list) or not fills:
        return []

    if cfg.get("mode") == "ratio":
        # Log lightweight fill notices, then rebalance to target ratio
        logged: list[dict] = []
        with _lock:
            book = load_paper()
            rows = list(book.get("fills") or [])
            for f in fills:
                if not isinstance(f, dict):
                    continue
                tid = str(f.get("tid") or f.get("hash") or "")
                if tid and any(str(x.get("target_tid") or "") == tid for x in rows):
                    continue
                meta = _wallet_meta(address)
                row = {
                    "id": str(uuid.uuid4())[:8],
                    "skipped": False,
                    "action": "signal",
                    "source": meta.get("id") or address[:10],
                    "target_address": address,
                    "coin": f.get("coin"),
                    "px": f.get("px"),
                    "target_sz": f.get("sz"),
                    "side": "buy" if str(f.get("side") or "").upper() in ("B", "BUY") else "sell",
                    "target_tid": tid or None,
                    "target_closed_pnl": f.get("closedPnl"),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                rows.insert(0, row)
                logged.append(row)
            book["fills"] = rows[:500]
            save_paper(book)
        sync_proportional()
        return logged

    applied: list[dict] = []
    for f in fills:
        if isinstance(f, dict):
            row = apply_target_fill(address, f)
            if row:
                applied.append(row)
    try:
        refresh_marks()
    except Exception:
        pass
    return applied

"""HL paper mirror → Binance USDT-M sub-accounts.

Routes: hl_binance_subaccounts.json (bot_id → BINANCE_SUB_* keys).
Live: HL_BINANCE_LIVE=1, DRY_RUN=0, enabled sub with credentials.
Debounce: HL_BINANCE_DEBOUNCE_MS (default 10000).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.hl_short_term import resolve_data_dir

logger = logging.getLogger(__name__)

LEDGER_NAME = "hl_binance_live.jsonl"

_bg_lock = threading.Lock()
_debounce_lock = threading.Lock()
_debounce_pending: list[dict[str, Any]] = []
_debounce_timer: threading.Timer | None = None
_debounce_gen = 0
_one_way_done: set[str] = set()
_symbol_locks: dict[str, threading.Lock] = {}
_symbol_locks_guard = threading.Lock()


def _env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def live_enabled() -> bool:
    return _env_truthy("HL_BINANCE_LIVE", default=False)


def dry_run() -> bool:
    return _env_truthy("HL_BINANCE_DRY_RUN", default=True)


def scale() -> float:
    try:
        return max(0.0, float(os.getenv("HL_BINANCE_SCALE", "1") or 1))
    except (TypeError, ValueError):
        return 1.0


def max_leverage() -> int:
    """Binance sub-accounts are often capped at 5x; default clamp for live O."""
    try:
        return max(1, int(float(os.getenv("HL_BINANCE_MAX_LEVERAGE", "5") or 5)))
    except (TypeError, ValueError):
        return 5


def margin_buffer() -> float:
    """Use this fraction of available USDT when sizing opens (headroom for fees)."""
    try:
        return min(1.0, max(0.1, float(os.getenv("HL_BINANCE_MARGIN_BUFFER", "0.9") or 0.9)))
    except (TypeError, ValueError):
        return 0.9


def min_notional() -> float:
    try:
        return max(0.0, float(os.getenv("HL_BINANCE_MIN_NOTIONAL", "5") or 5))
    except (TypeError, ValueError):
        return 5.0


def debounce_ms() -> float:
    raw = os.getenv("HL_BINANCE_DEBOUNCE_MS")
    if raw is None or str(raw).strip() == "":
        raw = os.getenv("HL_BITGET_DEBOUNCE_MS", "10000")
    try:
        return max(0.0, float(raw or 10000))
    except (TypeError, ValueError):
        return 10000.0


def allow_coins() -> set[str] | None:
    raw = (os.getenv("HL_BINANCE_ALLOW_COINS") or "").strip()
    if not raw:
        return None
    return {c.strip().upper() for c in raw.split(",") if c.strip()}


def live_ready() -> tuple[bool, str]:
    if not live_enabled():
        return False, "HL_BINANCE_LIVE=0"
    if dry_run():
        return False, "HL_BINANCE_DRY_RUN=1"
    try:
        from quant.engine.exchanges.binance.account import load_creds_from_env
        from utils.hl_binance_subaccounts import (
            enabled_routes,
            max_subaccounts,
            validate_routes,
        )
    except Exception as exc:
        return False, f"binance_sub_import: {exc}"
    routes = enabled_routes()
    if not routes:
        problems = validate_routes()
        if any("max" in p for p in problems):
            return False, problems[0]
        return False, "no enabled binance subaccounts (set enabled or HL_BINANCE_SUB_<ID>_ENABLED=1)"
    problems = validate_routes()
    if problems:
        return False, "; ".join(problems[:3])
    if len(routes) > max_subaccounts():
        return False, f"enabled > max {max_subaccounts()}"
    missing = [r.id for r in routes if not load_creds_from_env(r.env_prefix).ok()]
    if missing:
        return False, f"Railway credentials missing for: {','.join(missing)}"
    return True, "ok"


def status() -> dict[str, Any]:
    from utils.hl_binance_subaccounts import enabled_routes, load_subaccounts_doc, parse_routes

    ready, reason = live_ready()
    doc = load_subaccounts_doc()
    routes = parse_routes(doc)
    try:
        from quant.engine.exchanges.binance.account import load_creds_from_env
    except Exception:
        load_creds_from_env = None  # type: ignore
    rows = []
    for r in routes:
        creds_ok = bool(load_creds_from_env and load_creds_from_env(r.env_prefix).ok())
        rows.append(
            {
                "id": r.id,
                "bot_id": r.bot_id,
                "label": r.label,
                "enabled": r.enabled,
                "coins": sorted(r.coins) if r.coins is not None else None,
                "env_prefix": r.env_prefix,
                "credentials": creds_ok,
                "scale": r.scale,
            }
        )
    return {
        "live_enabled": live_enabled(),
        "dry_run": dry_run(),
        "live_ready": ready,
        "live_ready_reason": reason,
        "scale": scale(),
        "max_leverage": max_leverage(),
        "margin_buffer": margin_buffer(),
        "min_notional": min_notional(),
        "debounce_ms": debounce_ms(),
        "allow_coins": sorted(allow_coins()) if allow_coins() is not None else None,
        "ledger": str(_ledger_path()),
        "subaccounts": {
            "enabled_count": len(enabled_routes()),
            "routes": rows,
            "config_error": doc.get("error"),
        },
    }


def _ledger_path() -> Path:
    return resolve_data_dir() / LEDGER_NAME


def _append_ledger(row: dict[str, Any]) -> None:
    path = _ledger_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.exception("binance ledger write failed")


def _symbol_lock(symbol: str, *, account_id: str = "main") -> threading.Lock:
    key = f"{account_id}|{symbol}"
    with _symbol_locks_guard:
        lk = _symbol_locks.get(key)
        if lk is None:
            lk = threading.Lock()
            _symbol_locks[key] = lk
        return lk


def _ensure_one_way_once(*, account_id: str) -> None:
    if account_id in _one_way_done:
        return
    try:
        from quant.engine.exchanges.binance.account import ensure_one_way_mode

        ensure_one_way_mode()
        _one_way_done.add(account_id)
    except Exception as exc:
        logger.warning("binance ensure one-way failed [%s]: %s", account_id, exc)


def hl_coin_to_binance(
    coin: str, *, route_coins: set[str] | frozenset[str] | None = None
) -> str | None:
    from utils.hl_binance_symbol_map import hl_base_ticker, map_hl_coin_to_binance

    base = hl_base_ticker(coin)
    if not base:
        return None
    allow = allow_coins()
    if route_coins is not None:
        allowed = {str(c).strip().upper() for c in route_coins}
        if base not in allowed and str(coin).upper() not in allowed:
            return None
    elif allow is not None and base not in allow:
        return None
    return map_hl_coin_to_binance(coin)


def _load_bot(bot_id: str) -> dict[str, Any]:
    from utils.hl_paper_copy import load_paper

    book = load_paper()
    bot = (book.get("bots") or {}).get(bot_id) or {}
    if bot:
        return bot
    for b in (book.get("bots") or {}).values():
        if str(b.get("id") or "") == bot_id:
            return b if isinstance(b, dict) else {}
    return {}


def compute_bot_desired(
    bot_id: str,
    *,
    route_coins: set[str] | frozenset[str] | None = None,
    route_scale: float = 1.0,
) -> dict[str, float]:
    from utils.hl_paper_copy import is_live_only_bot

    bot = _load_bot(bot_id)
    sc = scale() * float(route_scale or 1.0)
    if is_live_only_bot(bot):
        return _desired_from_target_book(bot, route_coins=route_coins, route_scale=sc)
    net: dict[str, float] = {}
    for pos in (bot.get("positions") or {}).values():
        coin = str(pos.get("coin") or "")
        sym = hl_coin_to_binance(coin, route_coins=route_coins)
        if not sym:
            continue
        try:
            sz = float(pos.get("sz") or 0) * sc
        except (TypeError, ValueError):
            continue
        if abs(sz) < 1e-16:
            continue
        net[sym] = net.get(sym, 0.0) + sz
    return net


def _desired_from_target_book(
    bot: dict[str, Any],
    *,
    route_coins: set[str] | frozenset[str] | None = None,
    route_scale: float = 1.0,
) -> dict[str, float]:
    """Size by equity ratio (copy exposure). Margin headroom is handled later
    by scaling the *whole* book uniformly — not per-coin leverage shrink.

        our_sz = target_sz × (our_eq / target_equity) × scale
    """
    from utils.hl_paper_copy import target_sizing_equity

    try:
        av = float(target_sizing_equity(bot) or 0)
    except (TypeError, ValueError):
        av = 0.0
    try:
        from quant.engine.exchanges.binance.account import fetch_account_equity

        eq = float(fetch_account_equity().get("equity") or 0)
    except Exception as exc:
        logger.warning("binance equity for live sizing failed: %s", exc)
        eq = 0.0
    if av <= 1e-9 or eq <= 0:
        return {}
    ratio = (eq / av) * float(route_scale or 1.0)
    net: dict[str, float] = {}
    tpos = bot.get("target_positions") if isinstance(bot.get("target_positions"), dict) else {}
    for coin, tp in tpos.items():
        if not isinstance(tp, dict):
            continue
        sym = hl_coin_to_binance(str(coin), route_coins=route_coins)
        if not sym:
            continue
        try:
            sz = float(tp.get("sz") or 0) * ratio
        except (TypeError, ValueError):
            continue
        if abs(sz) < 1e-16:
            continue
        net[sym] = net.get(sym, 0.0) + sz
    return net


def _scale_book_to_margin(
    desired: dict[str, float],
    *,
    bot_id: str,
) -> dict[str, float]:
    """If target book margin exceeds equity×buffer, shrink all sizes by the same factor."""
    if not desired:
        return desired
    try:
        from quant.engine.exchanges.binance.account import fetch_account_equity

        eq = float(fetch_account_equity().get("equity") or 0)
    except Exception as exc:
        logger.warning("binance equity for book margin scale failed: %s", exc)
        return desired
    if eq <= 0:
        return desired
    budget = eq * margin_buffer()
    need = 0.0
    for sym, sz in desired.items():
        try:
            qty = abs(float(sz))
        except (TypeError, ValueError):
            continue
        if qty < 1e-16:
            continue
        mid = _paper_mark(bot_id, sym)
        if mid <= 0:
            continue
        lev = _paper_leverage(bot_id, sym) or max_leverage()
        lev = min(max(1, int(lev)), max_leverage())
        need += qty * mid / float(lev)
    if need <= budget + 1e-9:
        return desired
    factor = budget / need
    logger.warning(
        "HL→Binance book margin scale ×%.4f (need=%.2f USDT budget=%.2f eq=%.2f bot=%s)",
        factor,
        need,
        budget,
        eq,
        bot_id,
    )
    return {sym: float(sz) * factor for sym, sz in desired.items()}


def overlay_live_bots(book: dict[str, Any]) -> dict[str, Any]:
    """Mutate API response: fill Binance live-only seats with wallet/positions.

    Bitget paper→sub seats keep their paper book — do not overlay.
    """
    from utils.hl_paper_copy import is_live_only_bot
    from utils.hl_binance_subaccounts import enabled_routes, routes_for_bot

    bots = book.get("bots") if isinstance(book.get("bots"), dict) else {}
    if not bots:
        return book
    try:
        from quant.engine.exchanges.binance.account import (
            binance_creds,
            fetch_account_equity,
            fetch_position_risk_rows,
            load_creds_from_env,
        )
    except Exception as exc:
        logger.warning("binance overlay import failed: %s", exc)
        return book

    for bot in bots.values():
        if not is_live_only_bot(bot):
            continue
        venue = str(bot.get("venue") or "").strip().lower()
        if venue and venue != "binance":
            continue
        bid = str(bot.get("id") or "")
        routes = routes_for_bot(bid) or [r for r in enabled_routes() if r.bot_id == bid]
        if not routes:
            # Still show seat as live even if route disabled — try first matching env.
            from utils.hl_binance_subaccounts import parse_routes

            routes = [r for r in parse_routes() if r.bot_id == bid][:1]
        if not routes:
            bot["live_error"] = "no_binance_route"
            continue
        route = routes[0]
        creds = load_creds_from_env(route.env_prefix)
        if not creds.ok():
            bot["live_error"] = "credentials_missing"
            bot["equity"] = None
            bot["balance"] = None
            continue
        try:
            with binance_creds(creds):
                eq = fetch_account_equity()
                rows = fetch_position_risk_rows()
            bot["live_error"] = None
            bot["equity"] = eq.get("equity")
            bot["balance"] = eq.get("wallet")
            bot["u_pnl"] = eq.get("upnl")
            bot["live_available"] = eq.get("available")
            bot["paper_balance"] = eq.get("wallet")
            bot["realized_pnl"] = None
            positions: dict[str, Any] = {}
            for row in rows:
                try:
                    amt = float(row.get("positionAmt") or 0)
                except (TypeError, ValueError):
                    continue
                if abs(amt) < 1e-12:
                    continue
                sym = str(row.get("symbol") or "").upper()
                coin = sym[:-4] if sym.endswith("USDT") else sym
                try:
                    entry = float(row.get("entryPrice") or 0)
                except (TypeError, ValueError):
                    entry = 0.0
                try:
                    mark = float(row.get("markPrice") or 0)
                except (TypeError, ValueError):
                    mark = entry
                try:
                    upnl = float(row.get("unRealizedProfit") or 0)
                except (TypeError, ValueError):
                    upnl = 0.0
                try:
                    lev = float(row.get("leverage") or 0) or None
                except (TypeError, ValueError):
                    lev = None
                notional = abs(amt) * mark if mark > 0 else 0.0
                positions[f"{bid}:{coin}"] = {
                    "coin": coin,
                    "sz": amt,
                    "entry_px": entry,
                    "mark_px": mark,
                    "u_pnl": upnl,
                    "leverage": lev,
                    "notional": notional,
                    "source": bid,
                    "venue": "binance",
                    "live": True,
                }
            bot["positions"] = positions
            bot["live_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            logger.warning("binance overlay %s failed: %s", bid, exc)
            bot["live_error"] = str(exc)
    # Recompute desk totals including live equity for UI.
    desk_eq = 0.0
    desk_bal = 0.0
    for bot in bots.values():
        try:
            desk_eq += float(bot.get("equity") or 0)
        except (TypeError, ValueError):
            pass
        try:
            desk_bal += float(bot.get("balance") or 0)
        except (TypeError, ValueError):
            pass
    book["equity"] = round(desk_eq, 4)
    book["balance"] = round(desk_bal, 4)
    book["portfolio_equity"] = round(desk_eq, 4)
    return book


def _paper_leverage(bot_id: str | None, symbol: str) -> int | None:
    if not bot_id:
        return None
    bot = _load_bot(bot_id)
    best: int | None = None
    # Live-only: leverage from cached HL target book.
    tpos = bot.get("target_positions") if isinstance(bot.get("target_positions"), dict) else {}
    for coin, tp in tpos.items():
        if hl_coin_to_binance(str(coin)) != symbol:
            continue
        if not isinstance(tp, dict):
            continue
        try:
            lev = int(float(tp.get("leverage") or 0))
        except (TypeError, ValueError):
            continue
        if lev > 0:
            best = lev if best is None else max(best, lev)
    for pos in (bot.get("positions") or {}).values():
        coin = str(pos.get("coin") or "")
        sym = hl_coin_to_binance(coin)
        if sym != symbol:
            continue
        try:
            lev = int(float(pos.get("leverage") or 0))
        except (TypeError, ValueError):
            continue
        if lev > 0:
            best = lev if best is None else max(best, lev)
    if best is None:
        return None
    return min(best, max_leverage())


def _clamp_qty_to_margin(
    *,
    symbol: str,
    qty: float,
    leverage: int | None,
    reduce_only: bool,
    bot_id: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Shrink open size so notional/lev fits available USDT (subaccount 5x reality)."""
    info: dict[str, Any] = {}
    if reduce_only or qty <= 0:
        return qty, info
    lev = max(1, int(leverage or max_leverage()))
    lev = min(lev, max_leverage())
    mid = _paper_mark(bot_id, symbol)
    if mid <= 0:
        try:
            from binance_fapi import fetch_mark_price

            px = fetch_mark_price(symbol)
            mid = float(px) if px else 0.0
        except Exception:
            mid = 0.0
    if mid <= 0:
        return qty, info
    try:
        from quant.engine.exchanges.binance.account import fetch_usdt_available

        avail = float(fetch_usdt_available())
    except Exception as exc:
        info["margin_check"] = f"avail_failed:{exc}"
        return qty, info
    info["available_usdt"] = round(avail, 4)
    info["leverage_used"] = lev
    budget = avail * margin_buffer() * lev
    info["max_notional"] = round(budget, 4)
    if budget < min_notional():
        info["clamped"] = True
        info["clamp_reason"] = "insufficient_margin"
        return 0.0, info
    max_qty = budget / mid
    if qty > max_qty + 1e-16:
        info["clamped"] = True
        info["size_raw"] = qty
        info["size_clamped"] = max_qty
        logger.warning(
            "HL→Binance clamp %s qty %.6f→%.6f (avail=%.2f USDT lev=%sx mid=%.2f)",
            symbol,
            qty,
            max_qty,
            avail,
            lev,
            mid,
        )
        return max_qty, info
    return qty, info


def _paper_mark(bot_id: str | None, symbol: str) -> float:
    """Best-effort mid for min-notional checks (paper mark → HL mid → Binance ticker)."""
    if bot_id:
        bot = _load_bot(bot_id)
        for pos in (bot.get("positions") or {}).values():
            coin = str(pos.get("coin") or "")
            if hl_coin_to_binance(coin) != symbol:
                continue
            for key in ("mark_px", "entry_px", "px"):
                try:
                    px = float(pos.get(key) or 0)
                except (TypeError, ValueError):
                    px = 0.0
                if px > 0:
                    return px
    try:
        from utils.hl_paper_copy import fetch_all_mids

        base = symbol[:-4] if symbol.endswith("USDT") else symbol
        mids = fetch_all_mids()
        for key in (base, symbol):
            try:
                px = float(mids.get(key) or 0)
            except (TypeError, ValueError):
                px = 0.0
            if px > 0:
                return px
    except Exception:
        pass
    try:
        from binance_fapi import fetch_mark_price

        px = fetch_mark_price(symbol)
        return float(px) if px and float(px) > 0 else 0.0
    except Exception:
        return 0.0


def _route_allows_symbol(
    symbol: str, route_coins: set[str] | frozenset[str] | None
) -> bool:
    if route_coins is None:
        return True
    base = symbol[:-4] if symbol.upper().endswith("USDT") else symbol
    allowed = {str(c).strip().upper() for c in route_coins}
    return base.upper() in allowed or symbol.upper() in allowed


def make_client_oid(
    *, symbol: str, tid: str | None, desired: float, account_id: str = "O", tag: str = "bn"
) -> str:
    seed = f"{tag}|{account_id}|{symbol}|{tid or ''}|{desired:.8f}"
    digest = hashlib.sha1(seed.encode()).hexdigest()[:20]
    return f"hb{digest}"[:36]


def _place_one(
    *,
    symbol: str,
    side: str,
    size: float,
    client_oid: str,
    reduce_only: bool,
    meta: dict[str, Any],
    account_id: str = "O",
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        **meta,
        "ts": now,
        "venue": "binance",
        "account_id": account_id,
        "symbol": symbol,
        "side": side,
        "size": size,
        "reduce_only": reduce_only,
        "client_oid": client_oid,
        "dry_run": dry_run(),
        "live_enabled": live_enabled(),
    }
    if size <= 0:
        payload["status"] = "skipped"
        payload["reason"] = "zero_size"
        return payload

    try:
        from quant.engine.exchanges.binance.account import round_qty

        rounded = round_qty(symbol, size)
    except Exception:
        rounded = float(size)
    if rounded <= 0:
        payload["status"] = "skipped"
        payload["reason"] = "size_rounds_to_zero"
        payload["size_raw"] = size
        _append_ledger(payload)
        return payload
    payload["size"] = rounded

    if dry_run():
        payload["status"] = "dry_run"
        logger.info(
            "HL→Binance [%s] DRY %s %s size=%.6f reduceOnly=%s oid=%s",
            account_id,
            side,
            symbol,
            rounded,
            reduce_only,
            client_oid,
        )
        _append_ledger(payload)
        return payload

    ready, reason = live_ready()
    if not ready:
        payload["status"] = "blocked"
        payload["error"] = reason
        _append_ledger(payload)
        return payload

    _ensure_one_way_once(account_id=account_id)
    with _symbol_lock(symbol, account_id=account_id):
        try:
            from quant.engine.exchanges.binance.account import (
                fetch_signed_position,
                place_market_order,
                round_qty,
            )

            qty = float(rounded)
            lev_raw = meta.get("leverage")
            try:
                lev_i = int(float(lev_raw)) if lev_raw is not None else max_leverage()
            except (TypeError, ValueError):
                lev_i = max_leverage()
            lev_i = min(max(1, lev_i), max_leverage())
            payload["leverage"] = lev_i

            if reduce_only:
                have = float(fetch_signed_position(symbol))
                if abs(have) < 1e-12:
                    payload["status"] = "skipped"
                    payload["reason"] = "no_position_to_reduce"
                    _append_ledger(payload)
                    return payload
                if side.lower() == "sell" and have > 0:
                    qty = min(qty, abs(have))
                elif side.lower() == "buy" and have < 0:
                    qty = min(qty, abs(have))
                else:
                    payload["status"] = "skipped"
                    payload["reason"] = "reduce_side_mismatch"
                    _append_ledger(payload)
                    return payload
                try:
                    qty = round_qty(symbol, qty)
                except Exception:
                    pass
                if qty <= 0:
                    payload["status"] = "skipped"
                    payload["reason"] = "size_rounds_to_zero"
                    _append_ledger(payload)
                    return payload
            else:
                qty, clamp_info = _clamp_qty_to_margin(
                    symbol=symbol,
                    qty=qty,
                    leverage=lev_i,
                    reduce_only=False,
                    bot_id=str(meta.get("bot_id") or "") or None,
                )
                payload.update({k: v for k, v in clamp_info.items() if k != "size_clamped"})
                try:
                    qty = round_qty(symbol, qty)
                except Exception:
                    pass
                if qty <= 0:
                    payload["status"] = "skipped"
                    payload["reason"] = clamp_info.get("clamp_reason") or "insufficient_margin"
                    payload["size"] = 0
                    _append_ledger(payload)
                    return payload

            result = place_market_order(
                symbol=symbol,
                side=side,
                size=qty,
                client_oid=client_oid,
                reduce_only=reduce_only,
                leverage=lev_i if not reduce_only else None,
            )
            payload["size"] = qty
            payload["status"] = "deduped" if result.get("deduped") else "sent"
            payload["exchange"] = result
        except Exception as exc:
            logger.exception("HL→Binance place failed %s [%s]", symbol, account_id)
            payload["status"] = "error"
            payload["error"] = str(exc)
    _append_ledger(payload)
    return payload


def sync_symbol(
    symbol: str,
    desired: float,
    *,
    account_id: str,
    trigger_tid: str | None = None,
    bot_id: str | None = None,
    have_override: float | None = None,
) -> list[dict[str, Any]]:
    from quant.engine.exchanges.binance.account import fetch_signed_position

    eps = 1e-12
    desired = float(desired)
    have = 0.0
    if have_override is not None:
        have = float(have_override)
    else:
        try:
            have = float(fetch_signed_position(symbol))
        except Exception as exc:
            if not dry_run():
                logger.warning("binance sync fetch pos %s [%s]: %s", symbol, account_id, exc)
                return [
                    {
                        "status": "error",
                        "venue": "binance",
                        "account_id": account_id,
                        "symbol": symbol,
                        "error": f"fetch_position: {exc}",
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                ]
            have = 0.0

    delta = desired - have
    lev = _paper_leverage(bot_id, symbol)
    meta = {
        "action": "bn_sync",
        "venue": "binance",
        "desired": desired,
        "have": have,
        "delta": delta,
        "trigger_tid": trigger_tid,
        "bot_id": bot_id,
        "leverage": lev,
    }
    if abs(delta) < eps:
        return [
            {
                **meta,
                "status": "synced",
                "symbol": symbol,
                "account_id": account_id,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        ]

    results: list[dict[str, Any]] = []
    if abs(have) > eps and (desired == 0 or have * desired < 0):
        close_side = "sell" if have > 0 else "buy"
        oid_c = make_client_oid(
            symbol=symbol, tid=trigger_tid, desired=0.0, account_id=account_id, tag="bnc"
        )
        results.append(
            _place_one(
                symbol=symbol,
                side=close_side,
                size=abs(have),
                client_oid=oid_c,
                reduce_only=True,
                meta={**meta, "leg": "flatten"},
                account_id=account_id,
            )
        )
        have = 0.0
        delta = desired - have
        if abs(desired) < eps or abs(delta) < eps:
            return results
        time.sleep(0.05)

    # Skip dust opens / tiny adjusts (flatten already handled above)
    mid = _paper_mark(bot_id, symbol)
    notion = abs(delta) * mid if mid > 0 else 0.0
    if notion > 0 and notion < min_notional() and abs(desired) > eps:
        return results + [
            {
                **meta,
                "status": "skipped",
                "reason": "below_min_notional",
                "notional": notion,
                "symbol": symbol,
                "account_id": account_id,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        ]

    side = "buy" if delta > 0 else "sell"
    reduce_only = abs(have) > eps and abs(desired) < abs(have) - eps and have * desired >= 0
    oid = make_client_oid(
        symbol=symbol, tid=trigger_tid, desired=desired, account_id=account_id
    )
    results.append(
        _place_one(
            symbol=symbol,
            side=side,
            size=abs(delta),
            client_oid=oid,
            reduce_only=reduce_only,
            meta={**meta, "leg": "align"},
            account_id=account_id,
        )
    )
    return results


def sync_from_paper(rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if not live_enabled():
        return []

    from contextlib import nullcontext

    from quant.engine.exchanges.binance.account import (
        binance_creds,
        fetch_all_signed_positions,
        load_creds_from_env,
    )
    from utils.hl_binance_subaccounts import enabled_routes, routes_for_bot

    touched_bots: set[str] = set()
    touched_coins: set[str] = set()
    trigger_tid: str | None = None
    for row in rows or []:
        bid = str(row.get("source") or row.get("bot_id") or "")
        if bid:
            touched_bots.add(bid)
        coin = str(row.get("coin") or "").strip()
        if coin:
            touched_coins.add(coin)
        tids = row.get("target_tids") or []
        if not trigger_tid and isinstance(tids, list) and tids:
            trigger_tid = str(tids[0])
        elif not trigger_tid and row.get("target_tid"):
            trigger_tid = str(row.get("target_tid"))

    routes = enabled_routes()
    if not routes:
        logger.warning("HL→Binance: no enabled subaccounts")
        return [{"status": "blocked", "error": "no enabled binance subaccounts"}]

    if touched_bots:
        routes = [r for r in routes if r.bot_id in touched_bots]
        if not routes:
            for bid in touched_bots:
                routes.extend(routes_for_bot(bid))
            seen: set[str] = set()
            uniq = []
            for r in routes:
                if r.id in seen:
                    continue
                seen.add(r.id)
                uniq.append(r)
            routes = uniq

    out: list[dict[str, Any]] = []
    for route in routes:
        creds = load_creds_from_env(route.env_prefix)
        if not creds.ok() and not dry_run():
            out.append(
                {
                    "status": "blocked",
                    "account_id": route.id,
                    "bot_id": route.bot_id,
                    "error": "credentials_missing",
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )
            continue

        desired = compute_bot_desired(
            route.bot_id,
            route_coins=route.coins,
            route_scale=route.scale,
        )
        symbols = set(desired.keys())
        for coin in touched_coins:
            sym = hl_coin_to_binance(coin, route_coins=route.coins)
            if sym:
                symbols.add(sym)

        # Never fall back to main BINANCE_* when sub keys are missing.
        ctx = binance_creds(creds) if creds.ok() else nullcontext()
        with ctx:
            # Scale whole book under creds context (equity/marks need sub keys).
            desired = _scale_book_to_margin(desired, bot_id=route.bot_id)

            open_pos: dict[str, float] = {}
            scanned_open = False
            if creds.ok():
                try:
                    open_pos = fetch_all_signed_positions()
                    scanned_open = True
                    for sym, sz in open_pos.items():
                        if abs(sz) < 1e-12:
                            continue
                        if not _route_allows_symbol(sym, route.coins):
                            continue
                        symbols.add(sym)
                except Exception as exc:
                    logger.warning("binance open-pos scan [%s]: %s", route.id, exc)

            for sym in sorted(symbols):
                want = float(desired.get(sym) or 0.0)
                # After a successful book scan, missing symbol ⇒ flat (avoid N+1 risk calls).
                have_ov = float(open_pos.get(sym) or 0.0) if scanned_open else None
                out.extend(
                    sync_symbol(
                        sym,
                        want,
                        account_id=route.id,
                        trigger_tid=trigger_tid,
                        bot_id=route.bot_id,
                        have_override=have_ov,
                    )
                )
                time.sleep(0.05)
    return out


def maybe_execute_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows or not live_enabled():
        return []
    try:
        from utils.hl_binance_subaccounts import enabled_routes

        bots = {r.bot_id for r in enabled_routes()}
    except Exception:
        bots = set()
    if not bots:
        return []
    if not any(str(r.get("source") or r.get("bot_id") or "") in bots for r in rows):
        return []
    try:
        return sync_from_paper(rows)
    except Exception:
        logger.exception("HL Binance executor failed")
        return []


def _flush_debounced(gen: int) -> None:
    global _debounce_pending
    with _debounce_lock:
        if gen != _debounce_gen:
            return
        batch = list(_debounce_pending)
        _debounce_pending = []
    if not batch:
        return
    logger.info(
        "HL→Binance debounce flush n_rows=%s bots=%s",
        len(batch),
        sorted({str(r.get("source") or "") for r in batch if r}),
    )
    with _bg_lock:
        maybe_execute_rows(batch)


def maybe_execute_rows_async(
    rows: list[dict[str, Any]], *, immediate: bool = False
) -> None:
    if not rows or not live_enabled():
        return
    try:
        from utils.hl_binance_subaccounts import enabled_routes

        bots = {r.bot_id for r in enabled_routes()}
    except Exception:
        bots = set()
    if not bots:
        return
    if not any(str(r.get("source") or r.get("bot_id") or "") in bots for r in rows):
        return

    ms = 0.0 if immediate else debounce_ms()
    if ms <= 0:

        def _run() -> None:
            with _bg_lock:
                maybe_execute_rows(rows)

        threading.Thread(target=_run, name="hl-binance-exec", daemon=True).start()
        return

    global _debounce_timer, _debounce_gen
    with _debounce_lock:
        _debounce_pending.extend(rows)
        _debounce_gen += 1
        gen = _debounce_gen
        if _debounce_timer is not None:
            try:
                _debounce_timer.cancel()
            except Exception:
                pass
        t = threading.Timer(ms / 1000.0, _flush_debounced, args=(gen,))
        t.daemon = True
        _debounce_timer = t
        t.start()

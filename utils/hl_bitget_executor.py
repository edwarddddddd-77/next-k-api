"""HL paper mirror → Bitget USDT-M (vnpy REST).

MODE=sub (default): each paper bot maps to a Bitget sub-account
(hl_bitget_subaccounts.json). Positions never net across sub-accounts.

MODE=net: legacy single-account sum of bots.
MODE=delta: per-row intents (single bot / single account only).

Default dry-run. Live: HL_BITGET_LIVE=1, DRY_RUN=0, plus enabled subaccounts
with credentials (sub mode) or ALLOW_COINS+BOT_IDS (net/delta).

Burst fills: HL_BITGET_DEBOUNCE_MS (default 1000) coalesces paper→Bitget
syncs so one HL fill storm becomes one position align.
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

LEDGER_NAME = "hl_bitget_live.jsonl"

_symbol_locks: dict[str, threading.Lock] = {}
_symbol_locks_guard = threading.Lock()
_mode_ready_accounts: set[str] = set()
_mode_lock = threading.Lock()
_bg_lock = threading.Lock()

# Coalesce rapid paper fills into one Bitget position sync.
_debounce_lock = threading.Lock()
_debounce_timer: threading.Timer | None = None
_debounce_pending: list[dict[str, Any]] = []
_debounce_gen = 0


def _env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def live_enabled() -> bool:
    return _env_truthy("HL_BITGET_LIVE", default=False)


def dry_run() -> bool:
    return _env_truthy("HL_BITGET_DRY_RUN", default=True)


def scale() -> float:
    try:
        return max(0.0, float(os.getenv("HL_BITGET_SCALE", "1") or 1))
    except (TypeError, ValueError):
        return 1.0


def max_notional() -> float:
    try:
        return max(0.0, float(os.getenv("HL_BITGET_MAX_NOTIONAL", "0") or 0))
    except (TypeError, ValueError):
        return 0.0


def min_notional() -> float:
    try:
        return max(0.0, float(os.getenv("HL_BITGET_MIN_NOTIONAL", "5") or 5))
    except (TypeError, ValueError):
        return 5.0


def debounce_ms() -> float:
    """Wait this many ms after the last paper fill before Bitget sync (0 = off)."""
    try:
        return max(0.0, float(os.getenv("HL_BITGET_DEBOUNCE_MS", "1000") or 1000))
    except (TypeError, ValueError):
        return 1000.0


def allow_coins() -> set[str] | None:
    """Global allowlist (net/delta). Sub mode prefers per-route coins."""
    raw = (os.getenv("HL_BITGET_ALLOW_COINS") or "").strip()
    if not raw:
        return None
    return {c.strip().upper() for c in raw.split(",") if c.strip()}


def allow_bot_ids() -> set[str] | None:
    raw = (os.getenv("HL_BITGET_BOT_IDS") or "").strip()
    if not raw:
        return None
    return {c.strip() for c in raw.split(",") if c.strip()}


def skip_prefixes() -> tuple[str, ...]:
    """Prefixes to drop before mapping. Default empty — xyz: stocks are mapped.
    Example skip: HL_BITGET_SKIP_PREFIX=flx:,vntl:
    """
    raw = (os.getenv("HL_BITGET_SKIP_PREFIX") or "").strip()
    if not raw:
        return ()
    return tuple(p.strip().lower() for p in raw.split(",") if p.strip())


def log_skips() -> bool:
    return _env_truthy("HL_BITGET_LOG_SKIPS", default=False)


def exec_mode() -> str:
    """sub | net | delta. Default sub (isolated Bitget sub-accounts)."""
    raw = (os.getenv("HL_BITGET_MODE") or "sub").strip().lower()
    if raw in ("delta", "per_bot", "row"):
        return "delta"
    if raw in ("net", "sum", "legacy"):
        return "net"
    return "sub"


def status() -> dict[str, Any]:
    allow = allow_coins()
    bots = allow_bot_ids()
    ready, ready_reason = live_ready()
    sub_summary = _subaccount_status()
    return {
        "live_enabled": live_enabled(),
        "dry_run": dry_run(),
        "mode": exec_mode(),
        "live_ready": ready,
        "live_ready_reason": ready_reason,
        "scale": scale(),
        "max_notional": max_notional() or None,
        "min_notional": min_notional(),
        "debounce_ms": debounce_ms(),
        "allow_coins": sorted(allow) if allow is not None else None,
        "allow_bot_ids": sorted(bots) if bots is not None else None,
        "skip_prefixes": list(skip_prefixes()),
        "credentials": _credentials_ok(),
        "ledger": str(_ledger_path()),
        "subaccounts": sub_summary,
        "symbol_map": _symbol_map_status(),
    }


def _symbol_map_status() -> dict[str, Any]:
    try:
        from utils.hl_bitget_symbol_map import (
            bitget_contract_set,
            coin_aliases,
            verify_symbols_enabled,
        )

        known = bitget_contract_set()
        return {
            "verify_symbols": verify_symbols_enabled(),
            "aliases": coin_aliases(),
            "bitget_contracts": len(known) if known is not None else None,
            "unlisted_policy": "auto_skip",
            "samples": {
                "BTC": "BTCUSDT",
                "xyz:TSLA": "TSLAUSDT",
                "xyz:SILVER": "XAGUSDT",
                "xyz:GOOG": "GOOGLUSDT",
            },
        }
    except Exception as exc:
        return {"error": str(exc)}


def _subaccount_status() -> dict[str, Any]:
    try:
        from utils.hl_bitget_subaccounts import (
            load_subaccounts_doc,
            max_subaccounts,
            parse_routes,
            validate_routes,
        )
        from quant.engine.exchanges.bitget.account import load_creds_from_env
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    doc = load_subaccounts_doc()
    routes = parse_routes(doc)
    problems = validate_routes(routes)
    rows = []
    for r in routes:
        creds = load_creds_from_env(r.env_prefix)
        rows.append(
            {
                "id": r.id,
                "label": r.label,
                "bot_id": r.bot_id,
                "coins": sorted(r.coins) if r.coins is not None else None,
                "enabled": r.enabled,
                "env_prefix": r.env_prefix or "(missing — set in JSON)",
                "scale": r.scale,
                "credentials_ok": creds.ok(),
                "railway_keys": [
                    f"{r.env_prefix}_API_KEY",
                    f"{r.env_prefix}_API_SECRET",
                    f"{r.env_prefix}_PASSPHRASE",
                ]
                if r.env_prefix
                else [],
            }
        )
    return {
        "ok": not problems,
        "max_subaccounts": max_subaccounts(),
        "config_error": doc.get("error"),
        "problems": problems,
        "routes": rows,
        "enabled_count": sum(1 for r in routes if r.enabled),
    }


def live_ready() -> tuple[bool, str]:
    """Whether real (non-dry) orders are allowed."""
    if not live_enabled():
        return False, "HL_BITGET_LIVE=0"
    if dry_run():
        return False, "HL_BITGET_DRY_RUN=1"

    mode = exec_mode()
    if mode == "sub":
        try:
            from utils.hl_bitget_subaccounts import (
                enabled_routes,
                max_subaccounts,
                validate_routes,
            )
            from quant.engine.exchanges.bitget.account import load_creds_from_env
        except Exception as exc:
            return False, f"subaccounts_import: {exc}"
        routes = enabled_routes()
        if not routes:
            problems = validate_routes()
            if any("max" in p for p in problems):
                return False, problems[0]
            return False, "no enabled subaccounts (set enabled or HL_BITGET_SUB_<ID>_ENABLED=1)"
        problems = validate_routes()
        if problems:
            return False, "; ".join(problems[:3])
        if len(routes) > max_subaccounts():
            return False, f"enabled > max {max_subaccounts()}"
        missing = [r.id for r in routes if not load_creds_from_env(r.env_prefix).ok()]
        if missing:
            return False, f"Railway credentials missing for: {','.join(missing)}"
        return True, "ok"

    if not _credentials_ok():
        return False, "bitget_credentials_missing"
    if not allow_coins():
        return False, "HL_BITGET_ALLOW_COINS required for live"
    if not allow_bot_ids():
        return False, "HL_BITGET_BOT_IDS required (bots included in net book)"
    if mode == "delta" and len(allow_bot_ids() or []) > 1:
        return False, "HL_BITGET_MODE=delta only safe with one BOT_ID (use sub or net)"
    return True, "ok"


def _credentials_ok() -> bool:
    try:
        from quant.engine.exchanges.bitget.gateway import bitget_credentials_configured

        return bool(bitget_credentials_configured())
    except Exception:
        key = (os.getenv("BITGET_API_KEY") or "").strip()
        sec = (os.getenv("BITGET_API_SECRET") or "").strip()
        pwd = (os.getenv("BITGET_PASSPHRASE") or os.getenv("BITGET_API_PASSPHRASE") or "").strip()
        return bool(key and sec and pwd)


def _ledger_path() -> Path:
    return resolve_data_dir() / LEDGER_NAME


def _append_ledger(row: dict[str, Any]) -> None:
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _symbol_lock(sym: str, *, account_id: str = "main") -> threading.Lock:
    key = f"{account_id}:{sym}"
    with _symbol_locks_guard:
        lk = _symbol_locks.get(key)
        if lk is None:
            lk = threading.Lock()
            _symbol_locks[key] = lk
        return lk


def hl_coin_to_bitget(
    coin: str,
    *,
    route_coins: set[str] | frozenset[str] | None = None,
) -> str | None:
    """Map HL coin (crypto / xyz: stocks) to Bitget USDT-M; None = auto-skip."""
    from utils.hl_bitget_symbol_map import hl_base_ticker, resolve_bitget_symbol

    raw = str(coin or "").strip()
    if not raw:
        return None
    low = raw.lower()
    for pref in skip_prefixes():
        if low.startswith(pref):
            return None
    if raw.startswith("@"):
        return None

    base = hl_base_ticker(raw)
    if not base:
        return None

    def _in_allow(allow: set[str] | frozenset[str]) -> bool:
        normalized = {hl_base_ticker(c) or str(c).strip().upper() for c in allow}
        return base in normalized

    if route_coins is not None:
        if not _in_allow(route_coins):
            return None
    else:
        allow = allow_coins()
        if allow is not None and not _in_allow(allow):
            return None

    sym, _reason = resolve_bitget_symbol(raw)
    return sym


def hl_coin_map_detail(coin: str) -> dict[str, Any]:
    """For ledger / debugging: full map result including skip_reason."""
    from utils.hl_bitget_symbol_map import describe_mapping, hl_base_ticker

    raw = str(coin or "").strip()
    detail = describe_mapping(raw)
    if not raw:
        return detail
    low = raw.lower()
    for pref in skip_prefixes():
        if low.startswith(pref):
            return {**detail, "bitget": None, "ok": False, "skip_reason": "skip_prefix"}
    allow = allow_coins()
    base = hl_base_ticker(raw)
    if allow is not None and base and base not in {
        hl_base_ticker(c) or str(c).strip().upper() for c in allow
    }:
        return {**detail, "bitget": None, "ok": False, "skip_reason": "not_in_allow_coins"}
    return detail


def make_client_oid(
    *,
    bot_id: str,
    action: str,
    coin: str,
    tid: str | None,
    fp: str | None,
) -> str:
    """Stable idempotency key — do NOT include qty (ratio drift would re-fire)."""
    seed = "|".join(
        [
            str(bot_id or ""),
            str(action or ""),
            str(coin or "").upper(),
            str(tid or ""),
            str(fp or ""),
        ]
    )
    digest = hashlib.sha1(seed.encode()).hexdigest()[:20]
    return f"hl{digest}"


def _ensure_one_way_once(*, account_id: str = "main") -> None:
    with _mode_lock:
        if account_id in _mode_ready_accounts:
            return
        try:
            from quant.engine.exchanges.bitget.account import ensure_one_way_mode

            ensure_one_way_mode()
            _mode_ready_accounts.add(account_id)
        except Exception as exc:
            logger.warning("bitget ensure one-way failed [%s]: %s", account_id, exc)


def _close_side_from_row(row: dict[str, Any]) -> str | None:
    side = str(row.get("side") or "").lower()
    if side in ("buy", "sell"):
        return side
    return None


def _clamp_reduce_size(sym: str, side: str, qty: float) -> float:
    try:
        from quant.engine.exchanges.bitget.account import fetch_signed_position

        pos = fetch_signed_position(sym)
    except Exception as exc:
        logger.warning("clamp reduce: position fetch failed %s: %s", sym, exc)
        return qty
    if abs(pos) < 1e-12:
        return 0.0
    if side == "sell" and pos > 0:
        return min(qty, abs(pos))
    if side == "buy" and pos < 0:
        return min(qty, abs(pos))
    return 0.0


def row_to_intent(row: dict[str, Any]) -> dict[str, Any] | None:
    action = str(row.get("action") or "").replace("sync_", "")
    if action not in ("open", "increase", "reduce", "close"):
        return None

    bot_id = str(row.get("source") or "")
    bots = allow_bot_ids()
    if bots is not None and bot_id not in bots:
        return {
            "skip": True,
            "reason": "bot_not_allowed",
            "bot_id": bot_id,
            "action": action,
            "coin": row.get("coin"),
        }

    coin = str(row.get("coin") or "")
    detail = hl_coin_map_detail(coin)
    sym = detail.get("bitget")
    if not sym:
        return {
            "skip": True,
            "reason": str(detail.get("skip_reason") or "unmapped_or_filtered"),
            "coin": coin,
            "action": action,
            "bot_id": bot_id,
        }

    try:
        qty = abs(float(row.get("our_sz") or 0)) * scale()
    except (TypeError, ValueError):
        return {"skip": True, "reason": "bad_qty", "coin": coin, "symbol": sym}
    try:
        px = float(row.get("px") or 0)
    except (TypeError, ValueError):
        px = 0.0
    try:
        notion = qty * px if px > 0 else abs(float(row.get("notional") or 0)) * scale()
    except (TypeError, ValueError):
        notion = 0.0

    if qty <= 0:
        return {"skip": True, "reason": "zero_qty", "coin": coin, "symbol": sym}
    if notion > 0 and notion < min_notional():
        return {
            "skip": True,
            "reason": "below_min_notional",
            "coin": coin,
            "symbol": sym,
            "notional": notion,
        }
    cap = max_notional()
    if cap > 0 and notion > cap:
        if px > 0:
            qty = cap / px
            notion = cap
        else:
            return {"skip": True, "reason": "above_max_notional", "coin": coin, "symbol": sym}

    reduce_only = action in ("reduce", "close")
    if reduce_only:
        side = _close_side_from_row(row)
        if not side:
            return {"skip": True, "reason": "missing_close_side", "coin": coin, "symbol": sym}
    else:
        side = str(row.get("side") or "").lower()
        if side not in ("buy", "sell"):
            return {"skip": True, "reason": "missing_side", "coin": coin, "symbol": sym}

    tid = None
    tids = row.get("target_tids") or []
    if isinstance(tids, list) and tids:
        tid = str(tids[0])
    elif row.get("target_tid"):
        tid = str(row.get("target_tid"))
    fp = str(row.get("target_fp") or "") or None
    if not tid and not fp:
        fp = str(row.get("id") or "") or None

    oid = make_client_oid(
        bot_id=bot_id,
        action=action,
        coin=coin,
        tid=tid,
        fp=fp,
    )
    lev = row.get("leverage")
    try:
        lev_i = int(float(lev)) if lev is not None else None
    except (TypeError, ValueError):
        lev_i = None

    return {
        "skip": False,
        "action": action,
        "coin": coin,
        "symbol": sym,
        "side": side,
        "size": qty,
        "notional": notion,
        "reduce_only": reduce_only,
        "client_oid": oid,
        "leverage": lev_i,
        "bot_id": bot_id,
        "target_tid": tid,
        "paper_row_id": row.get("id"),
    }


def execute_intent(intent: dict[str, Any], *, account_id: str = "main") -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    if intent.get("skip"):
        out = {**intent, "ts": now, "status": "skipped", "account_id": account_id}
        if log_skips():
            _append_ledger(out)
        return out

    sym = str(intent["symbol"])
    payload = {
        **intent,
        "ts": now,
        "dry_run": dry_run(),
        "live_enabled": live_enabled(),
        "account_id": account_id,
    }

    if not live_enabled():
        payload["status"] = "disabled"
        return payload

    if dry_run():
        payload["status"] = "dry_run"
        logger.info(
            "HL→Bitget DRY [%s] %s %s %s size=%.6f oid=%s bot=%s",
            account_id,
            intent.get("action"),
            intent.get("side"),
            sym,
            float(intent.get("size") or 0),
            intent.get("client_oid"),
            intent.get("bot_id"),
        )
        _append_ledger(payload)
        return payload

    ready, reason = live_ready()
    if not ready:
        payload["status"] = "blocked"
        payload["error"] = reason
        _append_ledger(payload)
        logger.warning("HL→Bitget blocked: %s", reason)
        return payload

    _ensure_one_way_once(account_id=account_id)
    lk = _symbol_lock(sym, account_id=account_id)
    with lk:
        try:
            from quant.engine.exchanges.bitget.account import place_market_order

            size = float(intent["size"])
            side = str(intent["side"])
            if intent.get("reduce_only"):
                size = _clamp_reduce_size(sym, side, size)
                if size <= 0:
                    payload["status"] = "skipped"
                    payload["reason"] = "no_position_to_reduce"
                    _append_ledger(payload)
                    return payload
                payload["size"] = size

            result = place_market_order(
                symbol=sym,
                side=side,
                size=size,
                client_oid=str(intent["client_oid"]),
                reduce_only=bool(intent.get("reduce_only")),
                leverage=intent.get("leverage"),
            )
            payload["status"] = "deduped" if result.get("deduped") else "sent"
            payload["exchange"] = result
        except Exception as exc:
            logger.exception("HL→Bitget place failed %s [%s]", sym, account_id)
            payload["status"] = "error"
            payload["error"] = str(exc)
    _append_ledger(payload)
    return payload


def apply_mirror_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Execute after paper mirror."""
    if not rows:
        return []
    if not live_enabled():
        return []

    mode = exec_mode()
    if mode == "sub":
        return sync_subaccounts_from_paper(rows)
    if mode == "net":
        return sync_net_from_paper(rows)

    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("skipped"):
            continue
        intent = row_to_intent(row)
        if not intent:
            continue
        result = execute_intent(intent)
        out.append(result)
        if result.get("status") == "sent":
            time.sleep(0.05)
    return out


def compute_bot_desired(
    bot_id: str,
    *,
    route_coins: frozenset[str] | set[str] | None = None,
    route_scale: float = 1.0,
) -> dict[str, float]:
    """Paper positions for one bot → Bitget symbol signed sizes."""
    from utils.hl_paper_copy import load_paper

    book = load_paper()
    bot = (book.get("bots") or {}).get(bot_id) or {}
    # also allow lookup by id field if keyed differently
    if not bot:
        for b in (book.get("bots") or {}).values():
            if str(b.get("id") or "") == bot_id:
                bot = b
                break
    net: dict[str, float] = {}
    sc = scale() * float(route_scale or 1.0)
    for pos in (bot.get("positions") or {}).values():
        coin = str(pos.get("coin") or "")
        sym = hl_coin_to_bitget(coin, route_coins=route_coins)
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


def compute_net_desired() -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Sum paper bot positions → Bitget symbol net size (signed). Legacy net mode."""
    from utils.hl_paper_copy import load_paper

    book = load_paper()
    bots_filter = allow_bot_ids()
    net: dict[str, float] = {}
    parts: dict[str, dict[str, float]] = {}
    for bot in (book.get("bots") or {}).values():
        bid = str(bot.get("id") or "")
        if bots_filter is not None and bid not in bots_filter:
            continue
        for pos in (bot.get("positions") or {}).values():
            coin = str(pos.get("coin") or "")
            sym = hl_coin_to_bitget(coin)
            if not sym:
                continue
            try:
                sz = float(pos.get("sz") or 0) * scale()
            except (TypeError, ValueError):
                continue
            if abs(sz) < 1e-16:
                continue
            net[sym] = net.get(sym, 0.0) + sz
            parts.setdefault(sym, {})[bid] = parts.get(sym, {}).get(bid, 0.0) + sz
    return net, parts


def make_net_client_oid(*, symbol: str, tid: str | None, desired: float, account_id: str = "main") -> str:
    seed = f"sub|{account_id}|{symbol}|{tid or ''}|{desired:.8f}"
    digest = hashlib.sha1(seed.encode()).hexdigest()[:20]
    return f"hs{digest}"


def _place_one(
    *,
    symbol: str,
    side: str,
    size: float,
    client_oid: str,
    reduce_only: bool,
    meta: dict[str, Any],
    account_id: str = "main",
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        **meta,
        "ts": now,
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

    if dry_run():
        payload["status"] = "dry_run"
        logger.info(
            "HL→Bitget [%s] DRY %s %s size=%.6f reduceOnly=%s oid=%s desired=%s have=%s",
            account_id,
            side,
            symbol,
            size,
            reduce_only,
            client_oid,
            meta.get("desired"),
            meta.get("have"),
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
    lk = _symbol_lock(symbol, account_id=account_id)
    with lk:
        try:
            from quant.engine.exchanges.bitget.account import place_market_order

            qty = size
            if reduce_only:
                qty = _clamp_reduce_size(symbol, side, qty)
                if qty <= 0:
                    payload["status"] = "skipped"
                    payload["reason"] = "no_position_to_reduce"
                    _append_ledger(payload)
                    return payload
                payload["size"] = qty
            result = place_market_order(
                symbol=symbol,
                side=side,
                size=qty,
                client_oid=client_oid,
                reduce_only=reduce_only,
            )
            payload["status"] = "deduped" if result.get("deduped") else "sent"
            payload["exchange"] = result
        except Exception as exc:
            logger.exception("HL→Bitget place failed %s [%s]", symbol, account_id)
            payload["status"] = "error"
            payload["error"] = str(exc)
    _append_ledger(payload)
    return payload


def sync_account_symbol(
    symbol: str,
    desired: float,
    *,
    account_id: str = "main",
    parts: dict[str, float] | None = None,
    trigger_tid: str | None = None,
    mode_tag: str = "sync",
) -> list[dict[str, Any]]:
    """Move one Bitget account's position to desired (signed)."""
    from quant.engine.exchanges.bitget.account import fetch_signed_position

    eps = 1e-12
    desired = float(desired)
    have = 0.0
    try:
        have = float(fetch_signed_position(symbol))
    except Exception as exc:
        if not dry_run():
            logger.warning("sync fetch pos %s [%s]: %s", symbol, account_id, exc)
            return [
                {
                    "status": "error",
                    "symbol": symbol,
                    "account_id": account_id,
                    "error": f"fetch_position: {exc}",
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            ]
        have = 0.0

    delta = desired - have
    meta = {
        "action": mode_tag,
        "desired": desired,
        "have": have,
        "delta": delta,
        "parts": parts or {},
        "trigger_tid": trigger_tid,
        "mode": exec_mode(),
    }

    if abs(delta) < eps:
        out = {
            **meta,
            "status": "synced",
            "symbol": symbol,
            "account_id": account_id,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if log_skips():
            _append_ledger(out)
        return [out]

    results: list[dict[str, Any]] = []

    if abs(have) > eps and (desired == 0 or have * desired < 0):
        close_side = "sell" if have > 0 else "buy"
        oid_c = make_net_client_oid(
            symbol=symbol, tid=trigger_tid, desired=0.0, account_id=account_id
        ) + "c"
        oid_c = oid_c[:32]
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

    side = "buy" if delta > 0 else "sell"
    reduce_only = abs(have) > eps and abs(desired) < abs(have) - eps and have * desired >= 0
    oid = make_net_client_oid(
        symbol=symbol, tid=trigger_tid, desired=desired, account_id=account_id
    )
    results.append(
        _place_one(
            symbol=symbol,
            side=side,
            size=abs(delta),
            client_oid=oid,
            reduce_only=reduce_only,
            meta={**meta, "leg": "adjust"},
            account_id=account_id,
        )
    )
    return results


# Back-compat alias
def sync_net_symbol(
    symbol: str,
    desired: float,
    *,
    parts: dict[str, float] | None = None,
    trigger_tid: str | None = None,
) -> list[dict[str, Any]]:
    return sync_account_symbol(
        symbol,
        desired,
        account_id="main",
        parts=parts,
        trigger_tid=trigger_tid,
        mode_tag="net_sync",
    )


def _trigger_meta(rows: list[dict[str, Any]] | None) -> tuple[set[str], set[str], str | None]:
    """coins/symbols touched + bot ids + first tid."""
    coins: set[str] = set()
    bots: set[str] = set()
    trigger_tid = None
    for row in rows or []:
        if row.get("skipped"):
            continue
        bid = str(row.get("source") or "")
        if bid:
            bots.add(bid)
        c = str(row.get("coin") or "")
        if c:
            coins.add(c)
        tids = row.get("target_tids") or []
        if not trigger_tid and isinstance(tids, list) and tids:
            trigger_tid = str(tids[0])
        elif not trigger_tid and row.get("target_tid"):
            trigger_tid = str(row.get("target_tid"))
    return coins, bots, trigger_tid


def sync_subaccounts_from_paper(rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Per sub-account: sync that bot's filtered paper book onto its Bitget keys."""
    from quant.engine.exchanges.bitget.account import bitget_creds, load_creds_from_env
    from utils.hl_bitget_subaccounts import enabled_routes, routes_for_bot

    touched_coins, touched_bots, trigger_tid = _trigger_meta(rows)
    routes = enabled_routes()
    if not routes:
        logger.warning("HL→Bitget sub mode: no enabled subaccounts")
        return [
            {
                "status": "blocked",
                "error": "no enabled subaccounts",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        ]

    # Only sync routes whose bot was touched; if no bot in rows, sync all enabled
    if touched_bots:
        routes = [r for r in routes if r.bot_id in touched_bots]
        # also include routes for bots that share coin triggers via routes_for_bot
        if not routes:
            for bid in touched_bots:
                routes.extend(routes_for_bot(bid))
            # dedupe
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

        # Symbols to touch: trigger coins + desired + (if paper flat) open Bitget book
        symbols: set[str] = set(desired.keys())
        for c in touched_coins:
            sym = hl_coin_to_bitget(c, route_coins=route.coins)
            if sym:
                symbols.add(sym)

        with bitget_creds(creds if creds.ok() else None):
            if not symbols and (not touched_bots or route.bot_id in touched_bots):
                # Risk-halt / full flat: discover open Bitget positions and flatten
                try:
                    from quant.engine.exchanges.bitget.account import fetch_all_signed_positions

                    open_pos = fetch_all_signed_positions()
                    for sym, sz in open_pos.items():
                        if abs(sz) < 1e-12:
                            continue
                        base = sym[:-4] if sym.endswith("USDT") else sym
                        if not route.allows_coin(base):
                            continue
                        symbols.add(sym)
                except Exception as exc:
                    logger.warning("sub flatten discover failed [%s]: %s", route.id, exc)

            if not symbols:
                continue

            for sym in sorted(symbols):
                want = float(desired.get(sym) or 0.0)
                out.extend(
                    sync_account_symbol(
                        sym,
                        want,
                        account_id=route.id,
                        parts={route.bot_id: want},
                        trigger_tid=trigger_tid,
                        mode_tag="sub_sync",
                    )
                )
                time.sleep(0.05)
    return out


def sync_net_from_paper(rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Legacy: recompute net desires; sync on main BITGET_* account."""
    net, parts = compute_net_desired()
    coins_raw, _, trigger_tid = _trigger_meta(rows)
    coins: set[str] = set()
    for c in coins_raw:
        sym = hl_coin_to_bitget(c)
        if sym:
            coins.add(sym)
    coins.update(net.keys())

    out: list[dict[str, Any]] = []
    for sym in sorted(coins):
        desired = float(net.get(sym) or 0.0)
        out.extend(
            sync_account_symbol(
                sym,
                desired,
                account_id="main",
                parts=parts.get(sym),
                trigger_tid=trigger_tid,
                mode_tag="net_sync",
            )
        )
        time.sleep(0.05)
    return out


def maybe_execute_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        return apply_mirror_rows(rows)
    except Exception:
        logger.exception("HL Bitget executor failed")
        return []


def _flush_debounced(gen: int) -> None:
    """Timer callback: sync once using all rows accumulated for this generation."""
    global _debounce_timer
    with _debounce_lock:
        if gen != _debounce_gen:
            return
        batch = list(_debounce_pending)
        _debounce_pending.clear()
        _debounce_timer = None
    if not batch:
        return
    logger.info(
        "HL→Bitget debounce flush n_rows=%s bots=%s",
        len(batch),
        sorted({str(r.get("source") or r.get("bot_id") or "") for r in batch if r}),
    )
    with _bg_lock:
        maybe_execute_rows(batch)


def maybe_execute_rows_async(rows: list[dict[str, Any]]) -> None:
    """Queue Bitget sync after paper fills. Default: debounce burst fills (~1s)."""
    if not rows or not live_enabled():
        return

    ms = debounce_ms()
    if ms <= 0:
        def _run() -> None:
            with _bg_lock:
                maybe_execute_rows(rows)

        try:
            threading.Thread(target=_run, name="hl-bitget-exec", daemon=True).start()
        except Exception:
            logger.exception("HL Bitget async dispatch failed")
            maybe_execute_rows(rows)
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

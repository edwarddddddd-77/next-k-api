"""HL → Bitget sub-account routing config.

Each entry binds one paper bot (watchlist id) to one Bitget API key set,
optionally limited to a coin allowlist.

Credentials live in Railway/env only (never in JSON):
  {env_prefix}_API_KEY / _API_SECRET / _PASSPHRASE

Hard cap: at most HL_BITGET_MAX_SUBACCOUNTS enabled routes (default 5).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.hl_short_term import PROJECT_ROOT, resolve_data_dir

logger = logging.getLogger(__name__)

CONFIG_NAME = "hl_bitget_subaccounts.json"
DEFAULT_MAX_SUBACCOUNTS = 5

_lock = threading.Lock()
_cache: dict[str, Any] | None = None
_cache_mtime: float | None = None


@dataclass(frozen=True)
class SubAccountRoute:
    id: str
    label: str
    bot_id: str
    coins: frozenset[str] | None  # None = all coins for that bot
    enabled: bool
    env_prefix: str
    scale: float

    def allows_coin(self, coin: str) -> bool:
        if self.coins is None:
            return True
        from utils.hl_bitget_symbol_map import hl_base_ticker

        base = hl_base_ticker(coin)
        if not base:
            return False
        allowed = {hl_base_ticker(c) or str(c).strip().upper() for c in self.coins}
        return base in allowed


def max_subaccounts() -> int:
    try:
        n = int(os.getenv("HL_BITGET_MAX_SUBACCOUNTS", str(DEFAULT_MAX_SUBACCOUNTS)) or DEFAULT_MAX_SUBACCOUNTS)
    except (TypeError, ValueError):
        n = DEFAULT_MAX_SUBACCOUNTS
    return max(1, min(5, n))  # hard ceiling 5


def _config_path() -> Path:
    data = resolve_data_dir() / CONFIG_NAME
    if data.exists():
        return data
    return PROJECT_ROOT / CONFIG_NAME


def invalidate_cache() -> None:
    global _cache, _cache_mtime
    with _lock:
        _cache = None
        _cache_mtime = None


def load_subaccounts_doc(*, force: bool = False) -> dict[str, Any]:
    global _cache, _cache_mtime
    path = _config_path()
    with _lock:
        try:
            mtime = path.stat().st_mtime if path.exists() else None
        except OSError:
            mtime = None
        if not force and _cache is not None and mtime == _cache_mtime:
            return _cache
        if not path.exists():
            doc = {"updated": None, "subaccounts": [], "error": f"missing: {path}"}
            _cache = doc
            _cache_mtime = mtime
            return doc
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(doc, dict):
                doc = {"subaccounts": [], "error": "invalid root"}
        except Exception as exc:
            logger.warning("hl_bitget_subaccounts load failed: %s", exc)
            doc = {"subaccounts": [], "error": str(exc)}
        _cache = doc
        _cache_mtime = mtime
        return doc


def parse_routes(doc: dict[str, Any] | None = None) -> list[SubAccountRoute]:
    raw = doc if doc is not None else load_subaccounts_doc()
    out: list[SubAccountRoute] = []
    seen_ids: set[str] = set()
    for row in raw.get("subaccounts") or []:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id") or "").strip()
        bot_id = str(row.get("bot_id") or "").strip()
        if not rid or not bot_id:
            continue
        if rid in seen_ids:
            logger.warning("duplicate subaccount id skipped: %s", rid)
            continue
        seen_ids.add(rid)
        coins_raw = row.get("coins")
        coins: frozenset[str] | None
        if coins_raw is None or coins_raw == [] or coins_raw == "*":
            coins = None
        elif isinstance(coins_raw, str):
            coins = frozenset(c.strip().upper() for c in coins_raw.split(",") if c.strip())
        else:
            coins = frozenset(str(c).strip().upper() for c in coins_raw if str(c).strip())
            if not coins:
                coins = None
        try:
            scale = max(0.0, float(row.get("scale", 1) or 1))
        except (TypeError, ValueError):
            scale = 1.0
        enabled = bool(row.get("enabled", False))
        # Railway override: HL_BITGET_SUB_<ID>_ENABLED=1
        env_en = os.getenv(f"HL_BITGET_SUB_{rid.upper()}_ENABLED", "").strip().lower()
        if env_en in ("1", "true", "yes", "on"):
            enabled = True
        elif env_en in ("0", "false", "no", "off"):
            enabled = False
        out.append(
            SubAccountRoute(
                id=rid,
                label=str(row.get("label") or rid).strip(),
                bot_id=bot_id,
                coins=coins,
                enabled=enabled,
                env_prefix=str(row.get("env_prefix") or "").strip(),
                scale=scale,
            )
        )
    return out


def enabled_routes() -> list[SubAccountRoute]:
    """Enabled routes, hard-capped at max_subaccounts() (fail-closed over cap)."""
    enabled = [r for r in parse_routes() if r.enabled]
    cap = max_subaccounts()
    if len(enabled) > cap:
        logger.error(
            "enabled subaccounts %d > max %d — refusing all until trimmed",
            len(enabled),
            cap,
        )
        return []
    return enabled


def routes_for_bot(bot_id: str) -> list[SubAccountRoute]:
    bid = str(bot_id or "").strip()
    return [r for r in enabled_routes() if r.bot_id == bid]


def validate_routes(routes: list[SubAccountRoute] | None = None) -> list[str]:
    """Return human-readable config problems (empty = ok)."""
    routes = routes if routes is not None else parse_routes()
    problems: list[str] = []
    enabled = [r for r in routes if r.enabled]
    cap = max_subaccounts()
    if len(enabled) > cap:
        problems.append(f"enabled subaccounts {len(enabled)} > max {cap}")

    seen_bots: dict[str, str] = {}
    for r in enabled:
        if not r.env_prefix:
            problems.append(
                f"route {r.id}: env_prefix required (Railway: {r.id.upper()} API keys)"
            )
        if r.bot_id in seen_bots and seen_bots[r.bot_id] != r.id:
            # one bot → one subaccount recommended; coin-split still allowed via validate below
            pass
        seen_bots[r.bot_id] = r.id

    for i, a in enumerate(enabled):
        for b in enabled[i + 1 :]:
            if a.bot_id != b.bot_id:
                continue
            if a.coins is None or b.coins is None:
                problems.append(
                    f"bot {a.bot_id} mapped to both {a.id} and {b.id} with overlapping all-coins"
                )
                continue
            overlap = a.coins & b.coins
            if overlap:
                problems.append(
                    f"bot {a.bot_id} coin overlap {sorted(overlap)} on {a.id} and {b.id}"
                )
    return problems

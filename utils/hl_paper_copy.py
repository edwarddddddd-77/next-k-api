"""Hyperliquid paper copy — fill-follow with Mirror-style reduce + reconcile.

One bot per watchlist seat (default 1000U; override via watchlist paper_balance).
Multiple seats may share one HL address (e.g. former K paper + O live twin); target
health / WS fills are keyed by address and applied to every matching bot.

Sizing:
  open/add: our_delta = fill.sz × (bot_equity / target_equity) at fill.px
  target_equity = perp AV (main+xyz) + Core spot USDC — stable across spot↔perp transfers
  target_av stays perp-only (empty/inactive health); HyperEVM USDC is monitor-only
  flat-entry burst: coalesce same-sign clips only when leader pre≈0 (true new entry)
  orphan add (local flat, leader already had inventory): skip — no silent stub from Add
    HL often labels mid-book adds as Open*; trust startPosition / had_prior over dir text
  copy_current (watchlist, default off): on orphan, open full leader coin × ratio once
    (Dextrabot/Legend Copy Current — never open the Add delta alone)
  twin seats (mirror_of + live): after fills, Copy-Current sync to sibling paper × equity
  dust open: skip with reason dust_open (never silent)
  reduce: scale our size by leader remaining fraction (startPosition→post), not raw δ×ratio
  reconcile: leader flat on a coin → close our leg (Bitget syncs paper)

Live seats (Railway HL_BITGET_ENABLE_BOTS / SUB_*_ENABLED):
  1) live_only — no paper ledger; desk shows Bitget wallet/positions.
  2) Event-driven Bitget (copy_current=off): open/size-up only on leader
     fill signals; size-down only when that coin actually reduces/flattens;
     no signal → hold (no AV-chase top-up or shrink). Snapshots merge HIP-3 xyz.
  3) enter-live does NOT Bitget-align when copy_current=off.
  4) Paper JSON is atomic tmp+replace + .bak; corrupt load recovers bak.
  5) Seat disabled / removed → exchange flatten attempted.
  6) Non-enabled seats stay paper as before.

WS snapshots are ignored so deploy starts flat. Mark refresh updates uPnL and
optionally runs the per-bot hard-stop (OFF unless HL_DAILY_LOSS_PCT>0).
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
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
# Live venue ops queued outside the paper lock.
_pending_live_flatten: list[str] = []
_pending_live_align: list[str] = []
_dup_addr_warned: set[str] = set()


def _truthy_flag(raw: Any) -> bool:
    if raw is True:
        return True
    if raw is False or raw is None:
        return False
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _bot_copy_current(bot: dict[str, Any] | None) -> bool:
    """Watchlist copy_current — explicit mid-book entry (Dextrabot Copy Current)."""
    if not isinstance(bot, dict):
        return False
    return _truthy_flag(bot.get("copy_current"))


def _queue_live_flatten(bot_ids: list[str] | set[str]) -> None:
    for bid in bot_ids:
        s = str(bid or "").strip()
        if s and s not in _pending_live_flatten:
            _pending_live_flatten.append(s)


def _queue_live_align(bot_ids: list[str] | set[str]) -> None:
    """Align Bitget to leader×equity after entering live_only."""
    for bid in bot_ids:
        s = str(bid or "").strip()
        if s and s not in _pending_live_align:
            _pending_live_align.append(s)


def flush_pending_live_flatten() -> list[str]:
    """Run outside paper locks: flatten retired seats + align new live_only seats."""
    global _pending_live_flatten, _pending_live_align
    flat_ids = list(_pending_live_flatten)
    align_ids = list(_pending_live_align)
    _pending_live_flatten = []
    _pending_live_align = []
    if flat_ids:
        _sync_live_after_paper_reset(flat_ids)
    if align_ids:
        _sync_live_align(align_ids)
    return flat_ids + align_ids


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


def _bak_path() -> Path:
    return _data_dir() / f"{PAPER_NAME}.bak"


def _try_read_paper_dict(path: Path) -> dict[str, Any] | None:
    """Return parsed paper dict, or None if missing/empty/corrupt."""
    try:
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _atomic_write_json(
    path: Path,
    data: dict[str, Any],
    *,
    refresh_bak: bool = True,
) -> None:
    """Write JSON via tmp+replace. Refresh .bak only from a known-good primary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bak = path.with_name(path.name + ".bak")
    if refresh_bak and path.is_file() and path.stat().st_size > 8:
        prev = _try_read_paper_dict(path)
        if prev is not None:
            try:
                shutil.copy2(path, bak)
            except Exception:
                logger.warning("hl paper bak copy failed path=%s", path, exc_info=True)
    # Unique tmp avoids two writers truncating the same *.tmp.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    tmp.write_text(payload, encoding="utf-8")
    try:
        tmp.replace(path)
    except Exception:
        if tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _scrub_live_only_for_disk(data: dict[str, Any]) -> dict[str, Any]:
    """Drop exchange overlays before persisting (same as save_paper)."""
    for bot in (data.get("bots") or {}).values():
        if not isinstance(bot, dict) or not is_live_only_bot(bot):
            continue
        bot["positions"] = {}
        bot["balance"] = 0.0
        bot["equity"] = 0.0
        bot["realized_pnl"] = 0.0
        for k in ("u_pnl", "live_available", "live_error", "live_at", "paper_balance"):
            if k == "paper_balance":
                bot[k] = 0.0
            else:
                bot.pop(k, None)
    return data


def _rewrite_primary_from_recovered(data: dict[str, Any]) -> None:
    """Persist bak-recovered book without re-running _ensure_bots / save_paper.

    Avoids nested ensure side-effects and does not refresh .bak from a bad
    primary (refresh_bak=False).
    """
    payload = dict(data)
    payload = _scrub_live_only_for_disk(payload)
    payload = _aggregate(payload)
    payload["updated_at"] = _now()
    payload.pop("error", None)
    _atomic_write_json(_path(), payload, refresh_bak=False)


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
        # Per-bot hard stop vs cycle anchor. OFF by default (7d explore: −25% halt
        # cut desk +20%→+4% and worsened MDD). Re-enable via HL_DAILY_LOSS_PCT.
        "daily_loss_pct": _env_float("HL_DAILY_LOSS_PCT", 0.0),
        "bot_halt_cooldown_sec": _env_float("HL_BOT_HALT_COOLDOWN_SEC", 6 * 3600),
        # Desk peak-to-trough: flatten all when equity falls ≥N from running peak.
        "portfolio_peak_dd_pct": _env_float("HL_PORTFOLIO_PEAK_DD_PCT", 0.15),
        # Desk-wide TP/SL vs compound anchor (optional; off by default).
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
            "Fill-delta market follow. Per-bot hard stop OFF by default "
            "(set HL_DAILY_LOSS_PCT e.g. 0.25 to re-enable): flatten only, "
            "rebase anchor, keep following — no pause. "
            "Desk peak drawdown flatten ON at −15% "
            "(HL_PORTFOLIO_PEAK_DD_PCT; 0 disables). "
            "Desk soft/hard TP/SL vs anchor OFF by default "
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


# Gentle anti-martingale sleeve (watchlist anti_martingale=gentle / section=anti_martingale)
AM_SCALE_CAP = 2.0
AM_SCALE_STEP = 0.5
AM_MDD_BREAK = 0.15
AM_COOLDOWN_SEC = 24 * 3600


def _is_am_bot(bot: dict[str, Any] | None) -> bool:
    if not isinstance(bot, dict):
        return False
    if str(bot.get("section") or "").strip().lower() == "anti_martingale":
        return True
    mode = str(bot.get("anti_martingale") or "").strip().lower()
    return mode in ("gentle", "1", "true", "yes", "on")


def _am_scale_mult(bot: dict[str, Any]) -> float:
    if not _is_am_bot(bot):
        return 1.0
    try:
        scale = float(bot.get("am_scale") or 1.0)
    except (TypeError, ValueError):
        scale = 1.0
    if scale <= 0:
        return 0.0
    return max(1.0, min(AM_SCALE_CAP, scale))


def _am_settle_day(bot: dict[str, Any], end_eq: float) -> None:
    """Beijing day roll: green +0.5 (cap 2), red → 1, flat hold; respect cooldown."""
    try:
        start = float(bot.get("am_day_start_equity") or bot.get("day_start_equity") or end_eq)
    except (TypeError, ValueError):
        start = end_eq
    ret = (end_eq - start) / start if start > 1e-9 else 0.0
    try:
        old = float(bot.get("am_scale") or 1.0)
    except (TypeError, ValueError):
        old = 1.0
    now = time.time()
    try:
        cool = float(bot.get("am_cooldown_until") or 0)
    except (TypeError, ValueError):
        cool = 0.0
    if cool > now:
        bot["am_scale"] = 1.0
        bot["am_reason"] = "cooldown"
    elif abs(ret) <= 1e-6:
        bot["am_scale"] = max(1.0, min(AM_SCALE_CAP, old))
        bot["am_reason"] = "flat_hold"
    elif ret > 0:
        nxt = min(AM_SCALE_CAP, old + AM_SCALE_STEP)
        bot["am_scale"] = round(nxt, 4)
        bot["am_reason"] = "green_up" if nxt > old + 1e-12 else "green_cap"
    else:
        bot["am_scale"] = 1.0
        bot["am_reason"] = "red_reset"
    bot["am_last_day_ret_pct"] = round(ret * 100.0, 4)


def _am_maybe_mdd_break(bot: dict[str, Any]) -> None:
    if not _is_am_bot(bot):
        return
    _recompute_bot(bot)
    try:
        eq = float(bot.get("equity") or 0)
    except (TypeError, ValueError):
        return
    try:
        peak = float(bot.get("am_peak_equity") or 0)
    except (TypeError, ValueError):
        peak = 0.0
    if eq > peak:
        bot["am_peak_equity"] = round(eq, 4)
        peak = eq
    try:
        scale = float(bot.get("am_scale") or 1.0)
    except (TypeError, ValueError):
        scale = 1.0
    if peak > 1e-9 and scale > 1.0 + 1e-12:
        dd = (peak - eq) / peak
        if dd >= AM_MDD_BREAK:
            bot["am_scale"] = 1.0
            bot["am_cooldown_until"] = time.time() + AM_COOLDOWN_SEC
            bot["am_reason"] = "mdd_break"
            bot["am_last_dd_pct"] = round(dd * 100.0, 4)
            logger.info(
                "AM MDD break bot=%s dd=%.1f%% scale→1 cooldown=%sh",
                bot.get("id"),
                dd * 100.0,
                AM_COOLDOWN_SEC / 3600,
            )


def _desk_bots(book: dict[str, Any]) -> list[dict[str, Any]]:
    """Main desk seats — excludes anti-martingale sleeve (own risk island)."""
    return [b for b in _iter_bots(book) if not _is_am_bot(b)]


def _am_bots(book: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in _iter_bots(book) if _is_am_bot(b)]


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


def is_live_only_bot(bot: dict[str, Any] | None) -> bool:
    """True for seats that mirror on exchange only (no paper PnL book).

    Opt-in via paper:false / live_only / mode=live_only, or live+venue
    (binance|bitget). Explicit paper:true always wins (paper→sub twin).
    """
    if not isinstance(bot, dict):
        return False
    if bot.get("paper") is True:
        return False
    if bot.get("live_only") is True:
        return True
    if bot.get("paper") is False:
        return True
    venue = str(bot.get("venue") or "").strip().lower()
    if venue in ("binance", "bitget") and bot.get("live") is True and bot.get("paper") is not True:
        return True
    return False


def _empty_bot(wallet: dict[str, Any], balance: float) -> dict[str, Any]:
    bot = {
        "id": wallet.get("id") or str(wallet.get("address") or "")[:10],
        "address": wallet.get("address"),
        "balance": balance,
        "equity": balance,
        "realized_pnl": 0.0,
        "positions": {},
        "fills": [],
        "copy_ratio": None,
        "target_av": None,
        "target_equity": None,
        "target_spot_usdc": None,
        "target_last_fill_at": None,
        "target_health": None,
        "risk_halted": False,
        "day_key": None,
        "day_start_equity": balance,
        "risk_anchor_equity": balance,
    }
    section = str(wallet.get("section") or "").strip().lower() or None
    am = str(wallet.get("anti_martingale") or "").strip().lower() or None
    if section:
        bot["section"] = section
    if am:
        bot["anti_martingale"] = am
    if wallet.get("mirror_of"):
        bot["mirror_of"] = str(wallet.get("mirror_of"))
    if section == "anti_martingale" or am in ("gentle", "1", "true", "yes", "on"):
        bot["section"] = "anti_martingale"
        bot["anti_martingale"] = am or "gentle"
        bot["am_scale"] = 1.0
        bot["am_reason"] = "init"
        bot["am_peak_equity"] = balance
        bot["am_day_start_equity"] = balance
        bot["am_cooldown_until"] = 0
    return bot


def _rebase_desk_peak_anchor(data: dict[str, Any], bots: dict[str, Any], *, why: str) -> None:
    """Reset desk peak/anchor to current equity after membership changes.

    Without this, cutting seats (e.g. 11→4) leaves a stale high-water mark and the
    peak-DD stop can false-trip flatten + Bitget sync.
    """
    for bot in bots.values():
        if isinstance(bot, dict):
            _recompute_bot(bot)
    eq = 0.0
    n_desk = 0
    for bot in bots.values():
        if not isinstance(bot, dict):
            continue
        if _is_am_bot(bot) or is_live_only_bot(bot):
            continue
        try:
            eq += float(bot.get("equity") or bot.get("balance") or 0)
            n_desk += 1
        except (TypeError, ValueError):
            continue
    eq = round(eq, 4)
    data["portfolio_anchor_equity"] = eq
    data["portfolio_peak_equity"] = eq
    data["portfolio_peak_dd_pct"] = 0.0
    data["portfolio_return_pct"] = 0.0
    data["portfolio_soft_tp_taken"] = False
    data["portfolio_copy_scale"] = 1.0
    logger.info("desk peak/anchor rebase to %.2f (%s, desk_bots=%s)", eq, why, n_desk)


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

    before_ids = set(bots.keys())
    before_addrs = {
        k: str((v or {}).get("address") or "").strip().lower()
        for k, v in bots.items()
        if isinstance(v, dict)
    }
    membership_changed = False

    want_ids: set[str] = set()
    addr_seats: dict[str, list[str]] = {}
    for w in wallets:
        bid = str(w.get("id") or w.get("address") or "")[:32]
        want_ids.add(bid)
        a = str(w.get("address") or "").strip().lower()
        if a:
            addr_seats.setdefault(a, []).append(bid)
    # One address → one seat (warn once; never share an address with a live seat).
    try:
        from utils.hl_bitget_subaccounts import env_enable_bot_tokens

        enable_tokens = {t.lower() for t in env_enable_bot_tokens()}
    except Exception:
        enable_tokens = set()
    for a, ids in addr_seats.items():
        if len(ids) <= 1 or a in _dup_addr_warned:
            continue
        _dup_addr_warned.add(a)
        live_ids = [
            i
            for i in ids
            if i.lower() in enable_tokens
            or i.lower().replace("bot_", "") in enable_tokens
            or _truthy_flag((bots.get(i) or {}).get("live_only"))
        ]
        logger.warning(
            "watchlist duplicate address %s seats=%s%s",
            a[:14],
            ids,
            f" live={live_ids}" if live_ids else "",
        )
    for w in wallets:
        bid = str(w.get("id") or w.get("address") or "")[:32]
        init = _bot_initial_balance(w, cfg)
        new_addr = str(w.get("address") or "").strip().lower()
        if bid not in bots:
            bots[bid] = _empty_bot(w, init)
            bots[bid]["paper_balance"] = init
            membership_changed = True
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
                membership_changed = True
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
        # Railway-enabled Bitget seats are live_only (no paper). Watchlist flags
        # still work for non-Railway seats; Railway enable overrides paper:true.
        live_flag = bool(w.get("live"))
        paper_flag = w.get("paper")
        venue = str(w.get("venue") or "").strip().lower() or None
        mode = str(w.get("mode") or "").strip().lower()
        env_live = False
        try:
            from utils.hl_bitget_subaccounts import route_id_for_bot, seat_enabled_by_env

            rid = route_id_for_bot(bid)
            env_live = seat_enabled_by_env(route_id=rid, bot_id=bid) is True
        except Exception:
            env_live = False
        live_only = (
            env_live
            or paper_flag is False
            or mode == "live_only"
            or w.get("live_only") is True
            or (
                live_flag
                and venue in ("binance", "bitget")
                and paper_flag is not True
            )
        )
        if paper_flag is True and not env_live:
            live_only = False
        if env_live:
            live_flag = True
            if not venue:
                venue = "bitget"
            live_only = True
        bots[bid]["live"] = live_flag
        bots[bid]["venue"] = venue
        bots[bid]["live_only"] = bool(live_only)
        bots[bid]["paper"] = False if live_only else True
        # Explicit mid-book entry (default off — Legend/Dextrabot default).
        was_copy_current = _bot_copy_current(bots[bid])
        bots[bid]["copy_current"] = _truthy_flag(w.get("copy_current"))
        now_copy_current = _bot_copy_current(bots[bid])
        if live_only and not bots[bid].get("paper_cleared_for_live"):
            bots[bid]["positions"] = {}
            bots[bid]["fills"] = []
            bots[bid]["balance"] = 0.0
            bots[bid]["equity"] = 0.0
            bots[bid]["realized_pnl"] = 0.0
            bots[bid]["paper_balance"] = 0.0
            bots[bid]["paper_cleared_for_live"] = True
            membership_changed = True
            # copy_current=off: NEVER Bitget-align on enter. Align after a
            # corrupt-paper rebuild was topping up an already-open seat (C
            # 2026-08-05). New opens only come from real flat→open fills.
            if now_copy_current:
                _queue_live_align([bid])
                logger.info(
                    "entered live_only %s — queued Bitget align (copy_current)",
                    bid,
                )
            else:
                logger.info(
                    "entered live_only %s — Bitget align skipped "
                    "(copy_current=off; follow fills only)",
                    bid,
                )
        elif (
            live_only
            and bots[bid].get("paper_cleared_for_live")
            and now_copy_current
            and not was_copy_current
        ):
            # User flipped copy_current off→on: one-shot sync to leader×equity.
            membership_changed = True
            _queue_live_align([bid])
            logger.info(
                "copy_current ON %s — queued Bitget sync align",
                bid,
            )
        elif not live_only and bots[bid].get("paper_cleared_for_live"):
            # Leaving live-only: re-seed paper book; do not keep stale live/venue.
            init = _bot_initial_balance(w, cfg)
            keep_keys = (
                "allow_coins",
                "tag",
                "ht_style",
                "style_tags",
                "address",
                "target_av",
                "target_equity",
                "target_spot_usdc",
                "target_positions",
                "target_lev_by_coin",
                "target_last_fill_at",
            )
            kept = {k: bots[bid].get(k) for k in keep_keys}
            bots[bid].update(_empty_bot(w, init))
            bots[bid]["paper_balance"] = init
            bots[bid].pop("paper_cleared_for_live", None)
            bots[bid]["live_only"] = False
            bots[bid]["paper"] = True
            bots[bid]["live"] = live_flag
            bots[bid]["venue"] = venue
            for k, v in kept.items():
                if v is not None:
                    bots[bid][k] = v
            membership_changed = True
            _queue_live_flatten([bid])
            logger.info("restored paper book for %s after leaving live-only (%.0fU)", bid, init)
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
        # Anti-martingale sleeve metadata (paper-only twins of A/E/C etc.)
        section = str(w.get("section") or "").strip().lower() or None
        am_mode = str(w.get("anti_martingale") or "").strip().lower() or None
        if section == "anti_martingale" or am_mode in ("gentle", "1", "true", "yes", "on"):
            bots[bid]["section"] = "anti_martingale"
            bots[bid]["anti_martingale"] = am_mode or "gentle"
            bots[bid].setdefault("am_scale", 1.0)
            bots[bid].setdefault("am_reason", "init")
            bots[bid].setdefault("am_peak_equity", float(bots[bid].get("equity") or init))
            bots[bid].setdefault(
                "am_day_start_equity",
                float(bots[bid].get("day_start_equity") or bots[bid].get("equity") or init),
            )
            bots[bid].setdefault("am_cooldown_until", 0)
            if w.get("mirror_of"):
                bots[bid]["mirror_of"] = str(w.get("mirror_of"))
        else:
            bots[bid].pop("section", None)
            # leave leftover am_* keys harmless if seat was never AM

        # Drop bots removed from the watchlist (old dig ids clutter the desk)
    if want_ids:
        removed = sorted(set(bots.keys()) - want_ids)
        if removed:
            _queue_live_flatten(removed)
            logger.info("paper prune retired seats %s → live flatten queued", removed)
        bots = {k: v for k, v in bots.items() if k in want_ids}

    after_ids = set(bots.keys())
    if before_ids != after_ids:
        membership_changed = True
    else:
        for bid in after_ids:
            cur = str((bots.get(bid) or {}).get("address") or "").strip().lower()
            if before_addrs.get(bid) != cur:
                membership_changed = True
                break

    if membership_changed:
        _rebase_desk_peak_anchor(
            data,
            bots,
            why=f"membership {sorted(before_ids)}→{sorted(after_ids)}",
        )

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
    am_balance = 0.0
    am_equity = 0.0
    am_realized = 0.0
    alerts: list[str] = []
    for bot in bots.values():
        _recompute_bot(bot)
        # Refresh quiet-hours label without hitting HL
        if bot.get("target_av") is not None or bot.get("target_last_fill_at") is not None:
            bot["target_health"] = _compute_target_health(bot)
            h = bot["target_health"]
            if not h.get("ok"):
                alerts.append(f"{bot.get('id')}:{h.get('status')}")
        # Live-only seats do not contribute paper desk equity (real $ comes from overlay).
        if is_live_only_bot(bot):
            fills.extend(
                f
                for f in (bot.get("fills") or [])
                if isinstance(f, dict) and f.get("action") == "live_sync"
            )
            continue
        if _is_am_bot(bot):
            am_balance += float(bot.get("balance") or 0)
            am_equity += float(bot.get("equity") or 0)
            am_realized += float(bot.get("realized_pnl") or 0)
            for k, p in (bot.get("positions") or {}).items():
                positions[k] = p
            fills.extend(bot.get("fills") or [])
            continue
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
    data["am_balance"] = round(am_balance, 4)
    data["am_equity"] = round(am_equity, 4)
    data["am_realized_pnl"] = round(am_realized, 4)
    data["positions"] = positions
    data["fills"] = fills[:500]
    data["bot_count"] = len(bots)
    data["am_bot_count"] = sum(1 for b in bots.values() if isinstance(b, dict) and _is_am_bot(b))
    data["target_alerts"] = alerts
    data["ok"] = True
    data["mode"] = "fill_delta_market"
    data["config"] = paper_config()
    # Portfolio risk snapshot — main desk only (AM sleeve excluded)
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
    try:
        peak_f = float(data.get("portfolio_peak_equity") or 0)
    except (TypeError, ValueError):
        peak_f = 0.0
    if peak_f < equity:
        peak_f = equity
        if peak_f > 0:
            data["portfolio_peak_equity"] = round(peak_f, 4)
    data["portfolio_peak_equity"] = round(peak_f, 4) if peak_f > 0 else None
    if peak_f > 1e-9:
        data["portfolio_peak_dd_pct"] = round((peak_f - equity) / peak_f, 6)
    else:
        data["portfolio_peak_dd_pct"] = None
    data["portfolio_copy_scale"] = float(data.get("portfolio_copy_scale") or 1.0)
    data["portfolio_halted_count"] = sum(
        1
        for b in bots.values()
        if isinstance(b, dict) and b.get("risk_halted") and not _is_am_bot(b)
    )
    pr = data.get("portfolio_risk")
    data["portfolio_risk"] = pr if isinstance(pr, dict) else None
    return data


def slim_paper_for_api(
    data: dict[str, Any],
    *,
    per_bot_fills: int = 48,
    agg_fills: int = 100,
) -> dict[str, Any]:
    """Trim fill history for HTTP responses — disk keeps full book, UI only needs recent rows.

    Large multi-bot ledgers were ~0.9MB / 10–15s and caused intermittent
    「连接失败」on the mirror desk when the browser or edge timed out.
    """
    bots_in = data.get("bots") if isinstance(data.get("bots"), dict) else {}
    bots_out: dict[str, Any] = {}
    for bid, bot in bots_in.items():
        if not isinstance(bot, dict):
            continue
        b = dict(bot)
        fills = b.get("fills")
        if isinstance(fills, list) and len(fills) > per_bot_fills:
            b["fills"] = fills[:per_bot_fills]
        th = b.get("target_health")
        if isinstance(th, dict):
            th2 = dict(th)
            bal = th2.get("target_spot_balances")
            if isinstance(bal, list) and len(bal) > 8:
                th2["target_spot_balances"] = bal[:8]
            sf = th2.get("target_spot_fills")
            if isinstance(sf, list) and len(sf) > 8:
                th2["target_spot_fills"] = sf[:8]
            b["target_health"] = th2
        bots_out[str(bid)] = b
    out = dict(data)
    out["bots"] = bots_out
    top = out.get("fills")
    if isinstance(top, list) and len(top) > agg_fills:
        out["fills"] = top[:agg_fills]
    return out


def load_paper() -> dict[str, Any]:
    path = _path()
    bak = _bak_path()
    recovered_from_bak = False
    data = _try_read_paper_dict(path)
    if data is None:
        # Primary missing OR corrupt/empty — always try bak before empty shell.
        # (Bugbot: missing primary previously skipped bak and could wipe it later.)
        if path.exists() or bak.is_file():
            why = "corrupt/empty" if path.exists() else "missing"
            logger.error(
                "hl paper primary %s path=%s — trying bak",
                why,
                path,
            )
            data = _try_read_paper_dict(bak)
            if data is not None:
                recovered_from_bak = True
                logger.error("hl paper recovered from bak %s", bak)
            else:
                logger.error(
                    "hl paper unrecovered (primary+bak bad) — empty shell; "
                    "live seats will NOT Bitget-align on rebuild (copy_current=off)"
                )
                data = {
                    "bots": {},
                    "updated_at": _now(),
                    "error": "paper_corrupt_unrecovered",
                }
    if data is None:
        data = {"bots": {}, "updated_at": _now()}
    data = _ensure_bots(data)
    out = _aggregate(data)
    if recovered_from_bak:
        try:
            # Rewrite primary only; do not touch .bak / do not nest save_paper.
            _rewrite_primary_from_recovered(data)
        except Exception:
            logger.exception("hl paper rewrite-after-bak-recover failed")
    return out


def save_paper(data: dict[str, Any]) -> None:
    data = _ensure_bots(data)
    data = _scrub_live_only_for_disk(data)
    data = _aggregate(data)
    data["updated_at"] = _now()
    # Never persist an unrecovered empty shell over a good .bak.
    if str(data.get("error") or "") == "paper_corrupt_unrecovered":
        logger.error("hl paper save refused: unrecovered corrupt shell")
        return
    _atomic_write_json(_path(), data)


def reset_paper() -> dict[str, Any]:
    with _lock:
        data = _ensure_bots({"bots": {}})
        cfg = paper_config()
        wallets = {str(w.get("id") or "")[:32]: w for w in load_watchlist()}
        reset_ids: list[str] = []
        for bot in data["bots"].values():
            bid = str(bot.get("id") or "")
            w = wallets.get(bid) or {"id": bid}
            bal = _bot_initial_balance(w, cfg)
            bot.update(_empty_bot(bot, bal))
            bot["paper_balance"] = bal
            bot.pop("risk_halted_at", None)
            if bid:
                reset_ids.append(bid)
        save_paper(data)
        out = load_paper()
    _sync_live_after_paper_reset(reset_ids)
    return out


def reset_paper_bot(bot_id: str) -> dict[str, Any]:
    """Reset one seat. Live-only → clear signals + flatten exchange; paper → wipe ledger."""
    bid = str(bot_id or "").strip()
    if not bid:
        raise ValueError("bot_id required")
    with _lock:
        data = load_paper()
        bots = data.get("bots") or {}
        bot = bots.get(bid)
        if not isinstance(bot, dict):
            raise LookupError(f"unknown bot: {bid}")
        if is_live_only_bot(bot):
            bot["positions"] = {}
            bot["fills"] = []
            bot["balance"] = 0.0
            bot["equity"] = 0.0
            bot["realized_pnl"] = 0.0
            bot["target_positions"] = {}
            bot.pop("risk_halted_at", None)
            bot["risk_halted"] = False
            save_paper(data)
            out = load_paper()
        else:
            cfg = paper_config()
            wallets = {str(w.get("id") or "")[:32]: w for w in load_watchlist()}
            w = wallets.get(bid) or {
                "id": bid,
                "address": bot.get("address"),
                "paper_balance": bot.get("paper_balance"),
            }
            bal = _bot_initial_balance(w, cfg)
            keep_allow = bot.get("allow_coins")
            bot.update(_empty_bot({**bot, **w, "id": bid}, bal))
            bot["paper_balance"] = bal
            bot.pop("risk_halted_at", None)
            if keep_allow is not None:
                bot["allow_coins"] = keep_allow
            save_paper(data)
            out = load_paper()
    _sync_live_after_paper_reset([bid])
    try:
        from utils.hl_binance_executor import overlay_live_bots as overlay_binance

        out = overlay_binance(out)
    except Exception:
        pass
    try:
        from utils.hl_bitget_executor import overlay_live_bots as overlay_bitget

        return overlay_bitget(out)
    except Exception:
        return out


def _sync_live_after_paper_reset(bot_ids: list[str]) -> None:
    """Immediately align live venues to flat/reset paper (skip fill debounce)."""
    rows = [
        {
            "id": f"reset-{bid}",
            "source": bid,
            "bot_id": bid,
            "action": "reset",
        }
        for bid in bot_ids
        if str(bid or "").strip()
    ]
    if not rows:
        return
    try:
        from utils.hl_bitget_executor import maybe_execute_rows_async

        maybe_execute_rows_async(rows, immediate=True)
    except Exception:
        logger.exception("paper reset bitget sync bots=%s", [r["source"] for r in rows])
    try:
        from utils.hl_binance_executor import maybe_execute_rows_async as bn_exec

        bn_exec(rows, immediate=True)
    except Exception:
        logger.exception("paper reset binance sync bots=%s", [r["source"] for r in rows])


def _sync_live_align(bot_ids: list[str]) -> None:
    """Align live_only seats to leader×equity (enter live / post-enable).

    Rows are tagged ``live_align`` so Bitget will not increase an already-open
    leg when copy_current=off (fill path owns size-ups).
    """
    rows = [
        {
            "id": f"align-{bid}",
            "source": bid,
            "bot_id": bid,
            "action": "live_align",
            "live_only": True,
        }
        for bid in bot_ids
        if str(bid or "").strip()
    ]
    if not rows:
        return
    # Fresh seat: never inherit stale pending opens from a prior live period.
    try:
        from utils.hl_bitget_executor import clear_pending_fresh_account
        from utils.hl_bitget_subaccounts import route_id_for_bot

        for bid in bot_ids:
            rid = route_id_for_bot(str(bid or "").strip())
            if rid:
                clear_pending_fresh_account(rid)
    except Exception:
        logger.exception("clear pending fresh on live align failed")
    try:
        from utils.hl_bitget_executor import maybe_execute_rows_async

        maybe_execute_rows_async(rows, immediate=True)
    except Exception:
        logger.exception("live_only align bitget sync bots=%s", [r["source"] for r in rows])
    try:
        from utils.hl_binance_executor import maybe_execute_rows_async as bn_exec

        bn_exec(rows, immediate=True)
    except Exception:
        logger.exception("live_only align binance sync bots=%s", [r["source"] for r in rows])


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
            if is_live_only_bot(bot):
                continue
            _roll_day(bot, cfg)
            for pos in (bot.get("positions") or {}).values():
                coin = str(pos.get("coin") or "")
                mid = _mid_for_coin(mids, coin) or float(pos.get("mark_px") or 0)
                if mid > 0:
                    _mark_one(pos, mid)
            _enforce_notional_caps(bot, mids, cfg)
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
        try:
            from utils.hl_binance_executor import maybe_execute_rows_async as bn_exec

            bn_exec(halt_logged)
        except Exception:
            logger.exception("HL Binance live hook failed (mark halt)")
    # Enter/leave live queues align/flatten inside load_paper; flush here so
    # ENABLE_BOTS swaps do not wait for the next WS fill or desk poll.
    try:
        flush_pending_live_flatten()
    except Exception:
        logger.exception("HL pending live flatten flush failed (mark)")
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
    start_position = None
    for key in ("startPosition", "startPos", "start_position"):
        if fill.get(key) is None or fill.get(key) == "":
            continue
        try:
            start_position = float(fill.get(key))
            break
        except (TypeError, ValueError):
            continue
    return {
        "coin": coin.upper(),
        "target_delta": signed,
        "px": px,
        "tid": tid or None,
        "fill_time": fill_time,
        "side": "buy" if signed > 0 else "sell",
        "dir": str(fill.get("dir") or "").strip(),
        "start_position": start_position,
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
    _recompute_target_equity(bot)


def _recompute_target_equity(bot: dict[str, Any]) -> float:
    """Sizing denom: perp AV (main+xyz) + Core spot USDC. Not HyperEVM."""
    try:
        perp = float(bot.get("target_av") or 0)
    except (TypeError, ValueError):
        perp = 0.0
    try:
        spot = float(bot.get("target_spot_usdc") or 0)
    except (TypeError, ValueError):
        spot = 0.0
    if spot < 0:
        spot = 0.0
    equity = max(0.0, perp) + spot
    bot["target_equity"] = round(equity, 4)
    return equity


def target_sizing_equity(bot: dict[str, Any]) -> float:
    """Leader total equity used for copy ratio (always recompute from av+spot)."""
    return _recompute_target_equity(bot)


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
    """Flatten paper leftovers when target book is flat or perp AV is gone.

    Spot USDC is part of sizing equity, but must NOT block flatten when the
    leader's perp book is empty — otherwise leftovers stick forever.
    """
    if not (bot.get("positions") or {}):
        return False
    if _target_snap_flat(snap):
        return True
    try:
        av = float(bot.get("target_av") or 0)
    except (TypeError, ValueError):
        av = 0.0
    empty_thr = target_empty_av()
    # Snap missing + perp AV ~0 → clear zombies even if spot keeps ratio > 0.
    if snap is None and av < empty_thr:
        return True
    # Total sizing denom also gone (no spot either).
    if ratio <= 0 and av < empty_thr:
        return True
    return False


def _cache_target_positions(bot: dict[str, Any], snap: dict[str, Any]) -> None:
    """Update leader book / leverage maps from a clearinghouse snap."""
    lev_map: dict[str, float] = {}
    target_pos: dict[str, dict[str, float]] = {}
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
            pass
        try:
            szi = float(p.get("szi") or 0)
        except (TypeError, ValueError):
            continue
        if abs(szi) < 1e-16:
            continue
        row: dict[str, float] = {"sz": szi}
        try:
            if p.get("entryPx") is not None:
                row["entry_px"] = float(p["entryPx"])
        except (TypeError, ValueError):
            pass
        try:
            if p.get("lev") is not None:
                row["leverage"] = float(p["lev"])
        except (TypeError, ValueError):
            pass
        target_pos[c] = row
    bot["target_positions"] = target_pos
    if lev_map:
        prev = bot.get("target_lev_by_coin")
        if isinstance(prev, dict):
            prev.update(lev_map)
            bot["target_lev_by_coin"] = prev
        else:
            bot["target_lev_by_coin"] = lev_map


def _cache_target_meta(
    bot: dict[str, Any],
    snap: dict[str, Any] | None,
    *,
    update_av: bool = True,
) -> None:
    if not snap:
        return
    if update_av:
        try:
            av = float(snap.get("account_value") or 0)
        except (TypeError, ValueError):
            av = 0.0
        # Always record perp AV (incl. 0) so empty wallets are detectable.
        bot["target_av"] = av
        bot["target_av_at"] = time.time()
        _recompute_target_equity(bot)
    _cache_target_positions(bot, snap)


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
        # Same address; spot/EVM USDC does not restore *perp-empty* health
        # (sizing already uses total equity separately).
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

    try:
        te = _recompute_target_equity(bot) if av is not None else None
    except (TypeError, ValueError):
        te = (av or 0.0) + (spot or 0.0) if av is not None else None

    return {
        "status": status,
        "label": label,
        "ok": status == "ok",
        "empty": empty,
        "inactive": inactive,
        "target_av": None if av is None else round(av, 2),
        "target_equity": None if te is None else round(float(te), 2),
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
    """Poll target clearinghouse + recent fill age; tag empty/inactive bots.

    Bots that share an address (e.g. K paper + O live twin) fetch HL once and
    reuse the same perp/spot/EVM snapshot so UI labels stay consistent.
    """
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

        by_addr: dict[str, list[dict[str, Any]]] = {}
        for bot in (book.get("bots") or {}).values():
            if not isinstance(bot, dict):
                continue
            addr = str(bot.get("address") or "").strip()
            if not addr:
                continue
            by_addr.setdefault(addr.lower(), []).append(bot)

        for addr_l, bots in by_addr.items():
            addr = str(bots[0].get("address") or addr_l).strip()
            for bot in bots:
                if bot.get("target_watched_at") is None:
                    bot["target_watched_at"] = now

            snap = None
            spot = None
            evm: float | None = None
            seeded_last: float | None = None
            label = ",".join(str(b.get("id") or "") for b in bots) or addr[:10]

            try:
                snap = hl_snapshot_positions(addr)
            except Exception as exc:
                logger.warning("target health AV %s: %s", label, exc)
            try:
                spot = hl_snapshot_spot(addr, fill_limit=20)
            except Exception as exc:
                logger.warning("target health spot %s: %s", label, exc)
                try:
                    spot = {
                        "usdc": hl_snapshot_spot_usdc(addr),
                        "balances": [],
                        "recent_fills": [],
                    }
                except Exception as exc2:
                    logger.warning("target health spot usdc %s: %s", label, exc2)
            try:
                evm = hl_snapshot_hyperevm_usdc(addr)
            except Exception as exc:
                logger.warning("target health evm %s: %s", label, exc)
                evm = None

            # Seed last-fill once per address if no twin has seen a WS fill yet.
            if all(b.get("target_last_fill_at") is None for b in bots):
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
                        seeded_last = latest
                except Exception as exc:
                    logger.debug("target health fills %s: %s", label, exc)

            # Unify last-fill across twins (max of existing + seed).
            last_fill = seeded_last
            for bot in bots:
                try:
                    cur = float(bot["target_last_fill_at"]) if bot.get("target_last_fill_at") is not None else None
                except (TypeError, ValueError):
                    cur = None
                if cur is not None and (last_fill is None or cur > last_fill):
                    last_fill = cur

            for bot in bots:
                if snap is not None:
                    _cache_target_meta(bot, snap)
                if spot is not None:
                    _apply_spot_snapshot(bot, spot)
                if evm is None:
                    bot["target_evm_usdc"] = None
                else:
                    bot["target_evm_usdc"] = round(float(evm), 4)
                bot["target_evm_at"] = now
                if last_fill is not None:
                    bot["target_last_fill_at"] = last_fill

                prev = (
                    (bot.get("target_health") or {}).get("status")
                    if isinstance(bot.get("target_health"), dict)
                    else None
                )
                health = _compute_target_health(bot)
                bot["target_health"] = health
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
                else:
                    # Even if leader still has other coins, flatten legs they exited.
                    closed = _reconcile_flat_target_coins(bot, snap, mids, cfg)
                    if closed:
                        logger.warning(
                            "HL per-coin reconcile %s: closed %s paper rows",
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


def _target_coin_szi(snap: dict[str, Any] | None, coin: str) -> float | None:
    """Signed target size for coin from a clearinghouse snap, or None if unknown."""
    if not snap or not coin:
        return None
    want = str(coin).strip().upper()
    found = False
    total = 0.0
    for p in snap.get("positions") or []:
        if not isinstance(p, dict):
            continue
        if str(p.get("coin") or "").strip().upper() != want:
            continue
        found = True
        try:
            total += float(p.get("szi") or 0)
        except (TypeError, ValueError):
            continue
    return total if found else (0.0 if snap.get("positions") is not None else None)


def _target_had_prior_inventory(
    snap: dict[str, Any] | None,
    coin: str,
    target_delta: float,
) -> bool | None:
    """Infer whether target already had this coin before the fill (snap is post-fill).

    Returns True/False when snap is usable, None when unknown.
    """
    post = _target_coin_szi(snap, coin)
    if post is None:
        return None
    try:
        pre = float(post) - float(target_delta)
    except (TypeError, ValueError):
        return None
    return abs(pre) > 1e-9


def _leader_pre_post_sz(
    *,
    start_position: float | None,
    target_delta: float,
    snap: dict[str, Any] | None,
    coin: str,
) -> tuple[float | None, float | None]:
    """Leader signed size before/after this fill.

    Prefer HL fill ``startPosition``; else infer from post-fill clearinghouse snap.
    """
    try:
        delta = float(target_delta)
    except (TypeError, ValueError):
        return None, None
    if start_position is not None:
        try:
            pre = float(start_position)
        except (TypeError, ValueError):
            pre = None
        if pre is not None:
            return pre, pre + delta
    post = _target_coin_szi(snap, coin)
    if post is None:
        return None, None
    return float(post) - delta, float(post)


def _stamp_leader_start_positions(
    fresh: list[dict[str, Any]],
    snap: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Chronological stamp of startPosition from snap when HL omits it.

    Shared by paper + live_only so Bitget fresh-open gate never has to guess
    from unreliable Open* dir labels.
    """

    def _ft_key(it: dict[str, Any]) -> tuple[float, str]:
        return (_fill_time_epoch(it.get("fill_time")) or 0.0, str(it.get("tid") or ""))

    fresh_sorted = sorted(fresh, key=_ft_key)
    batch_delta: dict[str, float] = {}
    for it in fresh_sorted:
        c = str(it.get("coin") or "")
        try:
            batch_delta[c] = batch_delta.get(c, 0.0) + float(it.get("target_delta") or 0)
        except (TypeError, ValueError):
            continue
    leader_pos: dict[str, float] = {}
    if snap is not None:
        for c, dlt in batch_delta.items():
            post = _target_coin_szi(snap, c)
            if post is not None:
                leader_pos[c] = float(post) - float(dlt)

    chain = dict(leader_pos)
    for item in fresh_sorted:
        c = str(item.get("coin") or "")
        if item.get("start_position") is None and c in chain:
            item["start_position"] = chain[c]
        try:
            td = float(item.get("target_delta") or 0)
            sp = item.get("start_position")
            if sp is not None:
                chain[c] = float(sp) + td
            elif c in chain:
                chain[c] = float(chain[c]) + td
        except (TypeError, ValueError):
            pass
    return fresh_sorted


def _reduce_sz_by_leader_pct(
    old_sz: float,
    *,
    pre: float,
    post: float,
) -> float:
    """Mirror leader reduce by remaining fraction (Dextrabot-style).

    If leader goes from 10 → 5, we keep 50% of our size. Full flat / flip-through-zero
    on their book → we flatten (opposite open is a separate fill).
    """
    if abs(old_sz) < 1e-16:
        return 0.0
    if abs(pre) < 1e-16:
        return old_sz
    # Crossed or reached flat on leader → close our leg fully.
    if abs(post) < 1e-16 or pre * post <= 0:
        return 0.0
    remain = abs(post) / abs(pre)
    remain = max(0.0, min(1.0, remain))
    return old_sz * remain


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
    """equity / target_equity — sizing basis; optional soft-TP size_mult (e.g. 0.5).

    Denominator is perp AV (main+xyz) + Core spot USDC so spot↔perp transfers
    do not reprice the copy ratio. ``target_av`` alone stays for empty health.
    """
    _recompute_bot(bot)
    eq = float(bot.get("equity") or bot.get("balance") or cfg["bot_balance"])
    denom = target_sizing_equity(bot)
    if denom <= 1e-9 or eq <= 0:
        bot["copy_ratio"] = 0.0
        return 0.0
    try:
        mult = float(size_mult)
    except (TypeError, ValueError):
        mult = 1.0
    if mult <= 0:
        bot["copy_ratio"] = 0.0
        return 0.0
    ratio = (eq / denom) * mult
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
    """Leader leverage for coin — never Bitget overlay (that echoes venue lev)."""
    lev_map = bot.get("target_lev_by_coin") if isinstance(bot.get("target_lev_by_coin"), dict) else {}
    want = _scope_keys_for_coin(coin)
    raw = None
    for key, val in lev_map.items():
        if want & _scope_keys_for_coin(str(key)):
            raw = val
            break
    if raw is None:
        tpos = bot.get("target_positions") if isinstance(bot.get("target_positions"), dict) else {}
        for tcoin, tp in tpos.items():
            if not isinstance(tp, dict) or tp.get("leverage") is None:
                continue
            if want & _scope_keys_for_coin(str(tcoin)):
                raw = tp.get("leverage")
                break
    if raw is None:
        raw = 10.0
    return _adjusted_leverage(raw, cfg.get("leverage_adjustment", 1.0), coin)


def _pos_notional(pos: dict[str, Any], px: float | None = None) -> float:
    try:
        sz = abs(float(pos.get("sz") or 0))
    except (TypeError, ValueError):
        return 0.0
    if px is None or px <= 0:
        try:
            px = float(pos.get("mark_px") or pos.get("entry_px") or 0)
        except (TypeError, ValueError):
            px = 0.0
    return sz * float(px or 0)


def _gross_notional(
    bot: dict[str, Any],
    *,
    exclude_key: str | None = None,
    px_by_key: dict[str, float] | None = None,
) -> float:
    total = 0.0
    for key, pos in (bot.get("positions") or {}).items():
        if exclude_key and key == exclude_key:
            continue
        if not isinstance(pos, dict):
            continue
        override = None
        if px_by_key and key in px_by_key:
            override = px_by_key[key]
        total += _pos_notional(pos, override)
    return total


def _max_notional(bot: dict[str, Any], lev: int, cfg: dict[str, Any]) -> float:
    """Gross notional ceiling for the whole bot: equity × leverage."""
    _recompute_bot(bot)
    eq = float(bot.get("equity") or bot.get("balance") or cfg["bot_balance"])
    return max(0.0, eq * float(max(1, lev)))


def _clip_sz_to_notional(sz: float, px: float, max_notional: float) -> float:
    """Hard-cap |sz|*px ≤ max_notional (tiny haircut avoids float overshoot)."""
    if px <= 0 or max_notional <= 0:
        return 0.0
    cap = (max_notional * 0.999999) / px
    if abs(sz) <= cap + 1e-15:
        return sz
    return math.copysign(cap, sz)


def _margin_px_for_clip(px: float, mark: float, new_sz: float) -> float:
    """Conservative price so mark≥fill cannot silently breach notional cap."""
    fill = float(px or 0)
    mid = float(mark or 0)
    if fill <= 0:
        return max(mid, 0.0)
    if mid <= 0:
        return fill
    # Higher px → smaller |sz| for the same notional budget (long and short).
    return max(fill, mid)


def _enforce_notional_caps(
    bot: dict[str, Any],
    mids: dict[str, float],
    cfg: dict[str, Any],
) -> None:
    """If mark moves notionals above equity×lev, shrink positions pro-rata."""
    positions = bot.get("positions") or {}
    if not positions:
        return
    # Use the max leverage among open legs (fallback 10).
    lev = 1
    for pos in positions.values():
        if not isinstance(pos, dict):
            continue
        try:
            lev = max(lev, int(float(pos.get("leverage") or 0) or 0))
        except (TypeError, ValueError):
            continue
    if lev < 1:
        lev = 10
    cap = _max_notional(bot, lev, cfg)
    if cap <= 0:
        return
    gross = 0.0
    notionals: dict[str, float] = {}
    for key, pos in positions.items():
        if not isinstance(pos, dict):
            continue
        coin = str(pos.get("coin") or "")
        mark = _mid_for_coin(mids, coin) or float(pos.get("mark_px") or pos.get("entry_px") or 0)
        n = _pos_notional(pos, mark)
        notionals[key] = n
        gross += n
    if gross <= cap * 1.000001:
        return
    scale = cap / gross if gross > 0 else 0.0
    for key, pos in list(positions.items()):
        if not isinstance(pos, dict):
            continue
        try:
            sz = float(pos.get("sz") or 0)
        except (TypeError, ValueError):
            continue
        if abs(sz) < 1e-16:
            continue
        pos["sz"] = sz * scale
        coin = str(pos.get("coin") or "")
        mark = _mid_for_coin(mids, coin) or float(pos.get("mark_px") or pos.get("entry_px") or 0)
        if mark > 0:
            _mark_one(pos, mark)
    _recompute_bot(bot)


def _merge_tids(
    trigger_tid: str | None, extra_tids: list[str] | None = None
) -> list[str]:
    out: list[str] = []
    if trigger_tid is not None and str(trigger_tid).strip():
        out.append(str(trigger_tid))
    for t in extra_tids or []:
        s = str(t or "").strip()
        if s and s not in out:
            out.append(s)
    return out


def _sim_sz_after_fill(old_sz: float, tdelta: float, fill_dir: str | None) -> float:
    """Advance a coarse local-size sim used only by flat-entry coalesce.

    Close* → treat as flat so a later reopen burst in the same WS batch can
    coalesce. Partial reduces without Close stay non-flat (no merge).
    """
    dir_l = str(fill_dir or "").strip().lower()
    is_close = "close" in dir_l and "open" not in dir_l
    is_open = "open" in dir_l and "close" not in dir_l
    if abs(tdelta) < 1e-16:
        return old_sz
    if abs(old_sz) < 1e-16:
        return float(tdelta)
    if is_close:
        return 0.0
    if is_open and old_sz * tdelta < 0:
        # Flip labeled as Open opposite — flat then enter the new side.
        return float(tdelta)
    if old_sz * tdelta < 0:
        # Unlabeled reduce: keep non-flat so we do not coalesce mid-position.
        return old_sz
    return old_sz + float(tdelta)


def _coalesce_flat_entry_fills(
    bot: dict[str, Any], items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge consecutive same-sign clips while local flat (incl. intra-batch).

    A tiny first Open (common in HL bursts) can sit under min_notional after
    sizing on a small seat. Silently dropping it left the rest of the burst as
    orphan_add — K opened, O stayed flat. Coalesce the entry burst so the open
    is sized on the full clip sum.

    Tracks simulated size through the batch so Close→Open in one event still
    merges the reopen clips (Bugbot: coalesce must not only use pre-batch pos).
    """
    sim_sz: dict[str, float] = {}
    for key, pos in (bot.get("positions") or {}).items():
        if not isinstance(pos, dict):
            continue
        coin = str(pos.get("coin") or "")
        if not coin and isinstance(key, str) and ":" in key:
            coin = key.split(":", 1)[-1]
        if not coin:
            continue
        try:
            sim_sz[coin] = float(pos.get("sz") or 0)
        except (TypeError, ValueError):
            continue

    out: list[dict[str, Any]] = []
    i = 0
    n = len(items)
    while i < n:
        item = items[i]
        coin = str(item.get("coin") or "")
        old_sz = float(sim_sz.get(coin) or 0.0)
        try:
            td0 = float(item.get("target_delta") or 0)
        except (TypeError, ValueError):
            out.append(item)
            i += 1
            continue
        if abs(old_sz) > 1e-16 or abs(td0) < 1e-16:
            out.append(item)
            sim_sz[coin] = _sim_sz_after_fill(old_sz, td0, item.get("dir"))
            i += 1
            continue

        # Leader already in inventory → do not fuse Add clips into a fake open.
        try:
            sp0 = item.get("start_position")
            if sp0 is not None and abs(float(sp0)) > 1e-9:
                out.append(item)
                sim_sz[coin] = _sim_sz_after_fill(old_sz, td0, item.get("dir"))
                i += 1
                continue
        except (TypeError, ValueError):
            pass

        j = i + 1
        total_td = td0
        abs_td = abs(td0)
        px_weighted = abs_td * float(item.get("px") or 0)
        tids: list[str] = []
        if item.get("tid") is not None and str(item.get("tid")).strip():
            tids.append(str(item.get("tid")))
        open_dir = item.get("dir")
        while j < n:
            nxt = items[j]
            if str(nxt.get("coin") or "") != coin:
                break
            dir_n = str(nxt.get("dir") or "").strip().lower()
            if "close" in dir_n and "open" not in dir_n:
                break
            try:
                td = float(nxt.get("target_delta") or 0)
            except (TypeError, ValueError):
                break
            if abs(td) < 1e-16:
                j += 1
                continue
            if td * total_td < 0:
                break
            total_td += td
            abs_td += abs(td)
            px_weighted += abs(td) * float(nxt.get("px") or 0)
            if nxt.get("tid") is not None and str(nxt.get("tid")).strip():
                tids.append(str(nxt.get("tid")))
            nd = str(nxt.get("dir") or "").strip().lower()
            if "open" in nd and "close" not in nd:
                open_dir = nxt.get("dir")
            j += 1

        if j == i + 1:
            out.append(item)
            sim_sz[coin] = _sim_sz_after_fill(old_sz, td0, item.get("dir"))
            i += 1
            continue

        merged = dict(item)
        merged["target_delta"] = total_td
        merged["px"] = (px_weighted / abs_td) if abs_td > 1e-16 else float(item.get("px") or 0)
        merged["coalesced_n"] = j - i
        merged["extra_tids"] = tids
        if open_dir:
            merged["dir"] = open_dir
        out.append(merged)
        sim_sz[coin] = float(total_td)
        i = j
    return out


def _sync_paper_to_mirror_sibling(
    book: dict[str, Any],
    bot: dict[str, Any],
    mids: dict[str, float],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Dextrabot Copy-Current: align live twin paper to sibling × equity ratio.

    Follower_sz = sibling_sz × (our_equity / sibling_equity). Used for O←K so a
    late/flat twin does not open a stub from mid-book Add clips.

    Caps legs against a running gross-notional budget (same pass), and never
    force-closes a twin-held coin just because the scaled leg is under min_notional.
    """
    twin_id = str(bot.get("mirror_of") or "").strip()
    if not twin_id:
        return []
    twin = (book.get("bots") or {}).get(twin_id)
    if not isinstance(twin, dict):
        return []

    _recompute_bot(bot)
    _recompute_bot(twin)
    our_eq = float(bot.get("equity") or bot.get("balance") or 0)
    twin_eq = float(twin.get("equity") or twin.get("balance") or 0)
    if our_eq <= 1e-9 or twin_eq <= 1e-9:
        return []

    ratio = our_eq / twin_eq
    allow = _bot_allow_coins(bot)
    old = dict(bot.get("positions") or {})
    min_n = float(cfg.get("min_notional") or 0)

    # Phase 1: raw wants + dust keys (twin still holds; do not treat as twin-flat).
    raw: list[dict[str, Any]] = []
    dust_keys: set[str] = set()
    twin_keys: set[str] = set()
    for pos in (twin.get("positions") or {}).values():
        if not isinstance(pos, dict):
            continue
        coin = str(pos.get("coin") or "").upper()
        if not coin or not _coin_allowed(coin, allow):
            continue
        try:
            t_sz = float(pos.get("sz") or 0)
        except (TypeError, ValueError):
            continue
        if abs(t_sz) < 1e-16:
            continue
        key = f"{bot.get('id')}:{coin}"
        twin_keys.add(key)
        want = t_sz * ratio
        mid = _mid_for_coin(mids, coin) or float(
            pos.get("mark_px") or pos.get("entry_px") or 0
        )
        if abs(want) * float(mid or 0) < min_n:
            dust_keys.add(key)
            continue
        try:
            # Always leader lev map / target book — never our Bitget echo.
            lev = int(_lev_for_coin(bot, coin, cfg) or 1)
        except (TypeError, ValueError):
            lev = int(_lev_for_coin(bot, coin, cfg) or 1)
        raw.append(
            {
                "key": key,
                "coin": coin,
                "sz": want,
                "mid": float(mid or 0),
                "lev": max(1, lev),
                "entry_px": float(pos.get("entry_px") or mid or 0),
                "twin_sz": t_sz,
            }
        )

    # Phase 2: clip each leg against remaining gross budget (include same-pass reserved).
    raw.sort(key=lambda x: abs(float(x["sz"])) * float(x["mid"]), reverse=True)
    desired: dict[str, dict[str, Any]] = {}
    reserved = 0.0
    # Notionals we will keep (dust twin legs already on our book).
    for dk in dust_keys:
        op = old.get(dk)
        if isinstance(op, dict):
            reserved += _pos_notional(op)

    rows: list[dict[str, Any]] = []
    fills = list(bot.get("fills") or [])

    for item in raw:
        key = str(item["key"])
        coin = str(item["coin"])
        mid = float(item["mid"] or 0)
        lev = int(item["lev"])
        want = float(item["sz"])
        total_cap = _max_notional(bot, lev, cfg)
        # `reserved` already includes kept dust legs + previously accepted desired.
        max_n = max(0.0, total_cap - reserved)
        clipped = _clip_sz_to_notional(want, mid, max_n)
        if abs(clipped) * mid < min_n:
            dust_keys.add(key)
            fills.insert(
                0,
                {
                    "id": str(uuid.uuid4())[:8],
                    "action": "signal",
                    "skipped": True,
                    "reason": "twin_dust",
                    "skip_reason": "twin_dust",
                    "source": bot.get("id"),
                    "coin": coin,
                    "px": mid,
                    "our_sz": abs(clipped),
                    "target_sz": item.get("twin_sz"),
                    "copy_ratio": round(ratio, 10),
                    "twin_of": twin_id,
                    "min_notional": min_n,
                    "ts": _now(),
                },
            )
            rows.append(fills[0])
            logger.info(
                "HL twin_dust bot=%s←%s coin=%s want=%s mid=%s",
                bot.get("id"),
                twin_id,
                coin,
                want,
                mid,
            )
            continue
        desired[key] = {
            "key": key,
            "source": bot.get("id"),
            "coin": coin,
            "sz": clipped,
            "entry_px": float(item["entry_px"] or mid),
            "copy_ratio": round(ratio, 10),
            "leverage": lev,
            "target_address": bot.get("address"),
            "twin_of": twin_id,
            "opened_at": (old.get(key) or {}).get("opened_at") or _now(),
            "u_pnl": 0.0,
            "mark_px": mid or None,
        }
        reserved += abs(clipped) * mid

    new_positions: dict[str, dict[str, Any]] = {}

    # Preserve dust twin legs we already hold; log skip if twin has dust and we are flat.
    for key in dust_keys:
        op = old.get(key)
        coin = key.split(":", 1)[-1] if ":" in key else key
        mid = _mid_for_coin(mids, coin)
        if isinstance(op, dict) and abs(float(op.get("sz") or 0)) > 1e-16:
            if mid:
                _mark_one(op, mid)
            new_positions[key] = op
            continue
        fills.insert(
            0,
            {
                "id": str(uuid.uuid4())[:8],
                "action": "signal",
                "skipped": True,
                "reason": "twin_dust",
                "skip_reason": "twin_dust",
                "source": bot.get("id"),
                "coin": coin,
                "px": mid,
                "our_sz": 0,
                "copy_ratio": round(ratio, 10),
                "twin_of": twin_id,
                "min_notional": min_n,
                "ts": _now(),
            },
        )
        rows.append(fills[0])

    for key, pos in old.items():
        if key in desired or key in dust_keys:
            continue
        coin = str(pos.get("coin") or "").upper()
        mid = _mid_for_coin(mids, coin) or float(pos.get("mark_px") or pos.get("entry_px") or 0)
        try:
            old_sz = float(pos.get("sz") or 0)
        except (TypeError, ValueError):
            old_sz = 0.0
        if abs(old_sz) < 1e-16:
            continue
        pnl = _realize(bot, pos, float(mid or 0), abs(old_sz))
        side = "buy" if old_sz < 0 else "sell"
        pos_side = "short" if old_sz < 0 else "long"
        row = {
            "id": str(uuid.uuid4())[:8],
            "action": "close",
            "reason": "twin_copy_current",
            "source": bot.get("id"),
            "coin": coin,
            "px": mid,
            "our_sz": abs(old_sz),
            "notional": abs(old_sz) * float(mid or 0),
            "realized_pnl": pnl,
            "side": side,
            "pos": pos_side,
            "twin_of": twin_id,
            "ts": _now(),
        }
        rows.append(row)
        fills.insert(0, row)

    for key, want in desired.items():
        coin = str(want.get("coin") or "")
        mid = float(want.get("mark_px") or 0) or _mid_for_coin(mids, coin) or float(
            want.get("entry_px") or 0
        )
        new_sz = float(want.get("sz") or 0)
        old_pos = old.get(key)
        old_sz = float(old_pos.get("sz") or 0) if isinstance(old_pos, dict) else 0.0
        if abs(new_sz - old_sz) * float(mid or 0) < min_n * 0.5:
            if isinstance(old_pos, dict) and abs(old_sz) > 1e-16:
                _mark_one(old_pos, mid)
                new_positions[key] = old_pos
            elif abs(new_sz) > 1e-16:
                want["mark_px"] = mid
                _mark_one(want, mid)
                new_positions[key] = want
            continue

        if abs(old_sz) > 1e-16 and old_sz * new_sz < 0:
            pnl = _realize(bot, old_pos, mid, abs(old_sz))
            rows.append(
                {
                    "id": str(uuid.uuid4())[:8],
                    "action": "close",
                    "reason": "twin_copy_current",
                    "source": bot.get("id"),
                    "coin": coin,
                    "px": mid,
                    "our_sz": abs(old_sz),
                    "realized_pnl": pnl,
                    "pos": "long" if old_sz > 0 else "short",
                    "twin_of": twin_id,
                    "ts": _now(),
                }
            )
            fills.insert(0, rows[-1])
            old_sz = 0.0
            old_pos = None

        if abs(new_sz) < 1e-16:
            if abs(old_sz) > 1e-16 and isinstance(old_pos, dict):
                pnl = _realize(bot, old_pos, mid, abs(old_sz))
                rows.append(
                    {
                        "id": str(uuid.uuid4())[:8],
                        "action": "close",
                        "reason": "twin_copy_current",
                        "source": bot.get("id"),
                        "coin": coin,
                        "px": mid,
                        "our_sz": abs(old_sz),
                        "realized_pnl": pnl,
                        "pos": "long" if old_sz > 0 else "short",
                        "twin_of": twin_id,
                        "ts": _now(),
                    }
                )
                fills.insert(0, rows[-1])
            continue

        delta = new_sz - old_sz
        if abs(old_sz) < 1e-16:
            action = "open"
            qty = abs(new_sz)
        elif abs(new_sz) + 1e-12 < abs(old_sz):
            action = "reduce"
            qty = abs(old_sz) - abs(new_sz)
            if isinstance(old_pos, dict):
                _realize(bot, old_pos, mid, qty)
        else:
            action = "increase"
            qty = abs(new_sz) - abs(old_sz)

        side = "buy" if delta > 0 else "sell"
        pos_side = "long" if new_sz > 0 else "short"
        if action == "open" or not isinstance(old_pos, dict):
            pos = dict(want)
            pos["sz"] = new_sz
            pos["entry_px"] = mid
            pos["opened_at"] = _now()
        else:
            pos = dict(old_pos)
            if action == "increase":
                old_entry = float(pos.get("entry_px") or mid)
                old_abs = abs(old_sz)
                pos["entry_px"] = (old_entry * old_abs + mid * qty) / (old_abs + qty)
            pos["sz"] = new_sz
            pos["leverage"] = want.get("leverage")
            pos["copy_ratio"] = want.get("copy_ratio")
            pos["twin_of"] = twin_id
        _mark_one(pos, mid)
        new_positions[key] = pos
        row: dict[str, Any] = {
            "id": str(uuid.uuid4())[:8],
            "action": action,
            "reason": "twin_copy_current",
            "source": bot.get("id"),
            "coin": coin,
            "px": mid,
            "our_sz": qty,
            "notional": qty * float(mid or 0),
            "side": side,
            "pos": pos_side,
            "copy_ratio": round(ratio, 10),
            "twin_of": twin_id,
            "ts": _now(),
        }
        tpos = (twin.get("positions") or {}).get(f"{twin_id}:{coin}")
        if isinstance(tpos, dict):
            try:
                row["twin_sz"] = float(tpos.get("sz") or 0)
            except (TypeError, ValueError):
                pass
        rows.append(row)
        fills.insert(0, row)
        logger.info(
            "HL twin_copy_current bot=%s←%s coin=%s action=%s our=%s ratio=%.6g",
            bot.get("id"),
            twin_id,
            coin,
            action,
            new_sz,
            ratio,
        )

    if not rows and set(new_positions.keys()) == set(old.keys()):
        return []

    bot["positions"] = new_positions
    bot["fills"] = fills[:300]
    _recompute_bot(bot)
    return rows


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
    fill_dir: str | None = None,
    target_snap: dict[str, Any] | None = None,
    start_position: float | None = None,
    extra_tids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Apply one proportional fill at market px; enforce equity×lev notional cap.

    Live-safe rules (Dextrabot/Legend defaults):
    - Local flat + leader already had inventory → orphan skip (no stub from Add).
    - HL dir text is unreliable (Open Short with startPos≠0 is still an add).
    - Reduces scale our size by leader remaining fraction (not raw fill×ratio).
    - Never cross through zero into the opposite side on the same fill.
    """
    allow = _bot_allow_coins(bot)
    if not _coin_allowed(coin, allow):
        return []

    try:
        tdelta = float(target_delta)
    except (TypeError, ValueError):
        return []
    if abs(tdelta) < 1e-16:
        return []

    our_delta = tdelta * float(ratio)
    key = f"{bot.get('id')}:{coin}"
    positions = bot.setdefault("positions", {})
    old = positions.get(key)
    old_sz = float(old.get("sz") or 0) if isinstance(old, dict) else 0.0
    raw_new = old_sz + our_delta
    mark = _mid_for_coin(mids, coin) or px
    dir_l = str(fill_dir or "").strip().lower()
    is_add_dir = "add" in dir_l
    is_open_dir = "open" in dir_l and "close" not in dir_l
    pre_sz, post_sz = _leader_pre_post_sz(
        start_position=start_position,
        target_delta=tdelta,
        snap=target_snap,
        coin=coin,
    )

    # Prefer leader book to classify reduce vs increase when available.
    if pre_sz is not None and post_sz is not None:
        increasing = abs(post_sz) > abs(pre_sz) + 1e-12 or (
            abs(pre_sz) < 1e-16 and abs(post_sz) > 1e-16
        )
        decreasing = abs(post_sz) + 1e-12 < abs(pre_sz) or (
            abs(pre_sz) > 1e-16 and abs(post_sz) < 1e-16
        )
    else:
        increasing = abs(raw_new) > abs(old_sz) + 1e-12 or (
            abs(old_sz) < 1e-16 and abs(raw_new) > 1e-16
        )
        decreasing = abs(raw_new) + 1e-12 < abs(old_sz) or (
            abs(old_sz) > 1e-16 and abs(raw_new) < 1e-16
        )

    # Orphan add: flat locally but leader already had inventory.
    # Trust startPosition / had_prior over HL Open* labels.
    if abs(old_sz) < 1e-16 and increasing:
        had_prior = (
            abs(pre_sz) > 1e-9
            if pre_sz is not None
            else _target_had_prior_inventory(target_snap, coin, tdelta)
        )
        orphan = bool(is_add_dir) or (had_prior is True) or (
            had_prior is None
            and not is_open_dir
            and bool(dir_l)
            and ("close" in dir_l or "add" in dir_l)
        )
        if not orphan and had_prior is None and not dir_l:
            post = post_sz if post_sz is not None else _target_coin_szi(target_snap, coin)
            if post is not None and abs(post) > abs(tdelta) + 1e-9:
                orphan = True
        if is_open_dir and had_prior is True:
            orphan = True
        if orphan and _bot_copy_current(bot):
            # Dextrabot Copy Current: enter full leader leg × ratio, not Add stub.
            leader_sz = (
                post_sz
                if post_sz is not None
                else _target_coin_szi(target_snap, coin)
            )
            if leader_sz is not None and abs(float(leader_sz)) > 1e-16:
                total_cap = _max_notional(bot, lev, cfg)
                used_others = _gross_notional(bot, exclude_key=key)
                max_n = max(0.0, total_cap - used_others)
                margin_px = _margin_px_for_clip(px, mark, float(leader_sz) * float(ratio))
                new_sz = _clip_sz_to_notional(
                    float(leader_sz) * float(ratio), margin_px, max_n
                )
                if abs(new_sz) * float(px or 0) >= float(cfg.get("min_notional") or 0):
                    # Reuse normal open path by jumping to apply with forced size.
                    our_delta = new_sz - old_sz
                    raw_new = new_sz
                    increasing = True
                    decreasing = False
                    orphan = False
                    fill_dir = fill_dir or "Copy Current"
                    logger.info(
                        "HL copy_current bot=%s coin=%s leader=%s our=%s ratio=%.6g",
                        bot.get("id"),
                        coin,
                        leader_sz,
                        new_sz,
                        ratio,
                    )
                else:
                    row = {
                        "id": str(uuid.uuid4())[:8],
                        "action": "signal",
                        "skipped": True,
                        "reason": "copy_current_dust",
                        "skip_reason": "copy_current_dust",
                        "source": bot.get("id"),
                        "coin": coin,
                        "px": px,
                        "our_sz": abs(new_sz),
                        "target_delta": tdelta,
                        "leader_sz": leader_sz,
                        "copy_ratio": round(ratio, 10),
                        "leverage": lev,
                        "dir": fill_dir,
                        "target_tid": trigger_tid,
                        "fill_time": fill_time,
                        "ts": _now(),
                    }
                    fills = list(bot.get("fills") or [])
                    fills.insert(0, row)
                    bot["fills"] = fills[:300]
                    return []
            # fall through to orphan skip if no leader size
        if orphan:
            row = {
                "id": str(uuid.uuid4())[:8],
                "action": "signal",
                "skipped": True,
                "reason": "orphan_add",
                "skip_reason": "orphan_add",
                "source": bot.get("id"),
                "coin": coin,
                "px": px,
                "our_sz": 0,
                "target_delta": tdelta,
                "copy_ratio": round(ratio, 10),
                "leverage": lev,
                "dir": fill_dir,
                "target_tid": trigger_tid,
                "fill_time": fill_time,
                "ts": _now(),
            }
            fills = list(bot.get("fills") or [])
            fills.insert(0, row)
            bot["fills"] = fills[:300]
            logger.info(
                "HL skip orphan_add bot=%s coin=%s tdelta=%s dir=%s had_prior=%s",
                bot.get("id"),
                coin,
                tdelta,
                fill_dir,
                had_prior,
            )
            return []

    total_cap = _max_notional(bot, lev, cfg)
    used_others = _gross_notional(bot, exclude_key=key)
    max_n = max(0.0, total_cap - used_others)
    margin_px = _margin_px_for_clip(px, mark, raw_new) if (
        increasing or (old_sz * raw_new < 0)
    ) else float(px or mark or 0)

    if abs(old_sz) > 1e-16 and decreasing:
        # Percentage reduce vs leader pre→post (not fill×ratio absolute).
        if pre_sz is not None and post_sz is not None:
            new_sz = _reduce_sz_by_leader_pct(old_sz, pre=pre_sz, post=post_sz)
            # Leader flipped through zero into opposite: flatten only unless Open*.
            if (
                pre_sz * post_sz < 0
                and abs(post_sz) > 1e-16
                and is_open_dir
            ):
                new_sz = _clip_sz_to_notional(float(post_sz) * float(ratio), margin_px, max_n)
        else:
            if old_sz > 0:
                new_sz = max(0.0, raw_new)
            elif old_sz < 0:
                new_sz = min(0.0, raw_new)
            else:
                new_sz = 0.0
    elif abs(old_sz) > 1e-16 and old_sz * raw_new < 0 and not increasing:
        new_sz = 0.0
    elif increasing:
        new_sz = _clip_sz_to_notional(raw_new, margin_px, max_n)
        if abs(new_sz - old_sz) < 1e-12:
            row = {
                "id": str(uuid.uuid4())[:8],
                "action": "signal",
                "skipped": True,
                "reason": "margin_cap",
                "source": bot.get("id"),
                "coin": coin,
                "px": px,
                "our_sz": 0,
                "target_delta": tdelta,
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
        # Flat local + non-increasing (e.g. close with no local pos) → no-op
        if abs(old_sz) < 1e-16:
            return []
        if old_sz > 0:
            new_sz = max(0.0, raw_new)
        else:
            new_sz = min(0.0, raw_new)

    # Dust residual after pct-reduce → flatten (untradeable / bad for live sync)
    if (
        abs(old_sz) > 1e-16
        and abs(new_sz) > 1e-16
        and abs(new_sz) * float(px or 0) < float(cfg.get("min_notional") or 0)
    ):
        new_sz = 0.0

    # Dust open — must be visible; silent drop caused O to miss the whole entry burst.
    if abs(old_sz) < 1e-16 and abs(new_sz) * float(px or 0) < float(
        cfg.get("min_notional") or 0
    ):
        row = {
            "id": str(uuid.uuid4())[:8],
            "action": "signal",
            "skipped": True,
            "reason": "dust_open",
            "skip_reason": "dust_open",
            "source": bot.get("id"),
            "coin": coin,
            "px": px,
            "our_sz": abs(new_sz),
            "notional": abs(new_sz) * float(px or 0),
            "target_delta": tdelta,
            "copy_ratio": round(ratio, 10),
            "leverage": lev,
            "dir": fill_dir,
            "min_notional": cfg.get("min_notional"),
            "target_tid": trigger_tid,
            "target_tids": _merge_tids(trigger_tid, extra_tids),
            "fill_time": fill_time,
            "ts": _now(),
        }
        fills = list(bot.get("fills") or [])
        fills.insert(0, row)
        bot["fills"] = fills[:300]
        logger.info(
            "HL skip dust_open bot=%s coin=%s our_sz=%s notional=%.4f min=%s dir=%s",
            bot.get("id"),
            coin,
            new_sz,
            abs(new_sz) * float(px or 0),
            cfg.get("min_notional"),
            fill_dir,
        )
        return []

    # No effective change
    if abs(new_sz - old_sz) < 1e-16:
        return []

    rows: list[dict[str, Any]] = []
    fills = list(bot.get("fills") or [])

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
            "target_delta": tdelta,
            "target_tid": trigger_tid,
            "target_tids": _merge_tids(trigger_tid, extra_tids),
            "target_address": bot.get("address"),
            "fill_time": fill_time,
            "side": side,
            "pos": pos_side,
            "ts": _now(),
            "max_notional": round(max_n, 4),
        }
        if fill_dir:
            out["dir"] = fill_dir
        if pre_sz is not None:
            out["leader_pre_sz"] = pre_sz
        if post_sz is not None:
            out["leader_post_sz"] = post_sz
        if realized is not None:
            out["realized_pnl"] = realized
        return out

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
            "target_equity": bot.get("target_equity"),
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
        red_pos = "long" if old_sz > 0 else "short"
        red_row = _row("reduce", closed, px, pnl, side, pos_side=red_pos)
        rows.append(red_row)
        fills.insert(0, red_row)
        old["sz"] = new_sz
        old["leverage"] = lev
        old["copy_ratio"] = round(ratio, 10)
        old["target_av"] = bot.get("target_av")
        old["target_equity"] = bot.get("target_equity")
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
        old["target_equity"] = bot.get("target_equity")
        _mark_one(old, mark)
        positions[key] = old

    bot["fills"] = fills[:300]
    _recompute_bot(bot)
    return rows


def _reconcile_flat_target_coins(
    bot: dict[str, Any],
    snap: dict[str, Any] | None,
    mids: dict[str, float],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Close paper legs whose coin is flat on the leader (per-coin reconcile).

    Does not open missing coins (no silent Copy Current). Live Bitget sync follows
    the resulting paper flat.
    """
    if not snap or not isinstance(bot.get("positions"), dict):
        return []
    allow = _bot_allow_coins(bot)
    rows: list[dict[str, Any]] = []
    fills = list(bot.get("fills") or [])
    positions = bot["positions"]
    for key, pos in list(positions.items()):
        if not isinstance(pos, dict):
            continue
        coin = str(pos.get("coin") or "").strip().upper()
        if not coin or not _coin_allowed(coin, allow):
            continue
        try:
            our_sz = float(pos.get("sz") or 0)
        except (TypeError, ValueError):
            continue
        if abs(our_sz) < 1e-16:
            positions.pop(key, None)
            continue
        t_sz = _target_coin_szi(snap, coin)
        if t_sz is None or abs(t_sz) > 1e-16:
            continue
        px = _mid_for_coin(mids, coin) or float(pos.get("mark_px") or pos.get("entry_px") or 0)
        if px <= 0:
            continue
        pnl = _realize(bot, pos, px, abs(our_sz))
        close_side = "sell" if our_sz > 0 else "buy"
        close_pos = "long" if our_sz > 0 else "short"
        row = {
            "id": str(uuid.uuid4())[:8],
            "action": "close",
            "source": bot.get("id"),
            "bot_id": bot.get("id"),
            "coin": coin,
            "px": px,
            "our_sz": abs(our_sz),
            "notional": abs(our_sz) * px,
            "realized_pnl": pnl,
            "side": close_side,
            "pos": close_pos,
            "reason": "reconcile_target_flat",
            "target_address": bot.get("address"),
            "leverage": pos.get("leverage") or _lev_for_coin(bot, coin, cfg),
            "ts": _now(),
        }
        rows.append(row)
        fills.insert(0, row)
        positions.pop(key, None)
        logger.info(
            "HL reconcile flat bot=%s coin=%s closed_sz=%s px=%s",
            bot.get("id"),
            coin,
            our_sz,
            px,
        )
    if rows:
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
        # Re-fetch AV/book + spot under lock so ratio uses total equity.
        snap: dict[str, Any] | None = None
        try:
            snap = hl_snapshot_positions(address)
        except Exception as exc:
            logger.warning("target AV refresh failed %s: %s", address[:10], exc)
        spot_usdc: float | None = None
        try:
            spot_usdc = float(hl_snapshot_spot_usdc(address) or 0)
        except Exception as exc:
            logger.warning("target spot USDC refresh failed %s: %s", address[:10], exc)
        try:
            mids = fetch_all_mids()
        except Exception:
            mids = dict(_mids_cache)

        bots_for_addr = [
            b
            for b in (book.get("bots") or {}).values()
            if str(b.get("address") or "").lower() == addr
        ]
        # Parents before mirror_of twins so Copy Current sees updated sibling book.
        bots_for_addr.sort(
            key=lambda b: (
                1 if b.get("mirror_of") else 0,
                int(b.get("priority") or 0),
                str(b.get("id") or ""),
            )
        )

        for bot in bots_for_addr:
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
                # Still allow twin sync when sibling moved and we had no fresh fills.
                if bot.get("mirror_of") and bot.get("live") is True:
                    twin_rows = _sync_paper_to_mirror_sibling(book, bot, mids, cfg)
                    if twin_rows:
                        logged.extend(twin_rows)
                        bot["fills"] = (bot.get("fills") or [])[:300]
                continue

            # Spot + perp must move together for sizing. If spot refresh failed but
            # we already know spot USDC, update positions only — keep prior av+spot
            # equity so a spot↔perp transfer does not one-side the denom this tick.
            had_spot = bot.get("target_spot_usdc") is not None
            if spot_usdc is not None:
                bot["target_spot_usdc"] = round(float(spot_usdc), 4)
                bot["target_spot_at"] = recv_at
            if snap is not None:
                if spot_usdc is not None or not had_spot:
                    _cache_target_meta(bot, snap, update_av=True)
                else:
                    _cache_target_meta(bot, snap, update_av=False)
                    logger.warning(
                        "HL follow %s: spot refresh miss — positions updated, "
                        "kept prior target_equity=%s",
                        bot.get("id"),
                        bot.get("target_equity"),
                    )
            elif _need_target_av_refresh(bot):
                logger.warning(
                    "HL follow skip %s: no target_av (cannot size)", bot.get("id")
                )
                _stamp_skipped_fills(bot, fresh, reason="no_snap", note_activity=False)
                continue
            else:
                _recompute_target_equity(bot)

            # Live-only seat: update target meta + trigger exchange sync; no paper book.
            if is_live_only_bot(bot):
                # Stamp startPosition from snap (same as paper) so Bitget
                # copy_current=off gate can tell flat→open from mid-book Open*.
                for item in _stamp_leader_start_positions(fresh, snap):
                    note_target_fill(bot, item.get("fill_time"))
                    coin = item.get("coin")
                    lev = _lev_for_coin(bot, str(coin or ""), cfg)
                    tid = item.get("tid")
                    logged.append(
                        {
                            "id": str(uuid.uuid4())[:8],
                            "action": "live_sync",
                            "source": bot.get("id"),
                            "bot_id": bot.get("id"),
                            "coin": coin,
                            "px": item.get("px"),
                            "leverage": lev,
                            "target_tid": tid,
                            "target_tids": [tid] if tid else [],
                            "target_delta": item.get("target_delta"),
                            "fill_time": item.get("fill_time"),
                            "dir": item.get("dir"),
                            "start_position": item.get("start_position"),
                            "ts": _now(),
                            "live_only": True,
                        }
                    )
                    logger.info(
                        "HL live-sync bot=%s coin=%s tdelta=%s px=%s lev=%s "
                        "av=%s equity_denom=%s sp=%s",
                        bot.get("id"),
                        coin,
                        item.get("target_delta"),
                        item.get("px"),
                        lev,
                        bot.get("target_av"),
                        bot.get("target_equity"),
                        item.get("start_position"),
                    )
                    row = logged[-1]
                    fills = list(bot.get("fills") or [])
                    fills.insert(0, row)
                    bot["fills"] = fills[:80]
                continue

            # Flatten-only hard stop: never pause — still apply this fill batch
            # (e.g. target flip after our stop) on the rebased equity.
            halt_rows = _maybe_risk_halt(bot, mids, cfg)
            if halt_rows:
                logged.extend(halt_rows)

            if _is_am_bot(bot):
                _am_maybe_mdd_break(bot)
                size_mult = _am_scale_mult(bot)
            else:
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
                    # Still close coins the leader already exited (pct path needs ratio
                    # only for opens; leftovers must not stick when AV/ratio is 0).
                    rec_rows = _reconcile_flat_target_coins(bot, snap, mids, cfg)
                    if rec_rows:
                        logged.extend(rec_rows)
                        logger.warning(
                            "HL ratio=0 per-coin reconcile %s: closed %s",
                            bot.get("id"),
                            len(rec_rows),
                        )
                    logger.warning(
                        "HL follow skip opens %s: ratio=0 equity=%s av=%s "
                        "target_equity=%s snap_flat=%s",
                        bot.get("id"),
                        bot.get("equity"),
                        bot.get("target_av"),
                        bot.get("target_equity"),
                        _target_snap_flat(snap),
                    )
                    _stamp_skipped_fills(
                        bot, fresh, reason="ratio_zero", note_activity=False
                    )
                continue

            # Chronological fills + stamp startPosition from snap when missing.
            fresh_sorted = _stamp_leader_start_positions(fresh, snap)
            # Pre-batch leader size per coin (first stamped startPosition).
            leader_pos: dict[str, float] = {}
            for item in fresh_sorted:
                c = str(item.get("coin") or "")
                sp = item.get("start_position")
                if sp is not None and c and c not in leader_pos:
                    try:
                        leader_pos[c] = float(sp)
                    except (TypeError, ValueError):
                        pass

            # Flatten tiny first-clip → orphan cascade on small seats (K vs O).
            fresh_sorted = _coalesce_flat_entry_fills(bot, fresh_sorted)

            for item in fresh_sorted:
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
                start_pos = item.get("start_position")
                if start_pos is None and coin in leader_pos:
                    start_pos = leader_pos[coin]
                logger.info(
                    "HL market-follow bot=%s coin=%s tdelta=%s px=%s ratio=%.6g lev=%s "
                    "av=%s target_equity=%s equity=%s fill_time=%s lag_ms=%s startPos=%s",
                    bot.get("id"),
                    coin,
                    item["target_delta"],
                    item["px"],
                    ratio,
                    lev,
                    bot.get("target_av"),
                    bot.get("target_equity"),
                    bot.get("equity"),
                    fill_ts,
                    lag_ms,
                    start_pos,
                )
                extra_tids = item.get("extra_tids")
                if not isinstance(extra_tids, list):
                    extra_tids = None
                if item.get("coalesced_n"):
                    logger.info(
                        "HL coalesce flat-entry bot=%s coin=%s n=%s tdelta=%s",
                        bot.get("id"),
                        coin,
                        item.get("coalesced_n"),
                        item.get("target_delta"),
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
                    fill_dir=item.get("dir"),
                    target_snap=snap,
                    start_position=start_pos,
                    extra_tids=[str(t) for t in extra_tids] if extra_tids else None,
                )
                note_target_fill(bot, fill_ts)
                logged.extend(rows)
                # Advance chained leader size for the next fill on this coin.
                try:
                    if start_pos is not None:
                        leader_pos[coin] = float(start_pos) + float(item["target_delta"])
                    elif coin in leader_pos:
                        leader_pos[coin] = float(leader_pos[coin]) + float(
                            item["target_delta"]
                        )
                except (TypeError, ValueError):
                    pass
                # keep existing list in sync for multi-fill dedupe in same event
                existing = list(bot.get("fills") or [])

            # Per-coin reconcile: leader flat on a coin → we must be flat (no silent opens).
            if snap is not None:
                rec_rows = _reconcile_flat_target_coins(bot, snap, mids, cfg)
                if rec_rows:
                    logged.extend(rec_rows)

            # Live twin (e.g. O←K): Dextrabot-style Copy Current to sibling paper.
            # AM paper sleeves also set mirror_of — only live seats sync.
            if bot.get("mirror_of") and bot.get("live") is True:
                twin_rows = _sync_paper_to_mirror_sibling(book, bot, mids, cfg)
                if twin_rows:
                    logged.extend(twin_rows)

            bot["fills"] = (bot.get("fills") or [])[:300]
        port_rows = _maybe_portfolio_risk(book, mids, cfg)
        if port_rows:
            logged.extend(port_rows)
        save_paper(book)

    flush_pending_live_flatten()

    if logged:
        try:
            from utils.hl_bitget_executor import maybe_execute_rows_async

            maybe_execute_rows_async(logged)
        except Exception:
            logger.exception("HL Bitget live hook failed")
        try:
            from utils.hl_binance_executor import maybe_execute_rows_async as bn_exec

            bn_exec(logged)
        except Exception:
            logger.exception("HL Binance live hook failed")
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
    """Beijing calendar day roll. AM bots settle gentle anti-martingale scale here."""
    day = _beijing_day()
    sizing = float(bot.get("balance") or cfg["bot_balance"])
    _recompute_bot(bot)
    try:
        eq = float(bot.get("equity") or sizing)
    except (TypeError, ValueError):
        eq = sizing
    if bot.get("day_key") != day:
        prev = bot.get("day_key")
        if prev and _is_am_bot(bot):
            _am_settle_day(bot, eq)
        bot["day_key"] = day
        bot["day_start_equity"] = eq
        if _is_am_bot(bot):
            bot["am_day_start_equity"] = eq
        if bot.get("day_start_equity") is None:
            bot["day_start_equity"] = float(bot.get("equity") or sizing)
    elif bot.get("day_start_equity") is None:
        bot["day_start_equity"] = float(bot.get("equity") or sizing)
    if _is_am_bot(bot):
        _am_maybe_mdd_break(bot)


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
    """Clear per-bot halt and rebase its −25% anchor to current equity."""
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
        if _is_am_bot(bot):
            continue
        if active_only and bot.get("risk_halted"):
            continue
        _recompute_bot(bot)
        total += float(bot.get("equity") or bot.get("balance") or 0)
    return round(total, 4)


def _portfolio_active_anchor(book: dict[str, Any], cfg: dict[str, Any]) -> float:
    """Sum of per-bot risk anchors for non-halted bots (desk return basis)."""
    total = 0.0
    for bot in _active_bots(book):
        if _is_am_bot(bot):
            continue
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
    """Unlock per-bot halt after cooldown; rebase that bot's risk anchor.

    When daily_loss_pct is OFF (≤0), clear any leftover halts immediately so
    bots resume following after the feature is disabled.
    """
    released: list[str] = []
    if float(cfg.get("daily_loss_pct") or 0) <= 0:
        for bot in list(_halted_bots(book)):
            _reset_bot_after_portfolio_rebase(bot)
            released.append(str(bot.get("id") or ""))
            logger.info(
                "HL per-bot halt cleared %s (daily_loss_pct off)",
                bot.get("id"),
            )
        return released

    cool = float(cfg.get("bot_halt_cooldown_sec") or 0)
    if cool <= 0:
        return []
    now = time.time()
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
            if reason in ("portfolio_sl", "portfolio_peak_dd")
            else "risk_halt_close"
        )
    )
    sync_rows: list[dict[str, Any]] = []
    for bot in _iter_bots(book):
        if _is_am_bot(bot):
            # AM sleeve is its own island — desk hard rebase must not wipe it.
            continue
        if bot.get("positions"):
            sync_rows.extend(
                _flatten_bot_positions(
                    bot, mids, action=action, risk_reason=reason, keep_halted=False
                )
            )
        _reset_bot_after_portfolio_rebase(bot)

    new_eq = _portfolio_equity(book, active_only=False)
    book["portfolio_anchor_equity"] = new_eq
    book["portfolio_peak_equity"] = new_eq
    book["portfolio_peak_dd_pct"] = 0.0
    book["portfolio_return_pct"] = 0.0
    book["portfolio_soft_tp_taken"] = False
    book["portfolio_copy_scale"] = 1.0
    book["portfolio_risk"] = {
        "reason": reason,
        "tripped_at": _now(),
        "anchor_before": round(anchor, 4),
        "peak_before": round(anchor, 4) if reason == "portfolio_peak_dd" else None,
        "equity_before": round(equity, 4),
        "return_pct": round(ret, 6),
        "anchor_after": round(new_eq, 4),
        "peak_after": round(new_eq, 4),
        "tp_soft_pct": float(cfg.get("portfolio_tp_pct") or 0),
        "tp_hard_pct": float(cfg.get("portfolio_tp_hard_pct") or 0),
        "sl_pct": float(cfg.get("portfolio_sl_pct") or 0),
        "peak_dd_pct": float(cfg.get("portfolio_peak_dd_pct") or 0),
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
    """Full-desk risk: peak DD flatten; soft TP; hard TP/SL/multi-halt rebase."""
    _maybe_release_bot_halt_cooldown(book, cfg)
    _ensure_portfolio_anchor(book, cfg)

    soft_tp = float(cfg.get("portfolio_tp_pct") or 0)
    hard_tp = float(cfg.get("portfolio_tp_hard_pct") or 0)
    sl = float(cfg.get("portfolio_sl_pct") or 0)
    peak_dd_limit = float(cfg.get("portfolio_peak_dd_pct") or 0)
    soft_keep = float(cfg.get("portfolio_soft_reduce") or 0.5)
    try:
        halt_trigger = int(cfg.get("portfolio_halt_count_trigger") or 0)
    except (TypeError, ValueError):
        halt_trigger = 0

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

    # Running peak for peak-to-trough desk stop.
    try:
        peak = float(book.get("portfolio_peak_equity") or 0)
    except (TypeError, ValueError):
        peak = 0.0
    if peak < equity:
        peak = equity
    if peak <= 1e-9:
        peak = max(equity, anchor)
    book["portfolio_peak_equity"] = round(peak, 4)
    peak_dd = (peak - equity) / peak if peak > 1e-9 else 0.0
    book["portfolio_peak_dd_pct"] = round(peak_dd, 6)

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

    # Peak drawdown flatten (default −15% from high-water mark).
    if peak_dd_limit > 0 and peak_dd >= peak_dd_limit:
        return _hard_portfolio_rebase(
            book,
            mids,
            cfg,
            reason="portfolio_peak_dd",
            ret=-peak_dd,
            anchor=peak,
            equity=equity,
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
            if _is_am_bot(bot):
                continue
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
    """Per-bot hard stop: −daily_loss_pct vs risk_anchor_equity (OFF when ≤0).

    Flatten open legs, rebase the risk anchor to post-flat equity, and keep
    following — no pause / risk_halted lockout.
    """
    if float(cfg.get("daily_loss_pct") or 0) <= 0:
        # Clear any leftover pause from older builds so follow resumes.
        if bot.get("risk_halted"):
            _reset_bot_after_portfolio_rebase(bot)
        return None

    # Legacy pause flag: unlock immediately (flatten-only policy).
    if bot.get("risk_halted") and not (bot.get("positions") or {}):
        _reset_bot_after_portfolio_rebase(bot)
        return None

    _roll_day(bot, cfg)
    _recompute_bot(bot)
    sizing = float(bot.get("balance") or cfg["bot_balance"])
    anchor = _bot_risk_anchor(bot, cfg)
    if bot.get("risk_anchor_equity") is None:
        bot["risk_anchor_equity"] = round(anchor, 4)
    equity_now = float(bot.get("equity") or sizing)
    loss_pct = 0.0 if anchor <= 0 else (anchor - equity_now) / anchor
    if not (cfg["daily_loss_pct"] > 0 and loss_pct >= cfg["daily_loss_pct"]):
        return None

    rows = _flatten_bot_positions(
        bot,
        mids,
        action="risk_halt_close",
        risk_reason="bot_hard_stop",
        keep_halted=False,
    )
    _reset_bot_after_portfolio_rebase(bot)
    logger.warning(
        "HL per-bot hard stop %s loss_pct=%.1f%% equity=%.2f risk_anchor_was=%.2f "
        "→ flatten + rebase anchor=%.2f (keep following)",
        bot.get("id"),
        loss_pct * 100.0,
        equity_now,
        anchor,
        float(bot.get("risk_anchor_equity") or 0),
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
    # Hard stop flattens then rebases; continue aligning so we can reopen.
    halt_rows = _maybe_risk_halt(bot, mids, cfg) or []

    _recompute_bot(bot)
    # Mirror snaps only carry perp account_value; fold in cached Core spot USDC.
    try:
        perp = float(snap.get("account_value") or 0)
    except (TypeError, ValueError):
        perp = 0.0
    bot["target_av"] = perp
    target_eq = _recompute_target_equity(bot)
    your_eq = float(bot.get("equity") or bot.get("balance") or cfg["bot_balance"])
    ratio = (your_eq / target_eq) if target_eq > 1e-9 else 0.0
    bot["copy_ratio"] = round(ratio, 10)
    target_av = perp

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
            "target_equity": target_eq,
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
        pos["target_equity"] = target_eq
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
    return list(halt_rows) + rows

"""Desk follow candidate pool — prerequisite for dynamic address binding.

Builds `hl_desk_candidates.json` then splits into:

  ready  — day/swing · flat WR≥80% · trips 10–100 · wallet funds OK
  watch  — passed pool gates but not ready (quiet / style / thin WR…)
  bound  — currently on desk A–J (reference only)
  rejected — watchlist reject_for_now

Funds gate = perp AV + Core spot USDC + HyperEVM USDC (not perp-only).
Stocks OK when Bitget-mapped. Trips = flat round-trips (pos 0→open→0).

Does NOT auto-rebind; only maintains the pool.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from utils.hl_bitget_symbol_map import bitget_contract_set, map_hl_coin_to_bitget
from utils.hl_short_term import (
    load_watchlist,
    load_watchlist_doc,
    resolve_data_dir,
    snapshot_hyperevm_usdc,
    snapshot_spot_usdc,
)
from utils.hl_wr_screen import (
    LEADERBOARD_URL,
    _fetch_fills_7d,
    _hl_info,
    _leaderboard_candidates,
    _merge_leg_bursts,
    _parse_leaderboard_row,
    _round_trips_flat,
    _wr_from_round_trips,
)

logger = logging.getLogger(__name__)

BJ = timezone(timedelta(hours=8))
CANDIDATES_NAME = "hl_desk_candidates.json"

# ── Pool entry (deep score) ───────────────────────────────────────────
MIN_TRIPS = 10
MAX_TRIPS = 100  # flat round-trips in 7d (position 0→open→0)
MIN_WR = 0.65
MIN_WEEK_ROI = 0.10
MIN_PAIR = 0.50  # flat trips are always paired; kept for compat
MIN_PNL = 2_000.0
# Wallet funds = perp AV + Core spot USDC + HyperEVM USDC (NOT perp-only)
MIN_WALLET_USD = float(os.getenv("HL_CANDIDATE_MIN_WALLET_USD", "8000") or 8000)
# legacy env alias
if os.getenv("HL_CANDIDATE_MIN_LIVE"):
    MIN_WALLET_USD = float(os.getenv("HL_CANDIDATE_MIN_LIVE") or MIN_WALLET_USD)
MIN_LIVE = MIN_WALLET_USD  # back-compat name in gates dump
MAX_LPH24 = 8.0
# Anti-wash: raw fill spam + tiny average trip PnL
MAX_RAW_FPH24 = float(os.getenv("HL_CANDIDATE_MAX_RAW_FPH24", "12") or 12)
MIN_PNL_PER_TRIP = float(os.getenv("HL_CANDIDATE_MIN_PNL_PER_TRIP", "80") or 80)
BITGET_SHARE_MIN = float(os.getenv("HL_CANDIDATE_BITGET_SHARE_MIN", "0.20") or 0.20)
DEEP_N = int(float(os.getenv("HL_CANDIDATE_DEEP_N", "0") or 0))  # 0 = full coarse pool (~1k)
PICK_READY_N = int(float(os.getenv("HL_CANDIDATE_READY_N", "15") or 15))
PICK_WATCH_N = int(float(os.getenv("HL_CANDIDATE_WATCH_N", "40") or 40))
# Pace between wallets (fills already sleep internally)
SCORE_SLEEP_SEC = float(os.getenv("HL_CANDIDATE_SCORE_SLEEP_SEC", "0.85") or 0.85)
SCORE_ADDR_RETRIES = int(float(os.getenv("HL_CANDIDATE_ADDR_RETRIES", "3") or 3))

# ── Ready: day/swing · WR≥80% · trips 10–100 · wallet funds ───────────
READY_MIN_WR = float(os.getenv("HL_CANDIDATE_READY_MIN_WR", "0.80") or 0.80)
READY_MIN_TRIPS = int(float(os.getenv("HL_CANDIDATE_READY_MIN_TRIPS", "10") or 10))
READY_MAX_TRIPS = int(float(os.getenv("HL_CANDIDATE_READY_MAX_TRIPS", "100") or 100))
READY_MIN_PAIR = float(os.getenv("HL_CANDIDATE_READY_MIN_PAIR", "0.50") or 0.50)
READY_MIN_WEEK_ROI = float(os.getenv("HL_CANDIDATE_READY_MIN_WEEK_ROI", "0.10") or 0.10)
READY_MIN_PNL = float(os.getenv("HL_CANDIDATE_READY_MIN_PNL", "3000") or 3000)
READY_MIN_WALLET_USD = float(
    os.getenv("HL_CANDIDATE_READY_MIN_WALLET_USD")
    or os.getenv("HL_CANDIDATE_READY_MIN_LIVE")
    or "8000"
)
READY_MIN_LIVE = READY_MIN_WALLET_USD  # alias
READY_MAX_QUIET_H = float(os.getenv("HL_CANDIDATE_READY_MAX_QUIET_H", "24") or 24)
READY_LPH_MIN = float(os.getenv("HL_CANDIDATE_READY_LPH_MIN", "0") or 0)
READY_LPH_MAX = float(os.getenv("HL_CANDIDATE_READY_LPH_MAX", "6") or 6)
READY_BITGET_MIN = float(os.getenv("HL_CANDIDATE_READY_BITGET_MIN", "0.20") or 0.20)
READY_MAX_RAW_FPH24 = float(
    os.getenv("HL_CANDIDATE_READY_MAX_RAW_FPH24", str(MAX_RAW_FPH24)) or MAX_RAW_FPH24
)
READY_MIN_PNL_PER_TRIP = float(
    os.getenv("HL_CANDIDATE_READY_MIN_PNL_PER_TRIP", str(MIN_PNL_PER_TRIP))
    or MIN_PNL_PER_TRIP
)
# hyper-track style tags — ready only day + swing
READY_STYLES = {
    s.strip()
    for s in (
        os.getenv("HL_CANDIDATE_READY_STYLES", "day_trader,swing_trader") or "day_trader,swing_trader"
    ).split(",")
    if s.strip()
}

MAX_QUIET_H_READY = READY_MAX_QUIET_H
MIN_LIVE_READY = READY_MIN_WALLET_USD
MIN_WR = float(os.getenv("HL_CANDIDATE_MIN_WR", str(MIN_WR)) or MIN_WR)
MAX_TRIPS = int(float(os.getenv("HL_CANDIDATE_MAX_TRIPS", str(MAX_TRIPS)) or MAX_TRIPS))
MIN_TRIPS = int(float(os.getenv("HL_CANDIDATE_MIN_TRIPS", str(MIN_TRIPS)) or MIN_TRIPS))
MIN_WALLET_USD = float(os.getenv("HL_CANDIDATE_MIN_WALLET_USD", str(MIN_WALLET_USD)) or MIN_WALLET_USD)
MIN_LIVE = MIN_WALLET_USD


def candidates_path() -> Path:
    return resolve_data_dir() / CANDIDATES_NAME


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_candidates() -> dict[str, Any] | None:
    path = candidates_path()
    if not path.is_file():
        # fall back to repo root (local dev / committed snapshot)
        alt = _project_root() / CANDIDATES_NAME
        path = alt if alt.is_file() else path
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("load candidates failed: %s", exc)
        return None
    return data if isinstance(data, dict) else None


def save_candidates(data: dict[str, Any]) -> Path:
    path = candidates_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    # also mirror under project root when DATA_DIR != project (ops convenience)
    root_mirror = _project_root() / CANDIDATES_NAME
    if root_mirror.resolve() != path.resolve():
        try:
            root_mirror.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
    return path


def _desk_bound() -> dict[str, str]:
    """address_lower → bot_id for current A–J seats."""
    out: dict[str, str] = {}
    for w in load_watchlist():
        bid = str(w.get("id") or "").strip()
        addr = str(w.get("address") or "").strip().lower()
        if bid.startswith("bot_") and addr.startswith("0x"):
            out[addr] = bid
    return out


def _reject_set() -> set[str]:
    doc = load_watchlist_doc() or {}
    out: set[str] = set()
    for row in doc.get("reject_for_now") or []:
        if not isinstance(row, dict):
            continue
        addr = str(row.get("address") or "").strip().lower()
        if addr.startswith("0x"):
            out.add(addr)
    return out


def _fetch_leaderboard() -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(
                LEADERBOARD_URL, headers={"User-Agent": "desk-candidates"}
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            last_err = exc
            logger.warning("leaderboard try %s: %s", attempt + 1, exc)
            time.sleep(2 + attempt * 2)
    cache = _project_root() / "lb.json"
    if cache.is_file():
        logger.warning("using cached lb.json after fetch failures")
        return json.loads(cache.read_text(encoding="utf-8"))
    raise RuntimeError(f"leaderboard fetch failed: {last_err}")


def _classify_ht_style(trips: list[dict[str, Any]]) -> tuple[str, list[str], float, float]:
    """hyper-track style: scalper / day_trader / swing_trader / position_trader."""
    if not trips:
        return "unknown", [], 0.0, 0.0
    holds = [
        (int(t["close_time"]) - int(t["open_time"])) / 3600_000
        for t in trips
        if t.get("open_time") and t.get("close_time")
    ]
    avg_hold = sum(holds) / len(holds) if holds else 0.0
    span = max(int(t["close_time"]) for t in trips) - min(
        int(t["open_time"]) for t in trips
    )
    days = max(span / 86400_000, 1.0)
    freq = len(trips) / days
    wr = sum(1 for t in trips if float(t.get("pnl") or 0) > 0) / len(trips)

    if avg_hold < 1 and freq > 10:
        style = "scalper"
    elif avg_hold < 24 and freq > 2:
        style = "day_trader"
    elif avg_hold < 168:
        style = "swing_trader"
    else:
        style = "position_trader"

    tags: list[str] = []
    if avg_hold < 0.5 and freq > 20:
        tags.append("degen")
    if wr > 0.7 and avg_hold < 2:
        tags.append("sniper")
    if freq > 20 and 0.45 <= wr <= 0.55:
        tags.append("grinder")
    vols: dict[str, float] = defaultdict(float)
    for t in trips:
        vols[str(t.get("coin"))] += float(t.get("notional") or 0)
    tot = sum(vols.values()) or 1.0
    mx = max(vols.values()) / tot
    if mx > 0.8:
        tags.append("concentrated")
    elif len(vols) >= 10 and mx < 0.3:
        tags.append("diversified")
    return style, tags, avg_hold, freq


def _wallet_funds(addr: str) -> dict[str, float | None]:
    """Perp AV + Core spot USDC + HyperEVM USDC."""
    live = 0.0
    try:
        st = _hl_info({"type": "clearinghouseState", "user": addr})
        live = float(((st or {}).get("marginSummary") or {}).get("accountValue") or 0)
    except Exception as exc:
        logger.warning("perp AV %s: %s", addr[:12], exc)
    time.sleep(0.25)
    try:
        spot = float(snapshot_spot_usdc(addr) or 0)
    except Exception as exc:
        logger.warning("spot USDC %s: %s", addr[:12], exc)
        spot = 0.0
    time.sleep(0.2)
    try:
        evm = snapshot_hyperevm_usdc(addr)
    except Exception:
        evm = None
    evm_v = float(evm) if evm is not None else 0.0
    return {
        "live_av": round(live, 1),
        "spot_usdc": round(spot, 1),
        "evm_usdc": None if evm is None else round(evm_v, 1),
        "wallet_usd": round(live + spot + evm_v, 1),
    }


def score_addr(
    addr: str,
    c: dict[str, Any],
    *,
    now: datetime,
    start_ms: int,
    desk_map: dict[str, str],
) -> dict[str, Any] | None:
    """7d flat-trip score. Returns None if any pool gate fails."""
    fills = _fetch_fills_7d(addr, start_ms)
    if len(fills) < 12:
        return None
    bg = [f for f in fills if map_hl_coin_to_bitget(str(f.get("coin") or ""))]
    share = (len(bg) / len(fills)) if fills else 0.0
    if not bg or share < BITGET_SHARE_MIN:
        return None

    now_ms = now.timestamp() * 1000
    raw_24 = [f for f in fills if (now_ms - int(f.get("time") or 0)) < 86400_000]
    raw_fph24 = len(raw_24) / 24.0

    trips = _round_trips_flat(bg)
    wr, wins, losses, ncl = _wr_from_round_trips(trips)
    # ncl = closed flat round-trips in 7d (position 0→open→0)
    if (
        ncl < MIN_TRIPS
        or ncl > MAX_TRIPS
        or wr is None
        or wr < MIN_WR
    ):
        return None
    paired = sum(1 for t in trips if t.get("paired"))
    pr = paired / len(trips) if trips else 0.0
    if pr < MIN_PAIR:
        return None
    pnl = sum(float(t.get("pnl") or 0) for t in trips)
    if pnl < MIN_PNL:
        return None
    pnl_per_trip = pnl / float(ncl) if ncl > 0 else 0.0
    if pnl_per_trip < MIN_PNL_PER_TRIP:
        return None

    bg_24 = [f for f in bg if (now_ms - int(f.get("time") or 0)) < 86400_000]
    legs_24 = _merge_leg_bursts(bg_24, gap_ms=60_000)
    lph24 = len(legs_24) / 24.0
    trips_24 = [
        t for t in trips if (now_ms - int(t.get("close_time") or 0)) < 86400_000
    ]
    tph24 = len(trips_24) / 24.0
    if lph24 > MAX_LPH24:
        return None
    if raw_fph24 > MAX_RAW_FPH24:
        return None

    time.sleep(0.25)
    funds = _wallet_funds(addr)
    wallet = float(funds.get("wallet_usd") or 0)
    if wallet < MIN_WALLET_USD:
        return None

    style, style_tags, avg_hold, tpd = _classify_ht_style(trips)
    coins = Counter(str(f.get("coin")) for f in bg).most_common(3)
    last = datetime.fromtimestamp(int(fills[-1]["time"]) / 1000, BJ)
    quiet_h = (now - last).total_seconds() / 3600.0
    week_roi = float(c.get("week_roi") or 0)
    week_pnl = float(c.get("week_pnl") or 0)
    addr_l = addr.lower()
    return {
        "addr": addr_l,
        "desk": desk_map.get(addr_l),
        "live_av": funds.get("live_av"),
        "spot_usdc": funds.get("spot_usdc"),
        "evm_usdc": funds.get("evm_usdc"),
        "wallet_usd": funds.get("wallet_usd"),
        "lb_av": round(float(c.get("av") or 0), 1),
        "week_roi": round(week_roi, 4),
        "week_pnl": round(week_pnl, 1),
        "trips7": ncl,
        "wins": wins,
        "losses": losses,
        "paired": paired,
        "pair_rate": round(pr, 3),
        "wr7": round(wr, 4),
        "pnl7_closed": round(pnl, 1),
        "pnl_per_trip": round(pnl_per_trip, 2),
        "lph24": round(lph24, 2),
        "tph24": round(tph24, 2),
        "raw_fph24": round(raw_fph24, 2),
        "bitget_share": round(share, 3),
        "style": style,
        "style_tags": style_tags,
        "avg_hold_h": round(avg_hold, 2),
        "trips_per_day": round(tpd, 2),
        "coins": ",".join(k for k, _ in coins),
        "last_bj": last.strftime("%m-%d %H:%M"),
        "quiet_h": round(quiet_h, 2),
        "hl_url": f"https://app.hyperliquid.xyz/explorer/address/{addr_l}",
        "score": round(
            week_roi * 100.0
            + wr * 40.0
            + min(ncl, 40) * 0.5
            + min(pnl / 1000.0, 30),
            3,
        ),
    }


def _is_ready(row: dict[str, Any]) -> bool:
    """Bindable-now: day/swing · WR≥80% · wallet funds · anti-wash.

    Calibrated to match local ``能跟`` set (flat day/swing table).
    """
    try:
        wr = float(row.get("wr7") or 0)
        trips = int(row.get("trips7") or 0)
        # flat trips are always paired; missing field must not fail ready
        pair_raw = row.get("pair_rate")
        pair = 1.0 if pair_raw is None else float(pair_raw)
        week_roi = float(row.get("week_roi") or 0)
        pnl = float(row.get("pnl7_closed") or 0)
        wallet = float(row.get("wallet_usd") or 0)
        if wallet <= 0:
            # back-compat old snapshots: fall back to perp live
            wallet = float(row.get("live_av") or 0)
        quiet = float(row.get("quiet_h") or 999)
        lph = float(row.get("lph24") or 0)
        bg = float(row.get("bitget_share") or 0)
        raw_fph = float(row.get("raw_fph24") or 0)
        ppt = float(row.get("pnl_per_trip") or 0)
        style = str(row.get("style") or "").strip()
        if ppt <= 0 and trips > 0:
            ppt = pnl / float(trips)
    except (TypeError, ValueError):
        return False
    if READY_STYLES and style and style not in READY_STYLES:
        return False
    if wr < READY_MIN_WR:
        return False
    if trips < READY_MIN_TRIPS or trips > READY_MAX_TRIPS:
        return False
    if pair < READY_MIN_PAIR:
        return False
    if week_roi < READY_MIN_WEEK_ROI:
        return False
    if pnl < READY_MIN_PNL:
        return False
    if wallet < READY_MIN_WALLET_USD:
        return False
    if quiet > READY_MAX_QUIET_H:
        return False
    if lph < READY_LPH_MIN or lph > READY_LPH_MAX:
        return False
    # Trips already computed on Bitget-mapped fills; allow low fill-share
    # when listed top coins are mostly mappable (e.g. stock perps).
    if bg < READY_BITGET_MIN:
        parts = [c.strip() for c in str(row.get("coins") or "").split(",") if c.strip()]
        if not parts:
            return False
        mapped = sum(1 for c in parts if map_hl_coin_to_bitget(c)) / len(parts)
        if mapped < 0.5:
            return False
    if raw_fph > READY_MAX_RAW_FPH24:
        return False
    if ppt < READY_MIN_PNL_PER_TRIP:
        return False
    return True


def _sort_key(row: dict[str, Any]) -> tuple:
    tags = row.get("style_tags") or []
    concentrated = 1 if (isinstance(tags, list) and "concentrated" in tags) else 0
    return (
        concentrated,  # prefer 单币集中 in ready/watch lists
        float(row.get("score") or 0),
        float(row.get("week_roi") or 0),
        float(row.get("wr7") or 0),
        float(row.get("trips7") or 0),
        float(row.get("pnl7_closed") or 0),
    )


def build_candidate_pool(*, sleep_sec: float | None = None) -> dict[str, Any]:
    """Full leaderboard deep-scan → ready / watch / bound pools."""
    if sleep_sec is None:
        sleep_sec = SCORE_SLEEP_SEC
    bitget_contract_set(force=True)
    now = datetime.now(BJ)
    start_ms = int((now - timedelta(days=7)).timestamp() * 1000)
    desk_map = _desk_bound()
    reject = _reject_set()

    logger.info("desk candidates: fetch leaderboard…")
    lb = _fetch_leaderboard()
    parsed: list[dict] = []
    by_addr: dict[str, dict] = {}
    for row in lb.get("leaderboardRows") or []:
        item = _parse_leaderboard_row(row)
        if item:
            parsed.append(item)
            by_addr[str(item["addr"]).lower()] = item

    cands = _leaderboard_candidates(parsed, set())
    pool = [
        c
        for c in cands
        if 8_000 <= float(c.get("av") or 0) <= 2_000_000
        and float(c.get("week_roi") or 0) >= MIN_WEEK_ROI
        and float(c.get("week_pnl") or 0) >= 3_000
    ]
    pool.sort(
        key=lambda x: (float(x.get("week_roi") or 0), float(x.get("week_pnl") or 0)),
        reverse=True,
    )
    deep = list(pool) if DEEP_N <= 0 else pool[:DEEP_N]
    seen = {str(c["addr"]).lower() for c in deep}
    # Always score current desk seats so bound[] stays fresh
    for addr, bid in desk_map.items():
        if addr in seen:
            continue
        base = by_addr.get(addr) or {
            "addr": addr,
            "av": 0.0,
            "week_roi": 0.0,
            "week_pnl": 0.0,
        }
        deep.append(base)
        seen.add(addr)
        logger.info("force desk %s into candidate scan", bid)

    # Force-scan last local 「能跟」seeds so ready set does not drift off Deep-N cuts
    seed_path = _project_root() / "hl_ready_seed_addrs.json"
    if seed_path.is_file():
        try:
            seed_doc = json.loads(seed_path.read_text(encoding="utf-8"))
            seeds = seed_doc if isinstance(seed_doc, list) else (seed_doc.get("addrs") or [])
        except Exception:
            seeds = []
        for raw in seeds:
            addr = str(raw or "").strip().lower()
            if not addr.startswith("0x") or addr in seen or addr in reject:
                continue
            deep.append(
                by_addr.get(addr)
                or {"addr": addr, "av": 0.0, "week_roi": 0.0, "week_pnl": 0.0}
            )
            seen.add(addr)
            logger.info("force ready-seed %s into candidate scan", addr[:14])


    def _is_rate_limited(exc: BaseException) -> bool:
        code = getattr(exc, "code", None)
        msg = str(exc)
        return code == 429 or "429" in msg or "Too Many Requests" in msg

    hits: list[dict[str, Any]] = []
    for i, c in enumerate(deep):
        addr = str(c.get("addr") or "").strip().lower()
        if not addr.startswith("0x"):
            continue
        if addr in reject:
            continue
        row = None
        last_exc: Exception | None = None
        for attempt in range(max(1, SCORE_ADDR_RETRIES)):
            try:
                row = score_addr(addr, c, now=now, start_ms=start_ms, desk_map=desk_map)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if _is_rate_limited(exc) and attempt + 1 < SCORE_ADDR_RETRIES:
                    wait = min(45.0, 5.0 * (attempt + 1))
                    logger.warning(
                        "candidate 429 %s retry %s/%s sleep %.0fs",
                        addr[:12],
                        attempt + 1,
                        SCORE_ADDR_RETRIES,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                break
        if last_exc is not None and row is None:
            logger.warning("candidate score fail %s: %s", addr[:12], last_exc)
        if row:
            hits.append(row)
            logger.info(
                "candidate HIT%s %s wr=%.0f%% n=%s live=%.0f quiet=%.1fh",
                f" [{row['desk']}]" if row.get("desk") else "",
                addr[:14],
                float(row["wr7"]) * 100,
                row["trips7"],
                row["live_av"],
                row["quiet_h"],
            )
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        if (i + 1) % 20 == 0:
            logger.info("candidate progress %s/%s hits=%s", i + 1, len(deep), len(hits))

    hits.sort(key=_sort_key, reverse=True)

    bound = [h for h in hits if h.get("desk")]
    unbound = [h for h in hits if not h.get("desk")]
    ready = [h for h in unbound if _is_ready(h)][:PICK_READY_N]
    watch = [h for h in unbound if not _is_ready(h)][:PICK_WATCH_N]

    # Mark tiers on rows for consumers
    ready_set = {r["addr"] for r in ready}
    for h in hits:
        if h.get("desk"):
            h["tier"] = "bound"
        elif h["addr"] in ready_set:
            h["tier"] = "ready"
        else:
            h["tier"] = "watch"

    out = {
        "ok": True,
        "generated_at": now.isoformat(),
        "window": "7d",
        "note": (
            "Desk follow candidate pool. ready = day/swing · flat WR≥80% · "
            "trips 10–100 · wallet funds (perp+spot+evm); "
            "watch = pool hit but not ready; bound = current A–J. "
            "Stocks OK if Bitget-mapped. Auto-rebind not enabled."
        ),
        "gates": {
            "min_trips": MIN_TRIPS,
            "max_trips": MAX_TRIPS,
            "trips_note": "flat round-trips in 7d (per-coin position 0→open→0)",
            "min_wr": MIN_WR,
            "min_week_roi_pool": MIN_WEEK_ROI,
            "min_pair_rate": MIN_PAIR,
            "min_pnl": MIN_PNL,
            "min_wallet_usd": MIN_WALLET_USD,
            "min_live": MIN_WALLET_USD,
            "funds_note": "wallet_usd = perp AV + Core spot USDC + HyperEVM USDC",
            "max_lph24_merged_legs": MAX_LPH24,
            "max_raw_fph24": MAX_RAW_FPH24,
            "min_pnl_per_trip": MIN_PNL_PER_TRIP,
            "anti_wash": "trips 10-100 + raw_fph/lph caps + min pnl/trip",
            "bitget_share_min": BITGET_SHARE_MIN,
            "deep_n": DEEP_N,
            "score_sleep_sec": SCORE_SLEEP_SEC,
            "ready": {
                "styles": sorted(READY_STYLES),
                "min_wr": READY_MIN_WR,
                "min_trips": READY_MIN_TRIPS,
                "max_trips": READY_MAX_TRIPS,
                "min_pair": READY_MIN_PAIR,
                "min_week_roi": READY_MIN_WEEK_ROI,
                "min_pnl": READY_MIN_PNL,
                "min_wallet_usd": READY_MIN_WALLET_USD,
                "min_live": READY_MIN_WALLET_USD,
                "max_quiet_h": READY_MAX_QUIET_H,
                "lph24": [READY_LPH_MIN, READY_LPH_MAX],
                "max_raw_fph24": READY_MAX_RAW_FPH24,
                "min_pnl_per_trip": READY_MIN_PNL_PER_TRIP,
                "bitget_share_min": READY_BITGET_MIN,
            },
        },
        "scanned": len(deep),
        "hit_count": len(hits),
        "ready_count": len(ready),
        "watch_count": len(watch),
        "bound_count": len(bound),
        "reject_count": len(reject),
        "ready": ready,
        "watch": watch,
        "bound": bound,
        "rejects": sorted(reject),
    }
    path = save_candidates(out)
    logger.info(
        "desk candidates wrote %s ready=%s watch=%s bound=%s",
        path,
        len(ready),
        len(watch),
        len(bound),
    )
    return out


def get_candidates(*, refresh: bool = False) -> dict[str, Any]:
    if refresh:
        return build_candidate_pool()
    snap = load_candidates()
    if snap:
        return snap
    return {
        "ok": False,
        "generated_at": None,
        "note": "no candidate pool yet — run build or refresh",
        "ready": [],
        "watch": [],
        "bound": [],
        "ready_count": 0,
        "watch_count": 0,
        "bound_count": 0,
    }


def next_ready_candidate(
    *,
    exclude: set[str] | None = None,
) -> dict[str, Any] | None:
    """First ready address not excluded (for future auto-bind)."""
    snap = load_candidates() or {}
    skip = {a.lower() for a in (exclude or set())}
    skip |= set(_desk_bound().keys())
    skip |= _reject_set()
    for row in snap.get("ready") or []:
        if not isinstance(row, dict):
            continue
        addr = str(row.get("addr") or "").lower()
        if addr.startswith("0x") and addr not in skip and _is_ready(row):
            return row
    return None

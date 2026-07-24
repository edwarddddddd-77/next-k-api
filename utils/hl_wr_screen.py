"""HL fill / round-trip helpers shared by desk candidate pool and research scripts.

Product 「胜率筛」UI + daily board cron removed; use ``utils.hl_desk_candidates`` instead.
Still exposes ``run_screen`` for offline CLI experiments.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.hl_short_term import resolve_data_dir

logger = logging.getLogger(__name__)

LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
INFO_URL = "https://api.hyperliquid.xyz/info"
BOARD_NAME = "hl_wr_screen_board.json"

LIVE_AV_MIN = 8_000.0
WEEK_PNL_MIN = 3_000.0
WEEK_VLM_MIN = 30_000.0
WEEK_ROI_MIN = 0.10
DEEP_TOP_N = 120  # match 7d-good deep scan width
DEEP_WHALE_N = 30
DEEP_MID_N = 90
PICK_TOP_N = 15
MID_AV_MAX = 2_000_000.0

MAJORS = {
    "BTC",
    "ETH",
    "SOL",
    "HYPE",
    "ADA",
    "XRP",
    "BNB",
    "DOGE",
    "LINK",
    "AVAX",
    "AAVE",
    "XMR",
    "ZEC",
    "LTC",
    "SUI",
    "NEAR",
    "WLD",
    "TAO",
    "TRUMP",
    "MON",
    "TIA",
    "ATOM",
}
STOCKS = {
    "TSLA",
    "META",
    "NVDA",
    "AAPL",
    "MU",
    "COIN",
    "MSTR",
    "AMZN",
    "MSFT",
    "GOOG",
    "GOOGL",
    "CRCL",
    "SNDK",
    "CL",
    "SILVER",
    "HOOD",
    "SKHX",
    "SKHY",
    "SPCX",
    "CXMT",
}

CRITERIA_COPYABLE = {
    "id": "copyable",
    "label": "可跟",
    "week_pnl_min": WEEK_PNL_MIN,
    "week_vlm_min": WEEK_VLM_MIN,
    "week_roi_min": 0.0,  # pool uses WEEK_ROI_MIN=10%; deep pass matches 7d-good (desk B at ~4% still OK)
    "live_av_min": LIVE_AV_MIN,
    # Pace = 60s-merged open/close legs / hour (not raw fill fragments).
    "fph24_min": 0.0,
    "fph24_max": 8.0,
    "fph24_sweet": [0.1, 4.0],
    "fills7_min": 12,
    "closed7_min": 8,  # round-trips >7
    "wr7_min": 0.55,
    "pair_rate_min": 0.50,
    "pnl7_min": 2_000.0,  # trip closed PnL
    "pnl7_roi_min": 0.0,
    "month_pnl_min": None,
    "scratch_max": 1.0,  # off — not in 7d-good recipe
    "crypto_share_min": 0.0,
    "stock_share_max": 1.0,
    "major_share_min": 0.50,  # Bitget-mappable share
    "c2_min": 0.0,
    "npos_max": 99,
    "follow_coins_min": 0,
    "live_av_ratio_min": 0.0,  # shells OK if live_av gate passes
    "crypto_only": False,
    "deep_top_n": DEEP_TOP_N,
    "deep_whale_n": DEEP_WHALE_N,
    "deep_mid_n": DEEP_MID_N,
    "pick_top_n": PICK_TOP_N,
    "wr_window": "7d_round_trips_flat_bitget",
    "note": (
        "可跟=7d回合>7·WR≥55%·配对≥50%·周ROI≥10%·平仓≥2k·"
        "live≥8k·Bitget占比≥50%·合并腿lph≤8（非碎单fph）"
    ),
}

CRITERIA_WATCH = {
    **CRITERIA_COPYABLE,
    "id": "watch",
    "label": "宽观察",
    "fph24_min": 0.0,
    "fph24_max": 15.0,
    "wr7_min": 0.50,
    "pair_rate_min": 0.40,
    "pnl7_min": 1_000.0,
    "pnl7_roi_min": 0.0,
    "week_roi_min": 0.05,
    "major_share_min": 0.35,
    "closed7_min": 6,
    "fills7_min": 10,
    "note": "更宽观察；不直接当好跟绑仓",
}

_lock = threading.Lock()
_run_lock = threading.Lock()


def _board_path() -> Path:
    return resolve_data_dir() / BOARD_NAME


def _http_get_json(url: str, *, timeout: float = 120.0) -> Any:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "next-k-hl-wr-screen/2.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _hl_info(body: dict, *, retries: int = 8) -> Any:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                INFO_URL,
                data=json.dumps(body).encode(),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "next-k-hl-wr-screen/5.1",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            last_exc = exc
            code = getattr(exc, "code", None)
            msg = str(exc)
            is_429 = code == 429 or "429" in msg or "Too Many Requests" in msg
            if is_429:
                # Exponential backoff; HL info is aggressive on burst.
                wait = min(60.0, 2.0 * (2**attempt))
                retry_after = None
                try:
                    hdrs = getattr(exc, "headers", None)
                    if hdrs is not None:
                        retry_after = hdrs.get("Retry-After")
                except Exception:
                    retry_after = None
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except (TypeError, ValueError):
                        pass
                logger.warning(
                    "HL info 429 (attempt %s/%s) sleep %.1fs type=%s",
                    attempt + 1,
                    retries,
                    wait,
                    body.get("type"),
                )
                time.sleep(wait)
                continue
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _known_watch_addrs() -> set[str]:
    return set(_watchlist_by_addr())


def _watchlist_by_addr() -> dict[str, str]:
    """addr(lower) → watchlist id (bot_a…) or reject tag. Overlap with screen is allowed."""
    try:
        from utils.hl_short_term import load_watchlist_doc

        doc = load_watchlist_doc()
        out: dict[str, str] = {}
        for row in doc.get("wallets") or []:
            addr = str(row.get("address") or "").strip().lower()
            if addr:
                out[addr] = str(row.get("id") or "watch")
        for row in doc.get("reject_for_now") or []:
            addr = str(row.get("address") or "").strip().lower()
            if addr and addr not in out:
                out[addr] = "reject"
        return out
    except Exception:
        logger.exception("failed to load watchlist for screen overlap tags")
        return {}


def _coin_base(coin: str) -> str:
    c = str(coin or "").strip()
    if c.lower().startswith("xyz:"):
        return c.split(":", 1)[-1].upper()
    return c.upper()


def _is_stock_coin(coin: str) -> bool:
    c = str(coin or "").strip()
    if c.lower().startswith("xyz:"):
        return True
    return _coin_base(c) in STOCKS


def _is_crypto_major(coin: str) -> bool:
    if _is_stock_coin(coin):
        return False
    return _coin_base(coin) in MAJORS


def _is_bitget_mapped(coin: str) -> bool:
    """Strict Bitget contract map — same gate as screen_7d_good.py."""
    c = str(coin or "").strip()
    if not c:
        return False
    try:
        from utils.hl_bitget_symbol_map import map_hl_coin_to_bitget

        return bool(map_hl_coin_to_bitget(c))
    except Exception:
        return False


def _fetch_fills_7d(addr: str, start_ms: int) -> list[dict[str, Any]]:
    """Paginate userFillsByTime from start_ms — full 7d window (not truncated userFills)."""
    fills: list[dict[str, Any]] = []
    cursor = int(start_ms)
    seen: set[tuple[Any, ...]] = set()
    for _ in range(4):
        time.sleep(0.55)
        batch = _hl_info(
            {"type": "userFillsByTime", "user": addr, "startTime": cursor}
        )
        if not isinstance(batch, list) or not batch:
            break
        max_t = cursor
        for f in batch:
            if not isinstance(f, dict):
                continue
            try:
                ts = int(f.get("time") or 0)
            except (TypeError, ValueError):
                continue
            if ts < start_ms:
                continue
            key = (f.get("tid"), ts, f.get("coin"), f.get("sz"))
            if key in seen:
                continue
            seen.add(key)
            fills.append(f)
            max_t = max(max_t, ts)
        if len(batch) < 2000:
            break
        if max_t + 1 <= cursor:
            break
        cursor = max_t + 1
    fills.sort(key=lambda x: int(x.get("time") or 0))
    return fills


def _ensure_desk_in_deep(
    deep: list[dict[str, Any]],
    by_addr: dict[str, dict[str, Any]],
    desk_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Force-scan current watchlist wallets even if outside the deep cut."""
    seen = {str(r.get("addr") or "").lower() for r in deep}
    out = list(deep)
    for addr, wid in desk_map.items():
        if wid == "reject":
            continue
        a = str(addr or "").lower()
        if not a or a in seen:
            continue
        base = by_addr.get(a) or {
            "addr": a,
            "av": 0.0,
            "week_roi": 0.0,
            "week_pnl": 0.0,
            "week_vlm": 0.0,
            "day_pnl": 0.0,
            "month_pnl": 0.0,
        }
        out.append(dict(base))
        seen.add(a)
        logger.info("hl wr screen force desk %s %s", wid, a[:12])
    return out


def _parse_leaderboard_row(row: dict[str, Any]) -> dict[str, Any] | None:
    addr = (row.get("ethAddress") or "").lower()
    if not addr:
        return None
    av = float(row.get("accountValue") or 0)
    perf: dict[str, Any] = {}
    for w in row.get("windowPerformances") or []:
        if isinstance(w, list) and len(w) >= 2 and isinstance(w[1], dict):
            perf[w[0]] = w[1]
    day = perf.get("day") or {}
    week = perf.get("week") or {}
    month = perf.get("month") or {}
    day_pnl = float(day.get("pnl") or 0)
    week_pnl = float(week.get("pnl") or 0)
    month_pnl = float(month.get("pnl") or 0)
    week_vlm = float(week.get("vlm") or 0)
    month_vlm = float(month.get("vlm") or 0)
    week_roi = float(week.get("roi") or 0)
    month_roi = float(month.get("roi") or 0)
    turn_w = (week_vlm / av) if av > 0 else 0.0
    turn_m = (month_vlm / av) if av > 0 else 0.0
    return {
        "addr": addr,
        "av": round(av, 2),
        "day_pnl": round(day_pnl, 2),
        "week_pnl": round(week_pnl, 2),
        "month_pnl": round(month_pnl, 2),
        "week_vlm": round(week_vlm, 2),
        "month_vlm": round(month_vlm, 2),
        "week_roi": week_roi,
        "month_roi": month_roi,
        "turn_w": round(turn_w, 2),
        "turn_m": round(turn_m, 2),
    }


def _leaderboard_candidates(rows: list[dict[str, Any]], skip: set[str]) -> list[dict[str, Any]]:
    """Active + weekly profit. Rank by week PnL (with light ROI tie-break), not ROI-only ghosts."""
    cands: list[dict[str, Any]] = []
    for base in rows:
        if base["addr"] in skip:
            continue
        if float(base["week_pnl"]) <= WEEK_PNL_MIN:
            continue
        if float(base["week_vlm"]) < WEEK_VLM_MIN:
            continue
        if float(base["av"]) < LIVE_AV_MIN:
            continue
        if WEEK_ROI_MIN > 0 and float(base.get("week_roi") or 0) < WEEK_ROI_MIN:
            continue
        cands.append(dict(base))
    # Prefer real money made; ROI only breaks ties (avoids empty high-ROI shells)
    cands.sort(
        key=lambda x: (float(x.get("week_pnl") or 0), float(x.get("week_roi") or 0)),
        reverse=True,
    )
    return cands


def _select_deep_list(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Don't only deep-scan week-PnL whales (stock shells dominate that top).

    Mix: small whale tranche for context + mid-AV / high week-ROI hunt for copyable crypto.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    whales = sorted(
        cands,
        key=lambda x: (float(x.get("week_pnl") or 0), float(x.get("week_roi") or 0)),
        reverse=True,
    )
    for r in whales:
        if len(out) >= DEEP_WHALE_N:
            break
        addr = str(r.get("addr") or "")
        if not addr or addr in seen:
            continue
        seen.add(addr)
        out.append(r)

    mid = [
        r
        for r in cands
        if LIVE_AV_MIN <= float(r.get("av") or 0) <= MID_AV_MAX
        and str(r.get("addr") or "") not in seen
    ]
    mid.sort(
        key=lambda x: (float(x.get("week_roi") or 0), float(x.get("week_pnl") or 0)),
        reverse=True,
    )
    for r in mid:
        if len(out) >= DEEP_WHALE_N + DEEP_MID_N:
            break
        addr = str(r.get("addr") or "")
        if not addr or addr in seen:
            continue
        seen.add(addr)
        out.append(r)

    for r in whales:
        if len(out) >= DEEP_TOP_N:
            break
        addr = str(r.get("addr") or "")
        if not addr or addr in seen:
            continue
        seen.add(addr)
        out.append(r)
    return out[:DEEP_TOP_N]


def _fill_notional(f: dict[str, Any]) -> float:
    try:
        return abs(float(f.get("sz") or 0) * float(f.get("px") or 0))
    except (TypeError, ValueError):
        return 0.0


def _merge_closed_events(
    fills: list[dict[str, Any]],
    *,
    gap_ms: int = 60_000,
) -> list[dict[str, Any]]:
    """Collapse fragmented close fills into one event per coin/dir burst.

    Same coin + same dir (e.g. Close Long) within ``gap_ms`` → one trade.
    Win/loss uses summed closedPnl. Opens (pnl≈0) are dropped.
    """
    items: list[dict[str, Any]] = []
    for f in fills:
        if not isinstance(f, dict):
            continue
        if f.get("closedPnl") in (None, ""):
            continue
        try:
            cp = float(f.get("closedPnl") or 0)
        except (TypeError, ValueError):
            continue
        if abs(cp) < 1e-16:
            continue
        try:
            ts = int(f.get("time") or 0)
        except (TypeError, ValueError):
            continue
        coin = str(f.get("coin") or "").strip() or "?"
        direction = str(f.get("dir") or "").strip() or str(f.get("side") or "")
        items.append(
            {
                "time": ts,
                "coin": coin,
                "dir": direction,
                "pnl": cp,
                "notional": _fill_notional(f),
                "n_fills": 1,
            }
        )
    items.sort(key=lambda x: (x["time"], x["coin"], x["dir"]))

    merged: list[dict[str, Any]] = []
    for it in items:
        if merged:
            prev = merged[-1]
            if (
                prev["coin"] == it["coin"]
                and prev["dir"] == it["dir"]
                and it["time"] - int(prev["time_last"]) <= gap_ms
            ):
                prev["pnl"] = float(prev["pnl"]) + float(it["pnl"])
                prev["notional"] = float(prev["notional"]) + float(it["notional"])
                prev["n_fills"] = int(prev["n_fills"]) + 1
                prev["time_last"] = it["time"]
                continue
        merged.append(
            {
                "time": it["time"],
                "time_last": it["time"],
                "coin": it["coin"],
                "dir": it["dir"],
                "pnl": it["pnl"],
                "notional": it["notional"],
                "n_fills": 1,
            }
        )
    return merged


def _fill_leg(f: dict[str, Any]) -> tuple[str, str] | None:
    """Return (open|close, long|short) from HL fill dir; None if unusable."""
    d = str(f.get("dir") or "").strip().lower()
    side = "long" if "long" in d else ("short" if "short" in d else "")
    if "close" in d and side:
        return "close", side
    if "open" in d and side:
        return "open", side
    # Flip styles e.g. "Long > Short": treat closedPnl≠0 as close of prior side
    try:
        cp = float(f.get("closedPnl") or 0)
    except (TypeError, ValueError):
        cp = 0.0
    if abs(cp) > 1e-16 and side:
        return "close", side
    if side:
        return "open", side
    return None


def _merge_leg_bursts(
    fills: list[dict[str, Any]],
    *,
    gap_ms: int = 60_000,
) -> list[dict[str, Any]]:
    """Merge open/close fill fragments: same coin + leg + side within gap_ms."""
    items: list[dict[str, Any]] = []
    for f in fills:
        if not isinstance(f, dict):
            continue
        leg = _fill_leg(f)
        if not leg:
            continue
        kind, side = leg
        try:
            ts = int(f.get("time") or 0)
        except (TypeError, ValueError):
            continue
        try:
            cp = float(f.get("closedPnl") or 0)
        except (TypeError, ValueError):
            cp = 0.0
        coin = str(f.get("coin") or "").strip() or "?"
        items.append(
            {
                "time": ts,
                "time_last": ts,
                "coin": coin,
                "kind": kind,
                "side": side,
                "pnl": cp if kind == "close" else 0.0,
                "notional": _fill_notional(f),
                "n_fills": 1,
            }
        )
    items.sort(key=lambda x: (x["time"], x["coin"], x["kind"], x["side"]))

    merged: list[dict[str, Any]] = []
    for it in items:
        if merged:
            prev = merged[-1]
            if (
                prev["coin"] == it["coin"]
                and prev["kind"] == it["kind"]
                and prev["side"] == it["side"]
                and it["time"] - int(prev["time_last"]) <= gap_ms
            ):
                prev["pnl"] = float(prev["pnl"]) + float(it["pnl"])
                prev["notional"] = float(prev["notional"]) + float(it["notional"])
                prev["n_fills"] = int(prev["n_fills"]) + 1
                prev["time_last"] = it["time"]
                continue
        merged.append(dict(it))
    return merged


def _round_trips_60s(
    fills: list[dict[str, Any]],
    *,
    gap_ms: int = 60_000,
) -> list[dict[str, Any]]:
    """Legacy: open burst + matching close burst (60s merge) = 1 trip.

    Prefer ``_round_trips_flat`` for desk screening (true open→flat cycles).
    """
    bursts = _merge_leg_bursts(fills, gap_ms=gap_ms)
    open_q: dict[tuple[str, str], list[dict[str, Any]]] = {}
    trips: list[dict[str, Any]] = []

    for b in bursts:
        key = (str(b["coin"]), str(b["side"]))
        if b["kind"] == "open":
            open_q.setdefault(key, []).append(b)
            continue
        # close
        q = open_q.get(key) or []
        opn = q.pop(0) if q else None
        trips.append(
            {
                "coin": b["coin"],
                "side": b["side"],
                "pnl": float(b.get("pnl") or 0),
                "notional": float(b.get("notional") or 0),
                "open_time": None if opn is None else int(opn["time"]),
                "close_time": int(b["time_last"]),
                "paired": opn is not None,
                "n_fills": int(b.get("n_fills") or 1) + (int(opn.get("n_fills") or 0) if opn else 0),
            }
        )
    return trips


def _signed_fill_sz(f: dict[str, Any]) -> float:
    """HL fill → signed size (+ long / − short inventory), matching hyper-track."""
    try:
        sz = abs(float(f.get("sz") or 0))
    except (TypeError, ValueError):
        return 0.0
    if sz <= 0:
        return 0.0
    side = str(f.get("side") or "").strip().upper()
    if side in ("B", "BUY"):
        return sz
    if side in ("A", "S", "SELL", "ASK"):
        return -sz
    d = str(f.get("dir") or "").strip().lower()
    if "long" in d and "close" in d:
        return -sz
    if "short" in d and "close" in d:
        return sz
    if "long" in d:
        return sz
    if "short" in d:
        return -sz
    return 0.0


def _round_trips_flat(fills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """True round-trip: per-coin position flat → open → back to flat (or flip).

    Same idea as hyper-track: one cycle when inventory crosses/returns through 0.
    PnL = sum(closedPnl) − sum(fee) over fills in the cycle.
    Scaling in/out stays one trip until flat — does not inflate WR with fragments.
    """
    items: list[dict[str, Any]] = []
    for f in fills:
        if not isinstance(f, dict):
            continue
        signed = _signed_fill_sz(f)
        if signed == 0.0:
            continue
        try:
            ts = int(f.get("time") or 0)
        except (TypeError, ValueError):
            continue
        try:
            px = float(f.get("px") or 0)
        except (TypeError, ValueError):
            px = 0.0
        try:
            cp = float(f.get("closedPnl") or 0)
        except (TypeError, ValueError):
            cp = 0.0
        try:
            fee = float(f.get("fee") or 0)
        except (TypeError, ValueError):
            fee = 0.0
        coin = str(f.get("coin") or "").strip() or "?"
        items.append(
            {
                "time": ts,
                "coin": coin,
                "signed": signed,
                "price": px,
                "closed_pnl": cp,
                "fee": fee,
                "notional": abs(signed) * px,
            }
        )
    items.sort(key=lambda x: (x["time"], x["coin"]))

    # per coin running state
    state: dict[str, dict[str, Any]] = {}
    trips: list[dict[str, Any]] = []
    eps = 1e-12

    for it in items:
        coin = it["coin"]
        st = state.setdefault(
            coin,
            {
                "net": 0.0,
                "fills_pnl": 0.0,
                "fills_fee": 0.0,
                "fills_ntl": 0.0,
                "n_fills": 0,
                "entry_ntl": 0.0,
                "entry_sz": 0.0,
                "open_time": 0,
            },
        )
        prev = float(st["net"])
        signed = float(it["signed"])
        new = prev + signed
        st["fills_pnl"] = float(st["fills_pnl"]) + float(it["closed_pnl"])
        st["fills_fee"] = float(st["fills_fee"]) + float(it["fee"])
        st["fills_ntl"] = float(st["fills_ntl"]) + float(it["notional"])
        st["n_fills"] = int(st["n_fills"]) + 1

        if abs(prev) <= eps and abs(new) > eps:
            # open from flat
            st["open_time"] = int(it["time"])
            st["entry_ntl"] = float(it["notional"])
            st["entry_sz"] = abs(signed)
            st["fills_pnl"] = float(it["closed_pnl"])
            st["fills_fee"] = float(it["fee"])
            st["fills_ntl"] = float(it["notional"])
            st["n_fills"] = 1
        elif abs(prev) > eps and (prev > 0) == (new > 0) and abs(new) > abs(prev) + eps:
            # scale in same direction
            st["entry_ntl"] = float(st["entry_ntl"]) + float(it["notional"])
            st["entry_sz"] = float(st["entry_sz"]) + abs(signed)

        # Crossed flat or flipped side → one completed cycle
        flipped = abs(prev) > eps and abs(new) > eps and (prev > 0) != (new > 0)
        flat_now = abs(prev) > eps and abs(new) <= eps
        if flat_now or flipped:
            side = "long" if prev > 0 else "short"
            pnl_gross = float(st["fills_pnl"])
            fees = float(st["fills_fee"])
            net = pnl_gross - fees
            entry_sz = float(st["entry_sz"]) or abs(prev)
            avg_entry = (
                (float(st["entry_ntl"]) / entry_sz) if entry_sz > 0 else float(it["price"])
            )
            trips.append(
                {
                    "coin": coin,
                    "side": side,
                    "pnl": net,
                    "pnl_gross": pnl_gross,
                    "fees": fees,
                    "notional": float(st["entry_ntl"]) or abs(prev) * avg_entry,
                    "open_time": int(st.get("open_time") or it["time"]),
                    "close_time": int(it["time"]),
                    "paired": True,
                    "n_fills": int(st["n_fills"]),
                }
            )
            # reset; if flipped, remainder starts a new cycle
            if abs(new) > eps:
                st["net"] = new
                st["open_time"] = int(it["time"])
                st["entry_ntl"] = abs(new) * float(it["price"])
                st["entry_sz"] = abs(new)
                st["fills_pnl"] = 0.0
                st["fills_fee"] = 0.0
                st["fills_ntl"] = abs(new) * float(it["price"])
                st["n_fills"] = 1
            else:
                st["net"] = 0.0
                st["fills_pnl"] = 0.0
                st["fills_fee"] = 0.0
                st["fills_ntl"] = 0.0
                st["n_fills"] = 0
                st["entry_ntl"] = 0.0
                st["entry_sz"] = 0.0
                st["open_time"] = 0
            continue

        st["net"] = new

    return trips


def _wr_from_round_trips(trips: list[dict[str, Any]]) -> tuple[float | None, int, int, int]:
    """Return wr, wins, losses, n_trades (wins+losses; flat pnl ignored in WR)."""
    wins = losses = 0
    for t in trips:
        pnl = float(t.get("pnl") or 0)
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
    n = wins + losses
    wr = (wins / n) if n else None
    return wr, wins, losses, n



def _deep_screen_one(c: dict[str, Any], now_ms: int) -> dict[str, Any]:
    addr = c["addr"]
    state = _hl_info({"type": "clearinghouseState", "user": addr})
    start_ms = int(now_ms) - 7 * 86400 * 1000
    fills = _fetch_fills_7d(addr, start_ms)
    if not isinstance(fills, list):
        fills = []

    pos: list[str] = []
    for item in state.get("assetPositions") or []:
        p = item.get("position") or {}
        if abs(float(p.get("szi") or 0)) > 1e-12:
            pos.append(str(p.get("coin")))

    recent = [
        f
        for f in fills
        if isinstance(f, dict) and (now_ms - int(f.get("time") or 0)) < 7 * 86400 * 1000
    ]
    d1 = [
        f
        for f in fills
        if isinstance(f, dict) and (now_ms - int(f.get("time") or 0)) < 86400 * 1000
    ]
    raw_fph24 = len(d1) / 24.0

    # WR / trips / pace on Bitget-mapped fills only (screen_7d_good recipe).
    bg = [f for f in recent if _is_bitget_mapped(str(f.get("coin") or ""))]
    bg_24 = [f for f in d1 if _is_bitget_mapped(str(f.get("coin") or ""))]
    bitget_share = (len(bg) / len(recent)) if recent else 0.0

    trips = _round_trips_flat(bg)
    wr7, wins, losses, ncl = _wr_from_round_trips(trips)
    closed7 = ncl
    paired = sum(1 for t in trips if t.get("paired"))
    pair_rate = (paired / len(trips)) if trips else 0.0
    pnl7 = sum(float(t.get("pnl") or 0) for t in trips)

    legs_24 = _merge_leg_bursts(bg_24, gap_ms=60_000)
    lph24 = len(legs_24) / 24.0
    # Gate field fph24 = merged-leg pace (NOT raw fragments).
    fph24 = lph24

    # scratch: tiny |pnl|/notional on completed trips
    scratch = 0
    for t in trips:
        n = float(t.get("notional") or 0)
        cp = float(t.get("pnl") or 0)
        if n > 0 and abs(cp) / n < 0.0005:
            scratch += 1
    scratch_r = (scratch / len(trips)) if trips else None

    # keep legacy closed-burst merge available for debugging
    closed_raw = [f for f in recent if f.get("closedPnl") not in (None, "")]
    events = _merge_closed_events(closed_raw, gap_ms=60_000)

    coins = Counter(str(f.get("coin")) for f in bg if isinstance(f, dict))
    if not coins:
        coins = Counter(str(f.get("coin")) for f in recent if isinstance(f, dict))
    if not coins:
        coins = Counter(str(f.get("coin")) for f in fills[:200] if isinstance(f, dict))
    top = coins.most_common(8)
    total = sum(coins.values()) or 1
    c1 = top[0][1] / total if top else 0.0
    c2 = (top[0][1] + top[1][1]) / total if len(top) > 1 else c1
    stock_share = sum(v for k, v in coins.items() if _is_stock_coin(k)) / total
    crypto_share = 1.0 - stock_share
    # Bitget-mappable share (crypto + stock both OK when user allows stocks).
    major_share = bitget_share if recent else (
        sum(v for k, v in coins.items() if _is_bitget_mapped(k)) / total
    )
    follow_coins = [k for k, _ in top if _is_bitget_mapped(k)][:4]

    if c1 >= 0.55 and top:
        specialty = f"single:{top[0][0]}"
    elif c2 >= 0.70 and len(top) >= 2:
        specialty = f"duo:{top[0][0]}+{top[1][0]}"
    elif top:
        specialty = "mix:" + ",".join(x[0] for x in top[:3])
    else:
        specialty = "unknown"

    live_av = float((state.get("marginSummary") or {}).get("accountValue") or 0)
    lb_av = float(c.get("av") or 0)
    # ROI vs money actually on account (not inflated leaderboard shell)
    eq_roi = live_av if live_av >= 1_000 else max(live_av, lb_av, 1.0)
    pnl7_roi = (pnl7 / eq_roi) if eq_roi > 0 else 0.0
    live_av_ratio = (live_av / lb_av) if lb_av > 1e-9 else (1.0 if live_av > 0 else 0.0)

    # Score: profit + followability first; WR is supporting evidence only.
    wr_v = float(wr7 or 0)
    sweet_lo, sweet_hi = 0.1, 4.0
    if sweet_lo <= fph24 <= sweet_hi:
        pace_pen = 0.0
    else:
        pace_pen = abs(fph24 - 1.5) * 0.25
    conc_bonus = 1.0 if c2 >= 0.55 else 0.5
    crypto_bonus = min(max(crypto_share, 0.0), 1.0) * 0.35
    stock_pen = 0.0
    scratch_pen = float(scratch_r or 0) * 1.5
    roi_term = min(max(float(c.get("week_roi") or 0), 0.0) / 0.50, 3.0)
    pnl_term = min(max(pnl7, 0.0) / 20_000.0, 2.0)
    sample_pen = 1.2 if closed7 < 8 else (0.4 if closed7 < 15 else 0.0)
    mess_pen = 0.1 * max(0, len(pos) - 6)
    idle_pen = 0.5 if fph24 < 0.05 and closed7 < 12 else 0.0
    wr_term = wr_v * 1.0 if closed7 >= 8 else wr_v * 0.35
    pair_bonus = min(max(pair_rate, 0.0), 1.0) * 0.8
    copy_score = (
        roi_term
        + pnl_term
        + wr_term
        + pair_bonus
        + crypto_bonus
        + conc_bonus
        - stock_pen
        - pace_pen
        - scratch_pen
        - mess_pen
        - idle_pen
        - sample_pen
    )

    return {
        **c,
        "npos": len(pos),
        "coins_pos": pos[:8],
        "fills": len(fills),
        "fills7": len(recent),
        "closed7": closed7,
        "closed7_raw": len(closed_raw),
        "closed7_burst": len(events),
        "round_trips": closed7,
        "paired": paired,
        "pair_rate": round(pair_rate, 3),
        "wins": wins,
        "losses": losses,
        "wr7": None if wr7 is None else round(wr7, 4),
        "wr": None if wr7 is None else round(wr7, 4),
        "pnl7": round(pnl7, 2),
        "pnl7_roi": round(pnl7_roi, 4),
        "scratch": None if scratch_r is None else round(scratch_r, 3),
        "fph24": round(fph24, 2),
        "lph24": round(lph24, 2),
        "raw_fph24": round(raw_fph24, 2),
        "bitget_share": round(bitget_share, 3),
        "c1": round(c1, 3),
        "c2": round(c2, 3),
        "major_share": round(major_share, 3),
        "crypto_share": round(crypto_share, 3),
        "stock_share": round(stock_share, 3),
        "specialty": specialty,
        "follow_coins": follow_coins,
        "top_coins": [{"coin": k, "n": n} for k, n in top[:5]],
        "live_av": round(live_av, 2),
        "live_av_ratio": round(live_av_ratio, 3),
        "copy_score": round(copy_score, 3),
        "hl_url": f"https://app.hyperliquid.xyz/explorer/address/{addr}",
        "wr_note": (
            "wr7/pnl7/lph on Bitget-mappable 60s round-trips; "
            "fph24==lph24 merged legs (raw_fph24=fragments)"
        ),
        "easy_follow": bool(
            major_share >= 0.50
            and closed7 >= 8
            and pair_rate >= 0.50
            and float(wr7 or 0) >= 0.55
            and float(c.get("week_roi") or 0) >= 0.10
        ),
    }


def _passes_lane(r: dict[str, Any], criteria: dict[str, Any]) -> bool:
    # fph24 is merged-leg lph24 after v6
    fph = float(r.get("fph24") or r.get("lph24") or 0)
    if fph < float(criteria["fph24_min"]) or fph > float(criteria["fph24_max"]):
        return False
    if int(r.get("fills7") or 0) < int(criteria["fills7_min"]):
        return False
    if int(r.get("closed7") or 0) < int(criteria["closed7_min"]):
        return False
    wr = r.get("wr7")
    if wr is None or float(wr) < float(criteria["wr7_min"]):
        return False
    pair_min = float(criteria.get("pair_rate_min") or 0)
    if pair_min > 0 and float(r.get("pair_rate") or 0) < pair_min:
        return False
    week_roi_min = float(criteria.get("week_roi_min") or 0)
    if week_roi_min > 0 and float(r.get("week_roi") or 0) < week_roi_min:
        return False
    if float(r.get("pnl7") or 0) <= float(criteria.get("pnl7_min") or 0):
        return False
    roi_min = float(criteria.get("pnl7_roi_min") or 0)
    if roi_min > 0:
        roi = r.get("pnl7_roi")
        if roi is None:
            eq = max(float(r.get("live_av") or 0), float(r.get("av") or 0), 1.0)
            roi = float(r.get("pnl7") or 0) / eq
        if float(roi) < roi_min:
            return False
    # Month must also be green for returns-first copyable (skip if criteria says None)
    month_floor = criteria.get("month_pnl_min")
    if month_floor is not None and float(r.get("month_pnl") or 0) <= float(month_floor):
        return False
    scratch = r.get("scratch")
    if scratch is not None and float(scratch) > float(criteria["scratch_max"]):
        return False
    if float(r.get("live_av") or 0) < float(criteria["live_av_min"]):
        return False
    ratio_min = float(criteria.get("live_av_ratio_min") or 0)
    if ratio_min > 0 and float(r.get("live_av_ratio") or 0) < ratio_min:
        return False
    if float(r.get("major_share") or 0) < float(criteria["major_share_min"]):
        return False
    crypto_min = criteria.get("crypto_share_min")
    if crypto_min is not None and float(r.get("crypto_share") or 0) < float(crypto_min):
        return False
    stock_max = criteria.get("stock_share_max")
    if stock_max is not None and float(r.get("stock_share") or 0) > float(stock_max):
        return False
    if criteria.get("crypto_only"):
        top = r.get("top_coins") or []
        if top and _is_stock_coin(str((top[0] or {}).get("coin") or "")):
            return False
    if float(r.get("c2") or 0) < float(criteria.get("c2_min") or 0):
        return False
    if int(r.get("npos") or 0) > int(criteria.get("npos_max") or 999):
        return False
    follow = r.get("follow_coins") or []
    if len(follow) < int(criteria.get("follow_coins_min") or 0):
        return False
    if criteria.get("id") == "copyable" and r.get("watchlist_id") == "reject":
        return False
    return True


def _public_pick(r: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "addr",
        "av",
        "live_av",
        "live_av_ratio",
        "wr",
        "wr7",
        "fills7",
        "closed7",
        "fph24",
        "lph24",
        "raw_fph24",
        "pair_rate",
        "pnl7",
        "pnl7_roi",
        "scratch",
        "c1",
        "c2",
        "major_share",
        "bitget_share",
        "crypto_share",
        "stock_share",
        "specialty",
        "follow_coins",
        "copy_score",
        "easy_follow",
        "day_pnl",
        "week_pnl",
        "month_pnl",
        "week_vlm",
        "week_roi",
        "npos",
        "coins_pos",
        "top_coins",
        "hl_url",
        "on_watchlist",
        "watchlist_id",
        "wr_note",
    )
    return {k: r.get(k) for k in keys}


def _lane_payload(
    *,
    criteria: dict[str, Any],
    candidate_count: int,
    scanned_count: int,
    picks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": criteria["id"],
        "label": criteria["label"],
        "criteria": criteria,
        "candidate_count": candidate_count,
        "scanned_count": scanned_count,
        "pick_count": len(picks),
        "picks": picks,
    }


def run_screen(*, sleep_sec: float = 0.55) -> dict[str, Any]:
    """Run copyable + watch lanes; persist board JSON."""
    if not _run_lock.acquire(blocking=False):
        snap = load_board()
        if snap:
            out = dict(snap)
            out["refresh_skipped"] = True
            out["note"] = "screen already in progress"
            return out
        raise RuntimeError("hl wr screen already in progress")

    started = time.time()
    try:
        # Warm Bitget contract cache once so mappable checks are consistent
        try:
            from utils.hl_bitget_symbol_map import bitget_contract_set

            bitget_contract_set()
        except Exception:
            logger.warning("bitget contract warm failed; mappable checks use fallback")

        # Allow overlap with desk / reject list (tag only; do not skip)
        desk_map = _watchlist_by_addr()
        lb = _http_get_json(LEADERBOARD_URL)
        raw_rows = lb.get("leaderboardRows") or []
        parsed: list[dict[str, Any]] = []
        by_addr: dict[str, dict[str, Any]] = {}
        for row in raw_rows:
            item = _parse_leaderboard_row(row)
            if item:
                parsed.append(item)
                by_addr[str(item["addr"]).lower()] = item

        cands = _leaderboard_candidates(parsed, set())
        deep_list = _ensure_desk_in_deep(_select_deep_list(cands), by_addr, desk_map)

        now_ms = int(time.time() * 1000)
        scanned: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for c in deep_list:
            try:
                # pacing is mostly inside _fetch_fills_7d; keep a small gap between wallets
                if sleep_sec > 0:
                    time.sleep(min(float(sleep_sec), 0.25))
                scanned.append(_deep_screen_one(c, now_ms))
            except Exception as exc:
                logger.warning("deep screen failed %s: %s", c.get("addr"), exc)
                errors.append({"addr": str(c.get("addr")), "error": str(exc)})

        for r in scanned:
            wid = desk_map.get(str(r.get("addr") or "").lower())
            if wid:
                r["on_watchlist"] = True
                r["watchlist_id"] = wid

        # Same ranking as screen_7d_good: week ROI → WR → trips → closed PnL
        copy_picks = [_public_pick(r) for r in scanned if _passes_lane(r, CRITERIA_COPYABLE)]
        copy_picks.sort(
            key=lambda x: (
                float(x.get("week_roi") or 0),
                float(x.get("wr7") or 0),
                int(x.get("closed7") or 0),
                float(x.get("pnl7") or 0),
            ),
            reverse=True,
        )
        copy_picks = copy_picks[:PICK_TOP_N]

        watch_picks = [_public_pick(r) for r in scanned if _passes_lane(r, CRITERIA_WATCH)]
        # exclude already in copyable for cleaner second tab
        copy_addrs = {p["addr"] for p in copy_picks}
        watch_picks = [p for p in watch_picks if p.get("addr") not in copy_addrs]
        watch_picks.sort(
            key=lambda x: (
                float(x.get("copy_score") or 0),
                float(x.get("pnl7_roi") or 0),
                float(x.get("wr7") or 0),
            ),
            reverse=True,
        )
        watch_picks = watch_picks[:PICK_TOP_N]

        lanes = {
            "copyable": _lane_payload(
                criteria=CRITERIA_COPYABLE,
                candidate_count=len(cands),
                scanned_count=len(scanned),
                picks=copy_picks,
            ),
            "watch": _lane_payload(
                criteria=CRITERIA_WATCH,
                candidate_count=len(cands),
                scanned_count=len(scanned),
                picks=watch_picks,
            ),
            # legacy aliases so old UI/cache readers don't explode
            "turnover": _lane_payload(
                criteria={**CRITERIA_COPYABLE, "id": "turnover", "label": "可跟"},
                candidate_count=len(cands),
                scanned_count=len(scanned),
                picks=copy_picks,
            ),
            "midsize": _lane_payload(
                criteria={**CRITERIA_WATCH, "id": "midsize", "label": "宽观察"},
                candidate_count=len(cands),
                scanned_count=len(scanned),
                picks=watch_picks,
            ),
        }

        on_desk = sum(1 for p in copy_picks + watch_picks if p.get("on_watchlist"))
        board = {
            "ok": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "venue": "hyperliquid",
            "screen_version": 6,
            "lanes": lanes,
            "criteria": dict(CRITERIA_COPYABLE),
            "candidate_count": len(cands),
            "scanned_count": len(scanned),
            "pick_count": len(copy_picks),
            "picks": copy_picks,
            "skipped_known": 0,
            "on_watchlist_count": on_desk,
            "deep_unique": len(deep_list),
            "errors": errors[:20],
            "elapsed_sec": round(time.time() - started, 1),
            "fills_source": "userFillsByTime_7d",
            "rank": "week_roi,wr7,closed7,pnl7",
        }
        _save_board(board)
        logger.info(
            "hl wr screen v6 done: copy=%s watch=%s deep=%s on_desk=%s elapsed=%.1fs",
            len(copy_picks),
            len(watch_picks),
            len(deep_list),
            on_desk,
            board["elapsed_sec"],
        )
        return board
    finally:
        _run_lock.release()


def _save_board(board: dict[str, Any]) -> None:
    path = _board_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with _lock:
        tmp.write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def load_board() -> dict[str, Any] | None:
    path = _board_path()
    if not path.exists():
        return None
    try:
        with _lock:
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("failed to load %s", path)
        return None


def empty_board(*, note: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": True,
        "generated_at": None,
        "venue": "hyperliquid",
        "screen_version": 6,
        "lanes": {
            "copyable": _lane_payload(
                criteria=CRITERIA_COPYABLE,
                candidate_count=0,
                scanned_count=0,
                picks=[],
            ),
            "watch": _lane_payload(
                criteria=CRITERIA_WATCH,
                candidate_count=0,
                scanned_count=0,
                picks=[],
            ),
        },
        "criteria": dict(CRITERIA_COPYABLE),
        "candidate_count": 0,
        "scanned_count": 0,
        "pick_count": 0,
        "skipped_known": 0,
        "picks": [],
        "errors": [],
        "snapshot_source": "empty",
    }
    if note:
        out["note"] = note
    return out


def get_board(*, refresh: bool = False) -> dict[str, Any]:
    if refresh:
        board = run_screen()
        out = dict(board)
        out["snapshot_source"] = "live"
        return out
    snap = load_board()
    if snap:
        out = dict(snap)
        out["snapshot_source"] = "cache"
        lanes = out.get("lanes") or {}
        # migrate old boards: expose copyable/watch aliases
        if "copyable" not in lanes and "turnover" in lanes:
            lanes = dict(lanes)
            lanes["copyable"] = lanes.get("turnover")
            lanes["watch"] = lanes.get("midsize") or lanes.get("turnover")
            out["lanes"] = lanes
        return out
    return empty_board(note="尚无日筛快照；等待定时任务或手动 refresh=1")

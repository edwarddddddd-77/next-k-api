"""Daily Hyperliquid wallet screen — copyability first (not AV size bands).

Primary lane ``copyable`` (可跟):
  mid pace, positive 7D closedPnl, concentration, mappable coins, low scratch.

Secondary lane ``watch`` (宽观察):
  same idea, looser pace/scratch for research only.
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

LIVE_AV_MIN = 5_000.0
WEEK_PNL_MIN = 0.0
WEEK_VLM_MIN = 30_000.0
DEEP_TOP_N = 55
PICK_TOP_N = 15

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
}

CRITERIA_COPYABLE = {
    "id": "copyable",
    "label": "可跟",
    "week_pnl_min": WEEK_PNL_MIN,
    "week_vlm_min": WEEK_VLM_MIN,
    "live_av_min": LIVE_AV_MIN,
    "fph24_min": 0.15,
    "fph24_max": 25.0,
    "fph24_sweet": [1.0, 12.0],
    "fills7_min": 12,
    "closed7_min": 5,
    "wr7_min": 0.52,
    "pnl7_min": 0.0,
    "scratch_max": 0.78,
    "major_share_min": 0.25,
    "deep_top_n": DEEP_TOP_N,
    "pick_top_n": PICK_TOP_N,
    "wr_window": "7d_closedPnl",
    "note": "不计盘口大小；看节奏/真赚/可映射/scratch",
}

CRITERIA_WATCH = {
    **CRITERIA_COPYABLE,
    "id": "watch",
    "label": "宽观察",
    "fph24_max": 35.0,
    "wr7_min": 0.50,
    "scratch_max": 0.85,
    "major_share_min": 0.15,
    "note": "同可跟逻辑，节奏与 scratch 更宽，仅供观察",
}

_lock = threading.Lock()
_run_lock = threading.Lock()


def _board_path() -> Path:
    return resolve_data_dir() / BOARD_NAME


def _http_get_json(url: str, *, timeout: float = 60.0) -> Any:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "next-k-hl-wr-screen/2.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _hl_info(body: dict) -> Any:
    req = urllib.request.Request(
        INFO_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "next-k-hl-wr-screen/2.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


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


def _is_followable_coin(coin: str) -> bool:
    c = str(coin or "")
    cu = c.upper()
    if cu in MAJORS:
        return True
    if c.startswith("xyz:") or c.startswith("XYZ:"):
        return c.split(":", 1)[-1].upper() in STOCKS
    return False


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
    """No AV size band — only week profit + activity."""
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
        cands.append(dict(base))
    cands.sort(key=lambda x: (x["week_pnl"], x["week_vlm"]), reverse=True)
    return cands


def _deep_screen_one(c: dict[str, Any], now_ms: int) -> dict[str, Any]:
    addr = c["addr"]
    state = _hl_info({"type": "clearinghouseState", "user": addr})
    fills = _hl_info({"type": "userFills", "user": addr})
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
    fph24 = len(d1) / 24.0

    closed7 = [f for f in recent if f.get("closedPnl") not in (None, "")]
    wins = losses = 0
    pnl7 = 0.0
    scratch = 0
    for f in closed7:
        try:
            cp = float(f.get("closedPnl") or 0)
        except (TypeError, ValueError):
            continue
        pnl7 += cp
        if cp > 0:
            wins += 1
        elif cp < 0:
            losses += 1
        try:
            n = abs(float(f.get("sz") or 0) * float(f.get("px") or 0))
            if n > 0 and abs(cp) / n < 0.0005:
                scratch += 1
        except (TypeError, ValueError):
            pass
    ncl = wins + losses
    wr7 = (wins / ncl) if ncl else None
    scratch_r = (scratch / len(closed7)) if closed7 else None

    coins = Counter(str(f.get("coin")) for f in recent if isinstance(f, dict))
    if not coins:
        coins = Counter(str(f.get("coin")) for f in fills[:200] if isinstance(f, dict))
    top = coins.most_common(8)
    total = sum(coins.values()) or 1
    c1 = top[0][1] / total if top else 0.0
    c2 = (top[0][1] + top[1][1]) / total if len(top) > 1 else c1
    major_share = sum(v for k, v in coins.items() if _is_followable_coin(k)) / total
    follow_coins = [k for k, _ in top if _is_followable_coin(k)][:4]

    if c1 >= 0.55 and top:
        specialty = f"single:{top[0][0]}"
    elif c2 >= 0.70 and len(top) >= 2:
        specialty = f"duo:{top[0][0]}+{top[1][0]}"
    elif top:
        specialty = "mix:" + ",".join(x[0] for x in top[:3])
    else:
        specialty = "unknown"

    live_av = float((state.get("marginSummary") or {}).get("accountValue") or 0)

    # copy score (same spirit as deep_hl_copy_picks)
    wr_v = float(wr7 or 0)
    pace_pen = 0.0 if 1.0 <= fph24 <= 12.0 else abs(fph24 - 6.0) * 0.15
    conc_bonus = 1.2 if c2 >= 0.65 else (1.0 if c2 >= 0.45 else 0.7)
    scratch_pen = float(scratch_r or 0) * 1.5
    copy_score = wr_v * 2.0 + min(max(pnl7, 0) / 20000.0, 2.0) + conc_bonus - pace_pen - scratch_pen

    return {
        **c,
        "npos": len(pos),
        "coins_pos": pos[:8],
        "fills": len(fills),
        "fills7": len(recent),
        "closed7": ncl,
        "wr7": None if wr7 is None else round(wr7, 4),
        "wr": None if wr7 is None else round(wr7, 4),
        "pnl7": round(pnl7, 2),
        "scratch": None if scratch_r is None else round(scratch_r, 3),
        "fph24": round(fph24, 2),
        "c1": round(c1, 3),
        "c2": round(c2, 3),
        "major_share": round(major_share, 3),
        "specialty": specialty,
        "follow_coins": follow_coins,
        "top_coins": [{"coin": k, "n": n} for k, n in top[:5]],
        "live_av": round(live_av, 2),
        "copy_score": round(copy_score, 3),
        "hl_url": f"https://app.hyperliquid.xyz/explorer/address/{addr}",
    }


def _passes_lane(r: dict[str, Any], criteria: dict[str, Any]) -> bool:
    fph = float(r.get("fph24") or 0)
    if fph < float(criteria["fph24_min"]) or fph > float(criteria["fph24_max"]):
        return False
    if int(r.get("fills7") or 0) < int(criteria["fills7_min"]):
        return False
    if int(r.get("closed7") or 0) < int(criteria["closed7_min"]):
        return False
    wr = r.get("wr7")
    if wr is None or float(wr) < float(criteria["wr7_min"]):
        return False
    if float(r.get("pnl7") or 0) <= float(criteria["pnl7_min"]):
        return False
    scratch = r.get("scratch")
    if scratch is not None and float(scratch) > float(criteria["scratch_max"]):
        return False
    if float(r.get("live_av") or 0) < float(criteria["live_av_min"]):
        return False
    if float(r.get("major_share") or 0) < float(criteria["major_share_min"]):
        # allow if specialty top is followable
        follow = r.get("follow_coins") or []
        if not follow:
            return False
    return True


def _public_pick(r: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "addr",
        "av",
        "live_av",
        "wr",
        "wr7",
        "fills7",
        "closed7",
        "fph24",
        "pnl7",
        "scratch",
        "c1",
        "c2",
        "major_share",
        "specialty",
        "follow_coins",
        "copy_score",
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
        # Allow overlap with desk A–E / reject list (tag only; do not skip)
        desk_map = _watchlist_by_addr()
        lb = _http_get_json(LEADERBOARD_URL)
        raw_rows = lb.get("leaderboardRows") or []
        parsed: list[dict[str, Any]] = []
        for row in raw_rows:
            item = _parse_leaderboard_row(row)
            if item:
                parsed.append(item)

        cands = _leaderboard_candidates(parsed, set())
        deep_list = cands[:DEEP_TOP_N]

        now_ms = int(time.time() * 1000)
        scanned: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for c in deep_list:
            try:
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                scanned.append(_deep_screen_one(c, now_ms))
            except Exception as exc:
                logger.warning("deep screen failed %s: %s", c.get("addr"), exc)
                errors.append({"addr": str(c.get("addr")), "error": str(exc)})

        for r in scanned:
            wid = desk_map.get(str(r.get("addr") or "").lower())
            if wid:
                r["on_watchlist"] = True
                r["watchlist_id"] = wid

        copy_picks = [_public_pick(r) for r in scanned if _passes_lane(r, CRITERIA_COPYABLE)]
        copy_picks.sort(
            key=lambda x: (float(x.get("copy_score") or 0), float(x.get("pnl7") or 0)),
            reverse=True,
        )
        copy_picks = copy_picks[:PICK_TOP_N]

        watch_picks = [_public_pick(r) for r in scanned if _passes_lane(r, CRITERIA_WATCH)]
        # exclude already in copyable for cleaner second tab
        copy_addrs = {p["addr"] for p in copy_picks}
        watch_picks = [p for p in watch_picks if p.get("addr") not in copy_addrs]
        watch_picks.sort(
            key=lambda x: (float(x.get("copy_score") or 0), float(x.get("pnl7") or 0)),
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
            "screen_version": 2,
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
        }
        _save_board(board)
        logger.info(
            "hl wr screen v2 done: copy=%s watch=%s deep=%s on_desk=%s elapsed=%.1fs",
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
        "screen_version": 2,
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

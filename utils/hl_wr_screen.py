"""Daily Hyperliquid short-term high win-rate wallet screen (read-only).

Two lanes (both run each day):
1. turnover — original A/B style: high turnover + recent closedPnl WR
2. midsize  — C/D/E style: mid AV + 7D closedPnl WR + fill-rate cap
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

WR_MIN = 0.55
LIVE_AV_MIN = 10_000.0
PICK_TOP_N = 15

# Lane: turnover (original A/B)
TO_AV_MIN = 50_000.0
TO_AV_MAX = 3_000_000.0
TO_TURN_W_MIN = 15.0
TO_TURN_M_MIN = 40.0
TO_WEEK_ROI_MAX = 1.5
TO_DEEP_TOP_N = 25
TO_CLOSED_MIN = 10

# Lane: midsize (C/D/E)
MS_AV_MIN = 15_000.0
MS_AV_MAX = 350_000.0
MS_WEEK_VLM_MIN = 50_000.0
MS_DEEP_TOP_N = 30
MS_FILLS7_MIN = 8
MS_CLOSED7_MIN = 5
MS_FPH24_MAX = 40.0

CRITERIA_TURNOVER = {
    "id": "turnover",
    "label": "高换手",
    "av_band": [TO_AV_MIN, TO_AV_MAX],
    "turn_w_min": TO_TURN_W_MIN,
    "turn_m_min": TO_TURN_M_MIN,
    "week_pnl_min": 0,
    "week_roi_max": TO_WEEK_ROI_MAX,
    "wr_min": WR_MIN,
    "closed_min": TO_CLOSED_MIN,
    "live_av_min": LIVE_AV_MIN,
    "deep_top_n": TO_DEEP_TOP_N,
    "pick_top_n": PICK_TOP_N,
    "wr_window": "recent_closedPnl",
}

CRITERIA_MIDSIZE = {
    "id": "midsize",
    "label": "中盘7D",
    "av_band": [MS_AV_MIN, MS_AV_MAX],
    "week_vlm_min": MS_WEEK_VLM_MIN,
    "wr_min": WR_MIN,
    "fills7_min": MS_FILLS7_MIN,
    "closed7_min": MS_CLOSED7_MIN,
    "fph24_max": MS_FPH24_MAX,
    "live_av_min": LIVE_AV_MIN,
    "deep_top_n": MS_DEEP_TOP_N,
    "pick_top_n": PICK_TOP_N,
    "wr_window": "7d_closedPnl",
}

_lock = threading.Lock()
_run_lock = threading.Lock()


def _board_path() -> Path:
    return resolve_data_dir() / BOARD_NAME


def _http_get_json(url: str, *, timeout: float = 60.0) -> Any:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "next-k-hl-wr-screen/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _hl_info(body: dict) -> Any:
    req = urllib.request.Request(
        INFO_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "next-k-hl-wr-screen/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _known_watch_addrs() -> set[str]:
    try:
        from utils.hl_short_term import load_watchlist_doc

        doc = load_watchlist_doc()
        out: set[str] = set()
        for row in (doc.get("wallets") or []) + (doc.get("reject_for_now") or []):
            addr = str(row.get("address") or "").strip().lower()
            if addr:
                out.add(addr)
        return out
    except Exception:
        logger.exception("failed to load watchlist for screen skip set")
        return set()


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


def _turnover_candidates(rows: list[dict[str, Any]], skip: set[str]) -> list[dict[str, Any]]:
    cands: list[dict[str, Any]] = []
    for base in rows:
        if base["addr"] in skip:
            continue
        av = float(base["av"])
        if not (TO_AV_MIN <= av <= TO_AV_MAX):
            continue
        if base["turn_w"] < TO_TURN_W_MIN and base["turn_m"] < TO_TURN_M_MIN:
            continue
        if base["week_pnl"] <= 0:
            continue
        if base["week_roi"] > TO_WEEK_ROI_MAX:
            continue
        score = (
            base["turn_w"] * 0.4
            + (base["week_pnl"] / av) * 100
            + (1.0 if base["month_pnl"] > 0 else -5.0)
        )
        cands.append({**base, "score": round(score, 4)})
    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands


def _midsize_candidates(rows: list[dict[str, Any]], skip: set[str]) -> list[dict[str, Any]]:
    cands: list[dict[str, Any]] = []
    for base in rows:
        if base["addr"] in skip:
            continue
        av = float(base["av"])
        if not (MS_AV_MIN <= av <= MS_AV_MAX):
            continue
        if base["week_vlm"] < MS_WEEK_VLM_MIN:
            continue
        if base["month_pnl"] <= 0 and base["week_pnl"] <= 0:
            continue
        cands.append(dict(base))
    cands.sort(key=lambda x: (x["week_pnl"], x["month_pnl"]), reverse=True)
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

    # --- turnover WR: recent closedPnl sample (not window-limited) ---
    closed_all = [f for f in fills if isinstance(f, dict) and abs(float(f.get("closedPnl") or 0)) > 1e-9]
    if len(closed_all) < 20:
        closed_all = [
            f
            for f in fills
            if isinstance(f, dict) and f.get("closedPnl") not in (None, "0", 0, "0.0")
        ]
    wins_all = sum(1 for f in closed_all if float(f.get("closedPnl") or 0) > 0)
    losses_all = sum(1 for f in closed_all if float(f.get("closedPnl") or 0) < 0)
    closed_n = wins_all + losses_all
    wr_all = (wins_all / closed_n) if closed_n >= TO_CLOSED_MIN else None

    times = sorted(int(f.get("time") or 0) for f in fills if isinstance(f, dict) and f.get("time"))
    span_h = (times[-1] - times[0]) / 3_600_000 if len(times) >= 2 else 0.0
    tph = (len(fills) / span_h) if span_h > 0 else 0.0

    # --- midsize WR: last 7 days ---
    recent = [
        f
        for f in fills
        if isinstance(f, dict) and (now_ms - int(f.get("time") or 0)) < 7 * 86400 * 1000
    ]
    closed7 = [f for f in recent if abs(float(f.get("closedPnl") or 0)) > 1e-9]
    wins7 = [f for f in closed7 if float(f.get("closedPnl") or 0) > 0]
    wr7 = (len(wins7) / len(closed7)) if closed7 else None
    d1 = [
        f
        for f in fills
        if isinstance(f, dict) and (now_ms - int(f.get("time") or 0)) < 86400 * 1000
    ]
    fph24 = len(d1) / 24.0

    coins = Counter(str(f.get("coin")) for f in recent if isinstance(f, dict))
    if not coins:
        coins = Counter(str(f.get("coin")) for f in fills[:200] if isinstance(f, dict))

    live_av = float((state.get("marginSummary") or {}).get("accountValue") or 0)
    return {
        **c,
        "npos": len(pos),
        "coins_pos": pos[:8],
        "fills": len(fills),
        "closed_n": closed_n,
        "wr": None if wr_all is None else round(wr_all, 4),
        "tph": round(tph, 2),
        "fills7": len(recent),
        "closed7": len(closed7),
        "wr7": None if wr7 is None else round(wr7, 4),
        "fph24": round(fph24, 2),
        "top_coins": [{"coin": k, "n": n} for k, n in coins.most_common(5)],
        "live_av": round(live_av, 2),
        "hl_url": f"https://app.hyperliquid.xyz/explorer/address/{addr}",
    }


def _is_turnover_pick(r: dict[str, Any]) -> bool:
    wr = r.get("wr")
    if wr is None or wr < WR_MIN:
        return False
    if float(r.get("live_av") or 0) < LIVE_AV_MIN:
        return False
    if int(r.get("closed_n") or 0) < TO_CLOSED_MIN:
        return False
    return True


def _is_midsize_pick(r: dict[str, Any]) -> bool:
    wr = r.get("wr7")
    if wr is None or wr < WR_MIN:
        return False
    if int(r.get("fills7") or 0) < MS_FILLS7_MIN:
        return False
    if int(r.get("closed7") or 0) < MS_CLOSED7_MIN:
        return False
    if float(r.get("fph24") or 0) > MS_FPH24_MAX:
        return False
    if float(r.get("live_av") or 0) < LIVE_AV_MIN:
        return False
    return True


def _public_turnover(r: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "addr",
        "av",
        "live_av",
        "wr",
        "closed_n",
        "fills",
        "tph",
        "turn_w",
        "turn_m",
        "day_pnl",
        "week_pnl",
        "month_pnl",
        "week_vlm",
        "week_roi",
        "month_roi",
        "score",
        "npos",
        "coins_pos",
        "top_coins",
        "hl_url",
    )
    return {k: r.get(k) for k in keys}


def _public_midsize(r: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "addr",
        "av",
        "live_av",
        "wr",
        "fills7",
        "closed7",
        "fph24",
        "day_pnl",
        "week_pnl",
        "month_pnl",
        "week_vlm",
        "week_roi",
        "month_roi",
        "npos",
        "coins_pos",
        "top_coins",
        "hl_url",
    )
    out = {k: r.get(k) for k in keys}
    # expose 7D WR as primary `wr` for this lane
    out["wr"] = r.get("wr7")
    out["wr_all"] = r.get("wr")
    return out


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


def run_screen(*, sleep_sec: float = 0.4) -> dict[str, Any]:
    """Run both lanes; persist board JSON."""
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
        skip = _known_watch_addrs()
        lb = _http_get_json(LEADERBOARD_URL)
        raw_rows = lb.get("leaderboardRows") or []
        parsed: list[dict[str, Any]] = []
        for row in raw_rows:
            item = _parse_leaderboard_row(row)
            if item:
                parsed.append(item)

        to_cands = _turnover_candidates(parsed, skip)
        ms_cands = _midsize_candidates(parsed, skip)

        # Union deep-screen targets (dedupe, prefer richer cand fields)
        deep_map: dict[str, dict[str, Any]] = {}
        for c in to_cands[:TO_DEEP_TOP_N]:
            deep_map[c["addr"]] = dict(c)
        for c in ms_cands[:MS_DEEP_TOP_N]:
            if c["addr"] in deep_map:
                deep_map[c["addr"]].update(c)
            else:
                deep_map[c["addr"]] = dict(c)

        to_addrs = {c["addr"] for c in to_cands[:TO_DEEP_TOP_N]}
        ms_addrs = {c["addr"] for c in ms_cands[:MS_DEEP_TOP_N]}

        now_ms = int(time.time() * 1000)
        scanned: dict[str, dict[str, Any]] = {}
        errors: list[dict[str, str]] = []
        for addr, c in deep_map.items():
            try:
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                scanned[addr] = _deep_screen_one(c, now_ms)
            except Exception as exc:
                logger.warning("deep screen failed %s: %s", addr, exc)
                errors.append({"addr": addr, "error": str(exc)})

        to_scanned = [scanned[a] for a in to_addrs if a in scanned]
        ms_scanned = [scanned[a] for a in ms_addrs if a in scanned]

        to_picks = [_public_turnover(r) for r in to_scanned if _is_turnover_pick(r)]
        to_picks.sort(key=lambda x: (float(x.get("wr") or 0), float(x.get("turn_w") or 0)), reverse=True)
        to_picks = to_picks[:PICK_TOP_N]

        ms_picks = [_public_midsize(r) for r in ms_scanned if _is_midsize_pick(r)]
        ms_picks.sort(key=lambda x: (float(x.get("wr") or 0), float(x.get("week_pnl") or 0)), reverse=True)
        ms_picks = ms_picks[:PICK_TOP_N]

        lanes = {
            "turnover": _lane_payload(
                criteria=CRITERIA_TURNOVER,
                candidate_count=len(to_cands),
                scanned_count=len(to_scanned),
                picks=to_picks,
            ),
            "midsize": _lane_payload(
                criteria=CRITERIA_MIDSIZE,
                candidate_count=len(ms_cands),
                scanned_count=len(ms_scanned),
                picks=ms_picks,
            ),
        }

        board = {
            "ok": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "venue": "hyperliquid",
            "lanes": lanes,
            # backward-compatible flat fields → midsize
            "criteria": dict(CRITERIA_MIDSIZE),
            "candidate_count": len(ms_cands),
            "scanned_count": len(ms_scanned),
            "pick_count": len(ms_picks),
            "picks": ms_picks,
            "skipped_known": len(skip),
            "deep_unique": len(deep_map),
            "errors": errors[:20],
            "elapsed_sec": round(time.time() - started, 1),
        }
        _save_board(board)
        logger.info(
            "hl wr screen done: to_picks=%s ms_picks=%s deep=%s elapsed=%.1fs",
            len(to_picks),
            len(ms_picks),
            len(deep_map),
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
        "lanes": {
            "turnover": _lane_payload(
                criteria=CRITERIA_TURNOVER,
                candidate_count=0,
                scanned_count=0,
                picks=[],
            ),
            "midsize": _lane_payload(
                criteria=CRITERIA_MIDSIZE,
                candidate_count=0,
                scanned_count=0,
                picks=[],
            ),
        },
        "criteria": dict(CRITERIA_MIDSIZE),
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
        # older single-lane boards: synthesize lanes
        if "lanes" not in out:
            out["lanes"] = {
                "turnover": _lane_payload(
                    criteria=CRITERIA_TURNOVER,
                    candidate_count=0,
                    scanned_count=0,
                    picks=[],
                ),
                "midsize": _lane_payload(
                    criteria=out.get("criteria") or CRITERIA_MIDSIZE,
                    candidate_count=int(out.get("candidate_count") or 0),
                    scanned_count=int(out.get("scanned_count") or 0),
                    picks=list(out.get("picks") or []),
                ),
            }
        return out
    return empty_board(note="尚无日筛结果，等待每日任务或手动刷新")

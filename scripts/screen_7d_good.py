"""Canonical 7d-good screen recipe that produced the 15-hit board.

Gates (do not drift without user OK):
  trips ≥ 8, WR ≥ 55%, pair ≥ 50%, week-ROI pool ≥ 10%,
  trip PnL ≥ 2k, live ≥ 8k, Bitget share ≥ 50%,
  pace = 60s-merged legs/hour ≤ 8 (NOT raw fill fph),
  desk B–F force-included / not excluded.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.hl_bitget_symbol_map import bitget_contract_set, map_hl_coin_to_bitget  # noqa: E402
from utils.hl_wr_screen import (  # noqa: E402
    LEADERBOARD_URL,
    _fetch_fills_7d,
    _hl_info,
    _leaderboard_candidates,
    _merge_leg_bursts,
    _parse_leaderboard_row,
    _round_trips_60s,
    _wr_from_round_trips,
)

BJ = timezone(timedelta(hours=8))
OUT = ROOT / "hl_7d_good_screen.json"

DESK = {
    "0xa870c44f7e6e2e15d104185d3bbe5a54f9e2b52d": "bot_a",
    "0x9ffdf919da72213588f7517598394cc5535bce40": "bot_b",
    "0x93c3cff3b7a8492c581a609d5312920a2a60a0f4": "bot_c",
    "0xf42d1073367178b215f0ca6dfa04a5c889cfd112": "bot_d",
    "0xfbab890b61dc2e912718295dcdf45f499650ce9d": "bot_e",
    "0xa7405ff2687cb83b8a8a08eeaa4e4bc249344d23": "bot_f",
    "0xae1e846249f2ea246f36b53f326a2ce02aac17e4": "bot_g",
    "0x202c47f7c66e5d678db3e051deabc25478863022": "bot_h",
}

MIN_TRIPS = 8  # >7
MIN_WR = 0.55
MIN_WEEK_ROI = 0.10
MIN_PAIR = 0.50
MIN_PNL = 2_000.0
MIN_LIVE = 8_000.0
MAX_LPH24 = 8.0
BITGET_SHARE_MIN = 0.50
DEEP_N = 120


def fetch_fills_7d(addr: str, start_ms: int) -> list[dict]:
    return _fetch_fills_7d(addr, start_ms)


def score_addr(
    addr: str,
    c: dict,
    *,
    now: datetime,
    start_ms: int,
) -> dict | None:
    fills = fetch_fills_7d(addr, start_ms)
    if len(fills) < 12:
        return None
    bg = [f for f in fills if map_hl_coin_to_bitget(str(f.get("coin") or ""))]
    share = (len(bg) / len(fills)) if fills else 0.0
    if not bg or share < BITGET_SHARE_MIN:
        return None

    now_ms = now.timestamp() * 1000
    raw_24 = [f for f in fills if (now_ms - int(f.get("time") or 0)) < 86400_000]
    raw_fph24 = len(raw_24) / 24.0

    trips = _round_trips_60s(bg, gap_ms=60_000)
    wr, wins, losses, ncl = _wr_from_round_trips(trips)
    if ncl < MIN_TRIPS or wr is None or wr < MIN_WR:
        return None
    paired = sum(1 for t in trips if t.get("paired"))
    pr = paired / len(trips) if trips else 0.0
    if pr < MIN_PAIR:
        return None
    pnl = sum(float(t.get("pnl") or 0) for t in trips)
    if pnl < MIN_PNL:
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

    time.sleep(0.2)
    st = _hl_info({"type": "clearinghouseState", "user": addr})
    live = float((st.get("marginSummary") or {}).get("accountValue") or 0)
    if live < MIN_LIVE:
        return None

    coins = Counter(str(f.get("coin")) for f in bg).most_common(3)
    last = datetime.fromtimestamp(int(fills[-1]["time"]) / 1000, BJ)
    quiet_h = (now - last).total_seconds() / 3600.0
    week_roi = float(c.get("week_roi") or 0)
    week_pnl = float(c.get("week_pnl") or 0)
    return {
        "addr": addr,
        "desk": DESK.get(addr),
        "live_av": round(live, 1),
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
        "lph24": round(lph24, 2),
        "tph24": round(tph24, 2),
        "raw_fph24": round(raw_fph24, 2),
        "bitget_share": round(share, 3),
        "coins": ",".join(k for k, _ in coins),
        "last_bj": last.strftime("%m-%d %H:%M"),
        "quiet_h": round(quiet_h, 2),
        "hl_url": f"https://app.hyperliquid.xyz/explorer/address/{addr}",
    }


def main() -> None:
    bitget_contract_set(force=True)
    now = datetime.now(BJ)
    start_ms = int((now - timedelta(days=7)).timestamp() * 1000)
    print("window", datetime.fromtimestamp(start_ms / 1000, BJ), "->", now, flush=True)

    print("fetch leaderboard...", flush=True)
    lb = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(
                LEADERBOARD_URL, headers={"User-Agent": "7d-good"}
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                lb = json.loads(resp.read().decode())
            break
        except Exception as exc:
            print(f"leaderboard try {attempt+1}: {exc}", flush=True)
            time.sleep(2 + attempt * 2)
    if lb is None:
        cache = ROOT / "lb.json"
        if cache.is_file():
            print("using cached lb.json", flush=True)
            lb = json.loads(cache.read_text(encoding="utf-8"))
        else:
            raise RuntimeError("leaderboard fetch failed and no lb.json cache")
    parsed = []
    by_addr: dict[str, dict] = {}
    for row in lb.get("leaderboardRows") or []:
        item = _parse_leaderboard_row(row)
        if item:
            parsed.append(item)
            by_addr[item["addr"]] = item
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
    deep = pool[:DEEP_N]
    seen_addr = {c["addr"] for c in deep}
    for addr, label in DESK.items():
        if addr in seen_addr:
            continue
        base = by_addr.get(addr)
        if base:
            deep.append(base)
            print(f"force desk {label} into scan", flush=True)
        else:
            deep.append(
                {"addr": addr, "av": 0.0, "week_roi": 0.0, "week_pnl": 0.0}
            )
            print(f"force desk {label} (not on board row)", flush=True)
    print(f"pool={len(pool)} deep={len(deep)} (A–H included)", flush=True)

    hits: list[dict] = []
    desk_rows: list[dict] = []
    for i, c in enumerate(deep):
        addr = c["addr"]
        try:
            row = score_addr(addr, c, now=now, start_ms=start_ms)
            if not row:
                if addr in DESK:
                    print(f"desk miss {DESK[addr]} {addr[:14]}", flush=True)
                continue
            hits.append(row)
            if row.get("desk"):
                desk_rows.append(row)
            tag = f" [{row['desk']}]" if row.get("desk") else ""
            print(
                f"HIT{tag} {addr[:14]} weekROI={row['week_roi']:.0%} "
                f"wr={row['wr7']:.0%} n={row['trips7']} pair={row['pair_rate']:.0%} "
                f"pnl={row['pnl7_closed']:.0f} lph={row['lph24']} "
                f"raw_fph={row['raw_fph24']} {row['coins']}",
                flush=True,
            )
        except Exception as exc:
            print(f"fail {addr[:12]} {exc}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"progress {i+1}/{len(deep)} hits={len(hits)}", flush=True)

    hits.sort(
        key=lambda x: (
            float(x["week_roi"]),
            float(x["wr7"]),
            float(x["trips7"]),
            float(x["pnl7_closed"]),
        ),
        reverse=True,
    )
    out = {
        "ok": True,
        "generated_at": now.isoformat(),
        "window": "7d",
        "gates": {
            "min_trips": MIN_TRIPS,
            "min_wr": MIN_WR,
            "min_week_roi_pool": MIN_WEEK_ROI,
            "min_pair_rate": MIN_PAIR,
            "min_pnl": MIN_PNL,
            "min_live": MIN_LIVE,
            "max_lph24_merged_legs": MAX_LPH24,
            "pace_note": "lph24 = 60s-merged open/close bursts / hour (not raw fills)",
            "bitget_share_min": BITGET_SHARE_MIN,
            "exclude_desk_ah": False,
        },
        "scanned": len(deep),
        "count": len(hits),
        "desk_hits": desk_rows,
        "picks": hits[:40],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== TOP 7d good (trips>7, B–F included) ===", flush=True)
    for i, h in enumerate(hits[:20], 1):
        tag = f" [{h['desk']}]" if h.get("desk") else ""
        print(
            f"{i:02d}{tag} weekROI={h['week_roi']:.0%} wr={h['wr7']:.0%} "
            f"n={h['trips7']} pair={h['pair_rate']:.0%} "
            f"pnl={h['pnl7_closed']:.0f} lph={h['lph24']} "
            f"live={h['live_av']:.0f} {h['coins']} {h['addr']}",
            flush=True,
        )
    print("wrote", OUT, "count", len(hits), flush=True)


if __name__ == "__main__":
    main()

"""Rank HL wallets: 60s-merged WR + 7D ROI, Bitget-listed assets only → top 100."""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.hl_bitget_symbol_map import bitget_contract_set, map_hl_coin_to_bitget  # noqa: E402
from utils.hl_wr_screen import (  # noqa: E402
    LEADERBOARD_URL,
    _hl_info,
    _leaderboard_candidates,
    _parse_leaderboard_row,
    _round_trips_60s,
    _wr_from_round_trips,
)

OUT_NAME = "hl_bitget_roi_wr_top100.json"
DEEP_N = 450
SLEEP = 0.55
MIN_CLOSED = 4
MIN_LIVE_AV = 3_000.0
MIN_BITGET_SHARE = 0.50  # majority of 7d fills on Bitget-mappable coins


def _is_bitget(coin: str) -> bool:
    return map_hl_coin_to_bitget(coin) is not None


def _rank_one(c: dict, now_ms: int) -> dict | None:
    addr = c["addr"]
    state = _hl_info({"type": "clearinghouseState", "user": addr})
    fills = _hl_info({"type": "userFills", "user": addr})
    if not isinstance(fills, list):
        fills = []

    recent = [
        f
        for f in fills
        if isinstance(f, dict) and (now_ms - int(f.get("time") or 0)) < 7 * 86400 * 1000
    ]
    if not recent:
        return None

    n_all = len(recent)
    bg_fills = [f for f in recent if _is_bitget(str(f.get("coin") or ""))]
    bitget_share = len(bg_fills) / n_all if n_all else 0.0
    if bitget_share < MIN_BITGET_SHARE:
        return None

    # PnL / WR only on Bitget-listed coins; WR = 60s-merged open↔close round-trips.
    pnl7 = 0.0
    for f in bg_fills:
        if f.get("closedPnl") in (None, ""):
            continue
        try:
            pnl7 += float(f.get("closedPnl") or 0)
        except (TypeError, ValueError):
            continue

    trips = _round_trips_60s(bg_fills, gap_ms=60_000)
    wr7, wins, losses, ncl = _wr_from_round_trips(trips)
    if ncl < MIN_CLOSED:
        return None
    if wr7 is None:
        return None

    live_av = float((state.get("marginSummary") or {}).get("accountValue") or 0)
    if live_av < MIN_LIVE_AV:
        return None
    roi7 = pnl7 / live_av if live_av > 0 else 0.0

    from collections import Counter

    coins = Counter(str(f.get("coin")) for f in bg_fills)
    top = coins.most_common(5)
    follow = [k for k, _ in top if _is_bitget(k)][:4]

    pos = []
    for item in state.get("assetPositions") or []:
        p = item.get("position") or {}
        if abs(float(p.get("szi") or 0)) > 1e-12:
            coin = str(p.get("coin"))
            if _is_bitget(coin):
                pos.append(coin)

    score = (roi7 * 100.0) + (wr7 * 20.0)
    paired = sum(1 for t in trips if t.get("paired"))

    return {
        "addr": addr,
        "live_av": round(live_av, 2),
        "lb_av": round(float(c.get("av") or 0), 2),
        "week_pnl_lb": round(float(c.get("week_pnl") or 0), 2),
        "pnl7_bitget": round(pnl7, 2),
        "roi7": round(roi7, 4),
        "wr7": round(wr7, 4),
        "closed7_merged": ncl,
        "round_trips": len(trips),
        "paired_trips": paired,
        "wins": wins,
        "losses": losses,
        "fills7_bitget": len(bg_fills),
        "bitget_share": round(bitget_share, 3),
        "follow_coins": follow,
        "top_coins": [{"coin": k, "n": n} for k, n in top],
        "npos_bitget": len(pos),
        "coins_pos": pos[:8],
        "score": round(score, 3),
        "hl_url": f"https://app.hyperliquid.xyz/explorer/address/{addr}",
        "note": "wr=60s open↔close round-trips on Bitget-mappable; roi=pnl7/live_av",
    }


def _select_deep(cands: list[dict], n: int) -> list[dict]:
    """Mix week-PnL whales + mid-AV week-ROI so Bitget traders aren't crowded out."""
    seen: set[str] = set()
    out: list[dict] = []
    whales = sorted(cands, key=lambda x: float(x.get("week_pnl") or 0), reverse=True)
    mid = [
        r
        for r in cands
        if 8_000 <= float(r.get("av") or 0) <= 2_000_000
    ]
    mid.sort(
        key=lambda x: (float(x.get("week_roi") or 0), float(x.get("week_pnl") or 0)),
        reverse=True,
    )
    for src, take in ((whales, n // 3), (mid, n - n // 3)):
        for r in src:
            if len(out) >= n:
                break
            a = str(r.get("addr") or "")
            if not a or a in seen:
                continue
            seen.add(a)
            out.append(r)
    return out[:n]


def main() -> None:
    print("warm bitget contracts...", flush=True)
    known = bitget_contract_set(force=True)
    print("bitget contracts", len(known or []), flush=True)

    print("fetch leaderboard...", flush=True)
    t0 = time.time()
    req = urllib.request.Request(LEADERBOARD_URL, headers={"User-Agent": "next-k-bitget-rank/1"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        lb = json.loads(resp.read().decode())
    print("lb", len(lb.get("leaderboardRows") or []), "sec", round(time.time() - t0, 1), flush=True)

    parsed = []
    for row in lb.get("leaderboardRows") or []:
        item = _parse_leaderboard_row(row)
        if item:
            parsed.append(item)
    cands = _leaderboard_candidates(parsed, set())
    deep = _select_deep(cands, DEEP_N)
    print("cands", len(cands), "deep", len(deep), flush=True)

    now_ms = int(time.time() * 1000)
    ranked: list[dict] = []
    errors = 0
    for i, c in enumerate(deep):
        time.sleep(SLEEP)
        try:
            row = _rank_one(c, now_ms)
            if row:
                ranked.append(row)
        except Exception as exc:
            errors += 1
            print(f"fail {c.get('addr')}: {exc}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"scanned {i+1}/{len(deep)} kept {len(ranked)} err {errors}", flush=True)

    ranked.sort(key=lambda x: (float(x["roi7"]), float(x["wr7"]), float(x["score"])), reverse=True)
    top = ranked[:100]

    out = {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "merge_gap_ms": 60_000,
            "window": "7d",
            "assets": "bitget_mappable_only",
            "bitget_share_min": MIN_BITGET_SHARE,
            "closed_merged_min": MIN_CLOSED,
            "live_av_min": MIN_LIVE_AV,
            "rank": "roi7 desc, wr7 desc",
            "scanned": len(deep),
            "qualified": len(ranked),
            "errors": errors,
        },
        "count": len(top),
        "picks": top,
    }
    path = Path(OUT_NAME)
    # also under data dir if available
    try:
        from utils.hl_short_term import resolve_data_dir

        dpath = resolve_data_dir() / OUT_NAME
        dpath.parent.mkdir(parents=True, exist_ok=True)
        dpath.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", dpath, flush=True)
    except Exception:
        pass
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", path.resolve(), flush=True)
    print(f"TOP {len(top)} / qualified {len(ranked)} / scanned {len(deep)}", flush=True)
    for i, p in enumerate(top[:30], 1):
        print(
            f"{i:02d} roi={p['roi7']:.2%} wr={p['wr7']:.1%} "
            f"closed={p['closed7_merged']} pnl={p['pnl7_bitget']:.0f} "
            f"av={p['live_av']:.0f} bg={p['bitget_share']:.0%} "
            f"{','.join(p['follow_coins'][:3])} {p['addr'][:14]}…",
            flush=True,
        )
    if len(top) < 100:
        print(f"NOTE: only {len(top)} qualified under Bitget≥80% rules (want 100).", flush=True)


if __name__ == "__main__":
    main()

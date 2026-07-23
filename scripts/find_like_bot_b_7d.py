"""Find wallets like desk B using last-7d data only (Bitget assets)."""
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
    _hl_info,
    _leaderboard_candidates,
    _merge_leg_bursts,
    _parse_leaderboard_row,
    _round_trips_60s,
    _wr_from_round_trips,
)

BJ = timezone(timedelta(hours=8))
OUT = ROOT / "hl_like_bot_b_7d.json"
SKIP = {
    "0xa870c44f7e6e2e15d104185d3bbe5a54f9e2b52d",  # desk B
    "0x9ffdf919da72213588f7517598394cc5535bce40",  # desk C
}

# B-like gates (7d)
MIN_TRIPS = 8
MIN_PAIR = 0.55
MIN_WR = 0.60
MIN_PNL = 5_000.0
MIN_LIVE = 15_000.0
MIN_WEEK_ROI = 0.15  # HL board week ROI
MAX_LPH = 8.0  # 60s-merged open/close legs per hour (not raw fills)
DEEP_N = 100


def fetch_fills_7d(addr: str, start_ms: int) -> list[dict]:
    fills: list[dict] = []
    cursor = start_ms
    seen: set = set()
    for _ in range(4):
        time.sleep(0.4)
        batch = _hl_info(
            {"type": "userFillsByTime", "user": addr, "startTime": cursor}
        )
        if not isinstance(batch, list) or not batch:
            break
        max_t = cursor
        for f in batch:
            if not isinstance(f, dict):
                continue
            ts = int(f.get("time") or 0)
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


def main() -> None:
    bitget_contract_set(force=True)
    now = datetime.now(BJ)
    start_ms = int((now - timedelta(days=7)).timestamp() * 1000)
    print("window", datetime.fromtimestamp(start_ms / 1000, BJ), "->", now, flush=True)

    print("fetch leaderboard...", flush=True)
    req = urllib.request.Request(LEADERBOARD_URL, headers={"User-Agent": "like-b-7d"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        lb = json.loads(resp.read().decode())
    parsed = []
    for row in lb.get("leaderboardRows") or []:
        item = _parse_leaderboard_row(row)
        if item:
            parsed.append(item)
    cands = _leaderboard_candidates(parsed, set())

    # Prefer mid AV + strong week ROI / week PnL (B-like band)
    pool = [
        c
        for c in cands
        if c["addr"] not in SKIP
        and 15_000 <= float(c.get("av") or 0) <= 2_000_000
        and float(c.get("week_roi") or 0) >= MIN_WEEK_ROI
        and float(c.get("week_pnl") or 0) >= 8_000
    ]
    pool.sort(
        key=lambda x: (float(x.get("week_roi") or 0), float(x.get("week_pnl") or 0)),
        reverse=True,
    )
    deep = pool[:DEEP_N]
    print(f"pool={len(pool)} deep={len(deep)}", flush=True)

    hits: list[dict] = []
    for i, c in enumerate(deep):
        addr = c["addr"]
        try:
            fills = fetch_fills_7d(addr, start_ms)
            if len(fills) < 20:
                continue
            bg = [f for f in fills if map_hl_coin_to_bitget(str(f.get("coin") or ""))]
            if not bg or len(bg) / len(fills) < 0.6:
                continue
            now_ms = now.timestamp() * 1000
            bg_24 = [
                f
                for f in bg
                if (now_ms - int(f.get("time") or 0)) < 86400_000
            ]
            lph = len(_merge_leg_bursts(bg_24, gap_ms=60_000)) / 24.0
            if lph > MAX_LPH:
                continue

            trips = _round_trips_60s(bg, gap_ms=60_000)
            wr, wins, losses, ncl = _wr_from_round_trips(trips)
            if ncl < MIN_TRIPS or wr is None or wr < MIN_WR:
                continue
            paired = sum(1 for t in trips if t.get("paired"))
            pr = paired / len(trips) if trips else 0.0
            if pr < MIN_PAIR:
                continue
            pnl = sum(float(t.get("pnl") or 0) for t in trips)
            if pnl < MIN_PNL:
                continue

            time.sleep(0.25)
            st = _hl_info({"type": "clearinghouseState", "user": addr})
            live = float((st.get("marginSummary") or {}).get("accountValue") or 0)
            if live < MIN_LIVE:
                continue

            coins = Counter(str(f.get("coin")) for f in bg).most_common(3)
            last = datetime.fromtimestamp(int(fills[-1]["time"]) / 1000, BJ)
            quiet_h = (now - last).total_seconds() / 3600.0
            row = {
                "addr": addr,
                "live_av": round(live, 1),
                "lb_av": round(float(c.get("av") or 0), 1),
                "week_roi": round(float(c.get("week_roi") or 0), 4),
                "week_pnl": round(float(c.get("week_pnl") or 0), 1),
                "trips7": ncl,
                "paired": paired,
                "pair_rate": round(pr, 3),
                "wr7": round(wr, 4),
                "pnl7_closed": round(pnl, 1),
                "pnl7_roi_live": round(pnl / live, 4) if live else 0,
                "lph24": round(lph, 2),
                "fph24": round(lph, 2),
                "bitget_share": round(len(bg) / len(fills), 3),
                "coins": ",".join(k for k, _ in coins),
                "last_bj": last.strftime("%m-%d %H:%M"),
                "quiet_h": round(quiet_h, 2),
                "hl_url": f"https://app.hyperliquid.xyz/explorer/address/{addr}",
            }
            hits.append(row)
            print(
                f"HIT {addr[:14]} weekROI={row['week_roi']:.0%} "
                f"wr={row['wr7']:.0%} n={ncl} pair={pr:.0%} "
                f"pnl={pnl:.0f} {row['coins']}",
                flush=True,
            )
        except Exception as exc:
            print(f"fail {addr[:12]} {exc}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"progress {i+1}/{len(deep)} hits={len(hits)}", flush=True)

    hits.sort(
        key=lambda x: (
            float(x["pair_rate"]),
            float(x["week_roi"]),
            float(x["wr7"]),
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
            "min_pair_rate": MIN_PAIR,
            "min_wr": MIN_WR,
            "min_pnl": MIN_PNL,
            "min_live": MIN_LIVE,
            "min_week_roi": MIN_WEEK_ROI,
            "max_lph24_merged_legs": MAX_LPH,
            "bitget_share_min": 0.6,
        },
        "scanned": len(deep),
        "count": len(hits),
        "picks": hits[:30],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== TOP like B (7d) ===", flush=True)
    for i, h in enumerate(hits[:15], 1):
        print(
            f"{i:02d} weekROI={h['week_roi']:.0%} wr={h['wr7']:.0%} "
            f"n={h['trips7']} pair={h['pair_rate']:.0%} "
            f"pnl={h['pnl7_closed']:.0f} live={h['live_av']:.0f} "
            f"{h['coins']} {h['addr']}",
            flush=True,
        )
    print("wrote", OUT, "count", len(hits), flush=True)


if __name__ == "__main__":
    main()

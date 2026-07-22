"""Full-history trade analysis for desk B target address (since first fill)."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.hl_wr_screen import (  # noqa: E402
    _fill_leg,
    _hl_info,
    _merge_leg_bursts,
    _round_trips_60s,
    _wr_from_round_trips,
)

BJ = timezone(timedelta(hours=8))
ADDR = "0xa870c44f7e6e2e15d104185d3bbe5a54f9e2b52d"
OUT = ROOT / "bot_b_addr_lifetime_analysis.json"
# start well before known first activity (~2026-07-14)
START_MS = int(datetime(2026, 7, 1, tzinfo=BJ).timestamp() * 1000)


def fetch_all_fills(addr: str, start_ms: int) -> list[dict]:
    """Paginate userFillsByTime (max 2000 / response)."""
    all_fills: list[dict] = []
    cursor = start_ms
    seen: set[tuple] = set()
    while True:
        time.sleep(0.35)
        batch = _hl_info(
            {
                "type": "userFillsByTime",
                "user": addr,
                "startTime": cursor,
                "aggregateByTime": False,
            }
        )
        if not isinstance(batch, list) or not batch:
            break
        new = 0
        max_t = cursor
        for f in batch:
            if not isinstance(f, dict):
                continue
            tid = f.get("tid")
            key = (tid, f.get("time"), f.get("coin"), f.get("sz"), f.get("px"))
            if key in seen:
                continue
            seen.add(key)
            all_fills.append(f)
            new += 1
            max_t = max(max_t, int(f.get("time") or 0))
        print(f"batch +{new} total={len(all_fills)} cursor→{datetime.fromtimestamp(max_t/1000, BJ)}", flush=True)
        if len(batch) < 2000:
            break
        # advance past last timestamp; if stuck, bump 1ms
        nxt = max_t + 1
        if nxt <= cursor:
            break
        cursor = nxt
    all_fills.sort(key=lambda x: int(x.get("time") or 0))
    return all_fills


def main() -> None:
    print("fetch lifetime fills", ADDR, flush=True)
    fills = fetch_all_fills(ADDR, START_MS)
    if not fills:
        print("no fills", flush=True)
        return

    first = datetime.fromtimestamp(int(fills[0]["time"]) / 1000, BJ)
    last = datetime.fromtimestamp(int(fills[-1]["time"]) / 1000, BJ)
    print(f"fills={len(fills)} first={first} last={last}", flush=True)

    legs: Counter[str] = Counter()
    for f in fills:
        leg = _fill_leg(f)
        if leg:
            legs[f"{leg[0]} {leg[1]}"] += 1
        else:
            legs[f"other:{str(f.get('dir') or '')[:28]}"] += 1
    coins = Counter(str(f.get("coin")) for f in fills)
    print("legs", dict(legs), flush=True)
    print("coins", coins.most_common(10), flush=True)

    bursts = _merge_leg_bursts(fills, gap_ms=60_000)
    trips = _round_trips_60s(fills, gap_ms=60_000)
    wr, wins, losses, ncl = _wr_from_round_trips(trips)
    paired = sum(1 for t in trips if t.get("paired"))
    pnl = sum(float(t.get("pnl") or 0) for t in trips)
    print(
        f"bursts={len(bursts)} trips={len(trips)} paired={paired}/{len(trips)} "
        f"wr={None if wr is None else round(wr,4)} W/L={wins}/{losses} pnl={pnl:.2f}",
        flush=True,
    )

    # by day BJ
    by_day: Counter[str] = Counter()
    pnl_day: dict[str, float] = {}
    for t in trips:
        day = datetime.fromtimestamp(int(t["close_time"]) / 1000, BJ).strftime("%Y-%m-%d")
        by_day[day] += 1
        pnl_day[day] = pnl_day.get(day, 0.0) + float(t.get("pnl") or 0)

    print("--- round trips ---", flush=True)
    trip_rows = []
    for i, t in enumerate(trips, 1):
        ot = (
            datetime.fromtimestamp(t["open_time"] / 1000, BJ).strftime("%m-%d %H:%M:%S")
            if t.get("open_time")
            else "-"
        )
        ct = datetime.fromtimestamp(t["close_time"] / 1000, BJ).strftime("%m-%d %H:%M:%S")
        row = {
            "i": i,
            "coin": t["coin"],
            "side": t["side"],
            "pnl": round(float(t["pnl"]), 2),
            "paired": bool(t.get("paired")),
            "open_bj": ot,
            "close_bj": ct,
            "n_fills": t.get("n_fills"),
            "notional": round(float(t.get("notional") or 0), 2),
        }
        trip_rows.append(row)
        print(
            f"{i:03d} {t['coin']:8s} {t['side']:5s} pnl={float(t['pnl']):10.2f} "
            f"paired={str(t['paired']):5s} open={ot} close={ct}",
            flush=True,
        )

    print("--- bursts (merged 60s legs) ---", flush=True)
    burst_rows = []
    for i, b in enumerate(bursts, 1):
        t0 = datetime.fromtimestamp(b["time"] / 1000, BJ).strftime("%m-%d %H:%M:%S")
        burst_rows.append(
            {
                "i": i,
                "kind": b["kind"],
                "side": b["side"],
                "coin": b["coin"],
                "n_fills": b["n_fills"],
                "pnl": round(float(b["pnl"]), 2),
                "notional": round(float(b["notional"]), 2),
                "time_bj": t0,
            }
        )
        print(
            f"{i:03d} {b['kind']:5s} {b['side']:5s} {b['coin']:8s} "
            f"n={b['n_fills']:4d} pnl={float(b['pnl']):10.2f}  {t0}",
            flush=True,
        )

    print("--- by day (round-trip close day) ---", flush=True)
    for day in sorted(by_day):
        print(f"{day} trips={by_day[day]} pnl={pnl_day[day]:.2f}", flush=True)

    state = _hl_info({"type": "clearinghouseState", "user": ADDR})
    live_av = float((state.get("marginSummary") or {}).get("accountValue") or 0)
    pos = []
    for item in state.get("assetPositions") or []:
        p = item.get("position") or {}
        if abs(float(p.get("szi") or 0)) > 1e-12:
            pos.append(
                {
                    "coin": p.get("coin"),
                    "szi": p.get("szi"),
                    "entryPx": p.get("entryPx"),
                    "unrealizedPnl": p.get("unrealizedPnl"),
                    "positionValue": p.get("positionValue"),
                }
            )

    out = {
        "addr": ADDR,
        "scope": "address_lifetime_since_first_fill",
        "first_fill_bj": first.isoformat(),
        "last_fill_bj": last.isoformat(),
        "raw_fills": len(fills),
        "legs": dict(legs),
        "coins": [{"coin": k, "n": n} for k, n in coins.most_common()],
        "bursts": len(bursts),
        "round_trips": len(trips),
        "paired": paired,
        "wr": wr,
        "wins": wins,
        "losses": losses,
        "closed_pnl_sum": round(pnl, 2),
        "live_av": live_av,
        "positions": pos,
        "by_day": {d: {"trips": by_day[d], "pnl": round(pnl_day[d], 2)} for d in sorted(by_day)},
        "trip_rows": trip_rows,
        "burst_rows": burst_rows,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

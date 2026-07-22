"""Analyze desk B (bot_b) trades since paper bind — 60s merge + round-trips."""
from __future__ import annotations

import json
import sys
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
PAPER_PATH = ROOT / "bot_b_paper_fills.json"
OUT_PATH = ROOT / "bot_b_trade_analysis.json"


def parse_ts(f: dict) -> datetime | None:
    t = f.get("ts")
    if not t:
        return None
    return datetime.fromisoformat(str(t).replace("Z", "+00:00"))


def main() -> None:
    paper = json.loads(PAPER_PATH.read_text(encoding="utf-8"))
    fills = paper.get("fills") or []
    times = [parse_ts(f) for f in fills]
    times = [t for t in times if t]
    bind = min(times) if times else None
    last = max(times) if times else None
    bind_ms = int(bind.timestamp() * 1000) if bind else 0

    print("=== PAPER DESK B ===", flush=True)
    bot = paper.get("bot") or {}
    print(
        f"addr={bot.get('address')} equity={bot.get('equity')} "
        f"realized={bot.get('realized_pnl')} init={bot.get('paper_balance')}",
        flush=True,
    )
    print(
        f"bind={bind.astimezone(BJ) if bind else None} "
        f"last_paper={last.astimezone(BJ) if last else None}",
        flush=True,
    )
    print("paper rows", len(fills), "actions", dict(Counter(f.get("action") for f in fills)), flush=True)
    print(
        "skipped",
        sum(1 for f in fills if f.get("skipped")),
        dict(Counter(f.get("reason") for f in fills if f.get("skipped"))),
        flush=True,
    )
    print(
        "coins(non-skip)",
        Counter(f.get("coin") for f in fills if not f.get("skipped")).most_common(),
        flush=True,
    )
    pos = bot.get("positions") or {}
    if isinstance(pos, dict):
        for k, v in pos.items():
            if not isinstance(v, dict):
                continue
            print(
                "pos",
                k,
                {kk: v.get(kk) for kk in ("coin", "sz", "entry", "u_pnl", "notional", "side", "leverage")},
                flush=True,
            )

    print("=== TARGET HL since bind (60s merge) ===", flush=True)
    raw = _hl_info({"type": "userFills", "user": ADDR})
    if not isinstance(raw, list):
        raw = []
    since = [f for f in raw if isinstance(f, dict) and int(f.get("time") or 0) >= bind_ms]
    print(f"target fills since bind: {len(since)} / api_window {len(raw)}", flush=True)
    if raw:
        oldest = min(int(f.get("time") or 0) for f in raw)
        print("api oldest fill", datetime.fromtimestamp(oldest / 1000, BJ), flush=True)

    legs: Counter[str] = Counter()
    for f in since:
        leg = _fill_leg(f)
        if leg:
            legs[f"{leg[0]} {leg[1]}"] += 1
        else:
            legs[f"other:{str(f.get('dir') or '')[:24]}"] += 1
    print("legs", dict(legs), flush=True)
    print("coins", Counter(str(f.get("coin")) for f in since).most_common(10), flush=True)

    bursts = _merge_leg_bursts(since, gap_ms=60_000)
    trips = _round_trips_60s(since, gap_ms=60_000)
    wr, wins, losses, ncl = _wr_from_round_trips(trips)
    paired = sum(1 for t in trips if t.get("paired"))
    pnl = sum(float(t.get("pnl") or 0) for t in trips)
    print(
        f"bursts={len(bursts)} trips={len(trips)} paired={paired}/{len(trips)} "
        f"wr={wr} wins={wins} losses={losses} pnl={pnl:.2f}",
        flush=True,
    )

    print("--- merged leg bursts ---", flush=True)
    for i, b in enumerate(bursts, 1):
        t0 = datetime.fromtimestamp(b["time"] / 1000, BJ).strftime("%m-%d %H:%M:%S")
        print(
            f"{i:02d} {b['kind']:5s} {b['side']:5s} {b['coin']:8s} "
            f"n={b['n_fills']:3d} pnl={float(b['pnl']):10.2f} "
            f"notional={float(b['notional']):10.0f}  {t0}",
            flush=True,
        )

    print("--- round trips (open↔close) ---", flush=True)
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
        }
        trip_rows.append(row)
        print(
            f"{i:02d} {t['coin']} {t['side']} pnl={t['pnl']:.2f} "
            f"paired={t['paired']} open={ot} close={ct}",
            flush=True,
        )

    # paper executed notional / skipped
    exec_n = sum(float(f.get("notional") or 0) for f in fills if not f.get("skipped") and f.get("action") != "signal")
    skip_n = sum(1 for f in fills if f.get("skipped"))
    out = {
        "bot": "bot_b",
        "address": ADDR,
        "bind_at_bj": bind.astimezone(BJ).isoformat() if bind else None,
        "paper": {
            "equity": bot.get("equity"),
            "realized_pnl": bot.get("realized_pnl"),
            "paper_balance": bot.get("paper_balance"),
            "rows": len(fills),
            "skipped": skip_n,
            "exec_notional_sum": round(exec_n, 2),
            "actions": dict(Counter(f.get("action") for f in fills)),
            "skip_reasons": dict(Counter(f.get("reason") for f in fills if f.get("skipped"))),
        },
        "target_since_bind": {
            "raw_fills": len(since),
            "legs": dict(legs),
            "bursts": len(bursts),
            "round_trips": len(trips),
            "paired": paired,
            "wr": wr,
            "wins": wins,
            "losses": losses,
            "pnl": round(pnl, 2),
            "trips": trip_rows,
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()

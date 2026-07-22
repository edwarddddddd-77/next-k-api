"""Lifetime-scan a few B-like candidates (Bitget, high pair rate)."""
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

from utils.hl_bitget_symbol_map import bitget_contract_set, map_hl_coin_to_bitget  # noqa: E402
from utils.hl_wr_screen import (  # noqa: E402
    _hl_info,
    _round_trips_60s,
    _wr_from_round_trips,
)

BJ = timezone(timedelta(hours=8))
START = int(datetime(2026, 7, 1, tzinfo=BJ).timestamp() * 1000)

CANDS = [
    ("0x9ffdf919da72213588f7517598394cc5535bce40", "desk_C_HYPE"),
    ("0x34d7b1c11bebf407af7f67f08249a174d1316a6a", "LIT"),
    ("0x4f7634c03ec4e87e14725c84913ade523c6fad5a", "LIT_big"),
    ("0x9751cec58be43e0e27d57ba49ac3d68ec269a97e", "stocks_SKHY"),
    ("0x21316c7fa26028c7da41e03eed0ae9d7c2b7f727", "ZEC"),
    ("0x25833c0e1684182ad2370864c3202fa9a16db87a", "SOL_ETH"),
    ("0x6807f127eaf85e3d0b9dbc7971bbed2afd041f8d", "PUMP_BTC"),
    ("0x07fd993f0fa3a185f7207adccd29f7a87404689d", "AERO_alts"),
    ("0xd45ddb475b20ec2138ee0b316ef2896851390422", "BTC_SOL_HYPE"),
    ("0x212abcf1a07986aedb244fc4c6f5fe037913a8c6", "ETH_BTC_SOL"),
]


def fetch_fills(addr: str) -> list[dict]:
    fills: list[dict] = []
    cursor = START
    seen: set = set()
    for _ in range(8):
        time.sleep(0.35)
        batch = _hl_info({"type": "userFillsByTime", "user": addr, "startTime": cursor})
        if not isinstance(batch, list) or not batch:
            break
        max_t = cursor
        for f in batch:
            if not isinstance(f, dict):
                continue
            key = (f.get("tid"), f.get("time"), f.get("coin"), f.get("sz"))
            if key in seen:
                continue
            seen.add(key)
            fills.append(f)
            max_t = max(max_t, int(f.get("time") or 0))
        if len(batch) < 2000:
            break
        if max_t + 1 <= cursor:
            break
        cursor = max_t + 1
    fills.sort(key=lambda x: int(x.get("time") or 0))
    return fills


def ledger_net(addr: str) -> dict:
    ledger = _hl_info({"type": "userNonFundingLedgerUpdates", "user": addr})
    if not isinstance(ledger, list):
        return {}
    inn = out = 0.0
    for e in ledger:
        d = e.get("delta") or {}
        typ = d.get("type")
        if typ == "deposit":
            inn += float(d.get("usdc") or 0)
        elif typ == "withdraw":
            out += float(d.get("usdc") or 0)
        elif typ == "send":
            usdc = float(d.get("usdcValue") or d.get("amount") or 0)
            dest = (d.get("destination") or "").lower()
            src = (d.get("user") or "").lower()
            if dest == addr.lower():
                inn += usdc
            elif src == addr.lower():
                out += usdc
    live = float(
        ((_hl_info({"type": "clearinghouseState", "user": addr}).get("marginSummary") or {}).get("accountValue") or 0)
    )
    return {
        "in": round(inn, 1),
        "out": round(out, 1),
        "net_in": round(inn - out, 1),
        "live_av": round(live, 1),
        "approx_pnl": round(live + out - inn, 1),
    }


def main() -> None:
    bitget_contract_set(force=True)
    rows = []
    for addr, tag in CANDS:
        print("scan", tag, addr[:12], flush=True)
        fills = fetch_fills(addr)
        if not fills:
            print("  no fills", flush=True)
            continue
        bg = [f for f in fills if map_hl_coin_to_bitget(str(f.get("coin") or ""))]
        trips = _round_trips_60s(bg, gap_ms=60_000)
        wr, wins, losses, ncl = _wr_from_round_trips(trips)
        paired = sum(1 for t in trips if t.get("paired"))
        pr = paired / len(trips) if trips else 0
        pnl = sum(float(t.get("pnl") or 0) for t in trips)
        coins = Counter(str(f.get("coin")) for f in bg).most_common(3)
        first = datetime.fromtimestamp(int(fills[0]["time"]) / 1000, BJ)
        last = datetime.fromtimestamp(int(fills[-1]["time"]) / 1000, BJ)
        time.sleep(0.3)
        money = ledger_net(addr)
        row = {
            "tag": tag,
            "addr": addr,
            "fills": len(fills),
            "trips": ncl,
            "paired": paired,
            "pair_rate": round(pr, 3),
            "wr": None if wr is None else round(wr, 3),
            "wins": wins,
            "losses": losses,
            "closed_pnl": round(pnl, 1),
            "coins": ",".join(k for k, _ in coins),
            "first": first.strftime("%m-%d %H:%M"),
            "last": last.strftime("%m-%d %H:%M"),
            **money,
        }
        rows.append(row)
        print(
            f"  trips={ncl} pair={pr:.0%} wr={wr} pnl={pnl:.0f} "
            f"in={money.get('in')} out={money.get('out')} approxPnL={money.get('approx_pnl')} "
            f"{row['coins']}",
            flush=True,
        )

    # filter B-like
    good = [
        r
        for r in rows
        if (r.get("trips") or 0) >= 15
        and (r.get("pair_rate") or 0) >= 0.5
        and (r.get("wr") or 0) >= 0.55
        and (r.get("closed_pnl") or 0) > 0
        and (r.get("live_av") or 0) >= 8000
    ]
    good.sort(key=lambda x: (x["pair_rate"], x.get("approx_pnl") or 0, x["closed_pnl"]), reverse=True)
    out = {"ok": True, "compared_to": "bot_b", "all": rows, "similar": good}
    path = ROOT / "hl_similar_to_bot_b.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== SIMILAR ===", flush=True)
    for i, r in enumerate(good, 1):
        print(
            f"{i} {r['tag']} wr={r['wr']:.0%} n={r['trips']} pair={r['pair_rate']:.0%} "
            f"closed={r['closed_pnl']:.0f} approxPnL={r.get('approx_pnl')} "
            f"av={r.get('live_av')} {r['coins']} {r['addr']}",
            flush=True,
        )
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()

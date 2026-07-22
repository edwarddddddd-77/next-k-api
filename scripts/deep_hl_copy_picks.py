"""Deep mine HL leaderboard for copyable wallets beyond current watch/screen."""
from __future__ import annotations

import json
import time
import urllib.request
from collections import Counter

LEADER = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
INFO = "https://api.hyperliquid.xyz/info"

KNOWN = {
    "0xac64598f1b7748542bafd6cee762d2d8f0f4a637",
    "0x1642b8e59e5f5394e7005f2e221bb4213a9f9de9",
    "0xa870c44f7e6e2e15d104185d3bbe5a54f9e2b52d",
    "0x598945346525e8cdcf6a8d441eb2c88bb36199b8",
    "0x0884051c662892bfcd8a132d1404ab8a3a409cea",
    "0x34827044cbd4b808fc1b189fce9f50e6dafae7c9",
    "0xd5c7757d91cbd59543c8ef64738d806e57d2212b",
    "0x9c7035f6c578f0d0372155d23da1c9345e8c89d9",
    "0x1e9b03ec06f463bc5d81c15e050425639cc43855",
    "0x7ab12f7a0925ef24927343d47199e75a91fc78aa",
    "0xb67c4c91728e6865b216e11d54b50c75bbd3fd5f",
    "0x5a024dd5fa786c7759c36c578384ea705e0bd06f",
    "0xf71ab4e4172730ed096b832f37982942935c7215",
    "0x40c75d831744818e01c768db25aa093163d34927",
}

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
}


def http_get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "next-k-deep/1.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


def hl(body: dict):
    req = urllib.request.Request(
        INFO,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "next-k-deep/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode())


def is_followable_coin(coin: str) -> bool:
    c = (coin or "").upper()
    if c in MAJORS:
        return True
    if c.startswith("XYZ:"):
        return c.split(":", 1)[-1] in STOCKS
    return False


def score_copyable(addr: str) -> dict | None:
    fills = hl({"type": "userFills", "user": addr})
    if not isinstance(fills, list):
        fills = []
    now = int(time.time() * 1000)
    d1 = now - 86400000
    d7 = now - 7 * 86400000
    f1 = [f for f in fills if int(f.get("time") or 0) >= d1]
    f7 = [f for f in fills if int(f.get("time") or 0) >= d7]
    fph = len(f1) / 24.0
    if fph > 25 or fph < 0.15:
        return None
    if len(f7) < 12:
        return None

    coins = Counter(str(f.get("coin") or "") for f in f7)
    top = coins.most_common(8)
    total = sum(coins.values()) or 1
    c1 = top[0][1] / total if top else 0
    c2 = (top[0][1] + top[1][1]) / total if len(top) > 1 else c1

    closed = [f for f in f7 if f.get("closedPnl") not in (None, "")]
    wins = losses = 0
    pnl = 0.0
    scratch = 0
    for f in closed:
        try:
            c = float(f.get("closedPnl") or 0)
        except (TypeError, ValueError):
            continue
        pnl += c
        if c > 0:
            wins += 1
        elif c < 0:
            losses += 1
        try:
            n = abs(float(f.get("sz") or 0) * float(f.get("px") or 0))
            if n > 0 and abs(c) / n < 0.0005:
                scratch += 1
        except (TypeError, ValueError):
            pass
    ncl = wins + losses
    if ncl < 5 or pnl <= 0:
        return None
    wr = wins / ncl
    scratch_r = scratch / len(closed) if closed else 1.0
    if wr < 0.52 or scratch_r > 0.78:
        return None

    top_coin = top[0][0] if top else ""
    major_share = (
        sum(v for k, v in coins.items() if is_followable_coin(k)) / total
    )
    bad_primary = top_coin.upper() in {"CASHCAT", "FARTCOIN", "PUMP"} or top_coin.startswith("@")
    if bad_primary and major_share < 0.25:
        return None

    st = hl({"type": "clearinghouseState", "user": addr})
    av = float((st.get("marginSummary") or {}).get("accountValue") or 0)
    if av < 5000:
        return None

    if c1 >= 0.55:
        specialty = "single:" + top_coin
    elif c2 >= 0.70:
        specialty = "duo:" + top[0][0] + "+" + top[1][0]
    else:
        specialty = "mix:" + ",".join(x[0] for x in top[:3])

    pace_pen = 0.0 if 1 <= fph <= 12 else abs(fph - 6) * 0.15
    conc_bonus = 1.2 if c2 >= 0.65 else (1.0 if c2 >= 0.45 else 0.7)
    score = wr * 2.0 + min(pnl / 20000.0, 2.0) + conc_bonus - pace_pen - scratch_r * 1.5

    follow_coins = [k for k, _ in top if is_followable_coin(k)][:3]
    return {
        "addr": addr,
        "av": round(av, 1),
        "fph24": round(fph, 2),
        "fills7": len(f7),
        "wr7": round(wr, 3),
        "pnl7": round(pnl, 1),
        "scratch": round(scratch_r, 3),
        "c1": round(c1, 2),
        "c2": round(c2, 2),
        "specialty": specialty,
        "top": top[:5],
        "follow_coins": follow_coins,
        "score": round(score, 3),
        "major_share": round(major_share, 2),
    }


def main() -> None:
    raw = http_get(LEADER)
    rows = raw if isinstance(raw, list) else (raw.get("leaderboardRows") or raw.get("data") or [])
    print("leaderboard_rows", len(rows))

    parsed = []
    for row in rows:
        addr = (row.get("ethAddress") or "").lower()
        if not addr or addr in KNOWN:
            continue
        av = float(row.get("accountValue") or 0)
        if av < 8000:
            continue
        perf = {}
        for w in row.get("windowPerformances") or []:
            if isinstance(w, list) and len(w) >= 2 and isinstance(w[1], dict):
                perf[w[0]] = w[1]
        week = perf.get("week") or {}
        week_pnl = float(week.get("pnl") or 0)
        week_vlm = float(week.get("vlm") or 0)
        if week_pnl <= 0 or week_vlm < 30000:
            continue
        turn = week_vlm / av if av else 0
        parsed.append(
            {
                "addr": addr,
                "av": av,
                "week_pnl": week_pnl,
                "week_vlm": week_vlm,
                "turn": turn,
            }
        )
    print("profitable_week_candidates", len(parsed))

    pool = []
    seen = set()
    for p in sorted(parsed, key=lambda x: (-x["week_pnl"], -x["av"]))[:100]:
        if p["addr"] in seen:
            continue
        seen.add(p["addr"])
        pool.append(p)
    mid = [p for p in parsed if 2 <= p["turn"] <= 50]
    for p in sorted(mid, key=lambda x: -x["week_pnl"])[:80]:
        if p["addr"] in seen:
            continue
        seen.add(p["addr"])
        pool.append(p)
    print("deep_pool", len(pool))

    hits = []
    for p in pool[:120]:
        try:
            r = score_copyable(p["addr"])
            if not r:
                continue
            r["lb_week_pnl"] = round(p["week_pnl"], 1)
            r["lb_turn"] = round(p["turn"], 2)
            hits.append(r)
            print(
                "HIT",
                r["score"],
                r["addr"],
                r["specialty"],
                "fph",
                r["fph24"],
                "wr",
                r["wr7"],
                "pnl7",
                r["pnl7"],
                "follow",
                r["follow_coins"],
            )
        except Exception as exc:
            print("err", p["addr"][:12], type(exc).__name__, exc)
        time.sleep(0.22)
        if len(hits) >= 20:
            break

    hits.sort(key=lambda x: -x["score"])
    print("\n=== TOP NEW COPYABLE ===")
    print(json.dumps(hits[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

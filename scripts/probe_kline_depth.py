#!/usr/bin/env python3
"""探测 Bitget / Binance SPY/QQQ 永续 5m 可拉取深度。"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from quant.common.kline_cache import norm_symbol
from quant.market.binance import fetch_klines_forward as bn_fwd
from quant.market.bitget import _public_get, fetch_klines_forward as bg_fwd


def ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def fmt(msv: int) -> str:
    return datetime.fromtimestamp(msv / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def bitget_backward(sym: str, start_ms: int, end_ms: int) -> list[int]:
    sym = norm_symbol(sym)
    cur_end = end_ms
    out: list[int] = []
    seen: set[int] = set()
    while cur_end > start_ms and len(out) < 200_000:
        batch = _public_get(
            "/api/v2/mix/market/candles",
            {
                "symbol": sym,
                "productType": "USDT-FUTURES",
                "granularity": "5m",
                "endTime": str(cur_end),
                "limit": "1000",
            },
        )
        if not isinstance(batch, list) or not batch:
            break
        rows = sorted(batch, key=lambda r: int(r[0]))
        first = int(rows[0][0])
        for row in rows:
            ot = int(row[0])
            if ot < start_ms or ot > end_ms or ot in seen:
                continue
            seen.add(ot)
            out.append(ot)
        if first <= start_ms or len(rows) < 1000:
            break
        cur_end = first - 1
    out.sort()
    return out


def main() -> None:
    end = int(time.time() * 1000)
    tests = [
        ("QQQUSDT", "bitget", ms("2025-10-28")),
        ("SPYUSDT", "bitget", ms("2026-02-11")),
        ("QQQUSDT", "binance", ms("2026-04-06")),
        ("SPYUSDT", "binance", ms("2026-04-06")),
    ]
    for sym, ex, start in tests:
        print(f"=== {sym} @ {ex} from listing ===")
        if ex == "bitget":
            bwd = bitget_backward(sym, start, end)
            if bwd:
                print(f"  backward: {len(bwd):6d} bars  {fmt(bwd[0])} -> {fmt(bwd[-1])}")
            else:
                print("  backward: 0 bars")
            fwd = bg_fwd(sym, "5m", start, end)
            if fwd:
                print(f"  forward:  {len(fwd):6d} bars  {fmt(int(fwd[0][0]))} -> {fmt(int(fwd[-1][0]))}")
            else:
                print("  forward:  0 bars")
        else:
            fwd = bn_fwd(sym, "5m", start, end)
            if fwd:
                print(f"  forward:  {len(fwd):6d} bars  {fmt(int(fwd[0][0]))} -> {fmt(int(fwd[-1][0]))}")
            else:
                print("  forward:  0 bars")
        days = (end - start) / 86_400_000
        print(f"  listing->now ~{days:.0f} calendar days")
        print()


if __name__ == "__main__":
    main()

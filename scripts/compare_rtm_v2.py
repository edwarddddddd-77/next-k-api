#!/usr/bin/env python3
"""Quick BTC 4H RTM v2 scan summary."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.rtm_patterns import RTMConfig, scan_rtm_patterns
from quant.rtm_patterns.scanner import hits_to_dataframe, pattern_counts

path = ROOT.parent / "quant-nine-schools/nautilus_trader/tests/test_data/btc-perp-20211231-20220201_1m.csv"
df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
ohlc4h = df.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()

hits = scan_rtm_patterns(ohlc4h, config=RTMConfig())
counts = pattern_counts(hits)
out = hits_to_dataframe(hits)

print("=== RTM v2.1 BTC 4H ===")
print(f"Total: {len(hits)}")
for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {name}: {cnt}")
if not out.empty:
    zoned = int(out["zone_source"].notna().sum())
    print(f"Avg quality: {out['quality'].mean():.3f}")
    print(f"With S/D zone: {zoned}/{len(out)} ({100*zoned/len(out):.0f}%)")

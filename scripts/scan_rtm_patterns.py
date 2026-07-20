#!/usr/bin/env python3
"""Scan OHLC CSV/data for RTM institutional price action patterns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.rtm_patterns import RTMConfig, scan_rtm_patterns
from quant.rtm_patterns.scanner import RTM_PATTERN_IDS, hits_to_dataframe, pattern_counts


def _load_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    for alias, target in (
        ("datetime", "timestamp"),
        ("date", "timestamp"),
        ("time", "timestamp"),
    ):
        if alias in cols and target not in {c.lower() for c in df.columns}:
            df = df.rename(columns={cols[alias]: "timestamp"})
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan RTM institutional price action patterns")
    parser.add_argument("csv", type=Path, help="OHLC CSV (open,high,low,close)")
    parser.add_argument(
        "-p",
        "--pattern",
        action="append",
        choices=RTM_PATTERN_IDS,
        help="Filter to specific pattern(s); repeatable",
    )
    parser.add_argument("--pivot-left", type=int, default=3)
    parser.add_argument("--pivot-right", type=int, default=3)
    parser.add_argument("--eq-tolerance", type=float, default=0.15, help="Equal H/L tolerance %%")
    parser.add_argument("-o", "--output", type=Path, help="Save hits to CSV")
    parser.add_argument("--summary", action="store_true", help="Print pattern counts only")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"File not found: {args.csv}", file=sys.stderr)
        return 1

    df = _load_ohlc(args.csv)
    cfg = RTMConfig(
        pivot_left=args.pivot_left,
        pivot_right=args.pivot_right,
        eq_tolerance_pct=args.eq_tolerance,
    )
    hits = scan_rtm_patterns(df, config=cfg, patterns=args.pattern)

    if args.summary:
        counts = pattern_counts(hits)
        print(f"Total hits: {len(hits)}")
        for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {name}: {cnt}")
        return 0

    out = hits_to_dataframe(hits)
    if out.empty:
        print("No patterns detected.")
        return 0

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"Saved {len(out)} hits -> {args.output}")
    else:
        pd.set_option("display.max_rows", 50)
        pd.set_option("display.width", 200)
        print(out.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

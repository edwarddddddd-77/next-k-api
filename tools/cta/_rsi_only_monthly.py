"""RSI 汇合 only -?2022 Model monthly breakdown 2026 H1."""
from __future__ import annotations

import pandas as pd

from tools.cta.research_ict_master_suite import (
    SimConfig,
    backtest_2022_model,
    fetch_ohlcv,
    monthly_breakdown,
    print_data_coverage,
    resample_ohlcv,
)
from tools.cta.rsi_core_levels import RSICoreConfig, build_rsi_core_levels_df, near_level_row, row_at_time

PROX_VALUES = (0.005, 0.01)  # 0.5% and 1%


def run_one(df_5m, rsi_df, start_ms, end_ms, cfg, prox: float) -> None:
    def rsi_only(t: int, side: str, entry: float) -> bool:
        row = row_at_time(rsi_df, t)
        return near_level_row(row, side=side, entry_px=entry, proximity_pct=prox)

    res = backtest_2022_model(
        "ETHUSDT", df_5m, test_start_ms=start_ms, test_end_ms=end_ms, cfg=cfg, entry_allow=rsi_only
    )
    print(
        f"\n{'='*60}\n汇合 ±{prox*100:.1f}% | trades={len(res.trades)} win={res.win_rate:.1f}% "
        f"ret={res.total_return_pct:+.2f}% final=${res.final_equity:.2f}"
    )
    mb = monthly_breakdown(res)
    print("\n月度明细 (按平仓月, 复利):")
    print(mb.to_string(index=False))


def main() -> None:
    cfg = SimConfig(commission_pct=0, position_pct=1.0, rr=1.5, leverage=2.0)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-06-30 23:59:59", tz="UTC")
    warmup = start - pd.Timedelta(days=14)
    start_ms = int(start.value // 1_000_000)
    end_ms = int(end.value // 1_000_000)
    warm_ms = int(warmup.value // 1_000_000)

    print("RSI 汇合 only | 2022 Model | ETH 2026 H1")
    print("仓位: 全仓 position_pct=1.0 | 杠杆: 2x | R:R=1:1.5 | 无手续费")
    print("复利: 每笔按当-?equity -?notional (= equity × 1.0 × 2.0)\n")

    mid = pd.Timestamp("2026-03-15", tz="UTC")
    df_a = fetch_ohlcv("ETHUSDT", "1m", warm_ms, int(mid.value // 1_000_000))
    df_b = fetch_ohlcv("ETHUSDT", "1m", int(mid.value // 1_000_000), end_ms)
    df_1m = (
        pd.concat([df_a, df_b], ignore_index=True)
        .drop_duplicates(subset=["open_time"], keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    df_5m = resample_ohlcv(df_1m, "5m")
    print_data_coverage("ETHUSDT", df_5m, start_ms, end_ms)

    rsi_df = build_rsi_core_levels_df(df_5m, RSICoreConfig())
    for prox in PROX_VALUES:
        run_one(df_5m, rsi_df, start_ms, end_ms, cfg, prox)


if __name__ == "__main__":
    main()

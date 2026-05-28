#!/usr/bin/env python3
"""官方 Moss 工厂 vs Next-K moss_quant：同一 CSV、同一参数、同一 bar 窗口对照。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FACTORY_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "moss-trade-bot-skills-main"
    / "moss-trade-bot-factory-1.0.24"
    / "scripts"
)
DATA_CACHE = FACTORY_SCRIPTS / "data_cache"
BARS = 1500
CAPITAL = 10000.0

CASES = [
    {
        "name": "BTC 寻优最优 (trend)",
        "csv": "hyperliquid_BTCUSDC_15m_2025-10-06_148d.csv",
        "symbol": "BTCUSDC",
        "template": "trend",
        "tactical": {
            "entry_threshold": 0.48,
            "exit_threshold": 0.12,
            "sl_atr_mult": 2.0,
            "tp_rr_ratio": 2.5,
            "regime_sensitivity": 0.55,
        },
    },
    {
        "name": "BTC 默认 (balanced)",
        "csv": "hyperliquid_BTCUSDC_15m_2025-10-06_148d.csv",
        "symbol": "BTCUSDC",
        "template": "balanced",
        "tactical": {},
    },
    {
        "name": "SOL 寻优最优 (momentum)",
        "csv": "hyperliquid_SOLUSDC_15m_2025-10-06_148d.csv",
        "symbol": "SOLUSDC",
        "template": "momentum",
        "tactical": {
            "entry_threshold": 0.48,
            "exit_threshold": 0.12,
            "sl_atr_mult": 2.5,
            "tp_rr_ratio": 2.5,
            "regime_sensitivity": 0.55,
        },
    },
]


def _build_params(template: str, tactical: dict) -> dict:
    from moss_quant.params import build_initial_params, resolve_params_dict
    from moss_quant.params import cap_leverage_for_symbol

    p = build_initial_params(template=template)
    p.update(tactical)
    return p


def _slice_csv(csv_path: Path, bars: int, out_path: Path) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    tail = df.tail(bars).copy()
    t0 = str(tail["timestamp"].iloc[0])
    t1 = str(tail["timestamp"].iloc[-1])
    tail.to_csv(out_path, index=False)
    return tail, t0, t1


def _buy_hold_return(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    c0 = float(df["close"].iloc[0])
    c1 = float(df["close"].iloc[-1])
    if c0 <= 0:
        return 0.0
    return round((c1 - c0) / c0, 4)


def _run_factory(data_csv: Path, params: dict, symbol: str) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            str(FACTORY_SCRIPTS / "run_backtest.py"),
            "--data",
            str(data_csv),
            "--params",
            json.dumps(params),
            "--capital",
            str(CAPITAL),
            "--regime-version",
            "v1",
        ],
        cwd=str(FACTORY_SCRIPTS),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "factory backtest failed")
    out = json.loads(proc.stdout)
    br = out.get("backtest_result") or {}
    return {
        "total_return": round(float(br.get("total_return", 0)), 4),
        "sharpe": round(float(br.get("sharpe_ratio", 0)), 4),
        "max_drawdown": round(float(br.get("max_drawdown", 0)), 4),
        "total_trades": int(br.get("total_trades", 0)),
        "win_rate": round(float(br.get("win_rate", 0)), 4),
        "profit_factor": round(float(br.get("profit_factor", 0)), 4),
    }


def _run_nextk(df: pd.DataFrame, params: dict, symbol: str) -> dict:
    from moss_quant.core.backtest import run_backtest
    from moss_quant.core.decision import DecisionParams
    from moss_quant.core.regime import classify_regime
    from moss_quant.params import cap_leverage_for_symbol, resolve_params_dict

    params = cap_leverage_for_symbol(resolve_params_dict(params), symbol)
    regime = classify_regime(df, version="v1")
    p = DecisionParams.from_dict(params)
    result = run_backtest(
        df,
        p,
        regime,
        initial_capital=CAPITAL,
        symbol=symbol,
    )
    return {
        "total_return": round(float(result.total_return), 4),
        "sharpe": round(float(result.sharpe_ratio), 4),
        "max_drawdown": round(float(result.max_drawdown), 4),
        "total_trades": int(result.total_trades),
        "win_rate": round(float(result.win_rate), 4),
        "profit_factor": round(float(result.profit_factor), 4),
    }


def main() -> None:
    if not FACTORY_SCRIPTS.is_dir():
        print("ERROR: factory not found:", FACTORY_SCRIPTS, file=sys.stderr)
        sys.exit(1)

    tmp_dir = ROOT / "data" / "compare_factory_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in CASES:
        csv_path = DATA_CACHE / case["csv"]
        if not csv_path.is_file():
            print("SKIP missing csv:", csv_path)
            continue
        slug = case["csv"].replace(".csv", "")[-40:]
        slice_path = tmp_dir / f"slice_{slug}_{BARS}.csv"
        df, t0, t1 = _slice_csv(csv_path, BARS, slice_path)
        params = _build_params(case["template"], case["tactical"])
        bh = _buy_hold_return(df)

        fac = _run_factory(slice_path, params, case["symbol"])
        nk = _run_nextk(df, params, case["symbol"])

        rows.append(
            {
                "case": case["name"],
                "window": f"{t0} .. {t1}",
                "bars": len(df),
                "buy_hold_return": bh,
                "factory": fac,
                "next_k": nk,
                "delta_return": round(nk["total_return"] - fac["total_return"], 4),
            }
        )

    print("=" * 72)
    print("Moss 官方工厂 (v1.0.24 HL CSV) vs Next-K moss_quant")
    print(f"窗口: 最近 {BARS} 根 15m · 资金 {CAPITAL} · regime v1")
    print("=" * 72)
    for r in rows:
        print()
        print(f"【{r['case']}】")
        print(f"  时间: {r['window']} ({r['bars']} bars)")
        print(f"  持币涨跌 buy&hold: {r['buy_hold_return']:+.2%}")
        f, n = r["factory"], r["next_k"]
        print(
            f"  官方工厂  ret={f['total_return']:+.4f}  sharpe={f['sharpe']:.3f}  "
            f"trades={f['total_trades']}  win={f['win_rate']:.1%}  mdd={f['max_drawdown']:.4f}"
        )
        print(
            f"  Next-K    ret={n['total_return']:+.4f}  sharpe={n['sharpe']:.3f}  "
            f"trades={n['total_trades']}  win={n['win_rate']:.1%}  mdd={n['max_drawdown']:.4f}"
        )
        print(f"  收益差 (Next-K - 官方): {r['delta_return']:+.4f}")
    print()
    print("JSON:", json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

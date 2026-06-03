"""SUIUSDT：网格寻优 + 70% 归因 + 局部精修，打印对比。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi

load_env_oi()

from moss_quant import config as cfg
from moss_quant.backtest_service import run_full_backtest
from moss_quant.kline_cache import load_cached
from moss_quant.optimize_service import run_strategy_optimize
from moss_quant.params import build_initial_params, resolve_params_dict

SYM = "SUIUSDT"


def _best_params(best: dict) -> dict:
    tpl = str(best.get("template") or "balanced")
    params = build_initial_params(template=tpl)
    params.update(best.get("tactical_params") or {})
    if best.get("params"):
        for k, v in (best.get("params") or {}).items():
            if k in params or k in ("trailing_enabled",):
                params[k] = v
    return resolve_params_dict(params)


def _print_pipeline(sm: dict) -> None:
    pipe = sm.get("post_grid_pipeline") or {}
    if pipe.get("error"):
        print(f"  pipeline ERROR: {pipe.get('error')}")
        return
    print(f"  param_source={sm.get('param_source', '?')}  refine_improved={sm.get('refine_improved')}")
    print(
        f"  验证收益: grid {float(sm.get('grid_val_return') or 0) * 100:+.2f}% "
        f"-> final {float(sm.get('val_return') or 0) * 100:+.2f}%"
    )
    print(
        f"  验证Sharpe: grid {float(sm.get('grid_val_sharpe') or 0):.3f} "
        f"-> final {float(sm.get('val_sharpe') or 0):.3f}"
    )
    diag = pipe.get("tuning_diagnosis") or {}
    if diag.get("suggestion"):
        print(f"  70%归因: {diag['suggestion'].get('narrative', '')[:120]}")
    refine = pipe.get("local_refine") or {}
    if refine:
        print(
            f"  精修(WF): rounds={refine.get('rounds_run')} "
            f"improved={refine.get('improved_vs_grid')} "
            f"wf={refine.get('refined_wf_passed_folds')}/{refine.get('grid_wf_passed_folds')}折基线"
        )
        print(f"  {refine.get('narrative', '')}")
        for r in refine.get("rounds") or []:
            print(
                f"    round{r['round']}: improved={r.get('improved')} "
                f"val_ret={float(r.get('val_return') or 0) * 100:+.2f}% "
                f"cands={r.get('candidates_tested')}"
            )
            tact = r.get("best_tactical") or {}
            print(
                f"      entry={tact.get('entry_threshold')} sl={tact.get('sl_atr_mult')} "
                f"tp={tact.get('tp_rr_ratio')} exit={tact.get('exit_threshold')}"
            )
    ta = pipe.get("final_tactical_params") or {}
    if ta:
        print(
            f"  最终战术: entry={ta.get('entry_threshold')} sl={ta.get('sl_atr_mult')} "
            f"tp={ta.get('tp_rr_ratio')} exit={ta.get('exit_threshold')}"
        )


def main() -> None:
    capital = float(cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    print(
        f">>> {SYM} | source={cfg.MOSS_QUANT_DATA_SOURCE} "
        f"bars={cfg.MOSS_QUANT_RESEARCH_KLINE_BARS} "
        f"tuning_diag={cfg.MOSS_QUANT_OPTIMIZE_TUNING_DIAG_ENABLED} "
        f"local_refine={cfg.MOSS_QUANT_OPTIMIZE_LOCAL_REFINE_ENABLED} "
        f"max_rounds={cfg.MOSS_QUANT_OPTIMIZE_LOCAL_REFINE_MAX_ROUNDS}"
    )
    print("=" * 72)

    df = load_cached(SYM, refresh=True, research=True)
    print(f"K线 {len(df)} bars  {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")

    print("\n[1] balanced 全窗")
    out0 = run_full_backtest(
        symbol=SYM,
        params=build_initial_params(template="balanced"),
        capital=capital,
        refresh_klines=False,
    )
    s0 = out0.get("summary") or {}
    print(
        f"  return={float(s0.get('total_return') or 0) * 100:+.2f}% "
        f"sharpe={float(s0.get('sharpe') or 0):.3f} "
        f"win={float(s0.get('win_rate') or 0) * 100:.1f}% "
        f"trades={s0.get('total_trades')}"
    )

    print("\n[2] 网格寻优 + 70/30 精修")
    opt = run_strategy_optimize(symbol=SYM, capital=capital, refresh_klines=False, top_n=3)
    best = opt.get("best")
    if not best:
        print(f"  无结果: {opt.get('warning')}")
        return

    sm = best.get("summary") or {}
    tact = best.get("tactical_params") or {}
    print(
        f"  template={best.get('template')} train={float(sm.get('train_return', 0) or 0) * 100:+.2f}% "
        f"pool={sm.get('pool_tier')} sync={sm.get('sync_allowed')}"
    )
    print(
        f"  战术 entry={tact.get('entry_threshold')} sl={tact.get('sl_atr_mult')} "
        f"tp={tact.get('tp_rr_ratio')}"
    )
    tr = sm.get("post_grid_pipeline", {}).get("tuning_diagnosis", {}).get("train_analysis") or {}
    if tr:
        print(
            f"  训练窗: win={float(tr.get('win_rate') or 0) * 100:.1f}% "
            f"PF={tr.get('profit_factor')} trades={tr.get('total_trades')} "
            f"flip_ratio={tr.get('signal_exit_ratio')}"
        )
    _print_pipeline(sm)

    print("\n[3] 最终参数 全窗 replay")
    bt = run_full_backtest(
        symbol=SYM, params=_best_params(best), capital=capital, refresh_klines=False
    )
    s = bt.get("summary") or {}
    print(
        f"  return={float(s.get('total_return') or 0) * 100:+.2f}% "
        f"sharpe={float(s.get('sharpe') or 0):.3f} "
        f"win={float(s.get('win_rate') or 0) * 100:.1f}% "
        f"trades={s.get('total_trades')}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()

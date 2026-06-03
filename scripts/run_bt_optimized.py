"""寻优 → 用 best 模板+战术参数做全窗 replay 回测。"""
from __future__ import annotations

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

SYMS = ("HYPEUSDT", "ARBUSDT", "APTUSDT", "ADAUSDT", "DOGEUSDT")


def _best_params(best: dict) -> dict:
    tpl = str(best.get("template") or "balanced")
    params = build_initial_params(template=tpl)
    params.update(best.get("tactical_params") or {})
    if best.get("params"):
        for k, v in (best.get("params") or {}).items():
            if k in params or k in ("trailing_enabled",):
                params[k] = v
    return resolve_params_dict(params)


def main() -> None:
    capital = float(cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    print(
        f"optimize grid max={cfg.MOSS_QUANT_OPTIMIZE_MAX_COMBINATIONS} "
        f"bars={cfg.MOSS_QUANT_RESEARCH_KLINE_BARS} capital={capital}"
    )
    print("=" * 88)
    for sym in SYMS:
        print(f"\n>>> {sym}")
        try:
            load_cached(sym, refresh=True, research=True)
            opt = run_strategy_optimize(
                symbol=sym, capital=capital, refresh_klines=False, top_n=3
            )
            best = opt.get("best")
            if not best or not best.get("summary"):
                print(f"    寻优无有效结果: {opt.get('warning') or 'no_valid_result'}")
                print(f"    tested={opt.get('combinations_tested')} ok={opt.get('combinations_ok')}")
                continue
            sm = best.get("summary") or {}
            tpl = best.get("template")
            tact = best.get("tactical_params") or {}
            print(
                f"    寻优 best: template={tpl} entry={tact.get('entry_threshold')} "
                f"sl={tact.get('sl_atr_mult')} tp={tact.get('tp_rr_ratio')}"
            )
            print(
                f"    训练 return={float(sm.get('train_return', sm.get('total_return')) or 0)*100:+.2f}% "
                f"val={float(sm.get('val_return') or 0)*100:+.2f}% "
                f"WF={sm.get('wf_passed_folds')}/{sm.get('wf_folds')} "
                f"pool={sm.get('pool_tier')} sync={sm.get('sync_allowed')}"
            )
            if sm.get("wf_reason"):
                print(f"    {sm.get('wf_reason')}")
            if sm.get("sync_block_reason"):
                print(f"    sync_block: {sm.get('sync_block_reason')}")

            params = _best_params(best)
            bt = run_full_backtest(
                symbol=sym, params=params, capital=capital, refresh_klines=False
            )
            s = bt.get("summary") or {}
            df = load_cached(sym, refresh=False, research=True)
            ret = float(s.get("total_return") or 0) * 100
            mdd = abs(float(s.get("max_drawdown") or 0)) * 100
            print(
                f"    全窗回测: return={ret:+7.2f}% sharpe={float(s.get('sharpe') or 0):6.3f} "
                f"mdd={mdd:5.2f}% trades={int(s.get('total_trades') or 0):3} "
                f"win={float(s.get('win_rate') or 0)*100:5.1f}%"
            )
            print(f"    kline {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
        except Exception as e:
            print(f"    ERROR: {e}")
    print("\n" + "=" * 88)


if __name__ == "__main__":
    main()

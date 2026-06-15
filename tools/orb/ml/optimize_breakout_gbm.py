#!/usr/bin/env python3
"""从 K 线样本 → GBM 超参 + Gate min_p 联合优化 → 部署 orb_live。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.ml.gbm import (  # noqa: E402
    GbmHyperParams,
    rows_to_xy_gbm,
    save_gbm,
    score_gbm_holdout,
    train_gbm,
)
from orb.ml.gate import LiveGateConfig  # noqa: E402
from orb.ml.live_bundle import bootstrap_from_legacy  # noqa: E402
from orb.ml.live_gate_sim import cached_symbols, run_gate_eval_sessions, trading_dates_from_samples  # noqa: E402
from orb.ml.paths import V2_EVAL, default_shared_samples_path, ensure_v2_dirs  # noqa: E402
from orb.ml.profiles import load_profiles  # noqa: E402
from orb.ml.ranker import BreakoutRanker  # noqa: E402
from orb.ml.samples import parse_symbol_list, split_holdout_by_date  # noqa: E402
from orb.ml.model.paths import (  # noqa: E402
    GBM_META,
    GBM_PKL,
    GBM_TRAIN_REPORT,
    ensure_model_dirs,
)
from orb.v2.paths import resolve_gate_config_path, resolve_symbols_path  # noqa: E402
from orb.v2.robots import init_robot_wallets, robot_count_from_env, robot_equity_from_env  # noqa: E402
from tools.orb.ml.eval_live_gate import _ml_cfg  # noqa: E402
from tools.orb.ml.sweep_breakout_gbm import _eval_one, _param_grid, _walk_forward_folds  # noqa: E402
from tools.orb.v2.backtest_universe import universe_session_dates  # noqa: E402

OPTIMIZE_JSON = V2_EVAL / "gbm_optimize.json"


def _run_py(rel: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "tools" / "orb" / "ml" / rel), *args]
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def sweep_hold30(rows: List[Dict[str, Any]], *, fold_days: int, folds: int, quick: bool) -> List[Dict[str, Any]]:
    fold_list = _walk_forward_folds(rows, fold_days=fold_days, n_folds=folds)
    grid = _param_grid(quick=quick)
    results: List[Dict[str, Any]] = []
    for hp in grid:
        fold_metrics = []
        for train_rows, test_rows in fold_list:
            fold_metrics.append(_eval_one(train_rows, test_rows, label_mode="hold_30m", hp=hp))
        sep = [float(f.get("holdout_separation") or 0.0) for f in fold_metrics]
        acc = [float(f.get("holdout_accuracy") or 0.0) for f in fold_metrics]
        score = [float(f.get("score") or 0.0) for f in fold_metrics]
        results.append(
            {
                "label_mode": "hold_30m",
                "hyperparams": hp.to_dict(),
                "fold_metrics": fold_metrics,
                "folds": len(fold_metrics),
                "mean_holdout_separation": round(sum(sep) / len(sep), 4) if sep else 0.0,
                "mean_holdout_accuracy": round(sum(acc) / len(acc), 4) if acc else 0.0,
                "mean_score": round(sum(score) / len(score), 4) if score else 0.0,
            }
        )
    results.sort(
        key=lambda r: (float(r["mean_holdout_separation"]), float(r["mean_score"])),
        reverse=True,
    )
    return results


def _gate_pnl(
    gbm,
    *,
    dates: Sequence[str],
    symbols: List[str],
    gate: LiveGateConfig,
    robot_wallets: List[float],
) -> Dict[str, Any]:
    profiles = load_profiles()
    ranker = BreakoutRanker(gbm=gbm, profiles=profiles, use_prior=True)
    days = run_gate_eval_sessions(
        dates=list(dates),
        symbols=symbols,
        gate=gate,
        ranker=ranker,
        respect_env_filters=True,
        robot_wallets=robot_wallets,
    )
    day_pnls = [
        sum(float(o.get("pnl_usdt") or 0.0) for o in (d.get("opened") or []))
        for d in days
    ]
    total = sum(day_pnls)
    opens = sum(int(d.get("opens") or 0) for d in days)
    win = sum(1 for p in day_pnls if p > 0)
    loss = sum(1 for p in day_pnls if p < 0)
    return {
        "sessions": len(days),
        "total_pnl_usdt": round(total, 1),
        "avg_daily_pnl_usdt": round(total / max(len(days), 1), 1),
        "win_days": win,
        "loss_days": loss,
        "avg_opens": round(opens / max(len(days), 1), 1),
    }


def joint_optimize(
    rows: List[Dict[str, Any]],
    sweep_rows: List[Dict[str, Any]],
    *,
    top: int,
    min_p_grid: Sequence[float],
    bt_sessions: int,
    holdout_days: int,
) -> List[Dict[str, Any]]:
    syms = cached_symbols(parse_symbol_list(resolve_symbols_path().read_text(encoding="utf-8")))
    cfg = _ml_cfg(compound_per_symbol=True, respect_env_filters=True)
    cache_dates = universe_session_dates(syms, cfg)
    dates = cache_dates[-max(1, bt_sessions) :]
    rc = robot_count_from_env()
    re = robot_equity_from_env()
    robots = init_robot_wallets(count=rc, equity_usdt=re)
    gate_base = LiveGateConfig.from_json(resolve_gate_config_path())
    train_rows, _ = split_holdout_by_date(rows, holdout_days=holdout_days)

    candidates = sweep_rows[: max(1, top)]
    out: List[Dict[str, Any]] = []
    total_runs = len(candidates) * len(min_p_grid)
    run_i = 0
    for cand in candidates:
        hp = GbmHyperParams.from_dict(cand["hyperparams"])
        X, y = rows_to_xy_gbm(train_rows, label_mode="hold_30m")
        gbm = train_gbm(X, y, label_mode="hold_30m", hyperparams=hp)
        for min_p in min_p_grid:
            run_i += 1
            gate = LiveGateConfig.from_json(resolve_gate_config_path())
            gate.min_p_true = float(min_p)
            print(f"[pnl] {run_i}/{total_runs} sep={cand['mean_holdout_separation']:.4f} min_p={min_p:.2f} ...", flush=True)
            pnl = _gate_pnl(
                gbm,
                dates=dates,
                symbols=syms,
                gate=gate,
                robot_wallets=list(robots),
            )
            row = {
                "hyperparams": hp.to_dict(),
                "mean_holdout_separation": cand["mean_holdout_separation"],
                "mean_holdout_accuracy": cand["mean_holdout_accuracy"],
                "min_p_true": float(min_p),
                **pnl,
            }
            out.append(row)
            print(
                f"  -> pnl={pnl['total_pnl_usdt']:+.1f}U avg/d={pnl['avg_daily_pnl_usdt']:+.1f}U "
                f"opens={pnl['avg_opens']:.1f} W/L={pnl['win_days']}/{pnl['loss_days']}",
                flush=True,
            )
    out.sort(key=lambda x: float(x.get("total_pnl_usdt") or 0.0), reverse=True)
    return out


def deploy_best(
    rows: List[Dict[str, Any]],
    best: Dict[str, Any],
    *,
    holdout_days: int,
    update_gate: bool,
) -> Dict[str, Any]:
    hp = GbmHyperParams.from_dict(best["hyperparams"])
    train_rows, test_rows = split_holdout_by_date(rows, holdout_days=holdout_days)
    X, y = rows_to_xy_gbm(train_rows, label_mode="hold_30m")
    model = train_gbm(X, y, label_mode="hold_30m", hyperparams=hp)
    holdout = score_gbm_holdout(model, test_rows)

    ensure_model_dirs()
    save_gbm(model, GBM_PKL, GBM_META)
    report = {
        "kind": "gbm",
        "label_mode": "hold_30m",
        "hyperparams": hp.to_dict(),
        "samples_total": len(rows),
        "train_n": len(train_rows),
        "holdout_n": len(test_rows),
        **model.metrics,
        **holdout,
        "optimize_min_p": best.get("min_p_true"),
        "optimize_pnl_60d": best.get("total_pnl_usdt"),
    }
    GBM_TRAIN_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    gate_path = resolve_gate_config_path()
    gate_data = json.loads(gate_path.read_text(encoding="utf-8"))
    old_min_p = float(gate_data.get("min_p_true", 0.35))
    new_min_p = float(best.get("min_p_true") or old_min_p)
    if update_gate and abs(new_min_p - old_min_p) > 1e-6:
        gate_data["min_p_true"] = new_min_p
        gate_data["notes"] = (
            str(gate_data.get("notes") or "")
            + f" | optimize_breakout_gbm min_p {old_min_p:.2f}->{new_min_p:.2f}"
        ).strip()
        gate_path.write_text(json.dumps(gate_data, indent=2, ensure_ascii=False), encoding="utf-8")

    copied = bootstrap_from_legacy(overwrite=True)
    return {"report": report, "copied": copied, "gate_min_p": new_min_p}


def main() -> int:
    load_env_oi()
    ap = argparse.ArgumentParser(description="Optimize ORB GBM + Gate min_p from kline samples")
    ap.add_argument("--days", type=float, default=180.0)
    ap.add_argument("--holdout-days", type=int, default=10)
    ap.add_argument("--fold-days", type=int, default=7)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="较小超参网格")
    ap.add_argument("--top", type=int, default=5, help="进入 PnL 联合优化的 Top 超参数")
    ap.add_argument("--bt-sessions", type=int, default=60, help="PnL 回测 session 数")
    ap.add_argument("--min-p-grid", default="0.32,0.35,0.38,0.40,0.42")
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--skip-relabel", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true", help="跳过超参 sweep（用已有 gbm_optimize/sweep 的 top）")
    ap.add_argument("--skip-deploy", action="store_true")
    ap.add_argument("--no-update-gate", action="store_true")
    ap.add_argument("--json-out", default=str(OPTIMIZE_JSON))
    args = ap.parse_args()

    t0 = time.time()
    samples_path = default_shared_samples_path()

    if not args.skip_collect:
        _run_py("collect_shared_breakout_samples.py", "--days", str(args.days))
    if not args.skip_relabel:
        _run_py("relabel_hold30_samples.py", "--samples", str(samples_path))

    rows = list(json.loads(samples_path.read_text(encoding="utf-8")).get("rows") or [])
    if not rows or not any("hold30_true" in r for r in rows):
        print("No hold30 labeled samples")
        return 1
    print(f"[optimize] samples={len(rows)}", flush=True)

    sweep_rows = sweep_hold30(rows, fold_days=args.fold_days, folds=args.folds, quick=bool(args.quick))
    if args.skip_sweep and OPTIMIZE_JSON.is_file():
        prev = json.loads(OPTIMIZE_JSON.read_text(encoding="utf-8"))
        sweep_rows = prev.get("sweep_top10") or sweep_rows
    min_p_grid = [float(x.strip()) for x in args.min_p_grid.split(",") if x.strip()]
    joint_rows = joint_optimize(
        rows,
        sweep_rows,
        top=int(args.top),
        min_p_grid=min_p_grid,
        bt_sessions=int(args.bt_sessions),
        holdout_days=int(args.holdout_days),
    )
    best = joint_rows[0] if joint_rows else {}

    deploy_info: Optional[Dict[str, Any]] = None
    if best and not args.skip_deploy:
        deploy_info = deploy_best(
            rows,
            best,
            holdout_days=int(args.holdout_days),
            update_gate=not bool(args.no_update_gate),
        )

    out = {
        "samples": len(rows),
        "sweep_top10": sweep_rows[:10],
        "joint_top10": joint_rows[:10],
        "best": best,
        "deploy": deploy_info,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    ensure_v2_dirs()
    Path(args.json_out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== 最优配置 ===")
    if best:
        print(f"min_p_true={best.get('min_p_true')} hyperparams={best.get('hyperparams')}")
        print(
            f"PnL {best.get('total_pnl_usdt'):+.1f}U / {best.get('sessions')}d | "
            f"sep={best.get('mean_holdout_separation')} acc={best.get('mean_holdout_accuracy')}"
        )
    if deploy_info:
        r = deploy_info["report"]
        print(f"deployed holdout_separation={r.get('holdout_separation')} gate min_p={deploy_info.get('gate_min_p')}")
    print(f"report -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

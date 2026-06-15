#!/usr/bin/env python3
"""GBM 超参 + 标签 walk-forward 扫描；可选对 Top 候选做 Gate PnL 重放。"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from orb.ml.gbm import (  # noqa: E402
    DEFAULT_GBM_HYPERPARAMS,
    GbmHyperParams,
    score_gbm_holdout,
    train_gbm,
)
from orb.ml.gate import LiveGateConfig  # noqa: E402
from orb.ml.live_gate_sim import (  # noqa: E402
    cached_symbols,
    run_gate_eval_sessions,
    trading_dates_from_samples,
)
from orb.ml.paths import V2_GBM_SWEEP, default_shared_samples_path, ensure_v2_dirs  # noqa: E402
from orb.ml.profiles import load_profiles  # noqa: E402
from orb.ml.ranker import BreakoutRanker  # noqa: E402
from orb.ml.samples import parse_symbol_list, split_holdout_by_date  # noqa: E402
from orb.v2.paths import resolve_gate_config_path, resolve_symbols_path  # noqa: E402
from orb.v2.robots import init_robot_wallets, robot_count_from_env, robot_equity_from_env  # noqa: E402

LABEL_MODES = ("hold_30m", "true_breakout", "quality")


def _walk_forward_folds(
    rows: List[Dict[str, Any]], *, fold_days: int, n_folds: int
) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    dates = sorted({str(r.get("session_date") or "") for r in rows if r.get("session_date")})
    if len(dates) <= fold_days or n_folds <= 0:
        train, test = split_holdout_by_date(rows, holdout_days=fold_days)
        return [(train, test)] if test else []

    folds: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = []
    for i in range(n_folds):
        end_idx = len(dates) - i * fold_days
        start_idx = end_idx - fold_days
        if start_idx < fold_days:
            break
        hold = set(dates[start_idx:end_idx])
        train_dates = set(dates[:start_idx])
        train = [r for r in rows if str(r.get("session_date") or "") in train_dates]
        test = [r for r in rows if str(r.get("session_date") or "") in hold]
        if train and test:
            folds.append((train, test))
    return folds or []


def _eval_one(
    train_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    *,
    label_mode: str,
    hp: GbmHyperParams,
) -> Dict[str, Any]:
    from orb.ml.gbm import rows_to_xy_gbm

    X, y = rows_to_xy_gbm(train_rows, label_mode=label_mode)
    model = train_gbm(X, y, label_mode=label_mode, hyperparams=hp)
    holdout = score_gbm_holdout(model, test_rows)
    train_acc = float(model.metrics.get("train_accuracy") or 0.0)
    hold_acc = float(holdout.get("holdout_accuracy") or 0.0)
    separation = float(holdout.get("holdout_separation") or 0.0)
    brier = float(model.metrics.get("brier") or 0.0)
    overfit_gap = max(0.0, train_acc - hold_acc)
    score = separation - 0.35 * max(0.0, overfit_gap - 0.12) - 0.05 * max(0.0, brier - 0.18)
    return {
        "train_n": len(train_rows),
        "holdout_n": len(test_rows),
        "train_accuracy": train_acc,
        "holdout_accuracy": hold_acc,
        "holdout_separation": separation,
        "brier": brier,
        "overfit_gap": round(overfit_gap, 4),
        "score": round(score, 4),
        **holdout,
    }


def _aggregate_fold_metrics(folds: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not folds:
        return {}
    sep = [float(f.get("holdout_separation") or 0.0) for f in folds]
    acc = [float(f.get("holdout_accuracy") or 0.0) for f in folds]
    score = [float(f.get("score") or 0.0) for f in folds]
    return {
        "folds": len(folds),
        "mean_holdout_separation": round(sum(sep) / len(sep), 4),
        "mean_holdout_accuracy": round(sum(acc) / len(acc), 4),
        "mean_score": round(sum(score) / len(score), 4),
        "min_holdout_separation": round(min(sep), 4),
    }


def _param_grid(*, quick: bool) -> List[GbmHyperParams]:
    if quick:
        depths = (2, 3, 4)
        leaves = (20, 40, 80)
        l2s = (1.0, 5.0, 10.0, 20.0)
        lrs = (0.05, 0.08)
        iters = (200,)
    else:
        depths = (2, 3, 4)
        leaves = (20, 40, 60, 80)
        l2s = (1.0, 3.0, 5.0, 10.0, 15.0, 20.0)
        lrs = (0.04, 0.05, 0.08)
        iters = (150, 200, 300)
    out: List[GbmHyperParams] = []
    for depth, leaf, l2, lr, n_iter in itertools.product(depths, leaves, l2s, lrs, iters):
        out.append(
            GbmHyperParams(
                max_depth=depth,
                min_samples_leaf=leaf,
                l2_regularization=l2,
                learning_rate=lr,
                max_iter=n_iter,
            )
        )
    baseline = GbmHyperParams.from_dict(DEFAULT_GBM_HYPERPARAMS)
    if baseline not in out:
        out.insert(0, baseline)
    return out


def _gate_pnl_for_model(
    model,
    *,
    dates: Sequence[str],
    symbols: List[str],
    gate: LiveGateConfig,
    robot_wallets: List[float],
) -> Dict[str, Any]:
    profiles = load_profiles()
    ranker = BreakoutRanker(gbm=model, profiles=profiles, use_prior=True)
    days = run_gate_eval_sessions(
        dates=list(dates),
        symbols=symbols,
        gate=gate,
        ranker=ranker,
        respect_env_filters=True,
        robot_wallets=robot_wallets,
    )
    total = sum(
        float(o.get("pnl_usdt") or 0.0)
        for d in days
        for o in (d.get("opened") or [])
    )
    day_pnls = [
        sum(float(o.get("pnl_usdt") or 0.0) for o in (d.get("opened") or []))
        for d in days
    ]
    win = sum(1 for p in day_pnls if p > 0)
    loss = sum(1 for p in day_pnls if p < 0)
    opens = sum(int(d.get("opens") or 0) for d in days)
    return {
        "sessions": len(days),
        "total_pnl_usdt": round(total, 1),
        "avg_daily_pnl_usdt": round(total / max(len(days), 1), 1),
        "win_days": win,
        "loss_days": loss,
        "avg_opens": round(opens / max(len(days), 1), 1),
    }


def run_sweep(
    rows: List[Dict[str, Any]],
    *,
    fold_days: int,
    n_folds: int,
    quick: bool,
) -> List[Dict[str, Any]]:
    folds = _walk_forward_folds(rows, fold_days=fold_days, n_folds=n_folds)
    if not folds:
        print("No walk-forward folds — check sample dates.")
        return []

    grid = _param_grid(quick=quick)
    total = len(LABEL_MODES) * len(grid)
    print(f"=== GBM sweep: {len(LABEL_MODES)} labels × {len(grid)} params × {len(folds)} folds = {total * len(folds)} fits ===")

    results: List[Dict[str, Any]] = []
    done = 0
    for label_mode in LABEL_MODES:
        if label_mode == "hold_30m" and not any("hold30_true" in r for r in rows):
            print("skip hold_30m — missing hold30_true")
            continue
        for hp in grid:
            fold_metrics = []
            for train_rows, test_rows in folds:
                fold_metrics.append(_eval_one(train_rows, test_rows, label_mode=label_mode, hp=hp))
            agg = _aggregate_fold_metrics(fold_metrics)
            if not agg:
                continue
            results.append(
                {
                    "label_mode": label_mode,
                    "hyperparams": hp.to_dict(),
                    "fold_metrics": fold_metrics,
                    **agg,
                }
            )
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  progress {done}/{total}", flush=True)

    results.sort(
        key=lambda r: (
            float(r.get("mean_holdout_separation") or 0.0),
            float(r.get("mean_score") or 0.0),
            -float((r.get("fold_metrics") or [{}])[0].get("overfit_gap") or 0.0),
        ),
        reverse=True,
    )
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep ORB breakout GBM hyperparams + label modes")
    ap.add_argument("--samples", default=str(default_shared_samples_path()))
    ap.add_argument("--fold-days", type=int, default=7, help="每个 walk-forward fold 的 holdout 天数")
    ap.add_argument("--folds", type=int, default=3, help="walk-forward fold 数")
    ap.add_argument("--quick", action="store_true", help="较小网格（默认 full grid）")
    ap.add_argument("--top", type=int, default=5, help="Gate PnL 评估的 Top 候选数")
    ap.add_argument("--gate-sessions", type=int, default=30, help="Gate PnL 回测最近 N 个交易日")
    ap.add_argument("--gate-eval", action="store_true", help="对 Top 候选跑 Gate PnL（较慢）")
    ap.add_argument("--gate-eval-only", action="store_true", help="跳过 sweep，仅对已有 json 做 Gate PnL")
    ap.add_argument("--json-out", default=str(V2_GBM_SWEEP))
    args = ap.parse_args()

    samples_path = Path(args.samples)
    if not samples_path.is_file():
        print(f"Missing samples: {samples_path}")
        return 1

    rows = list(json.loads(samples_path.read_text(encoding="utf-8")).get("rows") or [])
    if len(rows) < 100:
        print(f"Too few samples: {len(rows)}")
        return 1

    out_path = Path(args.json_out)
    if args.gate_eval_only and out_path.is_file():
        sweep = json.loads(out_path.read_text(encoding="utf-8"))
        results = sweep.get("top10") or [sweep.get("best")]
        baseline = sweep.get("baseline")
    else:
        results = run_sweep(rows, fold_days=max(1, args.fold_days), n_folds=max(1, args.folds), quick=bool(args.quick))
        if not results:
            return 1
        baseline = next(
            (
                r
                for r in results
                if r.get("label_mode") == "hold_30m"
                and r.get("hyperparams") == GbmHyperParams.from_dict(DEFAULT_GBM_HYPERPARAMS).to_dict()
            ),
            None,
        )

        print("\n=== Top 10（mean holdout separation）===")
        print(f"{'label':<14} {'sep':>6} {'acc':>6} {'score':>6}  hyperparams")
        for r in results[:10]:
            hp = r["hyperparams"]
            hp_s = f"d={hp['max_depth']} leaf={hp['min_samples_leaf']} l2={hp['l2_regularization']} lr={hp['learning_rate']}"
            print(
                f"{r['label_mode']:<14} {r['mean_holdout_separation']:6.4f} "
                f"{r['mean_holdout_accuracy']:6.4f} {r['mean_score']:6.4f}  {hp_s}"
            )

        if baseline:
            print(
                f"\nbaseline hold_30m sep={baseline['mean_holdout_separation']:.4f} "
                f"acc={baseline['mean_holdout_accuracy']:.4f}"
            )

    holdout_best = results[0]
    best = holdout_best
    gate_eval_rows: List[Dict[str, Any]] = []
    if args.gate_eval or args.gate_eval_only:
        from env_loader import load_env_oi

        load_env_oi()
        syms = cached_symbols(parse_symbol_list(resolve_symbols_path().read_text(encoding="utf-8")))
        gate = LiveGateConfig.from_json(resolve_gate_config_path())
        dates = trading_dates_from_samples(last_sessions=int(args.gate_sessions))
        rc = robot_count_from_env()
        re = robot_equity_from_env()
        robot_wallets = init_robot_wallets(count=rc, equity_usdt=re) if rc > 0 else None
        print(f"\n=== Gate PnL eval Top {args.top} / {len(dates)} sessions ===")
        eval_candidates: List[Dict[str, Any]] = list(results[: max(1, args.top)])
        if baseline and baseline not in eval_candidates:
            eval_candidates.append(baseline)
        for r in eval_candidates:
            from orb.ml.gbm import rows_to_xy_gbm

            hp = GbmHyperParams.from_dict(r["hyperparams"])
            train_rows, _ = split_holdout_by_date(rows, holdout_days=max(1, args.fold_days))
            X, y = rows_to_xy_gbm(train_rows, label_mode=str(r["label_mode"]))
            model = train_gbm(X, y, label_mode=str(r["label_mode"]), hyperparams=hp)
            pnl = _gate_pnl_for_model(
                model,
                dates=dates,
                symbols=syms,
                gate=gate,
                robot_wallets=robot_wallets or [],
            )
            row = {
                "label_mode": r["label_mode"],
                "hyperparams": r["hyperparams"],
                "mean_holdout_separation": r["mean_holdout_separation"],
                **pnl,
            }
            gate_eval_rows.append(row)
            print(
                f"{r['label_mode']:<14} sep={r['mean_holdout_separation']:.4f} "
                f"pnl={pnl['total_pnl_usdt']:+.1f}U avg/d={pnl['avg_daily_pnl_usdt']:+.1f}U "
                f"W/L={pnl['win_days']}/{pnl['loss_days']}"
            )
        gate_eval_rows.sort(key=lambda x: float(x.get("total_pnl_usdt") or 0.0), reverse=True)
        if gate_eval_rows:
            best = gate_eval_rows[0]

    out = {
        "samples": len(rows),
        "fold_days": int(args.fold_days),
        "folds": int(args.folds),
        "baseline": baseline,
        "best_holdout": holdout_best,
        "best": best,
        "top10": results[:10],
        "gate_eval": gate_eval_rows,
        "all_count": len(results),
    }
    ensure_v2_dirs()
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten -> {out_path}")
    print(
        f"best: label={best['label_mode']} sep={best['mean_holdout_separation']:.4f} "
        f"params={best['hyperparams']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""共享 ORB 突破 GBM 排序模型（sklearn HistGradientBoosting）。"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from orb.ml.features import RANK_FEATURE_NAMES, rank_features

from orb.ml.model.paths import GBM_META, GBM_PKL

DEFAULT_GBM_PATH = GBM_PKL
DEFAULT_GBM_META = GBM_META

DEFAULT_GBM_HYPERPARAMS: Dict[str, Any] = {
    "max_depth": 4,
    "learning_rate": 0.08,
    "max_iter": 200,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": 42,
}


@dataclass(frozen=True)
class GbmHyperParams:
    max_depth: int = 4
    learning_rate: float = 0.08
    max_iter: int = 200
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    random_state: int = 42

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "GbmHyperParams":
        base = dict(DEFAULT_GBM_HYPERPARAMS)
        if d:
            base.update({k: v for k, v in d.items() if k in base})
        return cls(**base)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "random_state": self.random_state,
        }


@dataclass
class BreakoutGBM:
    target: str = "true"
    feature_names: Tuple[str, ...] = RANK_FEATURE_NAMES
    label_mode: str = "hold_30m"
    model: HistGradientBoostingClassifier = field(default_factory=HistGradientBoostingClassifier)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, feat: Dict[str, float], *, symbol: str = "", rank_only: bool = True) -> float:
        x = np.array([[float(rank_features(feat).get(k, 0.0) or 0.0) for k in self.feature_names]], dtype=np.float64)
        p = float(self.model.predict_proba(x)[0, 1])
        return max(0.0, min(1.0, p))

    def to_meta(self) -> Dict[str, Any]:
        return {
            "version": 3,
            "kind": "gbm",
            "target": self.target,
            "feature_names": list(self.feature_names),
            "label_mode": self.label_mode,
            "metrics": dict(self.metrics),
        }


def rows_to_xy_gbm(
    rows: List[Dict[str, Any]],
    *,
    label_mode: str = "hold_30m",
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.vstack(
        [
            [float(rank_features({k.replace('f_', '', 1): v for k, v in r.items() if str(k).startswith('f_')}).get(k, 0.0) or 0.0) for k in RANK_FEATURE_NAMES]
            for r in rows
        ]
    )
    y = np.array([_row_label(r, label_mode=label_mode) for r in rows], dtype=np.int32)
    return X, y


def _row_label(r: Dict[str, Any], *, label_mode: str) -> int:
    if label_mode == "hold_30m":
        if "hold30_true" in r:
            return int(r.get("hold30_true") or 0)
        return int(r.get("true_breakout") or 0)
    if label_mode == "quality":
        from orb.ml.features import label_quality

        return label_quality(str(r.get("outcome") or ""), float(r.get("pnl_usdt") or 0), float(r.get("pnl_r") or 0))
    return int(r.get("true_breakout") or 0)


def train_gbm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    label_mode: str = "hold_30m",
    hyperparams: Optional[GbmHyperParams | Dict[str, Any]] = None,
) -> BreakoutGBM:
    hp = hyperparams if isinstance(hyperparams, GbmHyperParams) else GbmHyperParams.from_dict(hyperparams)
    clf = HistGradientBoostingClassifier(**hp.to_dict())
    clf.fit(X, y)
    p_hat = clf.predict_proba(X)[:, 1]
    pred = p_hat >= 0.5
    actual = y >= 1
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    n = len(y)
    return BreakoutGBM(
        model=clf,
        label_mode=label_mode,
        metrics={
            "samples": n,
            "positive_rate": round(float(y.mean()), 4),
            "train_accuracy": round((tp + tn) / max(n, 1), 4),
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
            "brier": round(float(np.mean((p_hat - y) ** 2)), 4),
        },
    )


def score_gbm_holdout(model: BreakoutGBM, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    X, y = rows_to_xy_gbm(rows, label_mode=model.label_mode)
    p = model.model.predict_proba(X)[:, 1]
    pred = p >= 0.5
    hits = int((pred == (y >= 1)).sum())
    pos = p[y >= 1]
    neg = p[y < 1]
    return {
        "holdout_accuracy": round(hits / len(rows), 4),
        "holdout_n": len(rows),
        "holdout_p_mean_pos": round(float(pos.mean()), 4) if len(pos) else 0.0,
        "holdout_p_mean_neg": round(float(neg.mean()), 4) if len(neg) else 0.0,
        "holdout_separation": round(float(pos.mean() - neg.mean()), 4) if len(pos) and len(neg) else 0.0,
    }


def save_gbm(model: BreakoutGBM, path: Optional[Path] = None, meta_path: Optional[Path] = None) -> None:
    pkl = path or DEFAULT_GBM_PATH
    meta = meta_path or DEFAULT_GBM_META
    pkl.parent.mkdir(parents=True, exist_ok=True)
    with pkl.open("wb") as f:
        pickle.dump(model, f)
    meta.write_text(json.dumps(model.to_meta(), indent=2), encoding="utf-8")


def load_gbm(path: Optional[Path] = None) -> Optional[BreakoutGBM]:
    if path is None:
        from orb.ml.model.paths import resolve_gbm_path

        pkl = resolve_gbm_path()
    else:
        pkl = path
    if not pkl.is_file():
        return None
    try:
        with pkl.open("rb") as f:
            return pickle.load(f)
    except (pickle.PickleError, EOFError, AttributeError, TypeError):
        return None

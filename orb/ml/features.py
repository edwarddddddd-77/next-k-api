#!/usr/bin/env python3
"""ORB 突破概率：特征 + logistic 模型（含共享 symbol 编码）。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from orb.core.config import OrbConfig
from orb.core.session import session_anchor_ms
from orb.core.signals import OrbSignal

FEATURE_NAMES: Tuple[str, ...] = (
    "or_width_pct",
    "vol_ratio",
    "side_long",
    "vwap_dist_pct",
    "risk_frac_pct",
    "minutes_after_or",
    "gap_pct",
    "pm_rvol",
    "pm_regime_go",
    "pm_regime_fade",
    "atr_pct",
    "sync_same_side",
)

# 横截面排序用：不含 sync（sync 仅作日级风险，不参与 symbol 排序）
RANK_FEATURE_NAMES: Tuple[str, ...] = tuple(k for k in FEATURE_NAMES if k != "sync_same_side")

LABEL_MODES = ("eod", "no_sl", "quality")

from orb.ml.paths import (
    default_shared_fake_model_path,
    default_shared_samples_path,
    default_shared_true_model_path,
)


def _parse_reason_float(reasons: Sequence[str], prefix: str) -> float:
    for raw in reasons:
        r = str(raw).strip()
        if not r.startswith(prefix):
            continue
        try:
            return float(r[len(prefix) :].strip().rstrip("%"))
        except ValueError:
            continue
    return 0.0


def _parse_pm_regime(reasons: Sequence[str]) -> Tuple[float, float]:
    for raw in reasons:
        r = str(raw).strip()
        if not r.startswith("pm_regime="):
            continue
        regime = r.split("=", 1)[1].strip().lower()
        return (1.0 if regime == "gap_and_go" else 0.0, 1.0 if regime == "gap_and_fade" else 0.0)
    return 0.0, 0.0


def minutes_after_or(sig: OrbSignal, cfg: OrbConfig) -> float:
    bo = int(sig.entry_bar_open_ms or 0)
    if bo <= 0:
        return 0.0
    anchor = session_anchor_ms(bo, tz=cfg.session_tz, session_open_time=cfg.session_open_time)
    or_end = anchor + max(1, int(cfg.or_minutes)) * 60_000
    return max(0.0, (bo - or_end) / 60_000.0)


def extract_features(sig: OrbSignal, cfg: OrbConfig, *, sync_same_side: int = 0) -> Dict[str, float]:
    reasons = list(sig.reasons or [])
    vol = float(sig.volume or 0.0)
    vma = float(sig.vol_ma or 0.0)
    vol_ratio = vol / vma if vma > 0 else 1.0
    entry = float(sig.price or 0.0)
    sl = float(sig.sl_price or 0.0)
    risk_frac = abs(entry - sl) / entry * 100.0 if entry > 0 and sl > 0 else 0.0
    vwap = _parse_reason_float(reasons, "vwap=")
    vwap_dist = (entry - vwap) / vwap * 100.0 if vwap > 0 and entry > 0 else 0.0
    atr_raw = _parse_reason_float(reasons, "atr=")
    atr_pct = atr_raw / entry * 100.0 if entry > 0 and atr_raw > 0 else 0.0
    go, fade = _parse_pm_regime(reasons)
    return {
        "or_width_pct": float(sig.or_width_pct or 0.0),
        "vol_ratio": float(vol_ratio),
        "side_long": 1.0 if str(sig.side).upper() == "LONG" else 0.0,
        "vwap_dist_pct": float(vwap_dist),
        "risk_frac_pct": float(risk_frac),
        "minutes_after_or": float(minutes_after_or(sig, cfg)),
        "gap_pct": float(_parse_reason_float(reasons, "gap=")),
        "pm_rvol": float(_parse_reason_float(reasons, "pm_rvol=")),
        "pm_regime_go": float(go),
        "pm_regime_fade": float(fade),
        "atr_pct": float(atr_pct),
        "sync_same_side": float(max(0, int(sync_same_side))),
    }


def build_symbol_vocab(rows: Sequence[Dict[str, Any]]) -> List[str]:
    syms = sorted({str(r.get("symbol") or "").upper() for r in rows if r.get("symbol")})
    return syms[1:] if len(syms) > 1 else []


def feature_names_for(*, rank_only: bool = False) -> Tuple[str, ...]:
    return RANK_FEATURE_NAMES if rank_only else FEATURE_NAMES


def build_feature_vector(
    feat: Dict[str, float],
    *,
    symbol: str,
    vocab: Sequence[str],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> np.ndarray:
    base = np.array([float(feat.get(k, 0.0) or 0.0) for k in feature_names], dtype=np.float64)
    if not vocab:
        return base
    sym = str(symbol or "").upper()
    oh = np.array([1.0 if s == sym else 0.0 for s in vocab], dtype=np.float64)
    return np.concatenate([base, oh])


def rank_features(feat: Dict[str, float]) -> Dict[str, float]:
    return {k: float(feat.get(k, 0.0) or 0.0) for k in RANK_FEATURE_NAMES}


@dataclass
class BreakoutModel:
    target: str
    feature_names: Tuple[str, ...] = FEATURE_NAMES
    symbol_vocab: List[str] = field(default_factory=list)
    mean: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURE_NAMES)))
    std: np.ndarray = field(default_factory=lambda: np.ones(len(FEATURE_NAMES)))
    weights: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURE_NAMES)))
    bias: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, feat: Dict[str, float], *, symbol: str = "", rank_only: bool = False) -> float:
        names = RANK_FEATURE_NAMES if rank_only else self.feature_names
        use_feat = rank_features(feat) if rank_only else feat
        x = build_feature_vector(use_feat, symbol=symbol, vocab=self.symbol_vocab, feature_names=names)
        dim = len(names) + len(self.symbol_vocab)
        if len(x) != dim:
            pad = np.zeros(dim, dtype=np.float64)
            pad[: min(len(x), dim)] = x[: min(len(x), dim)]
            x = pad
        std = np.where(self.std > 1e-9, self.std, 1.0)
        z = (x - self.mean[: len(x)]) / std[: len(x)]
        logit = float(self.bias + np.dot(self.weights[: len(x)], z))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 2,
            "target": self.target,
            "feature_names": list(self.feature_names),
            "symbol_vocab": list(self.symbol_vocab),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BreakoutModel":
        feat = tuple(d.get("feature_names") or FEATURE_NAMES)
        vocab = list(d.get("symbol_vocab") or [])
        dim = len(feat) + len(vocab)

        def _pad(arr: np.ndarray) -> np.ndarray:
            return arr[:dim] if len(arr) >= dim else np.pad(arr, (0, dim - len(arr)))

        return cls(
            target=str(d.get("target") or "fake"),
            feature_names=feat,
            symbol_vocab=vocab,
            mean=_pad(np.array(d.get("mean") or [0.0] * dim, dtype=np.float64)),
            std=_pad(np.array(d.get("std") or [1.0] * dim, dtype=np.float64)),
            weights=_pad(np.array(d.get("weights") or [0.0] * dim, dtype=np.float64)),
            bias=float(d.get("bias") or 0.0),
            metrics=dict(d.get("metrics") or {}),
        )


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    target: str,
    symbol_vocab: Sequence[str],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> BreakoutModel:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    Z = (X - mean) / std
    n, d = Z.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    lr = 0.05
    for _ in range(800):
        z = b + Z @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = p - y
        w -= lr * ((Z.T @ err) / max(n, 1) + 0.01 * w)
        b -= lr * float(err.mean())
    p_hat = 1.0 / (1.0 + np.exp(-np.clip(b + Z @ w, -30, 30)))
    pred = p_hat >= 0.5
    actual = y >= 0.5
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    return BreakoutModel(
        target=target,
        feature_names=tuple(feature_names),
        symbol_vocab=list(symbol_vocab),
        mean=mean,
        std=std,
        weights=w,
        bias=b,
        metrics={
            "target": target,
            "samples": int(n),
            "positive_rate": round(float(y.mean()), 4),
            "train_accuracy": round((tp + tn) / max(n, 1), 4),
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
            "brier": round(float(np.mean((p_hat - y) ** 2)), 4),
        },
    )


def save_model(model: BreakoutModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")


def load_model(path: Optional[Path] = None) -> Optional[BreakoutModel]:
    if path is None or not path.is_file():
        return None
    try:
        return BreakoutModel.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def label_is_fake(outcome: str, pnl_usdt: float) -> int:
    if str(outcome).strip().lower() == "loss" or float(pnl_usdt or 0.0) < 0:
        return 1
    return 0


def label_is_true_breakout(outcome: str, pnl_usdt: float) -> int:
    return 1 if float(pnl_usdt or 0.0) > 0 else 0


def label_no_sl(outcome: str, pnl_usdt: float, pnl_r: float = 0.0) -> int:
    if str(outcome).strip().lower() == "loss":
        return 0
    return 1 if float(pnl_usdt or 0.0) > 0 else 0


def label_quality(outcome: str, pnl_usdt: float, pnl_r: float = 0.0) -> int:
    if str(outcome).strip().lower() == "loss":
        return 0
    return 1 if float(pnl_r or 0.0) >= 0.5 else 0


def label_for_target(
    target: str,
    outcome: str,
    pnl_usdt: float,
    *,
    pnl_r: float = 0.0,
    label_mode: str = "eod",
) -> int:
    mode = str(label_mode or "eod").strip().lower()
    if mode == "no_sl":
        positive = label_no_sl(outcome, pnl_usdt, pnl_r)
    elif mode == "quality":
        positive = label_quality(outcome, pnl_usdt, pnl_r)
    else:
        positive = label_is_true_breakout(outcome, pnl_usdt)
    if target == "true":
        return positive
    return 0 if positive else 1


def rows_to_xy(
    rows: List[Dict[str, Any]],
    *,
    target: str = "fake",
    vocab: Sequence[str] = (),
    rank_only: bool = False,
    label_mode: str = "eod",
) -> Tuple[np.ndarray, np.ndarray]:
    names = feature_names_for(rank_only=rank_only)

    def _feat(r: Dict[str, Any]) -> Dict[str, float]:
        raw = {k.replace("f_", "", 1): v for k, v in r.items() if str(k).startswith("f_")}
        return rank_features(raw) if rank_only else raw

    X = np.vstack([build_feature_vector(_feat(r), symbol=str(r.get("symbol") or ""), vocab=vocab, feature_names=names) for r in rows])
    y = np.array(
        [
            label_for_target(
                target,
                str(r.get("outcome") or ""),
                float(r.get("pnl_usdt") or 0),
                pnl_r=float(r.get("pnl_r") or 0),
                label_mode=label_mode,
            )
            for r in rows
        ],
        dtype=np.float64,
    )
    return X, y


def score_rows(
    model: BreakoutModel,
    rows: List[Dict[str, Any]],
    *,
    label_mode: str = "eod",
    rank_only: bool = False,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        feat = {k.replace("f_", "", 1): v for k, v in r.items() if str(k).startswith("f_")}
        p = model.predict_proba(feat, symbol=str(r.get("symbol") or ""), rank_only=rank_only)
        actual = label_for_target(
            model.target,
            str(r.get("outcome") or ""),
            float(r.get("pnl_usdt") or 0),
            pnl_r=float(r.get("pnl_r") or 0),
            label_mode=label_mode,
        )
        out.append(
            {
                "session_date": r.get("session_date"),
                "symbol": r.get("symbol"),
                "side": r.get("side"),
                "outcome": r.get("outcome"),
                "pnl_usdt": r.get("pnl_usdt"),
                "p": round(p, 4),
                "actual": actual,
                "sync": r.get("f_sync_same_side"),
            }
        )
    return out

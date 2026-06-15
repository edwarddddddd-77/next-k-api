#!/usr/bin/env python3
"""统一突破排序：GBM + 标的先验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from orb.ml.gbm import BreakoutGBM, load_gbm
from orb.ml.features import BreakoutModel, load_model, default_shared_true_model_path
from orb.ml.profiles import apply_symbol_prior, load_profiles


@dataclass
class BreakoutRanker:
    gbm: Optional[BreakoutGBM] = None
    logistic: Optional[BreakoutModel] = None
    profiles: Optional[Dict[str, Any]] = None
    use_prior: bool = True
    prior_model_weight: float = 0.75

    @classmethod
    def load(
        cls,
        *,
        use_prior: bool = True,
        gbm_path: Optional[Path] = None,
        profiles_path: Optional[Path] = None,
        logistic_path: Optional[Path] = None,
    ) -> "BreakoutRanker":
        gbm = load_gbm(gbm_path)
        logistic = None
        if gbm is None:
            logistic = load_model(logistic_path or default_shared_true_model_path())
        profiles = load_profiles(profiles_path) if use_prior else {"profiles": {}}
        return cls(gbm=gbm, logistic=logistic, profiles=profiles, use_prior=use_prior)

    @property
    def kind(self) -> str:
        return "gbm" if self.gbm else "logistic"

    def predict_true(self, feat: Dict[str, float], *, symbol: str = "") -> float:
        if self.gbm is not None:
            p = self.gbm.predict_proba(feat, symbol=symbol, rank_only=True)
        elif self.logistic is not None:
            p = self.logistic.predict_proba(feat, symbol=symbol, rank_only=True)
        else:
            return 0.5
        if self.use_prior and self.profiles:
            p = apply_symbol_prior(p, symbol, self.profiles, model_weight=self.prior_model_weight)
        return round(float(p), 4)

    def predict_fake(self, feat: Dict[str, float], *, symbol: str = "") -> float:
        return round(1.0 - self.predict_true(feat, symbol=symbol), 4)

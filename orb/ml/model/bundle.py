"""突破排序大模型统一加载与推理入口。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from orb.ml.ranker import BreakoutRanker
from orb.ml.model.paths import (
    GBM_META,
    MANIFEST_JSON,
    resolve_gbm_meta_path,
    resolve_gbm_path,
    resolve_logistic_true_path,
    resolve_profiles_path,
)


@dataclass
class BreakoutModelBundle:
    """GBM（主）+ 标的先验 + logistic 回退，Live/回测共用。"""

    ranker: BreakoutRanker
    gbm_path: Path
    profiles_path: Path
    meta: Dict[str, Any] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.ranker.kind

    @property
    def is_ready(self) -> bool:
        return self.ranker.gbm is not None or self.ranker.logistic is not None

    def predict_true(self, feat: Dict[str, float], *, symbol: str = "") -> float:
        return self.ranker.predict_true(feat, symbol=symbol)

    def predict_fake(self, feat: Dict[str, float], *, symbol: str = "") -> float:
        return self.ranker.predict_fake(feat, symbol=symbol)

    def status(self) -> Dict[str, Any]:
        prof_n = len((self.ranker.profiles or {}).get("profiles") or {})
        return {
            "ready": self.is_ready,
            "kind": self.kind,
            "gbm_path": str(self.gbm_path),
            "gbm_exists": self.gbm_path.is_file(),
            "profiles_path": str(self.profiles_path),
            "profiles_exists": self.profiles_path.is_file(),
            "profile_symbols": prof_n,
            "use_prior": bool(self.ranker.use_prior),
            "prior_model_weight": float(self.ranker.prior_model_weight),
            "label_mode": self.meta.get("label_mode"),
            "train_metrics": self.meta.get("metrics"),
            "manifest_updated": self.manifest.get("updated_at_utc"),
        }

    @classmethod
    def load(
        cls,
        *,
        gbm_path: Optional[Path] = None,
        profiles_path: Optional[Path] = None,
        logistic_path: Optional[Path] = None,
        use_prior: bool = True,
        prior_model_weight: float = 0.75,
    ) -> "BreakoutModelBundle":
        gbm_p = Path(gbm_path) if gbm_path else resolve_gbm_path()
        prof_p = Path(profiles_path) if profiles_path else resolve_profiles_path()
        log_p = Path(logistic_path) if logistic_path else resolve_logistic_true_path()
        ranker = BreakoutRanker.load(
            use_prior=use_prior,
            gbm_path=gbm_p,
            profiles_path=prof_p,
            logistic_path=log_p,
        )
        ranker.prior_model_weight = float(prior_model_weight)
        meta: Dict[str, Any] = {}
        meta_path = resolve_gbm_meta_path()
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        manifest: Dict[str, Any] = {}
        if MANIFEST_JSON.is_file():
            try:
                manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
        return cls(
            ranker=ranker,
            gbm_path=gbm_p,
            profiles_path=prof_p,
            meta=meta,
            manifest=manifest,
        )

    @classmethod
    def load_production(cls) -> "BreakoutModelBundle":
        """从 env 或 data/orb/live/（再回退 ml/models）加载 production 模型。"""
        gbm_raw = (os.getenv("ORB_V2_GBM_PATH") or "").strip()
        prof_raw = (os.getenv("ORB_V2_PROFILES_PATH") or "").strip()
        return cls.load(
            gbm_path=Path(gbm_raw) if gbm_raw else None,
            profiles_path=Path(prof_raw) if prof_raw else None,
        )

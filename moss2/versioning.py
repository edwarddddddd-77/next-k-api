"""参数版本：hash、发布、回滚。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Tuple

from moss2.params import merge_profile_params


def params_hash(profile: dict) -> str:
    merged = merge_profile_params(profile)
    blob = json.dumps(merged, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def effective_version(profile: dict) -> str:
    return str(profile.get("approved_params_version") or profile.get("params_version") or "v1")


def canary_scale(profile: dict) -> float:
    return max(0.05, min(1.0, float(profile.get("canary_scale") or 1.0)))


def apply_candidate_to_profile_fields(
    *,
    template: str,
    initial_params: dict,
    tactical_params: dict,
    version_label: str,
) -> Dict[str, Any]:
    return {
        "template": template,
        "initial_params": initial_params,
        "tactical_params": tactical_params,
        "params_version": version_label,
        "params_hash": None,
    }

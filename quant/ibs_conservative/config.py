"""IBS 保守 lane 配置 — entry 0.20 / exit 0.50 / SMA200。"""

from __future__ import annotations

from quant.ibs.config import IbsLaneConfig
from quant.ibs.profile import PROFILE_CONSERVATIVE
from quant.ibs_conservative.paths import resolve_ibs_conservative_symbols_path
from quant.ibs_conservative.switches import IBS_CONSERVATIVE_SWITCH


class IbsConservativeConfig(IbsLaneConfig):
    lane: str = "ibs_conservative"
    profile: str = PROFILE_CONSERVATIVE

    @classmethod
    def from_env(cls) -> "IbsConservativeConfig":
        base = IbsLaneConfig.from_env(
            lane="ibs_conservative",
            profile=PROFILE_CONSERVATIVE,
            switch=IBS_CONSERVATIVE_SWITCH,
            resolve_symbols_path=resolve_ibs_conservative_symbols_path,
            env_prefix="IBS_CONSERVATIVE",
        )
        return cls(**base.__dict__)

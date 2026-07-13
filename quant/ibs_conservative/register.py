"""IBS 保守 lane — vnpy 注册。"""

from __future__ import annotations

from quant.ibs.register import build_ibs_plugin
from quant.ibs_conservative.config import IbsConservativeConfig
from quant.ibs_conservative.switches import IBS_CONSERVATIVE_SWITCH

IBS_CONSERVATIVE_VNPY_PLUGIN = build_ibs_plugin(
    lane="ibs_conservative",
    profile="conservative",
    switch=IBS_CONSERVATIVE_SWITCH,
    load_config=IbsConservativeConfig.from_env,
    sync_prefix="ibs_c",
    name_prefix="ibs_c",
)

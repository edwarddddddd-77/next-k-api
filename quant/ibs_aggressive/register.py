"""IBS 激进 lane — vnpy 注册。"""

from __future__ import annotations

from quant.ibs.register import build_ibs_plugin
from quant.ibs_aggressive.config import IbsAggressiveConfig
from quant.ibs_aggressive.switches import IBS_AGGRESSIVE_SWITCH

IBS_AGGRESSIVE_VNPY_PLUGIN = build_ibs_plugin(
    lane="ibs_aggressive",
    profile="aggressive",
    switch=IBS_AGGRESSIVE_SWITCH,
    load_config=IbsAggressiveConfig.from_env,
    sync_prefix="ibs_a",
    name_prefix="ibs_a",
)

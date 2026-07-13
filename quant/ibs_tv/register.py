"""IBS TV lane — vnpy 注册。"""

from __future__ import annotations

from quant.ibs.config import IbsLaneConfig
from quant.ibs.register import build_ibs_plugin
from quant.ibs_tv.config import IbsTvConfig
from quant.ibs_tv.switches import IBS_TV_SWITCH

def _lane_config_for_symbol(cfg: IbsLaneConfig, symbol: str) -> IbsLaneConfig:
    tv_cfg = cfg if isinstance(cfg, IbsTvConfig) else IbsTvConfig.from_env()
    return tv_cfg.lane_config_for_symbol(symbol)


def _register_tv(cta_engine, cfg: IbsLaneConfig, wallet_cur) -> list[str]:
    from quant.ibs.register import register_ibs_vnpy_strategies

    return register_ibs_vnpy_strategies(
        cta_engine,
        cfg,
        wallet_cur,
        name_prefix="ibs_tv",
        lane_config_for_symbol=_lane_config_for_symbol,
    )


IBS_TV_VNPY_PLUGIN = build_ibs_plugin(
    lane="ibs_tv",
    profile="tv",
    switch=IBS_TV_SWITCH,
    load_config=IbsTvConfig.from_env,
    sync_prefix="ibs_tv",
    name_prefix="ibs_tv",
    register=_register_tv,
)

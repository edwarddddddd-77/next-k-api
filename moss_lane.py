"""Moss 实盘槽：Protocol 同一时间只接一个 Moss（默认 moss2 + 币安 EN，改 lane 请改本文件）。"""

from __future__ import annotations

from typing import Literal

MossLane = Literal["moss_quant", "moss2"]
MOSS_LANES: tuple[MossLane, ...] = ("moss_quant", "moss2")
# 固化默认：Moss2 占 Protocol 槽（引擎见 moss2/config MOSS2_OPS_VARIANT=en / 币安）
# 切回 Moss1 实盘请改为 "moss_quant"
DEFAULT_MOSS_ACTIVE_LANE: MossLane = "moss2"


def active_moss_lane() -> MossLane:
    return DEFAULT_MOSS_ACTIVE_LANE


def lane_allows_moss_quant() -> bool:
    return active_moss_lane() == "moss_quant"


def lane_allows_moss2() -> bool:
    return active_moss_lane() == "moss2"


def moss_lane_snapshot() -> dict:
    lane = active_moss_lane()
    return {
        "active_lane": lane,
        "moss_quant_protocol": lane == "moss_quant",
        "moss2_protocol": lane == "moss2",
    }

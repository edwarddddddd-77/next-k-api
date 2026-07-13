"""IBS TV lane 路径。"""

from __future__ import annotations

from pathlib import Path

from quant.common.paths import PROJECT_ROOT


def resolve_ibs_tv_symbols_path() -> Path:
    return PROJECT_ROOT / "config" / "ibs_tv" / "symbols.txt"

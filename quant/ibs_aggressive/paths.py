"""IBS 激进 lane 路径。"""

from __future__ import annotations

from pathlib import Path

from quant.common.paths import PROJECT_ROOT


def resolve_ibs_aggressive_symbols_path() -> Path:
    return PROJECT_ROOT / "config" / "ibs_aggressive" / "symbols.txt"

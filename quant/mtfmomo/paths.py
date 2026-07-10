"""MtfMomo 路径。"""

from __future__ import annotations

from pathlib import Path

from quant.common.paths import PROJECT_ROOT


def resolve_mtfmomo_symbols_path() -> Path:
    return PROJECT_ROOT / "config" / "mtfmomo" / "symbols.txt"

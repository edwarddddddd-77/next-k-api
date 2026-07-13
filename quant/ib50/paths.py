"""IB50 路径。"""

from __future__ import annotations

from pathlib import Path

from quant.common.paths import PROJECT_ROOT


def resolve_ib50_symbols_path() -> Path:
    return PROJECT_ROOT / "config" / "ib50" / "symbols.txt"

"""Smart Breakout 路径。"""

from __future__ import annotations

from pathlib import Path

from quant.common.paths import PROJECT_ROOT


def resolve_squeeze_breakout_symbols_path() -> Path:
    return PROJECT_ROOT / "config" / "squeeze_breakout" / "symbols.txt"

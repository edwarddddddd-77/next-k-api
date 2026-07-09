"""Trading ORB 路径。"""

from __future__ import annotations

from pathlib import Path

from quant.common.paths import PROJECT_ROOT


def resolve_orb_vnpy_symbols_path() -> Path:
    return PROJECT_ROOT / "config" / "trading_orb" / "symbols.txt"

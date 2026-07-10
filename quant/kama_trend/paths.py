"""KAMA Trend 路径。"""

from __future__ import annotations

from pathlib import Path

from quant.common.paths import PROJECT_ROOT


def resolve_kama_trend_symbols_path() -> Path:
    return PROJECT_ROOT / "config" / "kama_trend" / "symbols.txt"

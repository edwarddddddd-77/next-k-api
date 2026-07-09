from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_aberration_symbols_path() -> Path:
    return ROOT / "config" / "aberration" / "symbols.txt"

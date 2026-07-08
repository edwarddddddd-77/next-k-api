"""标的池路径（Trading ORB pool7）。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_symbols_path() -> Path:
    return ROOT / "config" / "trading_orb" / "symbols.txt"

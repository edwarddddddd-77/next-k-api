"""CLI wrapper for HL short-term high win-rate screen."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.hl_wr_screen import run_screen  # noqa: E402


def main() -> None:
    board = run_screen()
    print(json.dumps(board, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

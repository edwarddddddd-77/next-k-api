"""tools/orb 脚本 launcher 模板。"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run_tool(relative: str) -> None:
    target = Path(__file__).resolve().parents[1] / relative
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")

"""将 vendor/vnpy 框架加入 import 路径。"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_vnpy_path() -> Path:
    api_root = Path(__file__).resolve().parents[2]
    vnpy_root = api_root / "vendor" / "vnpy"
    if vnpy_root.is_dir():
        root_s = str(vnpy_root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)
    return vnpy_root

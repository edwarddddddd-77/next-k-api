"""持久化数据路径（K 线缓存等）。"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# K 线：data/orb/kline/<SYMBOL>/5m.csv ...
KLINE_ROOT = Path(os.getenv("ORB_KLINE_CACHE_ROOT", "") or (PROJECT_ROOT / "data" / "orb" / "kline"))

# output/ 下非 K 线目录（迁移时跳过）
OUTPUT_LEGACY_ROOT = PROJECT_ROOT / "output"
KLINE_MIGRATION_SKIP_DIRS = frozenset(
    {
        "orb",
        "orb_v2",
        "orb_paper",
        "orb_1m_batch",
        ".cache",
    }
)


def ensure_kline_dirs() -> Path:
    KLINE_ROOT.mkdir(parents=True, exist_ok=True)
    return KLINE_ROOT

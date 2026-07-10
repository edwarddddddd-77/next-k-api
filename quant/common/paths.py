"""持久化数据路径（DATA_DIR、K 线缓存等）。"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Railway / Docker 常见持久卷挂载点（未设 DATA_DIR 时自动探测）
_RAILWAY_DATA_CANDIDATES = (
    Path("/app/data"),
    Path("/data"),
)

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


def resolve_data_dir() -> Path:
    """DATA_DIR 环境变量 > Railway 常见卷路径 > 项目根目录。"""
    raw = (os.getenv("DATA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser()
    for candidate in _RAILWAY_DATA_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return PROJECT_ROOT


def resolve_kline_cache_root() -> Path:
    """ORB_KLINE_CACHE_ROOT > {DATA_DIR}/orb/kline > 本地 data/orb/kline。"""
    raw = (os.getenv("ORB_KLINE_CACHE_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser()
    if (os.getenv("DATA_DIR") or "").strip():
        return resolve_data_dir() / "orb" / "kline"
    return PROJECT_ROOT / "data" / "orb" / "kline"


# 兼容旧 import；运行时路径以 resolve_* 为准
KLINE_ROOT = resolve_kline_cache_root()


def ensure_kline_dirs() -> Path:
    root = resolve_kline_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return root

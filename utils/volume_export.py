"""将 DATA_DIR 打包为 zip/tar.gz，供临时导出接口使用。"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

from quant.common.paths import resolve_data_dir

__all__ = (
    "cleanup_export_paths",
    "create_data_archive",
    "export_volume_enabled",
    "resolve_data_dir",
    "summarize_data_dir",
)


def export_volume_enabled() -> bool:
    return os.getenv("NEXT_K_EXPORT_VOLUME_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def summarize_data_dir(root: Path | None = None) -> dict:
    """返回目录体量摘要（不打包）。"""
    data_dir = root or resolve_data_dir()
    if not data_dir.is_dir():
        return {
            "data_dir": str(data_dir),
            "exists": False,
            "file_count": 0,
            "total_bytes": 0,
        }
    file_count = 0
    total_bytes = 0
    for dirpath, _dirnames, filenames in os.walk(data_dir):
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                total_bytes += fp.stat().st_size
            except OSError:
                continue
            file_count += 1
    return {
        "data_dir": str(data_dir),
        "exists": True,
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def create_data_archive(
    *,
    fmt: str = "zip",
    root: Path | None = None,
) -> tuple[Path, Path]:
    """
    在系统临时目录下打包 DATA_DIR。

    Returns:
        (archive_path, work_dir) — 调用方在响应结束后应删除二者。
    """
    data_dir = root or resolve_data_dir()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    fmt_norm = (fmt or "zip").strip().lower()
    if fmt_norm in ("zip", ".zip"):
        archive_fmt = "zip"
    elif fmt_norm in ("tar.gz", "tgz", "gztar", ".tar.gz"):
        archive_fmt = "gztar"
    else:
        raise ValueError(f"unsupported archive format: {fmt}")

    work_dir = Path(tempfile.mkdtemp(prefix="nextk-export-"))
    stamp = data_dir.name or "data"
    base = work_dir / f"next-k-{stamp}"
    logger.info(
        "Creating %s archive for DATA_DIR=%s (this may take several minutes)",
        archive_fmt,
        data_dir,
    )
    try:
        archive_path_str = shutil.make_archive(
            str(base),
            archive_fmt,
            root_dir=str(data_dir),
        )
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    archive_path = Path(archive_path_str)
    if not archive_path.is_file():
        shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(f"archive not created: {archive_path}")

    logger.info(
        "Archive ready: %s (%s bytes)",
        archive_path,
        archive_path.stat().st_size,
    )
    return archive_path, work_dir


def cleanup_export_paths(archive_path: Path, work_dir: Path) -> None:
    try:
        if archive_path.is_file():
            archive_path.unlink()
    except OSError as e:
        logger.warning("Failed to remove archive %s: %s", archive_path, e)
    try:
        if work_dir.is_dir():
            shutil.rmtree(work_dir, ignore_errors=True)
    except OSError as e:
        logger.warning("Failed to remove work dir %s: %s", work_dir, e)

"""临时维护路由：DATA_DIR 卷导出（用后请关闭 NEXT_K_EXPORT_VOLUME_ENABLED）。"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from utils.volume_export import (
    cleanup_export_paths,
    create_data_archive,
    export_volume_enabled,
    resolve_data_dir,
    summarize_data_dir,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["maintenance"])


def _ensure_export_enabled() -> None:
    if not export_volume_enabled():
        raise HTTPException(
            status_code=403,
            detail="export_volume_disabled_set_NEXT_K_EXPORT_VOLUME_ENABLED=1",
        )


@router.get("/api/export-volume/info")
async def export_volume_info() -> dict:
    """打包前查看 DATA_DIR 路径与体量（不下载）。"""
    _ensure_export_enabled()
    summary = summarize_data_dir()
    summary["export_enabled"] = True
    summary["formats"] = ["zip", "tar.gz"]
    return summary


@router.get("/export-volume")
@router.get("/api/export-volume")
async def export_volume_download(
    fmt: str = Query("zip", description="zip 或 tar.gz"),
) -> FileResponse:
    """
    将 DATA_DIR（Railway Volume，如 /data 或 /app/data）打包下载。

    部署前设置 NEXT_K_EXPORT_VOLUME_ENABLED=1。
    下载完成后请关闭 NEXT_K_EXPORT_VOLUME_ENABLED 并重新部署。
    """
    _ensure_export_enabled()
    data_dir = resolve_data_dir()
    if not data_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"data_dir_not_found:{data_dir}",
        )

    try:
        archive_path, work_dir = await asyncio.to_thread(
            create_data_archive, fmt=fmt, root=data_dir
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except OSError as e:
        logger.exception("export-volume archive failed")
        raise HTTPException(
            status_code=507,
            detail=f"archive_failed:{e}",
        ) from e

    filename = archive_path.name
    media = (
        "application/gzip"
        if filename.endswith(".tar.gz")
        else "application/zip"
    )

    def _cleanup() -> None:
        cleanup_export_paths(archive_path, work_dir)

    return FileResponse(
        path=str(archive_path),
        media_type=media,
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
        background=BackgroundTask(_cleanup),
    )

"""FastAPI 依赖：维护类路由（鉴权已禁用）。"""

from __future__ import annotations

from utils.maintenance_token import (
    MaintenanceAuthError,
    maintenance_token_configured,
    verify_maintenance_token,
)

__all__ = [
    "MaintenanceAuthError",
    "maintenance_token_configured",
    "require_maintenance_token",
    "verify_maintenance_token",
]


async def require_maintenance_token() -> None:
    return None

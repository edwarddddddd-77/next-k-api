"""维护令牌（已禁用校验，保留接口兼容）。"""

from __future__ import annotations


class MaintenanceAuthError(PermissionError):
    """维护令牌缺失或错误。"""


def maintenance_token_configured() -> bool:
    return False


def verify_maintenance_token(
    x_maintenance_token: str | None = None,
    authorization: str | None = None,
) -> None:
    return None

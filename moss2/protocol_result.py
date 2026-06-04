"""Protocol ingest 响应解析（与 moss_quant.paper_scanner 同语义）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ProtocolActionResult:
    ok: bool
    error: str = ""
    position_id: Optional[int] = None
    client_ref: str = ""
    entry_price: Optional[float] = None


def protocol_ingest_action_result(
    resp: Any,
    *,
    fallback_error: str,
) -> ProtocolActionResult:
    if not isinstance(resp, dict):
        return ProtocolActionResult(ok=False, error="invalid_protocol_response")
    for detail in resp.get("details") or []:
        if isinstance(detail, dict) and detail.get("action") == "traded":
            pid = detail.get("position_id")
            entry_raw = detail.get("entry_price")
            entry_price = float(entry_raw) if entry_raw is not None else None
            return ProtocolActionResult(
                ok=True,
                position_id=int(pid) if pid is not None else None,
                client_ref=str(
                    detail.get("client_ref")
                    or detail.get("api_signal_id")
                    or ""
                ),
                entry_price=entry_price,
            )
    first = details[0] if (details := resp.get("details") or []) and isinstance(
        details[0], dict
    ) else {}
    error = (
        resp.get("error")
        or first.get("error")
        or first.get("reason")
        or first.get("action")
        or fallback_error
    )
    return ProtocolActionResult(ok=False, error=str(error))


def protocol_ingest_open_result(resp: Any) -> ProtocolActionResult:
    return protocol_ingest_action_result(
        resp, fallback_error="protocol_open_not_traded"
    )


def protocol_ingest_close_result(resp: Any) -> ProtocolActionResult:
    return protocol_ingest_action_result(
        resp, fallback_error="protocol_close_not_traded"
    )

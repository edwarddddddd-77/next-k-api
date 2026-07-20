"""Reverse-proxy clawby-quant (openclawby/clawby-quant) into Next K."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from utils.clawby_quant_runtime import clawby_base_url, status as runtime_status

logger = logging.getLogger(__name__)

router = APIRouter(tags=["clawby-quant"])

_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


@router.get("/api/clawby-quant/health")
async def clawby_health():
    """Next K + sidecar status (does not require clawby up)."""
    st = runtime_status()
    upstream = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            r = await client.get(f"{clawby_base_url()}/health")
            upstream = r.json() if r.status_code == 200 else {"ok": False, "status": r.status_code}
    except Exception as e:
        upstream = {"ok": False, "error": str(e)}
    return {"ok": True, "runtime": st, "upstream": upstream}


async def _proxy(request: Request, upstream_path: str) -> Response:
    base = clawby_base_url()
    url = f"{base}{upstream_path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }
    body = await request.body()

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=False) as client:
            upstream = await client.request(
                request.method,
                url,
                headers=headers,
                content=body,
            )
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=f"clawby_quant_unreachable:{clawby_base_url()} ({e})",
        ) from e
    except Exception as e:
        logger.warning("clawby proxy failed %s %s: %s", request.method, url, e)
        raise HTTPException(status_code=502, detail=f"clawby_proxy_failed:{e}") from e

    out_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }
    media = upstream.headers.get("content-type")
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=out_headers,
        media_type=media,
    )


@router.api_route(
    "/api/clawby-quant/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy_api(path: str, request: Request):
    # health handled above when path empty? path is required — health is separate
    return await _proxy(request, f"/api/{path}")


@router.api_route(
    "/clawby-ui",
    methods=["GET"],
)
async def clawby_ui_root(request: Request):
    return await _proxy(request, "/")


@router.api_route(
    "/clawby-ui/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_ui(path: str, request: Request):
    return await _proxy(request, f"/{path}")

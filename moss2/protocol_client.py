"""Moss2 → Protocol（source=moss2，与 moss_quant 隔离）。"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

SOURCE = "moss2"


@dataclass
class Moss2ProtocolClient:
    base_url: str
    token: str = ""
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "Moss2ProtocolClient":
        return cls(
            base_url=os.getenv("PROTOCOL_API_URL", "").strip().rstrip("/"),
            token=os.getenv("PROTOCOL_MAINTENANCE_TOKEN", "").strip(),
        )

    def enabled(self) -> bool:
        return bool(self.base_url)

    def headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Maintenance-Token"] = self.token
        return headers

    @staticmethod
    def _raise_protocol_error(resp: httpx.Response) -> None:
        try:
            payload = resp.json()
        except Exception:
            payload = None
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if isinstance(payload, dict):
                parts = [
                    str(v).strip()
                    for v in (
                        payload.get("detail"),
                        payload.get("code"),
                        payload.get("msg"),
                    )
                    if str(v).strip()
                ]
                if parts:
                    raise RuntimeError(" | ".join(parts)) from exc
            raise

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        resp = httpx.post(
            f"{self.base_url}{path}",
            json=body,
            headers=self.headers(),
            timeout=self.timeout,
        )
        self._raise_protocol_error(resp)
        return resp.json()

    def send_open(
        self,
        *,
        symbol: str,
        side: str,
        entry_price: Optional[float],
        margin_usdt: float,
        leverage: float,
        profile_id: int,
        params_version: str = "v1",
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        play: str = "",
        composite: float = 0.0,
        regime: str = "",
    ) -> Dict[str, Any]:
        client_ref = (
            f"m2:{profile_id}:{params_version}:open:{int(time.time() * 1000)}"
        )
        signal: Dict[str, Any] = {
            "source": SOURCE,
            "api_signal_id": client_ref,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "margin_usdt": round(margin_usdt, 6),
            "leverage": round(leverage, 6),
            "play": play or f"moss2:{params_version}",
            "regime": regime,
            "composite": composite,
            "profile_id": profile_id,
            "client_ref": client_ref,
            "action": "open",
            "lane": "moss2",
            "params_version": params_version,
        }
        if sl_price is not None:
            signal["sl_price"] = sl_price
        if tp_price is not None:
            signal["tp_price"] = tp_price
        return self._post("/api/binance/signals/ingest", {"signals": [signal]})

    def send_close(
        self,
        *,
        symbol: str,
        side: str,
        exit_rule: str,
        close_price: float,
        profile_id: int,
        params_version: str = "v1",
    ) -> Dict[str, Any]:
        client_ref = (
            f"m2:{profile_id}:{params_version}:close:{int(time.time() * 1000)}"
        )
        signal = {
            "source": SOURCE,
            "api_signal_id": client_ref,
            "symbol": symbol,
            "side": side,
            "close_price": close_price,
            "profile_id": profile_id,
            "client_ref": client_ref,
            "action": "close",
            "play": exit_rule,
            "lane": "moss2",
            "params_version": params_version,
        }
        return self._post("/api/binance/signals/ingest", {"signals": [signal]})

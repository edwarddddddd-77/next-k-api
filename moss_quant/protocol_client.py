"""Protocol client for Moss Quant live trading source of truth."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

SOURCE = "moss_quant"


@dataclass
class ProtocolClient:
    base_url: str
    token: str = ""
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "ProtocolClient":
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

    def _get(self, path: str, **params: Any) -> Any:
        resp = httpx.get(
            f"{self.base_url}{path}",
            params={k: v for k, v in params.items() if v is not None},
            headers=self.headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        resp = httpx.post(
            f"{self.base_url}{path}",
            json=body,
            headers=self.headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _put(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        resp = httpx.put(
            f"{self.base_url}{path}",
            json=body,
            headers=self.headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_account_summary(self) -> Dict[str, Any]:
        return self._get("/api/binance/account/summary")

    def get_moss_positions(
        self, status: Optional[str] = None, limit: int = 500
    ) -> List[Dict[str, Any]]:
        return self._get(
            "/api/binance/positions",
            source=SOURCE,
            status=status,
            limit=limit,
        )

    def get_moss_leverage(self) -> int:
        summary = self.get_account_summary()
        return int((summary.get("moss_quant") or {}).get("leverage") or 0)

    def send_open(
        self,
        *,
        symbol: str,
        side: str,
        entry_price: float,
        sl_price: float,
        tp_price: Optional[float],
        notional: float,
        profile_id: int,
        play: str = "",
        composite: float = 0.0,
        regime: str = "",
        action: str = "open",
    ) -> Dict[str, Any]:
        client_ref = f"moss:{profile_id}:{action}:{int(time.time() * 1000)}"
        signal = {
            "source": SOURCE,
            "api_signal_id": client_ref,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "notional_usdt": round(notional, 2),
            "play": play,
            "regime": regime,
            "profile_id": profile_id,
            "client_ref": client_ref,
            "action": action,
        }
        return self._post("/api/binance/signals/ingest", {"signals": [signal]})

    def send_close(
        self,
        *,
        symbol: str,
        side: str,
        exit_rule: str,
        close_price: float,
        profile_id: int,
        position_id: int = 0,
    ) -> Dict[str, Any]:
        client_ref = f"moss:{profile_id}:close:{int(time.time() * 1000)}"
        body = {
            "source": SOURCE,
            "api_signal_id": client_ref,
            "symbol": symbol,
            "side": side,
            "exit_rule": exit_rule,
            "close_price": close_price,
            "profile_id": profile_id,
            "client_ref": client_ref,
        }
        if position_id:
            body["position_id"] = position_id
        return self._post("/api/binance/positions/close", body)

    def send_update_sl(
        self,
        *,
        position_id: int,
        new_sl_price: float,
        profile_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        profile_ref = str(profile_id) if profile_id is not None else "unknown"
        client_ref = f"moss:{profile_ref}:update_sl:{int(time.time() * 1000)}"
        body: Dict[str, Any] = {
            "new_sl_price": new_sl_price,
            "client_ref": client_ref,
        }
        if profile_id is not None:
            body["profile_id"] = profile_id
        return self._put(f"/api/binance/positions/{position_id}/sl", body)

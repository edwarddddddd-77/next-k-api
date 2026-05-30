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
                detail = payload.get("detail")
                code = payload.get("code")
                msg = payload.get("msg")
                parts = [str(v).strip() for v in (detail, code, msg) if str(v).strip()]
                if parts:
                    raise RuntimeError(" | ".join(parts)) from exc
            raise

    def _get(self, path: str, **params: Any) -> Any:
        resp = httpx.get(
            f"{self.base_url}{path}",
            params={k: v for k, v in params.items() if v is not None},
            headers=self.headers(),
            timeout=self.timeout,
        )
        self._raise_protocol_error(resp)
        return resp.json()

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        resp = httpx.post(
            f"{self.base_url}{path}",
            json=body,
            headers=self.headers(),
            timeout=self.timeout,
        )
        self._raise_protocol_error(resp)
        return resp.json()

    def _put(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        resp = httpx.put(
            f"{self.base_url}{path}",
            json=body,
            headers=self.headers(),
            timeout=self.timeout,
        )
        self._raise_protocol_error(resp)
        return resp.json()

    def get_account_summary(self) -> Dict[str, Any]:
        return self._get("/api/binance/account/summary")

    def get_trading_config(self) -> Dict[str, Any]:
        return self._get("/api/binance/config")

    def get_moss_positions(
        self, status: Optional[str] = None, limit: int = 500
    ) -> List[Dict[str, Any]]:
        return self._get(
            "/api/binance/positions",
            status=status,
            limit=limit,
        )

    def get_moss_leverage(self) -> int:
        cfg = self.get_trading_config()
        return int(cfg.get("leverage") or 0)

    def send_open(
        self,
        *,
        symbol: str,
        side: str,
        entry_price: Optional[float],
        sl_price: float,
        tp_price: Optional[float],
        margin_usdt: float,
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
            "margin_usdt": round(margin_usdt, 6),
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
    ) -> Dict[str, Any]:
        client_ref = f"moss:{profile_id}:close:{int(time.time() * 1000)}"
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
        }
        return self._post("/api/binance/signals/ingest", {"signals": [signal]})

    def send_update_sl(
        self,
        *,
        symbol: str,
        side: str,
        new_sl_price: float,
        profile_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        profile_ref = str(profile_id) if profile_id is not None else "unknown"
        client_ref = f"moss:{profile_ref}:update_sl:{int(time.time() * 1000)}"
        signal: Dict[str, Any] = {
            "source": SOURCE,
            "api_signal_id": client_ref,
            "symbol": symbol,
            "side": side,
            "sl_price": new_sl_price,
            "client_ref": client_ref,
            "action": "update_sl",
        }
        if profile_id is not None:
            signal["profile_id"] = profile_id
        return self._post("/api/binance/signals/ingest", {"signals": [signal]})

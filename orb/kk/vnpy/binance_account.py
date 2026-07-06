"""币安账户设置（KK vnpy 直连，与 Protocol 无关）。"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import requests

from binance_fapi import FAPI
from orb.core.kline_cache import norm_symbol
from orb.kk.config import KKConfig
from orb.kk.live_exec import _leverage as _kk_leverage

logger = logging.getLogger(__name__)


def _api_key() -> str:
    return (os.getenv("BINANCE_API_KEY") or "").strip()


def _api_secret() -> bytes:
    return (os.getenv("BINANCE_API_SECRET") or "").strip().encode()


def _signed_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    key = _api_key()
    secret = _api_secret()
    if not key or not secret:
        raise RuntimeError("BINANCE_API_KEY/SECRET not configured")
    payload = dict(params or {})
    payload["timestamp"] = int(time.time() * 1000)
    query = urllib.parse.urlencode(sorted(payload.items()))
    sig = hmac.new(secret, query.encode(), hashlib.sha256).hexdigest()
    url = f"{FAPI}{path}?{query}&signature={sig}"
    resp = requests.get(url, headers={"X-MBX-APIKEY": key}, timeout=15)
    if resp.status_code >= 400:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise RuntimeError(f"Binance {path} HTTP {resp.status_code}: {body}")
    return resp.json()


def _signed_post(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    key = _api_key()
    secret = _api_secret()
    if not key or not secret:
        raise RuntimeError("BINANCE_API_KEY/SECRET not configured")
    payload = dict(params)
    payload["timestamp"] = int(time.time() * 1000)
    query = urllib.parse.urlencode(sorted(payload.items()))
    sig = hmac.new(secret, query.encode(), hashlib.sha256).hexdigest()
    url = f"{FAPI}{path}?{query}&signature={sig}"
    resp = requests.post(url, headers={"X-MBX-APIKEY": key}, timeout=15)
    if resp.status_code >= 400:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise RuntimeError(f"Binance {path} HTTP {resp.status_code}: {body}")
    return resp.json()


def set_symbol_leverage(symbol: str, leverage: int) -> None:
    sym = norm_symbol(symbol)
    lev = max(1, int(leverage))
    try:
        _signed_post("/fapi/v1/leverage", {"symbol": sym, "leverage": lev})
        logger.info("[kk-vnpy] leverage %s -> %sx", sym, lev)
    except RuntimeError as exc:
        msg = str(exc)
        if "-4028" in msg:
            return
        raise


def set_symbol_margin_isolated(symbol: str) -> None:
    sym = norm_symbol(symbol)
    try:
        _signed_post("/fapi/v1/marginType", {"symbol": sym, "marginType": "ISOLATED"})
    except RuntimeError as exc:
        msg = str(exc)
        if "-4046" in msg or "-4067" in msg or "No need to change margin type" in msg:
            return
        raise


def ensure_one_way_mode() -> None:
    """账户设为单向持仓（vnpy 仅支持 one-way）。"""
    try:
        _signed_post("/fapi/v1/positionSide/dual", {"dualSidePosition": "false"})
        logger.info("[kk-vnpy] position mode -> one-way")
    except RuntimeError as exc:
        msg = str(exc)
        if "-4059" in msg or "No need to change" in msg:
            return
        raise


def fetch_position_amounts(symbols: List[str]) -> Dict[str, float]:
    """symbol -> 净持仓张数（多为正、空为负）。"""
    return {sym: float(snap.get("amount") or 0.0) for sym, snap in fetch_position_snapshots(symbols).items()}


def _opt_float(raw: Any) -> Optional[float]:
    if raw in (None, ""):
        return None
    return float(raw)


def list_all_open_positions(*, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """币安 U 本位当前非零持仓（实时 positionRisk）。"""
    want = {norm_symbol(s) for s in symbols} if symbols else None
    rows = _signed_get("/fapi/v2/positionRisk", {})
    out: List[Dict[str, Any]] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        sym = norm_symbol(str(row.get("symbol") or ""))
        if want is not None and sym not in want:
            continue
        amt = float(row.get("positionAmt") or 0.0)
        if abs(amt) < 1e-12:
            continue
        leverage = _opt_float(row.get("leverage"))
        out.append(
            {
                "symbol": sym,
                "side": "LONG" if amt > 0 else "SHORT",
                "quantity": abs(amt),
                "entry_price": _opt_float(row.get("entryPrice")),
                "mark_price": _opt_float(row.get("markPrice")),
                "unrealized_pnl_usdt": float(
                    row.get("unRealizedProfit") or row.get("unrealizedProfit") or 0.0
                ),
                "leverage": int(leverage) if leverage is not None else None,
                "liquidation_price": _opt_float(row.get("liquidationPrice")),
                "margin_type": (str(row.get("marginType") or "").upper() or None),
            }
        )
    out.sort(key=lambda r: r["symbol"])
    return out


def fetch_account_summary() -> Dict[str, Any]:
    """U 本位账户 USDT 摘要。"""
    data = _signed_get("/fapi/v2/account", {})
    assets = data.get("assets") or []
    usdt = None
    for row in assets:
        if str(row.get("asset") or "").upper() == "USDT":
            usdt = row
            break
    if usdt is None:
        raise RuntimeError("USDT asset not found in futures account")
    return {
        "asset": "USDT",
        "wallet_balance_usdt": float(usdt.get("walletBalance") or 0.0),
        "available_balance_usdt": float(
            data.get("availableBalance") or usdt.get("availableBalance") or 0.0
        ),
        "unrealized_pnl_usdt": float(
            usdt.get("unrealizedProfit") or data.get("totalUnrealizedProfit") or 0.0
        ),
    }


def _trade_side_label(side: str, *, realized_pnl: float) -> str:
    s = str(side or "").upper()
    if abs(realized_pnl) > 1e-12:
        if s == "SELL":
            return "平多"
        if s == "BUY":
            return "平空"
    if s == "BUY":
        return "开多"
    if s == "SELL":
        return "开空"
    return s or "—"


def fetch_user_trades(
    symbols: List[str],
    *,
    days: int = 7,
    limit_per_symbol: int = 100,
) -> List[Dict[str, Any]]:
    """各标的 userTrades 合并，按成交时间倒序。"""
    if not symbols:
        return []
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - max(1, int(days)) * 86_400_000
    cap = max(1, min(1000, int(limit_per_symbol)))
    merged: List[Dict[str, Any]] = []
    for raw in symbols:
        sym = norm_symbol(raw)
        try:
            rows = _signed_get(
                "/fapi/v1/userTrades",
                {
                    "symbol": sym,
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": cap,
                },
            )
        except Exception as exc:
            logger.warning("[binance] userTrades %s failed: %s", sym, exc)
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            realized = float(row.get("realizedPnl") or 0.0)
            side = str(row.get("side") or "")
            commission = float(row.get("commission") or 0.0)
            merged.append(
                {
                    "id": int(row.get("id") or 0),
                    "order_id": int(row.get("orderId") or 0),
                    "symbol": sym,
                    "side": side,
                    "side_label": _trade_side_label(side, realized_pnl=realized),
                    "price": float(row.get("price") or 0.0),
                    "qty": float(row.get("qty") or 0.0),
                    "quote_qty": float(row.get("quoteQty") or 0.0),
                    "realized_pnl_usdt": realized,
                    "commission_usdt": commission if str(row.get("commissionAsset") or "").upper() == "USDT" else commission,
                    "commission_asset": str(row.get("commissionAsset") or ""),
                    "role": "挂单方" if row.get("maker") else "吃单方",
                    "time_ms": int(row.get("time") or 0),
                }
            )
    merged.sort(key=lambda r: r.get("time_ms") or 0, reverse=True)
    return merged


def fetch_position_snapshots(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """symbol -> {amount, entry}（仅非零持仓）。"""
    want = {norm_symbol(s) for s in symbols}
    rows = _signed_get("/fapi/v2/positionRisk", {})
    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        sym = norm_symbol(str(row.get("symbol") or ""))
        if sym not in want:
            continue
        amt = float(row.get("positionAmt") or 0.0)
        if abs(amt) < 1e-12:
            continue
        out[sym] = {
            "amount": amt,
            "entry": float(row.get("entryPrice") or 0.0),
        }
    return out


def _lane_leverage(cfg) -> int:
    lev = float(getattr(cfg, "live_leverage", 0.0) or 0.0)
    if lev > 0:
        return int(lev)
    if isinstance(cfg, KKConfig):
        return int(_kk_leverage(cfg, cfg.orb_session_cfg()))
    return 5


def ensure_pool_leverage(symbols: List[str], cfg) -> None:
    ensure_one_way_mode()
    lev = _lane_leverage(cfg)
    for raw in symbols:
        sym = norm_symbol(raw)
        set_symbol_margin_isolated(sym)
        set_symbol_leverage(sym, lev)

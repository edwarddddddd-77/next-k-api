"""Bitget 现货账户 REST（余额持仓）。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from quant.engine.exchanges.bitget_spot.rest import signed_request
from quant.market.bitget_spot import fetch_symbol_info

logger = logging.getLogger(__name__)

_symbol_meta: Dict[str, Dict[str, Any]] = {}


def _norm_pair(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def load_symbol_meta(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    want = {_norm_pair(s) for s in symbols}
    out: Dict[str, Dict[str, Any]] = {}
    for sym in sorted(want):
        if sym in _symbol_meta:
            out[sym] = _symbol_meta[sym]
            continue
        try:
            row = fetch_symbol_info(sym)
        except Exception as exc:
            logger.warning("[bitget_spot] symbol meta %s failed: %s", sym, exc)
            continue
        if not row:
            continue
        meta = {
            "symbol": sym,
            "base_coin": str(row.get("baseCoin") or ""),
            "quote_coin": str(row.get("quoteCoin") or "USDT"),
            "price_precision": int(str(row.get("pricePrecision") or "2")),
            "quantity_precision": int(str(row.get("quantityPrecision") or "6")),
            "min_trade_usdt": float(row.get("minTradeUSDT") or 1.0),
        }
        _symbol_meta[sym] = meta
        out[sym] = meta
    return out


def fetch_position_snapshots(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    want = {_norm_pair(s) for s in symbols}
    meta = load_symbol_meta(list(want))
    base_to_symbol = {str(m.get("base_coin") or "").upper(): sym for sym, m in meta.items()}
    out: Dict[str, Dict[str, float]] = {}
    try:
        rows = signed_request("GET", "/api/v2/spot/account/assets")
    except RuntimeError as exc:
        logger.warning("[bitget_spot] assets failed: %s", exc)
        return out
    if not isinstance(rows, list):
        return out
    for row in rows:
        coin = str(row.get("coin") or row.get("coinName") or "").upper()
        sym = base_to_symbol.get(coin)
        if not sym or sym not in want:
            continue
        available = float(row.get("available") or row.get("availableBalance") or 0.0)
        frozen = float(row.get("frozen") or row.get("locked") or 0.0)
        total = available + frozen
        if total <= 1e-12:
            continue
        entry = float(row.get("avgCost") or row.get("averageCost") or 0.0)
        out[sym] = {"amount": total, "entry": entry}
    return out


def fetch_position_amounts(symbols: List[str]) -> Dict[str, float]:
    return {sym: float(snap.get("amount") or 0.0) for sym, snap in fetch_position_snapshots(symbols).items()}


def ensure_pool_leverage(symbols: List[str], cfg) -> None:
    """现货无杠杆设置。"""
    return None

"""Bitget 现货 vnpy Gateway（REST + K 线轮询）。"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from quant.engine.bootstrap import ensure_vnpy_path
from quant.engine.exchanges.base import VnpyLiveGatewayMixin
from quant.engine.exchanges.bitget_spot import account as spot_account
from quant.engine.exchanges.bitget_spot.rest import base_url, credentials_configured, signed_request
from quant.market.bitget_spot import fetch_klines_forward, fetch_mark_price, fetch_symbol_info

ensure_vnpy_path()

from vnpy.trader.constant import Direction, Exchange, Interval, Offset, OrderType, Product, Status  # noqa: E402
from vnpy.trader.gateway import BaseGateway  # noqa: E402
from vnpy.trader.object import (  # noqa: E402
    AccountData,
    BarData,
    CancelRequest,
    ContractData,
    HistoryRequest,
    OrderData,
    OrderRequest,
    PositionData,
    SubscribeRequest,
    TickData,
    TradeData,
)

logger = logging.getLogger(__name__)

_VT_SUFFIX = "_SPOT_BITGET"
GATEWAY_NAME = "BITGET_SPOT"


def _norm_pair(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def vnpy_vt_symbol(symbol: str) -> str:
    return f"{_norm_pair(symbol)}{_VT_SUFFIX}.GLOBAL"


def symbol_from_vt(vt_symbol: str) -> str:
    raw = str(vt_symbol or "").split(".", 1)[0]
    if raw.endswith(_VT_SUFFIX):
        raw = raw[: -len(_VT_SUFFIX)]
    return _norm_pair(raw)


def bitget_spot_credentials_configured() -> bool:
    return credentials_configured()


def bitget_spot_connect_setting() -> dict:
    return {
        "API Key": "",
        "Secret Key": "",
        "Passphrase": "",
        "Server": "REAL",
        "Kline Stream": "True",
    }


def _price_tick(precision: int) -> float:
    p = max(0, int(precision))
    return 10 ** (-p) if p > 0 else 1.0


def _volume_step(precision: int) -> float:
    p = max(0, int(precision))
    return 10 ** (-p) if p > 0 else 0.000001


class VnpyBitgetSpotGateway(VnpyLiveGatewayMixin, BaseGateway):
    """Bitget 现货 Gateway — 市价买卖 + 5m K 线推送。"""

    default_name = GATEWAY_NAME
    default_setting = bitget_spot_connect_setting()
    exchanges = [Exchange.GLOBAL]

    def __init__(self, event_engine, gateway_name: str = GATEWAY_NAME) -> None:
        super().__init__(event_engine, gateway_name)
        self.name_contract_map: Dict[str, ContractData] = {}
        self._subscribed: Set[str] = set()
        self._symbol_meta: Dict[str, Dict] = {}
        self._poll_stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._last_bar_ms: Dict[str, int] = {}
        self._order_seq = 0

    def connect(self, setting: dict) -> None:
        if not credentials_configured():
            self.write_log("BITGET spot credentials missing — contracts only from public API")
        self.write_log(f"Bitget spot connecting {base_url()}")
        raw = str(setting.get("Symbols") or "").strip()
        preload = [_norm_pair(s) for s in raw.split(",") if s.strip()]
        self._load_contracts(preload)
        self._query_account()
        self._query_positions()
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_klines, name="bitget-spot-kline", daemon=True)
        self._poll_thread.start()
        self.write_log("Bitget spot connected")

    def close(self) -> None:
        self._poll_stop.set()
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=3.0)
        self.write_log("Bitget spot closed")

    def subscribe(self, req: SubscribeRequest) -> None:
        raw = str(req.symbol or "")
        if raw.endswith(_VT_SUFFIX):
            sym = _norm_pair(raw[: -len(_VT_SUFFIX)])
        else:
            sym = symbol_from_vt(f"{raw}.GLOBAL")
        self._subscribed.add(sym)
        self.write_log(f"subscribe {sym}")

    def send_order(self, req: OrderRequest) -> str:
        sym = symbol_from_vt(req.symbol)
        meta = self._symbol_meta.get(sym) or spot_account.load_symbol_meta([sym]).get(sym, {})
        contract = self.name_contract_map.get(sym)
        if contract is None:
            self.write_log(f"reject order unknown symbol {sym}")
            return ""
        self._order_seq += 1
        local_id = f"{int(time.time() * 1000)}_{self._order_seq}"
        order = req.create_order_data(local_id, self.gateway_name)
        order.status = Status.SUBMITTING
        self.on_order(order)

        side = "buy" if req.direction == Direction.LONG else "sell"
        vol = float(req.volume or 0.0)
        if vol <= 0:
            order.status = Status.REJECTED
            self.on_order(order)
            return order.vt_orderid

        px = fetch_mark_price(sym) or 0.0
        qty_prec = int(meta.get("quantity_precision") or 6)
        if side == "buy":
            quote_size = vol * px if px > 0 else vol
            size_s = f"{quote_size:.{max(2, qty_prec)}f}".rstrip("0").rstrip(".")
        else:
            size_s = f"{vol:.{qty_prec}f}".rstrip("0").rstrip(".")

        client_oid = f"ibs_{uuid.uuid4().hex[:20]}"
        try:
            data = signed_request(
                "POST",
                "/api/v2/spot/trade/place-order",
                body={
                    "symbol": sym,
                    "side": side,
                    "orderType": "market",
                    "force": "gtc",
                    "size": size_s,
                    "clientOid": client_oid,
                },
            )
            order_id = str((data or {}).get("orderId") or "")
            if not order_id:
                raise RuntimeError(f"missing orderId: {data}")
            fill = self._wait_order_fill(order_id)
            if not fill:
                order.status = Status.REJECTED
                self.on_order(order)
                return order.vt_orderid
            order.status = Status.ALLTRADED
            order.traded = float(fill.get("baseVolume") or vol)
            order.price = float(fill.get("priceAvg") or px or 0.0)
            self.on_order(order)
            trade = TradeData(
                symbol=contract.symbol,
                exchange=contract.exchange,
                orderid=local_id,
                tradeid=order_id,
                direction=req.direction,
                offset=req.offset,
                price=order.price,
                volume=order.traded,
                datetime=datetime.now(timezone.utc),
                gateway_name=self.gateway_name,
            )
            self.on_trade(trade)
            self._push_position(sym)
        except Exception as exc:
            self.write_log(f"spot order failed {sym}: {exc}")
            order.status = Status.REJECTED
            self.on_order(order)
        return order.vt_orderid

    def cancel_order(self, req: CancelRequest) -> None:
        self.write_log(f"cancel ignored for market order {req.orderid}")

    def query_account(self) -> None:
        self._query_account()

    def query_position(self) -> None:
        self._query_positions()

    def query_history(self, req: HistoryRequest):
        return []

    def _load_contracts(self, symbols: List[str]) -> None:
        for sym in sorted({_norm_pair(s) for s in symbols if s}):
            row = fetch_symbol_info(sym)
            if not row:
                self.write_log(f"contract load failed {sym}: not listed")
                continue
            sym = _norm_pair(str(row.get("symbol") or sym))
            if not sym:
                continue
            price_prec = int(str(row.get("pricePrecision") or "2"))
            qty_prec = int(str(row.get("quantityPrecision") or "6"))
            meta = {
                "symbol": sym,
                "base_coin": str(row.get("baseCoin") or ""),
                "quote_coin": str(row.get("quoteCoin") or "USDT"),
                "price_precision": price_prec,
                "quantity_precision": qty_prec,
                "min_trade_usdt": float(row.get("minTradeUSDT") or 1.0),
            }
            self._symbol_meta[sym] = meta
            contract = ContractData(
                symbol=f"{sym}{_VT_SUFFIX}",
                exchange=Exchange.GLOBAL,
                name=sym,
                product=Product.SPOT,
                size=1.0,
                pricetick=_price_tick(price_prec),
                min_volume=_volume_step(qty_prec),
                history_data=True,
                gateway_name=self.gateway_name,
            )
            self.name_contract_map[sym] = contract
            self.on_contract(contract)

    def _query_account(self) -> None:
        if not credentials_configured():
            return
        try:
            rows = signed_request("GET", "/api/v2/spot/account/assets")
        except Exception as exc:
            self.write_log(f"account query failed: {exc}")
            return
        usdt = 0.0
        if isinstance(rows, list):
            for row in rows:
                if str(row.get("coin") or "").upper() == "USDT":
                    usdt = float(row.get("available") or row.get("availableBalance") or 0.0)
                    break
        account = AccountData(
            accountid="USDT",
            balance=usdt,
            frozen=0.0,
            gateway_name=self.gateway_name,
        )
        self.on_account(account)

    def _query_positions(self) -> None:
        if not credentials_configured():
            return
        snaps = spot_account.fetch_position_snapshots(list(self.name_contract_map.keys()))
        for sym, snap in snaps.items():
            self._emit_position(sym, float(snap.get("amount") or 0.0))

    def _push_position(self, sym: str) -> None:
        snaps = spot_account.fetch_position_snapshots([sym])
        amt = float((snaps.get(sym) or {}).get("amount") or 0.0)
        self._emit_position(sym, amt)

    def _emit_position(self, sym: str, volume: float) -> None:
        contract = self.name_contract_map.get(sym)
        if contract is None:
            return
        pos = PositionData(
            symbol=contract.symbol,
            exchange=contract.exchange,
            direction=Direction.LONG,
            volume=max(0.0, volume),
            frozen=0.0,
            price=0.0,
            pnl=0.0,
            gateway_name=self.gateway_name,
        )
        self.on_position(pos)

    def _wait_order_fill(self, order_id: str, *, attempts: int = 8) -> Optional[dict]:
        for _ in range(max(1, attempts)):
            try:
                data = signed_request(
                    "GET",
                    "/api/v2/spot/trade/orderInfo",
                    params={"orderId": order_id},
                )
                row = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
                if isinstance(row, dict) and str(row.get("status") or "").lower() == "filled":
                    return row
            except Exception as exc:
                logger.debug("[bitget_spot] order poll %s: %s", order_id, exc)
            time.sleep(0.35)
        return None

    def _poll_klines(self) -> None:
        while not self._poll_stop.is_set():
            symbols = sorted(self._subscribed) or sorted(self.name_contract_map.keys())
            end_ms = int(time.time() * 1000)
            start_ms = end_ms - 15 * 60_000
            for sym in symbols:
                contract = self.name_contract_map.get(sym)
                if contract is None:
                    continue
                try:
                    rows = fetch_klines_forward(sym, "5m", start_ms, end_ms)
                except Exception as exc:
                    logger.debug("[bitget_spot] kline poll %s: %s", sym, exc)
                    continue
                if not rows:
                    continue
                row = rows[-1]
                open_ms = int(row[0])
                if self._last_bar_ms.get(sym) == open_ms:
                    continue
                self._last_bar_ms[sym] = open_ms
                dt = datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc)
                bar = BarData(
                    symbol=contract.symbol,
                    exchange=contract.exchange,
                    datetime=dt,
                    interval=Interval.MINUTE,
                    open_price=float(row[1]),
                    high_price=float(row[2]),
                    low_price=float(row[3]),
                    close_price=float(row[4]),
                    volume=float(row[5]),
                    gateway_name=self.gateway_name,
                )
                tick = TickData(
                    symbol=contract.symbol,
                    exchange=contract.exchange,
                    datetime=dt,
                    name=sym,
                    last_price=float(row[4]),
                    volume=float(row[5]),
                    gateway_name=self.gateway_name,
                )
                tick.extra = {"bar": bar}
                self.on_tick(tick)
            self._poll_stop.wait(20.0)


__all__ = [
    "GATEWAY_NAME",
    "VnpyBitgetSpotGateway",
    "bitget_spot_connect_setting",
    "bitget_spot_credentials_configured",
    "symbol_from_vt",
    "vnpy_vt_symbol",
]

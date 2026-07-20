"""Bitget / OKX execution clients (USDT perpetuals, cross margin, market orders).

Market data & factors stay on Binance; these clients only EXECUTE live orders
when meta executor_exchange = bitget|okx. Symbols use the Binance convention
(BTCUSDT, 1000PEPEUSDT) and are mapped per venue:

  OKX:    BTCUSDT -> BTC-USDT-SWAP; 1000PEPEUSDT -> PEPE-USDT-SWAP with the
          quantity multiplied by 1000 (OKX has no 1000x tickers), then
          converted to contract sheets via ctVal.
  Bitget: same ticker text (productType USDT-FUTURES), size in coin units.
"""
import base64
import hashlib
import hmac
import json
import logging
import time

import httpx

from . import config

log = logging.getLogger("exchanges")

OKX_BASE = "https://www.okx.com"
BITGET_BASE = "https://api.bitget.com"


def _split_1000(symbol):
    """('PEPE', 1000.0) for 1000PEPEUSDT; ('BTC', 1.0) for BTCUSDT."""
    coin = symbol.replace("USDT", "")
    if coin.startswith("1000") and len(coin) > 4:
        return coin[4:], 1000.0
    return coin, 1.0


# ── OKX ──────────────────────────────────────────────────────────────────────

def okx_inst_id(symbol):
    coin, _ = _split_1000(symbol)
    return f"{coin}-USDT-SWAP"


def _okx_sign(secret, ts, method, path, body=""):
    msg = f"{ts}{method}{path}{body}"
    return base64.b64encode(
        hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()).decode()


async def _okx(method, path, params=None, body=None, auth=True, timeout=15):
    query = ""
    if params:
        query = "?" + "&".join(f"{k}={v}" for k, v in params.items())
    body_s = json.dumps(body) if body else ""
    headers = {"Content-Type": "application/json"}
    if auth:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + \
            f".{int(time.time() * 1000) % 1000:03d}Z"
        headers.update({
            "OK-ACCESS-KEY": config.OKX_API_KEY,
            "OK-ACCESS-SIGN": _okx_sign(config.OKX_SECRET_KEY, ts, method,
                                        path + query, body_s),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": config.OKX_PASSPHRASE,
        })
    async with httpx.AsyncClient(timeout=timeout) as c:
        resp = await c.request(method, OKX_BASE + path + query,
                               headers=headers, content=body_s or None)
        resp.raise_for_status()
        d = resp.json()
    if str(d.get("code")) != "0":
        raise RuntimeError(f"OKX {path}: code={d.get('code')} "
                           f"msg={d.get('msg')} data={str(d.get('data'))[:200]}")
    return d.get("data")


_okx_ct = {}   # instId -> (ctVal, lotSz, minSz)


async def _okx_contract(inst_id):
    if inst_id not in _okx_ct:
        data = await _okx("GET", "/api/v5/public/instruments",
                          {"instType": "SWAP", "instId": inst_id}, auth=False)
        if not data:
            raise RuntimeError(f"OKX instrument not found: {inst_id}")
        it = data[0]
        _okx_ct[inst_id] = (float(it["ctVal"]), float(it["lotSz"]), float(it["minSz"]))
    return _okx_ct[inst_id]


async def okx_place_market(symbol, side, qty, reduce_only=False):
    """qty in Binance units (e.g. 1000PEPE lots) -> OKX contract sheets."""
    inst = okx_inst_id(symbol)
    _, mult = _split_1000(symbol)
    ct_val, lot, min_sz = await _okx_contract(inst)
    sheets = qty * mult / ct_val
    sheets = max(round(sheets / lot) * lot, min_sz)
    body = {"instId": inst, "tdMode": "cross",
            "side": "buy" if side.upper() == "BUY" else "sell",
            "ordType": "market", "sz": f"{sheets:.10g}"}
    if reduce_only:
        body["reduceOnly"] = True
    return await _okx("POST", "/api/v5/trade/order", body=body)


async def okx_set_leverage(symbol, leverage):
    try:
        return await _okx("POST", "/api/v5/account/set-leverage",
                          body={"instId": okx_inst_id(symbol),
                                "lever": str(int(leverage)), "mgnMode": "cross"})
    except Exception as exc:  # noqa: BLE001
        log.warning("okx set_leverage %s failed: %s", symbol, exc)
        return None


async def okx_equity():
    data = await _okx("GET", "/api/v5/account/balance", {"ccy": "USDT"})
    det = (data[0].get("details") or [{}])[0] if data else {}
    total = float(data[0].get("totalEq") or 0) if data else 0.0
    avail = float(det.get("availEq") or det.get("availBal") or 0)
    return total, avail


async def okx_test():
    total, avail = await okx_equity()
    return {"ok": True, "wallet_usdt": total, "available": avail}


# ── Bitget ───────────────────────────────────────────────────────────────────

def bitget_symbol(symbol):
    return symbol  # same ticker convention incl. 1000x contracts


def _bitget_sign(secret, ts, method, path_with_query, body=""):
    msg = f"{ts}{method}{path_with_query}{body}"
    return base64.b64encode(
        hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()).decode()


async def _bitget(method, path, params=None, body=None, timeout=15):
    query = ""
    if params:
        query = "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    body_s = json.dumps(body) if body else ""
    ts = str(int(time.time() * 1000))
    headers = {
        "Content-Type": "application/json", "locale": "en-US",
        "ACCESS-KEY": config.BITGET_API_KEY,
        "ACCESS-SIGN": _bitget_sign(config.BITGET_SECRET_KEY, ts, method,
                                    path + query, body_s),
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": config.BITGET_PASSPHRASE,
    }
    async with httpx.AsyncClient(timeout=timeout) as c:
        resp = await c.request(method, BITGET_BASE + path + query,
                               headers=headers, content=body_s or None)
        resp.raise_for_status()
        d = resp.json()
    if str(d.get("code")) != "00000":
        raise RuntimeError(f"Bitget {path}: code={d.get('code')} msg={d.get('msg')}")
    return d.get("data")


_bitget_spec = {}   # symbol -> (sizeMultiplier/minTradeNum step, minTradeNum)


async def _bitget_contract(symbol):
    if symbol not in _bitget_spec:
        data = await _bitget("GET", "/api/v2/mix/market/contracts",
                             {"productType": "USDT-FUTURES", "symbol": symbol})
        if not data:
            raise RuntimeError(f"Bitget contract not found: {symbol}")
        it = data[0]
        step = float(it.get("sizeMultiplier") or it.get("minTradeNum") or 0.001)
        _bitget_spec[symbol] = (step, float(it.get("minTradeNum") or step))
    return _bitget_spec[symbol]


async def bitget_place_market(symbol, side, qty, reduce_only=False):
    step, min_num = await _bitget_contract(symbol)
    size = max(round(qty / step) * step, min_num)
    body = {"symbol": symbol, "productType": "USDT-FUTURES",
            "marginMode": "crossed", "marginCoin": "USDT",
            "size": f"{size:.10g}", "orderType": "market",
            "side": "buy" if side.upper() == "BUY" else "sell",
            "reduceOnly": "YES" if reduce_only else "NO"}
    return await _bitget("POST", "/api/v2/mix/order/place-order", body=body)


async def bitget_set_leverage(symbol, leverage):
    try:
        return await _bitget("POST", "/api/v2/mix/account/set-leverage",
                             body={"symbol": symbol, "productType": "USDT-FUTURES",
                                   "marginCoin": "USDT", "leverage": str(int(leverage))})
    except Exception as exc:  # noqa: BLE001
        log.warning("bitget set_leverage %s failed: %s", symbol, exc)
        return None


async def bitget_equity():
    data = await _bitget("GET", "/api/v2/mix/account/accounts",
                         {"productType": "USDT-FUTURES"})
    row = next((a for a in (data or []) if a.get("marginCoin") == "USDT"), {})
    total = float(row.get("accountEquity") or 0)
    avail = float(row.get("crossedMaxAvailable") or row.get("available") or 0)
    return total, avail


async def bitget_test():
    total, avail = await bitget_equity()
    return {"ok": True, "wallet_usdt": total, "available": avail}


# ── venue-agnostic facade (executor uses these) ──────────────────────────────

SUPPORTED = ("binance", "bitget", "okx")


def has_credentials(venue):
    if venue == "binance":
        return bool(config.BINANCE_API_KEY and config.BINANCE_SECRET_KEY)
    if venue == "bitget":
        return bool(config.BITGET_API_KEY and config.BITGET_SECRET_KEY
                    and config.BITGET_PASSPHRASE)
    if venue == "okx":
        return bool(config.OKX_API_KEY and config.OKX_SECRET_KEY
                    and config.OKX_PASSPHRASE)
    return False


async def place_market(venue, symbol, side, qty, reduce_only=False):
    if venue == "bitget":
        return await bitget_place_market(symbol, side, qty, reduce_only)
    if venue == "okx":
        return await okx_place_market(symbol, side, qty, reduce_only)
    raise ValueError(f"unsupported venue {venue}")


async def set_leverage(venue, symbol, leverage):
    if venue == "bitget":
        return await bitget_set_leverage(symbol, leverage)
    if venue == "okx":
        return await okx_set_leverage(symbol, leverage)
    return None


async def equity(venue):
    if venue == "bitget":
        return await bitget_equity()
    if venue == "okx":
        return await okx_equity()
    raise ValueError(f"unsupported venue {venue}")

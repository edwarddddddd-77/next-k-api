"""HL → Bitget ticker aliases must cover known twins; no false friends."""

from __future__ import annotations

from unittest.mock import patch

from utils import hl_bitget_symbol_map as m

# Minimal listed set: real twins + a DXYZ trap (crypto, not dollar index).
_LISTED = {
    "BTCUSDT",
    "CLUSDT",
    "BZUSDT",
    "XAUUSDT",
    "XAGUSDT",
    "XPTUSDT",
    "XPDUSDT",
    "GOOGLUSDT",
    "SKHYUSDT",
    "SKHYNIXUSDT",
    "SAMSUNGUSDT",
    "GIGADEVICEUSDT",
    "NOKSTOCKUSDT",
    "NDX100USDT",
    "QQQUSDT",
    "DXYZUSDT",
    "SP500USDT",
}


def _resolve(coin: str):
    with (
        patch.object(m, "verify_symbols_enabled", return_value=True),
        patch.object(m, "bitget_contract_set", return_value=_LISTED),
    ):
        return m.resolve_bitget_symbol(coin)


def test_commodity_and_stock_ticker_twins():
    cases = {
        "xyz:BRENTOIL": "BZUSDT",
        "xyz:PLATINUM": "XPTUSDT",
        "xyz:PALLADIUM": "XPDUSDT",
        "xyz:GOLD": "XAUUSDT",
        "xyz:SILVER": "XAGUSDT",
        "xyz:SKHX": "SKHYNIXUSDT",
        "xyz:SKHY": "SKHYUSDT",
        "xyz:SMSN": "SAMSUNGUSDT",
        "xyz:GIGADEV": "GIGADEVICEUSDT",
        "xyz:NOK": "NOKSTOCKUSDT",
        "xyz:XYZ100": "NDX100USDT",
        "xyz:GOOG": "GOOGLUSDT",
        "xyz:CL": "CLUSDT",
        "BTC": "BTCUSDT",
    }
    for coin, want in cases.items():
        sym, reason = _resolve(coin)
        assert (sym, reason) == (want, None), coin


def test_no_false_friend_aliases():
    # Dollar index is not DXYZ; Nasdaq-100 is NDX100 not QQQ.
    assert _resolve("xyz:DXY") == (None, "not_on_bitget")
    assert _resolve("xyz:XYZ100")[0] != "QQQUSDT"
    assert _resolve("xyz:SOFTBANK") == (None, "not_on_bitget")
    assert _resolve("xyz:UNITREE") == (None, "not_on_bitget")

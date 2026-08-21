"""Per-coin realized PnL aggregation from Bitget history-position rows."""

from quant.engine.exchanges.bitget.account import aggregate_coin_realized_pnl


def test_aggregate_coin_realized_prefers_net_profit():
    rows = [
        {
            "symbol": "BTCUSDT",
            "netProfit": "10.5",
            "pnl": "12",
            "uTime": "1700000001000",
        },
        {
            "symbol": "BTCUSDT",
            "netProfit": "-3.25",
            "pnl": "0",
            "uTime": "1700000002000",
        },
        {
            "symbol": "DOGEUSDT",
            "pnl": "1.1",
            "cTime": "1700000000000",
        },
    ]
    out = aggregate_coin_realized_pnl(rows)
    assert set(out) == {"BTC", "DOGE"}
    assert out["BTC"]["realized"] == 7.25
    assert out["BTC"]["closes"] == 2
    assert out["BTC"]["last_ts"]
    assert out["DOGE"]["realized"] == 1.1
    assert out["DOGE"]["closes"] == 1


def test_aggregate_coin_realized_empty():
    assert aggregate_coin_realized_pnl(None) == {}
    assert aggregate_coin_realized_pnl([]) == {}

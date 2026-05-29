from __future__ import annotations


def test_live_summary_aggregates_protocol_positions():
    from routers.moss_quant import _summarize_protocol_moss

    summary = _summarize_protocol_moss(
        account={
            "wallet_balance_usdt": 1000,
            "available_balance_usdt": 900,
            "unrealized_pnl_usdt": 25,
            "moss_quant": {"leverage": 8},
        },
        positions=[
            {
                "profile_id": 1,
                "symbol": "BTCUSDT",
                "status": "open",
                "pnl_usdt": None,
                "notional_usdt": 500,
            },
            {
                "profile_id": 1,
                "symbol": "BTCUSDT",
                "status": "closed",
                "pnl_usdt": 12.5,
                "notional_usdt": 500,
            },
            {
                "profile_id": 2,
                "symbol": "ETHUSDT",
                "status": "closed",
                "pnl_usdt": -3.0,
                "notional_usdt": 300,
            },
        ],
        enabled_profiles=2,
    )

    assert summary["mode"] == "live"
    assert summary["wallet_balance_usdt"] == 1000
    assert summary["open_positions"] == 1
    assert summary["settled_count"] == 2
    assert summary["total_pnl_usdt"] == 9.5

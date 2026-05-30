from __future__ import annotations

import sqlite3


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
    assert summary["profile_capital_usdt"] == 500
    assert summary["open_positions"] == 1
    assert summary["settled_count"] == 2
    assert summary["total_pnl_usdt"] == 9.5


def test_live_unavailable_summary_does_not_emit_paper_wallet_defaults():
    from moss_quant import config as mq_cfg
    from moss_quant.db import migrate_moss_tables
    from routers.moss_quant import _moss_live_unavailable_summary

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        migrate_moss_tables(conn.cursor())

        summary = _moss_live_unavailable_summary(
            conn,
            mq_cfg,
            reason="protocol_api_url_missing",
            enabled_profiles=3,
        )
    finally:
        conn.close()

    assert summary["mode"] == "live_unavailable"
    assert summary["protocol_error"] == "protocol_api_url_missing"
    assert summary["wallet_initial_usdt"] is None
    assert summary["wallet_balance_usdt"] is None
    assert summary["available_balance_usdt"] is None
    assert summary["profile_capital_usdt"] is None
    assert summary["enabled_profiles"] == 3


def test_live_open_position_uses_mark_price_and_unrealized_pnl():
    from routers.moss_quant import (
        _position_to_moss_signal_row,
        _summarize_protocol_moss,
    )

    position = {
        "id": 88,
        "profile_id": 3,
        "symbol": "SOLUSDT",
        "status": "open",
        "side": "LONG",
        "entry_price": 150.0,
        "mark_price": 153.5,
        "close_price": None,
        "unrealized_pnl_usdt": 17.25,
        "pnl_usdt": 999.0,
        "notional_usdt": 600,
    }

    summary = _summarize_protocol_moss(
        account={"wallet_balance_usdt": 1000, "available_balance_usdt": 900},
        positions=[position],
        enabled_profiles=1,
    )
    signal = _position_to_moss_signal_row(position)

    assert summary["open_by_profile"] == [
        {
            "profile_id": 3,
            "symbol": "SOLUSDT",
            "open_count": 1,
            "unrealized_pnl_usdt": 17.25,
        }
    ]
    assert signal["mark_price"] == 153.5
    assert signal["unrealized_pnl_usdt"] == 17.25


def test_pending_entry_position_maps_to_pending_outcome():
    from routers.moss_quant import _position_to_moss_signal_row

    signal = _position_to_moss_signal_row(
        {
            "id": 89,
            "profile_id": 4,
            "symbol": "ETHUSDT",
            "status": "pending_entry",
            "entry_price": 3000.0,
            "mark_price": 3001.0,
            "pnl_usdt": None,
        }
    )

    assert signal["outcome"] == "pending_entry"
    assert signal["outcome_at_utc"] is None
    assert signal["exit_rule"] is None

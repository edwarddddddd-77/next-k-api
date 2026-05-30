from __future__ import annotations

import sqlite3


def _migrate(conn):
    from moss_quant.db import migrate_moss_tables

    conn.row_factory = sqlite3.Row
    migrate_moss_tables(conn.cursor())


def _seed_profile(conn, profile_id, symbol):
    now = "2024-01-01T00:00:00Z"
    conn.execute(
        """INSERT INTO moss_profiles(
               id, name, symbol, template, enabled, initial_params_json,
               tactical_params_json, created_at_utc, updated_at_utc)
           VALUES (?, ?, ?, 'balanced', 1, '{}', '{}', ?, ?)""",
        (profile_id, symbol.lower(), symbol, now, now),
    )


def test_live_summary_aggregates_protocol_positions():
    from routers.moss_quant import _summarize_protocol_moss

    conn = sqlite3.connect(":memory:")
    _migrate(conn)
    _seed_profile(conn, 1, "BTCUSDT")
    _seed_profile(conn, 2, "ETHUSDT")
    conn.execute(
        """INSERT INTO moss_settlements(
               settled_at_utc, signal_id, profile_id, symbol, side, outcome,
               entry_price, exit_price, pnl_usdt, virtual_notional_usdt, exit_rule)
           VALUES
           ('2024-01-01T00:00:00Z', 11, 1, 'BTCUSDT', 'LONG', 'win', 100, 110, 12.5, 500, 'tp'),
           ('2024-01-01T00:00:00Z', 12, 2, 'ETHUSDT', 'SHORT', 'loss', 100, 103, -3.0, 300, 'sl')"""
    )

    summary = _summarize_protocol_moss(
        conn=conn,
        account={
            "wallet_balance_usdt": 1000,
            "available_balance_usdt": 900,
            "unrealized_pnl_usdt": 25,
        },
        positions=[
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry_price": 65000,
                "mark_price": 65100,
                "quantity": 0.01,
                "unrealized_pnl_usdt": 1.0,
                "leverage": 8,
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
    conn.close()


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

    conn = sqlite3.connect(":memory:")
    _migrate(conn)
    _seed_profile(conn, 3, "SOLUSDT")

    position = {
        "symbol": "SOLUSDT",
        "side": "LONG",
        "entry_price": 150.0,
        "mark_price": 153.5,
        "quantity": 4.0,
        "unrealized_pnl_usdt": 17.25,
        "leverage": 4,
    }

    summary = _summarize_protocol_moss(
        conn=conn,
        account={"wallet_balance_usdt": 1000, "available_balance_usdt": 900},
        positions=[position],
        enabled_profiles=1,
    )
    signal = _position_to_moss_signal_row(position, profile_id=3)

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
    assert signal["virtual_notional_usdt"] == 600.0
    conn.close()


def test_live_position_maps_to_open_signal_row():
    from routers.moss_quant import _position_to_moss_signal_row

    signal = _position_to_moss_signal_row(
        {
            "symbol": "ETHUSDT",
            "side": "SHORT",
            "entry_price": 3000.0,
            "mark_price": 3001.0,
            "quantity": 0.2,
            "unrealized_pnl_usdt": -0.2,
            "leverage": 5,
        }
    )

    assert signal["outcome"] is None
    assert signal["outcome_at_utc"] is None
    assert signal["exit_rule"] is None

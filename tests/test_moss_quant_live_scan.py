from __future__ import annotations

import sqlite3


def test_positions_map_groups_protocol_positions_by_symbol():
    from moss_quant.paper_scanner import protocol_open_positions_by_symbol

    rows = [
        {"symbol": "BTCUSDT", "side": "LONG", "entry_price": 65000, "quantity": 0.01, "leverage": 8},
        {"symbol": "BTCUSDT", "side": "LONG", "entry_price": 66000, "quantity": 0.01, "leverage": 8},
        {"symbol": "ETHUSDT", "side": "SHORT", "entry_price": 3000, "quantity": 0.2, "leverage": 8},
    ]

    by_symbol = protocol_open_positions_by_symbol(rows)
    assert len(by_symbol["BTCUSDT"]) == 2
    assert by_symbol["ETHUSDT"][0]["side"] == "SHORT"


def test_can_send_live_open_blocks_real_mode_when_protocol_truth_unavailable():
    from moss_quant.paper_scanner import can_send_live_open

    sender = object()

    assert can_send_live_open(None, live_opens_allowed=False)
    assert can_send_live_open(sender, live_opens_allowed=True)
    assert not can_send_live_open(sender, live_opens_allowed=False)


def test_protocol_ingest_result_requires_traded_action():
    from moss_quant.paper_scanner import protocol_ingest_open_result

    ok = protocol_ingest_open_result(
        {"traded": 1, "details": [{"action": "traded", "position_id": 42}]}
    )
    rejected = protocol_ingest_open_result(
        {"traded": 0, "errors": 1, "details": [{"action": "error", "error": "disabled"}]}
    )

    assert ok.ok is True
    assert rejected.ok is False
    assert "disabled" in rejected.error


def test_mark_profile_open_signals_external_closed_only_open_positions():
    from moss_quant.db import (
        mark_profile_open_signals_external_closed,
        migrate_moss_tables,
    )

    conn = sqlite3.connect(":memory:")
    migrate_moss_tables(conn.cursor())
    now = "2024-01-01T00:00:00Z"
    conn.execute(
        """INSERT INTO moss_profiles(
               id, name, symbol, template, enabled, initial_params_json,
               tactical_params_json, created_at_utc, updated_at_utc)
           VALUES (1, 'btc', 'BTCUSDT', 'balanced', 1, '{}', '{}', ?, ?)""",
        (now, now),
    )
    conn.execute(
        """INSERT INTO moss_signals(
               id, profile_id, recorded_at_utc, side, symbol, entry_price,
               virtual_notional_usdt, mark_price, unrealized_pnl_usdt, updated_at_utc)
           VALUES (10, 1, ?, 'LONG', 'BTCUSDT', 100, 1000, 110, 100, ?)""",
        (now, now),
    )
    conn.execute(
        """INSERT INTO moss_signals(
               id, profile_id, recorded_at_utc, side, symbol, outcome,
               outcome_at_utc, updated_at_utc)
           VALUES (11, 1, ?, 'WAIT', 'BTCUSDT', NULL, NULL, ?)""",
        (now, now),
    )

    changed = mark_profile_open_signals_external_closed(conn, 1)

    assert changed == 1
    row = conn.execute("SELECT outcome, exit_rule, unrealized_pnl_usdt FROM moss_signals WHERE id=10").fetchone()
    assert row == ("external_closed", "external_closed", 0)
    wait_outcome = conn.execute("SELECT outcome FROM moss_signals WHERE id=11").fetchone()[0]
    assert wait_outcome is None

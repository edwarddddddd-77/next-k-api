from __future__ import annotations

import json
import sqlite3

import pandas as pd


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


def test_live_close_failure_keeps_moss_signal_open(monkeypatch):
    from moss_quant.db import migrate_moss_tables
    from moss_quant import paper_scanner

    conn = sqlite3.connect(":memory:")
    migrate_moss_tables(conn.cursor())
    now = "2024-01-01T00:00:00Z"
    conn.execute(
        """INSERT INTO moss_profiles(
               id, name, symbol, template, enabled, initial_params_json,
               tactical_params_json, virtual_equity_usdt, created_at_utc, updated_at_utc)
           VALUES (1, 'btc', 'BTCUSDT', 'balanced', 1, '{}', '{}', 10000, ?, ?)""",
        (now, now),
    )
    conn.execute(
        """INSERT INTO moss_signals(
               id, profile_id, recorded_at_utc, side, symbol, entry_price,
               virtual_notional_usdt, mark_price, unrealized_pnl_usdt, updated_at_utc)
           VALUES (10, 1, ?, 'LONG', 'BTCUSDT', 100, 1000, 110, 100, ?)""",
        (now, now),
    )
    conn.commit()

    class FakeSender:
        def send_close(self, **kwargs):
            return {"traded": 0, "errors": 1, "details": [{"action": "error", "error": "close_failed"}]}

    class FakeProtocolClient:
        def get_account_summary(self):
            return {
                "wallet_balance_usdt": 10000.0,
                "available_balance_usdt": 9000.0,
                "unrealized_pnl_usdt": 100.0,
            }

        def get_moss_positions(self, **kwargs):
            return [{
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry_price": 100.0,
                "mark_price": 110.0,
                "quantity": 10.0,
                "leverage": 2.0,
                "unrealized_pnl_usdt": 100.0,
            }]

    monkeypatch.setattr(paper_scanner, "_get_sender", lambda: FakeSender())
    monkeypatch.setattr(
        __import__("moss_quant.protocol_client", fromlist=["ProtocolClient"]).ProtocolClient,
        "from_env",
        classmethod(lambda cls: FakeProtocolClient()),
    )
    monkeypatch.setattr(
        paper_scanner,
        "load_cached",
        lambda symbol, refresh=False: pd.DataFrame({"close": [110.0], "high": [111.0], "low": [109.0]}),
    )
    monkeypatch.setattr(paper_scanner, "classify_regime", lambda df, version=None: pd.Series(["SIDEWAYS"]))
    monkeypatch.setattr(
        paper_scanner,
        "exit_snapshot",
        lambda **kwargs: {
            "exit_rule": "take_profit",
            "pnl_pct": 20.0,
            "sl_thresh_pct": -5.0,
            "tp_thresh_pct": 10.0,
            "signal": 0,
        },
    )

    stats = paper_scanner.run_paper_scan(conn)

    row = conn.execute(
        "SELECT outcome, exit_rule, exit_price, pnl_usdt FROM moss_signals WHERE id=10"
    ).fetchone()
    settlement_count = conn.execute("SELECT COUNT(*) FROM moss_settlements").fetchone()[0]

    assert stats["closes"] == 0
    assert tuple(row) == (None, None, None, None)
    assert settlement_count == 0


def test_live_rolling_failure_keeps_local_notional_and_meta_unchanged(monkeypatch):
    from moss_quant.db import migrate_moss_tables
    from moss_quant import paper_scanner

    conn = sqlite3.connect(":memory:")
    migrate_moss_tables(conn.cursor())
    now = "2024-01-01T00:00:00Z"
    tactical = {
        "base_leverage": 2,
        "max_leverage": 2,
        "risk_per_trade": 1.0,
        "max_position_pct": 1.0,
        "rolling_enabled": True,
        "rolling_max_times": 3,
        "rolling_trigger_pct": 0.1,
        "rolling_reinvest_pct": 0.5,
        "trailing_enabled": False,
    }
    conn.execute(
        """INSERT INTO moss_profiles(
               id, name, symbol, template, enabled, initial_params_json,
               tactical_params_json, virtual_equity_usdt, created_at_utc, updated_at_utc)
           VALUES (1, 'btc', 'BTCUSDT', 'balanced', 1, '{}', ?, 10000, ?, ?)""",
        (json.dumps(tactical), now, now),
    )
    conn.execute(
        """INSERT INTO moss_signals(
               id, profile_id, recorded_at_utc, side, symbol, entry_price,
               virtual_notional_usdt, mark_price, unrealized_pnl_usdt, meta_json, updated_at_utc)
           VALUES (10, 1, ?, 'LONG', 'BTCUSDT', 100, 1000, 110, 100, '{}', ?)""",
        (now, now),
    )
    conn.commit()

    class FakeSender:
        def send_rolling(self, **kwargs):
            return {"traded": 0, "errors": 1, "details": [{"action": "error", "error": "rolling_failed"}]}

    class FakeProtocolClient:
        def get_account_summary(self):
            return {
                "wallet_balance_usdt": 10000.0,
                "available_balance_usdt": 9000.0,
                "unrealized_pnl_usdt": 100.0,
            }

        def get_moss_positions(self, **kwargs):
            return [{
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry_price": 100.0,
                "mark_price": 110.0,
                "quantity": 10.0,
                "leverage": 2.0,
                "unrealized_pnl_usdt": 100.0,
            }]

    monkeypatch.setattr(paper_scanner, "_get_sender", lambda: FakeSender())
    monkeypatch.setattr(
        __import__("moss_quant.protocol_client", fromlist=["ProtocolClient"]).ProtocolClient,
        "from_env",
        classmethod(lambda cls: FakeProtocolClient()),
    )
    monkeypatch.setattr(
        paper_scanner,
        "load_cached",
        lambda symbol, refresh=False: pd.DataFrame(
            {"close": [110.0] * 20, "high": [111.0] * 20, "low": [109.0] * 20}
        ),
    )
    monkeypatch.setattr(paper_scanner, "classify_regime", lambda df, version=None: pd.Series(["SIDEWAYS"]))
    monkeypatch.setattr(
        paper_scanner,
        "exit_snapshot",
        lambda **kwargs: {
            "exit_rule": None,
            "pnl_pct": 20.0,
            "sl_thresh_pct": -5.0,
            "tp_thresh_pct": 10.0,
            "signal": 0,
        },
    )
    monkeypatch.setattr(paper_scanner, "_free_margin", lambda *args, **kwargs: 10000.0)
    monkeypatch.setattr(paper_scanner, "compute_atr", lambda df, n: pd.Series([1.0] * len(df)))

    paper_scanner.run_paper_scan(conn)

    row = conn.execute(
        "SELECT virtual_notional_usdt, meta_json FROM moss_signals WHERE id=10"
    ).fetchone()

    assert row[0] == 1000.0
    assert row[1] == "{}"

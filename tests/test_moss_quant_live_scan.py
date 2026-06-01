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
        {"traded": 1, "details": [{"action": "traded", "position_id": 42, "client_ref": "moss:1:open:1"}]}
    )
    rejected = protocol_ingest_open_result(
        {"traded": 0, "errors": 1, "details": [{"action": "error", "error": "disabled"}]}
    )

    assert ok.ok is True
    assert ok.position_id == 42
    assert ok.client_ref == "moss:1:open:1"
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

    changed = mark_profile_open_signals_external_closed(conn, 1, exit_price=95.0)

    assert changed == 1
    row = conn.execute(
        "SELECT outcome, exit_rule, unrealized_pnl_usdt, pnl_usdt, exit_price FROM moss_signals WHERE id=10"
    ).fetchone()
    assert row[0] == "external_closed"
    assert row[1] == "external_closed"
    assert row[2] == 0
    assert row[3] is not None
    assert row[4] == 95.0
    settled = conn.execute(
        "SELECT COUNT(*) FROM moss_settlements WHERE profile_id=1"
    ).fetchone()[0]
    assert settled == 1
    wait_outcome = conn.execute("SELECT outcome FROM moss_signals WHERE id=11").fetchone()[0]
    assert wait_outcome is None


def test_compute_paper_protective_prices_long():
    import pandas as pd
    from moss_quant.core.decision import DecisionParams
    from moss_quant.paper_scanner import compute_paper_protective_prices

    params = DecisionParams(sl_atr_mult=2.0, tp_rr_ratio=2.0, trailing_enabled=False)
    df = pd.DataFrame(
        {
            "high": [110.0],
            "low": [90.0],
            "close": [100.0],
            "open": [100.0],
        }
    )
    prices = compute_paper_protective_prices(
        side="LONG",
        entry=100.0,
        mark=100.0,
        params=params,
        df=df,
    )

    assert prices["sl_price"] == 60.0
    assert prices["tp_price"] == 180.0


def test_compute_paper_protective_prices_trailing_tightens_long_sl():
    import pandas as pd
    from moss_quant.core.decision import DecisionParams
    from moss_quant.paper_scanner import compute_paper_protective_prices

    params = DecisionParams(
        sl_atr_mult=2.0,
        tp_rr_ratio=2.0,
        trailing_enabled=True,
        trailing_activation_pct=0.01,
        trailing_distance_atr=1.0,
    )
    df = pd.DataFrame(
        {
            "high": [115.0],
            "low": [100.0],
            "close": [110.0],
            "open": [100.0],
        }
    )
    prices = compute_paper_protective_prices(
        side="LONG",
        entry=100.0,
        mark=110.0,
        params=params,
        df=df,
    )

    assert prices["atr_sl"] == 70.0
    assert prices["trailing_sl"] == 100.0
    assert prices["sl_price"] == 100.0


def test_sync_live_protective_orders_updates_meta(monkeypatch):
    import pandas as pd
    from moss_quant.core.decision import DecisionParams
    from moss_quant.paper_scanner import sync_live_protective_orders

    sent = {"sl": 0, "tp": 0}

    class FakeSender:
        def send_update_sl(self, **kwargs):
            sent["sl"] += 1
            return {"details": [{"action": "traded"}]}

        def send_update_tp(self, **kwargs):
            sent["tp"] += 1
            return {"details": [{"action": "traded"}]}

    params = DecisionParams(sl_atr_mult=2.0, tp_rr_ratio=2.0)
    df = pd.DataFrame(
        {"high": [110.0], "low": [90.0], "close": [100.0], "open": [100.0]}
    )
    meta = sync_live_protective_orders(
        sender=FakeSender(),
        symbol="BTCUSDT",
        side="LONG",
        entry=100.0,
        mark=100.0,
        params=params,
        df=df,
        profile_id=1,
        meta={},
        has_live_position=True,
    )

    assert sent["sl"] == 1
    assert sent["tp"] == 1
    assert meta["last_synced_sl"] == 60.0
    assert meta["last_synced_tp"] == 180.0
    assert meta.get("last_synced_at_utc")

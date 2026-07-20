"""Regression: db writers must never call get_meta/current_mode while holding
the non-reentrant _lock (2026-07-18 deadlock), and paper/live stay segregated."""
import os
import tempfile
import threading


def _fresh_db():
    os.environ["QB_DB_PATH"] = tempfile.mktemp(suffix=".db")
    import importlib

    from backend import config
    importlib.reload(config)
    from backend import db
    importlib.reload(db)
    db.init()
    return db


def _no_deadlock(fn, timeout=3):
    done = []
    t = threading.Thread(target=lambda: (fn(), done.append(1)), daemon=True)
    t.start()
    t.join(timeout)
    return bool(done)


def test_mode_aware_writers_do_not_deadlock_and_segregate():
    db = _fresh_db()
    assert _no_deadlock(lambda: db.record_equity(100, 100, 0)), "record_equity deadlock"
    assert _no_deadlock(lambda: db.open_position(
        strategy="T", symbol="BTCUSDT", side="long", qty=1,
        entry_price=100, max_hold_sec=60)), "open_position deadlock"
    # segregation: paper rows invisible in live mode and vice versa
    db.set_meta("mode", "live")
    assert db.open_positions() == []
    assert db.equity_series() == []
    assert _no_deadlock(lambda: db.record_equity(50, 50, 0))
    assert [r["equity"] for r in db.equity_series()] == [50]
    db.set_meta("mode", "paper")
    assert len(db.open_positions()) == 1
    assert [r["equity"] for r in db.equity_series()] == [100]

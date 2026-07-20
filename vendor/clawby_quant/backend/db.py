"""SQLite storage: factor values, signals, positions, trades, equity, meta."""
import json
import sqlite3
import threading
import time

from . import config

_lock = threading.Lock()
_conn = None


def init():
    global _conn
    _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    with _lock, _conn:
        _conn.executescript("""
        CREATE TABLE IF NOT EXISTS factor_values (
            factor TEXT NOT NULL, symbol TEXT NOT NULL DEFAULT '',
            ts INTEGER NOT NULL, value TEXT NOT NULL,
            PRIMARY KEY (factor, symbol)
        );
        CREATE TABLE IF NOT EXISTS factor_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            factor TEXT NOT NULL, symbol TEXT NOT NULL DEFAULT '',
            ts INTEGER NOT NULL, value TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_fh ON factor_history(factor, symbol, ts);
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL, strategy TEXT NOT NULL, symbol TEXT NOT NULL,
            side TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '',
            acted INTEGER NOT NULL DEFAULT 0, skip_reason TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
            qty REAL NOT NULL, entry_price REAL NOT NULL,
            stop_price REAL, take_profit REAL, trail_atr REAL,
            opened_at INTEGER NOT NULL, max_hold_sec INTEGER NOT NULL,
            time_exit_at INTEGER, extreme REAL,
            status TEXT NOT NULL DEFAULT 'open',
            closed_at INTEGER, close_price REAL, pnl REAL, close_reason TEXT
        );
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL, strategy TEXT NOT NULL, symbol TEXT NOT NULL,
            side TEXT NOT NULL, qty REAL NOT NULL, price REAL NOT NULL,
            fee REAL NOT NULL DEFAULT 0, kind TEXT NOT NULL,  -- open/close
            position_id INTEGER, pnl REAL, mode TEXT NOT NULL DEFAULT 'paper'
        );
        CREATE TABLE IF NOT EXISTS equity_curve (
            ts INTEGER NOT NULL, equity REAL NOT NULL, balance REAL NOT NULL,
            unrealized REAL NOT NULL DEFAULT 0,
            mode TEXT NOT NULL DEFAULT 'paper',
            PRIMARY KEY (ts, mode)
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL, level TEXT NOT NULL, msg TEXT NOT NULL
        );
        """)
        # additive migrations for pre-existing DBs
        for ddl in (
            "ALTER TABLE positions ADD COLUMN mfe_pct REAL NOT NULL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN mae_pct REAL NOT NULL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN entry_factors TEXT",
            "ALTER TABLE positions ADD COLUMN reason TEXT NOT NULL DEFAULT ''",
            # paper/live segregation: history predating the column is paper
            "ALTER TABLE positions ADD COLUMN mode TEXT NOT NULL DEFAULT 'paper'",
            "ALTER TABLE equity_curve ADD COLUMN mode TEXT NOT NULL DEFAULT 'paper'",
        ):
            try:
                _conn.execute(ddl)
            except sqlite3.OperationalError:
                pass  # column already exists
        # equity_curve pk migration: (ts) -> (ts, mode) so paper/live points
        # in the same minute never overwrite each other
        row = _conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='equity_curve'").fetchone()
        if row and "PRIMARY KEY (ts, mode)" not in row["sql"]:
            _conn.executescript("""
                ALTER TABLE equity_curve RENAME TO equity_curve_old;
                CREATE TABLE equity_curve (
                    ts INTEGER NOT NULL, equity REAL NOT NULL,
                    balance REAL NOT NULL, unrealized REAL NOT NULL DEFAULT 0,
                    mode TEXT NOT NULL DEFAULT 'paper', PRIMARY KEY (ts, mode));
                INSERT INTO equity_curve
                    SELECT ts, equity, balance, unrealized, mode FROM equity_curve_old;
                DROP TABLE equity_curve_old;
            """)


def current_mode():
    return get_meta("mode", "paper")


# -- factors ----------------------------------------------------------------

def set_factor(factor, symbol, value, keep_history=True):
    ts = int(time.time())
    payload = json.dumps(value, ensure_ascii=False)
    with _lock, _conn:
        _conn.execute("REPLACE INTO factor_values(factor,symbol,ts,value) VALUES(?,?,?,?)",
                      (factor, symbol or "", ts, payload))
        if keep_history:
            _conn.execute("INSERT INTO factor_history(factor,symbol,ts,value) VALUES(?,?,?,?)",
                          (factor, symbol or "", ts, payload))


def get_factor(factor, symbol="", max_age_sec=None):
    with _lock:
        row = _conn.execute("SELECT ts,value FROM factor_values WHERE factor=? AND symbol=?",
                            (factor, symbol or "")).fetchone()
    if not row:
        return None
    if max_age_sec and time.time() - row["ts"] > max_age_sec:
        return None
    return json.loads(row["value"])


def get_factor_ts(factor, symbol=""):
    with _lock:
        row = _conn.execute("SELECT ts FROM factor_values WHERE factor=? AND symbol=?",
                            (factor, symbol or "")).fetchone()
    return row["ts"] if row else 0


def factor_history(factor, symbol="", limit=200):
    with _lock:
        rows = _conn.execute(
            "SELECT ts,value FROM factor_history WHERE factor=? AND symbol=? "
            "ORDER BY ts DESC LIMIT ?", (factor, symbol or "", limit)).fetchall()
    return [{"ts": r["ts"], "value": json.loads(r["value"])} for r in reversed(rows)]


def all_factors_snapshot():
    with _lock:
        rows = _conn.execute("SELECT factor,symbol,ts,value FROM factor_values").fetchall()
    return [{"factor": r["factor"], "symbol": r["symbol"], "ts": r["ts"],
             "value": json.loads(r["value"])} for r in rows]


# -- signals ----------------------------------------------------------------

def log_signal(strategy, symbol, side, reason, acted, skip_reason=""):
    with _lock, _conn:
        cur = _conn.execute(
            "INSERT INTO signals(ts,strategy,symbol,side,reason,acted,skip_reason) "
            "VALUES(?,?,?,?,?,?,?)",
            (int(time.time()), strategy, symbol, side, reason[:400], int(acted), skip_reason[:200]))
    return cur.lastrowid


def recent_signals(limit=50):
    with _lock:
        rows = _conn.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def signal_count_since(strategy, symbol, since_ts):
    with _lock:
        row = _conn.execute(
            "SELECT COUNT(*) n FROM signals WHERE strategy=? AND symbol=? AND ts>=? AND acted=1",
            (strategy, symbol, since_ts)).fetchone()
    return row["n"]


# -- positions / trades -----------------------------------------------------

def open_position(**kw):
    mode = kw.get("mode") or current_mode()   # BEFORE _lock (non-reentrant)
    with _lock, _conn:
        cur = _conn.execute(
            "INSERT INTO positions(strategy,symbol,side,qty,entry_price,stop_price,"
            "take_profit,trail_atr,opened_at,max_hold_sec,time_exit_at,extreme,status,"
            "entry_factors,reason,mode) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,'open',?,?,?)",
            (kw["strategy"], kw["symbol"], kw["side"], kw["qty"], kw["entry_price"],
             kw.get("stop_price"), kw.get("take_profit"), kw.get("trail_atr"),
             int(time.time()), kw["max_hold_sec"], kw.get("time_exit_at"),
             kw["entry_price"], kw.get("entry_factors"), kw.get("reason", ""),
             mode))
    return cur.lastrowid


def get_position(pid):
    with _lock:
        row = _conn.execute("SELECT * FROM positions WHERE id=?", (pid,)).fetchone()
    return dict(row) if row else None


def open_positions(strategy=None, symbol=None, mode="auto"):
    """mode='auto' -> only the ACTIVE trading mode's positions (paper/live
    strictly segregated); mode=None -> all; or an explicit 'paper'/'live'."""
    q = "SELECT * FROM positions WHERE status='open'"
    args = []
    if mode == "auto":
        mode = current_mode()
    if mode:
        q += " AND mode=?"; args.append(mode)
    if strategy:
        q += " AND strategy=?"; args.append(strategy)
    if symbol:
        q += " AND symbol=?"; args.append(symbol)
    with _lock:
        rows = _conn.execute(q, args).fetchall()
    return [dict(r) for r in rows]


def update_position(pid, **fields):
    sets = ", ".join(f"{k}=?" for k in fields)
    with _lock, _conn:
        _conn.execute(f"UPDATE positions SET {sets} WHERE id=?", (*fields.values(), pid))


def close_position(pid, price, pnl, reason):
    with _lock, _conn:
        _conn.execute(
            "UPDATE positions SET status='closed', closed_at=?, close_price=?, pnl=?, "
            "close_reason=? WHERE id=?", (int(time.time()), price, pnl, reason, pid))


def log_trade(strategy, symbol, side, qty, price, fee, kind, position_id, pnl, mode):
    with _lock, _conn:
        _conn.execute(
            "INSERT INTO trades(ts,strategy,symbol,side,qty,price,fee,kind,position_id,pnl,mode) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (int(time.time()), strategy, symbol, side, qty, price, fee, kind,
             position_id, pnl, mode))


def recent_trades(limit=100, mode=None):
    q = "SELECT * FROM trades"
    args = []
    if mode == "auto":
        mode = current_mode()
    if mode:
        q += " WHERE mode=?"; args.append(mode)
    q += " ORDER BY id DESC LIMIT ?"; args.append(limit)
    with _lock:
        rows = _conn.execute(q, args).fetchall()
    return [dict(r) for r in rows]


def closed_positions_today(day_start_ts, mode="auto"):
    if mode == "auto":
        mode = current_mode()
    q = "SELECT * FROM positions WHERE status='closed' AND closed_at>=?"
    args = [day_start_ts]
    if mode:
        q += " AND mode=?"; args.append(mode)
    with _lock:
        rows = _conn.execute(q, args).fetchall()
    return [dict(r) for r in rows]


# -- equity / meta / logs ---------------------------------------------------

def record_equity(equity, balance, unrealized, mode=None):
    ts = int(time.time()) // 60 * 60  # minute buckets
    mode = mode or current_mode()     # BEFORE taking _lock (non-reentrant!)
    with _lock, _conn:
        _conn.execute(
            "REPLACE INTO equity_curve(ts,equity,balance,unrealized,mode) VALUES(?,?,?,?,?)",
            (ts, equity, balance, unrealized, mode))


def equity_series(limit=2880, mode="auto"):
    if mode == "auto":
        mode = current_mode()
    if mode:
        q = "SELECT * FROM equity_curve WHERE mode=? ORDER BY ts DESC LIMIT ?"
        args = (mode, limit)
    else:
        q = "SELECT * FROM equity_curve ORDER BY ts DESC LIMIT ?"
        args = (limit,)
    with _lock:
        rows = _conn.execute(q, args).fetchall()
    return [dict(r) for r in reversed(rows)]


def set_meta(key, value):
    with _lock, _conn:
        _conn.execute("REPLACE INTO meta(key,value) VALUES(?,?)", (key, str(value)))


def get_meta(key, default=""):
    with _lock:
        row = _conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def log(level, msg):
    with _lock, _conn:
        _conn.execute("INSERT INTO logs(ts,level,msg) VALUES(?,?,?)",
                      (int(time.time()), level, msg[:800]))


def recent_logs(limit=100):
    with _lock:
        rows = _conn.execute("SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def prune():
    cutoff = int(time.time()) - 14 * 86400
    with _lock, _conn:
        _conn.execute("DELETE FROM factor_history WHERE ts<?", (cutoff,))
        _conn.execute("DELETE FROM logs WHERE ts<?", (cutoff,))

"""从 SQLite 恢复当日 Live Gate 状态（跨 scan 持久）。"""

from __future__ import annotations

import sqlite3

from orb.ml.gate import LiveGateDayState
from orb.v2.config import OrbV2Config
from orb.v2.db import count_v2_opens_today, load_gate_day_meta, save_gate_day_meta


def load_gate_day_state(cur: sqlite3.Cursor, session_date: str, cfg: OrbV2Config) -> LiveGateDayState:
    """从 orb_v2_breakout_seen + orb_v2_gate_day 恢复当日 gate 状态。"""
    state = LiveGateDayState()
    state.opens = count_v2_opens_today(cur, session_date)
    meta = load_gate_day_meta(cur, session_date)
    state.scored_signals = int(meta.get("scored_signals") or 0)
    state.day_aborted = bool(meta.get("day_aborted"))
    state.recent_p = list(meta.get("recent_p") or [])
    return state


def persist_gate_day_state(cur: sqlite3.Cursor, session_date: str, state: LiveGateDayState) -> None:
    save_gate_day_meta(
        cur,
        session_date,
        scored_signals=state.scored_signals,
        recent_p=state.recent_p,
        day_aborted=state.day_aborted,
    )


def v2_session_traded(cur: sqlite3.Cursor, symbol: str, session_date: str, cfg: OrbV2Config) -> bool:
    """当日该标的是否已开仓（仅 opened=1；Gate 拒绝后可继续扫描）。"""
    from orb.v2.db import breakout_opened_today

    return breakout_opened_today(cur, symbol, session_date)

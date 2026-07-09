"""RTH / 会话辅助（vnpy lane 共用）。"""

from __future__ import annotations

import time
from typing import Optional

from quant.common.config import OrbConfig
from quant.common.session import is_trading_session, session_day_str


def _session_date_now(cfg: OrbConfig) -> str:
    return session_day_str(
        int(time.time() * 1000), tz=cfg.session_tz, session_open_time=cfg.session_open_time
    )


def in_regular_session(cfg: OrbConfig, *, now_ms: Optional[int] = None) -> bool:
    if not (cfg.session_open_time or "").strip():
        return True
    t = int(now_ms if now_ms is not None else time.time() * 1000)
    return is_trading_session(
        t,
        tz=cfg.session_tz,
        session_open_time=cfg.session_open_time,
        session_close_time=cfg.session_close_time,
        market=cfg.market,
    )

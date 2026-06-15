"""ORB 持仓结算：1m 前向 SL/TP。"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from orb.core.config import OrbConfig
from orb.core.session import effective_session_close_time, session_anchor_ms, session_close_ms


def bar_hit_long(
    h: float, l: float, sl: float, tp: Optional[float], *, same_bar_rule: str
) -> Tuple[Optional[str], float]:
    hit_sl = l <= sl
    hit_tp = tp is not None and h >= tp
    if hit_sl and hit_tp:
        return ("win", tp) if same_bar_rule == "optimistic" else ("loss", sl)
    if hit_sl:
        return "loss", sl
    if hit_tp:
        return "win", tp
    return None, 0.0


def bar_hit_short(
    h: float, l: float, sl: float, tp: Optional[float], *, same_bar_rule: str
) -> Tuple[Optional[str], float]:
    hit_sl = h >= sl
    hit_tp = tp is not None and l <= tp
    if hit_sl and hit_tp:
        return ("win", tp) if same_bar_rule == "optimistic" else ("loss", sl)
    if hit_sl:
        return "loss", sl
    if hit_tp:
        return "win", tp
    return None, 0.0


def pnl_r(side: str, entry: float, exit_px: float, sl: float) -> float:
    side_u = str(side).upper()
    if side_u == "LONG":
        risk = entry - sl
        return (exit_px - entry) / risk if risk > 0 else 0.0
    if side_u == "SHORT":
        risk = sl - entry
        return (entry - exit_px) / risk if risk > 0 else 0.0
    return 0.0


def pnl_usdt(side: str, entry: float, exit_px: float, notional: float) -> float:
    if entry <= 0 or notional <= 0:
        return 0.0
    side_u = str(side).upper()
    if side_u == "LONG":
        return notional * (exit_px - entry) / entry
    if side_u == "SHORT":
        return notional * (entry - exit_px) / entry
    return 0.0


def resolve_forward(
    df_kline: pd.DataFrame,
    *,
    entry: float,
    entry_bar_open_ms: int,
    side: str,
    sl: float,
    tp: Optional[float],
    hist_end_ms: int,
    bar_step_ms: int,
    cfg: OrbConfig,
) -> Tuple[Optional[str], float, str, int, Optional[int]]:
    start_ms = int(entry_bar_open_ms) + int(bar_step_ms)
    if start_ms > hist_end_ms:
        return None, float(entry), "start_after_hist_end", 0, None
    rule = cfg.same_bar_rule
    hold_ms = int(cfg.resolve_max_hold_ms)
    max_bars = int(cfg.resolve_max_bars)
    bar_ms = 60_000
    session_close_deadline = None
    close_time = effective_session_close_time(
        int(entry_bar_open_ms),
        tz=cfg.session_tz,
        session_open_time=cfg.session_open_time,
        session_close_time=cfg.session_close_time,
        market=cfg.market,
    )
    if cfg.resolve_at_session_close and close_time:
        anchor = session_anchor_ms(
            int(entry_bar_open_ms),
            tz=cfg.session_tz,
            session_open_time=cfg.session_open_time,
        )
        session_close_deadline = session_close_ms(
            anchor, tz=cfg.session_tz, session_close_time=close_time
        )
    deadline_ms = None if session_close_deadline else (int(entry_bar_open_ms) + hold_ms if hold_ms > 0 else None)
    early_ms = int(cfg.early_exit_minutes) * 60_000 if int(getattr(cfg, "early_exit_minutes", 0) or 0) > 0 else 0
    early_deadline = int(entry_bar_open_ms) + early_ms if early_ms > 0 else None
    early_checked = False
    outcome = None
    exit_px = float(entry)
    note = "resolved:auto"
    bars_seen = 0
    exit_bo = None
    sub = df_kline[df_kline["open_time"] >= start_ms].sort_values("open_time")
    for _, row in sub.iterrows():
        bo = int(row["open_time"])
        if bo > hist_end_ms:
            break
        h, low, c = float(row["high"]), float(row["low"]), float(row["close"])
        bars_seen += 1
        if side == "LONG":
            tag, px = bar_hit_long(h, low, sl, tp, same_bar_rule=rule)
        else:
            tag, px = bar_hit_short(h, low, sl, tp, same_bar_rule=rule)
        if tag == "win":
            outcome, exit_px, exit_bo = "win", float(px), bo
            break
        if tag == "loss":
            outcome, exit_px, exit_bo = "loss", float(px), bo
            break
        if early_deadline is not None and not early_checked and bo + bar_ms >= early_deadline:
            early_checked = True
            window = sub[(sub["open_time"] >= int(entry_bar_open_ms)) & (sub["open_time"] < early_deadline)]
            continued = False
            for _, wrow in window.iterrows():
                wh, wl = float(wrow["high"]), float(wrow["low"])
                if side == "LONG" and wh > float(entry):
                    continued = True
                    break
                if side == "SHORT" and wl < float(entry):
                    continued = True
                    break
            if not continued:
                outcome, exit_px, exit_bo, note = "early_exit", c, bo, "resolved:early_exit"
                break
        if session_close_deadline is not None and bo + bar_ms >= int(session_close_deadline):
            outcome, exit_px, exit_bo, note = "session_close", c, bo, "resolved:session_close"
            break
        if deadline_ms is not None and bo >= deadline_ms:
            outcome, exit_px, exit_bo, note = "expired", c, bo, "resolved:expired_time"
            break
        if session_close_deadline is None and max_bars > 0 and bars_seen >= max_bars:
            outcome, exit_px, exit_bo, note = "expired", c, bo, f"resolved:expired_{max_bars}bars"
            break
    if outcome is None:
        return None, float(entry), "no_touch", bars_seen, None
    return outcome, exit_px, note, bars_seen, exit_bo

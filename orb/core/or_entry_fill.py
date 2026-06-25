"""OR 入场成交模拟：Stop-Limit / 市价追单。"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from orb.core.config import OrbConfig
from orb.core.session import session_anchor_ms


def or_order_deadline_ms(*, or_end_ms: int, cfg: OrbConfig, session_close_ms: int) -> int:
    deadline = int(session_close_ms)
    if int(cfg.trade_window_minutes or 0) > 0:
        deadline = min(deadline, int(or_end_ms) + int(cfg.trade_window_minutes) * 60_000)
    return deadline


def bar_fills_stop_limit(*, side: str, entry_px: float, high: float, low: float) -> bool:
    side_u = str(side).upper()
    if side_u == "LONG":
        return high >= entry_px and low <= entry_px
    if side_u == "SHORT":
        return low <= entry_px and high >= entry_px
    return False


def find_or_stop_limit_fill(
    df1: pd.DataFrame,
    *,
    side: str,
    entry_px: float,
    after_ms: int,
    before_ms: int,
    gap_ok: bool = False,
    honest_fill: bool = False,
) -> Optional[tuple[int, float]]:
    """返回 (fill_bar_open_ms, fill_price)。"""
    if df1 is None or df1.empty or entry_px <= 0:
        return None
    side_u = str(side).upper()
    sub = df1[(df1["open_time"] > int(after_ms)) & (df1["open_time"] <= int(before_ms))]
    for _, row in sub.sort_values("open_time").iterrows():
        h, l, o = float(row["high"]), float(row["low"]), float(row["open"])
        if bar_fills_stop_limit(side=side_u, entry_px=entry_px, high=h, low=l):
            return int(row["open_time"]), float(entry_px)
        if gap_ok:
            if side_u == "LONG" and o >= entry_px:
                px = float(o) if honest_fill else float(entry_px)
                return int(row["open_time"]), px
            if side_u == "SHORT" and o <= entry_px:
                px = float(o) if honest_fill else float(entry_px)
                return int(row["open_time"]), px
    return None


def order_deadline_for_signal(
    *,
    scan_ms: int,
    cfg: OrbConfig,
    session_close_ms: int,
) -> int:
    anchor = session_anchor_ms(int(scan_ms), tz=cfg.session_tz, session_open_time=cfg.session_open_time)
    or_end_ms = anchor + max(1, int(cfg.or_minutes)) * 60_000
    return or_order_deadline_ms(or_end_ms=or_end_ms, cfg=cfg, session_close_ms=session_close_ms)


def resolve_entry_fill(
    *,
    mode: str,
    sym: str,
    sig: Any,
    session_date: str,
    scan_ms: int,
    df1: pd.DataFrame,
    df5: pd.DataFrame,
    close_ms: int,
    bar: int,
    cfg: OrbConfig,
    notional: float,
    wallet_before: float,
    robot_id: int,
    scans: Optional[list],
) -> tuple[Optional[Dict[str, Any]], str]:
    """返回 (trade_row, reason)。"""
    from dataclasses import replace

    from orb.ml.live_gate_sim import _resolve_trade_row

    mode_l = (mode or "signal").strip().lower()
    entry_bo = int(sig.entry_bar_open_ms or 0)
    signal_entry = float(sig.price)

    def _row_with_fill(sig_for_row: Any, fill_bo: int, fill_px: float, mode_name: str) -> tuple[Optional[Dict[str, Any]], str]:
        row = _resolve_trade_row(
            sym=sym,
            sig=sig_for_row,
            session_date=session_date,
            scan_ms=scan_ms,
            entry_bo=int(fill_bo),
            df1=df1,
            close_ms=close_ms,
            bar=bar,
            cfg=cfg,
            notional=notional,
            wallet_before=wallet_before,
            robot_id=robot_id,
            scans=scans,
        )
        if not row:
            return None, "no_trade_row"
        row["fill_bar_open_ms"] = int(fill_bo)
        row["entry_mode"] = mode_name
        row["signal_entry"] = signal_entry
        row["entry"] = float(fill_px)
        row["chase_slip"] = round(float(fill_px) - signal_entry, 6)
        return row, "ok"

    if mode_l == "signal":
        if entry_bo <= 0:
            return None, "no_entry_bar"
        row = _resolve_trade_row(
            sym=sym,
            sig=sig,
            session_date=session_date,
            scan_ms=scan_ms,
            entry_bo=entry_bo,
            df1=df1,
            close_ms=close_ms,
            bar=bar,
            cfg=cfg,
            notional=notional,
            wallet_before=wallet_before,
            robot_id=robot_id,
            scans=scans,
        )
        if row:
            row["entry_mode"] = "signal"
            row["signal_entry"] = signal_entry
        return (row, "ok") if row else (None, "no_trade_row")

    if mode_l in ("stoplimit", "stoplimit_gap", "stoplimit_honest", "stoplimit_gap_honest"):
        order_ms = int(scan_ms) + int(bar)
        deadline = order_deadline_for_signal(scan_ms=scan_ms, cfg=cfg, session_close_ms=int(close_ms))
        gap = mode_l in ("stoplimit_gap", "stoplimit_gap_honest")
        honest = mode_l in ("stoplimit_honest", "stoplimit_gap_honest")
        hit = find_or_stop_limit_fill(
            df1,
            side=str(sig.side),
            entry_px=signal_entry,
            after_ms=order_ms,
            before_ms=deadline,
            gap_ok=gap,
            honest_fill=honest,
        )
        if hit is None:
            return None, "or_limit_not_filled"
        fill_ms, fill_px = hit
        if honest or gap:
            fill_sig = replace(sig, price=round(fill_px, 8))
            return _row_with_fill(fill_sig, fill_ms, fill_px, mode_l)
        row, reason = _row_with_fill(sig, fill_ms, signal_entry, mode_l)
        return (row, reason) if row else (None, reason)

    if mode_l == "market":
        if entry_bo <= 0 or df5 is None or df5.empty:
            return None, "no_entry_bar"
        hit = df5[df5["open_time"] == entry_bo]
        if hit.empty:
            return None, "bar_not_found"
        chase_px = round(float(hit.iloc[-1]["close"]), 8)
        chase_sig = replace(sig, price=chase_px)
        row, reason = _row_with_fill(chase_sig, entry_bo, chase_px, "market_chase")
        return (row, reason) if row else (None, reason)

    return None, f"unknown_mode:{mode_l}"

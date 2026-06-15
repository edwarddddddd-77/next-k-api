#!/usr/bin/env python3
"""突破 horizon 标签：30min 内是否持住。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from orb.core.config import OrbConfig
from orb.core.kline_cache import load_klines
from orb.core.resolve import pnl_r, pnl_usdt, resolve_forward

HOLD_30M_MS = 30 * 60_000


def horizon_cfg(cfg: OrbConfig, *, hold_ms: int = HOLD_30M_MS) -> OrbConfig:
    return replace(
        cfg,
        resolve_at_session_close=False,
        resolve_max_hold_ms=int(hold_ms),
        resolve_max_bars=0,
    )


def resolve_horizon(
    df1: pd.DataFrame,
    *,
    entry: float,
    entry_bar_open_ms: int,
    side: str,
    sl: float,
    tp: Optional[float],
    cfg: OrbConfig,
    hold_ms: int = HOLD_30M_MS,
) -> Tuple[Optional[str], float, float, str]:
    """返回 (outcome, exit_px, pnl_r, note)。"""
    hcfg = horizon_cfg(cfg, hold_ms=hold_ms)
    end_ms = int(entry_bar_open_ms) + int(hold_ms) + 120_000
    out, ex_px, note, _, _ = resolve_forward(
        df1,
        entry=float(entry),
        entry_bar_open_ms=int(entry_bar_open_ms),
        side=str(side),
        sl=float(sl),
        tp=float(tp) if tp is not None else None,
        hist_end_ms=end_ms,
        bar_step_ms=hcfg.bar_step_ms(),
        cfg=hcfg,
    )
    if out is None:
        return None, float(entry), 0.0, note
    pr = pnl_r(str(side), float(entry), float(ex_px), float(sl))
    return str(out), float(ex_px), float(pr), str(note)


def label_hold_30m(outcome: Optional[str], pnl_r_val: float) -> int:
    if str(outcome or "").strip().lower() == "loss":
        return 0
    return 1 if float(pnl_r_val or 0.0) > 0.0 else 0


def relabel_row_horizon(
    row: Dict[str, Any],
    *,
    cfg: OrbConfig,
    hold_ms: int = HOLD_30M_MS,
) -> Dict[str, Any]:
    sym = str(row.get("symbol") or "").upper()
    bo = int(row.get("entry_bar_open_ms") or 0)
    if not sym or bo <= 0:
        return {}
    start = bo - 120_000
    end = bo + int(hold_ms) + 300_000
    df1 = load_klines(sym, "1m", start_ms=start, end_ms=end)
    if df1.empty:
        return {}
    out, ex_px, pr, note = resolve_horizon(
        df1,
        entry=float(row.get("entry") or 0),
        entry_bar_open_ms=bo,
        side=str(row.get("side") or ""),
        sl=float(row.get("sl") or 0),
        tp=row.get("tp"),
        cfg=cfg,
        hold_ms=hold_ms,
    )
    if out is None:
        return {}
    pu = pnl_usdt(str(row.get("side") or ""), float(row.get("entry") or 0), ex_px, 100.0)
    y = label_hold_30m(out, pr)
    return {
        "hold30_outcome": out,
        "hold30_exit_price": round(ex_px, 6),
        "hold30_pnl_r": round(pr, 6),
        "hold30_pnl_usdt": round(pu, 4),
        "hold30_true": y,
        "hold30_note": note,
    }

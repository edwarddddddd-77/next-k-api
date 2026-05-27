"""Moss 量化纸面 — 每 profile 单 symbol 单仓。"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from moss_quant import config as cfg
from moss_quant.core.decision import DecisionParams, compute_last_composite, compute_signals
from moss_quant.core.indicators import atr as compute_atr
from moss_quant.core.regime import classify_regime
from moss_quant.db import _utc_now, list_enabled_profiles
from moss_quant.kline_cache import load_cached
from moss_quant.params import cap_leverage_for_symbol, resolve_params_dict

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def compute_current_signal(df: pd.DataFrame, params: DecisionParams) -> int:
    regime = classify_regime(df, version=cfg.MOSS_QUANT_REGIME_VERSION)
    signals = compute_signals(df, params, regime)
    return int(signals.iloc[-1])


def check_exit(
    *,
    side: str,
    entry: float,
    mark: float,
    params: DecisionParams,
    df: pd.DataFrame,
    leverage: float,
) -> Optional[str]:
    if entry <= 0 or mark <= 0:
        return None
    side_u = side.upper()
    if side_u in ("LONG", "BUY"):
        pnl_pct = (mark - entry) / entry * leverage
    else:
        pnl_pct = (entry - mark) / entry * leverage

    atr_series = compute_atr(df, 14)
    atr_val = float(atr_series.iloc[-1])
    if np.isnan(atr_val) or atr_val <= 0:
        atr_val = mark * 0.02
    sl_dist = params.sl_atr_mult * atr_val / entry
    tp_dist = sl_dist * params.tp_rr_ratio
    if pnl_pct <= -sl_dist * leverage:
        return "stop_loss"
    if pnl_pct >= tp_dist * leverage:
        return "take_profit"
    sig = compute_current_signal(df, params)
    if side_u in ("LONG", "BUY") and sig == -1:
        return "signal_reverse"
    if side_u in ("SHORT", "SELL") and sig == 1:
        return "signal_reverse"
    return None


def pnl_usdt(side: str, entry: float, exit_px: float, notional: float) -> float:
    if entry <= 0 or notional <= 0:
        return 0.0
    if side.upper() in ("LONG", "BUY"):
        return notional * (exit_px - entry) / entry
    return notional * (entry - exit_px) / entry


def _effective_params(profile: Dict[str, Any]) -> dict:
    base = dict(profile["initial_params"])
    tactical = profile.get("tactical_params") or {}
    base.update(tactical)
    sym = str(profile.get("symbol") or "").upper()
    return cap_leverage_for_symbol(resolve_params_dict(base), sym)


def _notional(profile: Dict[str, Any], params: dict) -> float:
    equity = float(profile.get("virtual_equity_usdt") or cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    lev = min(float(params.get("base_leverage", 10)), float(params.get("max_leverage", 10)))
    risk = float(params.get("risk_per_trade", 0.1))
    max_pct = float(params.get("max_position_pct", 0.5))
    n = equity * risk * lev
    n = min(n, equity * max_pct * lev)
    return max(n, 10.0)


def run_paper_scan(conn: sqlite3.Connection) -> Dict[str, Any]:
    conn.row_factory = sqlite3.Row
    profiles = list_enabled_profiles(conn)
    stats: Dict[str, Any] = {
        "profiles_scanned": len(profiles),
        "opens": 0,
        "closes": 0,
        "details": [],
    }
    now = _utc_now()

    for profile in profiles:
        pid = int(profile["id"])
        symbol = str(profile["symbol"]).upper()
        params_d = _effective_params(profile)
        params = DecisionParams.from_dict(params_d)
        try:
            df = load_cached(symbol, refresh=True)
        except Exception as e:
            logger.warning("[moss] %s kline failed: %s", symbol, e)
            stats["details"].append({"profile_id": pid, "error": str(e)})
            continue

        mark = float(df["close"].iloc[-1])
        regime_s = classify_regime(df, version=cfg.MOSS_QUANT_REGIME_VERSION)
        regime_label = str(regime_s.iloc[-1]) if len(regime_s) else "SIDEWAYS"

        row = conn.execute(
            """SELECT * FROM moss_signals
               WHERE profile_id = ? AND outcome IS NULL AND side IN ('LONG','SHORT')
               LIMIT 1""",
            (pid,),
        ).fetchone()

        if row:
            side = str(row["side"])
            entry = float(row["entry_price"] or 0)
            notional = float(row["virtual_notional_usdt"] or 0)
            lev = float(params_d.get("base_leverage", 10))
            exit_rule = check_exit(
                side=side,
                entry=entry,
                mark=mark,
                params=params,
                df=df,
                leverage=lev,
            )
            if exit_rule:
                pnl = pnl_usdt(side, entry, mark, notional)
                outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "flat")
                conn.execute(
                    """UPDATE moss_signals SET outcome=?, outcome_at_utc=?, exit_price=?,
                       pnl_usdt=?, exit_rule=?, updated_at_utc=?, unrealized_pnl_usdt=0
                       WHERE id=?""",
                    (outcome, now, mark, pnl, exit_rule, now, row["id"]),
                )
                conn.execute(
                    """INSERT INTO moss_settlements(
                        settled_at_utc, signal_id, profile_id, symbol, side, outcome,
                        entry_price, exit_price, pnl_usdt, virtual_notional_usdt, exit_rule)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        now,
                        row["id"],
                        pid,
                        symbol,
                        side,
                        outcome,
                        entry,
                        mark,
                        pnl,
                        notional,
                        exit_rule,
                    ),
                )
                new_eq = float(profile["virtual_equity_usdt"]) + pnl
                conn.execute(
                    "UPDATE moss_profiles SET virtual_equity_usdt=?, updated_at_utc=? WHERE id=?",
                    (new_eq, now, pid),
                )
                stats["closes"] += 1
                stats["details"].append(
                    {"profile_id": pid, "action": "close", "rule": exit_rule, "pnl": pnl}
                )
            else:
                upnl = pnl_usdt(side, entry, mark, notional)
                conn.execute(
                    """UPDATE moss_signals SET mark_price=?, unrealized_pnl_usdt=?,
                       updated_at_utc=? WHERE id=?""",
                    (mark, upnl, now, row["id"]),
                )
                stats["details"].append({"profile_id": pid, "action": "hold", "upnl": upnl})
            continue

        sig = compute_current_signal(df, params)
        if sig == 0:
            stats["details"].append({"profile_id": pid, "action": "wait"})
            continue

        side = "LONG" if sig == 1 else "SHORT"
        notional = _notional(profile, params_d)
        regime_val = regime_label
        composite = compute_last_composite(df, params, regime_s)

        conn.execute(
            """INSERT INTO moss_signals(
                profile_id, recorded_at_utc, side, symbol, entry_price,
                virtual_notional_usdt, mark_price, composite, regime,
                unrealized_pnl_usdt, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,0,?)""",
            (
                pid,
                now,
                side,
                symbol,
                mark,
                notional,
                mark,
                composite,
                regime_val,
                now,
            ),
        )
        stats["opens"] += 1
        stats["details"].append(
            {"profile_id": pid, "action": "open", "side": side, "price": mark}
        )

    conn.execute(
        """INSERT INTO moss_paper_runs(ran_at_utc, profiles_scanned, opens, closes, detail_json)
           VALUES (?,?,?,?,?)""",
        (now, stats["profiles_scanned"], stats["opens"], stats["closes"], json.dumps(stats["details"])),
    )
    conn.commit()
    return stats

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


def _py_scalar(v: Any) -> Any:
    """numpy 标量 → Python float/int，便于 JSON 日志。"""
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _verbose() -> bool:
    return bool(getattr(cfg, "MOSS_QUANT_VERBOSE_LOG", True))


def _vlog(msg: str, *args: Any) -> None:
    if _verbose():
        logger.info("[moss] " + msg, *args)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _profile_label(profile: Dict[str, Any]) -> str:
    return "p%d:%s:%s" % (
        int(profile["id"]),
        str(profile.get("symbol") or "").upper(),
        str(profile.get("template") or ""),
    )


def format_scan_detail_message(label: str, detail: Dict[str, Any]) -> str:
    """与 Railway 日志同风格的单行摘要（供前端展示）。"""
    act = str(detail.get("action") or "")
    if act == "error":
        return f"[moss] {label} ERROR {detail.get('error', '')}"
    if act == "wait":
        return (
            f"[moss] {label} WAIT composite={detail.get('composite')} "
            f"thresh=±{detail.get('entry_threshold')} regime={detail.get('regime')} "
            f"reason={detail.get('reason')}"
        )
    if act == "hold":
        return (
            f"[moss] {label} HOLD {detail.get('side', '')} upnl={detail.get('upnl')}U "
            f"pnl%={detail.get('pnl_pct')} sl<={detail.get('sl_thresh_pct')}% "
            f"tp>={detail.get('tp_thresh_pct')}% sig={detail.get('signal')}"
        )
    if act == "close":
        return (
            f"[moss] {label} CLOSE {detail.get('side', '')} {detail.get('rule', '')} "
            f"pnl={detail.get('pnl')}U"
        )
    if act == "open":
        return (
            f"[moss] {label} OPEN {detail.get('side', '')} "
            f"notional={detail.get('notional')}U composite={detail.get('composite')} "
            f"regime={detail.get('regime', '')}"
        )
    return f"[moss] {label} {act}"


def _scan_detail(
    label: str, profile: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    d = {"profile_id": int(profile["id"]), "label": label, **payload}
    d["message"] = format_scan_detail_message(label, d)
    return d


def compute_current_signal(df: pd.DataFrame, params: DecisionParams) -> int:
    regime = classify_regime(df, version=cfg.MOSS_QUANT_REGIME_VERSION)
    signals = compute_signals(df, params, regime)
    return int(signals.iloc[-1])


def exit_snapshot(
    *,
    side: str,
    entry: float,
    mark: float,
    params: DecisionParams,
    df: pd.DataFrame,
    leverage: float,
) -> Dict[str, Any]:
    side_u = side.upper()
    if entry <= 0 or mark <= 0:
        return {"exit_rule": None}

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
    sl_thresh = -sl_dist * leverage
    tp_thresh = tp_dist * leverage
    sig = compute_current_signal(df, params)

    exit_rule: Optional[str] = None
    if pnl_pct <= sl_thresh:
        exit_rule = "stop_loss"
    elif pnl_pct >= tp_thresh:
        exit_rule = "take_profit"
    elif side_u in ("LONG", "BUY") and sig == -1:
        exit_rule = "signal_reverse"
    elif side_u in ("SHORT", "SELL") and sig == 1:
        exit_rule = "signal_reverse"

    return {
        "exit_rule": exit_rule,
        "pnl_pct": round(pnl_pct * 100, 3),
        "sl_thresh_pct": round(sl_thresh * 100, 3),
        "tp_thresh_pct": round(tp_thresh * 100, 3),
        "signal": sig,
        "atr": round(atr_val, 6),
    }


def check_exit(
    *,
    side: str,
    entry: float,
    mark: float,
    params: DecisionParams,
    df: pd.DataFrame,
    leverage: float,
) -> Optional[str]:
    return exit_snapshot(
        side=side,
        entry=entry,
        mark=mark,
        params=params,
        df=df,
        leverage=leverage,
    ).get("exit_rule")


def entry_snapshot(
    df: pd.DataFrame,
    params: DecisionParams,
    regime_s: pd.Series,
) -> Dict[str, Any]:
    sig = compute_current_signal(df, params)
    composite = compute_last_composite(df, params, regime_s)
    th = params.entry_threshold
    regime_label = str(regime_s.iloc[-1]) if len(regime_s) else "SIDEWAYS"
    if sig == 1:
        reason = "signal_long"
    elif sig == -1:
        reason = "signal_short"
    elif abs(composite) <= th:
        reason = "composite_below_threshold"
    else:
        reason = "no_discrete_signal"
    return {
        "signal": int(sig),
        "composite": round(float(composite), 4),
        "entry_threshold": round(float(_py_scalar(th)), 4),
        "regime": regime_label,
        "reason": reason,
        "bars": int(len(df)),
    }


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

    logger.info(
        "[moss] paper scan start enabled=%s verbose=%s",
        len(profiles),
        _verbose(),
    )
    if not profiles:
        logger.info("[moss] paper scan skip: no enabled profiles")
        conn.execute(
            """INSERT INTO moss_paper_runs(ran_at_utc, profiles_scanned, opens, closes, detail_json)
               VALUES (?,?,?,?,?)""",
            (now, 0, 0, 0, "[]"),
        )
        conn.commit()
        return stats

    for profile in profiles:
        pid = int(profile["id"])
        symbol = str(profile["symbol"]).upper()
        label = _profile_label(profile)
        params_d = _effective_params(profile)
        params = DecisionParams.from_dict(params_d)
        lev = float(params_d.get("base_leverage", 10))

        try:
            # HL：缓存过期时 load_cached 会自动 ccxt 更新，无需每次 refresh=True
            df = load_cached(symbol, refresh=False)
        except Exception as e:
            logger.warning("[moss] %s kline failed: %s", label, e)
            stats["details"].append(
                _scan_detail(label, profile, {"symbol": symbol, "action": "error", "error": str(e)})
            )
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
            snap = exit_snapshot(
                side=side,
                entry=entry,
                mark=mark,
                params=params,
                df=df,
                leverage=lev,
            )
            exit_rule = snap.get("exit_rule")
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
                    _scan_detail(
                        label,
                        profile,
                        {
                            "symbol": symbol,
                            "action": "close",
                            "side": side,
                            "rule": exit_rule,
                            "pnl": round(pnl, 4),
                        },
                    )
                )
                logger.info(
                    "[moss] %s CLOSE %s %s pnl=%.4fU mark=%.6g entry=%.6g",
                    label,
                    side,
                    exit_rule,
                    pnl,
                    mark,
                    entry,
                )
                _vlog("%s exit_diag %s", label, snap)
            else:
                upnl = pnl_usdt(side, entry, mark, notional)
                conn.execute(
                    """UPDATE moss_signals SET mark_price=?, unrealized_pnl_usdt=?,
                       updated_at_utc=? WHERE id=?""",
                    (mark, upnl, now, row["id"]),
                )
                stats["details"].append(
                    _scan_detail(
                        label,
                        profile,
                        {
                            "symbol": symbol,
                            "action": "hold",
                            "side": side,
                            "upnl": round(upnl, 4),
                            "pnl_pct": snap.get("pnl_pct"),
                            "sl_thresh_pct": snap.get("sl_thresh_pct"),
                            "tp_thresh_pct": snap.get("tp_thresh_pct"),
                            "signal": snap.get("signal"),
                        },
                    )
                )
                logger.info(
                    "[moss] %s HOLD %s upnl=%.4fU pnl%%=%s sl<=%s%% tp>=%s%% sig=%s",
                    label,
                    side,
                    upnl,
                    snap.get("pnl_pct"),
                    snap.get("sl_thresh_pct"),
                    snap.get("tp_thresh_pct"),
                    snap.get("signal"),
                )
                _vlog("%s hold_diag %s", label, snap)
            continue

        ent = entry_snapshot(df, params, regime_s)
        if ent["signal"] == 0:
            stats["details"].append(
                _scan_detail(
                    label,
                    profile,
                    {
                        "symbol": symbol,
                        "action": "wait",
                        "composite": ent["composite"],
                        "entry_threshold": ent["entry_threshold"],
                        "reason": ent["reason"],
                        "regime": regime_label,
                    },
                )
            )
            logger.info(
                "[moss] %s WAIT composite=%s thresh=±%s regime=%s reason=%s mark=%.6g",
                label,
                ent["composite"],
                ent["entry_threshold"],
                regime_label,
                ent["reason"],
                mark,
            )
            continue

        side = "LONG" if ent["signal"] == 1 else "SHORT"
        notional = _notional(profile, params_d)
        composite = ent["composite"]

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
                regime_label,
                now,
            ),
        )
        stats["opens"] += 1
        stats["details"].append(
            _scan_detail(
                label,
                profile,
                {
                    "symbol": symbol,
                    "action": "open",
                    "side": side,
                    "notional": round(notional, 2),
                    "composite": composite,
                    "regime": regime_label,
                },
            )
        )
        logger.info(
            "[moss] %s OPEN %s notional=%.2fU composite=%s regime=%s mark=%.6g",
            label,
            side,
            notional,
            composite,
            regime_label,
            mark,
        )

    detail_json = json.dumps(stats["details"], ensure_ascii=False, default=_py_scalar)
    conn.execute(
        """INSERT INTO moss_paper_runs(ran_at_utc, profiles_scanned, opens, closes, detail_json)
           VALUES (?,?,?,?,?)""",
        (now, stats["profiles_scanned"], stats["opens"], stats["closes"], detail_json),
    )
    conn.commit()

    logger.info(
        "[moss] paper scan done profiles=%s opens=%s closes=%s",
        stats["profiles_scanned"],
        stats["opens"],
        stats["closes"],
    )
    if _verbose() and stats["details"]:
        logger.info("[moss] paper details %s", detail_json)

    return stats

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
from moss_quant.db import _utc_now, list_profiles_for_paper_scan
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
            f"[moss] {label} HOLD {detail.get('side', '')} "
            f"entry={detail.get('entry_price')} mark={detail.get('mark_price')} "
            f"upnl={detail.get('upnl')}U pnl%={detail.get('pnl_pct')} "
            f"sl<={detail.get('sl_thresh_pct')}% tp>={detail.get('tp_thresh_pct')}% "
            f"sig={detail.get('signal')}"
        )
    if act == "close":
        return (
            f"[moss] {label} CLOSE {detail.get('side', '')} {detail.get('rule', '')} "
            f"pnl={detail.get('pnl')}U"
        )
    if act == "open":
        return (
            f"[moss] {label} OPEN {detail.get('side', '')} "
            f"entry={detail.get('entry_price')} notional={detail.get('notional')}U "
            f"upnl={detail.get('upnl', 0)}U composite={detail.get('composite')} "
            f"regime={detail.get('regime', '')}"
        )
    return f"[moss] {label} {act}"


def _scan_detail(
    label: str, profile: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    d = {"profile_id": int(profile["id"]), "label": label, **payload}
    d["message"] = format_scan_detail_message(label, d)
    return d


def _margin_pnl_pct(side: str, entry: float, mark: float, leverage: float) -> float:
    """相对保证金的收益率 %（与 hold 日志 pnl% 一致）。"""
    if entry <= 0 or mark <= 0 or leverage <= 0:
        return 0.0
    side_u = side.upper()
    if side_u in ("LONG", "BUY"):
        return (mark - entry) / entry * leverage * 100.0
    return (entry - mark) / entry * leverage * 100.0


def _position_fields(
    *,
    side: str,
    entry: float,
    mark: float,
    notional: float,
    upnl: float,
    leverage: float = 10.0,
) -> Dict[str, Any]:
    return {
        "side": side,
        "entry_price": round(entry, 8),
        "mark_price": round(mark, 8),
        "notional": round(notional, 2),
        "upnl": round(upnl, 4),
        "leverage": round(float(leverage), 2),
        "pnl_pct": round(_margin_pnl_pct(side, entry, mark, leverage), 3),
    }


def fetch_open_positions_map(conn: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
    """profile_id → 当前未平仓纸面单（每 profile 仅最新一条）。"""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT s.profile_id, s.side, s.symbol, s.entry_price, s.mark_price,
                  s.unrealized_pnl_usdt, s.virtual_notional_usdt
           FROM moss_signals s
           INNER JOIN (
               SELECT profile_id, MAX(id) AS max_id
               FROM moss_signals
               WHERE outcome IS NULL AND side IN ('LONG','SHORT')
               GROUP BY profile_id
           ) t ON s.id = t.max_id"""
    ).fetchall()
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        pid = int(row["profile_id"])
        entry = float(row["entry_price"] or 0)
        mark = float(row["mark_price"] or 0)
        notional = float(row["virtual_notional_usdt"] or 0)
        side = str(row["side"])
        upnl = float(row["unrealized_pnl_usdt"] or 0)
        if mark > 0 and entry > 0 and notional > 0:
            upnl = pnl_usdt(side, entry, mark, notional)
        out[pid] = {
            "profile_id": pid,
            "side": side,
            "symbol": str(row["symbol"] or "").upper(),
            "entry_price": round(entry, 8),
            "mark_price": round(mark, 8),
            "notional": round(notional, 2),
            "upnl": round(upnl, 4),
        }
    return out


def refresh_open_map_marks(
    open_map: Dict[int, Dict[str, Any]],
    conn: Optional[sqlite3.Connection] = None,
    *,
    persist: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """拉最新 K 线收盘价刷新现价、浮盈；可选写回 moss_signals。"""
    from moss_quant.db import get_profile

    now = _utc_now()
    for pid, pos in open_map.items():
        sym = str(pos.get("symbol") or "").upper()
        if not sym:
            continue
        lev = float(pos.get("leverage") or 10.0)
        if conn is not None:
            prof = get_profile(conn, int(pid))
            if prof:
                lev = float(_effective_params(prof).get("base_leverage", lev))
        try:
            df = load_cached(sym, refresh=True)
            mark = float(df["close"].iloc[-1])
        except Exception as e:
            logger.warning("[moss] refresh mark %s failed: %s", sym, e)
            continue
        entry = float(pos.get("entry_price") or 0)
        notional = float(pos.get("notional") or 0)
        side = str(pos.get("side") or "")
        if mark <= 0 or entry <= 0 or notional <= 0:
            continue
        upnl = round(pnl_usdt(side, entry, mark, notional), 4)
        pos.update(
            {
                "leverage": round(lev, 2),
                "mark_price": round(mark, 8),
                "upnl": upnl,
                "pnl_pct": round(_margin_pnl_pct(side, entry, mark, lev), 3),
            }
        )
        if persist and conn is not None:
            conn.execute(
                """UPDATE moss_signals SET mark_price=?, unrealized_pnl_usdt=?,
                   updated_at_utc=?
                   WHERE profile_id=? AND outcome IS NULL AND side IN ('LONG','SHORT')""",
                (pos["mark_price"], upnl, now, int(pid)),
            )
    if persist and conn is not None:
        conn.commit()
    return open_map


def append_missing_open_position_details(
    conn: sqlite3.Connection,
    details: List[Dict[str, Any]],
    open_map: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """上次扫描未包含的持仓（如当次 K 线失败）补一条 hold 明细。"""
    if not open_map:
        return details
    from moss_quant.db import get_profile

    seen = {
        int(d["profile_id"])
        for d in details
        if isinstance(d, dict) and d.get("profile_id") is not None
    }
    out = list(details)
    for pid, pos in open_map.items():
        if pid in seen:
            continue
        prof = get_profile(conn, int(pid))
        if not prof:
            continue
        label = _profile_label(prof)
        out.append(
            _scan_detail(
                label,
                prof,
                {
                    "symbol": pos.get("symbol") or prof.get("symbol"),
                    "action": "hold",
                    "template": prof.get("template"),
                    **_position_fields(
                        side=str(pos["side"]),
                        entry=float(pos["entry_price"]),
                        mark=float(pos["mark_price"]),
                        notional=float(pos["notional"]),
                        upnl=float(pos["upnl"]),
                    ),
                },
            )
        )
    return out


def enrich_scan_details_with_positions(
    details: List[Dict[str, Any]],
    open_map: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """用库内最新持仓覆盖/补全扫描明细中的开仓价与浮盈。"""
    if not open_map:
        return details
    enriched: List[Dict[str, Any]] = []
    for d in details:
        if not isinstance(d, dict):
            enriched.append(d)
            continue
        row = dict(d)
        pid = row.get("profile_id")
        if pid is None:
            enriched.append(row)
            continue
        pos = open_map.get(int(pid))
        if not pos:
            enriched.append(row)
            continue
        act = str(row.get("action") or "").lower()
        if act not in ("hold", "open", "close"):
            row["action"] = "hold"
        row.update(
            {
                "side": pos.get("side") or row.get("side"),
                "entry_price": pos["entry_price"],
                "mark_price": pos["mark_price"],
                "notional": pos.get("notional", row.get("notional")),
                "upnl": pos["upnl"],
                "leverage": pos.get("leverage", row.get("leverage")),
                "pnl_pct": pos.get("pnl_pct", row.get("pnl_pct")),
            }
        )
        label = str(
            row.get("label")
            or ("p%s:%s" % (pid, pos.get("symbol") or row.get("symbol") or ""))
        )
        row["label"] = label
        row["message"] = format_scan_detail_message(label, row)
        enriched.append(row)
    return enriched


def refresh_live_open_signals(conn: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
    """刷新所有未平仓纸面单的标记价与浮盈并写库。"""
    return refresh_open_map_marks(
        fetch_open_positions_map(conn), conn, persist=True
    )


def serialize_signal_rows(
    conn: sqlite3.Connection, rows: List[Any]
) -> List[Dict[str, Any]]:
    """将 moss_signals 行转为 API dict，并为持仓补充 leverage / pnl_pct。"""
    from moss_quant.db import get_profile

    out: List[Dict[str, Any]] = []
    prof_cache: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        d = dict(row)
        if d.get("outcome"):
            out.append(d)
            continue
        side = str(d.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            out.append(d)
            continue
        pid = int(d["profile_id"])
        if pid not in prof_cache:
            prof_cache[pid] = get_profile(conn, pid) or {}
        lev = 10.0
        if prof_cache[pid]:
            lev = float(_effective_params(prof_cache[pid]).get("base_leverage", 10))
        entry = float(d.get("entry_price") or 0)
        mark = float(d.get("mark_price") or 0)
        d["leverage"] = round(lev, 2)
        d["pnl_pct"] = round(_margin_pnl_pct(side, entry, mark, lev), 3)
        out.append(d)
    return out


def scan_detail_lines(details: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for d in details:
        if not isinstance(d, dict):
            continue
        label = str(
            d.get("label")
            or ("p%s:%s" % (d.get("profile_id", "?"), d.get("symbol", "")))
        )
        lines.append(format_scan_detail_message(label, d))
    return lines


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
    profiles = list_profiles_for_paper_scan(conn)
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
                            "template": profile.get("template"),
                            "sl_thresh_pct": snap.get("sl_thresh_pct"),
                            "tp_thresh_pct": snap.get("tp_thresh_pct"),
                            "signal": snap.get("signal"),
                            **_position_fields(
                                side=side,
                                entry=entry,
                                mark=mark,
                                notional=notional,
                                upnl=upnl,
                                leverage=lev,
                            ),
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
                    "template": profile.get("template"),
                    "composite": composite,
                    "regime": regime_label,
                    **_position_fields(
                        side=side,
                        entry=mark,
                        mark=mark,
                        notional=notional,
                        upnl=0.0,
                        leverage=lev,
                    ),
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

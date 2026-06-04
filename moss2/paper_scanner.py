"""Moss2 纸面/实盘：factory 信号 + discipline + Protocol。"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from moss2 import config as cfg
from moss2.dataset import resolve_csv_path
from moss2.db import _utc_now, list_profiles_for_paper_scan, refresh_open_signal_marks
from moss2.discipline.entry_quality import evaluate_open_signal
from moss2.discipline.gates import check_open_gate, regime_notional_scale
from moss2.kline_loader import load_market_df
from moss2.params import merge_profile_params
from moss2.paper_wallet import paper_source_of_truth, pnl_usdt
from moss2.protocol_result import protocol_ingest_close_result, protocol_ingest_open_result
from moss2.versioning import canary_scale, effective_version

logger = logging.getLogger(__name__)


def _variant_modules(variant: str):
    if str(variant).lower() == "en":
        from moss2.variants.en.core.decision import (
            DecisionParams,
            compute_last_composite,
            compute_signals,
        )
        from moss2.variants.en.core.indicators import atr as compute_atr
        from moss2.variants.en.core.regime import classify_regime

        cap = lambda p, s: p
    else:
        from moss2.variants.hl.core.decision import (
            DecisionParams,
            compute_last_composite,
            compute_signals,
        )
        from moss2.variants.hl.core.indicators import atr as compute_atr
        from moss2.variants.hl.core.regime import classify_regime
        from moss2.variants.hl.core.leverage_caps import cap_params_for_symbol

        cap = cap_params_for_symbol
    return DecisionParams, compute_signals, compute_atr, classify_regime, cap, compute_last_composite


def compute_current_signal(
    df: pd.DataFrame, params_dict: dict, variant: str
) -> Tuple[int, float, str]:
    (
        DecisionParams,
        compute_signals,
        _,
        classify_regime,
        cap,
        compute_last_composite,
    ) = _variant_modules(variant)
    sym = str(params_dict.get("_symbol") or "")
    clean = {k: v for k, v in params_dict.items() if not str(k).startswith("_")}
    clean = cap(dict(clean), sym)
    params = DecisionParams.from_dict(clean)
    regime = classify_regime(df, version=cfg.MOSS2_REGIME_VERSION)
    signals = compute_signals(df, params, regime)
    sig = int(signals.iloc[-1])
    comp = float(compute_last_composite(df, params, regime))
    regime_label = str(regime.iloc[-1]) if len(regime) else "SIDEWAYS"
    return sig, comp, regime_label


def _entry_threshold(params_dict: dict, variant: str) -> float:
    DecisionParams, _, _, _, cap, _ = _variant_modules(variant)
    clean = {k: v for k, v in params_dict.items() if not str(k).startswith("_")}
    clean = cap(dict(clean), str(params_dict.get("_symbol") or ""))
    return float(DecisionParams.from_dict(clean).entry_threshold)


def _check_exit(
    *,
    side: str,
    entry: float,
    mark: float,
    params_dict: dict,
    df: pd.DataFrame,
    variant: str,
    prev_regime: str,
    current_regime: str,
) -> Optional[str]:
    DecisionParams, compute_signals, compute_atr, classify_regime, cap, _ = (
        _variant_modules(variant)
    )
    clean = {k: v for k, v in params_dict.items() if not str(k).startswith("_")}
    params = DecisionParams.from_dict(
        cap(dict(clean), str(params_dict.get("_symbol") or ""))
    )
    side_u = side.upper()
    lev = min(float(params.base_leverage), float(params.max_leverage))
    if side_u in ("LONG", "BUY"):
        pnl_pct = (mark - entry) / entry * lev
    else:
        pnl_pct = (entry - mark) / entry * lev
    atr_series = compute_atr(df, 14)
    atr_val = float(atr_series.iloc[-1])
    if np.isnan(atr_val) or atr_val <= 0:
        atr_val = mark * 0.02
    sl_dist = params.sl_atr_mult * atr_val / entry
    tp_dist = sl_dist * params.tp_rr_ratio
    sl_thresh = -sl_dist * lev
    tp_thresh = tp_dist * lev
    regime = classify_regime(df, version=cfg.MOSS2_REGIME_VERSION)
    sig = int(compute_signals(df, params, regime).iloc[-1])
    if pnl_pct <= sl_thresh:
        return "stop_loss"
    if pnl_pct >= tp_thresh:
        return "take_profit"
    if params.exit_on_regime_change and prev_regime and current_regime != prev_regime:
        return "regime_change"
    if sig == 1 and side_u in ("SHORT", "SELL"):
        return "signal_reverse"
    if sig == -1 and side_u in ("LONG", "BUY"):
        return "signal_reverse"
    return None


def _signal_row_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "side": row["side"],
        "entry_price": row["entry_price"],
        "virtual_notional_usdt": row["virtual_notional_usdt"],
        "meta_json": row["meta_json"],
    }


def _wait_detail(
    label: str,
    composite: float,
    regime_label: str,
    params_d: dict,
    variant: str,
    extra: Optional[dict] = None,
) -> dict:
    from moss2.discipline.entry_quality import effective_entry_threshold

    eth = _entry_threshold(params_d, variant)
    th = effective_entry_threshold(eth)
    eff = float((extra or {}).get("entry_threshold_eff") or 0)
    if eff <= 0 and cfg.MOSS2_ENTRY_QUALITY_ENABLED:
        eff = th + float(cfg.MOSS2_ENTRY_MARGIN or 0)
    d: Dict[str, Any] = {
        "label": label,
        "action": "wait",
        "composite": round(composite, 4),
        "regime": regime_label,
        "entry_threshold": round(th, 4),
    }
    if cfg.MOSS2_PAPER_LOG_MARGIN:
        ref = eff if cfg.MOSS2_ENTRY_QUALITY_ENABLED else th
        d["margin"] = round(abs(composite) - ref, 4)
        if cfg.MOSS2_ENTRY_QUALITY_ENABLED:
            d["entry_threshold_eff"] = round(ref, 4)
    if extra:
        d.update(extra)
    return d


def run_paper_scan(conn) -> Dict[str, Any]:
    from moss2.db import insert_paper_run
    from moss2.signal_sender import is_real_mode, send_close, send_open

    conn.row_factory = sqlite3.Row
    profiles = list_profiles_for_paper_scan(conn)
    stats: Dict[str, Any] = {
        "lane": "moss2",
        "real_mode": is_real_mode(),
        "profiles_scanned": len(profiles),
        "opens": 0,
        "closes": 0,
        "protocol_opens": 0,
        "protocol_closes": 0,
        "details": [],
    }
    now = _utc_now()

    for profile in profiles:
        pid = profile["id"]
        symbol = profile["symbol"]
        variant = str(profile.get("variant") or cfg.MOSS2_OPS_VARIANT)
        pver = effective_version(profile)
        label = f"m2:{pid}:{symbol}:{variant}:{pver}"
        row = conn.execute(
            """SELECT * FROM moss2_signals
               WHERE profile_id=? AND outcome IS NULL
               AND side IN ('LONG','SHORT') LIMIT 1""",
            (pid,),
        ).fetchone()
        ops_ok = cfg.is_ops_variant(variant)
        if not ops_ok and not row:
            stats["details"].append(
                {
                    "label": label,
                    "action": "skip",
                    "reason": f"variant_{variant}_disabled",
                }
            )
            continue
        try:
            df = load_market_df(symbol, variant, limit=cfg.MOSS2_KLINE_LIMIT)
        except FileNotFoundError as e:
            stats["details"].append(
                {
                    "label": label,
                    "action": "error",
                    "reason": str(e),
                    "data_csv": str(resolve_csv_path(symbol, variant) or ""),
                }
            )
            continue
        mark = float(df["close"].iloc[-1])
        params_d = merge_profile_params(profile)
        params_d["_symbol"] = symbol
        params_d["_template"] = profile.get("template") or "balanced"
        eth = _entry_threshold(params_d, variant)

        if row:
            sig_row = _signal_row_dict(row)
            side = str(sig_row["side"])
            entry = float(sig_row["entry_price"] or mark)
            meta = json.loads(sig_row["meta_json"] or "{}") if sig_row["meta_json"] else {}
            prev_regime = str(meta.get("regime") or "")
            _sig, composite, regime_label = compute_current_signal(df, params_d, variant)
            refresh_open_signal_marks(
                conn, profile_id=pid, mark=mark, composite=composite
            )
            rule = _check_exit(
                side=side,
                entry=entry,
                mark=mark,
                params_dict=params_d,
                df=df,
                variant=variant,
                prev_regime=prev_regime,
                current_regime=regime_label,
            )
            if rule:
                notional = float(sig_row["virtual_notional_usdt"] or 0)
                pnl = round(pnl_usdt(side, entry, mark, notional), 4)
                outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "flat")
                proto_close_err = None
                if is_real_mode():
                    resp = send_close(
                        symbol=symbol,
                        side=side,
                        exit_rule=rule,
                        close_price=mark,
                        profile_id=pid,
                        params_version=pver,
                    )
                    close_result = protocol_ingest_close_result(resp)
                    if close_result.ok:
                        stats["protocol_closes"] += 1
                    else:
                        proto_close_err = close_result.error
                        logger.error(
                            "[moss2] %s protocol close notify failed: %s",
                            label,
                            proto_close_err,
                        )
                conn.execute(
                    """UPDATE moss2_signals SET outcome=?, outcome_at_utc=?,
                       exit_price=?, pnl_usdt=?, exit_rule=?, unrealized_pnl_usdt=0,
                       updated_at_utc=? WHERE id=?""",
                    (outcome, now, mark, pnl, rule, now, sig_row["id"]),
                )
                from moss2.db import insert_moss2_settlement, sync_moss2_wallet_from_settlements

                insert_moss2_settlement(
                    conn,
                    signal_id=int(sig_row["id"]),
                    profile_id=pid,
                    symbol=symbol,
                    side=side,
                    outcome=outcome,
                    entry_price=entry,
                    exit_price=mark,
                    pnl_usdt=pnl,
                    notional=notional,
                    exit_rule=rule,
                    settled_at=now,
                )
                sync_moss2_wallet_from_settlements(conn)
                conn.commit()
                stats["closes"] += 1
                close_d: Dict[str, Any] = {
                    "label": label,
                    "action": "close",
                    "rule": rule,
                    "pnl_usdt": pnl,
                }
                if proto_close_err:
                    close_d["protocol_error"] = proto_close_err
                stats["details"].append(close_d)
            else:
                upnl = round(pnl_usdt(side, entry, mark, float(sig_row["virtual_notional_usdt"] or 0)), 4)
                conn.execute(
                    """UPDATE moss2_signals SET mark_price=?, unrealized_pnl_usdt=?,
                       updated_at_utc=? WHERE id=?""",
                    (mark, upnl, now, sig_row["id"]),
                )
                conn.commit()
                hold_d: Dict[str, Any] = {
                    "label": label,
                    "action": "hold",
                    "upnl": upnl,
                }
                if cfg.MOSS2_PAPER_LOG_MARGIN:
                    from moss2.discipline.entry_quality import effective_entry_threshold

                    hold_th = effective_entry_threshold(eth)
                    hold_ref = (
                        hold_th + float(cfg.MOSS2_ENTRY_MARGIN or 0)
                        if cfg.MOSS2_ENTRY_QUALITY_ENABLED
                        else hold_th
                    )
                    hold_d["composite"] = round(composite, 4)
                    hold_d["margin"] = round(abs(composite) - hold_ref, 4)
                stats["details"].append(hold_d)
            continue

        if not ops_ok:
            continue

        open_lane = int(
            conn.execute(
                """SELECT COUNT(*) FROM moss2_signals
                   WHERE outcome IS NULL AND side IN ('LONG','SHORT')"""
            ).fetchone()[0]
            or 0
        )
        if open_lane >= int(cfg.MOSS2_PORTFOLIO_MAX_OPEN_POSITIONS):
            stats["details"].append(
                {
                    "label": label,
                    "action": "wait",
                    "reason": "portfolio_max_open_positions",
                    "open_count": open_lane,
                    "cap": int(cfg.MOSS2_PORTFOLIO_MAX_OPEN_POSITIONS),
                }
            )
            continue

        ev = evaluate_open_signal(
            df, params_d, variant, entry_threshold=eth
        )
        sig = int(ev["signal"])
        composite = float(ev["composite"])
        regime_label = str(ev["regime"])
        if sig == 0:
            extra_wait = {
                "gate": ev.get("reason"),
                "entry_threshold_eff": ev.get("entry_threshold_eff"),
                "confirm_bars": ev.get("confirm_bars"),
            }
            stats["details"].append(
                _wait_detail(
                    label, composite, regime_label, params_d, variant, extra_wait
                )
            )
            continue

        scan_th = float(ev.get("entry_threshold") or eth)
        allowed, gate_reason, gate_dbg = check_open_gate(
            conn, pid, composite=composite, entry_threshold=scan_th
        )
        if not allowed:
            stats["details"].append(
                _wait_detail(
                    label,
                    composite,
                    regime_label,
                    params_d,
                    variant,
                    {"gate": gate_reason, **gate_dbg},
                )
            )
            continue

        side = "LONG" if sig == 1 else "SHORT"
        base_notional = float(
            profile.get("virtual_equity_usdt") or cfg.MOSS2_PROFILE_CAPITAL
        )
        scale = regime_notional_scale(regime_label) * canary_scale(profile)
        notional = base_notional * scale
        lev = min(
            float(params_d.get("base_leverage", 10)),
            float(params_d.get("max_leverage", 10)),
        )
        margin_usdt = notional / lev if lev > 0 else notional / 10.0

        meta = {
            "regime": regime_label,
            "variant": variant,
            "params_version": pver,
            "notional_scale": scale,
            "lane": "moss2",
        }
        entry_px = mark
        protocol_ok = False
        proto_open_err = None
        if is_real_mode():
            resp = send_open(
                symbol=symbol,
                side=side,
                entry_price=mark,
                margin_usdt=margin_usdt,
                leverage=lev,
                profile_id=pid,
                params_version=pver,
                composite=composite,
                regime=regime_label,
            )
            open_result = protocol_ingest_open_result(resp)
            if open_result.ok:
                protocol_ok = True
                stats["protocol_opens"] += 1
                entry_px = open_result.entry_price or mark
                if open_result.client_ref:
                    meta["protocol_client_ref"] = open_result.client_ref
            else:
                proto_open_err = open_result.error
                logger.error("[moss2] %s protocol open failed: %s", label, proto_open_err)
                meta["protocol_open_error"] = proto_open_err
                if not paper_source_of_truth():
                    stats["details"].append(
                        {
                            "label": label,
                            "action": "protocol_error",
                            "error": proto_open_err,
                        }
                    )
                    continue

        conn.execute(
            """INSERT INTO moss2_signals(
                   profile_id, recorded_at_utc, side, symbol,
                   entry_price, virtual_notional_usdt, mark_price,
                   composite, regime, unrealized_pnl_usdt, meta_json, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,0,?,?)""",
            (
                pid,
                now,
                side,
                symbol,
                entry_px,
                notional,
                entry_px,
                composite,
                regime_label,
                json.dumps(meta, ensure_ascii=False),
                now,
            ),
        )
        conn.commit()
        stats["opens"] += 1
        open_d: Dict[str, Any] = {
            "label": label,
            "action": "open",
            "side": side,
            "composite": composite,
            "regime": regime_label,
            "notional_usdt": notional,
            "protocol": protocol_ok,
        }
        if proto_open_err:
            open_d["protocol_error"] = proto_open_err
        stats["details"].append(open_d)
        if cfg.MOSS2_VERBOSE_LOG:
            logger.info(
                "[moss2] %s OPEN %s composite=%.4f regime=%s scale=%.2f",
                label,
                side,
                composite,
                regime_label,
                scale,
            )

    insert_paper_run(conn, stats)
    return stats

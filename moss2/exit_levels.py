"""纸面持仓止盈/止损参考价（与 paper_scanner._check_exit 的 ATR 规则一致）。"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from moss2 import config as cfg
from moss2.db import get_profile
from moss2.kline_loader import load_market_df
from moss2.params import merge_profile_params

logger = logging.getLogger(__name__)


def _variant_modules(variant: str):
    if str(variant).lower() == "en":
        from moss2.variants.en.core.decision import DecisionParams
        from moss2.variants.en.core.indicators import atr as compute_atr

        cap = lambda p, s: p
    else:
        from moss2.variants.hl.core.decision import DecisionParams
        from moss2.variants.hl.core.indicators import atr as compute_atr
        from moss2.variants.hl.core.leverage_caps import cap_params_for_symbol

        cap = cap_params_for_symbol
    return DecisionParams, compute_atr, cap


def compute_exit_price_levels(
    *,
    side: str,
    entry: float,
    mark: float,
    params_dict: dict,
    df: pd.DataFrame,
    variant: str,
) -> Dict[str, float]:
    """返回 stop_loss / take_profit 等价触发价位（每轮扫描用最新 ATR 重算）。"""
    DecisionParams, compute_atr, cap = _variant_modules(variant)
    clean = {k: v for k, v in params_dict.items() if not str(k).startswith("_")}
    params = DecisionParams.from_dict(
        cap(dict(clean), str(params_dict.get("_symbol") or ""))
    )
    atr_series = compute_atr(df, 14)
    atr_val = float(atr_series.iloc[-1])
    if np.isnan(atr_val) or atr_val <= 0:
        atr_val = float(mark) * 0.02 if mark > 0 else 0.0
    sl_dist = float(params.sl_atr_mult) * atr_val
    tp_dist = sl_dist * float(params.tp_rr_ratio)
    side_u = str(side).upper()
    if side_u in ("LONG", "BUY"):
        sl = float(entry) - sl_dist
        tp = float(entry) + tp_dist
    else:
        sl = float(entry) + sl_dist
        tp = float(entry) - tp_dist
    return {
        "stop_loss": round(sl, 8),
        "take_profit": round(tp, 8),
        "atr14": round(atr_val, 8),
    }


def merge_exit_levels_into_meta(meta: Optional[dict], levels: Dict[str, float], *, at_utc: str) -> dict:
    out = dict(meta or {})
    out["stop_loss"] = levels["stop_loss"]
    out["take_profit"] = levels["take_profit"]
    out["atr14"] = levels.get("atr14")
    out["exit_levels_at_utc"] = at_utc
    return out


def parse_exit_levels_from_meta(meta_json: Optional[str]) -> Dict[str, Optional[float]]:
    if not meta_json:
        return {"stop_loss": None, "take_profit": None, "atr14": None}
    try:
        meta = json.loads(meta_json) if isinstance(meta_json, str) else dict(meta_json or {})
    except (TypeError, json.JSONDecodeError):
        return {"stop_loss": None, "take_profit": None, "atr14": None}
    return {
        "stop_loss": meta.get("stop_loss"),
        "take_profit": meta.get("take_profit"),
        "atr14": meta.get("atr14"),
    }


def enrich_position_exit_levels(
    conn,
    pos: Dict[str, Any],
    *,
    df: Optional[pd.DataFrame] = None,
    at_utc: Optional[str] = None,
    persist_meta: bool = False,
    signal_id: Optional[int] = None,
    meta_json: Optional[str] = None,
) -> Dict[str, Any]:
    pid = int(pos.get("profile_id") or 0)
    sym = str(pos.get("symbol") or "").upper()
    variant = str(pos.get("variant") or cfg.MOSS2_OPS_VARIANT)
    side = str(pos.get("side") or "")
    entry = float(pos.get("entry_price") or 0)
    mark = float(pos.get("mark_price") or entry)
    if not sym or side.upper() not in ("LONG", "SHORT") or entry <= 0:
        return pos
    prof = get_profile(conn, pid) if pid else None
    if not prof:
        return pos
    params_d = merge_profile_params(prof)
    params_d["_symbol"] = sym
    try:
        kdf = df if df is not None else load_market_df(sym, variant, limit=cfg.MOSS2_KLINE_LIMIT)
        levels = compute_exit_price_levels(
            side=side,
            entry=entry,
            mark=mark,
            params_dict=params_d,
            df=kdf,
            variant=variant,
        )
        pos.update(levels)
        if persist_meta and at_utc and signal_id is not None:
            meta = merge_exit_levels_into_meta(
                json.loads(meta_json or "{}") if meta_json else {},
                levels,
                at_utc=at_utc,
            )
            conn.execute(
                "UPDATE moss2_signals SET meta_json=?, updated_at_utc=? WHERE id=?",
                (json.dumps(meta, ensure_ascii=False), at_utc, int(signal_id)),
            )
    except Exception as exc:
        logger.warning("[moss2] exit levels %s failed: %s", sym, exc)
    return pos

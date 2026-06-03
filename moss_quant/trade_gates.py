"""开仓前轻量过滤：资金费率极端等（不替代 composite 信号）。"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

from moss_quant import config as cfg

logger = logging.getLogger(__name__)


def _funding_rate_from_db(conn: Optional[sqlite3.Connection], symbol: str) -> Optional[float]:
    if conn is None:
        return None
    try:
        row = conn.execute(
            """SELECT funding_rate FROM s2_funding_signals
               WHERE symbol=? ORDER BY recorded_at DESC LIMIT 1""",
            (symbol.upper(),),
        ).fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception:
        pass
    return None


def _funding_rate_live(symbol: str) -> Optional[float]:
    try:
        from accumulation_radar import api_get

        data = api_get("/fapi/v1/premiumIndex", {"symbol": symbol.upper()})
        if isinstance(data, dict) and data.get("lastFundingRate") is not None:
            return float(data["lastFundingRate"])
    except Exception as e:
        logger.debug("funding live %s: %s", symbol, e)
    return None


def _oi_radar_row(symbol: str) -> Optional[Dict[str, Any]]:
    """从 oi_radar_snapshot.json 读取标的 OI/价格变化（与收筹雷达同源）。"""
    sym = str(symbol or "").upper()
    base = sym.replace("USDT", "")
    try:
        from accumulation_radar import OI_RADAR_SNAPSHOT_PATH

        if not OI_RADAR_SNAPSHOT_PATH.is_file():
            return None
        raw = json.loads(OI_RADAR_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        for row in raw.get("coin_data") or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("sym") or "").upper() == sym:
                return row
            if str(row.get("coin") or "").upper() == base:
                return row
    except Exception as e:
        logger.debug("oi radar row %s: %s", sym, e)
    return None


def _oi_spike_flat_price(row: Dict[str, Any]) -> bool:
    """6h OI 增幅大但价格几乎不动 → 假突破风险。"""
    d6h = float(row.get("d6h") or 0)
    px = abs(float(row.get("px_chg") or 0))
    return d6h >= float(cfg.MOSS_QUANT_GATE_OI_D6H_MIN) and px <= float(
        cfg.MOSS_QUANT_GATE_OI_PX_FLAT_MAX
    )


def entry_trade_gate(
    symbol: str,
    *,
    side: str,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """
    返回 threshold_bump（提高开仓门槛）、allowed、reasons。
    不直接禁止时仅抬高 entry_threshold。
    """
    sym = str(symbol or "").upper()
    side_u = str(side or "").upper()
    bump = 0.0
    reasons: List[str] = []

    if cfg.MOSS_QUANT_GATE_FUNDING_EXTREME:
        fr = _funding_rate_from_db(conn, sym)
        if fr is None:
            fr = _funding_rate_live(sym)
        if fr is not None:
            extreme = float(cfg.MOSS_QUANT_GATE_FUNDING_ABS_MAX)
            # 做多时费率过正 = 多头拥挤；做空时费率过负 = 空头拥挤
            if side_u == "LONG" and fr > extreme:
                bump += float(cfg.MOSS_QUANT_GATE_FUNDING_BUMP)
                reasons.append(f"funding_high_long_{fr:.4f}")
            elif side_u == "SHORT" and fr < -extreme:
                bump += float(cfg.MOSS_QUANT_GATE_FUNDING_BUMP)
                reasons.append(f"funding_low_short_{fr:.4f}")

    if cfg.MOSS_QUANT_GATE_OI_SPIKE:
        oi_row = _oi_radar_row(sym)
        if oi_row and _oi_spike_flat_price(oi_row):
            bump += float(cfg.MOSS_QUANT_GATE_OI_BUMP)
            reasons.append(
                f"oi_spike_flat_d6h={float(oi_row.get('d6h') or 0):.1f}_px={float(oi_row.get('px_chg') or 0):.1f}"
            )

    allowed = True
    if bump >= float(cfg.MOSS_QUANT_GATE_BLOCK_BUMP) and cfg.MOSS_QUANT_GATE_HARD_BLOCK:
        allowed = False
        reasons.append("gate_hard_block")

    return {
        "allowed": allowed,
        "threshold_bump": round(bump, 4),
        "reasons": reasons,
    }


def effective_entry_threshold(
    base_threshold: float,
    *,
    gate_bump: float = 0.0,
    intraday_bump: float = 0.0,
    regime_delta: float = 0.0,
) -> float:
    """regime_delta<0 放宽，>0 收紧（在 gate/日内 bump 之后叠加）。"""
    t = (
        float(base_threshold)
        + float(gate_bump)
        + float(intraday_bump)
        + float(regime_delta)
    )
    return round(max(0.05, min(0.75, t)), 4)


def train_regime_note_from_summary(summary: Optional[Dict[str, Any]]) -> str:
    """从寻优 summary / recent_pick 提取训练窗 regime 标签（供验证可达性等）。"""
    if not summary:
        return ""
    rp = summary.get("recent_pick")
    if isinstance(rp, dict):
        note = str(rp.get("regime_note") or "").strip()
        if note:
            return note
    adj = summary.get("regime_adjustment")
    if isinstance(adj, dict):
        return str(adj.get("regime_note") or "").strip()
    return ""


def _side_stats_from_local_refine(block: Dict[str, Any]) -> Dict[str, Any]:
    rounds = block.get("rounds") or block.get("rounds_log") or []
    if not isinstance(rounds, list) or not rounds:
        return {}
    last = rounds[-1]
    if not isinstance(last, dict):
        return {}
    ta = last.get("train_analysis") or {}
    if isinstance(ta, dict):
        ss = ta.get("side_stats")
        if isinstance(ss, dict) and ss:
            return ss
    return {}


def _side_stats_from_summary(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not summary:
        return {}
    rp = summary.get("recent_pick")
    if isinstance(rp, dict):
        ss = rp.get("side_stats")
        if isinstance(ss, dict) and ss:
            return ss
    pipe = summary.get("post_grid_pipeline")
    if isinstance(pipe, dict):
        td = pipe.get("tuning_diagnosis")
        if isinstance(td, dict):
            ta = td.get("train_analysis") or {}
            if isinstance(ta, dict):
                ss = ta.get("side_stats")
                if isinstance(ss, dict) and ss:
                    return ss
        lr = pipe.get("local_refine")
        if isinstance(lr, dict):
            ss = _side_stats_from_local_refine(lr)
            if ss:
                return ss
    for key in ("train_analysis", "local_refine"):
        block = summary.get(key)
        if isinstance(block, dict):
            ss = block.get("side_stats")
            if isinstance(ss, dict) and ss:
                return ss
            ss = _side_stats_from_local_refine(block)
            if ss:
                return ss
    return {}


def _latest_completed_optimize_summary(
    conn: sqlite3.Connection, symbol: str
) -> Dict[str, Any]:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return {}
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT i.summary_json
           FROM moss_daily_optimize_items i
           INNER JOIN moss_daily_optimize_batches b ON b.id = i.batch_id
           WHERE i.symbol = ? AND b.status = 'completed'
           ORDER BY b.finished_at_utc DESC, b.id DESC
           LIMIT 1""",
        (sym,),
    ).fetchone()
    if not row or not row["summary_json"]:
        return {}
    try:
        return json.loads(row["summary_json"] or "{}")
    except json.JSONDecodeError:
        return {}


def latest_attribution_side_stats(
    conn: sqlite3.Connection, symbol: str
) -> Dict[str, Any]:
    """最近每日寻优批次中该标的训练/L3 多空胜率统计。"""
    return _side_stats_from_summary(_latest_completed_optimize_summary(conn, symbol))


def side_attribution_threshold_deltas(
    side_stats: Dict[str, Any],
    *,
    base_threshold: float,
) -> Dict[str, Any]:
    """纸面扫描：空/多侧胜率差 → 仅抬升弱势方向门槛。"""
    if not cfg.MOSS_QUANT_SIDE_BIAS_ADJUST_ENABLED or base_threshold <= 0:
        return {"long_delta": 0.0, "short_delta": 0.0, "reason": ""}
    lc = int(side_stats.get("long_count") or 0)
    sc = int(side_stats.get("short_count") or 0)
    min_side = int(cfg.MOSS_QUANT_SIDE_BIAS_MIN_SIDE_TRADES)
    lwr = side_stats.get("long_win_rate")
    swr = side_stats.get("short_win_rate")
    gap = float(cfg.MOSS_QUANT_SIDE_BIAS_WIN_GAP)
    if lc < min_side or sc < min_side or lwr is None or swr is None:
        return {"long_delta": 0.0, "short_delta": 0.0, "reason": ""}
    long_d = 0.0
    short_d = 0.0
    reason = ""
    if float(swr) < float(lwr) - gap:
        short_d = float(cfg.MOSS_QUANT_SIDE_BIAS_SHORT_EXTRA_BUMP)
        reason = "short_side_weak"
    elif float(lwr) < float(swr) - gap:
        long_d = float(cfg.MOSS_QUANT_SIDE_BIAS_LONG_EXTRA_BUMP)
        reason = "long_side_weak"
    return {
        "long_delta": round(long_d, 4),
        "short_delta": round(short_d, 4),
        "reason": reason,
        "side_stats": side_stats,
    }


def latest_train_regime_note(
    conn: sqlite3.Connection, symbol: str
) -> str:
    """最近完成的每日寻优批次里该标的训练窗 regime 标签。"""
    return train_regime_note_from_summary(
        _latest_completed_optimize_summary(conn, symbol)
    )


def regime_alignment_state(
    train_regime_note: str,
    live_regime: str,
    *,
    template: str = "balanced",
) -> str:
    """aligned | misaligned | neutral"""
    note = str(train_regime_note or "").strip()
    live = str(live_regime or "SIDEWAYS").upper()
    tpl = str(template or "balanced").lower()
    sideways_live = live in ("SIDEWAYS", "CHOP", "RANGE")
    if note == "trend_heavy" and live in ("BULL", "BEAR"):
        return "aligned"
    if note == "sideways_heavy" and sideways_live:
        return "aligned"
    if note == "trend_heavy" and sideways_live:
        return "misaligned"
    if note == "sideways_heavy" and live in ("BULL", "BEAR"):
        return "misaligned"
    return "neutral"


def regime_aligned_threshold_deltas(
    base_threshold: float,
    *,
    train_regime_note: str,
    live_regime: str,
    template: str = "balanced",
    allow_relax: bool = True,
) -> Dict[str, Any]:
    """
    返回 long/short 相对 base 的增量（负=放宽）。
    仅在纸面扫描开仓前使用，不改回测/寻优网格。
    """
    base = float(base_threshold)
    if not cfg.MOSS_QUANT_REGIME_ALIGN_ADJUST_ENABLED or base <= 0:
        return {
            "long_delta": 0.0,
            "short_delta": 0.0,
            "alignment": "neutral",
            "reason": "",
        }
    note = str(train_regime_note or "").strip()
    state = regime_alignment_state(note, live_regime, template=template)
    live = str(live_regime or "").upper()
    long_d = 0.0
    short_d = 0.0
    reason = ""

    if state == "misaligned":
        bump = base * float(cfg.MOSS_QUANT_REGIME_ALIGN_TIGHTEN_PCT)
        long_d = short_d = round(bump, 4)
        reason = "train_live_regime_mismatch"
    elif state == "aligned" and allow_relax:
        relax = base * float(cfg.MOSS_QUANT_REGIME_ALIGN_RELAX_PCT)
        if live == "BEAR":
            short_d = round(-relax, 4)
            reason = "bear_short_relax"
        elif live == "BULL":
            long_d = round(-relax, 4)
            reason = "bull_long_relax"
        elif live in ("SIDEWAYS", "CHOP", "RANGE") and note == "sideways_heavy":
            side_relax = base * float(cfg.MOSS_QUANT_REGIME_ALIGN_SIDEWAYS_RELAX_PCT)
            long_d = short_d = round(-side_relax, 4)
            reason = "sideways_symmetric_relax"
    return {
        "long_delta": long_d,
        "short_delta": short_d,
        "alignment": state,
        "reason": reason,
        "train_regime_note": note,
    }


def resolve_entry_thresholds_for_scan(
    conn: sqlite3.Connection,
    profile: Dict[str, Any],
    *,
    live_regime: str,
    symbol: str,
) -> Dict[str, Any]:
    """纸面扫描：base + 日内 + gate + regime 对齐 → 多空阈值。"""
    from moss_quant.optimize_policy import paper_recent_pnl_block_reason

    pid = int(profile["id"])
    base_th = float(
        (profile.get("tactical_params") or {}).get("entry_threshold")
        or (profile.get("initial_params") or {}).get("entry_threshold")
        or 0.44
    )
    cap = float(profile.get("virtual_equity_usdt") or 0) or None
    intra_bump = intraday_threshold_bump(conn, pid, profile_capital=cap)
    paper_block = bool(
        paper_recent_pnl_block_reason(conn, pid, profile_capital=cap)
    )
    allow_relax = not paper_block and intra_bump <= 0
    train_note = latest_train_regime_note(conn, symbol)
    template = str(profile.get("template") or "balanced")
    align = regime_aligned_threshold_deltas(
        base_th,
        train_regime_note=train_note,
        live_regime=live_regime,
        template=template,
        allow_relax=allow_relax,
    )
    side_stats = latest_attribution_side_stats(conn, symbol)
    side_bias = side_attribution_threshold_deltas(
        side_stats, base_threshold=base_th
    )
    gate_long = entry_trade_gate(symbol, side="LONG", conn=conn)
    gate_short = entry_trade_gate(symbol, side="SHORT", conn=conn)
    long_th = effective_entry_threshold(
        base_th,
        gate_bump=float(gate_long.get("threshold_bump") or 0),
        intraday_bump=intra_bump,
        regime_delta=float(align["long_delta"])
        + float(side_bias.get("long_delta") or 0),
    )
    short_th = effective_entry_threshold(
        base_th,
        gate_bump=float(gate_short.get("threshold_bump") or 0),
        intraday_bump=intra_bump,
        regime_delta=float(align["short_delta"])
        + float(side_bias.get("short_delta") or 0),
    )
    return {
        "base_threshold": round(base_th, 4),
        "long_threshold": long_th,
        "short_threshold": short_th,
        "intraday_bump": round(intra_bump, 4),
        "gate_long": gate_long,
        "gate_short": gate_short,
        "regime_align": align,
        "side_bias": side_bias,
        "paper_loss_block_relax": paper_block,
    }


def intraday_threshold_bump_from_pnl(pnl_pct: float) -> float:
    """按 Profile 盈亏占本金比例抬高开仓门槛（纸面/纸面回测共用）。"""
    if not cfg.MOSS_QUANT_INTRADAY_ADJUST_ENABLED:
        return 0.0
    pct = float(pnl_pct)
    if pct <= -float(cfg.MOSS_QUANT_INTRADAY_DRAWDOWN_PCT):
        return float(cfg.MOSS_QUANT_INTRADAY_DRAWDOWN_BUMP)
    if pct >= float(cfg.MOSS_QUANT_INTRADAY_PROFIT_PCT):
        return float(cfg.MOSS_QUANT_INTRADAY_PROFIT_BUMP)
    return 0.0


def intraday_threshold_bump(
    conn: sqlite3.Connection,
    profile_id: int,
    *,
    profile_capital: Optional[float] = None,
) -> float:
    """当日该 Profile 回撤/盈利过大时微调开仓门槛。"""
    if not cfg.MOSS_QUANT_INTRADAY_ADJUST_ENABLED:
        return 0.0
    cap = float(profile_capital or cfg.MOSS_QUANT_PROFILE_CAPITAL)
    if cap <= 0:
        return 0.0
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT COALESCE(SUM(
                   CASE WHEN outcome IS NOT NULL THEN pnl_usdt ELSE 0 END
               ), 0) AS realized,
                  COALESCE(SUM(
                   CASE WHEN outcome IS NULL THEN unrealized_pnl_usdt ELSE 0 END
               ), 0) AS unrealized
           FROM moss_signals WHERE profile_id=?""",
        (int(profile_id),),
    ).fetchone()
    if not row:
        return 0.0
    pnl = float(row["realized"] or 0) + float(row["unrealized"] or 0)
    return intraday_threshold_bump_from_pnl(pnl / cap)

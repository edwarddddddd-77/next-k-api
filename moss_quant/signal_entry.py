"""纸面开仓质量（余量/多根确认）与验证窗可达性子样本（对齐 gate_proxy 费率逻辑）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from moss_quant import config as cfg
from moss_quant.core.decision import DecisionParams, _composite_at_bar


def effective_entry_margin(*, alignment: str = "neutral") -> float:
    """aligned 时略降余量，与纸面 regime 放宽一致。"""
    m = float(cfg.MOSS_QUANT_ENTRY_MARGIN or 0)
    if m <= 0:
        return 0.0
    if (
        str(alignment or "").lower() == "aligned"
        and float(cfg.MOSS_QUANT_ENTRY_MARGIN_ALIGNED_RELAX or 0) > 0
    ):
        return max(0.0, m - float(cfg.MOSS_QUANT_ENTRY_MARGIN_ALIGNED_RELAX))
    return m


def entry_confirm_bars() -> int:
    """0 表示仅最后一根（等同单根确认，仍受余量约束）。"""
    return max(0, int(cfg.MOSS_QUANT_ENTRY_CONFIRM_BARS or 0))


def compute_trailing_composites(
    df: pd.DataFrame,
    params: DecisionParams,
    regime: pd.Series,
    *,
    bars: int,
) -> List[float]:
    """最近 bars 根 K 线的 composite（时间正序，末项为当前 bar）。"""
    params.normalize_weights()
    n = len(df)
    start_min = max(int(params.slow_ma_period), 50)
    if n <= start_min:
        return []
    need = max(1, int(bars))
    cache: dict = {}
    out: List[float] = []
    for i in range(max(start_min, n - need), n):
        out.append(float(_composite_at_bar(df, params, regime, i, cache)))
    return out


def _funding_bump_for_rate(fr: float, side: str) -> float:
    if not cfg.MOSS_QUANT_GATE_FUNDING_EXTREME:
        return 0.0
    extreme = float(cfg.MOSS_QUANT_GATE_FUNDING_ABS_MAX)
    bump = float(cfg.MOSS_QUANT_GATE_FUNDING_BUMP)
    side_u = str(side or "").upper()
    if side_u == "LONG" and fr > extreme:
        return bump
    if side_u == "SHORT" and fr < -extreme:
        return bump
    return 0.0


def _funding_rate_at_bar_ms(
    bar_ms: int,
    f_times: List[int],
    f_rates: List[float],
) -> float:
    if not f_times:
        return 0.0
    idx = 0
    for i, t_ms in enumerate(f_times):
        if t_ms <= bar_ms:
            idx = i
    return float(f_rates[idx])


def _bar_thresholds_funding_only(
    base: float,
    fr: float,
) -> Tuple[float, float]:
    long_th = base + _funding_bump_for_rate(fr, "LONG")
    short_th = base + _funding_bump_for_rate(fr, "SHORT")
    return long_th, short_th


def _regime_deltas_at_bar(
    base: float,
    *,
    train_regime_note: str,
    live_regime: str,
    template: str,
) -> Tuple[float, float]:
    from moss_quant.trade_gates import regime_aligned_threshold_deltas

    pack = regime_aligned_threshold_deltas(
        base,
        train_regime_note=train_regime_note,
        live_regime=live_regime,
        template=template,
        allow_relax=True,
    )
    return float(pack.get("long_delta") or 0), float(pack.get("short_delta") or 0)


def _passes_entry_at_composites(
    composites: List[float],
    *,
    long_threshold: float,
    short_threshold: float,
    entry_margin: float,
    confirm_bars: int,
) -> Tuple[int, str]:
    """返回 signal 1/-1/0 与 reason。"""
    if not composites:
        return 0, "composite_unavailable"
    long_eff = float(long_threshold) + float(entry_margin)
    short_eff = float(short_threshold) + float(entry_margin)
    c_last = float(composites[-1])
    raw_confirm = int(confirm_bars if confirm_bars is not None else 1)
    k = 1 if raw_confirm <= 0 else raw_confirm
    if len(composites) < k:
        return 0, "confirm_bars_insufficient"

    def long_ok() -> bool:
        tail = [float(x) for x in composites[-k:]]
        if not all(x > long_eff for x in tail):
            return False
        if k >= 2 and tail[-1] < tail[-2] - 1e-9:
            return False
        return True

    def short_ok() -> bool:
        tail = [float(x) for x in composites[-k:]]
        if not all(x < -short_eff for x in tail):
            return False
        if k >= 2 and tail[-1] > tail[-2] + 1e-9:
            return False
        return True

    if long_ok():
        return 1, "signal_long"
    if short_ok():
        return -1, "signal_short"
    if abs(c_last) <= min(long_eff, short_eff):
        return 0, "composite_below_threshold"
    if c_last > 0:
        return 0, "long_margin_or_confirm_failed"
    return 0, "short_margin_or_confirm_failed"


def evaluate_entry_signal(
    composites: List[float],
    *,
    long_threshold: float,
    short_threshold: float,
    entry_margin: Optional[float] = None,
    confirm_bars: Optional[int] = None,
    alignment: str = "neutral",
) -> Dict[str, Any]:
    margin = (
        float(entry_margin)
        if entry_margin is not None
        else effective_entry_margin(alignment=alignment)
    )
    confirm = (
        int(confirm_bars)
        if confirm_bars is not None
        else entry_confirm_bars()
    )
    if not cfg.MOSS_QUANT_ENTRY_QUALITY_ENABLED:
        margin = 0.0
        confirm = 1
    sig, reason = _passes_entry_at_composites(
        composites,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        entry_margin=margin,
        confirm_bars=confirm,
    )
    c_last = float(composites[-1]) if composites else 0.0
    return {
        "signal": int(sig),
        "reason": reason,
        "composite": round(c_last, 4),
        "entry_margin": round(margin, 4),
        "confirm_bars": confirm,
        "entry_threshold_long_eff": round(long_threshold + margin, 4),
        "entry_threshold_short_eff": round(short_threshold + margin, 4),
    }


def validation_reachable_stats(
    df_ctx: pd.DataFrame,
    regime: pd.Series,
    symbol: str,
    params: DecisionParams,
    *,
    base_threshold: float,
    val_start_idx: int,
    train_regime_note: str = "",
    template: str = "balanced",
) -> Dict[str, Any]:
    """
    验证窗内：composite 过线且过简化 funding gate 的 bar 占比与子样本 forward 收益。
    """
    empty = {
        "reachable_ratio": 0.0,
        "reachable_bars": 0,
        "val_bars_scored": 0,
        "reachable_sub_trades": 0,
        "reachable_sub_wins": 0,
        "reachable_sub_pf": 0.0,
        "reachable_sub_win_rate": 0.0,
        "reachable_sub_avg_ret": 0.0,
    }
    if not cfg.MOSS_QUANT_OPTIMIZE_REACHABLE_ENABLED:
        return empty

    from moss_quant.gate_proxy import fetch_binance_funding_history

    n = len(df_ctx)
    start_min = max(int(params.slow_ma_period), 50)
    if n <= start_min or val_start_idx >= n:
        return empty

    params.normalize_weights()
    if "timestamp" not in df_ctx.columns or "close" not in df_ctx.columns:
        return empty

    ts = pd.to_datetime(df_ctx["timestamp"], utc=True)
    start_ms = int(ts.iloc[val_start_idx].value // 1_000_000)
    end_ms = int(ts.iloc[-1].value // 1_000_000)
    funding = fetch_binance_funding_history(
        str(symbol).upper(), start_ms=start_ms, end_ms=end_ms
    )
    f_times = [int(r["time_ms"]) for r in funding]
    f_rates = [float(r["rate"]) for r in funding]

    margin = float(cfg.MOSS_QUANT_ENTRY_MARGIN or 0) if cfg.MOSS_QUANT_ENTRY_QUALITY_ENABLED else 0.0
    confirm = entry_confirm_bars() if cfg.MOSS_QUANT_ENTRY_QUALITY_ENABLED else 1
    use_regime = bool(cfg.MOSS_QUANT_OPTIMIZE_REACHABLE_REGIME_ALIGN)
    forward = max(1, int(cfg.MOSS_QUANT_OPTIMIZE_REACHABLE_FORWARD_BARS))
    base = float(base_threshold)

    cache: dict = {}
    reachable = 0
    scored = 0
    sub_returns: List[float] = []

    for i in range(max(val_start_idx, start_min), n):
        bar_ms = int(ts.iloc[i].value // 1_000_000)
        fr = _funding_rate_at_bar_ms(bar_ms, f_times, f_rates)
        long_th, short_th = _bar_thresholds_funding_only(base, fr)
        if use_regime and train_regime_note:
            live = str(regime.iloc[i] if i < len(regime) else "SIDEWAYS")
            ld, sd = _regime_deltas_at_bar(
                base,
                train_regime_note=train_regime_note,
                live_regime=live,
                template=template,
            )
            long_th = max(0.05, min(0.75, long_th + ld))
            short_th = max(0.05, min(0.75, short_th + sd))

        need = max(confirm, 1)
        if i - need + 1 < start_min:
            continue
        comps = []
        for j in range(i - need + 1, i + 1):
            comps.append(float(_composite_at_bar(df_ctx, params, regime, j, cache)))
        scored += 1
        sig, _ = _passes_entry_at_composites(
            comps,
            long_threshold=long_th,
            short_threshold=short_th,
            entry_margin=margin,
            confirm_bars=confirm,
        )
        if sig == 0:
            continue
        reachable += 1
        if i + forward < n:
            c0 = float(df_ctx["close"].iloc[i])
            c1 = float(df_ctx["close"].iloc[i + forward])
            if c0 > 0:
                ret = (c1 - c0) / c0
                if sig < 0:
                    ret = -ret
                sub_returns.append(ret)

    ratio = reachable / scored if scored else 0.0
    wins = sum(1 for r in sub_returns if r > 0)
    losses = sum(1 for r in sub_returns if r < 0)
    gross_win = sum(r for r in sub_returns if r > 0)
    gross_loss = abs(sum(r for r in sub_returns if r < 0))
    pf = (gross_win / gross_loss) if gross_loss > 1e-12 else (2.0 if gross_win > 0 else 0.0)
    avg_ret = sum(sub_returns) / len(sub_returns) if sub_returns else 0.0
    wr = wins / len(sub_returns) if sub_returns else 0.0

    return {
        "reachable_ratio": round(ratio, 4),
        "reachable_bars": reachable,
        "val_bars_scored": scored,
        "reachable_sub_trades": len(sub_returns),
        "reachable_sub_wins": wins,
        "reachable_sub_pf": round(pf, 4),
        "reachable_sub_win_rate": round(wr, 4),
        "reachable_sub_avg_ret": round(avg_ret, 6),
    }

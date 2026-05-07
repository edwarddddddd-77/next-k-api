#!/usr/bin/env python3
"""
突破 → 回踩 → 延续（Breakout, Pullback, Continuation）状态机

不含 OI；仅用价格行为 + 成交量（优先 Binance K 线 quote asset volume）。

实盘注意（已实现默认优化）：
  · 回踩触及：低价探入支撑带上沿即可（允许插针跌破下沿），无效破位以收盘价 invalidation 为准。
  · 阻力：滚动最高价与更长窗口内分型高点取强，减轻横盘「矮窗伪阻力」。
  · 单根 V 反：micro_hi 锚到突破位；同根 Pin/吞没可放宽缩量门禁。
  · 延续触发后：默认下一根 K 自动回到 idle，避免相位卡住（末根仍可停在 continuation）。

三状态（工程上拆成四段便于调试）：
  idle          — 等待突破
  post_breakout — 已确认突破，跟踪突破后高点
  pullback      — 已从峰值回踩至突破位（支撑带）附近，跟踪回踩段量能
  continuation  — 回踩结束：出现反转 K 或收复回踩段内高点（简化版 CHoCH）

K 线格式：Binance /fapi/v1/klines 原始 list，
  [0]=open time, [1]=open, [2]=high, [3]=low, [4]=close, [5]=volume, ...
  [7]=quote asset volume（若缺失则用 volume*close 近似）
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class BPCPhase(str, Enum):
    IDLE = "idle"
    POST_BREAKOUT = "post_breakout"
    PULLBACK = "pullback"
    CONTINUATION = "continuation"


@dataclass
class BPCParams:
    # 阻力窗口：收盘突破此前窗口最高价（可与 swing 强化叠加）
    resistance_lookback: int = 30
    # True：阻力 = max(滚动窗口最高, 近更长窗口内分型高点)；减轻横盘过长时滚动窗「变矮」导致的伪突破
    resistance_use_swing_reinforce: bool = True
    # 突破：收盘价高于阻力 * (1 + eps)
    breakout_eps: float = 0.0001
    # 突破放量：当前 quote vol >= 过去 vol_ma_period 均值 * mult
    vol_ma_period: int = 20
    breakout_vol_mult: float = 1.15
    # 突破视为失败：收盘跌破突破参考位以下 invalidation_pct（比例，0.012 = 1.2%）
    invalidation_pct: float = 0.012
    # 进入回踩：相对峰值最小回撤比例（0.0015 = 0.15%）
    min_retrace_from_peak_pct: float = 0.0015
    # 回踩触及支撑带上沿：最低价探到 band_hi 以下即算「探到支撑区」（允许插针跌破下沿，由收盘价 invalidation 兜底）
    retest_band_pct: float = 0.004
    # 回踩段相对突破 burst 量能：均额比例上限视为「缩量回踩」
    pullback_vol_ratio_max: float = 0.92
    # 突破 burst 取 breakout 起若干根均额
    breakout_burst_bars: int = 3
    # 延续：Pinbar 下影 >= pin_body_mult * 实体
    pin_body_mult: float = 2.0
    # 延续：收复回踩段内前序高点（简化 CHoCH）
    continuation_close_above_micro_high: bool = True
    # 同一根 K 线完成「触及支撑 + Pin/吞没」时，回踩均额无法拆分：跳过缩量门禁
    single_bar_pullback_pin_relaxes_volume: bool = True
    # 延续确认出现在 continuation_idx 的下一根 K 起自动 reset_idle，避免流式/多次调用卡在 continuation
    continuation_reset_next_bar: bool = True


@dataclass
class BPCState:
    phase: BPCPhase = BPCPhase.IDLE
    breakout_idx: Optional[int] = None
    breakout_level: float = 0.0
    peak_after_breakout: float = 0.0
    peak_idx: int = 0
    pullback_enter_idx: Optional[int] = None
    pullback_low: float = 0.0
    continuation_idx: Optional[int] = None
    continuation_reason: str = ""
    last_invalid_reason: str = ""


def _parse_klines_row(k: Sequence[Any]) -> Tuple[float, float, float, float, float]:
    o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
    return o, h, l, c


def _quote_vol(k: Sequence[Any]) -> float:
    vol = float(k[5])
    c = float(k[4])
    if len(k) > 7:
        return float(k[7])
    return vol * c


def _sma(vals: List[float], i_end_exclusive: int, length: int) -> float:
    """vals[j] for j in [i_end_exclusive - length, i_end_exclusive)."""
    if i_end_exclusive < length:
        return sum(vals[:i_end_exclusive]) / max(1, i_end_exclusive)
    s = sum(vals[i_end_exclusive - length : i_end_exclusive])
    return s / float(length)


def _max_high(highs: List[float], start: int, end_exclusive: int) -> float:
    """[start, end_exclusive)."""
    if start >= end_exclusive:
        return 0.0
    return max(highs[start:end_exclusive])


def _swing_high_max(highs: List[float], start: int, end_exclusive: int) -> float:
    """区间内分型高点（不低于左右邻居），用于锚定「真实前高」，减轻固定滚动窗盲区。"""
    if end_exclusive - start < 3:
        return 0.0
    hi_len = len(highs)
    best = 0.0
    for j in range(max(start + 1, 1), min(end_exclusive - 1, hi_len - 1)):
        if highs[j] >= highs[j - 1] and highs[j] >= highs[j + 1]:
            best = max(best, highs[j])
    return best


def _resistance_level(highs: List[float], i: int, lookback: int, use_swing: bool) -> float:
    lo = max(0, i - lookback)
    rolling = _max_high(highs, lo, i)
    if not use_swing or i < 3:
        return rolling
    span = max(lookback * 2, lookback + 20)
    slo = max(0, i - span)
    swing_m = _swing_high_max(highs, slo, i)
    if swing_m > 0:
        return max(rolling, swing_m)
    return rolling


def _is_bullish_engulfing(
    o0: float, h0: float, l0: float, c0: float, o1: float, h1: float, l1: float, c1: float
) -> bool:
    bear0 = c0 < o0
    bull1 = c1 > o1
    if not bull1:
        return False
    if bear0:
        return c1 >= o0 and o1 <= c0 and c1 > o0
    # 前一根也可为小阳，经典放宽：当前阳线实体吞没前一根实体
    body0 = abs(c0 - o0)
    body1 = abs(c1 - o1)
    return bull1 and body1 > body0 and c1 > max(c0, o0)


def _is_pin_bar_bullish(o: float, h: float, l: float, c: float, pin_body_mult: float) -> bool:
    body = abs(c - o)
    rng = h - l
    if rng <= 0:
        return False
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    if body < rng * 0.05:
        body = rng * 0.05
    if lower_wick < pin_body_mult * body:
        return False
    if upper_wick > lower_wick * 0.6:
        return False
    mid = (h + l) / 2.0
    return c >= mid


def evaluate_breakout_pullback_continuation(
    klines: Sequence[Sequence[Any]],
    params: Optional[BPCParams] = None,
) -> Dict[str, Any]:
    """
    对整段 K 线做一次前向扫描，返回**最后一根 bar 时刻**的状态机结论。
    """
    p = params or BPCParams()
    n = len(klines)
    min_warmup = max(p.resistance_lookback, p.vol_ma_period) + 2
    if n < min_warmup:
        return {
            "ok": False,
            "reason": f"need_at_least_{min_warmup}_bars",
            "phase": BPCPhase.IDLE.value,
        }

    highs: List[float] = []
    lows: List[float] = []
    closes: List[float] = []
    opens: List[float] = []
    qvols: List[float] = []
    for k in klines:
        o, h, l, c = _parse_klines_row(k)
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        qvols.append(_quote_vol(k))

    st = BPCState()

    def reset_idle(reason: str = "") -> None:
        st.phase = BPCPhase.IDLE
        st.breakout_idx = None
        st.breakout_level = 0.0
        st.peak_after_breakout = 0.0
        st.peak_idx = 0
        st.pullback_enter_idx = None
        st.pullback_low = 0.0
        st.continuation_idx = None
        st.continuation_reason = ""
        if reason:
            st.last_invalid_reason = reason

    i = min_warmup - 1
    while i < n:
        if (
            p.continuation_reset_next_bar
            and st.phase == BPCPhase.CONTINUATION
            and st.continuation_idx is not None
            and i > st.continuation_idx
        ):
            reset_idle("")

        resistance = _resistance_level(
            highs, i, p.resistance_lookback, p.resistance_use_swing_reinforce
        )
        vol_ma = _sma(qvols, i, p.vol_ma_period)
        qv = qvols[i]
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        # 状态机瀑布流转：去除了 elif，允许在单根 K 线上产生跨阶段演进
        if st.phase == BPCPhase.IDLE:
            brk = resistance > 0 and c > resistance * (1.0 + p.breakout_eps)
            brk_vol = vol_ma > 0 and qv >= vol_ma * p.breakout_vol_mult
            if brk and brk_vol:
                st.phase = BPCPhase.POST_BREAKOUT
                st.breakout_idx = i
                st.breakout_level = resistance
                st.peak_after_breakout = h
                st.peak_idx = i
                st.last_invalid_reason = ""

        if st.phase == BPCPhase.POST_BREAKOUT:
            bi = st.breakout_idx or i
            lvl = st.breakout_level
            if c < lvl * (1.0 - p.invalidation_pct):
                reset_idle("invalidated_below_breakout_level")
                i += 1
                continue
            
            if h > st.peak_after_breakout:
                st.peak_after_breakout = h
                st.peak_idx = i
            
            peak = st.peak_after_breakout
            band_hi = lvl * (1.0 + p.retest_band_pct)
            retraced = peak > 0 and (peak - l) / peak >= p.min_retrace_from_peak_pct
            touched = l <= band_hi
            if retraced and touched:
                st.phase = BPCPhase.PULLBACK
                st.pullback_enter_idx = i
                st.pullback_low = l

        if st.phase == BPCPhase.PULLBACK:
            bi = st.breakout_idx or 0
            pe = st.pullback_enter_idx or i
            lvl = st.breakout_level
            
            if c < lvl * (1.0 - p.invalidation_pct):
                reset_idle("invalidated_in_pullback")
                i += 1
                continue
                
            st.pullback_low = min(st.pullback_low, l)

            # 修复 Lookahead Bias：严格将突破均量窗口边界限制在当前已走出的 K 线 i+1 内
            burst_end = min(bi + p.breakout_burst_bars, i + 1)
            burst_avg = sum(qvols[bi:burst_end]) / float(max(1, burst_end - bi))

            seg_start = pe
            if i > seg_start:
                pullback_avg_pre = sum(qvols[seg_start:i]) / float(i - seg_start)
            else:
                pullback_avg_pre = qvols[seg_start]
                
            vol_ok = burst_avg > 0 and pullback_avg_pre <= burst_avg * p.pullback_vol_ratio_max

            pin = _is_pin_bar_bullish(o, h, l, c, p.pin_body_mult)
            engulf = False
            if i > 0:
                engulf = _is_bullish_engulfing(
                    opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1], o, h, l, c
                )
                
            prior_hi = _max_high(highs, seg_start, i) if i > seg_start else 0.0
            micro_hi = max(prior_hi, lvl)
            reclaim = p.continuation_close_above_micro_high and c > micro_hi and c > lvl

            if p.single_bar_pullback_pin_relaxes_volume and i == seg_start and (pin or engulf):
                vol_ok = True

            reason = ""
            if pin:
                reason = "pin_bar"
            elif engulf:
                reason = "bullish_engulfing"
            elif reclaim:
                reason = "reclaim_micro_high"

            pattern_ok = pin or engulf or reclaim
            if pattern_ok and vol_ok:
                st.phase = BPCPhase.CONTINUATION
                st.continuation_idx = i
                st.continuation_reason = reason

        i += 1

    burst_avg_f = 0.0
    pullback_avg_f = 0.0
    vol_contracted = False
    if st.breakout_idx is not None:
        bi = st.breakout_idx
        burst_end = min(bi + p.breakout_burst_bars, n)
        burst_avg_f = sum(qvols[bi:burst_end]) / float(max(1, burst_end - bi))
    if st.pullback_enter_idx is not None and st.continuation_idx is not None:
        pe = st.pullback_enter_idx
        ce = st.continuation_idx + 1
        pullback_avg_f = sum(qvols[pe:ce]) / float(max(1, ce - pe))
        vol_contracted = burst_avg_f > 0 and pullback_avg_f <= burst_avg_f * p.pullback_vol_ratio_max

    return {
        "ok": True,
        "phase": st.phase.value,
        "breakout_idx": st.breakout_idx,
        "breakout_level": st.breakout_level,
        "peak_after_breakout": st.peak_after_breakout,
        "peak_idx": st.peak_idx,
        "pullback_enter_idx": st.pullback_enter_idx,
        "pullback_low": st.pullback_low,
        "continuation_idx": st.continuation_idx,
        "continuation_reason": st.continuation_reason,
        "pullback_vol_contracted": vol_contracted,
        "breakout_burst_quote_vol_avg": burst_avg_f,
        "pullback_segment_quote_vol_avg": pullback_avg_f,
        "last_invalid_reason": st.last_invalid_reason,
        "bars": n,
    }


def demo_symbol_from_binance_get(
    api_get_fn: Any,
    symbol: str,
    interval: str = "1h",
    limit: int = 120,
    params: Optional[BPCParams] = None,
) -> Dict[str, Any]:
    """
    通过封装的 api_get_fn 拉取 K 线并评估。
    由于直连节点可能会遇到限流或网络阻断，可以在外层使用类似 LuckyAPI 等第三方代理环境来传递请求。
    部署在 Railway 时，建议保持这里的入参结构以便统一管理超时和重试逻辑。
    """
    kl = api_get_fn("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    if not kl:
        return {"ok": False, "reason": "no_klines"}
    return evaluate_breakout_pullback_continuation(kl, params)
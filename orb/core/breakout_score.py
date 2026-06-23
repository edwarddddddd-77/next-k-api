"""ORB 突破 bar 质量分（Mirage 五因子改编，仅评突破 K 线）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, TYPE_CHECKING, Optional

import pandas as pd

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from orb.core.config import OrbConfig
    from orb.core.signals import OrbSignal


@dataclass
class BreakoutScoreResult:
    score: float = 0.0
    close_pos: float = 0.0
    reclaim_atr: float = 0.0
    clean_wick_atr: float = 0.0
    atr: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "breakout_score": round(self.score, 2),
            "breakout_close_pos": round(self.close_pos, 4),
            "breakout_reclaim_atr": round(self.reclaim_atr, 4),
            "breakout_clean_wick_atr": round(self.clean_wick_atr, 4),
            "atr": round(self.atr, 6),
        }


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _vol_comp(volume: float, vol_sma: float, *, vol_mult: float = 1.5) -> float:
    if vol_sma <= 0:
        return 0.5
    vol_ratio = volume / vol_sma
    denom = max(vol_mult - 1.0, 0.1)
    return _clamp((vol_ratio - 1.0) / denom, 0.0, 1.0)


def score_breakout_bar(
    *,
    side: str,
    level: float,
    open_: float,
    high: float,
    low: float,
    close: float,
    atr: float,
    vol_comp: float = 0.5,
    htf_comp: float = 0.5,
) -> float:
    """
    突破 bar 质量分 0-100。

    权重：突破幅度 30% | 收盘位置 25% | 成交量 20% | K 线干净度 15% | HTF 10%（默认中性 0.5）
    """
    if level <= 0 or atr <= 0:
        return 0.0
    rng = high - low
    if rng <= 0:
        return 0.0
    side_u = str(side).upper()
    close_pos = (close - low) / rng
    if side_u == "LONG":
        reclaim = close - level
        cp_comp = close_pos
        opp_wick = high - max(open_, close)
    elif side_u == "SHORT":
        reclaim = level - close
        cp_comp = 1.0 - close_pos
        opp_wick = min(open_, close) - low
    else:
        return 0.0
    rcl_comp = _clamp(max(reclaim, 0.0) / atr, 0.0, 1.0)
    clean_comp = 1.0 - _clamp(max(opp_wick, 0.0) / atr, 0.0, 1.0)
    raw = rcl_comp * 0.30 + cp_comp * 0.25 + vol_comp * 0.20 + clean_comp * 0.15 + htf_comp * 0.10
    return raw * 100.0


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def analyze_breakout_bar(
    session_df: pd.DataFrame,
    *,
    or_high: float,
    or_low: float,
    signal_bar_open_ms: int,
    side: str,
    atr_period: int = 14,
    vol_ma_period: int = 21,
    vol_mult: float = 1.5,
) -> BreakoutScoreResult:
    """对信号 bar 计算突破质量分（用当日前序 K 线估 ATR / 量均线）。"""
    out = BreakoutScoreResult()
    if session_df.empty or or_high <= or_low:
        return out

    df = session_df.sort_values("open_time").reset_index(drop=True)
    win = df[df["open_time"] <= int(signal_bar_open_ms)].copy()
    if win.empty:
        return out

    vol = win["volume"].astype(float)
    vol_sma = vol.rolling(vol_ma_period, min_periods=1).mean()
    atr_s = _atr_series(win, period=atr_period)
    last_atr = float(atr_s.iloc[-1]) if len(atr_s) else 0.0
    out.atr = last_atr

    sig_rows = win[win["open_time"] == int(signal_bar_open_ms)]
    if sig_rows.empty or last_atr <= 0:
        return out

    row = sig_rows.iloc[-1]
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    rng = h - l
    side_u = str(side).upper()
    level = or_high if side_u == "LONG" else or_low
    sig_idx = sig_rows.index[-1]
    v_comp = _vol_comp(
        float(row["volume"]),
        float(vol_sma.loc[sig_idx]) if sig_idx in vol_sma.index else 0.0,
        vol_mult=vol_mult,
    )
    if rng > 0:
        close_pos = (c - l) / rng
        out.close_pos = close_pos if side_u == "LONG" else 1.0 - close_pos
    if side_u == "LONG":
        out.reclaim_atr = max(c - or_high, 0.0) / last_atr
        out.clean_wick_atr = max(h - max(o, c), 0.0) / last_atr
    else:
        out.reclaim_atr = max(or_low - c, 0.0) / last_atr
        out.clean_wick_atr = max(min(o, c) - l, 0.0) / last_atr
    out.score = score_breakout_bar(
        side=side_u,
        level=level,
        open_=o,
        high=h,
        low=l,
        close=c,
        atr=last_atr,
        vol_comp=v_comp,
    )
    return out


def _df5_has_entry_bar(df5: pd.DataFrame, entry_bar_open_ms: int, *, bar_step_ms: int, now_ms: int) -> bool:
    if df5 is None or df5.empty or entry_bar_open_ms <= 0:
        return False
    opens = {int(x) for x in df5["open_time"].tolist()}
    if int(entry_bar_open_ms) not in opens:
        return False
    last_open = int(df5["open_time"].iloc[-1])
    return last_open + int(bar_step_ms) >= int(now_ms)


def df5_for_breakout_score(
    sym: str,
    sig: "OrbSignal",
    cfg: "OrbConfig",
    *,
    session_day: str,
    now_ms: int,
    df5_cache: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """突破分用 5m K 线：优先 scan 缓存 / 本地文件，缺失时回退实时 Binance。"""
    from orb.core.kline_cache import load_klines
    from orb.core.paper import _load_signal_df

    bo = int(sig.entry_bar_open_ms or 0)
    bar_step = cfg.bar_step_ms()
    cached = df5_cache.get(sym) if df5_cache is not None and sym in df5_cache else None
    if _df5_has_entry_bar(cached, bo, bar_step_ms=bar_step, now_ms=now_ms):
        logger.debug(
            "[breakout_score] %s kline source=scan_cache rows=%d entry_bar=%s",
            sym,
            len(cached),
            bo,
        )
        return cached

    fetch_start, end_ms = breakout_kline_range_ms(session_day, cfg)
    disk_df = load_klines(
        sym,
        cfg.signal_interval,
        start_ms=fetch_start,
        end_ms=max(int(end_ms), int(now_ms)),
    )
    if _df5_has_entry_bar(disk_df, bo, bar_step_ms=bar_step, now_ms=now_ms):
        if df5_cache is not None:
            df5_cache[sym] = disk_df
        logger.info(
            "[breakout_score] %s kline source=disk_cache rows=%d entry_bar=%s",
            sym,
            len(disk_df),
            bo,
        )
        return disk_df

    if bo > 0:
        live_df = _load_signal_df(sym, cfg, now_ms=int(now_ms))
        if live_df is not None and not live_df.empty:
            if df5_cache is not None:
                df5_cache[sym] = live_df
            if cached is None or getattr(cached, "empty", True) or not _df5_has_entry_bar(cached, bo, bar_step_ms=bar_step, now_ms=now_ms):
                logger.info(
                    "[breakout_score] %s kline source=live_api rows=%d entry_bar=%s",
                    sym,
                    len(live_df),
                    bo,
                )
            return live_df

    logger.warning(
        "[breakout_score] %s no klines for breakout score (entry_bar=%s session=%s)",
        sym,
        bo,
        session_day,
    )
    return pd.DataFrame()


def breakout_kline_range_ms(session_date: str, cfg: "OrbConfig") -> tuple[int, int]:
    """突破分所需 5m K 线窗口（与 live_gate_sim 预热一致）。"""
    from orb.core.session import extended_fetch_anchor_ms, session_anchor_ms, session_close_ms

    tz = cfg.session_tz
    ts = pd.Timestamp(f"{session_date} 12:00:00", tz=tz)
    anchor = session_anchor_ms(int(ts.value // 1_000_000), tz=tz, session_open_time=cfg.session_open_time)
    close = session_close_ms(anchor, tz=tz, session_close_time=cfg.session_close_time)
    if close is None:
        close = anchor + 6 * 60 * 60 * 1000
    bar = cfg.bar_step_ms()
    warmup = cfg.daily_atr_warmup_ms() + bar * 96
    fetch_start = extended_fetch_anchor_ms(anchor, cfg) - warmup
    end_ms = int(close) + bar * 4
    return int(fetch_start), int(end_ms)


def passes_breakout_score(
    score: Optional[float], *, min_score: float = 50.0
) -> tuple[bool, str]:
    """min_score<=0 表示不过滤。"""
    if min_score <= 0:
        return True, "open_ok"
    if score is None:
        return False, "breakout_score_missing"
    if float(score) < min_score:
        return False, f"breakout_score<{min_score:.0f}"
    return True, "open_ok"


def breakout_score_for_signal(
    sig: "OrbSignal",
    df5: pd.DataFrame,
    cfg: "OrbConfig",
    *,
    now_ms: int,
) -> float:
    """从 ORB 信号 + 5m K 线计算突破 bar 质量分（供 live_gate 回测标注）。"""
    from orb.core.paper import _signal_df_from_bars

    bo = int(sig.entry_bar_open_ms or 0)
    if bo <= 0 or df5.empty:
        return 0.0
    or_high = float(sig.or_high or 0)
    or_low = float(sig.or_low or 0)
    if or_high <= or_low:
        return 0.0
    sig_df = _signal_df_from_bars(df5, cfg, now_ms=int(now_ms))
    if sig_df.empty:
        return 0.0
    vol_mult = float(cfg.vol_mult or 0)
    vol_mult = max(vol_mult, 1.5) if vol_mult > 0 else 1.5
    result = analyze_breakout_bar(
        sig_df,
        or_high=or_high,
        or_low=or_low,
        signal_bar_open_ms=bo,
        side=str(sig.side),
        vol_ma_period=int(cfg.vol_ma_period),
        vol_mult=vol_mult,
    )
    return float(result.score)

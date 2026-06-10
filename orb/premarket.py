"""盘前数据指标与 ORB 过滤器（美股 RTH 09:30 前窗口）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from orb.config import OrbConfig
from orb.session import session_anchor_ms, session_day_str
from orb.us_equity_calendar import is_us_equity_market


def uses_alpaca_premarket(cfg: OrbConfig) -> bool:
    if not cfg.premarket_filter or not is_us_equity_market(cfg.market):
        return False
    return (cfg.premarket_source or "alpaca").strip().lower() == "alpaca"


def uses_binance_premarket(cfg: OrbConfig) -> bool:
    if not cfg.premarket_filter or not is_us_equity_market(cfg.market):
        return False
    return (cfg.premarket_source or "alpaca").strip().lower() == "binance"


def premarket_anchor_ms(
    open_ms: int,
    *,
    tz: str,
    session_open_time: str,
    premarket_open_time: str,
) -> int:
    """盘前起点：当日 session 日历日 + premarket_open_time（默认 04:00 ET）。"""
    rth_anchor = session_anchor_ms(open_ms, tz=tz, session_open_time=session_open_time)
    tz_name = tz or "UTC"
    ts = pd.Timestamp(int(rth_anchor), unit="ms", tz=tz_name)
    parts = (premarket_open_time or "04:00").strip().split(":")
    hour = int(parts[0]) if parts else 4
    minute = int(parts[1]) if len(parts) > 1 else 0
    pm_ts = ts.normalize() + pd.Timedelta(hours=hour, minutes=minute)
    return int(pm_ts.value // 1_000_000)


def premarket_slice(
    df: pd.DataFrame,
    asof_open_ms: int,
    *,
    tz: str,
    session_open_time: str,
    premarket_open_time: str,
) -> pd.DataFrame:
    """[盘前起点, RTH 开盘) 内的 K 线（不含 09:30 那根）。"""
    if df.empty:
        return df
    rth = session_anchor_ms(asof_open_ms, tz=tz, session_open_time=session_open_time)
    pm = premarket_anchor_ms(
        asof_open_ms,
        tz=tz,
        session_open_time=session_open_time,
        premarket_open_time=premarket_open_time,
    )
    if int(asof_open_ms) < pm:
        return df.iloc[0:0].copy()
    end = min(int(asof_open_ms), rth - 1)
    if end < pm:
        return df.iloc[0:0].copy()
    return df[(df["open_time"] >= pm) & (df["open_time"] <= end)].copy()


def extended_fetch_anchor_ms(asof_open_ms: int, cfg: OrbConfig) -> int:
    """Binance 信号 K 线拉取起点：Alpaca 盘前时仍从 RTH 09:30 起。"""
    if uses_binance_premarket(cfg):
        return premarket_anchor_ms(
            asof_open_ms,
            tz=cfg.session_tz,
            session_open_time=cfg.session_open_time,
            premarket_open_time=cfg.premarket_open_time,
        )
    return session_anchor_ms(
        asof_open_ms, tz=cfg.session_tz, session_open_time=cfg.session_open_time
    )


def prev_daily_close(
    daily_df: pd.DataFrame,
    asof_open_ms: int,
    *,
    tz: str,
) -> Optional[float]:
    """asof 所属 session 日之前最近一根已完成日 K 的收盘价。"""
    if daily_df.empty:
        return None
    df = daily_df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time")
    tz_name = tz or "UTC"
    asof_day = pd.Timestamp(int(asof_open_ms), unit="ms", tz=tz_name).normalize()
    day_ts = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(tz_name).dt.normalize()
    completed = df[day_ts < asof_day]
    if completed.empty:
        return None
    val = float(completed["close"].iloc[-1])
    return val if val > 0 else None


def _pm_vwap(pm_df: pd.DataFrame) -> float:
    if pm_df.empty:
        return 0.0
    typical = (
        pm_df["high"].astype(float) + pm_df["low"].astype(float) + pm_df["close"].astype(float)
    ) / 3.0
    vol = pm_df["volume"].astype(float)
    total = float(vol.sum())
    if total <= 0:
        return float(pm_df["close"].iloc[-1])
    return float((typical * vol).sum() / total)


def _session_pm_volume(
    full_df: pd.DataFrame,
    session_date: str,
    *,
    cfg: OrbConfig,
) -> float:
    tz = cfg.session_tz
    day_open = pd.Timestamp(session_date, tz=tz)
    asof = int((day_open + pd.Timedelta(hours=16)).value // 1_000_000)
    pm = premarket_slice(
        full_df,
        asof,
        tz=tz,
        session_open_time=cfg.session_open_time,
        premarket_open_time=cfg.premarket_open_time,
    )
    if pm.empty:
        return 0.0
    return float(pm["volume"].astype(float).sum())


def _historical_pm_volumes(
    full_df: pd.DataFrame,
    asof_open_ms: int,
    *,
    cfg: OrbConfig,
) -> List[float]:
    """过去 N 个交易日的盘前成交量（不含当日）。"""
    if full_df.empty:
        return []
    tz = cfg.session_tz
    lookback = max(1, int(cfg.premarket_rvol_lookback))
    today = session_day_str(
        asof_open_ms, tz=tz, session_open_time=cfg.session_open_time
    )
    day_series = pd.to_datetime(full_df["open_time"], unit="ms", utc=True).dt.tz_convert(tz)
    dates = sorted({d.strftime("%Y-%m-%d") for d in day_series.dt.normalize().unique() if d.strftime("%Y-%m-%d") < today})
    vols: List[float] = []
    for d in dates[-lookback:]:
        v = _session_pm_volume(full_df, d, cfg=cfg)
        if v > 0:
            vols.append(v)
    return vols


@dataclass
class PremarketStats:
    pm_high: float = 0.0
    pm_low: float = 0.0
    pm_vwap: float = 0.0
    pm_volume: float = 0.0
    pm_rvol: float = 0.0
    gap_pct: float = 0.0
    rth_open: float = 0.0
    prev_close: float = 0.0
    regime: str = "neutral"
    pm_bars: int = 0
    pm_late_below_vwap: bool = False
    pm_rvol_hist_days: int = 0

    def reason_tags(self) -> List[str]:
        out = [
            f"pm_h={self.pm_high:.4f}",
            f"pm_l={self.pm_low:.4f}",
            f"pm_vwap={self.pm_vwap:.4f}",
            f"pm_vol={self.pm_volume:.0f}",
            f"pm_rvol={self.pm_rvol:.2f}",
            f"gap={self.gap_pct:.2f}%",
            f"pm_regime={self.regime}",
        ]
        return out


def compute_premarket_stats(
    full_df: pd.DataFrame,
    asof_open_ms: int,
    *,
    cfg: OrbConfig,
    daily_df: Optional[pd.DataFrame] = None,
    session_df: Optional[pd.DataFrame] = None,
    pm_history_df: Optional[pd.DataFrame] = None,
    pm_daily_df: Optional[pd.DataFrame] = None,
) -> PremarketStats:
    """计算盘前指标；Alpaca 模式用 pm_history_df / pm_daily_df，Binance 模式用 full_df。"""
    stats = PremarketStats()
    if not cfg.premarket_filter or not is_us_equity_market(cfg.market):
        return stats

    pm_source_df = pm_history_df if pm_history_df is not None else full_df
    gap_daily_df = pm_daily_df if pm_daily_df is not None else daily_df

    pm_df = premarket_slice(
        pm_source_df,
        asof_open_ms,
        tz=cfg.session_tz,
        session_open_time=cfg.session_open_time,
        premarket_open_time=cfg.premarket_open_time,
    )
    stats.pm_bars = len(pm_df)
    if not pm_df.empty:
        stats.pm_high = float(pm_df["high"].max())
        stats.pm_low = float(pm_df["low"].min())
        stats.pm_vwap = _pm_vwap(pm_df)
        stats.pm_volume = float(pm_df["volume"].astype(float).sum())
        hist = _historical_pm_volumes(
            pm_source_df, asof_open_ms, cfg=cfg
        )
        stats.pm_rvol_hist_days = len(hist)
        avg_hist = float(sum(hist) / len(hist)) if hist else 0.0
        stats.pm_rvol = stats.pm_volume / avg_hist if avg_hist > 0 else 0.0
        late_n = max(1, len(pm_df) // 3)
        late = pm_df.iloc[-late_n:]
        late_vwap = _pm_vwap(late) if len(late) else stats.pm_vwap
        stats.pm_late_below_vwap = late_vwap < stats.pm_vwap if stats.pm_vwap > 0 else False

    sess = session_df
    if sess is None:
        from orb.session import session_slice

        sess = session_slice(
            full_df,
            asof_open_ms,
            tz=cfg.session_tz,
            session_open_time=cfg.session_open_time,
        )

    rth_open: Optional[float] = None
    if pm_history_df is not None and not pm_history_df.empty and uses_alpaca_premarket(cfg):
        from orb.providers.alpaca import alpaca_rth_open

        rth_open = alpaca_rth_open(pm_history_df, int(asof_open_ms), cfg=cfg)
    if rth_open is None and not sess.empty:
        rth_open = float(sess["open"].iloc[0])
    if rth_open is not None and rth_open > 0:
        stats.rth_open = rth_open

    if gap_daily_df is not None and not gap_daily_df.empty:
        pc = prev_daily_close(gap_daily_df, asof_open_ms, tz=cfg.session_tz)
        if pc is not None:
            stats.prev_close = pc
            if stats.rth_open > 0 and pc > 0:
                stats.gap_pct = (stats.rth_open - pc) / pc * 100.0

    stats.regime = classify_premarket_regime(stats, cfg)
    return stats


def classify_premarket_regime(stats: PremarketStats, cfg: OrbConfig) -> str:
    """Gap and Go / Gap and Fade / neutral。"""
    if stats.pm_bars == 0:
        return "neutral"

    mode = (cfg.premarket_mode or "enhanced").strip().lower()
    if mode != "gap_go_fade":
        return "neutral"

    gap_up = stats.gap_pct >= cfg.premarket_min_gap_pct
    gap_down = stats.gap_pct <= -cfg.premarket_min_gap_pct if cfg.premarket_min_gap_pct > 0 else stats.gap_pct < 0

    vol_ok = stats.pm_volume >= cfg.premarket_min_volume if cfg.premarket_min_volume > 0 else True
    rvol_ok = stats.pm_rvol >= cfg.premarket_rvol_min if cfg.premarket_rvol_min > 0 else True
    above_vwap = stats.rth_open >= stats.pm_vwap if stats.pm_vwap > 0 else True

    if gap_up and vol_ok and rvol_ok and above_vwap and not stats.pm_late_below_vwap:
        return "gap_and_go"
    if gap_up and (not vol_ok or not rvol_ok or stats.pm_late_below_vwap):
        return "gap_and_fade"
    if gap_down and vol_ok and rvol_ok and not above_vwap:
        return "gap_and_go_short"
    if gap_down and (not vol_ok or not rvol_ok or above_vwap):
        return "gap_and_fade_short"
    return "neutral"


def apply_premarket_filter(
    side: str,
    stats: PremarketStats,
    *,
    cfg: OrbConfig,
    entry_px: float,
    session_high: float,
    session_low: float,
    or_high: float,
    or_low: float,
) -> Tuple[bool, str]:
    """
    返回 (allowed, reject_reason)。
    stats.pm_bars==0 时不拦截（数据不足时放行，避免误杀）。
    """
    if not cfg.premarket_filter or not is_us_equity_market(cfg.market):
        return True, ""
    if stats.pm_bars == 0:
        return True, ""

    side_u = str(side).upper()
    mode = (cfg.premarket_mode or "enhanced").strip().lower()

    if stats.pm_volume > 0 and cfg.premarket_min_volume > 0 and stats.pm_volume < cfg.premarket_min_volume:
        return False, "pm_vol_low"
    if cfg.premarket_rvol_min > 0 and stats.pm_rvol_hist_days >= 3:
        if stats.pm_rvol < cfg.premarket_rvol_min:
            return False, "pm_rvol_low"
    if cfg.premarket_max_gap_pct > 0 and abs(stats.gap_pct) > cfg.premarket_max_gap_pct:
        return False, "gap_too_wide"

    if mode == "gap_go_fade":
        if side_u == "LONG" and stats.regime in ("gap_and_fade", "gap_and_fade_short"):
            return False, "gap_fade_block_long"
        if side_u == "SHORT" and stats.regime in ("gap_and_go", "gap_and_go_short"):
            return False, "gap_go_block_short"
        if side_u == "LONG" and stats.regime == "gap_and_go":
            pass  # 继续检查 PMH/VWAP
        elif side_u == "SHORT" and stats.regime == "gap_and_fade":
            pass

    buf_bps = max(0.0, float(getattr(cfg, "premarket_pmh_buffer_bps", 0.0) or 0.0))
    buf_frac = buf_bps / 10_000.0

    if side_u == "LONG":
        if cfg.premarket_require_pmh_long and stats.pm_high > 0:
            eff_pmh = stats.pm_high * (1.0 - buf_frac)
            if session_high <= eff_pmh:
                return False, "below_pmh"
        if cfg.premarket_require_vwap_long and stats.pm_vwap > 0 and entry_px <= stats.pm_vwap:
            return False, "below_pm_vwap"
    elif side_u == "SHORT":
        if cfg.premarket_require_pml_short and stats.pm_low > 0:
            eff_pml = stats.pm_low * (1.0 + buf_frac)
            if session_low >= eff_pml:
                return False, "above_pml"
        if cfg.premarket_require_vwap_short and stats.pm_vwap > 0 and entry_px >= stats.pm_vwap:
            return False, "above_pm_vwap"
        if mode == "gap_go_fade" and stats.regime == "gap_and_fade":
            if session_high < or_high * 0.999 and stats.pm_high > or_high:
                return False, "fade_no_fakeout"

    return True, ""

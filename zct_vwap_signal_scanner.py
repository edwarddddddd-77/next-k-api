#!/usr/bin/env python3
"""
ZCT 风格 VWAP + 关键位 量化信号扫描（币安 U 本位永续）

依据材料中的框架实现可复现规则（非投资建议）：
- 会话 VWAP（UTC 日锚）与 ±1σ 带宽；斜率 / 带宽宽度 / 相对位置；会话内 VWAP 交叉次数
- 体制：顺势（陡斜率+宽轨） vs 反转（平斜率+窄轨 + 可选 MA30 纠缠）
- ZCT 层级参考：前日高/低 + 4H/1H/15m 前一根完整 K 的高/低

用法：
  python zct_vwap_signal_scanner.py              # 跑一次，stdout + 可选 TG
  python zct_vwap_signal_scanner.py --no-tg     # 仅打印

定时：由 next-k-api main.py APScheduler 调用（需 ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED=1），
      默认东八区每 15 分钟一次（xx:00/15/30/45）；亦可自建 cron 执行本脚本。

环境变量：
  ZCT_VWAP_SYMBOLS     逗号分隔永续标的；不设则默认含 BTC/ETH/SOL 及 XRP、ADA、
                        1000SHIB、1000PEPE、DOGE、BNB（见 _DEFAULT_ZCT_SYMBOLS）
  ZCT_VWAP_BAND_SIGMA  默认 1.0
  ZCT_VWAP_DB_SKIP_FLAT  设为 1 时不入库 side=FLAT 的行（减轻 NO_TRADE 噪音）
  TG_BOT_TOKEN / TG_CHAT_ID  与 accumulation 雷达相同；配置后即推送 Telegram
  ZCT_VWAP_TG_PUSH_MODE  扫描推送：summary（默认，每轮一条简报）| actionable（仅当有方向+SL/TP）
                        | all（每轮全文明细）| off（不推扫描，平仓推送仍受 NOTIFY_RESOLVE 控制）
  ZCT_VWAP_TG_NOTIFY_RESOLVE  平仓结算是否推 TG，默认 1

入库：accumulation.db 表 zct_vwap_signals（与 init_db 同源）。每笔定向信号写入
sl_price / tp_price / r_unit / entry_bar_open_ms；每次扫描后自动用后续 1m K 线判定先触发
止损或止盈，回填 outcome、exit_price、pnl_r（含超时 expired）。

环境变量（止盈止损）：
  ZCT_SWING_LOOKBACK      摆动窗口（根 1m），默认 20
  ZCT_MIN_SL_PCT          最小止损距离（占价比），默认 0.003
  ZCT_SL_BUFFER_BPS       σ 带 / 摆动外侧缓冲（基点），默认 2
  ZCT_RESOLVE_MAX_BARS    未触轨最长等待根数，默认 720（约 12h）
  ZCT_SAME_BAR_RULE       pessimistic | optimistic，同根同时触轨时先后，默认 pessimistic
  ZCT_VIRTUAL_NOTIONAL_USDT  虚拟仓位名义（USDT），默认 100；平仓时按 (价差/入场价)×名义 记 pnl_usdt

统计示例：

  SELECT outcome, COUNT(*) FROM zct_vwap_signals
    WHERE side IN ('LONG','SHORT') GROUP BY outcome;

  SELECT AVG(pnl_r) FROM zct_vwap_signals WHERE outcome='win';

  SELECT SUM(pnl_usdt) FROM zct_vwap_signals WHERE pnl_usdt IS NOT NULL;
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import io
import time
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass

FAPI = "https://fapi.binance.com"

# === 加载 .env（与 accumulation_radar.py：next-k-api/.env.oi）===
_env_file = Path(__file__).parent / ".env.oi"
if _env_file.exists():
    with open(_env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
# Telegram 推送：扫描结果 — summary=每轮简报（默认）；actionable=仅当有 LONG/SHORT 且含 SL/TP；
# all=每轮全文；off=不推扫描（仍打印 stdout）
TG_PUSH_MODE = os.getenv("ZCT_VWAP_TG_PUSH_MODE", "summary").strip().lower()
# 平仓结算是否单独推一条（平仓 id / 结果 / R / USDT）
TG_NOTIFY_RESOLVE = os.getenv("ZCT_VWAP_TG_NOTIFY_RESOLVE", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


# 默认监控 U 本位永续（可通过 ZCT_VWAP_SYMBOLS 覆盖）。
# SHIB/PEPE 在币安合约为 1000SHIBUSDT、1000PEPEUSDT（标的报价按「千枚」计）。
_DEFAULT_ZCT_SYMBOLS = (
    "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,ADAUSDT,1000SHIBUSDT,1000PEPEUSDT,"
    "DOGEUSDT,BNBUSDT"
)


def _symbols_from_env() -> List[str]:
    raw = os.getenv("ZCT_VWAP_SYMBOLS", _DEFAULT_ZCT_SYMBOLS).strip()
    parts = [x.strip().upper() for x in raw.split(",") if x.strip()]
    return parts or [x.strip() for x in _DEFAULT_ZCT_SYMBOLS.split(",") if x.strip()]


BAND_SIGMA = float(os.getenv("ZCT_VWAP_BAND_SIGMA", "1.0"))
# 斜率：最近 VWAP_SLOPE_BARS 根 1m K 上 VWAP 变化（基点）
VWAP_SLOPE_BARS = int(os.getenv("ZCT_VWAP_SLOPE_BARS", "20"))
SLOPE_STEEP_BPS = float(os.getenv("ZCT_SLOPE_STEEP_BPS", "2.5"))   # >= 视为陡
SLOPE_FLAT_BPS = float(os.getenv("ZCT_SLOPE_FLAT_BPS", "0.8"))    # <= 视为平
# 带宽：相对本会话带宽序列的中位数
WIDE_BAND_MULT = float(os.getenv("ZCT_WIDE_BAND_MULT", "1.15"))
TIGHT_BAND_MULT = float(os.getenv("ZCT_TIGHT_BAND_MULT", "0.88"))
# chop：会话内 VWAP 交叉次数
CHOPPY_CROSS_MIN = int(os.getenv("ZCT_CHOPPY_CROSS_MIN", "7"))
# MA30 交叉（近 MA_LOOKBACK 根）
MA_PERIOD = int(os.getenv("ZCT_MA_PERIOD", "30"))
MA_CHOPPY_CROSS_MIN = int(os.getenv("ZCT_MA_CHOPPY_CROSS_MIN", "10"))
MA_LOOKBACK = int(os.getenv("ZCT_MA_LOOKBACK", "120"))
# 触碰 σ 带判定：收盘距上下轨 within this fraction of band width
BAND_TOUCH_FRAC = float(os.getenv("ZCT_BAND_TOUCH_FRAC", "0.35"))
DB_SKIP_FLAT = os.getenv("ZCT_VWAP_DB_SKIP_FLAT", "").strip().lower() in ("1", "true", "yes", "on")

SWING_LOOKBACK = int(os.getenv("ZCT_SWING_LOOKBACK", "20"))
MIN_SL_PCT = float(os.getenv("ZCT_MIN_SL_PCT", "0.003"))
SL_BUFFER_BPS = float(os.getenv("ZCT_SL_BUFFER_BPS", "2"))
RESOLVE_MAX_BARS = int(os.getenv("ZCT_RESOLVE_MAX_BARS", "720"))
SAME_BAR_RULE = os.getenv("ZCT_SAME_BAR_RULE", "pessimistic").strip().lower()
# 虚拟仓位：每笔信号按固定名义 USDT 计纸面盈亏（U 本位线性近似）
VIRTUAL_NOTIONAL_USDT = float(os.getenv("ZCT_VIRTUAL_NOTIONAL_USDT", "100"))


def api_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    url = f"{FAPI}{endpoint}"
    for attempt in range(3):
        try:
            r = requests.get(url, params=params or {}, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(1.5)
            else:
                return None
        except Exception:
            time.sleep(0.8)
    return None


def fetch_klines(symbol: str, interval: str, limit: int) -> List[List[Any]]:
    data = api_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    return data if isinstance(data, list) else []


def fetch_klines_forward(symbol: str, interval: str, start_ms: int, end_ms: Optional[int] = None) -> List[List[Any]]:
    """从 start_ms 起分页拉取 K 线（含 start_ms 所在根），直到 end_ms 或接口无数据。"""
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    out: List[List[Any]] = []
    cur = start_ms
    cap = 20000
    while cur <= end_ms and len(out) < cap:
        batch = api_get(
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": int(cur),
                "endTime": int(end_ms),
                "limit": 1500,
            },
        )
        if not batch or not isinstance(batch, list):
            break
        out.extend(batch)
        last_open = int(batch[-1][0])
        nxt = last_open + 1
        if nxt <= cur:
            break
        cur = nxt
        if len(batch) < 1500:
            break
        time.sleep(0.05)
    return out


def klines_to_df(rows: List[List[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[[0, 1, 2, 3, 4, 5]].copy()
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["open_time"] = df["open_time"].astype(np.int64)
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def compute_vwap_bands_session(df: pd.DataFrame, sigma: float) -> pd.DataFrame:
    """UTC 当日会话内累积 VWAP 与 ±sigma 加权标准差轨。"""
    if df.empty:
        return df
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    v = df["volume"].values
    tpv = tp.values
    cum_pv = np.cumsum(tpv * v)
    cum_v = np.cumsum(v)
    vwap = cum_pv / np.maximum(cum_v, 1e-12)
    upper = np.zeros(len(df))
    lower = np.zeros(len(df))
    for i in range(len(df)):
        cv = cum_v[i]
        if cv <= 0:
            upper[i] = lower[i] = vwap[i]
            continue
        dev = tpv[: i + 1] - vwap[i]
        var = np.sum(v[: i + 1] * (dev ** 2)) / cv
        std = float(np.sqrt(max(var, 0.0)))
        upper[i] = vwap[i] + sigma * std
        lower[i] = vwap[i] - sigma * std
    out = df.copy()
    out["typical"] = tp
    out["vwap"] = vwap
    out["vwap_upper"] = upper
    out["vwap_lower"] = lower
    out["band_width_pct"] = np.where(
        vwap > 0, (upper - lower) / vwap * 100.0, 0.0
    )
    return out


def count_vwap_crosses(close: np.ndarray, vwap: np.ndarray) -> int:
    if len(close) < 2:
        return 0
    diff = close - vwap
    signs = np.sign(diff)
    crosses = 0
    for i in range(1, len(signs)):
        if signs[i] == 0 or signs[i - 1] == 0:
            continue
        if signs[i] != signs[i - 1]:
            crosses += 1
    return crosses


def count_ma_crosses(close: np.ndarray, ma: np.ndarray) -> int:
    valid = ~np.isnan(ma)
    if np.sum(valid) < 2:
        return 0
    diff = close - ma
    crosses = 0
    prev_s = None
    for i in range(len(diff)):
        if not valid[i]:
            continue
        s = np.sign(diff[i])
        if s == 0:
            continue
        if prev_s is not None and s != prev_s:
            crosses += 1
        prev_s = s
    return crosses


def session_cut_utc(df: pd.DataFrame) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return df[df["ts"] >= pd.Timestamp(day_start)].copy()


def ref_levels(symbol: str) -> Dict[str, float]:
    """前日高/低；各周期上一根完整 K 的 high/low（ZCT 层级参考）。"""
    out: Dict[str, float] = {}
    d1 = fetch_klines(symbol, "1d", 3)
    if len(d1) >= 2:
        prev = d1[-2]
        out["pdh"] = float(prev[2])
        out["pdl"] = float(prev[3])
    for interval, key_pfx in [("4h", "h4"), ("1h", "h1"), ("15m", "m15")]:
        kl = fetch_klines(symbol, interval, 4)
        if len(kl) >= 2:
            prev_bar = kl[-2]
            out[f"{key_pfx}_high"] = float(prev_bar[2])
            out[f"{key_pfx}_low"] = float(prev_bar[3])
    return out


def slope_bps(vwap_series: pd.Series, bars: int) -> float:
    if len(vwap_series) < bars + 1:
        return 0.0
    a = float(vwap_series.iloc[-bars - 1])
    b = float(vwap_series.iloc[-1])
    if a <= 0:
        return 0.0
    return (b / a - 1.0) * 10000.0


def nearest_level_distance_pct(price: float, levels: Dict[str, float]) -> List[Tuple[str, float, float]]:
    """返回 (name, level, dist_pct) 按距离排序。"""
    rows: List[Tuple[str, float, float]] = []
    for k, lv in levels.items():
        if lv and price > 0:
            rows.append((k, lv, abs(lv / price - 1.0) * 100.0))
    rows.sort(key=lambda x: x[2])
    return rows


@dataclass
class SignalResult:
    symbol: str
    price: float
    regime: str
    play: str
    side: str
    confidence: str
    reasons: List[str]
    vwap: float
    vwap_upper: float
    vwap_lower: float
    slope_bps: float
    band_width_pct: float
    vwap_crosses: int
    ma_crosses: int
    bands_wide: bool
    bands_tight: bool
    slope_steep: bool
    slope_flat: bool
    chop_score: str
    ref_levels: Dict[str, float]
    nearest_levels: List[Dict[str, Any]]
    entry_bar_open_ms: Optional[int] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    r_unit: Optional[float] = None


def compute_sl_tp(r: SignalResult, sdf: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    止损 / 止盈 / 风险单位（1R 的价位距离）。
    - 均值回归 PLAY03：止盈锚定 VWAP；止损在 σ 带外与摆动极值外。
    - 顺势 / 过渡偏置：1R 目标；止损在 VWAP 与近端摆动极值「错误侧」之外。
    """
    if r.side == "FLAT" or sdf is None or sdf.empty:
        return None, None, None
    n = min(SWING_LOOKBACK, len(sdf))
    lo = float(sdf["low"].iloc[-n:].min())
    hi = float(sdf["high"].iloc[-n:].max())
    buf = SL_BUFFER_BPS / 10000.0
    entry = r.price
    vw, vu, vl = r.vwap, r.vwap_upper, r.vwap_lower

    def clamp_long_sl(sl: float) -> float:
        if sl >= entry:
            sl = entry * (1 - MIN_SL_PCT) - 1e-12
        if entry - sl < entry * MIN_SL_PCT:
            return entry * (1 - MIN_SL_PCT)
        return sl

    def clamp_short_sl(sl: float) -> float:
        if sl <= entry:
            sl = entry * (1 + MIN_SL_PCT) + 1e-12
        if sl - entry < entry * MIN_SL_PCT:
            return entry * (1 + MIN_SL_PCT)
        return sl

    if r.side == "LONG":
        if r.play == "PLAY03_REV_LONG":
            tp = vw
            sl = min(lo, vl) * (1 - buf)
            sl = clamp_long_sl(sl)
            ru = entry - sl
            return round(sl, 8), round(tp, 8), round(ru, 8)
        sl = min(vw, lo) * (1 - buf)
        sl = clamp_long_sl(sl)
        ru = entry - sl
        tp = entry + ru
        return round(sl, 8), round(tp, 8), round(ru, 8)

    if r.side == "SHORT":
        if r.play == "PLAY03_REV_SHORT":
            tp = vw
            sl = max(hi, vu) * (1 + buf)
            sl = clamp_short_sl(sl)
            ru = sl - entry
            return round(sl, 8), round(tp, 8), round(ru, 8)
        sl = max(vw, hi) * (1 + buf)
        sl = clamp_short_sl(sl)
        ru = sl - entry
        tp = entry - ru
        return round(sl, 8), round(tp, 8), round(ru, 8)

    return None, None, None


def _bar_hit_long(
    _o: float, h: float, l: float, sl: float, tp: float
) -> Tuple[Optional[str], float]:
    """同根同时触轨：pessimistic 先判止损；optimistic 先判止盈。返回 (outcome_win_loss, exit_price)。"""
    hit_sl = l <= sl
    hit_tp = h >= tp
    if hit_sl and hit_tp:
        if SAME_BAR_RULE == "optimistic":
            return "win", tp
        return "loss", sl
    if hit_sl:
        return "loss", sl
    if hit_tp:
        return "win", tp
    return None, 0.0


def _bar_hit_short(
    _o: float, h: float, l: float, sl: float, tp: float
) -> Tuple[Optional[str], float]:
    hit_sl = h >= sl
    hit_tp = l <= tp
    if hit_sl and hit_tp:
        if SAME_BAR_RULE == "optimistic":
            return "win", tp
        return "loss", sl
    if hit_sl:
        return "loss", sl
    if hit_tp:
        return "win", tp
    return None, 0.0


def _pnl_r(side: str, entry: float, exit_px: float, sl: float, tp: float) -> float:
    if side == "LONG":
        risk = entry - sl
        if risk <= 0:
            return 0.0
        return (exit_px - entry) / risk
    if side == "SHORT":
        risk = sl - entry
        if risk <= 0:
            return 0.0
        return (entry - exit_px) / risk
    return 0.0


def _pnl_usdt(side: str, entry: float, exit_px: float, notional_usdt: float) -> float:
    """线性 USDT 本位近似：多 (exit-entry)/entry*N，空 (entry-exit)/entry*N。"""
    if entry <= 0 or notional_usdt <= 0:
        return 0.0
    if side == "LONG":
        return notional_usdt * (exit_px - entry) / entry
    if side == "SHORT":
        return notional_usdt * (entry - exit_px) / entry
    return 0.0


def classify_and_signal(
    symbol: str,
    sdf: pd.DataFrame,
    levels: Dict[str, float],
) -> SignalResult:
    last = sdf.iloc[-1]
    price = float(last["close"])
    vw = float(last["vwap"])
    up = float(last["vwap_upper"])
    lo = float(last["vwap_lower"])
    bw = float(last["band_width_pct"])

    close_a = sdf["close"].values
    vwap_a = sdf["vwap"].values
    crosses = count_vwap_crosses(close_a, vwap_a)

    med_bw = float(np.nanmedian(sdf["band_width_pct"].values)) if len(sdf) > 5 else bw
    bands_wide = bw >= med_bw * WIDE_BAND_MULT and med_bw > 0
    bands_tight = bw <= med_bw * TIGHT_BAND_MULT and med_bw > 0

    sb = slope_bps(sdf["vwap"], min(VWAP_SLOPE_BARS, len(sdf) - 1))
    slope_steep = abs(sb) >= SLOPE_STEEP_BPS
    slope_flat = abs(sb) <= SLOPE_FLAT_BPS

    sdf = sdf.copy()
    sdf["ma30"] = sdf["close"].rolling(MA_PERIOD).mean()
    ma_tail = sdf.tail(MA_LOOKBACK)
    ma_x = count_ma_crosses(
        ma_tail["close"].values, ma_tail["ma30"].values
    )

    vwap_choppy = crosses >= CHOPPY_CROSS_MIN
    ma_choppy = ma_x >= MA_CHOPPY_CROSS_MIN
    if vwap_choppy or ma_choppy:
        chop_score = "high"
    elif crosses >= 4:
        chop_score = "mid"
    else:
        chop_score = "low"

    reasons: List[str] = []
    play = "NONE"
    side = "FLAT"
    confidence = "low"
    regime = "mixed"

    band_half = (up - lo) / 2.0 if up > lo else 0.0
    touch_eps = max(band_half * BAND_TOUCH_FRAC, price * 1e-6)

    # 体制优先：陡+宽 -> 顺势；平+窄 -> 反转；否则观望或弱倾向
    if slope_steep and bands_wide:
        regime = "trend"
        if price > vw and price >= up - touch_eps:
            play = "PLAY01_BREAKOUT_LONG"
            side = "LONG"
            confidence = "medium"
            reasons.append("价格>VWAP，陡斜率，宽轨，贴近/站上上轨（顺势延续框架）")
        elif price < vw and price <= lo + touch_eps:
            play = "PLAY02_BREAKDOWN_SHORT"
            side = "SHORT"
            confidence = "medium"
            reasons.append("价格<VWAP，陡斜率，宽轨，贴近/站下下轨（顺势延续框架）")
        elif price > vw:
            play = "PLAY01_BIAS_LONG"
            side = "LONG"
            confidence = "low"
            reasons.append("价格>VWAP+陡斜率+宽轨，未极端贴轨，偏多观望/回踩试多框架")
        elif price < vw:
            play = "PLAY02_BIAS_SHORT"
            side = "SHORT"
            confidence = "low"
            reasons.append("价格<VWAP+陡斜率+宽轨，偏空观望/反抽试空框架")
    elif slope_flat and bands_tight:
        regime = "range"
        if price <= lo + touch_eps * 1.2:
            play = "PLAY03_REV_LONG"
            side = "LONG"
            confidence = "medium"
            reasons.append("平斜率+窄轨，贴近下轨，均值回归做多（目标 VWAP）")
        elif price >= up - touch_eps * 1.2:
            play = "PLAY03_REV_SHORT"
            side = "SHORT"
            confidence = "medium"
            reasons.append("平斜率+窄轨，贴近上轨，均值回归做空（目标 VWAP）")
        else:
            play = "PLAY03_WATCH"
            side = "FLAT"
            confidence = "low"
            reasons.append("震荡体制：等待贴近 σ 带再 fade")
    else:
        regime = "transition"
        if price > vw and bands_wide:
            side = "LONG"
            play = "TRANSITION_BIAS_LONG"
            reasons.append("过渡：价在 VWAP 上方且带宽偏宽")
        elif price < vw and bands_wide:
            side = "SHORT"
            play = "TRANSITION_BIAS_SHORT"
            reasons.append("过渡：价在 VWAP 下方且带宽偏宽")
        else:
            play = "NO_TRADE"
            reasons.append("斜率/带宽未同时满足顺势或反转模板")

    if chop_score == "high" and regime == "trend":
        reasons.append("会话 VWAP 交叉或 MA 纠缠偏多→谨慎追涨杀跌，易震荡")
        confidence = "low"

    near = nearest_level_distance_pct(price, levels)[:6]
    near_json = [{"level": n, "price": lv, "dist_pct": round(d, 4)} for n, lv, d in near]

    return SignalResult(
        symbol=symbol,
        price=round(price, 8),
        regime=regime,
        play=play,
        side=side,
        confidence=confidence,
        reasons=reasons,
        vwap=round(vw, 8),
        vwap_upper=round(up, 8),
        vwap_lower=round(lo, 8),
        slope_bps=round(sb, 4),
        band_width_pct=round(bw, 6),
        vwap_crosses=crosses,
        ma_crosses=ma_x,
        bands_wide=bands_wide,
        bands_tight=bands_tight,
        slope_steep=slope_steep,
        slope_flat=slope_flat,
        chop_score=chop_score,
        ref_levels={k: round(v, 8) for k, v in levels.items()},
        nearest_levels=near_json,
    )


def analyze_symbol(symbol: str) -> Optional[SignalResult]:
    kl = fetch_klines(symbol, "1m", 1500)
    df = klines_to_df(kl)
    if df.empty:
        return None
    sdf = session_cut_utc(df)
    if len(sdf) < 30:
        return None
    sdf = compute_vwap_bands_session(sdf, BAND_SIGMA)
    levels = ref_levels(symbol)
    res = classify_and_signal(symbol, sdf, levels)
    entry_ms = int(sdf.iloc[-1]["open_time"])
    sl, tp, ru = compute_sl_tp(res, sdf)
    return replace(
        res,
        entry_bar_open_ms=entry_ms,
        sl_price=sl,
        tp_price=tp,
        r_unit=ru,
    )


def send_telegram(text: str) -> None:
    """与 accumulation_radar.send_telegram 同源：分段、Markdown、失败回落纯文本。"""
    if not TG_BOT_TOKEN:
        print("\n[TG] No token, stdout:\n")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    chunks: List[str] = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > 3800:
            chunks.append(current)
            current = line
        else:
            current += "\n" + line if current else line
    if current:
        chunks.append(current)

    for chunk in chunks:
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id": TG_CHAT_ID,
                    "text": chunk,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                print(f"[TG] Sent ✓ ({len(chunk)} chars)")
            else:
                resp2 = requests.post(
                    url,
                    json={
                        "chat_id": TG_CHAT_ID,
                        "text": chunk.replace("*", "").replace("_", ""),
                    },
                    timeout=10,
                )
                print(f"[TG] Sent plain ({'✓' if resp2.status_code == 200 else '✗'})")
        except Exception as e:
            print(f"[TG] Error: {e}")
        time.sleep(0.5)


def _persist_results_db(
    recorded_at_utc: str,
    rows: List[SignalResult],
    scan_params: Dict[str, Any],
) -> Tuple[int, str]:
    """写入 accumulation.db；返回 (插入条数, db 文件路径)。"""
    from accumulation_radar import DB_PATH, init_db

    conn = init_db()
    try:
        cur = conn.cursor()
        params_json = json.dumps(scan_params, ensure_ascii=False)
        inserted = 0
        ins = """
            INSERT INTO zct_vwap_signals (
                recorded_at_utc, symbol, play, side, confidence, regime,
                entry_price, entry_bar_open_ms, sl_price, tp_price, r_unit,
                virtual_notional_usdt,
                vwap, vwap_upper, vwap_lower,
                slope_bps, band_width_pct, vwap_crosses, ma_crosses, chop_score,
                bands_wide, bands_tight, slope_steep, slope_flat,
                ref_levels_json, nearest_levels_json, reasons_json, scan_params_json
            ) VALUES (
                ?,?,?,?,?,?,
                ?,?,?,?,?,
                ?,
                ?,?,?,
                ?,?,?,?,?,
                ?,?,?,?,
                ?,?,?,?
            )
        """
        for r in rows:
            if DB_SKIP_FLAT and r.side == "FLAT":
                continue
            cur.execute(
                ins,
                (
                    recorded_at_utc,
                    r.symbol,
                    r.play,
                    r.side,
                    r.confidence,
                    r.regime,
                    r.price,
                    r.entry_bar_open_ms,
                    r.sl_price,
                    r.tp_price,
                    r.r_unit,
                    VIRTUAL_NOTIONAL_USDT,
                    r.vwap,
                    r.vwap_upper,
                    r.vwap_lower,
                    r.slope_bps,
                    r.band_width_pct,
                    r.vwap_crosses,
                    r.ma_crosses,
                    r.chop_score,
                    int(bool(r.bands_wide)),
                    int(bool(r.bands_tight)),
                    int(bool(r.slope_steep)),
                    int(bool(r.slope_flat)),
                    json.dumps(r.ref_levels, ensure_ascii=False),
                    json.dumps(r.nearest_levels, ensure_ascii=False),
                    json.dumps(r.reasons, ensure_ascii=False),
                    params_json,
                ),
            )
            inserted += 1
        conn.commit()
        return inserted, str(DB_PATH)
    finally:
        conn.close()


def format_result(r: SignalResult) -> str:
    lines = [
        f"*{r.symbol}*  `{r.play}`  side={r.side}  conf={r.confidence}",
        f"price={r.price}  VWAP={r.vwap}  σ[{r.vwap_lower}, {r.vwap_upper}]  slope={r.slope_bps} bps  bandW%={r.band_width_pct}",
        f"regime={r.regime}  VWAP_x={r.vwap_crosses}  MA30_x={r.ma_crosses}  chop={r.chop_score}",
    ]
    if r.sl_price is not None and r.tp_price is not None:
        lines.append(
            f"SL={r.sl_price}  TP={r.tp_price}  R={r.r_unit}"
            + (" (MR→VWAP)" if r.play in ("PLAY03_REV_LONG", "PLAY03_REV_SHORT") else " (1R)")
        )
    if r.side in ("LONG", "SHORT"):
        lines.append(f"paper_notional={VIRTUAL_NOTIONAL_USDT:g} USDT")
    lines.extend(
        [
            "reasons: " + "; ".join(r.reasons),
            "nearest levels: " + ", ".join(
                f"{x['level']}@{x['price']:.8f} ({x['dist_pct']:.3f}%)"
                for x in r.nearest_levels[:4]
            ),
        ]
    )
    return "\n".join(lines)


def resolve_open_signals_from_db() -> Dict[str, Any]:
    """
    对 outcome 为空且已写入 SL/TP 的记录，用信号 K 线之后的 1m 数据判定先触发止损或止盈。
    返回 stats + resolved_events（供 Telegram 平仓推送）。
    """
    from accumulation_radar import DB_PATH, init_db

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stats: Dict[str, Any] = {"checked": 0, "resolved": 0, "skipped": 0}
    resolved_events: List[Dict[str, Any]] = []
    conn = init_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, symbol, side, entry_price, sl_price, tp_price, entry_bar_open_ms,
                   COALESCE(virtual_notional_usdt, ?) AS notion
            FROM zct_vwap_signals
            WHERE outcome IS NULL
              AND sl_price IS NOT NULL AND tp_price IS NOT NULL
              AND side IN ('LONG','SHORT')
            ORDER BY id ASC
            """,
            (VIRTUAL_NOTIONAL_USDT,),
        )
        rows = cur.fetchall()
        end_ms = int(time.time() * 1000)
        for row in rows:
            stats["checked"] += 1
            sid, sym, side, entry, sl, tp, bar_open, notion = row
            if bar_open is None:
                stats["skipped"] += 1
                continue
            start_ms = int(bar_open) + 60_000
            if start_ms > end_ms:
                stats["skipped"] += 1
                continue
            kl = fetch_klines_forward(sym, "1m", start_ms, end_ms)
            if not kl:
                stats["skipped"] += 1
                continue
            outcome: Optional[str] = None
            exit_px: float = entry
            note = "resolved:auto"
            bars_seen = 0
            for k in kl:
                if int(k[0]) < start_ms:
                    continue
                bars_seen += 1
                o, h, low, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
                if side == "LONG":
                    tag, px = _bar_hit_long(o, h, low, sl, tp)
                else:
                    tag, px = _bar_hit_short(o, h, low, sl, tp)
                if tag == "win":
                    outcome = "win"
                    exit_px = px
                    break
                if tag == "loss":
                    outcome = "loss"
                    exit_px = px
                    break
                if bars_seen >= RESOLVE_MAX_BARS:
                    outcome = "expired"
                    exit_px = c
                    note = f"resolved:auto_expired_after_{RESOLVE_MAX_BARS}_bars"
                    break
            if outcome is None:
                stats["skipped"] += 1
                continue
            pnl = _pnl_r(side, entry, exit_px, sl, tp)
            pnl_u = _pnl_usdt(side, entry, exit_px, float(notion))
            cur.execute(
                """
                UPDATE zct_vwap_signals
                SET outcome = ?, outcome_at_utc = ?, exit_price = ?, pnl_r = ?, pnl_usdt = ?,
                    notes = CASE WHEN notes IS NULL OR notes = '' THEN ?
                                 ELSE notes || '; ' || ? END
                WHERE id = ? AND outcome IS NULL
                """,
                (
                    outcome,
                    now_utc,
                    exit_px,
                    round(pnl, 6),
                    round(pnl_u, 4),
                    note,
                    note,
                    sid,
                ),
            )
            if cur.rowcount:
                stats["resolved"] += 1
                resolved_events.append(
                    {
                        "id": sid,
                        "symbol": sym,
                        "side": side,
                        "outcome": outcome,
                        "exit_price": exit_px,
                        "pnl_r": round(pnl, 6),
                        "pnl_usdt": round(pnl_u, 4),
                    }
                )
        conn.commit()
        print(f"[resolve] checked={stats['checked']} resolved={stats['resolved']} skipped={stats['skipped']} db={DB_PATH}")
        stats["resolved_events"] = resolved_events
        return stats
    finally:
        conn.close()


def _tg_push_summary_text(
    ts: str,
    syms: List[str],
    per_symbol_lines: List[str],
    n_actionable: int,
) -> str:
    """每轮一条精简结论（每个标的一行），便于定时任务必达推送。"""
    lines: List[str] = [
        f"📊 ZCT VWAP 扫描结论  {ts} UTC",
        f"标的 {len(syms)} 个 · 本轮回合方向单（含 SL/TP）: {n_actionable}",
        "",
        *per_symbol_lines,
        "",
        "★=方向单+SL/TP；全文请设 ZCT_VWAP_TG_PUSH_MODE=all",
    ]
    return "\n".join(lines)


def _tg_push_scan_text(ts: str, syms: List[str], result_objs: List[SignalResult]) -> str:
    """组装发 TG 的扫描正文（无 markdown）。"""
    lines: List[str] = [
        f"ZCT VWAP 信号扫描 {ts} UTC",
        f"标的: {', '.join(syms)}",
    ]
    actionable = [
        r
        for r in result_objs
        if r.side in ("LONG", "SHORT") and r.sl_price is not None and r.tp_price is not None
    ]
    if actionable:
        lines.append("")
        lines.append("—— 方向单 ——")
        for r in actionable:
            lines.append("")
            lines.append(format_result(r).replace("*", "").replace("`", ""))
    else:
        lines.append("")
        lines.append("本轮无方向单（观望/NO_TRADE）；明细见日志或看板。")
    return "\n".join(lines)


def _tg_push_resolve_text(events: List[Dict[str, Any]]) -> str:
    if not events:
        return ""
    lines = ["📌 ZCT VWAP 平仓结算", ""]
    for e in events:
        lines.append(
            f"#{e['id']} {e['symbol']} {e['side']} → {e['outcome']} | "
            f"exit={e['exit_price']} | R={e['pnl_r']} | {e['pnl_usdt']} U"
        )
    return "\n".join(lines)


def run_scan(use_tg: bool = True, *, do_resolve: bool = True) -> Dict[str, Any]:
    syms = _symbols_from_env()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results: List[Dict[str, Any]] = []
    result_objs: List[SignalResult] = []
    tg_summary_lines: List[str] = []
    text_blocks: List[str] = [f"ZCT VWAP 信号扫描 `{ts}` UTC\n标的: {', '.join(syms)}"]

    for sym in syms:
        try:
            res = analyze_symbol(sym)
            if res is None:
                text_blocks.append(f"\n{sym}: 数据不足（会话 K 过少或无 K 线）")
                tg_summary_lines.append(f"· {sym}  数据不足")
                continue
            results.append(asdict(res))
            result_objs.append(res)
            text_blocks.append("\n" + format_result(res))
            actionable = (
                res.side in ("LONG", "SHORT")
                and res.sl_price is not None
                and res.tp_price is not None
            )
            mark = "★" if actionable else "·"
            tg_summary_lines.append(
                f"{mark} {sym}  {res.side}  {res.play}  conf={res.confidence}  {res.regime}"
            )
        except Exception as e:
            text_blocks.append(f"\n{sym}: ERROR {e}")
            tg_summary_lines.append(f"· {sym}  ERROR {e}")

    scan_params: Dict[str, Any] = {
        "symbols_scanned": syms,
        "band_sigma": BAND_SIGMA,
        "slope_bars": VWAP_SLOPE_BARS,
        "slope_steep_bps": SLOPE_STEEP_BPS,
        "slope_flat_bps": SLOPE_FLAT_BPS,
        "wide_band_mult": WIDE_BAND_MULT,
        "tight_band_mult": TIGHT_BAND_MULT,
        "db_skip_flat": DB_SKIP_FLAT,
        "swing_lookback": SWING_LOOKBACK,
        "min_sl_pct": MIN_SL_PCT,
        "resolve_max_bars": RESOLVE_MAX_BARS,
        "same_bar_rule": SAME_BAR_RULE,
        "virtual_notional_usdt": VIRTUAL_NOTIONAL_USDT,
    }
    payload = {
        "generated_at_utc": ts,
        "symbols": syms,
        "params": scan_params,
        "results": results,
    }

    if result_objs:
        try:
            n, dbp = _persist_results_db(ts, result_objs, scan_params)
            print(f"[db] zct_vwap_signals inserted={n} → {dbp}")
        except Exception as e:
            print(f"[db] persist failed: {e}")

    resolve_stats: Dict[str, Any] = {}
    if do_resolve:
        try:
            resolve_stats = resolve_open_signals_from_db()
        except Exception as e:
            print(f"[resolve] failed: {e}")

    msg = "\n".join(text_blocks)
    print(msg)

    if use_tg and TG_PUSH_MODE != "off":
        mode = TG_PUSH_MODE if TG_PUSH_MODE in ("all", "summary", "actionable") else "summary"
        n_actionable = sum(
            1
            for r in result_objs
            if r.side in ("LONG", "SHORT")
            and r.sl_price is not None
            and r.tp_price is not None
        )
        if mode == "all":
            send_telegram(msg.replace("*", "").replace("`", ""))
        elif mode == "summary":
            send_telegram(_tg_push_summary_text(ts, syms, tg_summary_lines, n_actionable))
        else:
            # actionable：仅当有 LONG/SHORT 且含 SL/TP 时推送（与旧默认一致）
            if n_actionable > 0:
                send_telegram(_tg_push_scan_text(ts, syms, result_objs))
            else:
                print("[TG] 本轮无方向单，跳过扫描推送（ZCT_VWAP_TG_PUSH_MODE=actionable）")

    if (
        use_tg
        and TG_NOTIFY_RESOLVE
        and resolve_stats.get("resolved_events")
    ):
        send_telegram(_tg_push_resolve_text(resolve_stats["resolved_events"]))

    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="ZCT VWAP signal scanner")
    ap.add_argument("--no-tg", action="store_true", help="Do not send Telegram")
    ap.add_argument("--no-resolve", action="store_true", help="Skip SL/TP outcome resolution pass")
    ap.add_argument(
        "--resolve-only",
        action="store_true",
        help="Only run DB resolution for open signals (no market scan)",
    )
    args = ap.parse_args()
    if args.resolve_only:
        rs = resolve_open_signals_from_db()
        if not args.no_tg and TG_NOTIFY_RESOLVE and rs.get("resolved_events"):
            send_telegram(_tg_push_resolve_text(rs["resolved_events"]))
        return
    run_scan(use_tg=not args.no_tg, do_resolve=not args.no_resolve)


if __name__ == "__main__":
    main()

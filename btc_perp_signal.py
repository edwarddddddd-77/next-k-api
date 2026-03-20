# -*- coding: utf-8 -*-
"""
永续合约信号脚本 - 策略完全体 V8 (V7 + FVG 进场共振)
策略要点 (V7 基础):
  - 波动率三档、Funding 拥挤过滤、L/S 与 Taker、OI 方向、清算/VP/盘口、CVD/MACD/4H、净差与 SOP
  - Funding 梯度/一票否决、盘口深度衰减、4H 超买超卖从严
V8 新增:
  - FVG (Fair Value Gap)：扫描最近 N 根 30m K 线识别看涨/看跌 FVG，过滤已回补缺口
  - 进场逻辑：若存在未回补的 FVG 且其中轴与 VWAP 共振(偏离≤fvg_vwap_max_dev_pct)，优先以 FVG 区间为进场带；否则沿用 V7 价值区/VWAP
"""
import sys
import io
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
import os

# Windows 控制台 UTF-8 输出兼容
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass

# ================= 配置区 =================
CONFIG = {
    "symbols": ["BTCUSDT", "ETHUSDT"],  
    "kline_interval": "30m",
    "kline_limit": 60,  
    "htf_interval": "4h",
    "htf_limit": 60,
    "depth_limit": 100,
    "request_timeout": 8,
    "max_retries": 3,
}

# ================= 策略参数（可调） =================
STRATEGY = {
    # 波动率 regime：BBW 与均线比值
    "bbw_low": 0.7,           # < 此值为低波盘整
    "bbw_high": 1.3,          # > 此值为高波趋势
    "bbw_mid_low": 0.85,      # 正常偏盘整上界
    "bbw_mid_high": 1.15,     # 正常偏趋势下界
    "vol_trend_low": 0.5,     # 低波时趋势乘数（原 0，现给 0.5 弱趋势）
    "vol_mean_rev_low": 1.5,
    "vol_trend_high": 1.5,
    "vol_mean_rev_high": 0.5,
    # Funding 拥挤过滤：|fund| >= 此值(%) 时同向得分打折；极端费率一票否决
    "funding_crowd_pct": 0.01,
    "funding_crowd_dampen": 0.7,
    "funding_extreme_pct": 0.03,       # >= 此值视为极端拥挤，同向得分极重降权
    "funding_extreme_dampen": 0.25,    # 极端时同向乘数（接近一票否决）
    "funding_extreme_veto": True,      # True=极端时同向方向直接判为中性
    # L/S 与 Taker 一致加分、冲突降权
    "ls_align_bonus": 0.5,
    "ls_conflict_dampen": 0.7,
    # OI 方向：价涨 OI 升=趋势延续多+0.5，价涨 OI 降=平多空+0.5
    "oi_trend_bonus": 0.5,
    # 清算磁吸：触发距离(%) 上限、与波动率挂钩系数
    "liq_dist_max_pct": 0.6,
    "liq_atr_mult": 0.3,      # 有效距离 = min(liq_dist_max_pct, atr_pct * liq_atr_mult)
    # 价值区内弱信号：距 VWAP 超过此比例才给 ±0.5
    "vp_inside_vwap_thresh_pct": 0.003,
    # 盘口：近档权重（前 N 档）、深度衰减（按距 mid 距离指数衰减）、高波时 OB 得分降权
    "depth_near_levels": 20,
    "depth_decay_factor": 0.5,         # 挂单权重 = exp(-factor * |价距mid%|)，远档贡献小
    "ob_high_vol_dampen": 0.65,        # 高波动时盘口信号乘数（前档虚单多）
    # CVD：多窗口 3/5；价差阈值 % 用于 A/B 级
    "cvd_price_thresh_pct": 0.2,
    "cvd_low_vol_dampen": 0.8,
    # MACD：需连续 2 根 hist 同向扩大
    # 4H 顺大势：仅当价格离 EMA50 超过此比例或 EMA 斜率同向
    "htf_min_dist_pct": 0.005,
    # 4H 超买/超卖：仅「极端偏离」时降权，强趋势日不误杀
    "htf_ob_os_thresh_pct": 0.08,      # 价格离 4H EMA50 超过 8% 才视为超买/超卖（4%时强势日也被压分，改为8%）
    "htf_ob_os_dampen": 0.65,          # 触发时同向乘数（仅极端偏离时用）
    # 净差与等级
    "min_net_gap": 2.0,
    "min_net_gap_strong": 2.5,
    "both_high_thresh": 4.0,
    "near_high_gap": 1.5,
    "near_high_both": 3.0,
    "s_bull_score": 7.0,               # S级略放宽：7.5→7.0，强趋势日有机会出「强力看多/看空」
    "s_net_score": 3.0,               # 净差 3.5→3.0，多空领先幅度仍明显大于 A 级
    "a_bull_score": 4.5,               # A级：4.5 分即偏多/偏空（净差仍要求 1.5，避免摇摆）
    "a_net_score": 1.5,               # A级净差放宽：2.0→1.5，多空不必拉得极开
    # SOP 止损 ATR 倍数、时效说明
    "sop_atr_sl_mult": 0.75,
    "sop_valid_hours": 4,   # 逢低做多需等回踩，4h 与 4H 周期对齐且给足入场窗口
    # 买入/卖出区间：基于价值区(VAL/VWAP/VAH)，不追价；V8 可与 FVG 共振
    "entry_below_vwap_pct": 0.005,   # 做多下沿可接受 VWAP 下方 0.5% 挂单
    "entry_above_vwap_pct": 0.01,    # 做多上沿不超过 VWAP+1%（或 VAH，取高者但不超现价）
    # FVG (Fair Value Gap)：SMC 订单流失衡缺口，回踩中轴高胜率
    "fvg_lookback": 12,              # 扫描最近 12 根 K 线寻找 FVG
    "fvg_vwap_max_dev_pct": 0.01,    # FVG 中轴与 VWAP 偏离 ≤1% 视为共振，才用 FVG 做进场区
    "fvg_min_width_pct": 0.05,       # FVG 宽度至少 0.05% 才用 FVG 做进场区，过窄则退回 VWAP/价值区
}

# 📧 邮件配置
EMAIL_CONFIG = {
    "smtp_host": "smtp.163.com",
    "smtp_port": 465,
    "from_email": "",
    "to_email": "",
    "password": "",  
}
# ==========================================

def _asset(symbol):
    return symbol.replace("USDT", "") if isinstance(symbol, str) else str(symbol)


def _fvg_near_vwap(fvg, vwap, max_dev_pct=0.01):
    """FVG 中轴与 VWAP 偏离 ≤ max_dev_pct 视为共振。"""
    if not fvg or not vwap or vwap <= 0:
        return False
    mid = fvg.get("mid")
    if mid is None:
        return False
    return abs(mid - vwap) / vwap <= max_dev_pct


def find_latest_fvgs(df, current_price, lookback=12):
    """
    扫描最近 lookback 根 K 线，找出最近的有效看涨 FVG 与看跌 FVG（未回补）。
    看涨 FVG: K1.High < K3.Low，缺口 [K1.High, K3.Low]；若当前价已跌破 bottom 视为已回补。
    看跌 FVG: K1.Low > K3.High，缺口 [K3.High, K1.Low]；若当前价已突破 top 视为已回补。
    返回 {"bull": {...} or None, "bear": {...} or None}，每项含 type, top, mid, bottom。
    """
    if df is None or len(df) < 3:
        return {"bull": None, "bear": None}
    end = len(df) - 3
    start = max(0, end - lookback + 1)
    if end < start:
        return {"bull": None, "bear": None}

    fvg_bull = None
    fvg_bear = None
    for i in range(end, start - 1, -1):
        k1, k3 = df.iloc[i], df.iloc[i + 2]
        h1, l1 = float(k1["high"]), float(k1["low"])
        h3, l3 = float(k3["high"]), float(k3["low"])
        if fvg_bull is None and h1 < l3 and current_price >= h1:
            fvg_bull = {"type": "BULL", "top": l3, "mid": (l3 + h1) / 2, "bottom": h1}
        if fvg_bear is None and l1 > h3 and current_price <= l1:
            fvg_bear = {"type": "BEAR", "top": l1, "mid": (l1 + h3) / 2, "bottom": h3}
        if fvg_bull is not None and fvg_bear is not None:
            break
    return {"bull": fvg_bull, "bear": fvg_bear}


def fetch_json(url, log=None):
    for attempt in range(CONFIG["max_retries"]):
        try:
            res = requests.get(url, timeout=CONFIG["request_timeout"])
            res.raise_for_status()
            return res.json()
        except Exception as e:
            if attempt == CONFIG["max_retries"] - 1:
                if log: log(f"  ❌ 请求失败 ({url}): {e}")
                return None
            time.sleep(0.5)

def get_binance_data_optimized(symbol, log=None):
    def _log(msg):
        if log: log(msg)

    base_url = "https://fapi.binance.com"
    spot_url = "https://api.binance.com"
    sym = symbol

    endpoints = {
        "ticker_24h": f"{base_url}/fapi/v1/ticker/24hr?symbol={sym}",
        "premium": f"{base_url}/fapi/v1/premiumIndex?symbol={sym}",
        "oi": f"{base_url}/fapi/v1/openInterest?symbol={sym}",
        "funding": f"{base_url}/fapi/v1/fundingRate?symbol={sym}&limit=1",
        "klines": f"{base_url}/fapi/v1/klines?symbol={sym}&interval={CONFIG['kline_interval']}&limit={CONFIG['kline_limit']}",
        "klines_spot": f"{spot_url}/api/v3/klines?symbol={sym}&interval={CONFIG['kline_interval']}&limit={CONFIG['kline_limit']}", 
        "klines_htf": f"{base_url}/fapi/v1/klines?symbol={sym}&interval={CONFIG['htf_interval']}&limit={CONFIG['htf_limit']}",
        "ls_pos": f"{base_url}/futures/data/topLongShortPositionRatio?symbol={sym}&period=30m&limit=1",
        "ls_acct": f"{base_url}/futures/data/globalLongShortAccountRatio?symbol={sym}&period=30m&limit=1",
        "taker_ratio": f"{base_url}/futures/data/takerlongshortRatio?symbol={sym}&period=30m&limit=1",
        "depth": f"{base_url}/fapi/v1/depth?symbol={sym}&limit={CONFIG['depth_limit']}",
        "oi_hist": f"{base_url}/futures/data/openInterestHist?symbol={sym}&period=30m&limit={CONFIG['kline_limit']}",
    }

    _log(f"  → [{_asset(sym)}] 并发请求 {len(endpoints)} 个接口...")
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        future_to_key = {executor.submit(fetch_json, url, log): key for key, url in endpoints.items()}
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as e:
                _log(f"  ❌ [{_asset(sym)}] {key} 异常: {e}")
                results[key] = None
    _log(f"  ✓ [{_asset(sym)}] 拉取完成")

    current_ms = int(time.time() * 1000)
    
    df = pd.DataFrame()
    if results.get("klines"):
        df = pd.DataFrame(results["klines"], columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore"])
        df[["open", "high", "low", "close", "volume", "tb_base_av", "close_time"]] = df[["open", "high", "low", "close", "volume", "tb_base_av", "close_time"]].astype(float)
        if df["close_time"].iloc[-1] > current_ms: df = df.iloc[:-1]

    df_spot = pd.DataFrame()
    if results.get("klines_spot"):
        df_spot = pd.DataFrame(results["klines_spot"], columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore"])
        df_spot[["open", "high", "low", "close", "volume", "tb_base_av", "close_time"]] = df_spot[["open", "high", "low", "close", "volume", "tb_base_av", "close_time"]].astype(float)
        if df_spot["close_time"].iloc[-1] > current_ms: df_spot = df_spot.iloc[:-1]

    df_htf = pd.DataFrame()
    if results.get("klines_htf"):
        df_htf = pd.DataFrame(results["klines_htf"], columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore"])
        df_htf[["close", "close_time"]] = df_htf[["close", "close_time"]].astype(float)
        if df_htf["close_time"].iloc[-1] > current_ms: df_htf = df_htf.iloc[:-1]

    return (results["ticker_24h"], results["premium"], results["oi"], results["funding"], df, df_spot, df_htf, results["ls_pos"], results["ls_acct"], results["taker_ratio"], results["depth"], results["oi_hist"])


def calculate_and_analyze(data, symbol, log=None):
    def _log(msg):
        if log: log(msg)

    asset = _asset(symbol)
    ticker_24h, premium, oi, funding, df, df_spot, df_htf, ls_pos, ls_acct, taker_ratio, depth, oi_hist = data
    
    if df.empty or df_htf.empty or df_spot.empty or not all([ticker_24h, premium, oi, funding, depth, ls_pos, ls_acct, taker_ratio, oi_hist]):
        return f"❌ [{asset}] 接口数据缺失或超时，本轮跳过计算以防崩溃。", "⚖️"

    _log(f"  → [{asset}] 解析快照与计算微观结构...")
    current_price = float(ticker_24h["lastPrice"])
    mark_price = float(premium["markPrice"])
    index_price = float(premium.get("indexPrice") or mark_price)
    premium_pct = (mark_price - index_price) / index_price * 100 if index_price else 0
    next_funding_ts = int(premium.get("nextFundingTime") or 0)
    high_24h = float(ticker_24h["highPrice"])
    low_24h = float(ticker_24h["lowPrice"])
    vol_24h = float(ticker_24h["quoteVolume"])
    price_change_pct = float(ticker_24h["priceChangePercent"])
    
    swing_high = df["high"].max()
    swing_low = df["low"].min()

    bull_score = 0.0
    bear_score = 0.0
    signals = {"bull": [], "bear": []}

    # ================= 🚀 模块 1: 波动率与宏观环境 =================
    df["sma20"] = df["close"].rolling(window=20).mean()
    df["std20"] = df["close"].rolling(window=20).std()
    df["bbw"] = (df["std20"] * 4) / df["sma20"] * 100
    current_bbw = df["bbw"].iloc[-1]
    avg_bbw = df["bbw"].rolling(window=20).mean().iloc[-1]

    price_pos_pct = (current_price - low_24h) / (high_24h - low_24h) * 100 if high_24h > low_24h else 50
    market_snapshot_msg = f"温和波动, 价格处于 24h 区间 {price_pos_pct:.0f}% 位置"
    if price_change_pct > 3: market_snapshot_msg = f"强势拉升, 价格处于 24h 区间 {price_pos_pct:.0f}% 位置"
    if price_change_pct < -3: market_snapshot_msg = f"急速下挫, 价格处于 24h 区间 {price_pos_pct:.0f}% 位置"

    # 波动率三档：低 / 正常偏盘整 / 正常偏趋势 / 高（与均线 BBW 比较）
    if current_bbw < avg_bbw * STRATEGY["bbw_low"]:
        vol_regime = "📉 低波动盘整 (均值回归市)"
        vol_multiplier_trend = STRATEGY["vol_trend_low"]
        vol_multiplier_mean_rev = STRATEGY["vol_mean_rev_low"]
    elif current_bbw > avg_bbw * STRATEGY["bbw_high"]:
        vol_regime = "📈 高波动单边 (趋势市)"
        vol_multiplier_trend = STRATEGY["vol_trend_high"]
        vol_multiplier_mean_rev = STRATEGY["vol_mean_rev_high"]
    elif current_bbw < avg_bbw * STRATEGY["bbw_mid_low"]:
        vol_regime = "📊 正常偏盘整"
        vol_multiplier_trend = 0.75
        vol_multiplier_mean_rev = 1.2
    elif current_bbw > avg_bbw * STRATEGY["bbw_mid_high"]:
        vol_regime = "📊 正常偏趋势"
        vol_multiplier_trend = 1.2
        vol_multiplier_mean_rev = 0.75
    else:
        vol_regime = "正常波动"
        vol_multiplier_trend = 1.0
        vol_multiplier_mean_rev = 1.0

    # 提前计算 ATR（供清算距离自适应与后续 SOP 使用）
    df["prev_close"] = df["close"].shift(1)
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["prev_close"]).abs(),
            (df["low"] - df["prev_close"]).abs()
        )
    )
    df["tr"] = tr
    df["atr"] = df["tr"].rolling(window=14).mean()
    current_atr = float(df["atr"].iloc[-1]) if len(df) >= 14 else (high_24h - low_24h) / 10
    atr_pct = current_atr / current_price * 100 if current_price else 0

    # ================= 原版衍生品状态 =================
    current_oi = float(oi["openInterest"]) if oi else 0
    oi_change_str = ""
    if isinstance(oi_hist, list) and len(oi_hist) > 0:
        oi_old = float(oi_hist[0].get("sumOpenInterest") or oi_hist[0].get("openInterest", 0))
        if oi_old > 0:
            oi_change_pct = (current_oi - oi_old) / oi_old * 100
            oi_change_str = f" ({oi_change_pct:+.1f}% 24h)"

    fund_rate = float(funding[0]["fundingRate"]) * 100 if funding else 0
    fund_tag = "[偏高]" if fund_rate >= 0.005 else "[偏低]" if fund_rate <= -0.005 else "[正常]"
    
    ls_pos_val = float(ls_pos[0]["longShortRatio"]) if ls_pos else 1.0
    ls_acct_val = float(ls_acct[0]["longShortRatio"]) if ls_acct else 1.0
    taker_val = float(taker_ratio[0]["buySellRatio"]) if taker_ratio else 1.0

    taker_str = "Taker买卖均衡"
    if taker_val > 1.1:
        bull_score += (1.0 * vol_multiplier_trend)
        signals["bull"].append(f"Taker主买 (+{1.0 * vol_multiplier_trend:.1f})")
        taker_str = "Taker主动买入偏多, 短期买压较强"
    elif taker_val < 0.9:
        bear_score += (1.0 * vol_multiplier_trend)
        signals["bear"].append(f"Taker主卖 (+{1.0 * vol_multiplier_trend:.1f})")
        taker_str = "Taker主动卖出偏多, 短期抛压较强"

    # L/S 与 Taker 一致加分、冲突降权
    if taker_val > 1.1 and ls_acct_val > 1.05:
        bull_score += STRATEGY["ls_align_bonus"]
        signals["bull"].append(f"L/S与Taker一致多 (+{STRATEGY['ls_align_bonus']:.1f})")
    elif taker_val < 0.9 and ls_acct_val < 0.95:
        bear_score += STRATEGY["ls_align_bonus"]
        signals["bear"].append(f"L/S与Taker一致空 (+{STRATEGY['ls_align_bonus']:.1f})")
    if taker_val > 1.1 and ls_acct_val < 0.95:
        bull_score *= STRATEGY["ls_conflict_dampen"]
    elif taker_val < 0.9 and ls_acct_val > 1.05:
        bear_score *= STRATEGY["ls_conflict_dampen"]

    deriv_msg = f"Funding 费率{fund_tag}, 大户多空均衡 ({ls_pos_val:.2f}); {taker_str}"

    # ================= 关键价位 (VP) & 清算 =================
    vwap = (df["close"] * df["volume"]).sum() / df["volume"].sum()
    bins = np.linspace(df["low"].min(), df["high"].max(), 50)
    df["bucket"] = pd.cut(df["close"], bins=bins)
    vp = df.groupby("bucket", observed=False)["volume"].sum()
    poc_idx = vp.argmax()
    poc = vp.index[poc_idx].mid

    target_vol = vp.sum() * 0.7
    current_vol = vp.iloc[poc_idx]
    up_idx, down_idx = poc_idx + 1, poc_idx - 1
    n_vp = len(vp)
    while current_vol < target_vol and (up_idx < n_vp or down_idx >= 0):
        vol_up = vp.iloc[up_idx] if up_idx < n_vp else -1
        vol_down = vp.iloc[down_idx] if down_idx >= 0 else -1
        if vol_up > vol_down:
            if vol_up > 0: current_vol += vol_up
            up_idx += 1
        else:
            if vol_down > 0: current_vol += vol_down
            down_idx -= 1
        if up_idx >= n_vp and down_idx < 0: break
    vah = vp.index[min(up_idx, n_vp - 1)].mid
    val = vp.index[max(down_idx, 0)].mid

    vp_status_msg = f"当前价格位于价值区域内，距 VWAP {(current_price-vwap)/vwap*100:+.2f}%"
    if current_price > vah:
        bull_score += (1.0 * vol_multiplier_trend)
        signals["bull"].append(f"价格在VAH上方 (+{1.0 * vol_multiplier_trend:.1f})")
        vp_status_msg = f"当前价格位于 VAH 上方(偏强区域)，距 VWAP {(current_price-vwap)/vwap*100:+.2f}%"
    elif current_price < val:
        bear_score += (1.0 * vol_multiplier_trend)
        signals["bear"].append(f"价格在VAL下方 (+{1.0 * vol_multiplier_trend:.1f})")
        vp_status_msg = f"当前价格位于 VAL 下方(偏弱区域)，距 VWAP {(current_price-vwap)/vwap*100:+.2f}%"
    else:
        # 价值区内弱信号：距 VWAP 超过阈值给弱方向
        vwap_dev_pct = (current_price - vwap) / vwap if vwap else 0
        if vwap_dev_pct >= STRATEGY["vp_inside_vwap_thresh_pct"]:
            bull_score += (0.5 * vol_multiplier_trend)
            signals["bull"].append(f"价值区内偏上 (+{0.5 * vol_multiplier_trend:.1f})")
        elif vwap_dev_pct <= -STRATEGY["vp_inside_vwap_thresh_pct"]:
            bear_score += (0.5 * vol_multiplier_trend)
            signals["bear"].append(f"价值区内偏下 (+{0.5 * vol_multiplier_trend:.1f})")

    # V8: FVG 扫描（供 SOP 进场区共振使用）
    fvgs = find_latest_fvgs(df, current_price, STRATEGY.get("fvg_lookback", 12))
    fvg_vwap_dev = STRATEGY.get("fvg_vwap_max_dev_pct", 0.01)

    liq_short_100x = swing_high * 1.012  
    liq_long_100x = swing_low * 0.988    
    
    dist_short_pct = (liq_short_100x - current_price) / current_price * 100
    dist_long_pct = (current_price - liq_long_100x) / current_price * 100

    # 清算磁吸触发距离：与波动率挂钩，高波用更大距离
    liq_trigger_pct = min(STRATEGY["liq_dist_max_pct"], max(0.25, atr_pct * STRATEGY["liq_atr_mult"]))
    
    liq_msg = "未靠近核心清算区, 未出现猎杀反转动作"
    if 0 < dist_short_pct < liq_trigger_pct:
        liq_msg = f"🧲 磁吸警告: 距上方空头清算区仅剩 {dist_short_pct:.2f}%！"
        bull_score += 1.5
    elif 0 < dist_long_pct < liq_trigger_pct:
        liq_msg = f"🧲 磁吸警告: 距下方多头清算区仅剩 {dist_long_pct:.2f}%！"
        bear_score += 1.5
    elif current_price >= liq_short_100x and (df["high"].iloc[-1] - max(df["open"].iloc[-1], df["close"].iloc[-1])) > (df["high"].iloc[-1] - df["low"].iloc[-1])*0.5:
        liq_msg = f"🩸 猎杀完成: 价格刺穿上方清算区遭拒绝(长上影)，看跌！"
        bear_score += 2.5
        bull_score = max(0.0, bull_score - 1.0)
    elif current_price <= liq_long_100x and (min(df["open"].iloc[-1], df["close"].iloc[-1]) - df["low"].iloc[-1]) > (df["high"].iloc[-1] - df["low"].iloc[-1])*0.5:
        liq_msg = f"🩸 猎杀完成: 价格刺穿下方清算区并收回(长下影)，看涨！"
        bull_score += 2.5
        bear_score = max(0.0, bear_score - 1.0)

    # ================= 盘口与 CVD 背离（深度衰减 + 高波降权） =================
    # 深度衰减：挂单权重 = exp(-factor * |价距 mid%|)，远档贡献小；高波时 OB 得分再乘 ob_high_vol_dampen
    ob_mult = STRATEGY["ob_high_vol_dampen"] if vol_regime.startswith("📈") else 1.0
    if depth and depth.get("bids") and depth.get("asks"):
        bids_list = [(float(p), float(q)) for p, q in depth["bids"]]
        asks_list = [(float(p), float(q)) for p, q in depth["asks"]]
        if bids_list and asks_list:
            best_bid = bids_list[0][0]
            best_ask = asks_list[0][0]
            mid = (best_bid + best_ask) / 2
            decay = STRATEGY["depth_decay_factor"]
            def weighted_vol(levels, mid_price, is_bid):
                total = 0.0
                for price, qty in levels:
                    dist_pct = abs(mid_price - price) / mid_price * 100
                    w = np.exp(-decay * dist_pct)
                    total += w * qty
                return total
            bids_weighted = weighted_vol(bids_list, mid, True)
            asks_weighted = weighted_vol(asks_list, mid, False)
            bid_ask_ratio = bids_weighted / asks_weighted if asks_weighted > 0 else 1.0
        else:
            bid_ask_ratio = 1.0
    else:
        bid_ask_ratio = 1.0
    ob_tag = "[买方强]" if bid_ask_ratio > 1.5 else "[卖方强]" if bid_ask_ratio < 0.6 else "[均衡]"

    ob_msg = "买卖挂单力量相对均衡"
    if bid_ask_ratio < 0.6:
        bear_score += (1.0 * vol_multiplier_trend * ob_mult)
        signals["bear"].append(f"上方抛压重 (+{1.0 * vol_multiplier_trend * ob_mult:.1f})")
        ob_msg = "卖方挂单远多于买方, 上方压力较大"
    elif bid_ask_ratio > 1.5:
        bull_score += (1.0 * vol_multiplier_trend * ob_mult)
        signals["bull"].append(f"下方托单强 (+{1.0 * vol_multiplier_trend * ob_mult:.1f})")
        ob_msg = "买方挂单托底强劲, 下方支撑明显"

    df["taker_sell_vol"] = df["volume"] - df["tb_base_av"]
    df["cvd"] = (df["tb_base_av"] - df["taker_sell_vol"]).cumsum()
    df_spot["taker_sell_vol"] = df_spot["volume"] - df_spot["tb_base_av"]
    df_spot["cvd"] = (df_spot["tb_base_av"] - df_spot["taker_sell_vol"]).cumsum()

    cvd_title = "量价平稳"
    spot_cvd_status = "✅ 健康 (现货合约同频买入)"
    cvd_msg = "现货与合约量价平稳，未见明显背离"
    perp_cvd_delta, spot_cvd_delta, oi_delta_pct = None, None, None

    if len(df) >= 5 and len(df_spot) >= 5:
        p_curr, p_prev = df["close"].iloc[-1], df["close"].iloc[-5]
        price_delta_pct = (p_curr - p_prev) / p_prev * 100

        perp_cvd_delta = df["cvd"].iloc[-1] - df["cvd"].iloc[-5]
        spot_cvd_delta = df_spot["cvd"].iloc[-1] - df_spot["cvd"].iloc[-5]
        
        oi_curr_hist = float(oi_hist[-1].get("sumOpenInterest", 0))
        oi_prev_hist = float(oi_hist[-5].get("sumOpenInterest", 0))
        oi_delta_pct = (oi_curr_hist - oi_prev_hist) / oi_prev_hist * 100 if oi_prev_hist > 0 else 0

        # OI 方向：价涨 OI 升=趋势延续多+0.5，价涨 OI 降=平多空+0.5
        if price_delta_pct > 0.05 and oi_delta_pct > 0.3:
            bull_score += STRATEGY["oi_trend_bonus"]
            signals["bull"].append(f"价涨OI升延续 (+{STRATEGY['oi_trend_bonus']:.1f})")
        elif price_delta_pct > 0.05 and oi_delta_pct < -0.3:
            bear_score += STRATEGY["oi_trend_bonus"]
            signals["bear"].append(f"价涨OI降平多 (+{STRATEGY['oi_trend_bonus']:.1f})")
        elif price_delta_pct < -0.05 and oi_delta_pct < -0.3:
            bear_score += STRATEGY["oi_trend_bonus"]
            signals["bear"].append(f"价跌OI降延续 (+{STRATEGY['oi_trend_bonus']:.1f})")
        elif price_delta_pct < -0.05 and oi_delta_pct > 0.3:
            bull_score += STRATEGY["oi_trend_bonus"]
            signals["bull"].append(f"价跌OI升平空 (+{STRATEGY['oi_trend_bonus']:.1f})")

        cvd_spot_mult = STRATEGY["cvd_low_vol_dampen"] if vol_regime.startswith("📉") else 1.0
        if perp_cvd_delta > 0 and spot_cvd_delta < 0:
            add_bear = 2.5 * cvd_spot_mult
            bear_score += add_bear
            bull_score = max(0.0, bull_score - 2.5)  # 强空信号：多分做减法，拉开净差
            signals["bear"].append(f"🚨现货抛售背离 (+{add_bear:.1f})")
            spot_cvd_status = "🚨 危险: 合约涨现货跌"
            cvd_msg = "价格新高但现货CVD未新高，纯杠杆推升，注意趋势反转风险"
            if len(df) >= 3 and len(df_spot) >= 3:
                p3 = df["cvd"].iloc[-1] - df["cvd"].iloc[-3]
                s3 = df_spot["cvd"].iloc[-1] - df_spot["cvd"].iloc[-3]
                if p3 > 0 and s3 < 0:
                    bear_score += 0.3
                    signals["bear"].append("3K同向背离 (+0.3)")
        elif perp_cvd_delta < 0 and spot_cvd_delta > 0:
            add_bull = 2.5 * cvd_spot_mult
            bull_score += add_bull
            bear_score = max(0.0, bear_score - 2.5)  # 强多信号：空分做减法，拉开净差
            signals["bull"].append(f"🚨现货吸筹背离 (+{add_bull:.1f})")
            spot_cvd_status = "🚨 提示: 合约跌现货涨"
            cvd_msg = "价格新低但现货CVD未新低，极佳抄底信号"
            if len(df) >= 3 and len(df_spot) >= 3:
                p3 = df["cvd"].iloc[-1] - df["cvd"].iloc[-3]
                s3 = df_spot["cvd"].iloc[-1] - df_spot["cvd"].iloc[-3]
                if p3 < 0 and s3 > 0:
                    bull_score += 0.3
                    signals["bull"].append("3K同向背离 (+0.3)")

        if price_delta_pct > STRATEGY["cvd_price_thresh_pct"] and perp_cvd_delta < 0:
            if oi_delta_pct > 0.8:
                bear_score += (3.0 * vol_multiplier_mean_rev)
                bull_score = max(0.0, bull_score - 1.5)  # A级顶背离：压制多方，过滤假多
                cvd_title = "🚨 A级顶部背离 (主力派发)"
            else:
                bear_score += (1.5 * vol_multiplier_mean_rev)
                bull_score = max(0.0, bull_score - 0.5)  # B级：小幅压制
                cvd_title = "⚠️ B级顶部背离 (动能衰竭)"
        elif price_delta_pct < -STRATEGY["cvd_price_thresh_pct"] and perp_cvd_delta > 0:
            if oi_delta_pct > 0.8:
                bull_score += (3.0 * vol_multiplier_mean_rev)
                bear_score = max(0.0, bear_score - 1.5)  # A级底背离：压制空方，过滤假空
                cvd_title = "🚨 A级底部背离 (主力吸收)"
            else:
                bull_score += (1.5 * vol_multiplier_mean_rev)
                bear_score = max(0.0, bear_score - 0.5)  # B级：小幅压制
                cvd_title = "⚠️ B级底部背离 (动能衰竭)"

    # ================= MACD（两格确认或金叉/死叉） =================
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal_line
    macd_str = "震荡 | Histogram: 平缓"
    hist_bull = len(hist) >= 3 and hist.iloc[-1] > hist.iloc[-2] and hist.iloc[-2] >= hist.iloc[-3]
    hist_bear = len(hist) >= 3 and hist.iloc[-1] < hist.iloc[-2] and hist.iloc[-2] <= hist.iloc[-3]
    golden_cross = len(macd) >= 2 and macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]
    death_cross = len(macd) >= 2 and macd.iloc[-1] < signal_line.iloc[-1] and macd.iloc[-2] >= signal_line.iloc[-2]
    if (macd.iloc[-1] > 0 and (hist_bull or golden_cross)):
        bull_score += (1.0 * vol_multiplier_trend)
        signals["bull"].append(f"MACD多头 (+{1.0 * vol_multiplier_trend:.1f})")
        macd_str = "多头增强 | Histogram: 扩大"
    elif (macd.iloc[-1] < 0 and (hist_bear or death_cross)):
        bear_score += (1.0 * vol_multiplier_trend)
        signals["bear"].append(f"MACD空头 (+{1.0 * vol_multiplier_trend:.1f})")
        macd_str = "空头扩散 | Histogram: 扩大"

    # ================= HTF 大级别（仅趋势明确时加分）+ 4H 超买/超卖 =================
    htf_bullish = None
    htf_ema = None
    htf_ob = False  # 4H 超买：价格远高于 EMA50，做多从严
    htf_os = False  # 4H 超卖：价格远低于 EMA50，做空从严
    if not df_htf.empty and len(df_htf) > 10:
        df_htf["ema50"] = df_htf["close"].ewm(span=50, adjust=False).mean()
        htf_ema = float(df_htf["ema50"].iloc[-1])
        dist_pct = (current_price - htf_ema) / htf_ema if htf_ema else 0
        htf_ob = dist_pct >= STRATEGY["htf_ob_os_thresh_pct"]
        htf_os = dist_pct <= -STRATEGY["htf_ob_os_thresh_pct"]
        slope_up = len(df_htf) >= 3 and df_htf["ema50"].iloc[-1] > df_htf["ema50"].iloc[-3]
        slope_down = len(df_htf) >= 3 and df_htf["ema50"].iloc[-1] < df_htf["ema50"].iloc[-3]
        trend_bull_clear = dist_pct >= STRATEGY["htf_min_dist_pct"] or (dist_pct > 0 and slope_up)
        trend_bear_clear = dist_pct <= -STRATEGY["htf_min_dist_pct"] or (dist_pct < 0 and slope_down)
        if current_price > htf_ema:
            htf_bullish = True
            if trend_bull_clear:
                bull_score += (1.0 * vol_multiplier_trend)
                signals["bull"].append(f"顺大势 (+{1.0 * vol_multiplier_trend:.1f})")
        else:
            htf_bullish = False
            if trend_bear_clear:
                bear_score += (1.0 * vol_multiplier_trend)
                signals["bear"].append(f"顺大势 (+{1.0 * vol_multiplier_trend:.1f})")

    # 4H 超买/超卖：大级别极端时同向得分降权（30m 背离易被 4H 拉平）
    if htf_ob:
        bull_score *= STRATEGY["htf_ob_os_dampen"]
    if htf_os:
        bear_score *= STRATEGY["htf_ob_os_dampen"]

    # ================= Funding 拥挤过滤（梯度 + 极端一票否决） =================
    if fund_rate >= STRATEGY["funding_crowd_pct"]:
        bull_score *= STRATEGY["funding_crowd_dampen"]
    if fund_rate <= -STRATEGY["funding_crowd_pct"]:
        bear_score *= STRATEGY["funding_crowd_dampen"]
    if fund_rate >= STRATEGY["funding_extreme_pct"]:
        bull_score *= STRATEGY["funding_extreme_dampen"]   # 极端多拥挤，多分极重降权
    if fund_rate <= -STRATEGY["funding_extreme_pct"]:
        bear_score *= STRATEGY["funding_extreme_dampen"]  # 极端空拥挤，空分极重降权

    # ================= 多因子冲突降权 =================
    has_cvd_bear = any("现货抛售" in s for s in signals["bear"])
    has_cvd_bull = any("现货吸筹" in s for s in signals["bull"])
    if has_cvd_bear and (taker_val > 1.1 or bid_ask_ratio > 1.5):
        bear_score *= STRATEGY["ls_conflict_dampen"]
    if has_cvd_bull and (taker_val < 0.9 or bid_ask_ratio < 0.6):
        bull_score *= STRATEGY["ls_conflict_dampen"]

    # ================= 净胜差强制拉开（禁止双高，高级因子做减法） =================
    MIN_NET_GAP = STRATEGY["min_net_gap"]
    MIN_NET_GAP_STRONG = STRATEGY["min_net_gap_strong"]
    BOTH_HIGH_THRESH = STRATEGY["both_high_thresh"]

    if bull_score >= BOTH_HIGH_THRESH and bear_score >= BOTH_HIGH_THRESH:
        # 双高：强制落后方压低，确保净差 >= MIN_NET_GAP_STRONG
        if bull_score >= bear_score:
            bear_score = min(bear_score, bull_score - MIN_NET_GAP_STRONG)
        else:
            bull_score = min(bull_score, bear_score - MIN_NET_GAP_STRONG)

    raw_gap = bull_score - bear_score
    if abs(raw_gap) < MIN_NET_GAP and (bull_score >= 5.0 or bear_score >= 5.0):
        # 净差不足但有一方上了 5 分 → 似是而非，用减法压低落后方
        if raw_gap > 0:
            bear_score = min(bear_score, bull_score - MIN_NET_GAP)
        else:
            bull_score = min(bull_score, bear_score - MIN_NET_GAP)
    if abs(raw_gap) < STRATEGY["near_high_gap"] and bull_score >= STRATEGY["near_high_both"] and bear_score >= STRATEGY["near_high_both"]:
        # 净差<1.5 且两边都>=3：准双高，强制拉开到至少 MIN_NET_GAP
        if raw_gap > 0:
            bear_score = min(bear_score, bull_score - MIN_NET_GAP)
        else:
            bull_score = min(bull_score, bear_score - MIN_NET_GAP)

    bull_score = max(0.0, bull_score)
    bear_score = max(0.0, bear_score)
    net_score = abs(bull_score - bear_score)

    # =============== 核心量化逻辑推导（S/A 级使用 STRATEGY 阈值） ===============
    if bull_score >= STRATEGY["s_bull_score"] and net_score >= STRATEGY["s_net_score"]:
        final_judgment = "🟢 强力看多 (S级共振，高胜率)"
        is_trade = "long"
    elif bull_score >= STRATEGY["a_bull_score"] and net_score >= STRATEGY["a_net_score"]:
        final_judgment = "↗️ 偏多 (A级机会，标准仓位)"
        is_trade = "long"
    elif bear_score >= STRATEGY["s_bull_score"] and net_score >= STRATEGY["s_net_score"]:
        final_judgment = "🔴 强力看空 (S级共振，高胜率)"
        is_trade = "short"
    elif bear_score >= STRATEGY["a_bull_score"] and net_score >= STRATEGY["a_net_score"]:
        final_judgment = "↘️ 偏空 (A级机会，标准仓位)"
        is_trade = "short"
    else:
        final_judgment = "⚖️ 中性/观望 (Neutral)"
        is_trade = "none"

    # Funding 极端一票否决：极端多拥挤时禁止做多、极端空拥挤时禁止做空
    if STRATEGY.get("funding_extreme_veto"):
        if is_trade == "long" and fund_rate >= STRATEGY["funding_extreme_pct"]:
            is_trade = "none"
            final_judgment = "⚖️ 中性/观望 (Funding 极端多拥挤否决)"
        elif is_trade == "short" and fund_rate <= -STRATEGY["funding_extreme_pct"]:
            is_trade = "none"
            final_judgment = "⚖️ 中性/观望 (Funding 极端空拥挤否决)"

    # ================= 构建最终报告排版（看盘优先：判断 + SOP 置顶） =================
    text = f"🔔 {asset} Perp Signal — {datetime.now().strftime('%m-%d %H:%M')} SGT\n\n"

    # 【置顶】终极判断（含得分构成，便于理解「具体怎么得的分」）
    text += f"💡 判断: {final_judgment}\n"
    text += f"· 多方得分: {bull_score:.1f} | 空方得分: {bear_score:.1f} | 净差: {bull_score - bear_score:+.1f}\n"
    text += f"· 多方得分构成: {', '.join(signals['bull']) if signals['bull'] else '无'}\n"
    text += f"· 空方得分构成: {', '.join(signals['bear']) if signals['bear'] else '无'}\n"
    text += "· 说明: 上为各信号原始加分；最终得分已含 Funding/冲突/4H 超买超卖等全局调整\n\n"

    # 【置顶】量化 SOP 交易计划
    text += "📋 量化 SOP 交易计划\n"
    if is_trade == "long":
        # V8: 若有未回补看涨 FVG 且与 VWAP 共振，优先用 FVG 区间；否则 V7 价值区/VWAP
        below_pct = STRATEGY.get("entry_below_vwap_pct", 0.005)
        above_pct = STRATEGY.get("entry_above_vwap_pct", 0.01)
        entry_zone_source = "VWAP"
        fvg_bull = fvgs.get("bull")
        use_fvg_long = fvg_bull and _fvg_near_vwap(fvg_bull, vwap, fvg_vwap_dev)
        if use_fvg_long and fvg_bull.get("mid"):
            fvg_width_pct = (fvg_bull["top"] - fvg_bull["bottom"]) / fvg_bull["mid"]
            if fvg_width_pct < STRATEGY.get("fvg_min_width_pct", 0.05) / 100.0:
                use_fvg_long = False
        if use_fvg_long:
            entry_low = max(val, fvg_bull["bottom"])
            entry_high = min(fvg_bull["top"], vwap * (1 + above_pct))
            if current_price > vwap:
                entry_high = min(entry_high, current_price)
            if entry_low >= entry_high:
                use_fvg_long = False
            else:
                entry_zone_source = "FVG"
        if not use_fvg_long:
            if current_price <= vwap:
                entry_low = max(val, current_price * (1 - below_pct))
                entry_high = min(vwap * (1 + above_pct), vwap + current_atr * 0.5)
                if entry_low > entry_high:
                    entry_low = max(val, vwap * (1 - below_pct))
                    entry_high = vwap
            else:
                entry_low = max(val, vwap * (1 - below_pct))
                entry_high = min(current_price, max(vah, vwap * (1 + above_pct)))
                if entry_low > entry_high:
                    entry_low = vwap * (1 - below_pct)
                    entry_high = min(current_price, vwap * (1 + above_pct))
        # 区间过窄时略放宽上沿（至少约 0.3% 宽度便于挂单）
        zone_width_pct = (entry_high - entry_low) / entry_low if entry_low > 0 else 0
        if 0 < zone_width_pct < 0.003 and current_price > vwap:
            entry_high = min(current_price, entry_low * 1.005)
        entry_avg = (entry_high + entry_low) / 2
        atr_sl = STRATEGY["sop_atr_sl_mult"] * current_atr
        sl_price = swing_low - atr_sl
        risk = entry_avg - sl_price
        # 止损必须低于进场区间下沿，否则一入场就处于止损之上
        if sl_price >= entry_low:
            sl_price = entry_low - 0.5 * current_atr
            risk = entry_avg - sl_price
        min_risk = 0.5 * current_atr
        if risk < min_risk:
            risk = min_risk
            sl_price = entry_avg - risk
        tp1_price = entry_avg + (risk * 1.5)
        tp2_price = swing_high if swing_high > tp1_price + (risk * 0.5) else tp1_price + risk
        if tp2_price < tp1_price:
            tp2_price = tp1_price + risk  # 做多 TP2 不得低于 TP1

        rr_tp2 = (tp2_price - entry_avg) / risk if risk > 0 else 0
        risk_status = "🟢 准许执行"
        if htf_ob:
            risk_status = "🔴 风控拦截 (4H 超买，做多风险极高)"
        elif rr_tp2 < 1.5:
            risk_status = "🔴 风控拦截 (整体盈亏比结构不足)"
        elif not htf_bullish:
            risk_status = "🔴 风控拦截 (逆 4H 大级别趋势)"

        text += "· 策略: 逢低做多 (Limit Long)\n"
        if current_price > entry_high:
            hint = "回踩FVG与VWAP共振区挂单，当前价已高不追" if entry_zone_source == "FVG" else "回踩价值区/VWAP挂单，当前价已高不追"
            text += f"· 进场: ${entry_low:,.0f} - ${entry_high:,.0f} ({hint})\n"
        else:
            hint = f"FVG与VWAP共振区 (中轴≈VWAP)" if entry_zone_source == "FVG" else f"价值区至VWAP+{above_pct*100:.0f}%"
            text += f"· 进场: ${entry_low:,.0f} - ${entry_high:,.0f} ({hint})\n"
        text += f"· 止损: ${sl_price:,.0f} (前低+{STRATEGY['sop_atr_sl_mult']}×ATR缓冲，必低于进场下沿)\n"
        text += f"· 1R = ${risk:,.0f} (按区间中值) | TP2 盈亏比: {rr_tp2:.1f}R\n"
        text += f"· 止盈: TP1 ${tp1_price:,.0f} (1.5R) | TP2 ${tp2_price:,.0f} (波段)\n"
        text += f"· 风控: {risk_status}\n"
        text += f"· 时效: 信号生成后 {STRATEGY['sop_valid_hours']} 小时内有效，若下一4H开盘前未入场可视为失效\n\n"

    elif is_trade == "short":
        # V8: 若有未回补看跌 FVG 且与 VWAP 共振，优先用 FVG 区间；否则 V7 价值区/VWAP
        above_pct = STRATEGY.get("entry_below_vwap_pct", 0.005)
        below_pct = STRATEGY.get("entry_above_vwap_pct", 0.01)
        entry_zone_source = "VWAP"
        fvg_bear = fvgs.get("bear")
        use_fvg_short = fvg_bear and _fvg_near_vwap(fvg_bear, vwap, fvg_vwap_dev)
        if use_fvg_short and fvg_bear.get("mid"):
            fvg_width_pct = (fvg_bear["top"] - fvg_bear["bottom"]) / fvg_bear["mid"]
            if fvg_width_pct < STRATEGY.get("fvg_min_width_pct", 0.05) / 100.0:
                use_fvg_short = False
        if use_fvg_short:
            entry_low = max(fvg_bear["bottom"], vwap * (1 - below_pct))
            entry_high = min(fvg_bear["top"], vah if current_price < vwap else current_price)
            if entry_low >= entry_high:
                use_fvg_short = False
            else:
                entry_zone_source = "FVG"
        if not use_fvg_short:
            if current_price >= vwap:
                entry_low = max(vah, vwap * (1 - below_pct))
                entry_high = current_price
                if entry_low > entry_high:
                    entry_low = vwap * (1 - below_pct)
                    entry_high = current_price
            else:
                entry_high = min(vah, vwap * (1 + above_pct))
                entry_low = max(current_price, min(val, vwap * (1 - below_pct)))
                if entry_low > entry_high:
                    entry_high = vwap
                    entry_low = max(current_price, vwap * (1 - below_pct))
        zone_width_pct = (entry_high - entry_low) / entry_low if entry_low > 0 else 0
        if 0 < zone_width_pct < 0.003 and current_price < vwap:
            entry_low = max(entry_high * 0.995, val)
        entry_avg = (entry_high + entry_low) / 2
        atr_sl = STRATEGY["sop_atr_sl_mult"] * current_atr
        sl_price = swing_high + atr_sl
        risk = sl_price - entry_avg
        # 止损必须高于进场区间上沿，否则空单一入场就处于止损之下
        if sl_price <= entry_high:
            sl_price = entry_high + 0.5 * current_atr
            risk = sl_price - entry_avg
        min_risk = 0.5 * current_atr
        if risk < min_risk:
            risk = min_risk
            sl_price = entry_avg + risk
        tp1_price = entry_avg - (risk * 1.5)
        tp2_price = swing_low if swing_low < tp1_price - (risk * 0.5) else tp1_price - risk
        if tp2_price > tp1_price:
            tp2_price = tp1_price - risk  # 做空 TP2 不得高于 TP1

        rr_tp2 = (entry_avg - tp2_price) / risk if risk > 0 else 0
        risk_status = "🟢 准许执行"
        if htf_os:
            risk_status = "🔴 风控拦截 (4H 超卖，做空风险极高)"
        elif rr_tp2 < 1.5:
            risk_status = "🔴 风控拦截 (整体盈亏比结构不足)"
        elif htf_bullish:
            risk_status = "🔴 风控拦截 (逆 4H 大级别趋势)"

        text += "· 策略: 逢高做空 (Limit Short)\n"
        if current_price < entry_low:
            hint = "反弹至FVG与VWAP共振区挂单，当前价已低不追" if entry_zone_source == "FVG" else "反弹价值区/VWAP挂单，当前价已低不追"
            text += f"· 进场: ${entry_low:,.0f} - ${entry_high:,.0f} ({hint})\n"
        else:
            hint = "FVG与VWAP共振区 (中轴≈VWAP)" if entry_zone_source == "FVG" else "价值区至VWAP"
            text += f"· 进场: ${entry_low:,.0f} - ${entry_high:,.0f} ({hint})\n"
        text += f"· 止损: ${sl_price:,.0f} (前高+{STRATEGY['sop_atr_sl_mult']}×ATR缓冲，必高于进场上沿)\n"
        text += f"· 1R = ${risk:,.0f} (按区间中值) | TP2 盈亏比: {rr_tp2:.1f}R\n"
        text += f"· 止盈: TP1 ${tp1_price:,.0f} (1.5R) | TP2 ${tp2_price:,.0f} (波段)\n"
        text += f"· 风控: {risk_status}\n"
        text += f"· 时效: 信号生成后 {STRATEGY['sop_valid_hours']} 小时内有效，若下一4H开盘前未入场可视为失效\n\n"
    else:
        text += "· 策略: 暂停交易 (系统静默过滤)\n\n"

    # 【以下为明细】市场快照 / 衍生品 / 关键价位 / 边缘信号
    text += "📊 市场快照\n"
    text += f"· 价格: ${current_price:,.2f} | Mark: ${mark_price:,.2f}\n"
    text += f"· 24h: {price_change_pct:+.2f}% | H: ${high_24h:,.0f} | L: ${low_24h:,.0f}\n"
    text += f"· 24h Vol: ${vol_24h/1e9:,.2f}B | 波动率: {vol_regime} | BBW: {current_bbw:.2f}% (均线: {avg_bbw:.2f}%)\n"
    text += f"💬 {market_snapshot_msg}\n\n"

    next_funding_str = datetime.utcfromtimestamp(next_funding_ts / 1000).strftime("%m-%d %H:%M UTC") if next_funding_ts else "—"
    text += "📈 衍生品\n"
    text += f"· OI: {current_oi:,.0f} {asset}{oi_change_str}\n"
    text += f"· Funding: {fund_rate:.4f}% (8h) {fund_tag} | 下次结算: {next_funding_str}\n"
    text += f"· 溢价: {premium_pct:+.4f}% (Mark vs Index)\n"
    text += f"· L/S Ratio (Pos): {ls_pos_val:.3f} | (Acct): {ls_acct_val:.3f}\n"
    text += f"· Taker B/S: {taker_val:.3f}\n"
    text += f"💬 {deriv_msg}\n\n"

    text += "🎯 关键价位\n"
    text += f"· Resistance: ${swing_high:,.0f} (Swing High) | 向上清算: ${liq_short_100x:,.0f}\n"
    text += f"· Support: ${swing_low:,.0f} (Swing Low) | 向下清算: ${liq_long_100x:,.0f}\n"
    text += f"· 距上方清算: {dist_short_pct:.2f}% | 距下方清算: {dist_long_pct:.2f}%\n"
    text += f"· VWAP 24h: ${vwap:,.0f}\n"
    text += f"· VP → VAH: ${vah:,.0f} | POC: ${poc:,.0f} | VAL: ${val:,.0f}\n"
    if fvgs.get("bull") or fvgs.get("bear"):
        fvg_parts = []
        if fvgs.get("bull"):
            b = fvgs["bull"]
            fvg_parts.append(f"看多 ${b['bottom']:,.0f}-${b['top']:,.0f} (中轴${b['mid']:,.0f})")
        if fvgs.get("bear"):
            b = fvgs["bear"]
            fvg_parts.append(f"看空 ${b['bottom']:,.0f}-${b['top']:,.0f} (中轴${b['mid']:,.0f})")
        text += f"· FVG(30m): {' | '.join(fvg_parts)}\n"
    text += f"· ATR(14): ${current_atr:,.2f} (止损缓冲参考)\n"
    text += f"💬 {vp_status_msg}\n\n"

    htf_ema_str = f"${htf_ema:,.0f}" if htf_ema is not None else "—"
    cvd_detail = ""
    if perp_cvd_delta is not None and spot_cvd_delta is not None:
        cvd_detail = f" | 近5K 合约CVD: {perp_cvd_delta:+.0f} 现货: {spot_cvd_delta:+.0f}"
        if oi_delta_pct is not None:
            cvd_detail += f" | OI: {oi_delta_pct:+.1f}%"
    text += "📡 边缘信号\n"
    text += f"· CVD: {spot_cvd_status} | {cvd_title}{cvd_detail}\n"
    text += f"· 4H EMA50: {htf_ema_str}\n"
    text += f"· MACD: {macd_str}\n"
    text += f"· Order Book: Bid/Ask = {bid_ask_ratio:.2f} {ob_tag}\n"
    text += f"💬 {cvd_msg}; {ob_msg}\n\n"

    judgment_emoji = final_judgment[0] if final_judgment else "⚖️"
    return text, judgment_emoji

# ================= 📧 发送模块与主引擎 =================
def send_report_by_email(full_report_text, subject_title="Crypto", log=None):
    import smtplib
    from email.mime.text import MIMEText
    from email.utils import formataddr

    cfg = EMAIL_CONFIG
    to_email = (cfg.get("to_email") or "").strip()
    if not to_email: return False
    from_email = (cfg.get("from_email") or "").strip() or to_email
    smtp_host = cfg.get("smtp_host") or "smtp.qq.com"
    smtp_port = int(cfg.get("smtp_port") or 465)
    password = (cfg.get("password") or os.environ.get("EMAIL_PASSWORD") or "").strip()
    if not password: return False
    
    try:
        msg = MIMEText(full_report_text, "plain", "utf-8")
        msg["Subject"] = f"{subject_title} — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        msg["From"] = formataddr((subject_title, from_email))
        msg["To"] = to_email
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(from_email, password)
            server.sendmail(from_email, [e.strip() for e in to_email.split(",")], msg.as_string())
        if log: log(f"  📧 报告已发送至: {to_email}")
        return True
    except Exception as e:
        if log: log(f"  ❌ 发邮件失败: {e}")
        return False

def next_half_hour_with_delay(delay_seconds=5):
    now = datetime.now()
    if now.minute < 30:
        next_run = now.replace(minute=30, second=0, microsecond=0)
    else:
        next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    next_run = next_run + timedelta(seconds=delay_seconds)
    sec = (next_run - now).total_seconds()
    return next_run, max(0, sec)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(os.path.dirname(__file__), "perp_signal_result.txt")
    symbols = CONFIG.get("symbols") or ["BTCUSDT", "ETHUSDT"]

    def log(s):
        print(s, flush=True)
        return s

    while True:
        log_lines = []
        _log = lambda s: log_lines.append(log(s))

        try:
            t0 = time.time()
            asset_names = [_asset(s) for s in symbols]
            _log(f"🚀 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 启动微观结构分析: {', '.join(asset_names)}")
            reports = []
            subject_parts = []
            
            for sym in symbols:
                try:
                    raw_data = get_binance_data_optimized(sym, log=_log)
                    message, judgment_emoji = calculate_and_analyze(raw_data, sym, log=_log)
                    reports.append(message)
                    subject_parts.append(f"{_asset(sym)}({judgment_emoji})")
                except Exception as e:
                    _log(f"  ❌ [{_asset(sym)}] 处理异常: {e}")
                    reports.append(f"❌ [{_asset(sym)}] 遭遇核心崩溃: {e}")
                    subject_parts.append(f"{_asset(sym)}(⚖️)")

            t1 = time.time()
            elapsed = t1 - t0
            subject_title = "微观雷达 - " + "、".join(subject_parts)
            separator = "\n" + "=" * 50 + "\n\n"
            message = separator.join(reports)

            print("\n" + "=" * 50 + "\n")
            print(message)
            print(f"\n⚡ 引擎耗时: {elapsed:.2f} 秒")
            print("=" * 50 + "\n")

            full_report = "\n".join(log_lines) + "\n\n" + "=" * 50 + "\n\n" + message + f"\n⚡ 引擎耗时: {elapsed:.2f} 秒\n" + "=" * 50 + "\n"
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(full_report)
                _log(f"📁 结果已更新至本地日志")
            except Exception as e:
                _log(f"  ⚠ 写入文件失败: {e}")

            email_body = message + f"\n\n⚡ 引擎耗时: {elapsed:.2f} 秒"
            send_report_by_email(email_body, subject_title=subject_title, log=_log)

        except Exception as e:
            _log(f"\n❌ 挂机引擎执行时出错: {e}")
            
        next_run, sleep_sec = next_half_hour_with_delay(delay_seconds=5)
        _log(f"⏰ 报告已生成，休眠中... 下次精准扫描时间: {next_run.strftime('%Y-%m-%d %H:%M:%S')} （约 {int(sleep_sec)} 秒后）\n")
        
        time.sleep(sleep_sec)
#!/usr/bin/env python3
"""
ZCT 风格 VWAP + 关键位 量化信号扫描（币安 U 本位永续）

依据 **ZCT「VWAP // MASTERCLASS」** 海报实现可复现规则（非投资建议）：
- **锚定**：会话 VWAP，**UTC 自然日**重置；**±1σ** 带宽（与海报一致）。
- **三张主信号（按海报顺序）**：① 价位 vs VWAP（方向）② VWAP **斜率**（动能/震荡）③ **带宽**（宽=延续，窄=犹豫/均值回归）。
- **Bonus：会话内价 vs VWAP 交叉次数** — 海报刻度 **0–3**（偏趋势突破）、**4–6**（混合）、**7+**（偏震荡/反转语境）。
- **Play 01/02**：价在锚一侧 + **陡斜率 + 宽轨** → 顺势（突破多 / 破位空）；**Play 03**：**平斜率 + 窄轨** → 贴轨做均值回归，目标收回 VWAP。
- **Setup level 1–3**：三信号与模板的一致程度；海报 **「use level 3+」** 对应本脚本 `setup_level==3`（严格模板：PLAY01_BREAKOUT / PLAY02_BREAKDOWN / PLAY03_REV）。默认 **`ZCT_ENFORCE_SETUP_LEVEL` 开启** 且 **`ZCT_MIN_SETUP_LEVEL=3`**，仅该档保留带 SL/TP 的方向单；可关 enforce 或降为 2 以放宽。
- ZCT 关键位参考：前日高/低 + 4H/1H/15m 前一根完整 K 的高/低（辅助，非海报核心三信号）。
- **实盘与 walk-forward 对齐**：以最后一根 1m 的 `open_time` 为 **asof**；当日 UTC 0 点起 forward 拉 1m、**`session_slice_utc_day`** 切会话、**`RefLevelResolver`** 给关键位；Play03 的 SPIKE ATR 经 **`classify_and_signal(..., spike_klines_end_ms=asof)`** 截断，避免多看未来外周期。
- **近端 recycled 否决（默认关）**：设 `ZCT_RECYCLED_NEAR_VETO_ENABLED=1` 时，PLAY01/02 若头顶最近结构阻力（或脚下最近结构支撑）在距离阈值内且 `_level_freshness_row` 为 `recycled`，则改为 NO_TRADE（ZCT S/R：烂墙附近少追顺势）。见 `ZCT_RECYCLED_NEAR_VETO_*`。

用法：
  python zct_vwap_signal_scanner.py              # 跑一次，stdout + 可选 TG
  python zct_vwap_signal_scanner.py --no-tg     # 仅打印

定时：由 next-k-api main.py APScheduler 调用（需 ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED=1），
      默认全量扫描每 12 分钟、独立结算(resolve-only)每 5 分钟（IntervalTrigger，环境变量可调）；
      主 lane 子进程注入 **ZCT_TOUCH_POOL_UNIVERSE=1**，标的仅从 **accumulation.db / zct_vwap_touch_pool**
      读取（须先跑触轨资产池 daily job 或 touch-pool-scan）；表空则本轮跳过扫描。
      亦可自建 cron 执行本脚本。

信号与结算统一写入 **zct_vwap_signals / zct_vwap_settlements**（旧 zct_hot_oi_* 在 accumulation init_db 时一次性并入后删除）。

环境变量：
  ZCT_VWAP_SYMBOLS     逗号分隔永续标的；不设则用内置默认列表（见 `_DEFAULT_ZCT_SYMBOLS`）。
                        **若 ZCT_TOUCH_POOL_UNIVERSE=1，本项被忽略**（以触轨表为准）。
  ZCT_VWAP_BAND_SIGMA  默认 1.0
  ZCT_VWAP_DB_SKIP_FLAT  设为 1 时不入库 side=FLAT 的行（减轻 NO_TRADE 噪音）
  ZCT_BTC_MACRO_FILTER_ENABLED  默认 **开启**「BTC 大盘红绿灯」：BTC VWAP 斜率极陡且非高震荡时，
                        拦截山寨逆势多/空；强多时尚禁空，并对多单做相对强弱(RS)与极端斜率熔断（BTC 自身不过滤；
                        本轮将 BTCUSDT 提前扫描以刷新缓存）；设为 0/false/off 关闭
  ZCT_BTC_MACRO_SLOPE_THRESHOLD_BPS  判定强单边行情的 |斜率| 下限（基点），默认 3.0，与 res.slope_bps 同单位
  ZCT_BTC_MACRO_RS_MIN_RATIO  大盘强多(BTC_STRONG_UP)时，山寨做多要求 slope_bps ≥ BTC×该比例，默认 0.5（防吸血假突破）
  ZCT_BTC_MACRO_LONG_FUSE_SLOPE_BPS  大盘强多且 BTC 斜率超过该值(bps)时一律拒接山寨多单，默认 8.0；设为 0 关闭该熔断
  ZCT_ENFORCE_SETUP_LEVEL  默认 **开启**：仅当 setup_level≥ZCT_MIN_SETUP_LEVEL 才保留方向单+SL/TP；设为 0/false/off 关闭
  ZCT_MIN_SETUP_LEVEL      默认 **3**（海报 level 3+）；设为 2 可在开启 gate 时纳入 BIAS/TRANSITION 等
  ZCT_VWAP_CROSS_MAX_LOW   VWAP 交叉刻度「0–3」上界，默认 3
  ZCT_VWAP_CROSS_MAX_MID   「4–6」上界，默认 6；交叉数>此值归入「7+」
  # P1/P2（默认名义仍为 保证金×杠杆；开启固定风险见下一行）
  ZCT_ACCOUNT_EQUITY_USDT    纸面权益（USDT），默认 10000；用于风险名义与日损熔断分母
  ZCT_RISK_PCT_PER_TRADE     单笔风险占权益，默认 0.005
  ZCT_USE_RISK_SIZED_NOTIONAL 设为 1 时按「权益×风险÷止损距离」推算名义（上限 ZCT_MAX_NOTIONAL_CAP_USDT）
  ZCT_MAX_DAILY_LOSS_PCT     当日已实现合计亏损 ≥ 权益×该比例 则暂停新开方向单（UTC 日），默认 0.05；0=关闭
  ZCT_MAX_BAND_WIDTH_PCT     band_width_pct 大于则跳过方向单；默认 15（极端宽轨过滤）；设为 0 关闭
  平仓冷却（毫秒）：脚本内常量 COOLDOWN_AFTER_LOSS_MS / COOLDOWN_AFTER_WIN_MS / COOLDOWN_AFTER_CLOSE_MS
                    （止损/止盈默认各 30min、任意平仓额外间隔默认 0；写入 zct_symbol_cooldown，非环境变量）
  同标的「持仓中」保护：若已有未平仓 LONG/SHORT（与看板一致），默认**跳过**入库以免洗掉 SL/TP。
                        **例外**：与持仓**方向相反**时先按扫描价纸面平仓（supersede）再写入快照。
                        观望(FLAT)是否也平仓：见代码常量 SCAN_SUPERSEDE_ON_FLAT（默认 False=仅反向平仓，FLAT 保留仓至 resolve）。
  TG_BOT_TOKEN / TG_CHAT_ID  与 accumulation 雷达相同；配置后即推送 Telegram
  ZCT_VWAP_TG_PUSH_MODE  扫描推送：summary（默认，每轮一条简报）| actionable（仅当有方向+SL/TP）
                        | all（每轮全文明细）| off（不推扫描，平仓推送仍受 NOTIFY_RESOLVE 控制）
  ZCT_VWAP_TG_NOTIFY_RESOLVE  平仓结算是否推 TG，默认 1
  ZCT_STRICT_PA_FILTERS  默认 1 启用「A 级」附加过滤：①顺势需两根收盘站轨外+vol>均量+慢磨靠近
                        ②反转 Play03 需近窗刺穿柱+假破收回轨内+震荡量能条件；设为 0/false/off 关闭
  ZCT_VOL_MA_PERIOD / ZCT_SPIKE_LOOKBACK / ZCT_SPIKE_RANGE_RATIO / ZCT_GRIND_LOOKBACK /
  ZCT_GRIND_MAX_NET_MOVE_PCT / ZCT_LEVEL_TOUCH_LOOKBACK_BARS / ZCT_LEVEL_FRESH_MIN_BARS /
  ZCT_LEVEL_RECYCLE_TOUCH_MIN  用于 nearest_levels / fresh 判定（触碰≥此值标 recycled）；**分类决策**见下「近端 recycled 否决」
  ZCT_LEVEL_FRESH_MIN_HOURS  与 Koroush S/R「约 6–8h 未触碰」对齐：>0 时 fresh 判定优先用墙上时钟（小时），0=仅用根数 ZCT_LEVEL_FRESH_MIN_BARS
  ZCT_RECYCLED_NEAR_VETO_ENABLED  默认 **0（关）**；设为 **1|true|yes|on** 开启（只要不是 `0|false|no|off|disabled` 即视为开）：上方/下方最近结构位（pdh/h4/h1/m15 高或低）距现价 ≤ `ZCT_RECYCLED_NEAR_MAX_DIST_PCT` 且新鲜度为 **recycled** 时，否决 **PLAY01_BREAKOUT_LONG** / **PLAY02_BREAKDOWN_SHORT**
  ZCT_RECYCLED_NEAR_MAX_DIST_PCT  否决用距离上限（占现价 **%**），默认 **0.2**；≤0 时回退为 0.2
  ZCT_PLAY03_TP_MODE  PLAY03 止盈：vwap（默认，回锚）| 1r（与 My Reversal Lesson4 一致：与 SL 等距 1:1）
  ZCT_KOROUSH_MIN_STOP_DISTANCE_PCT  止损距进场最小占价比（默认 **0.01=1%**）；不足时扩大摆动窗寻更远极值（Koroush SL）；设为 **0** 关闭扩止损（仅保留 ZCT_MIN_SL_PCT）
  ZCT_PSYCH_LEVELS  设为 1 时将大整数心理位并入 nearest_levels 距离排序（ZCT S/R 文 bonus）
  ZCT_BREAKOUT_MAX_MA_CROSSES  顺势突破/破位：近窗 MA30 交叉数超过则观望（0=关闭，对齐 Breakout 文「minimal crossovers」）
  ZCT_SPIKE_USE_ATR_15M   默认 1：Play03 刺穿阈值用「15m ATR%×倍数」动态伸缩（山寨相对 BTC 更严/更松随波动率）；0 关闭
  ZCT_SPIKE_ATR_INTERVAL  默认 15m；ZCT_SPIKE_ATR_PERIOD 默认 14；ZCT_SPIKE_ATR_MULT 默认 1.25
  ZCT_SPIKE_ATR_RATIO_FLOOR / ZCT_SPIKE_ATR_RATIO_CAP  动态阈值上下限（占价比小数）；ZCT_SPIKE_ATR_KLINE_LIMIT 拉线根数

入库：accumulation.db 表 zct_vwap_signals **每永续标的仅一行**（UPSERT），表示当前观望/方向单快照；
已平仓记录写入 zct_vwap_settlements（汇总与「已结算」列表）。定向单写入
sl_price / tp_price / r_unit / entry_bar_open_ms；resolve 用 1m K 判定 SL/TP，
回填 outcome 并归档 settlements。

环境变量（止盈止损）：
  ZCT_SWING_LOOKBACK      摆动窗口（根 1m），默认 20
  ZCT_MIN_SL_PCT          最小止损距离（占价比），默认 0.003
  ZCT_SL_BUFFER_BPS       σ 带 / 摆动外侧缓冲（基点），默认 2
  ZCT_RESOLVE_MAX_HOLD_MS  自 entry_bar_open_ms 起最长持仓（毫秒），默认 21600000（6h）；0=仅用根数上限
  ZCT_RESOLVE_MAX_BARS    未触轨最长等待根数（安全阀），默认 720；与墙上时钟满足其一即 expired
  ZCT_RESOLVE_INTER_SYMBOL_SLEEP_SEC  结算(resolve)时按标的顺序请求币安 K 线，每处理完上一标的后休眠秒数；默认 0；
                        标的多或结算 cron 较频时可设 5，减轻权重限制风险。
                        更高频时亦可考虑 U 本位 K 线 WebSocket 做 SL/TP、REST 仅断线补偿（当前实现为每持仓标的 REST 拉 1m）
  ZCT_SAME_BAR_RULE       pessimistic | optimistic，同根同时触轨时先后，默认 pessimistic
  ZCT_VIRTUAL_NOTIONAL_USDT  单笔保证金（USDT），默认 100；名义敞口 = 保证金 × ZCT_LEVERAGE
  ZCT_LEVERAGE               杠杆倍数，默认 10；盈亏按名义敞口计算（等价于保证金×杠杆）
  OI 过滤：`LIQUIDITY_OI_FILTER_ENABLED`（默认 False 关闭）为 True 时拉 `openInterestHist`，
  用前一根相对前前一根的环比；未达阈值则在 `analyze_symbol` **硬挡方向单**。详见脚本内流动性常量。

统计示例：

  SELECT outcome, COUNT(*) FROM zct_vwap_signals
    WHERE side IN ('LONG','SHORT') GROUP BY outcome;

  SELECT AVG(pnl_r) FROM zct_vwap_signals WHERE outcome='win';

  SELECT SUM(pnl_usdt) FROM zct_vwap_signals WHERE pnl_usdt IS NOT NULL;
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import io
import time
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

# 主 lane：触轨资产库标的（与 main 定时子进程注入一致）
if "--touch-pool" in sys.argv:
    os.environ.setdefault("ZCT_TOUCH_POOL_UNIVERSE", "1")

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


def _sql_ident(name: str, default: str) -> str:
    s = (name or "").strip()
    if s and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s):
        return s
    return default


ZCT_DB_SIGNALS_TABLE = _sql_ident(
    os.getenv("ZCT_DB_SIGNALS_TABLE", "zct_vwap_signals"), "zct_vwap_signals"
)
ZCT_DB_SETTLEMENTS_TABLE = _sql_ident(
    os.getenv("ZCT_DB_SETTLEMENTS_TABLE", "zct_vwap_settlements"),
    "zct_vwap_settlements",
)


def _touch_pool_universe_enabled() -> bool:
    return os.getenv("ZCT_TOUCH_POOL_UNIVERSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _filter_symbols_to_binance_usdt_perps(raw: List[str]) -> List[str]:
    """
    worth_watch 里可能出现非 U 本位永续、已下架或格式不一致的 symbol。
    仅保留币安 /fapi exchangeInfo 中 status=TRADING 的 USDT 永续，其余跳过。
    """
    if not raw:
        return []
    try:
        from accumulation_radar import get_all_perp_symbols

        allowed = set(get_all_perp_symbols())
    except Exception as e:
        print(f"[hot_oi] get_all_perp_symbols 失败，沿用原始列表（可能含无效合约）: {e}")
        return [s.strip().upper() for s in raw if s and str(s).strip()]
    kept: List[str] = []
    skipped: List[str] = []
    seen: Set[str] = set()
    for s in raw:
        u = str(s).strip().upper()
        if not u or u in seen:
            continue
        seen.add(u)
        if u in allowed:
            kept.append(u)
        else:
            skipped.append(u)
    if skipped:
        preview = skipped[:15]
        tail = "..." if len(skipped) > 15 else ""
        print(
            f"[hot_oi] 跳过 {len(skipped)} 个非 USDT 永续或未上市合约: "
            f"{preview}{tail}"
        )
    return kept


def _symbols_hot_oi_from_db() -> List[str]:
    """标的来自值得关注 · 🔥⚡热度+OI（worth_watch_hot_oi）。"""
    from accumulation_radar import init_db

    conn = init_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT symbol FROM worth_watch_hot_oi
            ORDER BY COALESCE(rank_in_category, 999) ASC, symbol ASC
            """
        )
        raw = [str(x[0]).strip().upper() for x in cur.fetchall() if x and x[0]]
        filtered = _filter_symbols_to_binance_usdt_perps(raw)
        if raw and not filtered:
            print(
                "[warn] hot_oi: worth_watch_hot_oi 有标的但均无有效币安 U 本位永续(TRADING)，"
                "请核对是否与合约代码一致（如 1000SHIB 对应 1000SHIBUSDT）"
            )
        return filtered
    except Exception as e:
        print(f"[hot_oi] worth_watch_hot_oi 读取失败: {e}")
        return []
    finally:
        conn.close()


def hot_oi_watchlist_symbols() -> List[str]:
    """worth_watch_hot_oi 当前标的（已过滤为币安 U 本位永续；供触轨资产池合并等）。"""
    return _symbols_hot_oi_from_db()


def _scan_title_short() -> str:
    if _touch_pool_universe_enabled():
        return "ZCT VWAP · 触轨资产池"
    return "ZCT VWAP"


# 默认监控 U 本位永续（可通过 ZCT_VWAP_SYMBOLS 覆盖）。
# SHIB/PEPE 在币安合约为 1000SHIBUSDT、1000PEPEUSDT（标的报价按「千枚」计）。
_DEFAULT_ZCT_SYMBOLS = (
    "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,ADAUSDT,1000SHIBUSDT,1000PEPEUSDT,"
    "DOGEUSDT,BNBUSDT,LINKUSDT,GALAUSDT,LTCUSDT,BCHUSDT,SUIUSDT,"
    "DOTUSDT,UNIUSDT,AVAXUSDT,AXSUSDT,MANAUSDT,ZECUSDT,TAOUSDT,ONDOUSDT,"
    "ARBUSDT,OPUSDT,NEARUSDT,ATOMUSDT,WLDUSDT,INJUSDT,JUPUSDT,TIAUSDT,"
    "HBARUSDT,1000BONKUSDT,WIFUSDT"
)


def _symbols_touch_pool_from_db() -> List[str]:
    """主 lane：标的来自触轨入选表 zct_vwap_touch_pool（与 daily job / touch-pool-scan 写入同源）。"""
    from zct_vwap_touch_pool_db import touch_pool_list_symbols

    return touch_pool_list_symbols()


def _symbols_from_env() -> List[str]:
    if _touch_pool_universe_enabled():
        syms = _symbols_touch_pool_from_db()
        if syms:
            return syms
        print(
            "[warn] ZCT_TOUCH_POOL_UNIVERSE：zct_vwap_touch_pool 无标的，本轮跳过扫描"
            "（请先跑 zct_vwap_asset_pool_daily_job 或 POST /api/zct-vwap/touch-pool-scan）"
        )
        return []
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

# --- BTC 大盘红绿灯（山寨逆势宏观熔断；判定逻辑见 check_btc_macro_permission）---
_BTC_MACRO_RAW = os.getenv("ZCT_BTC_MACRO_FILTER_ENABLED", "1").strip().lower()
BTC_MACRO_FILTER_ENABLED = _BTC_MACRO_RAW not in ("0", "false", "no", "off", "disabled")
BTC_MACRO_SLOPE_THRESHOLD_BPS = float(os.getenv("ZCT_BTC_MACRO_SLOPE_THRESHOLD_BPS", "3.0"))
try:
    BTC_MACRO_RS_MIN_RATIO = float(os.getenv("ZCT_BTC_MACRO_RS_MIN_RATIO", "0.5"))
except ValueError:
    BTC_MACRO_RS_MIN_RATIO = 0.5
BTC_MACRO_RS_MIN_RATIO = max(0.0, BTC_MACRO_RS_MIN_RATIO)
try:
    BTC_MACRO_LONG_FUSE_SLOPE_BPS = float(os.getenv("ZCT_BTC_MACRO_LONG_FUSE_SLOPE_BPS", "8.0"))
except ValueError:
    BTC_MACRO_LONG_FUSE_SLOPE_BPS = 8.0
_BTC_MACRO_STATE: Dict[str, Any] = {"slope_bps": 0.0, "chop": "high"}

# --- A 级 PA 过滤（教程：收盘确认 / 量 / 刺穿速度 / 关键位新鲜度）---
_STRICT_PA_RAW = os.getenv("ZCT_STRICT_PA_FILTERS", "1").strip().lower()
STRICT_PA_FILTERS = _STRICT_PA_RAW not in ("0", "false", "no", "off", "disabled")
VOL_MA_PERIOD = int(os.getenv("ZCT_VOL_MA_PERIOD", "10"))
SPIKE_LOOKBACK = int(os.getenv("ZCT_SPIKE_LOOKBACK", "5"))
# 近 SPIKE_LOOKBACK 根内 max((H-L)/mid) ≥ 阈值视为「刺穿」（Play03）；默认阈值由 15m ATR% 动态生成，失败时回退本固定比例
SPIKE_RANGE_RATIO = float(os.getenv("ZCT_SPIKE_RANGE_RATIO", "0.004"))
_SPIKE_ATR_USE_RAW = os.getenv("ZCT_SPIKE_USE_ATR_15M", "1").strip().lower()
SPIKE_USE_ATR_15M = _SPIKE_ATR_USE_RAW not in ("0", "false", "no", "off", "disabled")
SPIKE_ATR_INTERVAL = os.getenv("ZCT_SPIKE_ATR_INTERVAL", "15m").strip() or "15m"
SPIKE_ATR_PERIOD = int(os.getenv("ZCT_SPIKE_ATR_PERIOD", "14"))
SPIKE_ATR_MULT = float(os.getenv("ZCT_SPIKE_ATR_MULT", "1.25"))
# 动态阈值 = clamp(mult * ATR/close, floor, cap)；低波动币不低于 floor，极端行情不超过 cap
SPIKE_ATR_RATIO_FLOOR = float(os.getenv("ZCT_SPIKE_ATR_RATIO_FLOOR", "0.0009"))
SPIKE_ATR_RATIO_CAP = float(os.getenv("ZCT_SPIKE_ATR_RATIO_CAP", "0.025"))
SPIKE_ATR_KLINE_LIMIT = int(os.getenv("ZCT_SPIKE_ATR_KLINE_LIMIT", "64"))
GRIND_LOOKBACK = int(os.getenv("ZCT_GRIND_LOOKBACK", "6"))
# 顺势突破：近 GRIND_LOOKBACK 根净位移占价比上限（「慢磨台阶」）
GRIND_MAX_NET_MOVE_PCT = float(os.getenv("ZCT_GRIND_MAX_NET_MOVE_PCT", "0.0035"))
LEVEL_TOUCH_LOOKBACK_BARS = int(os.getenv("ZCT_LEVEL_TOUCH_LOOKBACK_BARS", "480"))  # ~8h 1m
LEVEL_FRESH_MIN_BARS = int(os.getenv("ZCT_LEVEL_FRESH_MIN_BARS", "360"))  # ~6h 未再测
LEVEL_RECYCLE_TOUCH_MIN = int(os.getenv("ZCT_LEVEL_RECYCLE_TOUCH_MIN", "3"))
try:
    _lfh = os.getenv("ZCT_LEVEL_FRESH_MIN_HOURS", "0").strip()
    LEVEL_FRESH_MIN_HOURS = float(_lfh) if _lfh else 0.0
except ValueError:
    LEVEL_FRESH_MIN_HOURS = 0.0

# PLAY03 止盈：vwap=回锚（默认）；1r=与 Koroush Reversal Lesson4 一致的 1:1 机械目标
_PLAY03_TP_RAW = os.getenv("ZCT_PLAY03_TP_MODE", "vwap").strip().lower()
PLAY03_TP_1R = _PLAY03_TP_RAW in ("1r", "one_r", "risk1")

try:
    _kms = os.getenv("ZCT_KOROUSH_MIN_STOP_DISTANCE_PCT", "0.01").strip()
    if _kms == "":
        KOROUSH_MIN_STOP_DISTANCE_PCT = 0.01
    else:
        KOROUSH_MIN_STOP_DISTANCE_PCT = float(_kms)
except ValueError:
    KOROUSH_MIN_STOP_DISTANCE_PCT = 0.01

_PSYCH_RAW = os.getenv("ZCT_PSYCH_LEVELS", "0").strip().lower()
PSYCH_LEVELS_ENABLED = _PSYCH_RAW in ("1", "true", "yes", "on")

try:
    _bma = os.getenv("ZCT_BREAKOUT_MAX_MA_CROSSES", "0").strip()
    BREAKOUT_MAX_MA_CROSSES = int(_bma) if _bma else 0
except ValueError:
    BREAKOUT_MAX_MA_CROSSES = 0

# --- ZCT S/R：近端 recycled 结构位否决顺势 PLAY01/02（默认关）---
_RECYCLED_VETO_RAW = os.getenv("ZCT_RECYCLED_NEAR_VETO_ENABLED", "0").strip().lower()
RECYCLED_NEAR_VETO_ENABLED = _RECYCLED_VETO_RAW not in (
    "0",
    "false",
    "no",
    "off",
    "disabled",
)
try:
    _rdm = os.getenv("ZCT_RECYCLED_NEAR_MAX_DIST_PCT", "0.2").strip()
    RECYCLED_NEAR_MAX_DIST_PCT = float(_rdm) if _rdm else 0.2
except ValueError:
    RECYCLED_NEAR_MAX_DIST_PCT = 0.2
if RECYCLED_NEAR_MAX_DIST_PCT <= 0:
    RECYCLED_NEAR_MAX_DIST_PCT = 0.2

# 海报：会话内 VWAP 交叉刻度 0–3 / 4–6 / 7+
VWAP_CROSS_MAX_LOW = int(os.getenv("ZCT_VWAP_CROSS_MAX_LOW", "3"))
VWAP_CROSS_MAX_MID = int(os.getenv("ZCT_VWAP_CROSS_MAX_MID", "6"))
# 海报「use level 3+」：setup_level 与 ZCT_MIN_SETUP_LEVEL 比较；默认 enforce 开、min=3
_ENFORCE_SETUP_RAW = os.getenv("ZCT_ENFORCE_SETUP_LEVEL", "1").strip().lower()
ENFORCE_SETUP_LEVEL = _ENFORCE_SETUP_RAW not in ("0", "false", "no", "off", "disabled")
try:
    MIN_SETUP_LEVEL_FOR_SIDE = int(os.getenv("ZCT_MIN_SETUP_LEVEL", "3"))
except ValueError:
    MIN_SETUP_LEVEL_FOR_SIDE = 3

# --- P1：固定风险名义 + 日损熔断（账户为纸面权益基准）---
ACCOUNT_EQUITY_USDT = float(os.getenv("ZCT_ACCOUNT_EQUITY_USDT", "10000"))
RISK_PCT_PER_TRADE = float(os.getenv("ZCT_RISK_PCT_PER_TRADE", "0.005"))
USE_RISK_SIZED_NOTIONAL = os.getenv("ZCT_USE_RISK_SIZED_NOTIONAL", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MAX_NOTIONAL_CAP_USDT = float(os.getenv("ZCT_MAX_NOTIONAL_CAP_USDT", "0") or 0)
MAX_DAILY_LOSS_PCT = float(os.getenv("ZCT_MAX_DAILY_LOSS_PCT", "0.05"))

# --- P2：平仓后冷却（毫秒，代码常量；非环境变量）+ 极端带宽跳过 ---
COOLDOWN_AFTER_LOSS_MS = 30 * 60 * 1000  # 止损后
COOLDOWN_AFTER_WIN_MS = 30 * 60 * 1000  # 止盈后
COOLDOWN_AFTER_CLOSE_MS = 0  # 任意平仓额外间隔；0=关闭
_DEFAULT_MAX_BAND_WIDTH_PCT = 15.0
MAX_BAND_WIDTH_PCT = float(
    os.getenv("ZCT_MAX_BAND_WIDTH_PCT", str(_DEFAULT_MAX_BAND_WIDTH_PCT))
    or _DEFAULT_MAX_BAND_WIDTH_PCT
)

SWING_LOOKBACK = int(os.getenv("ZCT_SWING_LOOKBACK", "20"))
MIN_SL_PCT = float(os.getenv("ZCT_MIN_SL_PCT", "0.003"))
SL_BUFFER_BPS = float(os.getenv("ZCT_SL_BUFFER_BPS", "2"))
RESOLVE_MAX_BARS = int(os.getenv("ZCT_RESOLVE_MAX_BARS", "720"))
_DEFAULT_RESOLVE_MAX_HOLD_MS = 6 * 60 * 60 * 1000  # 默认 6h 墙上时钟；与 RESOLVE_MAX_BARS 二者满足其一即 expired
_RESOLVE_HOLD_RAW = os.getenv("ZCT_RESOLVE_MAX_HOLD_MS")
try:
    if _RESOLVE_HOLD_RAW is None or str(_RESOLVE_HOLD_RAW).strip() == "":
        RESOLVE_MAX_HOLD_MS = _DEFAULT_RESOLVE_MAX_HOLD_MS
    else:
        RESOLVE_MAX_HOLD_MS = max(0, int(float(str(_RESOLVE_HOLD_RAW).strip())))
except ValueError:
    RESOLVE_MAX_HOLD_MS = _DEFAULT_RESOLVE_MAX_HOLD_MS
# 结算循环里「上一标的 → 下一标的」之间的休眠（秒），减轻 /fapi/v1/klines 频率；0=不休眠
RESOLVE_INTER_SYMBOL_SLEEP_SEC = float(
    os.getenv("ZCT_RESOLVE_INTER_SYMBOL_SLEEP_SEC", "0") or 0
)
SAME_BAR_RULE = os.getenv("ZCT_SAME_BAR_RULE", "pessimistic").strip().lower()
# 虚拟仓位：保证金 × 杠杆 = 名义敞口 USDT，用于纸面 pnl_usdt（与 s6 默认 10x 对齐）
_ZCT_MARGIN_USDT = float(os.getenv("ZCT_VIRTUAL_NOTIONAL_USDT", "100"))
ZCT_LEVERAGE = float(os.getenv("ZCT_LEVERAGE", "10"))
VIRTUAL_NOTIONAL_USDT = _ZCT_MARGIN_USDT * ZCT_LEVERAGE

# 流动性（仅 OI）：False 时不请求接口、不挡单（暂时屏蔽用）
LIQUIDITY_OI_FILTER_ENABLED = False
# 币安 U 本位 openInterestHist
LIQUIDITY_OI_PERIOD = "15m"  # 5m / 15m / 30m / 1h / 2h / 4h / 6h / 12h / 1d
# 顺势单：OI 环比 ≤ 该阈值则由 analyze_symbol 硬抑制方向单（小数；LONG 默认须为正增长）。
LIQUIDITY_OI_MIN_REL_LONG = 0.0
# 空单略放宽：破位时多头平仓可导致总 OI 小幅下降，-0.002 ≈ 允许 -0.2% 环比仍不降级。
LIQUIDITY_OI_MIN_REL_SHORT = -0.002
# OI 环比定义：见 fetch_liquidity_data（当前为「前一根 vs 前前一根」）
LIQUIDITY_OI_COMPARE_MODE = "prev_vs_prev2"

# 全量扫描入库：开放持仓遇本轮 FLAT 是否 supersede（扫描价平仓）。
# False = 仅 LONG↔SHORT 反向时平仓；True = 与旧逻辑一致，观望(FLAT)也会平仓。
SCAN_SUPERSEDE_ON_FLAT = False


def _circuit_breaker_halted() -> bool:
    """P1：当日已实现盈亏累计 ≤ -账户×MAX_DAILY_LOSS_PCT 则暂停新开方向单。"""
    if MAX_DAILY_LOSS_PCT <= 0:
        return False
    from accumulation_radar import init_db

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_iso = f"{today}T00:00:00"
    conn = init_db()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT COALESCE(SUM(pnl_usdt), 0)
            FROM {ZCT_DB_SETTLEMENTS_TABLE}
            WHERE settled_at_utc >= ?
            """,
            (start_iso,),
        )
        pnl_sum = float(cur.fetchone()[0] or 0)
        limit_neg = ACCOUNT_EQUITY_USDT * MAX_DAILY_LOSS_PCT
        return pnl_sum <= -limit_neg
    finally:
        conn.close()


def _cooldown_blocks(symbol: str) -> bool:
    """P2：该标的仍在任意冷却窗口内（止盈/止损/通用平仓间隔）。"""
    if (
        COOLDOWN_AFTER_LOSS_MS <= 0
        and COOLDOWN_AFTER_WIN_MS <= 0
        and COOLDOWN_AFTER_CLOSE_MS <= 0
    ):
        return False
    from accumulation_radar import init_db

    sym = symbol.strip().upper()
    now_ms = int(time.time() * 1000)
    conn = init_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT cooldown_until_ms FROM zct_symbol_cooldown WHERE symbol = ?",
            (sym,),
        )
        row = cur.fetchone()
        if not row:
            return False
        return int(row[0]) > now_ms
    finally:
        conn.close()


def _merge_symbol_cooldown(cur, symbol: str, until_ms: int) -> None:
    """将标的冷却截止时间设为 max(已有行, until_ms)，避免短窗口覆盖长窗口。"""
    sym = str(symbol).strip().upper()
    cur.execute(
        "SELECT cooldown_until_ms FROM zct_symbol_cooldown WHERE symbol = ?",
        (sym,),
    )
    row = cur.fetchone()
    prev = int(row[0]) if row else 0
    final = max(prev, int(until_ms))
    cur.execute(
        """
        INSERT OR REPLACE INTO zct_symbol_cooldown (symbol, cooldown_until_ms)
        VALUES (?, ?)
        """,
        (sym, final),
    )


def _apply_settlement_cooldowns(
    cur,
    *,
    symbol: str,
    outcome: str,
    pnl_usdt: float,
) -> None:
    """settlements 写入成功后更新冷却：close / win / loss 及 supersede 按盈亏映射到 win/loss。"""
    now_ms = int(time.time() * 1000)
    ends: List[int] = []
    if COOLDOWN_AFTER_CLOSE_MS > 0:
        ends.append(now_ms + COOLDOWN_AFTER_CLOSE_MS)
    if outcome == "win" and COOLDOWN_AFTER_WIN_MS > 0:
        ends.append(now_ms + COOLDOWN_AFTER_WIN_MS)
    if outcome == "loss" and COOLDOWN_AFTER_LOSS_MS > 0:
        ends.append(now_ms + COOLDOWN_AFTER_LOSS_MS)
    if outcome == "supersede":
        if pnl_usdt > 0 and COOLDOWN_AFTER_WIN_MS > 0:
            ends.append(now_ms + COOLDOWN_AFTER_WIN_MS)
        elif pnl_usdt < 0 and COOLDOWN_AFTER_LOSS_MS > 0:
            ends.append(now_ms + COOLDOWN_AFTER_LOSS_MS)
    if not ends:
        return
    _merge_symbol_cooldown(cur, symbol, max(ends))


def _paper_notional_for_signal(res: SignalResult) -> float:
    """P1：按单笔风险占权益比例反推名义（线性 USDT 本位近似）。"""
    if not USE_RISK_SIZED_NOTIONAL or res.side not in ("LONG", "SHORT"):
        return float(VIRTUAL_NOTIONAL_USDT)
    if res.sl_price is None or res.price is None or float(res.price) <= 0:
        return float(VIRTUAL_NOTIONAL_USDT)
    entry = float(res.price)
    sl = float(res.sl_price)
    risk_budget = ACCOUNT_EQUITY_USDT * RISK_PCT_PER_TRADE
    if res.side == "LONG":
        risk_frac = (entry - sl) / entry
    else:
        risk_frac = (sl - entry) / entry
    if risk_frac <= 1e-12:
        return float(VIRTUAL_NOTIONAL_USDT)
    n = risk_budget / risk_frac
    if MAX_NOTIONAL_CAP_USDT > 0:
        n = min(n, MAX_NOTIONAL_CAP_USDT)
    return float(max(n, 1.0))


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


def fetch_klines(
    symbol: str,
    interval: str,
    limit: int,
    *,
    end_time_ms: Optional[int] = None,
) -> List[List[Any]]:
    params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    data = api_get("/fapi/v1/klines", params)
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


# K 线周期时长（ms），多周期关键位 RefLevelResolver 用
ZCT_REF_BAR_DUR_MS: Dict[str, int] = {
    "1d": 86_400_000,
    "4h": 14_400_000,
    "1h": 3_600_000,
    "15m": 900_000,
}


def utc_day_floor_ms(ms: int) -> int:
    t = pd.Timestamp(int(ms), unit="ms", tz="UTC").floor("D")
    return int(t.value // 1_000_000)


def session_slice_utc_day(full_kline: pd.DataFrame, asof_open_ms: int) -> pd.DataFrame:
    """UTC 日历日 00:00 起至 asof 根（含）的 1m 子集（对齐会话 VWAP 与 walk-forward）。"""
    if full_kline.empty:
        return full_kline
    t = pd.Timestamp(int(asof_open_ms), unit="ms", tz="UTC")
    day0 = t.normalize()
    return full_kline[
        (full_kline["open_time"] <= int(asof_open_ms)) & (full_kline["ts"] >= day0)
    ].copy()


def _preload_ref_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = fetch_klines_forward(symbol, interval, start_ms, end_ms)
    return klines_to_df(rows)


class RefLevelResolver:
    """预拉多周期 K，按 ref_levels 语义在 asof 时刻给出关键位（无未来函数）。"""

    def __init__(self, symbol: str, start_ms: int, end_ms: int) -> None:
        self.symbol = str(symbol).strip().upper()
        pad = 120 * 86_400_000
        s0 = int(start_ms) - pad
        e0 = int(end_ms)
        self._d1 = _preload_ref_klines(self.symbol, "1d", s0, e0)
        self._h4 = _preload_ref_klines(self.symbol, "4h", s0, e0)
        self._h1 = _preload_ref_klines(self.symbol, "1h", s0, e0)
        self._m15 = _preload_ref_klines(self.symbol, "15m", s0, e0)

    def levels(self, asof_open_ms: int) -> Dict[str, float]:
        out: Dict[str, float] = {}
        t = int(asof_open_ms)

        def _tail_closed(df: pd.DataFrame, dur_ms: int, n: int) -> pd.DataFrame:
            if df is None or df.empty or "open_time" not in df.columns:
                return pd.DataFrame()
            ot = df["open_time"].astype("int64")
            closed = df.loc[ot + int(dur_ms) <= t]
            return closed.tail(int(n))

        d1t = _tail_closed(self._d1, ZCT_REF_BAR_DUR_MS["1d"], 3)
        if len(d1t) >= 2:
            prev = d1t.iloc[-2]
            out["pdh"] = float(prev["high"])
            out["pdl"] = float(prev["low"])
        for iv_df, dur_ms, pfx in (
            (self._h4, ZCT_REF_BAR_DUR_MS["4h"], "h4"),
            (self._h1, ZCT_REF_BAR_DUR_MS["1h"], "h1"),
            (self._m15, ZCT_REF_BAR_DUR_MS["15m"], "m15"),
        ):
            tt = _tail_closed(iv_df, dur_ms, 4)
            if len(tt) >= 2:
                prev_bar = tt.iloc[-2]
                out[f"{pfx}_high"] = float(prev_bar["high"])
                out[f"{pfx}_low"] = float(prev_bar["low"])
        return out


def compute_vwap_bands_session(df: pd.DataFrame, sigma: float) -> pd.DataFrame:
    """UTC 当日会话内累积 VWAP 与 ±sigma 加权标准差轨。

    会话内加权方差 Var = E[X^2] - E[X]^2，其中权重为各根成交量：
    sum(v·tp^2)/sum(v) - vwap^2，全程 cumsum 向量化，O(N) 而非逐根二重循环。
    """
    if df.empty:
        return df
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    v = df["volume"].values
    tpv = tp.values
    cum_pv = np.cumsum(tpv * v)
    cum_v = np.cumsum(v)
    cum_tp2v = np.cumsum(v * tpv * tpv)
    vwap = cum_pv / np.maximum(cum_v, 1e-12)
    # 加权总体方差（与旧实现 sum(v*(tp-vwap_i)^2)/cum_v 在代数上等价）
    var = cum_tp2v / np.maximum(cum_v, 1e-12) - vwap * vwap
    std = np.sqrt(np.maximum(var, 0.0))
    std = np.where(cum_v > 0, std, 0.0)
    upper = vwap + sigma * std
    lower = vwap - sigma * std
    out = df.copy()
    out["typical"] = tp
    out["vwap"] = vwap
    out["vwap_upper"] = upper
    out["vwap_lower"] = lower
    out["band_width_pct"] = np.where(
        vwap > 0, (upper - lower) / vwap * 100.0, 0.0
    )
    return out


def build_classify_inputs_at_asof(
    symbol: str,
    *,
    end_ms: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], int]:
    """以最后一根已返回 1m 的 open_time 为 asof：UTC 当日 0 点 forward 拉线 → 会话切片 → 关键位 → VWAP 轨。

    与 walk-forward 共用 `session_slice_utc_day` / `RefLevelResolver`；`classify_and_signal` 需再传
    `spike_klines_end_ms=asof` 对齐 Play03 ATR。
    """
    su = str(symbol).strip().upper()
    end_ms = int(time.time() * 1000) if end_ms is None else int(end_ms)
    day0_ms = utc_day_floor_ms(end_ms)
    rows = fetch_klines_forward(su, "1m", day0_ms, end_ms)
    df = klines_to_df(rows)
    if df.empty:
        return df, {}, end_ms
    asof_ms = int(df.iloc[-1]["open_time"])
    sdf0 = session_slice_utc_day(df, asof_ms)
    if sdf0.empty or len(sdf0) < 30:
        return pd.DataFrame(), {}, asof_ms
    pad_anchor = utc_day_floor_ms(asof_ms)
    levels = RefLevelResolver(su, pad_anchor, asof_ms).levels(asof_ms)
    sdf = compute_vwap_bands_session(sdf0, BAND_SIGMA)
    return sdf, levels, asof_ms


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


def vwap_crossover_bucket(crosses: int) -> str:
    """海报刻度：0–3 偏趋势突破；4–6 混合；7+ 偏震荡/反转语境。"""
    if crosses <= VWAP_CROSS_MAX_LOW:
        return "0-3"
    if crosses <= VWAP_CROSS_MAX_MID:
        return "4-6"
    return "7+"


def position_vs_vwap_label(
    price: float,
    vw: float,
    up: float,
    lo: float,
    touch_eps: float,
) -> str:
    """价位相对 VWAP / σ 带（海报 Signal 1）。"""
    if up > lo and price >= up - touch_eps:
        return "at_upper_band"
    if up > lo and price <= lo + touch_eps:
        return "at_lower_band"
    if price > vw:
        return "above_vwap"
    if price < vw:
        return "below_vwap"
    return "at_vwap"


def masterclass_setup_level(play: str) -> int:
    """
    海报「level 3+」：三信号与 Play01/02/03 严格模板一致。
    - 3：PLAY01_BREAKOUT / PLAY02_BREAKDOWN / PLAY03_REV（有方向的均值回归）
    - 2：偏置、过渡、观望等待贴轨
    - 1：无模板匹配
    """
    if play in (
        "PLAY01_BREAKOUT_LONG",
        "PLAY02_BREAKDOWN_SHORT",
        "PLAY03_REV_LONG",
        "PLAY03_REV_SHORT",
    ):
        return 3
    if play in (
        "PLAY01_BIAS_LONG",
        "PLAY02_BIAS_SHORT",
        "PLAY03_WATCH",
        "TRANSITION_BIAS_LONG",
        "TRANSITION_BIAS_SHORT",
    ):
        return 2
    return 1


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
    """兼容旧调用：以 df 最后一根 `open_time` 为 asof，切 UTC 当日会话（与 `session_slice_utc_day` 一致）。"""
    if df.empty:
        return df.copy()
    asof = int(df.iloc[-1]["open_time"])
    return session_slice_utc_day(df, asof)


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


def psych_levels_for_price(price: float, each_side: int = 2) -> Dict[str, float]:
    """ZCT S/R 文：大整数心理位。按数量级取步长，在现价附近各若干档。"""
    p = max(float(price), 1e-18)
    e = int(math.floor(math.log10(p)))
    step = 10.0 ** (e - 1)
    step = max(step, p * 0.002)
    step = min(step, p * 0.25)
    mid = round(p / step) * step
    out: Dict[str, float] = {}
    for i in range(-each_side, each_side + 1):
        if i == 0:
            continue
        lv = mid + float(i) * step
        if lv > 0:
            out[f"psych_{i:+d}"] = float(lv)
    return out


def _widen_sl_min_risk_long(
    entry: float,
    sl_init: float,
    sdf: pd.DataFrame,
    buf: float,
    clamp_long_sl,
) -> float:
    """Koroush：止损距进场若不足最小占价比，则扩大摆动窗取更远摆动低（多单）。"""
    need = max(MIN_SL_PCT, KOROUSH_MIN_STOP_DISTANCE_PCT) if KOROUSH_MIN_STOP_DISTANCE_PCT > 0 else MIN_SL_PCT
    lows = sdf["low"].astype(float)
    best = clamp_long_sl(sl_init)
    if entry <= 0 or (entry - best) / entry >= need - 1e-15:
        return best
    for mult in range(1, 14):
        win = min(max(SWING_LOOKBACK, 1) * mult, len(sdf), 720)
        cand_raw = float(lows.iloc[-win:].min()) * (1.0 - buf)
        cand = clamp_long_sl(cand_raw)
        if entry - cand >= entry * need - 1e-15:
            return cand
        if cand < best - 1e-15:
            best = cand
    return best


def _widen_sl_min_risk_short(
    entry: float,
    sl_init: float,
    sdf: pd.DataFrame,
    buf: float,
    clamp_short_sl,
) -> float:
    need = max(MIN_SL_PCT, KOROUSH_MIN_STOP_DISTANCE_PCT) if KOROUSH_MIN_STOP_DISTANCE_PCT > 0 else MIN_SL_PCT
    highs = sdf["high"].astype(float)
    best = clamp_short_sl(sl_init)
    if entry <= 0 or (best - entry) / entry >= need - 1e-15:
        return best
    for mult in range(1, 14):
        win = min(max(SWING_LOOKBACK, 1) * mult, len(sdf), 720)
        cand_raw = float(highs.iloc[-win:].max()) * (1.0 + buf)
        cand = clamp_short_sl(cand_raw)
        if cand - entry >= entry * need - 1e-15:
            return cand
        if cand > best + 1e-15:
            best = cand
    return best


def _vol_ma_last(sdf: pd.DataFrame, period: int) -> Tuple[float, float]:
    """当前根成交量与其简单均量。"""
    if sdf.empty or "volume" not in sdf.columns:
        return float("nan"), float("nan")
    v = sdf["volume"].astype(float)
    if len(v) < 1:
        return float("nan"), float("nan")
    last_v = float(v.iloc[-1])
    w = min(max(period, 1), len(v))
    ma = float(v.iloc[-w:].mean())
    return last_v, ma


def _wilder_atr_last_pct(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> Optional[float]:
    """最后一根 K 上的 Wilder ATR / close（占价比）。"""
    n = len(close)
    if n < period + 2 or period < 1:
        return None
    tr = np.zeros(n, dtype=float)
    tr[0] = float(high[0]) - float(low[0])
    for i in range(1, n):
        hl = float(high[i]) - float(low[i])
        hpc = abs(float(high[i]) - float(close[i - 1]))
        lpc = abs(float(low[i]) - float(close[i - 1]))
        tr[i] = max(hl, hpc, lpc)
    atr = np.full(n, np.nan, dtype=float)
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    last_atr = float(atr[-1])
    last_c = float(close[-1])
    if last_c <= 0 or not np.isfinite(last_atr) or last_atr <= 0:
        return None
    return last_atr / last_c


def _fetch_last_atr_pct(
    symbol: str, *, end_time_ms: Optional[int] = None
) -> Optional[float]:
    sym = str(symbol).strip().upper()
    if not sym:
        return None
    need = max(SPIKE_ATR_PERIOD + 5, 32)
    lim = max(SPIKE_ATR_KLINE_LIMIT, need)
    rows = fetch_klines(sym, SPIKE_ATR_INTERVAL, lim, end_time_ms=end_time_ms)
    if not rows or len(rows) < need:
        return None
    df = klines_to_df(rows)
    if df.empty or len(df) < need:
        return None
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    return _wilder_atr_last_pct(h, l, c, SPIKE_ATR_PERIOD)


def _resolve_spike_range_ratio(
    symbol: str, *, end_time_ms: Optional[int] = None
) -> Tuple[float, str]:
    """
    Play03 刺穿阈值：优先用 15m ATR%×倍数并夹在 floor/cap；否则回退 SPIKE_RANGE_RATIO。
    高波动山寨 ATR% 大 → 阈值抬高，减少「0.4% 一碰就过」的无效反转单。
    """
    if not SPIKE_USE_ATR_15M:
        return SPIKE_RANGE_RATIO, f"固定 {SPIKE_RANGE_RATIO*100:.3f}%（ZCT_SPIKE_RANGE_RATIO）"
    atrp = _fetch_last_atr_pct(symbol, end_time_ms=end_time_ms)
    if atrp is None or atrp <= 0:
        return SPIKE_RANGE_RATIO, (
            f"ATR 不可用→回退固定 {SPIKE_RANGE_RATIO*100:.3f}%（{SPIKE_ATR_INTERVAL}）"
        )
    raw = SPIKE_ATR_MULT * float(atrp)
    thr = max(SPIKE_ATR_RATIO_FLOOR, min(SPIKE_ATR_RATIO_CAP, raw))
    return thr, (
        f"动态 {SPIKE_ATR_INTERVAL} ATR%≈{atrp*100:.3f}% ×{SPIKE_ATR_MULT:g} "
        f"→ {thr*100:.3f}%（夹 {SPIKE_ATR_RATIO_FLOOR*100:.2f}%–{SPIKE_ATR_RATIO_CAP*100:.2f}%）"
    )


def _is_spike_window(sdf: pd.DataFrame, lookback: int, ratio_thr: float) -> bool:
    """近 lookback 根内是否存在「高波动刺穿」柱：max((H-L)/mid) ≥ ratio_thr。"""
    if len(sdf) < max(lookback, 2):
        return False
    tail = sdf.iloc[-lookback:]
    mid = (tail["high"].astype(float) + tail["low"].astype(float)) / 2.0
    mid = mid.replace(0, np.nan)
    rng = (tail["high"].astype(float) - tail["low"].astype(float)) / mid
    mx = float(np.nanmax(rng.values))
    return mx >= ratio_thr


def _grind_approach_ok(sdf: pd.DataFrame, lookback: int, max_net_move_pct: float) -> bool:
    """顺势突破语境：近端净位移小 → 慢磨靠近阻力区。"""
    if len(sdf) < lookback + 1:
        return True
    c0 = float(sdf["close"].iloc[-lookback - 1])
    c1 = float(sdf["close"].iloc[-1])
    if c1 <= 0:
        return True
    return abs(c1 - c0) / c1 <= max_net_move_pct


def _two_closes_strictly_above_upper(sdf: pd.DataFrame) -> bool:
    if len(sdf) < 2:
        return False
    a = sdf.iloc[-2]
    b = sdf.iloc[-1]
    return float(a["close"]) > float(a["vwap_upper"]) and float(b["close"]) > float(
        b["vwap_upper"]
    )


def _two_closes_strictly_below_lower(sdf: pd.DataFrame) -> bool:
    if len(sdf) < 2:
        return False
    a = sdf.iloc[-2]
    b = sdf.iloc[-1]
    return float(a["close"]) < float(a["vwap_lower"]) and float(b["close"]) < float(
        b["vwap_lower"]
    )


def _false_break_reclaim_short(sdf: pd.DataFrame) -> bool:
    """Play03 空：前根刺穿上轨，当前根收盘收回上轨内侧。"""
    if len(sdf) < 2:
        return False
    p = sdf.iloc[-2]
    c = sdf.iloc[-1]
    pierce = float(p["high"]) > float(p["vwap_upper"]) or float(p["close"]) > float(
        p["vwap_upper"]
    )
    inside = float(c["close"]) < float(c["vwap_upper"])
    return pierce and inside


def _false_break_reclaim_long(sdf: pd.DataFrame) -> bool:
    """Play03 多：前根刺穿下轨，当前根收盘收回下轨内侧。"""
    if len(sdf) < 2:
        return False
    p = sdf.iloc[-2]
    c = sdf.iloc[-1]
    pierce = float(p["low"]) < float(p["vwap_lower"]) or float(p["close"]) < float(
        p["vwap_lower"]
    )
    inside = float(c["close"]) > float(c["vwap_lower"])
    return pierce and inside


def _volume_ok_for_regime(regime: str, sdf: pd.DataFrame) -> bool:
    """trend: vol>vol_ma；range: vol<vol_ma 或近端量斜率为负。"""
    last_v, vma = _vol_ma_last(sdf, VOL_MA_PERIOD)
    if not np.isfinite(last_v) or not np.isfinite(vma) or vma <= 0:
        return True
    if regime == "trend":
        return last_v > vma
    if regime == "range":
        if last_v < vma:
            return True
        if len(sdf) >= 5:
            vs = sdf["volume"].astype(float).iloc[-5:].values
            if float(vs[-1]) < float(vs[0]):
                return True
        return False
    return True


def _level_freshness_row(
    sdf: pd.DataFrame, lv: float, lookback: int
) -> Dict[str, Any]:
    """关键位：窗口内触碰次数、距上一根触碰的 bar 数、fresh/recycled 标签。"""
    out: Dict[str, Any] = {
        "touch_count": 0,
        "bars_since_touch": None,
        "hours_since_touch": None,
        "freshness": "unknown",
    }
    if lv <= 0 or sdf.empty or lookback <= 0:
        return out
    tail = sdf.iloc[-min(lookback, len(sdf)) :]
    touched = (tail["low"].astype(float) <= lv) & (tail["high"].astype(float) >= lv)
    out["touch_count"] = int(touched.sum())
    hours_since: Optional[float] = None
    if len(sdf) >= 2:
        hist = sdf.iloc[:-1].iloc[-min(lookback, len(sdf) - 1) :]
        if not hist.empty:
            th = (hist["low"].astype(float) <= lv) & (
                hist["high"].astype(float) >= lv
            )
            if th.any():
                rel_idx = int(np.where(th.values)[0][-1])
                out["bars_since_touch"] = int(len(hist) - 1 - rel_idx)
                if (
                    LEVEL_FRESH_MIN_HOURS > 0
                    and "open_time" in hist.columns
                    and "open_time" in sdf.columns
                ):
                    try:
                        touch_ms = int(hist.iloc[rel_idx]["open_time"])
                        last_ms = int(sdf.iloc[-1]["open_time"])
                        hours_since = max(0.0, (last_ms - touch_ms) / 3_600_000.0)
                    except (TypeError, ValueError):
                        hours_since = None
            else:
                out["bars_since_touch"] = int(len(hist))
                hours_since = None
    out["hours_since_touch"] = (
        round(hours_since, 4) if hours_since is not None else None
    )
    bst = out["bars_since_touch"]
    tc = out["touch_count"]
    if LEVEL_FRESH_MIN_HOURS > 0:
        hs = hours_since
        hours_ok = hs is None or hs >= LEVEL_FRESH_MIN_HOURS
        fresh_ok = hours_ok and tc <= 2
    else:
        fresh_ok = bst is not None and bst >= LEVEL_FRESH_MIN_BARS and tc <= 2
    if fresh_ok:
        out["freshness"] = "fresh"
    elif tc >= LEVEL_RECYCLE_TOUCH_MIN:
        out["freshness"] = "recycled"
    else:
        out["freshness"] = "mixed"
    return out


def _nearest_structural_resistance_above(
    price: float, levels: Dict[str, float]
) -> Optional[Tuple[str, float, float]]:
    """严格高于现价的最近结构阻力 -> (key, level, dist_pct)；无则 None。不含心理位。"""
    if price <= 0 or not levels:
        return None
    best: Optional[Tuple[str, float, float]] = None
    for k in ("pdh", "h4_high", "h1_high", "m15_high"):
        raw = levels.get(k)
        if raw is None:
            continue
        try:
            lv = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(lv) or lv <= price:
            continue
        d_pct = (lv - price) / price * 100.0
        if best is None or d_pct < best[2]:
            best = (k, lv, d_pct)
    return best


def _nearest_structural_support_below(
    price: float, levels: Dict[str, float]
) -> Optional[Tuple[str, float, float]]:
    """严格低于现价的最近结构支撑 -> (key, level, dist_pct)；无则 None。不含心理位。"""
    if price <= 0 or not levels:
        return None
    best: Optional[Tuple[str, float, float]] = None
    for k in ("pdl", "h4_low", "h1_low", "m15_low"):
        raw = levels.get(k)
        if raw is None:
            continue
        try:
            lv = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(lv) or lv >= price:
            continue
        d_pct = (price - lv) / price * 100.0
        if best is None or d_pct < best[2]:
            best = (k, lv, d_pct)
    return best


def fetch_liquidity_data(symbol: str) -> Dict[str, Any]:
    """
    仅用币安免费 REST：U 本位持仓量历史，拉 **3 根**，环比取 **前一根 ÷ 前前一根 - 1**（不使用列表里最新一根）。

    数组按时间升序：[-3]=前前周期、[-2]=前一周期、[-1]=最新周期。
    好处：最新统计点刚落盘时数值有时不稳定；用滞后一期差分更稳，代价是信号慢约一个 LIQUIDITY_OI_PERIOD。

    接口：GET /futures/data/openInterestHist（limit≥3）。
    失败或数据不足时 ok=False。
    """
    sym = str(symbol).strip().upper()
    data = api_get(
        "/futures/data/openInterestHist",
        {"symbol": sym, "period": LIQUIDITY_OI_PERIOD, "limit": 3},
    )
    if not isinstance(data, list) or len(data) < 3:
        return {"ok": False}
    try:
        oi_prev_prev = float(data[-3]["sumOpenInterest"])
        oi_prev_bar = float(data[-2]["sumOpenInterest"])
        oi_latest = float(data[-1]["sumOpenInterest"])
    except (KeyError, TypeError, ValueError, IndexError):
        return {"ok": False}
    if oi_prev_prev <= 0:
        return {"ok": False}
    oi_change_pct = (oi_prev_bar - oi_prev_prev) / oi_prev_prev
    return {
        "ok": True,
        "oi_change_pct": float(oi_change_pct),
        "oi_prev": float(oi_prev_prev),
        "oi_now": float(oi_prev_bar),
        "oi_latest": float(oi_latest),
        "oi_period": LIQUIDITY_OI_PERIOD,
        "oi_compare_mode": LIQUIDITY_OI_COMPARE_MODE,
    }


def _liquidity_oi_suppresses_direction(res: "SignalResult", liq: Dict[str, Any]) -> bool:
    """顺势方向单：OI 环比 ≤ 多空阈值则抑制入库（需 liq ok 且能读到 oi_change_pct）。"""
    if res.regime != "trend" or res.side not in ("LONG", "SHORT"):
        return False
    if not liq.get("ok"):
        return False
    oi_pct = liq.get("oi_change_pct")
    if oi_pct is None:
        return False
    op = float(oi_pct)
    if res.side == "LONG":
        return op <= LIQUIDITY_OI_MIN_REL_LONG
    return op <= LIQUIDITY_OI_MIN_REL_SHORT


def _liquidity_oi_suppress_reason(res: "SignalResult", liq: Dict[str, Any]) -> str:
    op = float(liq["oi_change_pct"])
    per = liq.get("oi_period", "?")
    if res.side == "LONG":
        return (
            f"P2 流动性（OI）：多单环比 {op*100:.4f}% ≤ 阈值 {LIQUIDITY_OI_MIN_REL_LONG*100:.4f}%"
            f"（{per}），方向单已抑制"
        )
    return (
        f"P2 流动性（OI）：空单环比 {op*100:.4f}% ≤ 阈值 {LIQUIDITY_OI_MIN_REL_SHORT*100:.4f}%"
        f"（{per}），方向单已抑制"
    )


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
    # Masterclass 对齐字段
    setup_level: int = 1
    position_vs_vwap: str = ""
    vwap_cross_bucket: str = ""
    entry_bar_open_ms: Optional[int] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    r_unit: Optional[float] = None
    paper_notional_usdt: Optional[float] = None
    # Koroush Breakout：第二根确认轨上的限价回踩参考（扫描器不代为挂单）
    suggested_limit_entry: Optional[float] = None


def compute_sl_tp(r: SignalResult, sdf: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    止损 / 止盈 / 风险单位（1R 的价位距离）。
    - 均值回归 PLAY03：默认止盈锚定 VWAP；ZCT_PLAY03_TP_MODE=1r 时为与 SL 等距的 1R（Koroush Reversal Lesson4）。
    - 顺势 / 过渡偏置：1R 目标；止损在 VWAP 与近端摆动极值「错误侧」之外。
    - Koroush SL：默认 ZCT_KOROUSH_MIN_STOP_DISTANCE_PCT=1%；若止损距进场仍不足则扩大摆动窗寻更远极值；环境变量设为 0 可关闭。
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
            sl_raw = min(lo, vl) * (1 - buf)
            sl = _widen_sl_min_risk_long(entry, sl_raw, sdf, buf, clamp_long_sl)
            ru = entry - sl
            if PLAY03_TP_1R:
                tp = entry + ru
            else:
                tp = vw
            return round(sl, 8), round(tp, 8), round(ru, 8)
        sl_raw = min(vw, lo) * (1 - buf)
        sl = _widen_sl_min_risk_long(entry, sl_raw, sdf, buf, clamp_long_sl)
        ru = entry - sl
        tp = entry + ru
        return round(sl, 8), round(tp, 8), round(ru, 8)

    if r.side == "SHORT":
        if r.play == "PLAY03_REV_SHORT":
            sl_raw = max(hi, vu) * (1 + buf)
            sl = _widen_sl_min_risk_short(entry, sl_raw, sdf, buf, clamp_short_sl)
            ru = sl - entry
            if PLAY03_TP_1R:
                tp = entry - ru
            else:
                tp = vw
            return round(sl, 8), round(tp, 8), round(ru, 8)
        sl_raw = max(vw, hi) * (1 + buf)
        sl = _widen_sl_min_risk_short(entry, sl_raw, sdf, buf, clamp_short_sl)
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
    *,
    spike_klines_end_ms: Optional[int] = None,
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

    if STRICT_PA_FILTERS:
        min_b = max(6, VOL_MA_PERIOD + 1, SPIKE_LOOKBACK + 1, GRIND_LOOKBACK + 1)
        if len(sdf) < min_b:
            reasons.append(
                f"严格PA：本会话 1m 根数不足 {min_b}，未应用收盘确认/量/刺穿速度过滤"
            )
        else:
            v_ok = _volume_ok_for_regime(regime, sdf)
            is_spike = False
            spike_rr = SPIKE_RANGE_RATIO
            spike_src = f"固定 {SPIKE_RANGE_RATIO*100:.3f}%（未评 Play03）"
            if play in ("PLAY03_REV_LONG", "PLAY03_REV_SHORT"):
                spike_rr, spike_src = _resolve_spike_range_ratio(
                    symbol, end_time_ms=spike_klines_end_ms
                )
                is_spike = _is_spike_window(sdf, SPIKE_LOOKBACK, spike_rr)
                reasons.append(
                    f"严格PA：刺穿判据 max(H-L)/mid ≥ {spike_rr*100:.3f}%（{spike_src}）"
                )
            if play == "PLAY01_BREAKOUT_LONG":
                if not _two_closes_strictly_above_upper(sdf):
                    reasons.append(
                        "严格PA：突破多需连续两根 1m 收盘站上各自当根动态上轨（未满足→观望）"
                    )
                    play, side, confidence = "NO_TRADE", "FLAT", "low"
                elif not v_ok:
                    reasons.append(
                        "严格PA：顺势体制要求量能高于近期均量（vol>vol_ma，未满足→观望）"
                    )
                    play, side, confidence = "NO_TRADE", "FLAT", "low"
                elif not _grind_approach_ok(
                    sdf, GRIND_LOOKBACK, GRIND_MAX_NET_MOVE_PCT
                ):
                    reasons.append(
                        "严格PA：突破前宜慢磨靠近（近端净位移过大，未满足→观望）"
                    )
                    play, side, confidence = "NO_TRADE", "FLAT", "low"
            elif play == "PLAY02_BREAKDOWN_SHORT":
                if not _two_closes_strictly_below_lower(sdf):
                    reasons.append(
                        "严格PA：破位空需连续两根 1m 收盘站下各自当根动态下轨（未满足→观望）"
                    )
                    play, side, confidence = "NO_TRADE", "FLAT", "low"
                elif not v_ok:
                    reasons.append(
                        "严格PA：顺势体制要求量能高于近期均量（vol>vol_ma，未满足→观望）"
                    )
                    play, side, confidence = "NO_TRADE", "FLAT", "low"
                elif not _grind_approach_ok(
                    sdf, GRIND_LOOKBACK, GRIND_MAX_NET_MOVE_PCT
                ):
                    reasons.append(
                        "严格PA：破位前宜慢磨靠近（近端净位移过大，未满足→观望）"
                    )
                    play, side, confidence = "NO_TRADE", "FLAT", "low"
            elif play == "PLAY03_REV_LONG":
                if not is_spike:
                    reasons.append(
                        "严格PA：反转多要求近窗存在高波动刺穿柱（is_spike，未满足→观望）"
                    )
                    play, side, confidence = "NO_TRADE", "FLAT", "low"
                elif not _false_break_reclaim_long(sdf):
                    reasons.append(
                        "严格PA：反转多需假下破后收回下轨内侧（前根刺穿下轨、当根收盘于内侧）"
                    )
                    play, side, confidence = "NO_TRADE", "FLAT", "low"
                elif not v_ok:
                    reasons.append(
                        "严格PA：震荡体制要求量能低于均量或近端缩量（未满足→观望）"
                    )
                    play, side, confidence = "NO_TRADE", "FLAT", "low"
            elif play == "PLAY03_REV_SHORT":
                if not is_spike:
                    reasons.append(
                        "严格PA：反转空要求近窗存在高波动刺穿柱（is_spike，未满足→观望）"
                    )
                    play, side, confidence = "NO_TRADE", "FLAT", "low"
                elif not _false_break_reclaim_short(sdf):
                    reasons.append(
                        "严格PA：反转空需假上破后收回上轨内侧（前根刺穿上轨、当根收盘于内侧）"
                    )
                    play, side, confidence = "NO_TRADE", "FLAT", "low"
                elif not v_ok:
                    reasons.append(
                        "严格PA：震荡体制要求量能低于均量或近端缩量（未满足→观望）"
                    )
                    play, side, confidence = "NO_TRADE", "FLAT", "low"

    if (
        BREAKOUT_MAX_MA_CROSSES > 0
        and play in ("PLAY01_BREAKOUT_LONG", "PLAY02_BREAKDOWN_SHORT")
        and ma_x > BREAKOUT_MAX_MA_CROSSES
    ):
        reasons.append(
            f"Breakout 环境滤：近窗 MA30 与均线交叉数 {ma_x} > {BREAKOUT_MAX_MA_CROSSES} "
            f"（Koroush: minimal crossovers for momentum）"
        )
        play, side, confidence = "NO_TRADE", "FLAT", "low"

    if RECYCLED_NEAR_VETO_ENABLED and levels:
        if play == "PLAY01_BREAKOUT_LONG":
            nr = _nearest_structural_resistance_above(price, levels)
            if nr is not None:
                nk, nlv, nd = nr
                if nd <= RECYCLED_NEAR_MAX_DIST_PCT:
                    fr = _level_freshness_row(
                        sdf, float(nlv), LEVEL_TOUCH_LOOKBACK_BARS
                    )
                    if fr.get("freshness") == "recycled":
                        reasons.append(
                            f"关键位滤：上方最近结构阻力 {nk}={nlv:g} 距现价 {nd:.3f}% 已 recycled，"
                            "近端偏震荡/假突破语境，否决 PLAY01（ZCT_RECYCLED_NEAR_VETO）"
                        )
                        play, side, confidence = "NO_TRADE", "FLAT", "low"
        elif play == "PLAY02_BREAKDOWN_SHORT":
            ns = _nearest_structural_support_below(price, levels)
            if ns is not None:
                sk, sup_lv, sd_pct = ns
                if sd_pct <= RECYCLED_NEAR_MAX_DIST_PCT:
                    fr = _level_freshness_row(
                        sdf, float(sup_lv), LEVEL_TOUCH_LOOKBACK_BARS
                    )
                    if fr.get("freshness") == "recycled":
                        reasons.append(
                            f"关键位滤：下方最近结构支撑 {sk}={sup_lv:g} 距现价 {sd_pct:.3f}% 已 recycled，"
                            "近端偏震荡/假反弹语境，否决 PLAY02（ZCT_RECYCLED_NEAR_VETO）"
                        )
                        play, side, confidence = "NO_TRADE", "FLAT", "low"

    pos_lbl = position_vs_vwap_label(price, vw, up, lo, touch_eps)
    xbuck = vwap_crossover_bucket(crosses)
    setup_lvl = masterclass_setup_level(play)
    reasons.append(
        f"Masterclass 信号① 价位: {pos_lbl} · ②斜率: {'陡' if slope_steep else ('平' if slope_flat else '过渡')} · "
        f"③带宽: {'宽' if bands_wide else ('窄' if bands_tight else '过渡')}"
    )
    reasons.append(
        f"Masterclass VWAP 交叉刻度={xbuck}（海报: 0–3 偏突破/趋势 · 4–6 混合 · 7+ 偏震荡）"
    )
    reasons.append(
        f"Masterclass setup_level={setup_lvl}（海报 level 3+ = 三信号与 Play01/02/03 严格模板一致）"
    )

    levels_for_near = dict(levels)
    if PSYCH_LEVELS_ENABLED and price > 0:
        levels_for_near.update(psych_levels_for_price(price))

    near = nearest_level_distance_pct(price, levels_for_near)[:8]
    near_json: List[Dict[str, Any]] = []
    for n, lv, d in near:
        fr = _level_freshness_row(sdf, float(lv), LEVEL_TOUCH_LOOKBACK_BARS)
        near_json.append(
            {
                "level": n,
                "price": lv,
                "dist_pct": round(d, 4),
                "touch_count": fr["touch_count"],
                "bars_since_touch": fr["bars_since_touch"],
                "hours_since_touch": fr.get("hours_since_touch"),
                "freshness": fr["freshness"],
            }
        )

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
        setup_level=setup_lvl,
        position_vs_vwap=pos_lbl,
        vwap_cross_bucket=xbuck,
    )


def _btc_macro_reset_for_scan() -> None:
    """每轮扫描开始重置缓存；本轮无 BTC 或未开启过滤时由调用方决定是否调用。"""
    global _BTC_MACRO_STATE
    _BTC_MACRO_STATE = {"slope_bps": 0.0, "chop": "high"}


def check_btc_macro_permission(
    current_symbol_slope: float,
    btc_slope: float,
    btc_chop: str,
    target_side: str,
    *,
    slope_threshold: float = 3.0,
    rs_min_ratio: float = 0.5,
    long_fuse_slope_bps: float = 8.0,
) -> Tuple[bool, str]:
    """
    BTC 宏观方向过滤器（大盘红绿灯 + 强多时的 RS / 多单熔断）。

    :param current_symbol_slope: 当前标的 VWAP 斜率（bps），用于强多时的相对强弱
    :param btc_slope: 当前 BTCUSDT 的 VWAP 斜率（bps）
    :param btc_chop: 当前 BTCUSDT 的震荡档位（"high" | "mid" | "low"）
    :param target_side: 预判方向 "LONG" / "SHORT"
    :param slope_threshold: 视为 BTC 极端单边的 |斜率| 下限（bps）
    :param rs_min_ratio: BTC_STRONG_UP 时做多要求 current_symbol_slope >= btc_slope × 该比例
    :param long_fuse_slope_bps: BTC_STRONG_UP 且 btc 斜率超过该值时拒接所有山寨多单；≤0 关闭
    :return: (是否允许开仓, 拒绝原因；允许时原因为空串)
    """
    if target_side not in ("LONG", "SHORT"):
        return True, ""

    btc_strong_up = (btc_slope > slope_threshold) and (btc_chop != "high")
    btc_strong_down = (btc_slope < -slope_threshold) and (btc_chop != "high")

    if target_side == "LONG" and btc_strong_down:
        reason = (
            f"宏观红灯：大盘(BTC)处于强空头趋势 (slope={btc_slope:.1f}bps, chop={btc_chop})，"
            "严禁山寨逆势接多"
        )
        return False, reason

    if target_side == "SHORT" and btc_strong_up:
        reason = (
            f"宏观拦截：BTC正处于强多头逼空 (slope={btc_slope:.1f}bps, chop={btc_chop})，"
            "严禁山寨逆势做空（防被动带飞）"
        )
        return False, reason

    if target_side == "LONG" and btc_strong_up:
        min_req = btc_slope * rs_min_ratio
        if current_symbol_slope < min_req:
            reason = (
                f"吸血过滤：大盘强多头(BTC_slope={btc_slope:.1f}bps)，该标的动能(slope={current_symbol_slope:.1f}bps)"
                f"未达相对强弱下限({min_req:.1f}bps≈大盘×{rs_min_ratio:.0%})，极易假突破"
            )
            return False, reason
        if long_fuse_slope_bps > 0 and btc_slope > long_fuse_slope_bps:
            reason = (
                f"吸血熔断：BTC处于极端吸血拉升(slope={btc_slope:.1f}bps)，资金抽干，山寨突破胜率极低"
            )
            return False, reason

    return True, ""


def analyze_symbol(
    symbol: str,
    *,
    halt_daily_circuit: bool = False,
) -> Optional[SignalResult]:
    sdf, levels, asof_ms = build_classify_inputs_at_asof(symbol)
    if sdf.empty or len(sdf) < 30:
        return None
    liq = (
        fetch_liquidity_data(symbol)
        if LIQUIDITY_OI_FILTER_ENABLED
        else {"ok": False, "disabled": True}
    )
    res = classify_and_signal(symbol, sdf, levels, spike_klines_end_ms=asof_ms)
    if LIQUIDITY_OI_FILTER_ENABLED and _liquidity_oi_suppresses_direction(res, liq):
        res = replace(
            res,
            side="FLAT",
            play="NO_TRADE",
            confidence="low",
            reasons=res.reasons + [_liquidity_oi_suppress_reason(res, liq)],
            sl_price=None,
            tp_price=None,
            r_unit=None,
            entry_bar_open_ms=None,
            paper_notional_usdt=None,
        )
    entry_ms = int(sdf.iloc[-1]["open_time"])
    sl, tp, ru = compute_sl_tp(res, sdf)
    if res.side in ("LONG", "SHORT"):
        res = replace(
            res,
            entry_bar_open_ms=entry_ms,
            sl_price=sl,
            tp_price=tp,
            r_unit=ru,
        )
    else:
        res = replace(
            res,
            entry_bar_open_ms=None,
            sl_price=None,
            tp_price=None,
            r_unit=None,
        )
    if (
        ENFORCE_SETUP_LEVEL
        and res.side in ("LONG", "SHORT")
        and res.setup_level < MIN_SETUP_LEVEL_FOR_SIDE
    ):
        res = replace(
            res,
            side="FLAT",
            play="NO_TRADE",
            confidence="low",
            reasons=res.reasons
            + [
                f"已应用 ZCT_ENFORCE_SETUP_LEVEL：setup_level={res.setup_level} < {MIN_SETUP_LEVEL_FOR_SIDE}"
                f"{'（海报 level 3+ 档）' if MIN_SETUP_LEVEL_FOR_SIDE >= 3 else ''}，方向单已抑制",
            ],
            sl_price=None,
            tp_price=None,
            r_unit=None,
            entry_bar_open_ms=None,
            paper_notional_usdt=None,
        )
    if (
        res.side in ("LONG", "SHORT")
        and MAX_BAND_WIDTH_PCT > 0
        and res.band_width_pct > MAX_BAND_WIDTH_PCT
    ):
        res = replace(
            res,
            side="FLAT",
            play="NO_TRADE",
            confidence="low",
            reasons=res.reasons
            + [
                f"P2 波动过滤：band_width_pct={res.band_width_pct:.4f} > MAX_BAND_WIDTH_PCT={MAX_BAND_WIDTH_PCT}",
            ],
            sl_price=None,
            tp_price=None,
            r_unit=None,
            entry_bar_open_ms=None,
            paper_notional_usdt=None,
        )
    if res.side in ("LONG", "SHORT") and _cooldown_blocks(symbol):
        res = replace(
            res,
            side="FLAT",
            play="NO_TRADE",
            confidence="low",
            reasons=res.reasons
            + ["P2 止损冷却：该标的仍在冷却窗口内，跳过新开方向单"],
            sl_price=None,
            tp_price=None,
            r_unit=None,
            entry_bar_open_ms=None,
            paper_notional_usdt=None,
        )
    if halt_daily_circuit and res.side in ("LONG", "SHORT"):
        res = replace(
            res,
            side="FLAT",
            play="NO_TRADE",
            confidence="low",
            reasons=res.reasons
            + [
                f"P1 日损熔断：当日已实现盈亏已达 -{MAX_DAILY_LOSS_PCT:.1%}×权益 上限，暂停新开仓",
            ],
            sl_price=None,
            tp_price=None,
            r_unit=None,
            entry_bar_open_ms=None,
            paper_notional_usdt=None,
        )
    if BTC_MACRO_FILTER_ENABLED:
        if symbol == "BTCUSDT":
            _BTC_MACRO_STATE["slope_bps"] = float(res.slope_bps or 0.0)
            _BTC_MACRO_STATE["chop"] = str(res.chop_score or "high") or "high"
        elif res.side in ("LONG", "SHORT"):
            ok, mreason = check_btc_macro_permission(
                float(res.slope_bps or 0.0),
                float(_BTC_MACRO_STATE["slope_bps"]),
                str(_BTC_MACRO_STATE.get("chop") or "high"),
                res.side,
                slope_threshold=BTC_MACRO_SLOPE_THRESHOLD_BPS,
                rs_min_ratio=BTC_MACRO_RS_MIN_RATIO,
                long_fuse_slope_bps=BTC_MACRO_LONG_FUSE_SLOPE_BPS,
            )
            if not ok:
                res = replace(
                    res,
                    side="FLAT",
                    play="NO_TRADE",
                    confidence="low",
                    reasons=res.reasons + [mreason],
                    sl_price=None,
                    tp_price=None,
                    r_unit=None,
                    entry_bar_open_ms=None,
                    paper_notional_usdt=None,
                    suggested_limit_entry=None,
                )
    if res.side in ("LONG", "SHORT"):
        lim_hint: Optional[float] = None
        if res.play == "PLAY01_BREAKOUT_LONG" and len(sdf) >= 2:
            lim_hint = float(sdf.iloc[-2]["vwap_upper"])
        elif res.play == "PLAY02_BREAKDOWN_SHORT" and len(sdf) >= 2:
            lim_hint = float(sdf.iloc[-2]["vwap_lower"])
        new_reasons = list(res.reasons)
        if lim_hint is not None:
            new_reasons.append(
                f"执行提示（Koroush Breakout）：第二根确认后可于参考轨挂限价回踩 ≈ {lim_hint:g}"
            )
        res = replace(
            res,
            paper_notional_usdt=_paper_notional_for_signal(res),
            suggested_limit_entry=lim_hint,
            reasons=new_reasons,
        )
    else:
        res = replace(res, paper_notional_usdt=None, suggested_limit_entry=None)
    return res


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


def _is_open_hold_row(r: SignalResult) -> bool:
    """与 zct_vwap_api._display_status「持仓中」一致：方向单且已算止损价。"""
    return r.side in ("LONG", "SHORT") and r.sl_price is not None


def _fetch_symbols_with_open_positions(cur) -> Set[str]:
    """已平仓(outcome 非空)之前，同一标的不再新开方向单。"""
    cur.execute(
        f"""
        SELECT DISTINCT symbol FROM {ZCT_DB_SIGNALS_TABLE}
        WHERE outcome IS NULL
          AND sl_price IS NOT NULL
          AND side IN ('LONG', 'SHORT')
        """
    )
    return {str(row[0]) for row in cur.fetchall() if row and row[0]}


def _scan_supersedes_open_hold(db_side: str, r: SignalResult) -> bool:
    """是否应用扫描价 supersede：反向一定触发；FLAT 仅当 SCAN_SUPERSEDE_ON_FLAT 为 True。"""
    if r.side in ("LONG", "SHORT") and db_side in ("LONG", "SHORT"):
        return r.side != db_side
    if r.side == "FLAT":
        return SCAN_SUPERSEDE_ON_FLAT
    return False


def _settle_open_for_scan_supersede(
    cur,
    *,
    settled_at_utc: str,
    sid: int,
    sym: str,
    side: str,
    play: Optional[str],
    entry: float,
    sl: float,
    tp: Optional[float],
    notion: float,
    exit_px: float,
) -> None:
    """
    信号翻转 / 转观望：按本轮扫描价平仓，写入 settlements（与 resolve 同源盈亏公式）。
    不单独 UPDATE signals，随后 upsert 会覆盖为本轮快照。
    """
    tp_f = float(tp) if tp is not None else float(sl)
    en = float(entry)
    sx = float(exit_px)
    pnl = _pnl_r(side, en, sx, float(sl), tp_f)
    pnl_u = _pnl_usdt(side, en, sx, float(notion))
    outcome = "supersede"
    cur.execute(
        f"""
        INSERT INTO {ZCT_DB_SETTLEMENTS_TABLE} (
            settled_at_utc, signal_id, symbol, side, play, outcome,
            entry_price, exit_price, pnl_r, pnl_usdt, virtual_notional_usdt
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            settled_at_utc,
            sid,
            sym,
            side,
            play,
            outcome,
            en,
            sx,
            round(pnl, 6),
            round(pnl_u, 4),
            float(notion),
        ),
    )
    _apply_settlement_cooldowns(
        cur, symbol=str(sym).upper(), outcome="supersede", pnl_usdt=float(pnl_u)
    )
    print(
        f"[db] scan_supersede settle id={sid} {sym} {side} @ exit={sx:g} outcome={outcome} "
        f"pnl_u={round(pnl_u, 4)} (signal flip / FLAT)"
    )


def _persist_results_db(
    recorded_at_utc: str,
    rows: List[SignalResult],
    scan_params: Dict[str, Any],
) -> Tuple[int, str]:
    """UPSERT：每 symbol 一行当前快照；返回 (写入条数, db 路径)。"""
    from accumulation_radar import DB_PATH, init_db

    conn = init_db()
    try:
        cur = conn.cursor()
        open_syms = _fetch_symbols_with_open_positions(cur)
        params_json = json.dumps(scan_params, ensure_ascii=False)
        written = 0
        skipped_open = 0
        sig_tbl = ZCT_DB_SIGNALS_TABLE
        upsert = f"""
            INSERT INTO {sig_tbl} (
                recorded_at_utc, symbol, play, side, confidence, regime,
                entry_price, entry_bar_open_ms, sl_price, tp_price, r_unit,
                virtual_notional_usdt,
                vwap, vwap_upper, vwap_lower,
                slope_bps, band_width_pct, vwap_crosses, ma_crosses, chop_score,
                bands_wide, bands_tight, slope_steep, slope_flat,
                ref_levels_json, nearest_levels_json, reasons_json, scan_params_json,
                setup_level, vwap_cross_bucket, position_vs_vwap,
                outcome, outcome_at_utc, exit_price, pnl_r, pnl_usdt
            ) VALUES (
                ?,?,?,?,?,?,
                ?,?,?,?,?,
                ?,
                ?,?,?,
                ?,?,?,?,?,
                ?,?,?,?,
                ?,?,?,?,
                ?,?,?,
                NULL, NULL, NULL, NULL, NULL
            )
            ON CONFLICT(symbol) DO UPDATE SET
                recorded_at_utc = excluded.recorded_at_utc,
                play = excluded.play,
                side = excluded.side,
                confidence = excluded.confidence,
                regime = excluded.regime,
                entry_price = excluded.entry_price,
                entry_bar_open_ms = excluded.entry_bar_open_ms,
                sl_price = excluded.sl_price,
                tp_price = excluded.tp_price,
                r_unit = excluded.r_unit,
                virtual_notional_usdt = excluded.virtual_notional_usdt,
                vwap = excluded.vwap,
                vwap_upper = excluded.vwap_upper,
                vwap_lower = excluded.vwap_lower,
                slope_bps = excluded.slope_bps,
                band_width_pct = excluded.band_width_pct,
                vwap_crosses = excluded.vwap_crosses,
                ma_crosses = excluded.ma_crosses,
                chop_score = excluded.chop_score,
                bands_wide = excluded.bands_wide,
                bands_tight = excluded.bands_tight,
                slope_steep = excluded.slope_steep,
                slope_flat = excluded.slope_flat,
                ref_levels_json = excluded.ref_levels_json,
                nearest_levels_json = excluded.nearest_levels_json,
                reasons_json = excluded.reasons_json,
                scan_params_json = excluded.scan_params_json,
                setup_level = excluded.setup_level,
                vwap_cross_bucket = excluded.vwap_cross_bucket,
                position_vs_vwap = excluded.position_vs_vwap,
                outcome = excluded.outcome,
                outcome_at_utc = excluded.outcome_at_utc,
                exit_price = excluded.exit_price,
                pnl_r = excluded.pnl_r,
                pnl_usdt = excluded.pnl_usdt,
                manual_entry_price = {sig_tbl}.manual_entry_price,
                manual_exit_price = {sig_tbl}.manual_exit_price,
                manual_notes = {sig_tbl}.manual_notes,
                notes = {sig_tbl}.notes
        """
        for r in rows:
            # 必须先于 FLAT 删除：否则未平仓行会被 DB_SKIP_FLAT 删掉，或被 FLAT upsert 清空 sl/tp，resolve 永远选不中
            if r.symbol in open_syms:
                cur.execute(
                    f"""
                    SELECT id, symbol, side, play, entry_price, sl_price, tp_price,
                           COALESCE(virtual_notional_usdt, ?)
                    FROM {ZCT_DB_SIGNALS_TABLE}
                    WHERE symbol = ? AND outcome IS NULL
                      AND sl_price IS NOT NULL AND side IN ('LONG','SHORT')
                    """,
                    (VIRTUAL_NOTIONAL_USDT, r.symbol),
                )
                hold = cur.fetchone()
                if not hold:
                    open_syms.discard(r.symbol)
                else:
                    db_side = str(hold[2])
                    if _scan_supersedes_open_hold(db_side, r):
                        _settle_open_for_scan_supersede(
                            cur,
                            settled_at_utc=recorded_at_utc,
                            sid=int(hold[0]),
                            sym=str(hold[1]),
                            side=str(hold[2]),
                            play=hold[3],
                            entry=float(hold[4]),
                            sl=float(hold[5]),
                            tp=hold[6],
                            notion=float(hold[7]),
                            exit_px=float(r.price),
                        )
                        open_syms.discard(r.symbol)
                    else:
                        skipped_open += 1
                        print(
                            f"[db] skip {r.symbol}: 已有未平仓记录（持仓中），保留该行（不覆盖、不删除）"
                        )
                        continue
            if DB_SKIP_FLAT and r.side == "FLAT":
                cur.execute(
                    f"DELETE FROM {ZCT_DB_SIGNALS_TABLE} WHERE symbol = ?",
                    (r.symbol,),
                )
                continue
            cur.execute(
                upsert,
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
                    (
                        r.paper_notional_usdt
                        if r.paper_notional_usdt is not None
                        else VIRTUAL_NOTIONAL_USDT
                    ),
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
                    r.setup_level,
                    r.vwap_cross_bucket,
                    r.position_vs_vwap,
                ),
            )
            written += 1
            if _is_open_hold_row(r):
                open_syms.add(r.symbol)
        conn.commit()
        if skipped_open:
            print(f"[db] skipped_open_hold={skipped_open}")
        return written, str(DB_PATH)
    finally:
        conn.close()


def format_result(r: SignalResult) -> str:
    lines = [
        f"*{r.symbol}*  `{r.play}`  side={r.side}  conf={r.confidence}",
        f"Masterclass setup_level={r.setup_level}  pos={r.position_vs_vwap}  VWAP交叉刻度={r.vwap_cross_bucket}",
        f"price={r.price}  VWAP={r.vwap}  σ[{r.vwap_lower}, {r.vwap_upper}]  slope={r.slope_bps} bps  bandW%={r.band_width_pct}",
        f"regime={r.regime}  VWAP_x={r.vwap_crosses}  MA30_x={r.ma_crosses}  chop={r.chop_score}",
    ]
    if r.sl_price is not None and r.tp_price is not None:
        tp_note = " (1R)"
        if r.play in ("PLAY03_REV_LONG", "PLAY03_REV_SHORT"):
            tp_note = " (MR→1R)" if PLAY03_TP_1R else " (MR→VWAP)"
        lines.append(
            f"SL={r.sl_price}  TP={r.tp_price}  R={r.r_unit}{tp_note}"
        )
    if r.suggested_limit_entry is not None:
        lines.append(
            f"suggested_limit_entry≈{r.suggested_limit_entry:g}（Breakout 回踩参考，非扫描成交价）"
        )
    if r.side in ("LONG", "SHORT"):
        eff_n = (
            r.paper_notional_usdt
            if r.paper_notional_usdt is not None
            else VIRTUAL_NOTIONAL_USDT
        )
        lines.append(
            f"paper notional={eff_n:g} USDT"
            + (
                f"（固定风险 {RISK_PCT_PER_TRADE:.2%}×权益）"
                if USE_RISK_SIZED_NOTIONAL
                else f"（保证金 {_ZCT_MARGIN_USDT:g} × {ZCT_LEVERAGE:g}x）"
            )
        )
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

    当前实现：每个待结算标的各请求一次 REST /fapi/v1/klines；间隔休眠见 RESOLVE_INTER_SYMBOL_SLEEP_SEC。
    过期：优先按墙上时钟（从 entry_bar_open_ms 起 RESOLVE_MAX_HOLD_MS），再辅以根数上限 RESOLVE_MAX_BARS。
    若将来需要更高频判定且权重吃紧，可在进程内维护 K 线 WebSocket，REST 仅作补偿。
    """
    from accumulation_radar import DB_PATH, init_db

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stats: Dict[str, Any] = {"checked": 0, "resolved": 0, "skipped": 0, "skip_detail": []}
    resolved_events: List[Dict[str, Any]] = []
    conn = init_db()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id, symbol, side, play, entry_price, sl_price, tp_price, entry_bar_open_ms,
                   COALESCE(virtual_notional_usdt, ?) AS notion
            FROM {ZCT_DB_SIGNALS_TABLE}
            WHERE outcome IS NULL
              AND sl_price IS NOT NULL AND tp_price IS NOT NULL
              AND side IN ('LONG','SHORT')
            ORDER BY id ASC
            """,
            (VIRTUAL_NOTIONAL_USDT,),
        )
        rows = cur.fetchall()
        end_ms = int(time.time() * 1000)
        if rows and RESOLVE_INTER_SYMBOL_SLEEP_SEC > 0:
            print(
                f"[resolve] inter-symbol sleep={RESOLVE_INTER_SYMBOL_SLEEP_SEC:g}s "
                f"({len(rows)} open row(s))"
            )
        for idx, row in enumerate(rows):
            if idx > 0 and RESOLVE_INTER_SYMBOL_SLEEP_SEC > 0:
                time.sleep(RESOLVE_INTER_SYMBOL_SLEEP_SEC)
            stats["checked"] += 1
            sid, sym, side, play, entry, sl, tp, bar_open, notion = row
            if bar_open is None:
                stats["skipped"] += 1
                stats["skip_detail"].append(
                    {"id": sid, "symbol": sym, "reason": "entry_bar_open_ms_null"}
                )
                continue
            start_ms = int(bar_open) + 60_000
            if start_ms > end_ms:
                stats["skipped"] += 1
                stats["skip_detail"].append(
                    {"id": sid, "symbol": sym, "reason": "start_ms_after_now"}
                )
                continue
            kl = fetch_klines_forward(sym, "1m", start_ms, end_ms)
            if not kl:
                stats["skipped"] += 1
                stats["skip_detail"].append(
                    {"id": sid, "symbol": sym, "reason": "empty_klines"}
                )
                continue
            outcome: Optional[str] = None
            exit_px: float = entry
            note = "resolved:auto"
            bars_seen = 0
            deadline_ms = (
                int(bar_open) + RESOLVE_MAX_HOLD_MS if RESOLVE_MAX_HOLD_MS > 0 else None
            )
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
                bo = int(k[0])
                if deadline_ms is not None and bo >= deadline_ms:
                    outcome = "expired"
                    exit_px = c
                    note = "resolved:auto_expired_wall_clock"
                    break
                if bars_seen >= RESOLVE_MAX_BARS:
                    outcome = "expired"
                    exit_px = c
                    note = f"resolved:auto_expired_after_{RESOLVE_MAX_BARS}_bars"
                    break
            if outcome is None:
                stats["skipped"] += 1
                hi_max = None
                lo_min = None
                for k in kl:
                    if int(k[0]) < start_ms:
                        continue
                    hi_max = float(k[2]) if hi_max is None else max(hi_max, float(k[2]))
                    lo_min = float(k[3]) if lo_min is None else min(lo_min, float(k[3]))
                detail: Dict[str, Any] = {
                    "id": sid,
                    "symbol": sym,
                    "reason": "no_sl_tp_touch_yet",
                    "bars_from_entry": bars_seen,
                    "sl": sl,
                    "tp": tp,
                    "max_high_in_window": hi_max,
                    "min_low_in_window": lo_min,
                }
                if side == "LONG" and hi_max is not None:
                    detail["tp_gap"] = float(tp) - hi_max
                if side == "SHORT" and lo_min is not None:
                    detail["tp_gap"] = lo_min - float(tp)
                stats["skip_detail"].append(detail)
                continue
            pnl = _pnl_r(side, entry, exit_px, sl, tp)
            pnl_u = _pnl_usdt(side, entry, exit_px, float(notion))
            cur.execute(
                f"""
                UPDATE {ZCT_DB_SIGNALS_TABLE}
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
                cur.execute(
                    f"""
                    INSERT INTO {ZCT_DB_SETTLEMENTS_TABLE} (
                        settled_at_utc, signal_id, symbol, side, play, outcome,
                        entry_price, exit_price, pnl_r, pnl_usdt, virtual_notional_usdt
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        now_utc,
                        sid,
                        sym,
                        side,
                        play,
                        outcome,
                        entry,
                        exit_px,
                        round(pnl, 6),
                        round(pnl_u, 4),
                        float(notion),
                    ),
                )
                _apply_settlement_cooldowns(
                    cur,
                    symbol=str(sym).upper(),
                    outcome=str(outcome),
                    pnl_usdt=float(pnl_u),
                )
        conn.commit()
        lane_tag = "[touch_pool] " if _touch_pool_universe_enabled() else ""
        print(
            f"[resolve]{lane_tag}checked={stats['checked']} resolved={stats['resolved']} "
            f"skipped={stats['skipped']} db={DB_PATH} table={ZCT_DB_SIGNALS_TABLE}"
        )
        for d in stats.get("skip_detail") or []:
            if d.get("reason") == "no_sl_tp_touch_yet":
                print(
                    f"[resolve] skip id={d.get('id')} {d.get('symbol')}: "
                    f"bars={d.get('bars_from_entry')} max_high={d.get('max_high_in_window')} "
                    f"tp={d.get('tp')} min_low={d.get('min_low')} sl={d.get('sl')} tp_gap={d.get('tp_gap')}"
                )
            else:
                print(f"[resolve] skip {d}")
        stats["resolved_events"] = resolved_events
        return stats
    finally:
        conn.close()


def _merge_resolve_stats(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """合并两轮 resolve（先结算旧仓 → 入库 → 再结算本轮新开），用于日志与 TG。"""
    ev_a = list(a.get("resolved_events") or [])
    ev_b = list(b.get("resolved_events") or [])
    sd_a = list(a.get("skip_detail") or [])
    sd_b = list(b.get("skip_detail") or [])
    return {
        "checked": int(a.get("checked", 0)) + int(b.get("checked", 0)),
        "resolved": int(a.get("resolved", 0)) + int(b.get("resolved", 0)),
        "skipped": int(a.get("skipped", 0)) + int(b.get("skipped", 0)),
        "resolved_events": ev_a + ev_b,
        "skip_detail": sd_a + sd_b,
    }


def _tg_push_summary_text(
    ts: str,
    syms: List[str],
    per_symbol_lines: List[str],
    n_actionable: int,
) -> str:
    """每轮一条精简结论（每个标的一行），便于定时任务必达推送。"""
    lines: List[str] = [
        f"📊 {_scan_title_short()} 扫描结论  {ts} UTC",
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
        f"{_scan_title_short()} 信号扫描 {ts} UTC",
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
    lines = [
        "📌 ZCT 平仓结算",
        "",
    ]
    for e in events:
        lines.append(
            f"#{e['id']} {e['symbol']} {e['side']} → {e['outcome']} | "
            f"exit={e['exit_price']} | R={e['pnl_r']} | {e['pnl_usdt']} U"
        )
    return "\n".join(lines)


def run_scan(use_tg: bool = True, *, do_resolve: bool = True) -> Dict[str, Any]:
    syms = _symbols_from_env()
    if BTC_MACRO_FILTER_ENABLED:
        _btc_macro_reset_for_scan()
        if "BTCUSDT" in syms:
            syms = ["BTCUSDT"] + [s for s in syms if s != "BTCUSDT"]
    halt_day = _circuit_breaker_halted()
    if halt_day:
        print(
            f"[risk] P1 日损熔断开启：当日 settlements 累计已达 ≤-{MAX_DAILY_LOSS_PCT:.0%}×"
            f"{ACCOUNT_EQUITY_USDT:g} USDT，本轮跳过新开方向单"
        )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results: List[Dict[str, Any]] = []
    result_objs: List[SignalResult] = []
    tg_summary_lines: List[str] = []
    text_blocks: List[str] = [
        f"{_scan_title_short()} 信号扫描 `{ts}` UTC\n标的: {', '.join(syms) if syms else '(无)'}"
    ]

    for sym in syms:
        try:
            res = analyze_symbol(sym, halt_daily_circuit=halt_day)
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
        "lane": "touch_pool" if _touch_pool_universe_enabled() else "vwap_default",
        "signals_table": ZCT_DB_SIGNALS_TABLE,
        "settlements_table": ZCT_DB_SETTLEMENTS_TABLE,
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
        "resolve_max_hold_ms": RESOLVE_MAX_HOLD_MS,
        "same_bar_rule": SAME_BAR_RULE,
        "zct_margin_usdt": _ZCT_MARGIN_USDT,
        "zct_leverage": ZCT_LEVERAGE,
        "virtual_notional_usdt": VIRTUAL_NOTIONAL_USDT,
        "enforce_setup_level": ENFORCE_SETUP_LEVEL,
        "min_setup_level": MIN_SETUP_LEVEL_FOR_SIDE,
        "vwap_cross_bucket_breakpoints": [VWAP_CROSS_MAX_LOW, VWAP_CROSS_MAX_MID],
        "account_equity_usdt": ACCOUNT_EQUITY_USDT,
        "risk_pct_per_trade": RISK_PCT_PER_TRADE,
        "use_risk_sized_notional": USE_RISK_SIZED_NOTIONAL,
        "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
        "max_band_width_pct": MAX_BAND_WIDTH_PCT,
        "cooldown_after_loss_ms": COOLDOWN_AFTER_LOSS_MS,
        "cooldown_after_win_ms": COOLDOWN_AFTER_WIN_MS,
        "cooldown_after_close_ms": COOLDOWN_AFTER_CLOSE_MS,
        "max_notional_cap_usdt": MAX_NOTIONAL_CAP_USDT,
        "liquidity_oi_filter_enabled": LIQUIDITY_OI_FILTER_ENABLED,
        "scan_supersede_on_flat": SCAN_SUPERSEDE_ON_FLAT,
        "liquidity_oi_period": LIQUIDITY_OI_PERIOD,
        "liquidity_oi_compare_mode": LIQUIDITY_OI_COMPARE_MODE,
        "liquidity_oi_min_rel_long": LIQUIDITY_OI_MIN_REL_LONG,
        "liquidity_oi_min_rel_short": LIQUIDITY_OI_MIN_REL_SHORT,
        "strict_pa_filters": STRICT_PA_FILTERS,
        "spike_use_atr_15m": SPIKE_USE_ATR_15M,
        "spike_atr_interval": SPIKE_ATR_INTERVAL,
        "spike_atr_period": SPIKE_ATR_PERIOD,
        "spike_atr_mult": SPIKE_ATR_MULT,
        "spike_atr_ratio_floor": SPIKE_ATR_RATIO_FLOOR,
        "spike_atr_ratio_cap": SPIKE_ATR_RATIO_CAP,
        "spike_range_ratio_fallback": SPIKE_RANGE_RATIO,
        "spike_lookback_1m": SPIKE_LOOKBACK,
        "level_fresh_min_hours": LEVEL_FRESH_MIN_HOURS,
        "play03_tp_1r": PLAY03_TP_1R,
        "koroush_min_stop_distance_pct": KOROUSH_MIN_STOP_DISTANCE_PCT,
        "psych_levels_enabled": PSYCH_LEVELS_ENABLED,
        "breakout_max_ma_crosses": BREAKOUT_MAX_MA_CROSSES,
        "recycled_near_veto_enabled": RECYCLED_NEAR_VETO_ENABLED,
        "recycled_near_max_dist_pct": RECYCLED_NEAR_MAX_DIST_PCT,
        "btc_macro_filter_enabled": BTC_MACRO_FILTER_ENABLED,
        "btc_macro_slope_threshold_bps": BTC_MACRO_SLOPE_THRESHOLD_BPS,
        "btc_macro_rs_min_ratio": BTC_MACRO_RS_MIN_RATIO,
        "btc_macro_long_fuse_slope_bps": BTC_MACRO_LONG_FUSE_SLOPE_BPS,
    }
    payload = {
        "generated_at_utc": ts,
        "symbols": syms,
        "params": scan_params,
        "results": results,
    }

    # 定时任务语义：先纸面结算上一轮遗留，再写入本轮快照；若本轮有新快照再跑一遍 resolve，
    # 以免「仅 resolve→persist」时本轮新开仓要等到下一趟才判 SL/TP。
    resolve_stats: Dict[str, Any] = {}
    if do_resolve:
        try:
            print("[resolve] pass=pre_persist (先结算再扫描入库)")
            resolve_stats = resolve_open_signals_from_db()
        except Exception as e:
            print(f"[resolve] failed: {e}")

    if result_objs:
        try:
            n, dbp = _persist_results_db(ts, result_objs, scan_params)
            print(f"[db] {ZCT_DB_SIGNALS_TABLE} upserted={n} → {dbp}")
        except Exception as e:
            print(f"[db] persist failed: {e}")

    if do_resolve and result_objs:
        try:
            print("[resolve] pass=post_persist (本轮入库后再判触轨)")
            rs2 = resolve_open_signals_from_db()
            resolve_stats = _merge_resolve_stats(resolve_stats, rs2)
        except Exception as e:
            print(f"[resolve] post_persist failed: {e}")

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
    ap.add_argument(
        "--touch-pool",
        action="store_true",
        help="标的仅从 zct_vwap_touch_pool（触轨资产库）读取；与 main 定时子进程一致",
    )
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

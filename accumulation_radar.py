#!/usr/bin/env python3
"""
庄家收筹雷达 v1 — 发现庄家横盘吸筹 + OI异动

核心逻辑（Patrick教的）：
1. 庄家拉盘前必须先收筹 → 长期横盘+低量 = 收筹中
2. OI暴涨 = 大资金进场建仓 = 即将拉盘
3. 两个信号叠加 = 最强信号

两个模块：
A. 横盘收筹标的池（每天扫一次）→ 找正在被庄家收筹的币
B. OI异动监控（每小时扫）→ 标的池内的币有OI异动立即报警

数据源：币安合约API（免费公开，零成本）
"""

import json
import os
import sys
import time
import requests
import sqlite3
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from datetime import date as date_cls, datetime, timezone, timedelta
from pathlib import Path

# 兼容 Windows 控制台编码，避免 emoji 打印导致崩溃，并启用行缓冲便于看进度
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

# === 加载 .env ===
env_file = Path(__file__).parent / ".env.oi"
if env_file.exists():
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# === 配置 ===
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
# Telegram：暂时屏蔽原有长文推送（pool 日报 / 每小时 OI 雷达报告）；仅推送下方 BPC continuation，
# 且标的仅限「值得关注七类看板 worth_watch_* ∪ 重点关注 focus_watch」（不含收筹池 watchlist）。
TELEGRAM_SEND_LEGACY_POOL_SCAN_REPORT = False
TELEGRAM_SEND_LEGACY_OI_HOURLY_REPORT = False
# 1H BPC（突破—回踩—延续）原由 breakout_pullback_fsm 实现；模块已移除，热表不再重算 bpc_json。
FAPI = "https://fapi.binance.com"
db_dir = os.getenv("DATA_DIR", Path(__file__).parent)
DB_PATH = Path(db_dir) / "accumulation.db"
OI_RADAR_SNAPSHOT_PATH = Path(db_dir) / "oi_radar_snapshot.json"
HEAT_ACCUM_RETENTION_DAYS = 2  # 含今天在内共 2 个日历日
AMBUSH_WATCH_RETENTION_DAYS = 2  # 暗流 / 低市值埋伏看盘，与热度收筹一致
# 每轮 OI 雷达：暗流 / 低市值+OI 每类仅取埋伏榜（total 降序）中命中条件的前 N 名
AMBUSH_WATCH_TOP_N = 2
# 「📍 Patrick核心」：收筹池内 + |6h OI| 达标；与热度无关，按 |OI| 强度排序
PATRICK_CORE_OI_MIN_ABS_PCT = 3.0
PATRICK_CORE_RETENTION_DAYS = 2  # patrick_core_watch 表；含今天在内共 2 个日历日
# 收筹池 watchlist：每日 pool（默认 10:00 CST）UPSERT；落选行不更新 added_date。
# 仅保留 added_date 日期落在最近 N 个自然日内（含当日）；修剪仅在 pool 落库路径执行。默认 14 天。
try:
    WATCHLIST_RETENTION_DAYS = max(1, int(os.getenv("WATCHLIST_RETENTION_DAYS", "14").strip() or "14"))
except Exception:
    WATCHLIST_RETENTION_DAYS = 14
# 「值得关注」七类：动态入选（分数门槛 + 每类上限 + 并列带宽）；无人达标时回退为至少 FALLBACK_MIN_K 名
WORTH_CATEGORY_MAX_K = 5
WORTH_CATEGORY_FALLBACK_MIN_K = 2
WORTH_CATEGORY_TIE_EPS = 4.0
WORTH_MIN_SCORE_HEAT_ACCUM = 32.0
WORTH_MIN_SCORE_PATRICK_COMPOSITE = 62.0  # pool_sc + abs(d6h)*2.5
WORTH_MIN_SCORE_HOT_OI = 38.0  # heat + d6h*1.5
WORTH_MIN_FR_STRENGTH_CHASE_FIRE = 0.035  # -fr_pct，约 -0.035%
WORTH_MIN_COMBINED_TOTAL_DUAL = 72.0
WORTH_MIN_AMBUSH_TOTAL = 40.0
# 兼容旧逻辑命名：固定「至少保留」人数（动态模式下与 FALLBACK_MIN_K 一致）
WORTH_CATEGORY_TOP_N = WORTH_CATEGORY_FALLBACK_MIN_K
# 值得关注七张 worth_watch_* 表：按 generated_date 保留最近 N 个日历日（含当日）；七类统一默认 7 天。
try:
    WORTH_WATCH_RETENTION_DAYS = max(
        1, int(os.getenv("WORTH_WATCH_RETENTION_DAYS", "7").strip() or "7")
    )
except Exception:
    WORTH_WATCH_RETENTION_DAYS = 7
WORTH_HIGHLIGHT_RETENTION_DAYS = WORTH_WATCH_RETENTION_DAYS
try:
    WORTH_HOT_OI_RETENTION_DAYS = max(
        1,
        int(
            os.getenv(
                "WORTH_HOT_OI_RETENTION_DAYS",
                str(WORTH_WATCH_RETENTION_DAYS),
            ).strip()
            or str(WORTH_WATCH_RETENTION_DAYS)
        ),
    )
except Exception:
    WORTH_HOT_OI_RETENTION_DAYS = WORTH_WATCH_RETENTION_DAYS
WORTH_WATCH_TABLE_BY_CATEGORY: Dict[str, str] = {
    "heat_accum": "worth_watch_heat_accum",
    "patrick_core": "worth_watch_patrick_core",
    "hot_oi": "worth_watch_hot_oi",
    "chase_fire": "worth_watch_chase_fire",
    "dual_list": "worth_watch_dual_list",
    "ambush_dark": "worth_watch_ambush_dark",
    "ambush_gem": "worth_watch_ambush_gem",
}
WORTH_HIGHLIGHT_CATEGORY_ORDER: Tuple[str, ...] = (
    "heat_accum",
    "patrick_core",
    "hot_oi",
    "chase_fire",
    "dual_list",
    "ambush_dark",
    "ambush_gem",
)
WORTH_HIGHLIGHT_CATEGORY_LABEL_ZH: Dict[str, str] = {
    "heat_accum": "🔥💤 热度+收筹",
    "patrick_core": "📍 Patrick核心",
    "hot_oi": "🔥⚡ 热度+OI",
    "chase_fire": "🔥 追多·费率加速",
    "dual_list": "⭐ 追多+综合双榜",
    "ambush_dark": "🎯 埋伏·暗流",
    "ambush_gem": "💎 埋伏·低市值+OI",
}
# 值得关注合并后的 API / 电报展示条数上限（7 类 × 每类至多 MAX_K 条）
WORTH_HIGHLIGHTS_MAX = WORTH_CATEGORY_MAX_K * len(WORTH_HIGHLIGHT_CATEGORY_ORDER)
# 流动性掠夺（弹簧）：仅扫描收筹池内高分标的，控制 API 量
LIQUIDITY_SWEEP_POOL_MAX_SCAN = 45
LIQUIDITY_SWEEP_PIERCE_FRAC = 0.0012  # 相对 zone_low 下探深度
LIQUIDITY_SWEEP_RECOVERY_BARS = 5
LIQUIDITY_SWEEP_OI_MAX_DROP = 0.15  # 相对洗盘 K 附近 OI，若之后萎缩超过该比例则不作弹簧
# 👑 重点关注：否决 + 三通道（入库 focus_watch，保留 2 日）
TOP_FOCUS_RETENTION_DAYS = 2
TOP_FOCUS_MAX = 15
FOCUS_VETO_PX_POS_OI_NEG_PCT = -5.0  # px_chg>0 且 d6h 低于此值 → 否决（空头平仓上涨）
FOCUS_SQUEEZE_SW_MIN = 60
FOCUS_SQUEEZE_FR_MAX = -0.05  # fr_pct <= -0.05%
FOCUS_DARK_ABS_PX_MAX = 3.0
FOCUS_DARK_D6H_MIN = 4.0
FOCUS_DARK_MCAP_MAX_USD = 200_000_000
FOCUS_VOL_IGNITE_BREAKOUT_MIN = 10.0
FOCUS_VOL_IGNITE_STATUS_NEEDLE = "放量启动"
FOCUS_CHANNEL_PRIORITY: Dict[str, int] = {
    "squeeze": 1,
    "volume_ignite": 2,
    "dark_flow": 3,
}
FOCUS_CHANNEL_LABEL_ZH: Dict[str, str] = {
    "squeeze": "极致逼空",
    "volume_ignite": "绝地天量",
    "dark_flow": "纯正暗流",
}
FOCUS_CHANNEL_EMOJI: Dict[str, str] = {
    "squeeze": "🚀",
    "volume_ignite": "🌋",
    "dark_flow": "🥷",
}
FOCUS_STRATEGY_TIP_ZH: Dict[str, str] = {
    "squeeze": "关注突破前高与费率变化，谨防末端踩踏反转。",
    "volume_ignite": "等待 1h 缩量回踩价值区上沿再跟随，慎追阳线末端。",
    "dark_flow": "贴近 POC/价值区观察，仓位随波动分批。",
}
_LEGACY_HEAT_ACCUM_JSON = Path(db_dir) / "heat_accum_watchlist.json"
# 热度收筹表：突破—回踩—延续状态机（1h K 线，不含 OI）
HEAT_ACCUM_BPC_INTERVAL = "1h"
HEAT_ACCUM_BPC_KLINE_LIMIT = 120
# API / 看盘展示：与 BPC phase 字段对应（避免「待突破」暗示必做多）
BPC_PHASE_ZH: Dict[str, str] = {
    "idle": "观望",
    "post_breakout": "突破跟进",
    "pullback": "回踩中",
    "continuation": "延续确认",
}
# continuation_reason 英文 key → 中文（用于前端/推送）
BPC_CONTINUATION_REASON_ZH: Dict[str, str] = {
    "pin_bar": "长下影·Pin",
    "bullish_engulfing": "看涨吞没",
    "reclaim_micro_high": "收复回踩段前高",
}


def worth_pick_dynamic(
    ordered: List[Dict[str, Any]],
    *,
    score_fn: Callable[[Dict[str, Any]], float],
    score_min: float,
    max_k: int = WORTH_CATEGORY_MAX_K,
    fallback_k: int = WORTH_CATEGORY_FALLBACK_MIN_K,
    tie_eps: float = WORTH_CATEGORY_TIE_EPS,
) -> List[Dict[str, Any]]:
    """
    按既有排序依次入选：优先 score>=score_min，上限 max_k；若无达标则取前 fallback_k（上限仍为 max_k）。
    若已满一档但因并列接近（分差<=tie_eps），可补缺至 max_k。
    """
    if not ordered:
        return []
    primary: List[Dict[str, Any]] = []
    for x in ordered:
        if score_fn(x) >= score_min:
            primary.append(x)
            if len(primary) >= max_k:
                break
    if not primary:
        cap = min(max_k, fallback_k, len(ordered))
        return ordered[:cap]
    if len(primary) >= max_k:
        return primary[:max_k]
    floor = score_fn(primary[-1]) - tie_eps
    seen_id = {id(x) for x in primary}
    for x in ordered:
        if id(x) in seen_id:
            continue
        if score_fn(x) >= floor:
            primary.append(x)
            seen_id.add(id(x))
            if len(primary) >= max_k:
                break
    return primary[:max_k]


def volume_profile_from_daily_window(
    window_rows: List[Dict[str, Any]],
    *,
    num_bins: int = 42,
    value_area_pct: float = 0.70,
) -> Tuple[float, float, float]:
    """横盘窗口内日线堆量：POC + 价值区上下沿（成交量占比 value_area_pct）。"""
    if not window_rows:
        return 0.0, 0.0, 0.0
    lo = min(r["low"] for r in window_rows)
    hi = max(r["high"] for r in window_rows)
    if hi <= lo:
        return window_rows[-1]["close"], lo, hi
    tot_vol = sum(float(r["vol"]) for r in window_rows)
    if tot_vol <= 0:
        return (lo + hi) / 2.0, lo, hi
    n = max(int(num_bins), 12)
    step = (hi - lo) / float(n)
    if step <= 0:
        return (lo + hi) / 2.0, lo, hi
    bins = [0.0] * n
    for r in window_rows:
        a, b, va = float(r["low"]), float(r["high"]), float(r["vol"])
        if va <= 0:
            continue
        ia = int((a - lo) / step)
        ib = int((b - lo) / step)
        ia = max(0, min(n - 1, ia))
        ib = max(0, min(n - 1, ib))
        if ia > ib:
            ia, ib = ib, ia
        span = ib - ia + 1
        per = va / float(span)
        for k in range(ia, ib + 1):
            bins[k] += per
    peak_i = max(range(n), key=lambda i: bins[i])
    poc = lo + (peak_i + 0.5) * step
    totv = sum(bins)
    if totv <= 0:
        return poc, lo, hi
    target = totv * float(value_area_pct)
    acc = bins[peak_i]
    L = R = peak_i
    while acc < target and (L > 0 or R < n - 1):
        lv = bins[L - 1] if L > 0 else -1.0
        rv = bins[R + 1] if R < n - 1 else -1.0
        if lv >= rv:
            if L > 0:
                L -= 1
                acc += bins[L]
            elif R < n - 1:
                R += 1
                acc += bins[R]
            else:
                break
        else:
            if R < n - 1:
                R += 1
                acc += bins[R]
            elif L > 0:
                L -= 1
                acc += bins[L]
            else:
                break
    va_lo = lo + L * step
    va_hi = lo + (R + 1) * step
    return poc, va_lo, va_hi


def _nearest_oi_hist_value(oi_hist: List[Dict[str, Any]], target_ms: int) -> float:
    best_v = 0.0
    best_d = 10**18
    for row in oi_hist:
        ts = int(row.get("timestamp") or 0)
        if not ts:
            continue
        d = abs(ts - target_ms)
        if d < best_d:
            best_d = d
            best_v = float(row.get("sumOpenInterestValue") or 0)
    return best_v


def evaluate_liquidity_spring(symbol: str, zone_low: float, zone_high: float) -> Dict[str, Any]:
    """
    假跌破弹簧：1h 内最低价刺穿收筹下沿后，短期收回区间内；且洗盘时刻附近 OI 未显著出逃。
    zone_high 预留对称扩展，当前逻辑以下沿为主。
    """
    _ = zone_high
    out: Dict[str, Any] = {
        "detected": False,
        "reason": "",
        "bars_ago": None,
        "oi_drop_since_sweep_pct": None,
    }
    if zone_low <= 0:
        out["reason"] = "no_zone"
        return out
    kl = api_get("/fapi/v1/klines", {"symbol": symbol, "interval": "1h", "limit": 72})
    if not kl or len(kl) < 12:
        out["reason"] = "no_klines"
        return out
    oi_hist = api_get("/futures/data/openInterestHist", {"symbol": symbol, "period": "1h", "limit": 72})
    if not oi_hist or len(oi_hist) < 4:
        out["reason"] = "no_oi"
        return out
    thr_low = zone_low * (1.0 - LIQUIDITY_SWEEP_PIERCE_FRAC)
    n = len(kl)
    sweep_i: Optional[int] = None
    for i in range(n - 1, 4, -1):
        low_i = float(kl[i][3])
        if low_i >= thr_low:
            continue
        end = min(n, i + LIQUIDITY_SWEEP_RECOVERY_BARS + 1)
        recovered = False
        for j in range(i, end):
            if float(kl[j][4]) >= zone_low:
                recovered = True
                break
        if not recovered:
            continue
        sweep_i = i
        break
    if sweep_i is None:
        out["reason"] = "no_pattern"
        return out
    t_i = int(kl[sweep_i][0])
    oi_at = _nearest_oi_hist_value(oi_hist, t_i)
    oi_now = float(oi_hist[-1].get("sumOpenInterestValue") or 0)
    drop = 0.0
    if oi_at > 0:
        drop = (oi_now - oi_at) / oi_at
        if drop < -LIQUIDITY_SWEEP_OI_MAX_DROP:
            out["reason"] = "oi_flush"
            out["oi_drop_since_sweep_pct"] = drop * 100.0
            return out
    out["detected"] = True
    out["reason"] = "spring"
    out["bars_ago"] = n - 1 - sweep_i
    out["oi_drop_since_sweep_pct"] = drop * 100.0 if oi_at > 0 else None
    return out


def enrich_liquidity_spring_batch(coin_data: Dict[str, Dict[str, Any]], pool_map: Dict[str, Any]) -> None:
    """对收筹池内标的批量标注 liquidity_spring（限前 N 名 pool_score，控制请求量）。"""
    ranked = sorted(
        pool_map.items(),
        key=lambda kv: float(kv[1].get("pool_score") or 0),
        reverse=True,
    )[:LIQUIDITY_SWEEP_POOL_MAX_SCAN]
    for sym, pd in ranked:
        low = float(pd.get("low_price") or 0)
        high = float(pd.get("high_price") or 0)
        if low <= 0:
            continue
        row = coin_data.get(sym)
        if not row:
            continue
        ev = evaluate_liquidity_spring(sym, low, high)
        row["liquidity_spring"] = ev
        time.sleep(0.03)


def _persist_oi_radar_snapshot(payload: Dict[str, Any]) -> None:
    """供 GET /api/accumulation/oi-radar 读盘；定时任务与后台 refresh 写入同一路径。"""
    if not payload.get("ok"):
        return
    tmp = OI_RADAR_SNAPSHOT_PATH.with_suffix(".json.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp.replace(OI_RADAR_SNAPSHOT_PATH)
        print(f"  💾 OI 快照已写入 {OI_RADAR_SNAPSHOT_PATH}")
    except Exception as e:
        print(f"⚠️ OI 快照写入失败: {e}")


def _heat_accum_summary_line(sig: Dict[str, Any]) -> str:
    """与「值得关注」热力+收筹条一致的单行摘要（CST 语境）。"""
    tags = list(sig.get("tags") or [])
    coin = sig.get("coin") or ""
    sw = sig.get("sideways_days") or 0
    return f"🔥💤 {coin} 热度({'+'.join(tags)})+收筹{sw}天=OI将涨"


def _patrick_core_summary_line(sig: Dict[str, Any]) -> str:
    """Patrick 核心看盘摘要（与「值得关注」📍 条一致）。"""
    coin = sig.get("coin") or ""
    sw = int(sig.get("sideways_days") or 0)
    d6 = float(sig.get("d6h") or 0)
    return f"📍 {coin} 收筹{sw}天+OI{d6:+.0f}%（Patrick核心）"


def _heat_accum_now_cst(now: datetime) -> datetime:
    cst = timezone(timedelta(hours=8))
    if now.tzinfo is None:
        return now.replace(tzinfo=cst)
    return now.astimezone(cst)


def _watchlist_cutoff_date_iso(now: datetime) -> str:
    """早于该日（不含）的 watchlist 行删除；含今天在内共 WATCHLIST_RETENTION_DAYS 个自然日。"""
    now_cst = _heat_accum_now_cst(now)
    today = now_cst.date()
    cutoff = today - timedelta(days=WATCHLIST_RETENTION_DAYS - 1)
    return cutoff.isoformat()


def _watchlist_prune(conn: sqlite3.Connection, now: datetime) -> int:
    """按 added_date 日期删超窗行（及无效 added_date）。仅在每日 pool 写入前调用。"""
    cutoff_s = _watchlist_cutoff_date_iso(now)
    cur = conn.cursor()
    cur.execute(
        """
        DELETE FROM watchlist
        WHERE added_date IS NULL OR length(trim(added_date)) < 10
           OR substr(added_date, 1, 10) < ?
        """,
        (cutoff_s,),
    )
    n = int(cur.rowcount or 0)
    conn.commit()
    return n


def _heat_accum_cutoff_iso(now: datetime) -> str:
    """早于该生成日（不含）的行删除；含今天在内共 RETENTION_DAYS 个日历日。"""
    now_cst = _heat_accum_now_cst(now)
    today = now_cst.date()
    cutoff = today - timedelta(days=HEAT_ACCUM_RETENTION_DAYS - 1)
    return cutoff.isoformat()


def _heat_accum_prune(conn: sqlite3.Connection, now: datetime) -> None:
    cutoff_s = _heat_accum_cutoff_iso(now)
    conn.execute("DELETE FROM heat_accum_watch WHERE generated_date < ?", (cutoff_s,))


def _parse_bpc_for_item(bpc_json: Optional[str], bpc_updated_cst: Optional[str]) -> Optional[Dict[str, Any]]:
    if not bpc_json:
        return None
    try:
        d = json.loads(bpc_json)
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    ph = str(d.get("phase") or "idle")
    # last_continuation_reason 仅为窗口内快照；副标题只在「当前处于延续相位」时展示，
    # 否则观望/回踩也会被套上陈旧形态（多为同一种吞没结论）。
    cr_s = ""
    cr_zh = ""
    if ph == "continuation":
        cr_raw = d.get("continuation_reason") or d.get("last_continuation_reason")
        cr_s = str(cr_raw).strip() if cr_raw else ""
        cr_zh = BPC_CONTINUATION_REASON_ZH.get(cr_s, cr_s) if cr_s else ""
    return {
        "ok": d.get("ok", True),
        "phase": ph,
        "phase_zh": BPC_PHASE_ZH.get(ph, ph),
        "reason": d.get("reason"),
        "continuation_reason": cr_s or None,
        "continuation_reason_zh": cr_zh or None,
        "pullback_vol_contracted": d.get("pullback_vol_contracted"),
        "breakout_level": d.get("breakout_level"),
        "peak_after_breakout": d.get("peak_after_breakout"),
        "last_invalid_reason": d.get("last_invalid_reason"),
        "interval": d.get("interval") or HEAT_ACCUM_BPC_INTERVAL,
        "evaluated_at_cst": bpc_updated_cst,
    }


def _sqlite_row_to_watch_item(row: Tuple[Any, ...]) -> Dict[str, Any]:
    (
        symbol,
        coin,
        generated_date,
        last_seen_cst,
        heat,
        sideways_days,
        tags_json,
        low_price,
        high_price,
        price,
        zone_json,
        zone_reason,
        summary_line,
        bpc_json,
        bpc_updated_cst,
    ) = row
    tags: List[Any] = []
    if tags_json:
        try:
            t = json.loads(tags_json)
            if isinstance(t, list):
                tags = t
        except Exception:
            pass
    zone: Optional[Any] = None
    if zone_json:
        try:
            zone = json.loads(zone_json)
        except Exception:
            zone = None
    return {
        "symbol": symbol,
        "coin": coin,
        "generated_date": generated_date,
        "last_seen_cst": last_seen_cst,
        "heat": heat,
        "tags": tags,
        "sideways_days": sideways_days,
        "low_price": low_price,
        "high_price": high_price,
        "price": price,
        "zone": zone,
        "zone_reason": zone_reason,
        "summary_line": summary_line,
        "bpc": _parse_bpc_for_item(
            str(bpc_json) if bpc_json else None,
            str(bpc_updated_cst) if bpc_updated_cst else None,
        ),
    }


def _heat_accum_fetch_payload(conn: sqlite3.Connection, now: datetime) -> Dict[str, Any]:
    now_cst = _heat_accum_now_cst(now)
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"
    cur = conn.cursor()
    cur.execute(
        """
        SELECT symbol, coin, generated_date, last_seen_cst, heat, sideways_days, tags_json,
               low_price, high_price, price, zone_json, zone_reason, summary_line,
               bpc_json, bpc_updated_cst
        FROM heat_accum_watch
        ORDER BY generated_date DESC, last_seen_cst DESC, symbol DESC
        """
    )
    rows = cur.fetchall()
    items = [_sqlite_row_to_watch_item(tuple(r)) for r in rows]
    seen_times = [it.get("last_seen_cst") for it in items if isinstance(it.get("last_seen_cst"), str)]
    updated_at = max(seen_times) if seen_times else now_label
    bpc_times: List[str] = []
    for it in items:
        b = it.get("bpc")
        if isinstance(b, dict) and b.get("evaluated_at_cst"):
            bpc_times.append(str(b["evaluated_at_cst"]))
    bpc_snapshot_cst = max(bpc_times) if bpc_times else None
    return {
        "ok": True,
        "items": items,
        "updated_at_cst": updated_at,
        "bpc_snapshot_cst": bpc_snapshot_cst,
        "bpc_interval": HEAT_ACCUM_BPC_INTERVAL,
        "retention_days": HEAT_ACCUM_RETENTION_DAYS,
        "storage": "sqlite",
    }


def load_heat_accum_watchlist_from_db(
    conn: sqlite3.Connection,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """供 HTTP GET：按保留策略清理过期行后返回当前列表。"""
    if now is None:
        now = datetime.now(timezone(timedelta(hours=8)))
    _heat_accum_prune(conn, now)
    conn.commit()
    return _heat_accum_fetch_payload(conn, now)


def refresh_all_heat_accum_watch_prices(
    conn: sqlite3.Connection,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    对 heat_accum_watch 全表刷新现价、last_seen、摘要；清空 zone（已不再计算买入/进场区间）。
    由「热度看盘整表刷新」每小时与其它步骤一并调用。
    """
    if now is None:
        now = datetime.now(timezone(timedelta(hours=8)))
    now_cst = _heat_accum_now_cst(now)
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"

    _heat_accum_prune(conn, now)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT symbol, coin, heat, sideways_days, tags_json, low_price, high_price, price
        FROM heat_accum_watch
        """
    )
    rows = cur.fetchall()
    if not rows:
        conn.commit()
        return _heat_accum_fetch_payload(conn, now)

    ticker_map: Dict[str, float] = {}
    t24 = api_get("/fapi/v1/ticker/24hr")
    if isinstance(t24, list):
        for r in t24:
            try:
                sym = str(r.get("symbol") or "")
                px = float(r.get("lastPrice") or 0.0)
                if sym and px > 0:
                    ticker_map[sym] = px
            except Exception:
                continue

    recalculated = 0
    for row in rows:
        symbol = str(row[0] or "")
        if not symbol:
            continue
        coin = row[1]
        heat = row[2]
        sideways_days = row[3]
        tags_raw = row[4]
        low_price = float(row[5] or 0.0)
        high_price = float(row[6] or 0.0)
        old_price = float(row[7] or 0.0)
        price = float(ticker_map.get(symbol) or old_price or 0.0)

        tags: List[Any] = []
        if isinstance(tags_raw, str) and tags_raw:
            try:
                parsed = json.loads(tags_raw)
                if isinstance(parsed, list):
                    tags = parsed
            except Exception:
                tags = []

        sig = {
            "coin": coin,
            "symbol": symbol,
            "heat": heat,
            "tags": tags,
            "sideways_days": sideways_days,
            "low_price": low_price,
            "high_price": high_price,
            "price": price,
        }
        summary = _heat_accum_summary_line(sig)
        cur.execute(
            """
            UPDATE heat_accum_watch SET
                price = ?, last_seen_cst = ?, zone_json = ?, zone_reason = ?, summary_line = ?
            WHERE symbol = ?
            """,
            (price, now_label, None, None, summary, symbol),
        )
        recalculated += 1

    conn.commit()
    try:
        patch_oi_radar_snapshot_watchlists_from_db(conn)
    except Exception as e:
        print(f"⚠️ patch oi_radar snapshot after heat watch price refresh failed: {e}")

    payload = _heat_accum_fetch_payload(conn, now)
    payload["recalculated"] = recalculated
    return payload


def refresh_all_worth_watch_bpc_states(
    conn: sqlite3.Connection,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    原 worth_watch / focus 各行 1h BPC（breakout_pullback_fsm）已移除；保留空操作以兼容 main 调用链。
    """
    conn.commit()
    return {
        "worth_watch_bpc_recalculated": 0,
        "worth_watch_bpc_failed_klines": 0,
        "worth_watch_bpc_symbols": 0,
        "bpc_disabled": True,
    }


def refresh_all_heat_accum_watch_full(
    conn: sqlite3.Connection,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    heat_accum_watch 整表：同步现价/摘要并清空 zone（1h BPC 重算已随 breakout_pullback_fsm 移除）。
    每小时定时与单一维护接口共用。
    """
    if now is None:
        now = datetime.now(timezone(timedelta(hours=8)))
    p_px = refresh_all_heat_accum_watch_prices(conn, now=now)
    out = dict(p_px)
    out["recalculated_prices"] = p_px.get("recalculated")
    out["price_rows"] = p_px.get("recalculated")
    out["bpc_recalculated"] = 0
    out["bpc_failed_klines"] = 0
    out["bpc_disabled"] = True
    return out


def merge_and_persist_heat_accum_watchlist(
    conn: sqlite3.Connection,
    hot_pool_signals: List[Dict[str, Any]],
    now: datetime,
) -> Dict[str, Any]:
    """
    增量写入 accumulation.db：当前轮 hot_pool（热度+收筹）；按 symbol 主键去重；
    首次出现记 generated_date（CST，精确到分），再次命中刷新区间等字段但保留生成时间；
    仅保留生成日在最近 HEAT_ACCUM_RETENTION_DAYS 个日历日内的条目。
    """
    now_cst = _heat_accum_now_cst(now)
    generated_at_s = now_cst.strftime("%Y-%m-%d %H:%M")
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"

    _heat_accum_prune(conn, now)
    cur = conn.cursor()

    for sig in hot_pool_signals:
        if not isinstance(sig, dict):
            continue
        sym = sig.get("symbol")
        if not sym:
            continue
        sym = str(sym)
        summary = _heat_accum_summary_line(sig)
        tags_json = json.dumps(sig.get("tags") or [], ensure_ascii=False)
        zj = sig.get("zone_meta")
        zone_json_s = json.dumps(zj, ensure_ascii=False) if isinstance(zj, dict) and zj else None

        cur.execute("SELECT generated_date FROM heat_accum_watch WHERE symbol = ?", (sym,))
        ex = cur.fetchone()
        if ex:
            cur.execute(
                """
                UPDATE heat_accum_watch SET
                    coin = ?, last_seen_cst = ?, heat = ?, sideways_days = ?, tags_json = ?,
                    low_price = ?, high_price = ?, price = ?, zone_json = ?, zone_reason = ?, summary_line = ?
                WHERE symbol = ?
                """,
                (
                    sig.get("coin"),
                    now_label,
                    sig.get("heat"),
                    sig.get("sideways_days"),
                    tags_json,
                    sig.get("low_price"),
                    sig.get("high_price"),
                    sig.get("price"),
                    zone_json_s,
                    None,
                    summary,
                    sym,
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO heat_accum_watch (
                    symbol, coin, generated_date, last_seen_cst, heat, sideways_days, tags_json,
                    low_price, high_price, price, zone_json, zone_reason, summary_line,
                    bpc_json, bpc_updated_cst
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    sym,
                    sig.get("coin"),
                    generated_at_s,
                    now_label,
                    sig.get("heat"),
                    sig.get("sideways_days"),
                    tags_json,
                    sig.get("low_price"),
                    sig.get("high_price"),
                    sig.get("price"),
                    zone_json_s,
                    None,
                    summary,
                ),
            )

    conn.commit()
    print(f"  💾 热度+收筹看盘已写入 SQLite ({DB_PATH})")
    return _heat_accum_fetch_payload(conn, now)


def _migrate_legacy_heat_accum_json(conn: sqlite3.Connection) -> None:
    """一次性：旧 heat_accum_watchlist.json → DB，成功后改名为 .bak。"""
    path = _LEGACY_HEAT_ACCUM_JSON
    if not path.is_file():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            return
        cur = conn.cursor()
        migrated = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            sym = it.get("symbol")
            if not sym:
                continue
            gd = it.get("generated_date")
            if not gd or not isinstance(gd, str):
                continue
            cur.execute("SELECT 1 FROM heat_accum_watch WHERE symbol = ?", (str(sym),))
            if cur.fetchone():
                continue
            tags_json = json.dumps(it.get("tags") or [], ensure_ascii=False)
            zone = it.get("zone")
            zone_json = json.dumps(zone, ensure_ascii=False) if zone else None
            cur.execute(
                """
                INSERT OR IGNORE INTO heat_accum_watch (
                    symbol, coin, generated_date, last_seen_cst, heat, sideways_days, tags_json,
                    low_price, high_price, price, zone_json, zone_reason, summary_line,
                    bpc_json, bpc_updated_cst
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    str(sym),
                    it.get("coin"),
                    gd[:10],
                    str(it.get("last_seen_cst") or "") or "(迁移)",
                    it.get("heat"),
                    it.get("sideways_days"),
                    tags_json,
                    it.get("low_price"),
                    it.get("high_price"),
                    it.get("price"),
                    zone_json,
                    str(it.get("zone_reason")) if it.get("zone_reason") is not None else None,
                    str(it.get("summary_line") or ""),
                ),
            )
            if cur.rowcount == 1:
                migrated += 1
        conn.commit()
        bak = path.with_suffix(".json.bak")
        path.rename(bak)
        print(f"  📦 已迁移热度+收筹看盘 JSON → DB（新增 {migrated} 条），原文件改为 {bak.name}")
    except Exception as e:
        print(f"⚠️ 迁移 heat_accum_watchlist.json 跳过: {e}")


def _ambush_watch_cutoff_iso(now: datetime) -> str:
    now_cst = _heat_accum_now_cst(now)
    today = now_cst.date()
    cutoff = today - timedelta(days=AMBUSH_WATCH_RETENTION_DAYS - 1)
    return cutoff.isoformat()


def _ambush_watch_prune(conn: sqlite3.Connection, now: datetime) -> None:
    cutoff_s = _ambush_watch_cutoff_iso(now)
    conn.execute("DELETE FROM ambush_watch WHERE generated_date < ?", (cutoff_s,))


def _sqlite_row_to_ambush_item(row: Tuple[Any, ...]) -> Dict[str, Any]:
    (
        symbol,
        signal_type,
        coin,
        generated_date,
        last_seen_cst,
        d6h,
        px_chg,
        est_mcap,
        ambush_total,
        summary_line,
    ) = row
    return {
        "symbol": symbol,
        "signal_type": signal_type,
        "coin": coin,
        "generated_date": generated_date,
        "last_seen_cst": last_seen_cst,
        "d6h": d6h,
        "px_chg": px_chg,
        "est_mcap": est_mcap,
        "ambush_total": ambush_total,
        "summary_line": summary_line,
    }


def _ambush_watch_fetch_payload(conn: sqlite3.Connection, now: datetime) -> Dict[str, Any]:
    now_cst = _heat_accum_now_cst(now)
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"
    cur = conn.cursor()
    cur.execute(
        """
        SELECT symbol, signal_type, coin, generated_date, last_seen_cst,
               d6h, px_chg, est_mcap, ambush_total, summary_line
        FROM ambush_watch
        ORDER BY generated_date DESC, last_seen_cst DESC,
                 signal_type ASC, (ambush_total IS NULL) ASC, ambush_total DESC, symbol ASC
        """
    )
    rows = cur.fetchall()
    items = [_sqlite_row_to_ambush_item(tuple(r)) for r in rows]
    seen_times = [it.get("last_seen_cst") for it in items if isinstance(it.get("last_seen_cst"), str)]
    updated_at = max(seen_times) if seen_times else now_label
    return {
        "ok": True,
        "items": items,
        "updated_at_cst": updated_at,
        "retention_days": AMBUSH_WATCH_RETENTION_DAYS,
        "storage": "sqlite",
    }


def load_ambush_watchlist_from_db(
    conn: sqlite3.Connection,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """供 HTTP GET：按保留策略清理过期行后返回当前列表。"""
    if now is None:
        now = datetime.now(timezone(timedelta(hours=8)))
    _ambush_watch_prune(conn, now)
    conn.commit()
    return _ambush_watch_fetch_payload(conn, now)


def patch_oi_radar_snapshot_watchlists_from_db(conn: sqlite3.Connection) -> bool:
    """
    将磁盘 oi_radar_snapshot.json 内的嵌套看盘列表与当前 SQLite 对齐。
    仅清库而未重跑雷达时调用，避免前端仍读到快照里的陈旧副本。
    """
    if not OI_RADAR_SNAPSHOT_PATH.is_file():
        return False
    try:
        raw = json.loads(OI_RADAR_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(raw, dict) or not raw.get("ok"):
        return False
    now = datetime.now(timezone(timedelta(hours=8)))
    raw["ambush_watchlist"] = load_ambush_watchlist_from_db(conn, now=now)
    raw["heat_accum_watchlist"] = load_heat_accum_watchlist_from_db(conn, now=now)
    raw["patrick_core_watchlist"] = load_patrick_core_watchlist_from_db(conn, now=now)
    raw["worth_highlight_watchlist"] = load_worth_highlight_watchlist_from_db(conn, now=now)
    raw["focus_watchlist"] = load_focus_watchlist_from_db(conn, now=now)
    _persist_oi_radar_snapshot(raw)
    return True


def patch_oi_radar_snapshot_after_watchlist_clear(conn: sqlite3.Connection) -> bool:
    """
    清空 watchlist 后更新磁盘快照：标记不可用并清空依赖收筹池的列表字段，
    同时把嵌套看盘列表与当前 DB 对齐。
    """
    if not OI_RADAR_SNAPSHOT_PATH.is_file():
        return False
    try:
        raw = json.loads(OI_RADAR_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(raw, dict):
        return False
    raw["ok"] = False
    raw["error"] = "watchlist_empty"
    raw["message"] = (
        "收筹池已清空。请先运行 pool 扫描（定时每日 10:00 或维护面板「pool 收筹池」），再点刷新。"
    )
    for key in (
        "coin_data",
        "hot_coins",
        "chase",
        "combined",
        "ambush",
        "highlights",
        "hot_pool_signals",
    ):
        if key in raw:
            raw[key] = []
    now = datetime.now(timezone(timedelta(hours=8)))
    raw["ambush_watchlist"] = load_ambush_watchlist_from_db(conn, now=now)
    raw["heat_accum_watchlist"] = load_heat_accum_watchlist_from_db(conn, now=now)
    raw["patrick_core_watchlist"] = load_patrick_core_watchlist_from_db(conn, now=now)
    raw["worth_highlight_watchlist"] = load_worth_highlight_watchlist_from_db(conn, now=now)
    raw["focus_watchlist"] = load_focus_watchlist_from_db(conn, now=now)
    _persist_oi_radar_snapshot(raw)
    return True


def clear_watchlist_table(conn: sqlite3.Connection) -> int:
    """清空收筹标的池表 watchlist。返回清空前行数。"""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM watchlist")
    n = int(cur.fetchone()[0] or 0)
    cur.execute("DELETE FROM watchlist")
    conn.commit()
    return n


def clear_ambush_watch_table(conn: sqlite3.Connection) -> int:
    """清空表 ambush_watch。返回清空前行数。"""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ambush_watch")
    n = int(cur.fetchone()[0] or 0)
    cur.execute("DELETE FROM ambush_watch")
    conn.commit()
    return n


def clear_heat_accum_watch_table(conn: sqlite3.Connection) -> int:
    """清空表 heat_accum_watch。返回清空前行数。"""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM heat_accum_watch")
    n = int(cur.fetchone()[0] or 0)
    cur.execute("DELETE FROM heat_accum_watch")
    conn.commit()
    return n


def _patrick_core_cutoff_iso(now: datetime) -> str:
    """早于该生成日（不含）的行删除；含今天在内共 PATRICK_CORE_RETENTION_DAYS 个日历日。"""
    now_cst = _heat_accum_now_cst(now)
    today = now_cst.date()
    cutoff = today - timedelta(days=PATRICK_CORE_RETENTION_DAYS - 1)
    return cutoff.isoformat()


def _patrick_core_prune(conn: sqlite3.Connection, now: datetime) -> None:
    cutoff_s = _patrick_core_cutoff_iso(now)
    conn.execute("DELETE FROM patrick_core_watch WHERE generated_date < ?", (cutoff_s,))


def _sqlite_row_to_patrick_item(row: Tuple[Any, ...]) -> Dict[str, Any]:
    (
        symbol,
        coin,
        generated_date,
        last_seen_cst,
        sideways_days,
        d6h,
        px_chg,
        est_mcap,
        price,
        low_price,
        high_price,
        summary_line,
    ) = row
    return {
        "symbol": symbol,
        "coin": coin,
        "generated_date": generated_date,
        "last_seen_cst": last_seen_cst,
        "sideways_days": sideways_days,
        "d6h": d6h,
        "px_chg": px_chg,
        "est_mcap": est_mcap,
        "price": price,
        "low_price": low_price,
        "high_price": high_price,
        "summary_line": summary_line,
    }


def _patrick_core_fetch_payload(conn: sqlite3.Connection, now: datetime) -> Dict[str, Any]:
    now_cst = _heat_accum_now_cst(now)
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"
    cur = conn.cursor()
    cur.execute(
        """
        SELECT symbol, coin, generated_date, last_seen_cst,
               sideways_days, d6h, px_chg, est_mcap, price, low_price, high_price, summary_line
        FROM patrick_core_watch
        ORDER BY generated_date DESC, last_seen_cst DESC, symbol ASC
        """
    )
    rows = cur.fetchall()
    items = [_sqlite_row_to_patrick_item(tuple(r)) for r in rows]
    seen_times = [it.get("last_seen_cst") for it in items if isinstance(it.get("last_seen_cst"), str)]
    updated_at = max(seen_times) if seen_times else now_label
    return {
        "ok": True,
        "items": items,
        "updated_at_cst": updated_at,
        "retention_days": PATRICK_CORE_RETENTION_DAYS,
        "storage": "sqlite",
    }


def load_patrick_core_watchlist_from_db(
    conn: sqlite3.Connection,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """供 HTTP GET：按保留策略清理过期行后返回当前列表。"""
    if now is None:
        now = datetime.now(timezone(timedelta(hours=8)))
    _patrick_core_prune(conn, now)
    conn.commit()
    return _patrick_core_fetch_payload(conn, now)


def merge_and_persist_patrick_core_watchlist(
    conn: sqlite3.Connection,
    patrick_signals: List[Dict[str, Any]],
    now: datetime,
) -> Dict[str, Any]:
    """
    增量写入 patrick_core_watch：收筹池 + |OI| 达标；按 symbol 主键 upsert；
    首次出现记 generated_date（CST，精确到分），再次命中刷新指标与 last_seen；
    仅保留生成日在最近 PATRICK_CORE_RETENTION_DAYS 个日历日内的条目。
    """
    now_cst = _heat_accum_now_cst(now)
    generated_at_s = now_cst.strftime("%Y-%m-%d %H:%M")
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"

    _patrick_core_prune(conn, now)
    cur = conn.cursor()

    for sig in patrick_signals:
        if not isinstance(sig, dict):
            continue
        sym = sig.get("symbol")
        if not sym:
            continue
        sym = str(sym)
        summary = _patrick_core_summary_line(sig)

        cur.execute("SELECT generated_date FROM patrick_core_watch WHERE symbol = ?", (sym,))
        ex = cur.fetchone()
        if ex:
            cur.execute(
                """
                UPDATE patrick_core_watch SET
                    coin = ?, last_seen_cst = ?, sideways_days = ?, d6h = ?, px_chg = ?,
                    est_mcap = ?, price = ?, low_price = ?, high_price = ?, summary_line = ?
                WHERE symbol = ?
                """,
                (
                    sig.get("coin"),
                    now_label,
                    int(sig.get("sideways_days") or 0),
                    float(sig.get("d6h") or 0),
                    float(sig.get("px_chg") or 0),
                    float(sig.get("est_mcap") or 0),
                    float(sig.get("price") or 0),
                    float(sig.get("low_price") or 0),
                    float(sig.get("high_price") or 0),
                    summary,
                    sym,
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO patrick_core_watch (
                    symbol, coin, generated_date, last_seen_cst,
                    sideways_days, d6h, px_chg, est_mcap, price, low_price, high_price, summary_line
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sym,
                    sig.get("coin"),
                    generated_at_s,
                    now_label,
                    int(sig.get("sideways_days") or 0),
                    float(sig.get("d6h") or 0),
                    float(sig.get("px_chg") or 0),
                    float(sig.get("est_mcap") or 0),
                    float(sig.get("price") or 0),
                    float(sig.get("low_price") or 0),
                    float(sig.get("high_price") or 0),
                    summary,
                ),
            )

    conn.commit()
    print(f"  💾 Patrick核心看盘已写入 SQLite ({DB_PATH})")
    return _patrick_core_fetch_payload(conn, now)


def clear_patrick_core_watch_table(conn: sqlite3.Connection) -> int:
    """清空表 patrick_core_watch。返回清空前行数。"""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM patrick_core_watch")
    n = int(cur.fetchone()[0] or 0)
    cur.execute("DELETE FROM patrick_core_watch")
    conn.commit()
    return n


def _worth_watch_cutoff_iso(now: datetime, retention_days: int) -> str:
    """早于该生成日（不含）的行删除；含今天在内共 retention_days 个日历日。"""
    now_cst = _heat_accum_now_cst(now)
    today = now_cst.date()
    d = max(1, int(retention_days))
    cutoff = today - timedelta(days=d - 1)
    return cutoff.isoformat()


def _worth_watch_prune_all(conn: sqlite3.Connection, now: datetime) -> None:
    cutoff_s = _worth_watch_cutoff_iso(now, WORTH_WATCH_RETENTION_DAYS)
    for _cat, tbl in WORTH_WATCH_TABLE_BY_CATEGORY.items():
        conn.execute(f"DELETE FROM {tbl} WHERE generated_date < ?", (cutoff_s,))


def _sqlite_row_to_worth_item(row: Tuple[Any, ...], category: str) -> Dict[str, Any]:
    sym, coin, gd, ls, rk, summ, det = row[:7]
    bpc_json = row[7] if len(row) > 7 else None
    bpc_updated_cst = row[8] if len(row) > 8 else None
    detail: Optional[Dict[str, Any]] = None
    if det:
        try:
            d = json.loads(str(det))
            detail = d if isinstance(d, dict) else None
        except Exception:
            detail = None
    return {
        "category": category,
        "symbol": sym,
        "coin": coin,
        "generated_date": gd,
        "last_seen_cst": ls,
        "rank_in_category": rk,
        "summary_line": summ,
        "detail": detail or {},
        "bpc": _parse_bpc_for_item(
            str(bpc_json) if bpc_json else None,
            str(bpc_updated_cst) if bpc_updated_cst else None,
        ),
    }


def _worth_watch_fetch_payload(
    conn: sqlite3.Connection,
    now: datetime,
    *,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    now_cst = _heat_accum_now_cst(now)
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"
    cur = conn.cursor()
    items: List[Dict[str, Any]] = []
    order_sql = """
        ORDER BY generated_date DESC, last_seen_cst DESC,
                 COALESCE(rank_in_category, 999) ASC, symbol ASC
    """
    if category:
        cat = str(category).strip()
        tbl = WORTH_WATCH_TABLE_BY_CATEGORY.get(cat)
        if not tbl:
            raise ValueError(f"unknown worth category: {cat}")
        cur.execute(
            f"""
            SELECT symbol, coin, generated_date, last_seen_cst,
                   rank_in_category, summary_line, detail_json,
                   bpc_json, bpc_updated_cst
            FROM {tbl}
            {order_sql}
            """
        )
        items = [_sqlite_row_to_worth_item(tuple(r), cat) for r in cur.fetchall()]
    else:
        for cat in WORTH_HIGHLIGHT_CATEGORY_ORDER:
            tbl = WORTH_WATCH_TABLE_BY_CATEGORY[cat]
            cur.execute(
                f"""
                SELECT symbol, coin, generated_date, last_seen_cst,
                       rank_in_category, summary_line, detail_json,
                       bpc_json, bpc_updated_cst
                FROM {tbl}
                {order_sql}
                """
            )
            for r in cur.fetchall():
                items.append(_sqlite_row_to_worth_item(tuple(r), cat))
    seen_times = [it.get("last_seen_cst") for it in items if isinstance(it.get("last_seen_cst"), str)]
    updated_at = max(seen_times) if seen_times else now_label
    bpc_times: List[str] = []
    for it in items:
        b = it.get("bpc")
        if isinstance(b, dict) and b.get("evaluated_at_cst"):
            bpc_times.append(str(b["evaluated_at_cst"]))
    worth_bpc_snapshot = max(bpc_times) if bpc_times else None
    categories: Dict[str, Any] = {}
    for key in WORTH_HIGHLIGHT_CATEGORY_ORDER:
        sub = [it for it in items if it.get("category") == key]
        categories[key] = {
            "label_zh": WORTH_HIGHLIGHT_CATEGORY_LABEL_ZH.get(key, key),
            "items": sub,
            "table": WORTH_WATCH_TABLE_BY_CATEGORY.get(key),
        }
    return {
        "ok": True,
        "items": items,
        "categories": categories,
        "tables": dict(WORTH_WATCH_TABLE_BY_CATEGORY),
        "updated_at_cst": updated_at,
        "retention_days": WORTH_WATCH_RETENTION_DAYS,
        "hot_oi_retention_days": WORTH_HOT_OI_RETENTION_DAYS,
        "storage": "sqlite",
        "bpc_interval": HEAT_ACCUM_BPC_INTERVAL,
        "bpc_snapshot_cst": worth_bpc_snapshot,
    }


def load_worth_highlight_watchlist_from_db(
    conn: sqlite3.Connection,
    *,
    now: Optional[datetime] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    if now is None:
        now = datetime.now(timezone(timedelta(hours=8)))
    _worth_watch_prune_all(conn, now)
    conn.commit()
    return _worth_watch_fetch_payload(conn, now, category=category)


def merge_and_persist_worth_watch_category_tables(
    conn: sqlite3.Connection,
    buckets: Dict[str, List[Dict[str, Any]]],
    now: datetime,
) -> Dict[str, Any]:
    """
    七类「值得关注」：动态门槛 + 每类至多 WORTH_CATEGORY_MAX_K 条，分别 upsert 到对应物理表。
    每条需含: symbol, coin, summary_line, rank_in_category, detail(可选 dict，会写入 detail_json)。
    """
    now_cst = _heat_accum_now_cst(now)
    generated_at_s = now_cst.strftime("%Y-%m-%d %H:%M")
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"
    _worth_watch_prune_all(conn, now)
    cur = conn.cursor()
    for cat in WORTH_HIGHLIGHT_CATEGORY_ORDER:
        tbl = WORTH_WATCH_TABLE_BY_CATEGORY[cat]
        for ent in buckets.get(cat) or []:
            if not isinstance(ent, dict):
                continue
            sym = str(ent.get("symbol") or "").strip()
            if not sym:
                continue
            det = ent.get("detail")
            det_s = json.dumps(det, ensure_ascii=False) if isinstance(det, dict) else None
            cur.execute(f"SELECT generated_date FROM {tbl} WHERE symbol = ?", (sym,))
            ex = cur.fetchone()
            row_core = (
                str(ent.get("coin") or ""),
                now_label,
                int(ent.get("rank_in_category") or 0),
                str(ent.get("summary_line") or ""),
                det_s,
            )
            if ex:
                cur.execute(
                    f"""
                    UPDATE {tbl} SET
                        coin = ?, last_seen_cst = ?, rank_in_category = ?, summary_line = ?, detail_json = ?
                    WHERE symbol = ?
                    """,
                    row_core + (sym,),
                )
            else:
                cur.execute(
                    f"""
                    INSERT INTO {tbl} (
                        symbol, coin, generated_date, last_seen_cst,
                        rank_in_category, summary_line, detail_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sym,
                        str(ent.get("coin") or ""),
                        generated_at_s,
                        now_label,
                        int(ent.get("rank_in_category") or 0),
                        str(ent.get("summary_line") or ""),
                        det_s,
                    ),
                )
    conn.commit()
    print(f"  💾 值得关注七类看盘已写入 SQLite（{len(WORTH_WATCH_TABLE_BY_CATEGORY)} 张表） ({DB_PATH})")
    return _worth_watch_fetch_payload(conn, now)


# 兼容旧调用名
merge_and_persist_worth_highlight_watchlist = merge_and_persist_worth_watch_category_tables


def clear_one_worth_watch_category_table(conn: sqlite3.Connection, table_name: str) -> int:
    """清空单张 worth_watch_* 表。table_name 须为白名单内的物理表名。"""
    if table_name not in set(WORTH_WATCH_TABLE_BY_CATEGORY.values()):
        raise ValueError(f"unknown worth watch table: {table_name}")
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    n = int(cur.fetchone()[0] or 0)
    cur.execute(f"DELETE FROM {table_name}")
    conn.commit()
    return n


def clear_all_worth_watch_category_tables(conn: sqlite3.Connection) -> Dict[str, int]:
    """清空全部七张 worth_watch_* 表。返回各表删除前行数。"""
    out: Dict[str, int] = {}
    for tbl in sorted(set(WORTH_WATCH_TABLE_BY_CATEGORY.values())):
        out[tbl] = clear_one_worth_watch_category_table(conn, tbl)
    return out


def clear_worth_highlight_watch_table(conn: sqlite3.Connection) -> int:
    """兼容旧维护接口：清空全部 worth 分类表，返回删除总行数。"""
    parts = clear_all_worth_watch_category_tables(conn)
    return int(sum(parts.values()))


def merge_and_persist_ambush_watchlist(
    conn: sqlite3.Connection,
    ambush_dark: List[Dict[str, Any]],
    ambush_gem: List[Dict[str, Any]],
    now: datetime,
    mcap_str_fn,
) -> Dict[str, Any]:
    """
    与 heat_accum_watch 一致：每轮对「值得关注」里筛出的暗流/低市值候选做增量写入。
    按 (symbol, signal_type) 主键 upsert；首次出现记 generated_date，再次命中更新指标与 last_seen。
    不按轮次删行；仅 _ambush_watch_prune 按 generated_date 保留最近 AMBUSH_WATCH_RETENTION_DAYS 个日历日。
    调用方传入每类至多 AMBUSH_WATCH_TOP_N 条（与值得关注 🎯/💎 及入库条数一致）。
    """
    now_cst = _heat_accum_now_cst(now)
    generated_at_s = now_cst.strftime("%Y-%m-%d %H:%M")
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"

    _ambush_watch_prune(conn, now)
    cur = conn.cursor()

    def upsert(sig: Dict[str, Any], signal_type: str, summary: str) -> None:
        sym = sig.get("sym") or sig.get("symbol")
        if not sym:
            return
        sym = str(sym)
        cur.execute(
            "SELECT generated_date FROM ambush_watch WHERE symbol = ? AND signal_type = ?",
            (sym, signal_type),
        )
        ex = cur.fetchone()
        atotal = float(sig.get("total") or 0)
        row = (
            sig.get("coin"),
            now_label,
            float(sig.get("d6h") or 0),
            float(sig.get("px_chg") or 0),
            float(sig.get("est_mcap") or 0),
            atotal,
            summary,
        )
        if ex:
            cur.execute(
                """
                UPDATE ambush_watch SET
                    coin = ?, last_seen_cst = ?, d6h = ?, px_chg = ?, est_mcap = ?, ambush_total = ?, summary_line = ?
                WHERE symbol = ? AND signal_type = ?
                """,
                row + (sym, signal_type),
            )
        else:
            cur.execute(
                """
                INSERT INTO ambush_watch (
                    symbol, signal_type, coin, generated_date, last_seen_cst,
                    d6h, px_chg, est_mcap, ambush_total, summary_line
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sym,
                    signal_type,
                    sig.get("coin"),
                    generated_at_s,
                    now_label,
                    float(sig.get("d6h") or 0),
                    float(sig.get("px_chg") or 0),
                    float(sig.get("est_mcap") or 0),
                    atotal,
                    summary,
                ),
            )

    for s in ambush_dark:
        if not isinstance(s, dict):
            continue
        summ = (
            f"🎯 {s['coin']} 暗流！OI{s['d6h']:+.0f}%但价格没动，市值仅{mcap_str_fn(s['est_mcap'])}"
        )
        upsert(s, "dark_flow", summ)

    for s in ambush_gem:
        if not isinstance(s, dict):
            continue
        summ = (
            f"💎 {s['coin']} 低市值{mcap_str_fn(s['est_mcap'])}+OI{s['d6h']:+.0f}%，埋伏首选"
        )
        upsert(s, "low_mcap_oi", summ)

    conn.commit()
    print(f"  💾 暗流/低市值埋伏看盘已写入 SQLite ({DB_PATH})")
    return _ambush_watch_fetch_payload(conn, now)


# 收筹标的池参数
MIN_SIDEWAYS_DAYS = 45        # 至少横盘45天
MAX_RANGE_PCT = 80            # 横盘期价格波动<80%（宽松点，庄家盘波动可以大）
MAX_AVG_VOL_USD = 20_000_000  # 日均成交<$20M（低量才是收筹）
MIN_DATA_DAYS = 50            # 至少50天数据

# OI异动参数
MIN_OI_DELTA_PCT = 3.0        # OI变化至少3%
MIN_OI_USD = 2_000_000        # 最低OI门槛 $2M

# 放量突破参数
VOL_BREAKOUT_MULT = 3.0       # 当日Vol > 3x均值 = 放量

def api_get(endpoint, params=None):
    """币安API请求"""
    url = f"{FAPI}{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(2)
            else:
                return None
        except:
            time.sleep(1)
    return None


def _migrate_zct_settlements_drop_unique_signal_id(c: sqlite3.Cursor) -> None:
    """
    旧版 settlements 表 UNIQUE(signal_id) 与「每标的一行 signals（id 固定）」冲突：
    同一标的多次开平仓会复用同一 signal 行 id，第二次结算时 INSERT OR IGNORE 会被静默丢弃，
    导致看板「已结算」「累计盈亏」不更新。迁移为允许同一 signal_id 多条历史结算记录。
    """
    try:
        c.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='zct_vwap_settlements'"
        )
        row = c.fetchone()
        tbl_sql = (row[0] or "") if row else ""
        if not tbl_sql or "UNIQUE(signal_id)" not in tbl_sql:
            return
        c.execute("DROP TABLE IF EXISTS zct_vwap_settlements__mig")
        c.execute(
            """CREATE TABLE zct_vwap_settlements__mig (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            settled_at_utc TEXT NOT NULL,
            signal_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT,
            play TEXT,
            outcome TEXT NOT NULL,
            entry_price REAL,
            exit_price REAL,
            pnl_r REAL,
            pnl_usdt REAL,
            virtual_notional_usdt REAL
        )"""
        )
        c.execute("INSERT INTO zct_vwap_settlements__mig SELECT * FROM zct_vwap_settlements")
        c.execute("DROP TABLE zct_vwap_settlements")
        c.execute("ALTER TABLE zct_vwap_settlements__mig RENAME TO zct_vwap_settlements")
    except sqlite3.OperationalError:
        pass


def _migrate_zct_vwap_snapshot_and_settlements(c: sqlite3.Cursor) -> None:
    """
    ZCT VWAP：每标的仅保留一行当前状态；已结算记录写入 zct_vwap_settlements 供汇总/历史。
    """
    c.execute(
        """CREATE TABLE IF NOT EXISTS zct_vwap_settlements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        settled_at_utc TEXT NOT NULL,
        signal_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT,
        play TEXT,
        outcome TEXT NOT NULL,
        entry_price REAL,
        exit_price REAL,
        pnl_r REAL,
        pnl_usdt REAL,
        virtual_notional_usdt REAL
    )"""
    )
    _migrate_zct_settlements_drop_unique_signal_id(c)
    c.execute(
        "CREATE INDEX IF NOT EXISTS ix_zct_settle_symbol ON zct_vwap_settlements(symbol)"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS ix_zct_settle_time ON zct_vwap_settlements(settled_at_utc)"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS ix_zct_settle_signal_id ON zct_vwap_settlements(signal_id)"
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS zct_symbol_cooldown (
        symbol TEXT PRIMARY KEY,
        cooldown_until_ms INTEGER NOT NULL
    )"""
    )
    try:
        c.execute("SELECT COUNT(*) FROM zct_vwap_signals")
        n_sig = int(c.fetchone()[0] or 0)
        if n_sig > 0:
            c.execute("SELECT DISTINCT symbol FROM zct_vwap_signals")
            for (sym,) in c.fetchall():
                c.execute(
                    """
                    SELECT id FROM zct_vwap_signals
                    WHERE symbol = ?
                    ORDER BY
                      CASE WHEN outcome IS NULL AND side IN ('LONG', 'SHORT')
                                AND sl_price IS NOT NULL THEN 0 ELSE 1 END,
                      recorded_at_utc DESC,
                      id DESC
                    """,
                    (sym,),
                )
                ids = [r[0] for r in c.fetchall()]
                if len(ids) <= 1:
                    continue
                keep, drop = ids[0], ids[1:]
                c.execute(
                    f"DELETE FROM zct_vwap_signals WHERE id IN ({','.join('?' * len(drop))})",
                    drop,
                )
    except sqlite3.OperationalError:
        pass
    try:
        c.execute(
            """
            INSERT INTO zct_vwap_settlements (
                settled_at_utc, signal_id, symbol, side, play, outcome,
                entry_price, exit_price, pnl_r, pnl_usdt, virtual_notional_usdt
            )
            SELECT s.outcome_at_utc, s.id, s.symbol, s.side, s.play, s.outcome,
                   s.entry_price, s.exit_price, s.pnl_r, s.pnl_usdt, s.virtual_notional_usdt
            FROM zct_vwap_signals s
            WHERE s.outcome IS NOT NULL AND s.outcome_at_utc IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM zct_vwap_settlements z
                WHERE z.signal_id = s.id AND z.settled_at_utc = s.outcome_at_utc
              )
            """
        )
    except sqlite3.OperationalError:
        pass
    try:
        c.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_zct_vwap_symbol ON zct_vwap_signals(symbol)"
        )
    except sqlite3.OperationalError:
        pass


def _migrate_zct_hot_oi_merge_into_vwap_unified(c: sqlite3.Cursor) -> None:
    """
    一次性：旧版 zct_hot_oi_* 并入 zct_vwap_* 后删除热度 lane 专用表。

    合并范围（与「结算」一致）：
    1) **快照** `zct_hot_oi_signals` → `zct_vwap_signals`（仅 vwap 尚无该 symbol 的行）；
    2) **已平仓历史** `zct_hot_oi_settlements` → `zct_vwap_settlements`（`signal_id` 按 symbol
       对齐到合并后 vwap 表中的 id；按 symbol+时间+outcome+pnl 去重）；
    3) 再 DROP 两张 hot 表。之后 `resolve_open_signals_from_db` 只读写 `zct_vwap_*`。

    仅当库中仍存在 zct_hot_oi_signals 时执行；已合并则跳过（见 _zct_migration_flags）。
    """
    c.execute(
        """CREATE TABLE IF NOT EXISTS _zct_migration_flags (
            k TEXT PRIMARY KEY,
            v TEXT
        )"""
    )
    c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='zct_hot_oi_signals'"
    )
    if not c.fetchone():
        return
    c.execute("SELECT 1 FROM _zct_migration_flags WHERE k = 'hot_oi_unified_v1'")
    if c.fetchone():
        return
    c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='zct_hot_oi_settlements'"
    )
    has_hot_set = bool(c.fetchone())
    try:
        c.execute(
            """
            INSERT INTO zct_vwap_signals (
                recorded_at_utc, symbol, play, side, confidence, regime, entry_price,
                entry_bar_open_ms, sl_price, tp_price, r_unit, virtual_notional_usdt, pnl_usdt,
                vwap, vwap_upper, vwap_lower, slope_bps, band_width_pct, vwap_crosses, ma_crosses,
                chop_score, bands_wide, bands_tight, slope_steep, slope_flat,
                ref_levels_json, nearest_levels_json, reasons_json, scan_params_json,
                setup_level, vwap_cross_bucket, position_vs_vwap, outcome, outcome_at_utc,
                exit_price, pnl_r, manual_entry_price, manual_exit_price, manual_notes, notes
            )
            SELECT
                h.recorded_at_utc, h.symbol, h.play, h.side, h.confidence, h.regime, h.entry_price,
                h.entry_bar_open_ms, h.sl_price, h.tp_price, h.r_unit, h.virtual_notional_usdt, h.pnl_usdt,
                h.vwap, h.vwap_upper, h.vwap_lower, h.slope_bps, h.band_width_pct, h.vwap_crosses, h.ma_crosses,
                h.chop_score, h.bands_wide, h.bands_tight, h.slope_steep, h.slope_flat,
                h.ref_levels_json, h.nearest_levels_json, h.reasons_json, h.scan_params_json,
                h.setup_level, h.vwap_cross_bucket, h.position_vs_vwap, h.outcome, h.outcome_at_utc,
                h.exit_price, h.pnl_r, h.manual_entry_price, h.manual_exit_price, h.manual_notes, h.notes
            FROM zct_hot_oi_signals h
            WHERE NOT EXISTS (
                SELECT 1 FROM zct_vwap_signals v WHERE v.symbol = h.symbol
            )
            """
        )
        if has_hot_set:
            c.execute(
                """
                INSERT INTO zct_vwap_settlements (
                    settled_at_utc, signal_id, symbol, side, play, outcome,
                    entry_price, exit_price, pnl_r, pnl_usdt, virtual_notional_usdt
                )
                SELECT
                    h.settled_at_utc, v.id, h.symbol, h.side, h.play, h.outcome,
                    h.entry_price, h.exit_price, h.pnl_r, h.pnl_usdt, h.virtual_notional_usdt
                FROM zct_hot_oi_settlements h
                INNER JOIN zct_vwap_signals v ON v.symbol = h.symbol
                WHERE NOT EXISTS (
                    SELECT 1 FROM zct_vwap_settlements z
                    WHERE z.symbol = h.symbol
                      AND ifnull(z.settled_at_utc, '') = ifnull(h.settled_at_utc, '')
                      AND ifnull(z.outcome, '') = ifnull(h.outcome, '')
                      AND abs(ifnull(z.pnl_usdt, 0) - ifnull(h.pnl_usdt, 0)) < 0.0000001
                )
                """
            )
        if has_hot_set:
            c.execute("DROP TABLE IF EXISTS zct_hot_oi_settlements")
        c.execute("DROP TABLE IF EXISTS zct_hot_oi_signals")
        c.execute(
            "INSERT OR REPLACE INTO _zct_migration_flags (k, v) VALUES ('hot_oi_unified_v1', '1')"
        )
    except sqlite3.OperationalError:
        return


def init_db():
    """初始化数据库（WAL + busy_timeout 提升并发读写）。"""
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist (
        symbol TEXT PRIMARY KEY,
        coin TEXT,
        added_date TEXT,
        sideways_days INT,
        range_pct REAL,
        avg_vol REAL,
        low_price REAL,
        high_price REAL,
        current_price REAL,
        score REAL,
        status TEXT DEFAULT 'watching',
        last_oi_alert TEXT,
        notes TEXT
    )""")
    for _col in ("poc_price", "va_low", "va_high", "vol_breakout"):
        try:
            c.execute(f"ALTER TABLE watchlist ADD COLUMN {_col} REAL")
        except sqlite3.OperationalError:
            pass
    c.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        alert_type TEXT,
        alert_time TEXT,
        price REAL,
        oi_delta_pct REAL,
        vol_ratio REAL,
        details TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS heat_accum_watch (
        symbol TEXT PRIMARY KEY,
        coin TEXT,
        generated_date TEXT NOT NULL,
        last_seen_cst TEXT NOT NULL,
        heat REAL,
        sideways_days INTEGER,
        tags_json TEXT,
        low_price REAL,
        high_price REAL,
        price REAL,
        zone_json TEXT,
        zone_reason TEXT,
        summary_line TEXT,
        bpc_json TEXT,
        bpc_updated_cst TEXT
    )""")
    try:
        c.execute("ALTER TABLE heat_accum_watch ADD COLUMN bpc_json TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE heat_accum_watch ADD COLUMN bpc_updated_cst TEXT")
    except sqlite3.OperationalError:
        pass
    c.execute("""CREATE TABLE IF NOT EXISTS ambush_watch (
        symbol TEXT NOT NULL,
        signal_type TEXT NOT NULL,
        coin TEXT,
        generated_date TEXT NOT NULL,
        last_seen_cst TEXT NOT NULL,
        d6h REAL,
        px_chg REAL,
        est_mcap REAL,
        ambush_total REAL,
        summary_line TEXT,
        PRIMARY KEY (symbol, signal_type)
    )""")
    try:
        c.execute("ALTER TABLE ambush_watch ADD COLUMN ambush_total REAL")
    except sqlite3.OperationalError:
        pass
    c.execute("""CREATE TABLE IF NOT EXISTS patrick_core_watch (
        symbol TEXT PRIMARY KEY,
        coin TEXT,
        generated_date TEXT NOT NULL,
        last_seen_cst TEXT NOT NULL,
        sideways_days INTEGER,
        d6h REAL,
        px_chg REAL,
        est_mcap REAL,
        price REAL,
        low_price REAL,
        high_price REAL,
        summary_line TEXT
    )""")
    # 值得关注七类 · 各一张表（schema 初版一致；后续可按类 ALTER）；bpc_* 由每小时任务写入
    _worth_watch_shared_sql = """
        symbol TEXT PRIMARY KEY,
        coin TEXT,
        generated_date TEXT NOT NULL,
        last_seen_cst TEXT NOT NULL,
        rank_in_category INTEGER,
        summary_line TEXT,
        detail_json TEXT,
        bpc_json TEXT,
        bpc_updated_cst TEXT
    """
    for _tbl in sorted(set(WORTH_WATCH_TABLE_BY_CATEGORY.values())):
        c.execute(f"CREATE TABLE IF NOT EXISTS {_tbl} ({_worth_watch_shared_sql})")
    for _tbl in sorted(set(WORTH_WATCH_TABLE_BY_CATEGORY.values())):
        try:
            c.execute(f"ALTER TABLE {_tbl} ADD COLUMN bpc_json TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            c.execute(f"ALTER TABLE {_tbl} ADD COLUMN bpc_updated_cst TEXT")
        except sqlite3.OperationalError:
            pass
    try:
        c.execute("DROP TABLE IF EXISTS worth_highlight_watch")
    except sqlite3.OperationalError:
        pass
    c.execute("""CREATE TABLE IF NOT EXISTS focus_watch (
        symbol TEXT PRIMARY KEY,
        coin TEXT,
        generated_date TEXT NOT NULL,
        last_seen_cst TEXT NOT NULL,
        channel TEXT NOT NULL,
        priority INTEGER NOT NULL,
        rank_in_list INTEGER,
        summary_line TEXT,
        strategy_tip TEXT,
        detail_json TEXT,
        bpc_json TEXT,
        bpc_updated_cst TEXT
    )""")
    for _fc in ("bpc_json", "bpc_updated_cst"):
        try:
            c.execute(f"ALTER TABLE focus_watch ADD COLUMN {_fc} TEXT")
        except sqlite3.OperationalError:
            pass
    c.execute("""CREATE TABLE IF NOT EXISTS bpc_telegram_dedup (
        symbol TEXT PRIMARY KEY,
        last_cont_bar_open_ms INTEGER NOT NULL
    )""")
    # Removed Groq trade plan feature: drop legacy table if present.
    c.execute("DROP TABLE IF EXISTS ai_groq_trade_plan")
    c.execute("""CREATE TABLE IF NOT EXISTS s2_funding_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recorded_at TEXT NOT NULL,
        symbol TEXT NOT NULL,
        coin TEXT,
        price REAL,
        price_chg_24h REAL,
        prev_fr REAL,
        current_fr REAL,
        oi_change_pct REAL,
        oi_segment_avgs_json TEXT,
        volume_usd REAL,
        est_mcap_usd REAL,
        has_spot INTEGER DEFAULT 0,
        square_posts INTEGER DEFAULT 0,
        square_views INTEGER DEFAULT 0
    )""")
    c.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_s2_recorded_symbol "
        "ON s2_funding_signals(recorded_at, symbol)"
    )
    c.execute("""CREATE TABLE IF NOT EXISTS zct_vwap_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recorded_at_utc TEXT NOT NULL,
        symbol TEXT NOT NULL,
        play TEXT NOT NULL,
        side TEXT NOT NULL,
        confidence TEXT,
        regime TEXT,
        entry_price REAL NOT NULL,
        entry_bar_open_ms INTEGER,
        sl_price REAL,
        tp_price REAL,
        r_unit REAL,
        virtual_notional_usdt REAL DEFAULT 100,
        pnl_usdt REAL,
        vwap REAL,
        vwap_upper REAL,
        vwap_lower REAL,
        slope_bps REAL,
        band_width_pct REAL,
        vwap_crosses INTEGER,
        ma_crosses INTEGER,
        chop_score TEXT,
        bands_wide INTEGER NOT NULL DEFAULT 0,
        bands_tight INTEGER NOT NULL DEFAULT 0,
        slope_steep INTEGER NOT NULL DEFAULT 0,
        slope_flat INTEGER NOT NULL DEFAULT 0,
        ref_levels_json TEXT,
        nearest_levels_json TEXT,
        reasons_json TEXT,
        scan_params_json TEXT,
        setup_level INTEGER,
        vwap_cross_bucket TEXT,
        position_vs_vwap TEXT,
        outcome TEXT,
        outcome_at_utc TEXT,
        exit_price REAL,
        pnl_r REAL,
        manual_entry_price REAL,
        manual_exit_price REAL,
        manual_notes TEXT,
        notes TEXT
    )""")
    for _col, _typ in (
        ("entry_bar_open_ms", "INTEGER"),
        ("sl_price", "REAL"),
        ("tp_price", "REAL"),
        ("r_unit", "REAL"),
        ("virtual_notional_usdt", "REAL"),
        ("pnl_usdt", "REAL"),
        ("manual_entry_price", "REAL"),
        ("manual_exit_price", "REAL"),
        ("manual_notes", "TEXT"),
        ("setup_level", "INTEGER"),
        ("vwap_cross_bucket", "TEXT"),
        ("position_vs_vwap", "TEXT"),
    ):
        try:
            c.execute(f"ALTER TABLE zct_vwap_signals ADD COLUMN {_col} {_typ}")
        except sqlite3.OperationalError:
            pass
    for _ix_sql in (
        "CREATE INDEX IF NOT EXISTS ix_zct_vwap_recorded ON zct_vwap_signals(recorded_at_utc)",
        "CREATE INDEX IF NOT EXISTS ix_zct_vwap_symbol_recorded ON zct_vwap_signals(symbol, recorded_at_utc)",
        "CREATE INDEX IF NOT EXISTS ix_zct_vwap_play ON zct_vwap_signals(play)",
        "CREATE INDEX IF NOT EXISTS ix_zct_vwap_side ON zct_vwap_signals(side)",
    ):
        try:
            c.execute(_ix_sql)
        except sqlite3.OperationalError:
            pass
    _migrate_zct_vwap_snapshot_and_settlements(c)
    _migrate_zct_hot_oi_merge_into_vwap_unified(c)
    try:
        from supertrend_db import migrate_st_tables

        migrate_st_tables(c)
    except ImportError:
        pass
    conn.commit()
    _migrate_legacy_heat_accum_json(conn)
    return conn


def get_all_perp_symbols():
    """获取所有USDT永续合约"""
    info = api_get("/fapi/v1/exchangeInfo")
    if not info:
        return []
    return [s["symbol"] for s in info["symbols"]
            if s["quoteAsset"] == "USDT" 
            and s["contractType"] == "PERPETUAL"
            and s["status"] == "TRADING"]


def analyze_accumulation(symbol, klines):
    """分析单个币的收筹特征"""
    if len(klines) < MIN_DATA_DAYS:
        return None
    
    data = []
    for k in klines:
        data.append({
            "ts": k[0],
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "vol": float(k[7]),  # quote volume (USDT)
        })
    
    coin = symbol.replace("USDT", "")
    
    # === 排除稳定币和指数 ===
    EXCLUDE = {"USDC", "USDP", "TUSD", "FDUSD", "BTCDOM", "DEFI", "USDM"}
    if coin in EXCLUDE:
        return None
    
    # === 排除已经暴涨过+崩盘的币 ===
    # 最近 7 根日 K vs 更早区间的均价（非看盘「保留天数」）；涨超 300% 则跳过
    recent_7d = data[-7:]
    prior = data[:-7]
    if not prior:
        return None
    
    recent_avg_px = sum(d["close"] for d in recent_7d) / len(recent_7d)
    prior_avg_px = sum(d["close"] for d in prior) / len(prior)
    
    if prior_avg_px > 0 and ((recent_avg_px - prior_avg_px) / prior_avg_px) > 3.0:
        return None  # 已经涨了300%+，来不及了
    
    # === 寻找横盘区间 ===
    # 从最近往回找，找最长的横盘期（价格波动<MAX_RANGE_PCT%）
    # 关键：必须是真横盘（斜率接近零），阴跌不算横盘！
    best_sideways = 0
    best_range = 0
    best_low = 0
    best_high = 0
    best_avg_vol = 0
    best_slope_pct = 0
    best_window_data: Optional[List[Dict[str, Any]]] = None
    
    # 用滑动窗口从60天到全部
    for window in range(MIN_SIDEWAYS_DAYS, len(prior) + 1):
        window_data = prior[-window:]
        lows = [d["low"] for d in window_data]
        highs = [d["high"] for d in window_data]
        
        w_low = min(lows)
        w_high = max(highs)
        
        if w_low <= 0:
            continue
        
        range_pct = ((w_high - w_low) / w_low) * 100
        
        if range_pct <= MAX_RANGE_PCT:
            avg_vol = sum(d["vol"] for d in window_data) / len(window_data)
            if avg_vol <= MAX_AVG_VOL_USD:
                # 线性回归算斜率：阴跌/暴涨不算横盘
                closes = [d["close"] for d in window_data]
                n = len(closes)
                x_mean = (n - 1) / 2.0
                y_mean = sum(closes) / n
                num = sum((i - x_mean) * (c - y_mean) for i, c in enumerate(closes))
                den = sum((i - x_mean) ** 2 for i in range(n))
                slope = num / den if den > 0 else 0
                # 累计变化占起始价的百分比
                slope_pct = (slope * n / closes[0] * 100) if closes[0] > 0 else 0
                
                # 斜率过滤：累计变化超过±20%不算横盘
                if abs(slope_pct) > 20:
                    continue
                
                if window > best_sideways:
                    best_sideways = window
                    best_range = range_pct
                    best_low = w_low
                    best_high = w_high
                    best_avg_vol = avg_vol
                    best_slope_pct = slope_pct
                    best_window_data = list(window_data)
    
    if best_sideways < MIN_SIDEWAYS_DAYS:
        return None
    
    poc_price, va_low, va_high = volume_profile_from_daily_window(best_window_data or [])
    
    # === 计算收筹评分 ===
    # 横盘越久越好（庄家需要时间吸筹）
    days_score = min(best_sideways / 90, 1.0) * 25  # 90天满分25
    
    # 区间越窄越好（控盘紧）
    range_score = max(0, (1 - best_range / MAX_RANGE_PCT)) * 20  # 越窄越高，满分20
    
    # 成交量越低越好（死水一潭 = 筹码集中）
    vol_score = max(0, (1 - best_avg_vol / MAX_AVG_VOL_USD)) * 20  # 越低越高，满分20
    
    # 最近是否开始放量？（放量是启动信号）
    recent_vol = sum(d["vol"] for d in recent_7d) / len(recent_7d)
    vol_breakout = recent_vol / best_avg_vol if best_avg_vol > 0 else 0
    breakout_score = min(vol_breakout / VOL_BREAKOUT_MULT, 1.0) * 15  # 放量加分，满分15
    
    # 市值越低空间越大（核心！Patrick: 低市值=大空间）
    # 用当前价格*日均成交量/换手率来粗估市值排名
    # 实际市值在推送时用CoinGecko补充
    est_mcap = data[-1]["close"] * best_avg_vol * 30  # 粗略估算
    if est_mcap > 0 and est_mcap < 50_000_000:
        mcap_score = 20  # <$50M 满分
    elif est_mcap < 100_000_000:
        mcap_score = 15
    elif est_mcap < 200_000_000:
        mcap_score = 10
    elif est_mcap < 500_000_000:
        mcap_score = 5
    else:
        mcap_score = 0
    
    total_score = days_score + range_score + vol_score + breakout_score + mcap_score
    
    # 横盘质量加分：斜率越接近零越好（真横盘bonus，满分+5）
    flatness_bonus = max(0, (1 - abs(best_slope_pct) / 20)) * 5
    total_score += flatness_bonus
    
    # 状态判断
    if vol_breakout >= VOL_BREAKOUT_MULT:
        status = "🔥放量启动"
    elif vol_breakout >= 1.5:
        status = "⚡开始放量"
    else:
        status = "💤收筹中"
    
    return {
        "symbol": symbol,
        "coin": coin,
        "sideways_days": best_sideways,
        "range_pct": best_range,
        "slope_pct": best_slope_pct,
        "low_price": best_low,
        "high_price": best_high,
        "poc_price": poc_price,
        "va_low": va_low,
        "va_high": va_high,
        "avg_vol": best_avg_vol,
        "current_price": data[-1]["close"],
        "recent_vol": recent_vol,
        "vol_breakout": vol_breakout,
        "score": total_score,
        "status": status,
        "data_days": len(data),
    }


def scan_accumulation_pool():
    """扫描全市场，找正在被收筹的币"""
    print("📊 扫描全市场收筹标的...")
    
    symbols = get_all_perp_symbols()
    print(f"  共 {len(symbols)} 个合约")
    
    results = []
    
    for i, sym in enumerate(symbols):
        klines = api_get("/fapi/v1/klines", {
            "symbol": sym, "interval": "1d", "limit": 180
        })
        
        if klines and isinstance(klines, list):
            r = analyze_accumulation(sym, klines)
            if r:
                results.append(r)
        
        if (i + 1) % 10 == 0:
            time.sleep(0.5)
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(symbols)}... 已发现{len(results)}个")
    
    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"  ✅ 发现 {len(results)} 个收筹标的")
    return results


def scan_oi_changes(watchlist_symbols):
    """对标的池内的币扫描OI异动"""
    print(f"📊 扫描OI异动（{len(watchlist_symbols)}个标的）...")
    
    alerts = []
    
    for sym in watchlist_symbols:
        # OI历史
        oi_hist = api_get("/futures/data/openInterestHist", {
            "symbol": sym, "period": "1h", "limit": 3
        })
        
        if not oi_hist or len(oi_hist) < 2:
            continue
        
        prev_oi = float(oi_hist[-2]["sumOpenInterestValue"])
        curr_oi = float(oi_hist[-1]["sumOpenInterestValue"])
        
        if prev_oi <= 0 or curr_oi < MIN_OI_USD:
            continue
        
        delta_pct = ((curr_oi - prev_oi) / prev_oi) * 100
        
        if abs(delta_pct) >= MIN_OI_DELTA_PCT:
            # 拿当前价格
            ticker = api_get("/fapi/v1/ticker/24hr", {"symbol": sym})
            if not ticker:
                continue
            
            price = float(ticker["lastPrice"])
            vol_24h = float(ticker["quoteVolume"])
            px_chg = float(ticker["priceChangePercent"])
            
            # 拿费率
            funding = api_get("/fapi/v1/fundingRate", {"symbol": sym, "limit": 1})
            fr = float(funding[0]["fundingRate"]) if funding else 0
            
            coin = sym.replace("USDT", "")
            
            alerts.append({
                "symbol": sym,
                "coin": coin,
                "price": price,
                "oi_usd": curr_oi,
                "oi_delta_pct": delta_pct,
                "oi_delta_usd": curr_oi - prev_oi,
                "vol_24h": vol_24h,
                "px_chg_pct": px_chg,
                "funding_rate": fr,
            })
        
        time.sleep(0.3)
    
    alerts.sort(key=lambda x: abs(x["oi_delta_pct"]), reverse=True)
    print(f"  ✅ 发现 {len(alerts)} 个OI异动")
    return alerts


def format_usd(v):
    if v >= 1e9: return f"${v/1e9:.1f}B"
    if v >= 1e6: return f"${v/1e6:.1f}M"
    if v >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


def build_pool_report(results, top_n=25):
    """生成收筹标的池报告"""
    if not results:
        return ""
    
    now = datetime.now(timezone(timedelta(hours=8)))
    
    lines = [
        f"🏦 **庄家收筹雷达** — 标的池更新",
        f"⏰ {now.strftime('%Y-%m-%d %H:%M')} CST",
        f"━━━━━━━━━━━━━━━━━━",
        f"扫描 {len(results)} 个合约，发现标的：",
        "",
    ]
    
    # 分组：放量启动 > 开始放量 > 收筹中
    firing = [r for r in results if "放量启动" in r["status"]]
    warming = [r for r in results if "开始放量" in r["status"]]
    sleeping = [r for r in results if "收筹中" in r["status"]]
    
    if firing:
        lines.append(f"🔥 **放量启动** ({len(firing)}个) — 最高优先级！")
        for r in firing[:10]:
            lines.append(
                f"  🔥 **{r['coin']}** | 分:{r['score']:.0f} | "
                f"横盘{r['sideways_days']}天 | 波动{r['range_pct']:.0f}% | "
                f"Vol放大{r['vol_breakout']:.1f}x"
            )
            lines.append(
                f"     ${r['current_price']:.6f} | "
                f"区间: ${r['low_price']:.6f}~${r['high_price']:.6f} | "
                f"日均Vol: {format_usd(r['avg_vol'])}"
            )
        lines.append("")
    
    if warming:
        lines.append(f"⚡ **开始放量** ({len(warming)}个) — 关注中")
        for r in warming[:10]:
            lines.append(
                f"  ⚡ {r['coin']} | 分:{r['score']:.0f} | "
                f"横盘{r['sideways_days']}天 | 波动{r['range_pct']:.0f}% | "
                f"Vol{r['vol_breakout']:.1f}x"
            )
        lines.append("")
    
    if sleeping:
        lines.append(f"💤 **收筹中** ({len(sleeping)}个) — 持续监控")
        for r in sleeping[:15]:
            lines.append(
                f"  💤 {r['coin']} | 分:{r['score']:.0f} | "
                f"横盘{r['sideways_days']}天 | 波动{r['range_pct']:.0f}% | "
                f"日均Vol {format_usd(r['avg_vol'])}"
            )
    
    return "\n".join(lines)


def build_oi_alert_report(alerts, watchlist_coins):
    """生成OI异动报告（只报标的池内的）"""
    if not alerts:
        return ""
    
    now = datetime.now(timezone(timedelta(hours=8)))
    
    # 区分：池内 vs 池外
    in_pool = [a for a in alerts if a["symbol"] in watchlist_coins]
    out_pool = [a for a in alerts if a["symbol"] not in watchlist_coins]
    
    lines = [
        f"📊 **OI异动扫描** [收筹池]",
        f"⏰ {now.strftime('%Y-%m-%d %H:%M')} CST",
        f"━━━━━━━━━━━━━━━━━━",
        "",
    ]
    
    if in_pool:
        lines.append(f"🎯 **收筹池内异动** ({len(in_pool)}个) ⚠️ 重点关注!")
        for a in in_pool[:10]:
            emoji = "🟢" if a["oi_delta_pct"] > 0 else "🔴"
            lines.append(
                f"  {emoji} **{a['coin']}** | OI: {a['oi_delta_pct']:+.1f}% "
                f"({format_usd(a['oi_usd'])}) | 价格: {a['px_chg_pct']:+.1f}%"
            )
            # 信号解读
            if a["oi_delta_pct"] > 0 and abs(a["px_chg_pct"]) < 3:
                lines.append(f"     ⚡ 暗流涌动！OI涨但价格平 = 庄家建仓中")
            elif a["oi_delta_pct"] > 0 and a["px_chg_pct"] > 3:
                lines.append(f"     🚀 放量拉升！OI+价格同涨 = 启动中")
        lines.append("")
    
    if out_pool:
        lines.append(f"📋 池外异动 ({len(out_pool)}个)")
        for a in out_pool[:8]:
            emoji = "🟢" if a["oi_delta_pct"] > 0 else "🔴"
            lines.append(
                f"  {emoji} {a['coin']} | OI: {a['oi_delta_pct']:+.1f}% | "
                f"价格: {a['px_chg_pct']:+.1f}%"
            )
    
    return "\n".join(lines)


def send_telegram(text):
    """发送TG消息"""
    if not TG_BOT_TOKEN:
        print("\n[TG] No token, stdout:\n")
        print(text)
        return
    
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    
    # 分段发送（TG限制4096字）
    chunks = []
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
            resp = requests.post(url, json={
                "chat_id": TG_CHAT_ID,
                "text": chunk,
                "parse_mode": "Markdown"
            }, timeout=10)
            if resp.status_code == 200:
                print(f"[TG] Sent ✓ ({len(chunk)} chars)")
            else:
                # Markdown失败就用纯文本
                resp2 = requests.post(url, json={
                    "chat_id": TG_CHAT_ID,
                    "text": chunk.replace("*", "").replace("_", ""),
                }, timeout=10)
                print(f"[TG] Sent plain ({'✓' if resp2.status_code == 200 else '✗'})")
        except Exception as e:
            print(f"[TG] Error: {e}")
        time.sleep(0.5)


def save_watchlist(conn, results):
    """保存标的池到数据库（每日 pool；先按 WATCHLIST_RETENTION_DAYS 修剪落选超窗行再 UPSERT）。"""
    c = conn.cursor()
    now_cst = datetime.now(timezone(timedelta(hours=8)))
    now = now_cst.strftime("%Y-%m-%d %H:%M")
    pruned = _watchlist_prune(conn, now_cst)
    if pruned:
        print(
            f"  🧹 收筹池修剪：删除 {pruned} 行（added_date 早于最近 {WATCHLIST_RETENTION_DAYS} 个自然日）",
        )

    for r in results:
        c.execute(
            """INSERT OR REPLACE INTO watchlist 
            (symbol, coin, added_date, sideways_days, range_pct, avg_vol, 
             low_price, high_price, current_price, score, status,
             poc_price, va_low, va_high, vol_breakout)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["symbol"],
                r["coin"],
                now,
                r["sideways_days"],
                r["range_pct"],
                r["avg_vol"],
                r["low_price"],
                r["high_price"],
                r["current_price"],
                r["score"],
                r["status"],
                float(r.get("poc_price") or 0),
                float(r.get("va_low") or 0),
                float(r.get("va_high") or 0),
                float(r.get("vol_breakout") or 0),
            ),
        )
    
    conn.commit()
    print(f"  💾 保存 {len(results)} 个标的到数据库")


def load_watchlist_symbols(conn):
    """从数据库加载标的池（小时 oi 只读；超窗清理在每日 pool 的 save_watchlist 中）。"""
    c = conn.cursor()
    c.execute("SELECT symbol FROM watchlist WHERE status != 'removed'")
    return [row[0] for row in c.fetchall()]


def scan_short_fuel():
    """策略2: 空头燃料 — 涨了+费率负+OI大 = 庄家拉盘爆空单"""
    print("📊 扫描空头燃料（费率为负+在涨的币）...")
    
    tickers = api_get("/fapi/v1/ticker/24hr")
    premiums = api_get("/fapi/v1/premiumIndex")
    
    if not tickers or not premiums:
        return [], []
    
    funding_map = {p["symbol"]: float(p["lastFundingRate"]) 
                   for p in premiums if p["symbol"].endswith("USDT")}
    
    fuel_targets = []     # 已在涨+费率负 = 正在squeeze
    squeeze_targets = []  # 费率极负+还没大涨 = 潜在squeeze
    
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        
        px_chg = float(t["priceChangePercent"])
        vol = float(t["quoteVolume"])
        fr = funding_map.get(sym, 0)
        coin = sym.replace("USDT", "")
        price = float(t["lastPrice"])
        
        item = {
            "coin": coin, "symbol": sym,
            "px_chg": px_chg, "funding": fr,
            "vol": vol, "price": price,
        }
        
        # 正在squeeze: 涨>5% + 费率负 + Vol>$5M
        if px_chg > 5 and fr < -0.0003 and vol > 5_000_000:
            item["fuel_score"] = abs(fr) * 10000 * px_chg
            fuel_targets.append(item)
        
        # 潜在squeeze: 费率很负 + 还没大涨(<10%) + Vol>$2M
        elif fr < -0.0005 and px_chg < 10 and vol > 2_000_000:
            item["fuel_score"] = abs(fr) * 10000
            squeeze_targets.append(item)
    
    fuel_targets.sort(key=lambda x: x["fuel_score"], reverse=True)
    squeeze_targets.sort(key=lambda x: x["fuel_score"], reverse=True)
    
    print(f"  ✅ 正在squeeze: {len(fuel_targets)}个, 潜在squeeze: {len(squeeze_targets)}个")
    return fuel_targets, squeeze_targets


def build_fuel_report(fuel_targets, squeeze_targets):
    """生成空头燃料报告"""
    if not fuel_targets and not squeeze_targets:
        return ""
    
    now = datetime.now(timezone(timedelta(hours=8)))
    lines = [
        f"🔥 **空头燃料扫描**",
        f"⏰ {now.strftime('%Y-%m-%d %H:%M')} CST",
        f"━━━━━━━━━━━━━━━━━━",
        f"逻辑：费率负=大量做空，庄家拉盘爆空单+收资金费",
        "",
    ]
    
    if fuel_targets:
        lines.append(f"🚀 **正在Squeeze** ({len(fuel_targets)}个) — 涨了+空头还在扛")
        for t in fuel_targets[:8]:
            fr_pct = t["funding"] * 100
            flag = "🎯极度!" if fr_pct < -0.1 else "⚠️"
            lines.append(
                f"  {flag} **{t['coin']}** | 涨{t['px_chg']:+.1f}% | "
                f"费率🧊{fr_pct:.4f}% | Vol {format_usd(t['vol'])}"
            )
        lines.append("")
    
    if squeeze_targets:
        lines.append(f"🎯 **潜在Squeeze** ({len(squeeze_targets)}个) — 费率极负+还没大涨")
        for t in squeeze_targets[:8]:
            fr_pct = t["funding"] * 100
            lines.append(
                f"  🧊 {t['coin']} | 价格{t['px_chg']:+.1f}% | "
                f"费率{fr_pct:.4f}% | Vol {format_usd(t['vol'])}"
            )
    
    return "\n".join(lines)

def _focus_cutoff_iso(now: datetime) -> str:
    """含今天在内共 TOP_FOCUS_RETENTION_DAYS 个日历日。"""
    now_cst = _heat_accum_now_cst(now)
    today = now_cst.date()
    cutoff = today - timedelta(days=TOP_FOCUS_RETENTION_DAYS - 1)
    return cutoff.isoformat()


def _focus_prune(conn: sqlite3.Connection, now: datetime) -> None:
    cutoff_s = _focus_cutoff_iso(now)
    conn.execute("DELETE FROM focus_watch WHERE generated_date < ?", (cutoff_s,))


def _focus_sort_score(channel: str, d: Dict[str, Any], vb: float) -> float:
    if channel == "squeeze":
        return abs(float(d.get("fr_pct") or 0)) + max(float(d.get("d6h") or 0), 0.0)
    if channel == "volume_ignite":
        return float(vb) + float(d.get("heat") or 0) * 0.1
    return float(d.get("d6h") or 0)


def _focus_build_summary_line(coin: str, channel: str, d: Dict[str, Any], vb: float) -> str:
    lab = FOCUS_CHANNEL_LABEL_ZH[channel]
    em = FOCUS_CHANNEL_EMOJI[channel]
    if channel == "squeeze":
        return (
            f"👑 {coin} · {em}{lab} | 费率{d['fr_pct']:.2f}% OI{d['d6h']:+.0f}% 横盘{int(d.get('sw_days') or 0)}天"
        )
    if channel == "volume_ignite":
        return (
            f"👑 {coin} · {em}{lab} | Vol×{vb:.1f} 热度{float(d.get('heat') or 0):.0f} 横盘{int(d.get('sw_days') or 0)}天"
        )
    m = float(d.get("est_mcap") or 0)
    m_s = f"${m/1e6:.0f}M" if m >= 1e6 else f"${m/1e3:.0f}K"
    return (
        f"👑 {coin} · {em}{lab} | OI{d['d6h']:+.0f}% 涨跌{d['px_chg']:+.1f}% ~{m_s}"
    )


def compute_top_focus_candidates(
    coin_data: Dict[str, Dict[str, Any]],
    pool_map: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    重点关注候选：先否决（涨 + OI 大幅流出），再在收筹池内判定三通道之一；
    同一标的只保留优先级最高的一条（squeeze > volume_ignite > dark_flow）。
    """
    picked: List[Dict[str, Any]] = []
    for sym, d in coin_data.items():
        if not d.get("in_pool"):
            continue
        pd = pool_map.get(sym) or {}
        px = float(d.get("px_chg") or 0)
        d6 = float(d.get("d6h") or 0)
        if px > 0 and d6 < FOCUS_VETO_PX_POS_OI_NEG_PCT:
            continue
        st = str(pd.get("status") or "")
        vb = float(pd.get("vol_breakout") or 0)
        sw = int(d.get("sw_days") or 0)
        fr = float(d.get("fr_pct") or 0)
        heat = float(d.get("heat") or 0)
        mcap = float(d.get("est_mcap") or 0)
        hits: List[str] = []
        if sw >= FOCUS_SQUEEZE_SW_MIN and fr <= FOCUS_SQUEEZE_FR_MAX and d6 > 0:
            hits.append("squeeze")
        if abs(px) < FOCUS_DARK_ABS_PX_MAX and d6 >= FOCUS_DARK_D6H_MIN and mcap < FOCUS_DARK_MCAP_MAX_USD:
            hits.append("dark_flow")
        if FOCUS_VOL_IGNITE_STATUS_NEEDLE in st and vb >= FOCUS_VOL_IGNITE_BREAKOUT_MIN and heat > 0:
            hits.append("volume_ignite")
        if not hits:
            continue
        best_ch = min(hits, key=lambda c: FOCUS_CHANNEL_PRIORITY[c])
        pri = FOCUS_CHANNEL_PRIORITY[best_ch]
        summary = _focus_build_summary_line(str(d.get("coin") or ""), best_ch, d, vb)
        tip = FOCUS_STRATEGY_TIP_ZH.get(best_ch, "")
        detail = {
            "sym": sym,
            "coin": d.get("coin"),
            "channel": best_ch,
            "priority": pri,
            "px_chg": px,
            "d6h": d6,
            "fr_pct": fr,
            "heat": heat,
            "sw_days": sw,
            "est_mcap": mcap,
            "vol_breakout": vb,
            "pool_status": st,
            "poc_price": d.get("poc_price"),
            "va_low": d.get("va_low"),
            "va_high": d.get("va_high"),
            "liquidity_spring": d.get("liquidity_spring"),
        }
        picked.append(
            {
                "symbol": sym,
                "coin": d.get("coin"),
                "channel": best_ch,
                "priority": pri,
                "sort_score": _focus_sort_score(best_ch, d, vb),
                "summary_line": summary,
                "strategy_tip": tip,
                "detail": detail,
            }
        )
    picked.sort(key=lambda x: (int(x["priority"]), -float(x["sort_score"])))
    return picked[:TOP_FOCUS_MAX]


def _sqlite_row_to_focus_item(row: Tuple[Any, ...]) -> Dict[str, Any]:
    n = len(row)
    (
        symbol,
        coin,
        generated_date,
        last_seen_cst,
        channel,
        priority,
        rank_in_list,
        summary_line,
        strategy_tip,
        detail_json,
    ) = row[:10]
    bpc_json = row[10] if n > 10 else None
    bpc_updated_cst = row[11] if n > 11 else None
    det: Optional[Dict[str, Any]] = None
    if detail_json:
        try:
            t = json.loads(detail_json)
            det = t if isinstance(t, dict) else None
        except Exception:
            det = None
    return {
        "symbol": symbol,
        "coin": coin,
        "generated_date": generated_date,
        "last_seen_cst": last_seen_cst,
        "channel": channel,
        "channel_label_zh": FOCUS_CHANNEL_LABEL_ZH.get(str(channel), channel),
        "priority": int(priority or 0),
        "rank_in_list": int(rank_in_list or 0),
        "summary_line": summary_line,
        "strategy_tip": strategy_tip,
        "detail": det,
        "bpc": _parse_bpc_for_item(
            str(bpc_json) if bpc_json else None,
            str(bpc_updated_cst) if bpc_updated_cst else None,
        ),
    }


def _focus_fetch_payload(conn: sqlite3.Connection, now: datetime) -> Dict[str, Any]:
    now_cst = _heat_accum_now_cst(now)
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"
    cur = conn.cursor()
    cur.execute(
        """
        SELECT symbol, coin, generated_date, last_seen_cst, channel, priority,
               rank_in_list, summary_line, strategy_tip, detail_json,
               bpc_json, bpc_updated_cst
        FROM focus_watch
        ORDER BY priority ASC, rank_in_list ASC, symbol ASC
        """
    )
    rows = cur.fetchall()
    items = [_sqlite_row_to_focus_item(tuple(r)) for r in rows]
    seen_times = [it.get("last_seen_cst") for it in items if isinstance(it.get("last_seen_cst"), str)]
    updated_at = max(seen_times) if seen_times else now_label
    bpc_times: List[str] = []
    for it in items:
        b = it.get("bpc")
        if isinstance(b, dict) and b.get("evaluated_at_cst"):
            bpc_times.append(str(b["evaluated_at_cst"]))
    focus_bpc_snapshot = max(bpc_times) if bpc_times else None
    return {
        "ok": True,
        "items": items,
        "updated_at_cst": updated_at,
        "retention_days": TOP_FOCUS_RETENTION_DAYS,
        "storage": "sqlite",
        "channels": dict(FOCUS_CHANNEL_LABEL_ZH),
        "bpc_interval": HEAT_ACCUM_BPC_INTERVAL,
        "bpc_snapshot_cst": focus_bpc_snapshot,
    }


def load_focus_watchlist_from_db(
    conn: sqlite3.Connection,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    if now is None:
        now = datetime.now(timezone(timedelta(hours=8)))
    _focus_prune(conn, now)
    conn.commit()
    return _focus_fetch_payload(conn, now)


def merge_and_persist_focus_watch(
    conn: sqlite3.Connection,
    entries: List[Dict[str, Any]],
    now: datetime,
) -> Dict[str, Any]:
    """写入 focus_watch：当前轮命中列表 upsert；保留最近 TOP_FOCUS_RETENTION_DAYS 个生成日。"""
    now_cst = _heat_accum_now_cst(now)
    generated_at_s = now_cst.strftime("%Y-%m-%d %H:%M")
    now_label = now_cst.strftime("%Y-%m-%d %H:%M") + " CST"
    _focus_prune(conn, now)
    cur = conn.cursor()
    for rank, ent in enumerate(entries, start=1):
        if not isinstance(ent, dict):
            continue
        sym = str(ent.get("symbol") or "").strip()
        if not sym:
            continue
        det = ent.get("detail")
        det_s = json.dumps(det, ensure_ascii=False) if isinstance(det, dict) else None
        cur.execute("SELECT generated_date FROM focus_watch WHERE symbol = ?", (sym,))
        ex = cur.fetchone()
        row_core = (
            str(ent.get("coin") or ""),
            now_label,
            str(ent.get("channel") or ""),
            int(ent.get("priority") or 99),
            rank,
            str(ent.get("summary_line") or ""),
            str(ent.get("strategy_tip") or ""),
            det_s,
        )
        if ex:
            cur.execute(
                """
                UPDATE focus_watch SET
                    coin = ?, last_seen_cst = ?, channel = ?, priority = ?,
                    rank_in_list = ?, summary_line = ?, strategy_tip = ?, detail_json = ?
                WHERE symbol = ?
                """,
                row_core + (sym,),
            )
        else:
            cur.execute(
                """
                INSERT INTO focus_watch (
                    symbol, coin, generated_date, last_seen_cst,
                    channel, priority, rank_in_list, summary_line, strategy_tip, detail_json,
                    bpc_json, bpc_updated_cst
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    sym,
                    str(ent.get("coin") or ""),
                    generated_at_s,
                    now_label,
                    str(ent.get("channel") or ""),
                    int(ent.get("priority") or 99),
                    rank,
                    str(ent.get("summary_line") or ""),
                    str(ent.get("strategy_tip") or ""),
                    det_s,
                ),
            )
    conn.commit()
    print(f"  💾 重点关注看盘已写入 SQLite（{len(entries)} 条） ({DB_PATH})")
    return _focus_fetch_payload(conn, now)


def clear_focus_watch_table(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM focus_watch")
    n = int(cur.fetchone()[0] or 0)
    cur.execute("DELETE FROM focus_watch")
    conn.commit()
    return n


def run_oi_hourly_radar(conn: sqlite3.Connection, *, notify: bool = True) -> Dict[str, Any]:
    """综合扫描：OI + 费率 + 收筹。notify=False 时不推 Telegram，供 HTTP 刷新写快照。"""
    # === 综合扫描：OI + 费率 + 收筹 三维合一 ===
    watchlist = load_watchlist_symbols(conn)

    if not watchlist:
        print("⚠️ 标的池为空，先运行 pool 模式")
        return {"ok": False, "error": "watchlist_empty", "message": "标的池为空，先运行 pool 模式"}
    
    # 1. 拿全市场费率+行情
    tickers_raw = api_get("/fapi/v1/ticker/24hr")
    premiums_raw = api_get("/fapi/v1/premiumIndex")
    
    if not tickers_raw or not premiums_raw:
        print("❌ API失败")
        return {"ok": False, "error": "ticker_api", "message": "币安全市场行情 API 失败"}
    
    ticker_map = {}
    for t in tickers_raw:
        if t["symbol"].endswith("USDT"):
            ticker_map[t["symbol"]] = {
                "px_chg": float(t["priceChangePercent"]),
                "vol": float(t["quoteVolume"]),
                "price": float(t["lastPrice"]),
            }
    
    funding_map = {}
    for p in premiums_raw:
        if p["symbol"].endswith("USDT"):
            funding_map[p["symbol"]] = float(p["lastFundingRate"])
    
    # 1.5 拉真实流通市值（币安现货API，一次全量）
    mcap_map = {}  # coin名 -> marketCap
    try:
        import requests as _req
        _r = _req.get("https://www.binance.com/bapi/composite/v1/public/marketing/symbol/list", timeout=10)
        if _r.status_code == 200:
            for item in _r.json().get("data", []):
                name = item.get("name", "")
                mc = item.get("marketCap", 0)
                if name and mc:
                    mcap_map[name] = float(mc)
            print(f"✅ 拉到 {len(mcap_map)} 个币的真实市值")
    except Exception as e:
        print(f"⚠️ 市值API失败，走fallback: {e}")
    
    # 2. 拉热度数据（CoinGecko Trending + 成交量暴增）
    heat_map = {}  # coin名 -> heat_score (0-100)
    cg_trending = set()
    try:
        import requests as _req
        _r = _req.get("https://api.coingecko.com/api/v3/search/trending", timeout=10)
        if _r.status_code == 200:
            for item in _r.json().get("coins", []):
                sym = item["item"]["symbol"].upper()
                rank = item["item"].get("score", 99)
                cg_trending.add(sym)
                heat_map[sym] = heat_map.get(sym, 0) + max(50 - rank * 3, 10)  # top1=50分, top10=20分
            print(f"🔥 CoinGecko Trending: {len(cg_trending)}个币")
    except Exception as e:
        print(f"⚠️ CG Trending失败: {e}")
    
    # 成交量暴增检测（24hVol vs 5日均Vol）
    vol_surge_coins = set()
    for sym, tk in ticker_map.items():
        coin = sym.replace("USDT", "")
        vol_24h = tk["vol"]
        # 快速拿5天均量（用ticker的数据粗估，精确版在后面OI扫描时补充）
        # 这里先标记vol > $20M的为候选
        if vol_24h > 20_000_000:
            kl = api_get("/fapi/v1/klines", {"symbol": sym, "interval": "1d", "limit": 6})
            if kl and len(kl) >= 5:
                avg_5d = sum(float(k[7]) for k in kl[:-1]) / (len(kl)-1)
                if avg_5d > 0:
                    ratio = vol_24h / avg_5d
                    if ratio >= 2.5:  # 成交量放大2.5倍以上
                        vol_surge_coins.add(coin)
                        heat_map[coin] = heat_map.get(coin, 0) + min(ratio * 10, 50)  # 最高50分
                import time; time.sleep(0.05)
    
    print(f"📈 成交量暴增(≥2.5x): {len(vol_surge_coins)}个币")
    # 双重热度
    dual_heat = cg_trending & vol_surge_coins
    if dual_heat:
        for coin in dual_heat:
            heat_map[coin] = heat_map.get(coin, 0) + 20  # 双重信号bonus
        print(f"🔥🔥 双重热度: {dual_heat}")
    
    # 3. 从DB读收筹数据（含横盘高低点，供热度+收筹方案C多因子区间）
    c2 = conn.cursor()
    c2.execute(
        "SELECT symbol, score, sideways_days, range_pct, avg_vol, status, low_price, high_price, "
        "poc_price, va_low, va_high, vol_breakout FROM watchlist"
    )
    pool_map = {}
    for row in c2.fetchall():
        pool_map[row[0]] = {
            "pool_score": row[1],
            "sideways_days": row[2],
            "range_pct": row[3],
            "avg_vol": row[4],
            "status": row[5],
            "low_price": row[6],
            "high_price": row[7],
            "poc_price": row[8],
            "va_low": row[9],
            "va_high": row[10],
            "vol_breakout": row[11],
        }
    
    # 3. 扫OI（标的池中放量的 + Top100）
    scan_syms = set()
    for sym, pd in pool_map.items():
        if "放量" in pd.get("status", "") or "开始" in pd.get("status", ""):
            scan_syms.add(sym)
    top_by_vol = sorted(ticker_map.items(), key=lambda x: x[1]["vol"], reverse=True)[:100]
    for sym, _ in top_by_vol:
        scan_syms.add(sym)
    
    oi_map = {}
    for i, sym in enumerate(scan_syms):
        oi_hist = api_get("/futures/data/openInterestHist", {"symbol": sym, "period": "1h", "limit": 6})
        if oi_hist and len(oi_hist) >= 2:
            curr = float(oi_hist[-1]["sumOpenInterestValue"])
            prev_1h = float(oi_hist[-2]["sumOpenInterestValue"])
            prev_6h = float(oi_hist[0]["sumOpenInterestValue"])
            d1h = ((curr - prev_1h) / prev_1h * 100) if prev_1h > 0 else 0
            d6h = ((curr - prev_6h) / prev_6h * 100) if prev_6h > 0 else 0
            circ_supply = float(oi_hist[-1].get("CMCCirculatingSupply", 0))
            oi_map[sym] = {"oi_usd": curr, "d1h": d1h, "d6h": d6h, "circ_supply": circ_supply}
        if (i+1) % 10 == 0:
            import time; time.sleep(0.5)
    
    # 4. 三策略独立评分
    
    # 共用数据预处理
    all_syms = set(list(pool_map.keys()) + list(oi_map.keys()))
    coin_data = {}
    for sym in all_syms:
        tk = ticker_map.get(sym, {})
        if not tk: continue
        pool = pool_map.get(sym, {})
        oi = oi_map.get(sym, {})
        fr = funding_map.get(sym, 0)
        coin = sym.replace("USDT", "")
        
        d6h = oi.get("d6h", 0)
        fr_pct = fr * 100
        oi_usd = oi.get("oi_usd", 0)
        # 真实流通市值：优先现货API，fallback合约OI接口的CMC数据，最后粗估
        if coin in mcap_map:
            est_mcap = mcap_map[coin]
        else:
            circ_supply = oi.get("circ_supply", 0)
            price = tk.get("price", 0) if isinstance(tk, dict) else 0
            if circ_supply > 0 and price > 0:
                est_mcap = circ_supply * price
            else:
                est_mcap = max(tk["vol"] * 0.3, oi_usd * 2) if oi_usd > 0 else tk["vol"] * 0.3
        sw_days = pool.get("sideways_days", 0) if pool else 0
        pool_sc = pool.get("pool_score", 0) if pool else 0
        
        heat = heat_map.get(coin, 0)
        poc_v = pool.get("poc_price") if pool else None
        va_lo_v = pool.get("va_low") if pool else None
        va_hi_v = pool.get("va_high") if pool else None
        
        coin_data[sym] = {
            "coin": coin, "sym": sym,
            "px_chg": tk["px_chg"], "vol": tk["vol"],
            "price": tk["price"],
            "fr_pct": fr_pct, "d6h": d6h,
            "oi_usd": oi_usd, "est_mcap": est_mcap,
            "sw_days": sw_days, "pool_sc": pool_sc,
            "in_pool": bool(pool), "heat": heat,
            "in_cg": coin in cg_trending,
            "vol_surge": coin in vol_surge_coins,
            "low_price": float(pool.get("low_price") or 0) if pool else 0.0,
            "high_price": float(pool.get("high_price") or 0) if pool else 0.0,
            "poc_price": float(poc_v or 0) if pool else 0.0,
            "va_low": float(va_lo_v or 0) if pool else 0.0,
            "va_high": float(va_hi_v or 0) if pool else 0.0,
            "liquidity_spring": None,
        }
    
    enrich_liquidity_spring_batch(coin_data, pool_map)
    
    # ═══════════════════════════════════════
    # 策略1: 追多 — 纯费率排名
    # ═══════════════════════════════════════
    chase = []
    for sym, d in coin_data.items():
        if d["px_chg"] > 3 and d["fr_pct"] < -0.005 and d["vol"] > 1_000_000:
            # 查费率趋势
            fr_hist = api_get("/fapi/v1/fundingRate", {"symbol": sym, "limit": 5})
            fr_rates = [float(f["fundingRate"]) * 100 for f in fr_hist] if fr_hist else [d["fr_pct"]]
            fr_prev = fr_rates[-2] if len(fr_rates) >= 2 else d["fr_pct"]
            fr_delta = d["fr_pct"] - fr_prev
            
            trend = "🔥加速" if fr_delta < -0.05 else "⬇️变负" if fr_delta < -0.01 else "➡️" if abs(fr_delta) < 0.01 else "⬆️回升"
            
            chase.append({**d, "fr_delta": fr_delta, "trend": trend,
                          "rates": " → ".join([f"{x:.3f}" for x in fr_rates[-3:]])})
            import time; time.sleep(0.2)
    
    # 纯按费率绝对值排序（越负越前）
    chase.sort(key=lambda x: x["fr_pct"])
    
    # ═══════════════════════════════════════
    # 策略2: 综合 — 各维度均衡(各25分)
    # ═══════════════════════════════════════
    combined = []
    for sym, d in coin_data.items():
        # 费率分(25) — 越负越好
        fr = d["fr_pct"]
        if fr < -0.5: f_sc = 25
        elif fr < -0.1: f_sc = 22
        elif fr < -0.05: f_sc = 18
        elif fr < -0.03: f_sc = 14
        elif fr < -0.01: f_sc = 10
        elif fr < 0: f_sc = 5
        else: f_sc = 0
        
        # 市值分(25) — 用真实流通市值
        mc = d["est_mcap"]
        if mc > 0 and mc < 50e6: m_sc = 25
        elif mc < 100e6: m_sc = 22
        elif mc < 200e6: m_sc = 20
        elif mc < 300e6: m_sc = 17
        elif mc < 500e6: m_sc = 12
        elif mc < 1e9: m_sc = 7
        else: m_sc = 0
        
        # 横盘分(25)
        sw = d["sw_days"]
        if sw >= 120: s_sc = 25
        elif sw >= 90: s_sc = 22
        elif sw >= 75: s_sc = 18
        elif sw >= 60: s_sc = 14
        elif sw >= 45: s_sc = 10
        else: s_sc = 0
        
        # OI分(25)
        abs6 = abs(d["d6h"])
        if abs6 >= 15: o_sc = 25
        elif abs6 >= 8: o_sc = 22
        elif abs6 >= 5: o_sc = 18
        elif abs6 >= 3: o_sc = 14
        elif abs6 >= 2: o_sc = 10
        else: o_sc = 0
        
        total = f_sc + m_sc + s_sc + o_sc
        if total < 25: continue
        
        combined.append({**d, "total": total,
                        "f_sc": f_sc, "m_sc": m_sc, "s_sc": s_sc, "o_sc": o_sc})
    
    combined.sort(key=lambda x: x["total"], reverse=True)
    
    # ═══════════════════════════════════════
    # 策略3: 埋伏 — 市值>OI>横盘>费率
    # ═══════════════════════════════════════
    ambush = []
    for sym, d in coin_data.items():
        if not d["in_pool"]: continue  # 必须在收筹池
        if d["px_chg"] > 50: continue  # 已经暴涨的排除
        
        # 1.市值(35分) — 核心！越低越好（真实流通市值）
        mc = d["est_mcap"]
        if mc > 0 and mc < 50e6: m_sc = 35
        elif mc < 100e6: m_sc = 32
        elif mc < 150e6: m_sc = 28
        elif mc < 200e6: m_sc = 25
        elif mc < 300e6: m_sc = 20
        elif mc < 500e6: m_sc = 12
        elif mc < 1e9: m_sc = 5
        else: m_sc = 0
        
        # 2.OI异动(30分) — OI涨+市值低=极好
        abs6 = abs(d["d6h"])
        if abs6 >= 10: o_sc = 30
        elif abs6 >= 5: o_sc = 25
        elif abs6 >= 3: o_sc = 20
        elif abs6 >= 2: o_sc = 14
        elif abs6 >= 1: o_sc = 8
        else: o_sc = 0
        # 暗流加分：OI涨但价格平
        if d["d6h"] > 2 and abs(d["px_chg"]) < 5:
            o_sc = min(o_sc + 5, 30)
        
        # 3.横盘(20分)
        sw = d["sw_days"]
        if sw >= 120: s_sc = 20
        elif sw >= 90: s_sc = 17
        elif sw >= 75: s_sc = 14
        elif sw >= 60: s_sc = 10
        elif sw >= 45: s_sc = 6
        else: s_sc = 0
        
        # 4.负费率(15分) — 有负费率是bonus
        fr = d["fr_pct"]
        if fr < -0.1: f_sc = 15
        elif fr < -0.05: f_sc = 12
        elif fr < -0.03: f_sc = 9
        elif fr < -0.01: f_sc = 6
        elif fr < 0: f_sc = 3
        else: f_sc = 0
        
        total = m_sc + o_sc + s_sc + f_sc
        if total < 20: continue
        
        ambush.append({**d, "total": total,
                      "m_sc": m_sc, "o_sc": o_sc, "s_sc": s_sc, "f_sc": f_sc})
    
    ambush.sort(key=lambda x: x["total"], reverse=True)
    
    # ═══════════════════════════════════════
    # 5. 生成推送 + 值得关注提醒
    # ═══════════════════════════════════════
    def mcap_str(v):
        if v >= 1e6: return f"${v/1e6:.0f}M"
        if v >= 1e3: return f"${v/1e3:.0f}K"
        return f"${v:.0f}"
    
    now = datetime.now(timezone(timedelta(hours=8)))
    focus_entries = compute_top_focus_candidates(coin_data, pool_map)
    focus_watchlist_payload = merge_and_persist_focus_watch(conn, focus_entries, now)

    lines = [
        f"🏦 **庄家雷达** 三策略+热度",
        f"⏰ {now.strftime('%Y-%m-%d %H:%M')} CST",
    ]
    if focus_entries:
        lines.append("")
        lines.append("👑 **重点关注**（否决：涨且 6h OI < -5%）")
        for ent in focus_entries:
            lines.append(f"  {ent.get('summary_line') or ''}")
            tip = ent.get("strategy_tip") or ""
            if tip:
                lines.append(f"     💡 {tip}")
    
    # 表0: 热度榜（最重要，放最前面）
    hot_coins = sorted(
        [d for d in coin_data.values() if d["heat"] > 0],
        key=lambda x: x["heat"], reverse=True
    )
    if hot_coins:
        lines.append(f"\n🔥 **热度榜** (CG趋势+成交量暴增)")
        for s in hot_coins[:8]:
            tags = []
            if s["in_cg"]: tags.append("🌐CG热搜")
            if s["vol_surge"]: tags.append("📈放量")
            oi_tag = f"OI{s['d6h']:+.0f}%" if abs(s["d6h"]) >= 3 else ""
            if oi_tag: tags.append(f"⚡{oi_tag}")
            if s["in_pool"]: tags.append(f"💤池{s['sw_days']}天")
            fr_tag = f"🧊{s['fr_pct']:.2f}%" if s["fr_pct"] < -0.03 else ""
            if fr_tag: tags.append(fr_tag)
            lines.append(
                f"  {s['coin']:<8} ~{mcap_str(s['est_mcap'])} 涨{s['px_chg']:+.0f}% | {' '.join(tags)}"
            )
    
    # 表1: 追多
    lines.append(f"\n🔥 **追多** (按费率排名)")
    if chase:
        for s in chase[:8]:
            lines.append(
                f"  {s['coin']:<7} 费率{s['fr_pct']:+.3f}% {s['trend']}"
                f" | 涨{s['px_chg']:+.0f}% | ~{mcap_str(s['est_mcap'])}"
            )
    else:
        lines.append("  暂无（需涨>3%+费率负）")
    
    # 表2: 综合
    lines.append(f"\n📊 **综合** (费率+市值+横盘+OI 各25)")
    for s in combined[:8]:
        dims = []
        if s["f_sc"] >= 10: dims.append(f"🧊{s['fr_pct']:.2f}%")
        if s["m_sc"] >= 12: dims.append(f"💎{mcap_str(s['est_mcap'])}")
        if s["s_sc"] >= 10: dims.append(f"💤{s['sw_days']}天")
        if s["o_sc"] >= 10: dims.append(f"⚡OI{s['d6h']:+.0f}%")
        lines.append(
            f"  {s['coin']:<7} {s['total']}分 | {' '.join(dims)}"
        )
    
    # 表3: 埋伏
    lines.append(f"\n🎯 **埋伏** (市值35+OI30+横盘20+费率15)")
    for s in ambush[:8]:
        tags = [f"~{mcap_str(s['est_mcap'])}"]
        if abs(s["d6h"]) >= 2: tags.append(f"OI{s['d6h']:+.0f}%")
        if s["d6h"] > 2 and abs(s["px_chg"]) < 5: tags.append("🎯暗流")
        if s["sw_days"] >= 45: tags.append(f"横盘{s['sw_days']}天")
        if s["fr_pct"] < -0.01: tags.append(f"费率{s['fr_pct']:.2f}%")
        lines.append(
            f"  {s['coin']:<7} {s['total']}分 | {' '.join(tags)}"
        )
    
    # ═══ 值得关注提醒（七类 · 动态门槛 + 每类至多 MAX_K 条；入库 worth_watch_* 七表）═══
    worth_buckets: Dict[str, List[Dict[str, Any]]] = {k: [] for k in WORTH_HIGHLIGHT_CATEGORY_ORDER}
    hot_pool_signals: List[Dict[str, Any]] = []
    coin_row_by_coin = {str(d["coin"]): d for d in coin_data.values() if d.get("coin")}
    combined_by_coin = {str(x["coin"]): x for x in combined}

    def _zone_meta_for_row(s: Dict[str, Any]) -> Dict[str, Any]:
        zm: Dict[str, Any] = {}
        if float(s.get("poc_price") or 0) > 0:
            zm["poc_price"] = float(s["poc_price"])
        if float(s.get("va_low") or 0) > 0 and float(s.get("va_high") or 0) > 0:
            zm["va_low"] = float(s["va_low"])
            zm["va_high"] = float(s["va_high"])
        ls = s.get("liquidity_spring")
        if isinstance(ls, dict) and ls:
            zm["liquidity_spring"] = ls
        return zm

    # 1 热度+收筹
    hot_pool = [d for d in coin_data.values() if d["heat"] > 0 and d["in_pool"]]
    heat_pick = worth_pick_dynamic(
        sorted(hot_pool, key=lambda x: x["heat"], reverse=True),
        score_fn=lambda x: float(x["heat"]),
        score_min=WORTH_MIN_SCORE_HEAT_ACCUM,
    )
    for rank, s in enumerate(heat_pick, start=1):
        tags = []
        if s["in_cg"]:
            tags.append("CG热搜")
        if s["vol_surge"]:
            tags.append("放量")
        summary = f"🔥💤 {s['coin']} 热度({'+'.join(tags)})+收筹{s['sw_days']}天=OI将涨"
        ls = s.get("liquidity_spring") if isinstance(s.get("liquidity_spring"), dict) else {}
        if ls and ls.get("detected"):
            summary += "·弹簧"
        zm = _zone_meta_for_row(s)
        hot_pool_signals.append(
            {
                "coin": s["coin"],
                "symbol": s["sym"],
                "heat": s["heat"],
                "tags": list(tags),
                "sideways_days": s["sw_days"],
                "low_price": s["low_price"],
                "high_price": s["high_price"],
                "price": s["price"],
                "zone_meta": zm,
            }
        )
        det = {
            "heat": s["heat"],
            "d6h": s["d6h"],
            "sw_days": s["sw_days"],
            "fr_pct": s["fr_pct"],
            "est_mcap": s["est_mcap"],
            "px_chg": s["px_chg"],
            "poc_price": s.get("poc_price"),
            "va_low": s.get("va_low"),
            "va_high": s.get("va_high"),
            "liquidity_spring": s.get("liquidity_spring"),
        }
        worth_buckets["heat_accum"].append(
            {
                "symbol": s["sym"],
                "coin": s["coin"],
                "summary_line": summary,
                "rank_in_category": rank,
                "detail": det,
            }
        )

    # 2 Patrick 核心
    patrick_core_pool = [
        d for d in coin_data.values() if d["in_pool"] and abs(d["d6h"]) >= PATRICK_CORE_OI_MIN_ABS_PCT
    ]
    patrick_sorted = sorted(patrick_core_pool, key=lambda x: abs(x["d6h"]), reverse=True)

    def _patrick_comp(s: Dict[str, Any]) -> float:
        return float(s["pool_sc"]) + abs(float(s["d6h"])) * 2.5

    patrick_pick = worth_pick_dynamic(
        patrick_sorted,
        score_fn=_patrick_comp,
        score_min=WORTH_MIN_SCORE_PATRICK_COMPOSITE,
    )
    patrick_core_signals = []
    for s in patrick_pick:
        patrick_core_signals.append(
            {
                "coin": s["coin"],
                "symbol": s["sym"],
                "sideways_days": s["sw_days"],
                "d6h": s["d6h"],
                "px_chg": s["px_chg"],
                "est_mcap": s["est_mcap"],
                "price": s["price"],
                "low_price": s["low_price"],
                "high_price": s["high_price"],
            }
        )
    for rank, s in enumerate(patrick_pick, start=1):
        summary = f"📍 {s['coin']} 收筹{s['sw_days']}天+OI{s['d6h']:+.0f}%（Patrick核心）"
        ls = s.get("liquidity_spring") if isinstance(s.get("liquidity_spring"), dict) else {}
        if ls and ls.get("detected"):
            summary += "·弹簧"
        worth_buckets["patrick_core"].append(
            {
                "symbol": s["sym"],
                "coin": s["coin"],
                "summary_line": summary,
                "rank_in_category": rank,
                "detail": {
                    "sw_days": s["sw_days"],
                    "d6h": s["d6h"],
                    "px_chg": s["px_chg"],
                    "est_mcap": s["est_mcap"],
                    "poc_price": s.get("poc_price"),
                    "liquidity_spring": s.get("liquidity_spring"),
                },
            }
        )

    # 3 热度+OI
    hot_oi = [d for d in coin_data.values() if d["heat"] > 0 and d["d6h"] > 5]
    hot_oi_pick = worth_pick_dynamic(
        sorted(hot_oi, key=lambda x: x["d6h"], reverse=True),
        score_fn=lambda s: float(s["heat"]) + float(s["d6h"]) * 1.5,
        score_min=WORTH_MIN_SCORE_HOT_OI,
    )
    for rank, s in enumerate(hot_oi_pick, start=1):
        summary = f"🔥⚡ {s['coin']} 热度+OI{s['d6h']:+.0f}%双涨！"
        worth_buckets["hot_oi"].append(
            {
                "symbol": s["sym"],
                "coin": s["coin"],
                "summary_line": summary,
                "rank_in_category": rank,
                "detail": {"heat": s["heat"], "d6h": s["d6h"], "px_chg": s["px_chg"]},
            }
        )

    # 4 追多·费率加速
    chase_fire_raw = [s for s in chase[:16] if "加速" in s.get("trend", "")]
    chase_fire = worth_pick_dynamic(
        chase_fire_raw,
        score_fn=lambda s: -float(s["fr_pct"]),
        score_min=WORTH_MIN_FR_STRENGTH_CHASE_FIRE,
    )
    for rank, s in enumerate(chase_fire, start=1):
        summary = f"🔥 {s['coin']} 费率{s['fr_pct']:.3f}%加速恶化，空头涌入中"
        worth_buckets["chase_fire"].append(
            {
                "symbol": s["sym"],
                "coin": s["coin"],
                "summary_line": summary,
                "rank_in_category": rank,
                "detail": {
                    "fr_pct": s["fr_pct"],
                    "px_chg": s["px_chg"],
                    "trend": s.get("trend"),
                },
            }
        )

    # 5 追多+综合双榜（按综合分排序，不再纯字母序）
    chase_coins = set(s["coin"] for s in chase[:12])
    combined_coins = set(s["coin"] for s in combined[:12])
    overlap_2 = chase_coins & combined_coins
    overlap_rows: List[Dict[str, Any]] = []
    for c in overlap_2:
        row = coin_row_by_coin.get(str(c))
        comb = combined_by_coin.get(str(c))
        if row and comb:
            overlap_rows.append(
                {
                    "sym": row["sym"],
                    "coin": c,
                    "px_chg": row["px_chg"],
                    "d6h": row["d6h"],
                    "est_mcap": row["est_mcap"],
                    "combined_total": float(comb["total"]),
                }
            )
    overlap_rows.sort(key=lambda x: x["combined_total"], reverse=True)
    dual_pick = worth_pick_dynamic(
        overlap_rows,
        score_fn=lambda r: float(r["combined_total"]),
        score_min=WORTH_MIN_COMBINED_TOTAL_DUAL,
    )
    for rank, row in enumerate(dual_pick, start=1):
        c = row["coin"]
        summary = f"⭐ {c} 追多+综合双榜上榜"
        worth_buckets["dual_list"].append(
            {
                "symbol": row["sym"],
                "coin": c,
                "summary_line": summary,
                "rank_in_category": rank,
                "detail": {
                    "px_chg": row["px_chg"],
                    "d6h": row["d6h"],
                    "est_mcap": row["est_mcap"],
                    "combined_total": row["combined_total"],
                },
            }
        )

    # 6 / 7 埋伏暗流、低市值+OI（与 ambush_watch 写入同源）
    ambush_dark_all = [s for s in ambush if s["d6h"] > 2 and abs(s["px_chg"]) < 5]
    ambush_dark = worth_pick_dynamic(
        ambush_dark_all,
        score_fn=lambda s: float(s["total"]),
        score_min=WORTH_MIN_AMBUSH_TOTAL,
    )
    for rank, s in enumerate(ambush_dark, start=1):
        summary = f"🎯 {s['coin']} 暗流！OI{s['d6h']:+.0f}%但价格没动，市值仅{mcap_str(s['est_mcap'])}"
        worth_buckets["ambush_dark"].append(
            {
                "symbol": s["sym"],
                "coin": s["coin"],
                "summary_line": summary,
                "rank_in_category": rank,
                "detail": {
                    "d6h": s["d6h"],
                    "px_chg": s["px_chg"],
                    "est_mcap": s["est_mcap"],
                    "ambush_total": s.get("total"),
                },
            }
        )

    ambush_gem_all = [s for s in ambush if s["est_mcap"] < 100e6 and abs(s["d6h"]) >= 3]
    ambush_gem = worth_pick_dynamic(
        ambush_gem_all,
        score_fn=lambda s: float(s["total"]),
        score_min=WORTH_MIN_AMBUSH_TOTAL,
    )
    for rank, s in enumerate(ambush_gem, start=1):
        summary = f"💎 {s['coin']} 低市值{mcap_str(s['est_mcap'])}+OI{s['d6h']:+.0f}%，埋伏首选"
        worth_buckets["ambush_gem"].append(
            {
                "symbol": s["sym"],
                "coin": s["coin"],
                "summary_line": summary,
                "rank_in_category": rank,
                "detail": {
                    "d6h": s["d6h"],
                    "px_chg": s["px_chg"],
                    "est_mcap": s["est_mcap"],
                    "ambush_total": s.get("total"),
                },
            }
        )

    highlights: List[str] = []
    for cat in WORTH_HIGHLIGHT_CATEGORY_ORDER:
        for ent in worth_buckets[cat]:
            highlights.append(str(ent.get("summary_line") or ""))

    highlights = highlights[:WORTH_HIGHLIGHTS_MAX]
    if highlights:
        lines.append(f"\n💡 **值得关注**")
        for h in highlights:
            lines.append(f"  {h}")
    
    # 图例说明
    lines.append(f"\n📖 **图例**")
    lines.append("  🔥热度=CG热搜+成交量暴增(OI领先指标)")
    lines.append("  费率负=空头燃料 | 💎市值 | 💤横盘(收筹)")
    lines.append("  🔥💤热度+收筹=最强预判 | 🔥⚡热度+OI=正在发生")
    lines.append("  📍收筹池+OI异动=Patrick核心（可无热度）")
    lines.append("  👑重点关注=逼空/天量/暗流三通道+否决假上涨（详见段首）")
    
    report = "\n".join(lines)
    # 🎯 暗流 / 💎 低市值+OI：与上文 highlights 中对应条目同源，每类至多 AMBUSH_WATCH_TOP_N 条写入 ambush_watch
    ambush_watchlist = merge_and_persist_ambush_watchlist(
        conn, ambush_dark, ambush_gem, now, mcap_str
    )
    heat_accum_watchlist = merge_and_persist_heat_accum_watchlist(conn, hot_pool_signals, now)
    patrick_core_watchlist = merge_and_persist_patrick_core_watchlist(conn, patrick_core_signals, now)
    worth_highlight_watchlist = merge_and_persist_worth_highlight_watchlist(conn, worth_buckets, now)
    payload = {
        "ok": True,
        "generated_at_cst": now.strftime("%Y-%m-%d %H:%M") + " CST",
        "highlights": highlights,
        "hot_pool_signals": hot_pool_signals,
        "heat_accum_watchlist": heat_accum_watchlist,
        "ambush_watchlist": ambush_watchlist,
        "patrick_core_watchlist": patrick_core_watchlist,
        "worth_highlight_watchlist": worth_highlight_watchlist,
        "focus_watchlist": focus_watchlist_payload,
        "report_markdown": report,
        "hot_coins": hot_coins[:16],
        "chase": chase[:16],
        "combined": combined[:16],
        "ambush": ambush[:16],
        "coin_data": list(coin_data.values()),
    }
    _persist_oi_radar_snapshot(payload)
    if notify:
        if TELEGRAM_SEND_LEGACY_OI_HOURLY_REPORT:
            send_telegram(report)
    return payload


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    
    print(f"🏦 庄家收筹雷达 v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   模式: {mode}\n")
    
    conn = init_db()
    
    if mode in ("full", "pool"):
        # === 模块A: 更新收筹标的池 ===
        results = scan_accumulation_pool()
        
        if results:
            save_watchlist(conn, results)
            report = build_pool_report(results)
            if report and TELEGRAM_SEND_LEGACY_POOL_SCAN_REPORT:
                send_telegram(report)
        else:
            now_cst = datetime.now(timezone(timedelta(hours=8)))
            n_pr = _watchlist_prune(conn, now_cst)
            if n_pr:
                print(
                    f"  🧹 收筹池修剪：删除 {n_pr} 行（本日 pool 无新入选；"
                    f"added_date 早于最近 {WATCHLIST_RETENTION_DAYS} 个自然日）",
                )
    
    if mode in ("full", "oi"):
        run_oi_hourly_radar(conn, notify=True)
    
    conn.close()
    print("\n✅ 完成")


if __name__ == "__main__":
    main()

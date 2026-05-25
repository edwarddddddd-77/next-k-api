#!/usr/bin/env python3
"""
VP Regime Scanner — 量价环境扫描（独立脚本，与 ZCT VWAP 扫描并行）

依据 Koroush「Volume Analysis Masterclass」可规则化部分实现（非投资建议）：

1. **VolUSD 代理**：优先使用币安 U 本位 K 线第 8 列 *Quote asset volume*（USDT 计价成交额）；
   若无则退化为 ``close * volume``。
2. **流动性门**：``VolUSD`` 的 ``VP_VOL_MA_PERIOD`` 周期均线（默认 **5m×12≈1h**；1m 时为 60）；
   若 ``VP_MIN_VOL_USD_MA`` > 0 且均线低于阈值 → ``scheme=NO_TRADE``（仅执行质量过滤）。
3. **三态量环境**（在**最后一根已收盘** K 上判定，避免用未收盘根；默认 **5m**）：
   - **spike_price_spike**：当根量相对均量倍数 + 当根振幅占收盘比 ≥ 阈值 → 偏「耗尽/反转观察」
   - **increasing**：近窗内成交量阶梯抬升 → 偏「动量/延续语境」
   - **flat**：近窗成交量变异系数低 → 偏「均衡/均值回归语境」
   - **mixed**：介于之间

输出 ``scheme``：``MOMENTUM`` | ``MEAN_REVERT`` | ``REVERSAL_WATCH`` | ``WATCH`` | ``NO_TRADE``

用法：
  python vp_regime_scanner.py
  python vp_regime_scanner.py --watchlist --max-watchlist 40   # 收筹池每轮最多 40 个
  python vp_regime_scanner.py --no-db
  python vp_regime_scanner.py --no-tg

环境变量（节选）：
  VP_WATCHLIST_UNIVERSE   设为 1|true|on 时，标的来自 accumulation.db **收筹池**表 ``watchlist``
                          （``status != 'removed'`` 且代码以 USDT 结尾），不再使用 VP_SYMBOLS 默认列表；
                          池为空时本轮跳过并打 [warn]（需先跑收筹雷达每日 pool 写入 watchlist）
  VP_WATCHLIST_MAX_SYMBOLS  收筹池模式下每轮最多请求 K 线的标的数；**默认 60**；按 ``VP_WATCHLIST_ORDER`` 排序后截断。
                          设为 **0** 表示**不限制**（池很大时会慢、易触发 API 限频）。
  VP_WATCHLIST_ORDER        ``score_desc``（默认，按 pool score 从高到低）| ``alpha``（按 symbol 字母序）
  VP_SYMBOLS              逗号分隔永续标的（仅当未开启 VP_WATCHLIST_UNIVERSE 时使用）
  VP_INTER_SYMBOL_SLEEP_SEC  每标的拉完 K 线后的基础休眠（秒），默认 0.05
  VP_INTER_SYMBOL_JITTER_SEC 在基础休眠上追加 ``Uniform(0, jitter)`` 秒，默认 **0.04**，打散固定节拍
  VP_API_ORDER_SHUFFLE    默认 **1|on**：请求币安前**随机打乱**本轮标的顺序（不改变入选集合）；0|false|off 关闭
  VP_SHUFFLE_SEED         可选整数，设置后 shuffle **可复现**；不设则每轮顺序不同
  VP_KLINE_INTERVAL       K 线周期，默认 5m（可选 1m）
  VP_KLINE_LIMIT          拉取根数；未设时 5m 默认 150、1m 默认 300
  VP_VOL_MA_PERIOD        均量根数；未设时 5m 默认 12（≈1h）、1m 默认 60
  VP_MIN_VOL_USD_MA       均量门槛（USDT），默认 100000；设为 0 关闭流动性门
  VP_SPIKE_VOL_MULT       当根 vol_usd / vol_ma ≥ 该值参与 spike 判定，默认 2.5
  VP_SPIKE_RANGE_MIN_PCT  当根 (high-low)/close 最小占价比，默认 0.004
  VP_SPIKE_BURST_MULT     当根 vol / mean(前 5 根 vol) ≥ 该值，默认 1.8
  VP_INCREASE_LOOKBACK    递增判定窗口长度，默认 7
  VP_INCREASE_MIN_UP      窗口内 vol[i]>vol[i-1] 最少次数，默认 4
  VP_FLAT_LOOKBACK        平坦判定窗口，默认 12
  VP_FLAT_CV_MAX          std/mean 上限判 flat，默认 0.22
  VP_DB_SKIP_NO_TRADE     1 时入库跳过仅因流动性失败的行（减轻噪音）
  VP_DB_TABLE             默认 vp_regime_snapshots
  TG_BOT_TOKEN / TG_CHAT_ID  与 accumulation 雷达相同；``use_tg`` 为真时推送（API 默认 ``notify_tg=true``；长文按行拆多条）

定时：可自行 cron / APScheduler 调用本脚本；默认不写入 main.py。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# 兼容 Windows 控制台
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

_env_oi = Path(__file__).resolve().parent / ".env.oi"
if _env_oi.is_file():
    with open(_env_oi, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

FAPI = "https://fapi.binance.com"

_DEFAULT_VP_SYMBOLS = (
    "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,ADAUSDT,1000SHIBUSDT,1000PEPEUSDT,"
    "DOGEUSDT,BNBUSDT,LINKUSDT,GALAUSDT,LTCUSDT,BCHUSDT,SUIUSDT,"
    "DOTUSDT,UNIUSDT,AVAXUSDT,AXSUSDT,MANAUSDT,ZECUSDT,TAOUSDT,ONDOUSDT"
)


def _vp_watchlist_universe() -> bool:
    """是否从收筹池 watchlist 取标的（每次调用读环境，便于 main 里 argparse 先写入）。"""
    raw = os.getenv("VP_WATCHLIST_UNIVERSE", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _watchlist_max_symbols() -> int:
    """收筹池每轮最大扫描数；0=不限制。"""
    try:
        v = int(os.getenv("VP_WATCHLIST_MAX_SYMBOLS", "60").strip() or "60")
    except ValueError:
        v = 60
    return max(0, v)


def _watchlist_order_clause() -> str:
    raw = os.getenv("VP_WATCHLIST_ORDER", "score_desc").strip().lower()
    if raw in ("alpha", "symbol", "alphabetical"):
        return "ORDER BY UPPER(TRIM(symbol)) ASC"
    return "ORDER BY COALESCE(score, 0) DESC, UPPER(TRIM(symbol)) ASC"


def _symbols_from_watchlist() -> Tuple[List[str], int]:
    """收筹池：按 score 排序后可选截断。返回 (本轮标的列表, 池内 USDT 永续总数)。"""
    from accumulation_radar import init_db

    order_sql = _watchlist_order_clause()
    conn = init_db()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT symbol FROM watchlist
            WHERE status != 'removed'
              AND UPPER(TRIM(symbol)) LIKE '%USDT'
            {order_sql}
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    seen: set[str] = set()
    ordered: List[str] = []
    for (sym,) in rows:
        u = str(sym).strip().upper()
        if not u.endswith("USDT"):
            continue
        if u in seen:
            continue
        seen.add(u)
        ordered.append(u)

    pool_n = len(ordered)
    cap = _watchlist_max_symbols()
    if cap > 0 and len(ordered) > cap:
        print(
            f"[info] 收筹池 USDT 永续共 {pool_n} 个，VP_WATCHLIST_MAX_SYMBOLS={cap}，"
            f"本轮扫描前 {cap} 个（排序：{os.getenv('VP_WATCHLIST_ORDER', 'score_desc').strip() or 'score_desc'}）",
            flush=True,
        )
        ordered = ordered[:cap]

    return ordered, pool_n


def _symbols_from_env() -> Tuple[List[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {"universe": "default"}
    if _vp_watchlist_universe():
        meta["universe"] = "watchlist"
        w, pool_n = _symbols_from_watchlist()
        meta["watchlist_pool_usdt"] = pool_n
        meta["watchlist_scanned"] = len(w)
        meta["watchlist_max"] = _watchlist_max_symbols()
        if w:
            return w, meta
        print(
            "[warn] VP_WATCHLIST_UNIVERSE：watchlist 无有效 USDT 永续标的，本轮跳过扫描",
            flush=True,
        )
        return [], meta
    raw = os.getenv("VP_SYMBOLS", _DEFAULT_VP_SYMBOLS).strip()
    parts = [x.strip().upper() for x in raw.split(",") if x.strip()]
    out = parts or [x.strip() for x in _DEFAULT_VP_SYMBOLS.split(",") if x.strip()]
    return out, meta


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default)).strip() or default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip() or default)
    except ValueError:
        return default


def _env_int_optional(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def _interval_defaults(interval: str) -> Tuple[int, int]:
    """返回 (vol_ma_period, kline_limit) 未显式 env 时的默认值。"""
    iv = (interval or "5m").strip().lower()
    if iv == "1m":
        return 60, 300
    return 12, 150


@dataclass
class VPSettings:
    interval: str
    kline_limit: int
    vol_ma_period: int
    min_vol_usd_ma: float
    spike_vol_mult: float
    spike_range_min_pct: float
    spike_burst_mult: float
    increase_lookback: int
    increase_min_up: int
    flat_lookback: int
    flat_cv_max: float


def load_vp_settings(*, interval: Optional[str] = None) -> VPSettings:
    iv = (interval or os.getenv("VP_KLINE_INTERVAL", "5m") or "5m").strip().lower()
    def_ma, def_lim = _interval_defaults(iv)
    vol_ma = _env_int_optional("VP_VOL_MA_PERIOD") or def_ma
    klim = _env_int_optional("VP_KLINE_LIMIT") or max(def_lim, vol_ma + 30)
    return VPSettings(
        interval=iv,
        kline_limit=klim,
        vol_ma_period=vol_ma,
        min_vol_usd_ma=_env_float("VP_MIN_VOL_USD_MA", 100_000.0),
        spike_vol_mult=_env_float("VP_SPIKE_VOL_MULT", 2.5),
        spike_range_min_pct=_env_float("VP_SPIKE_RANGE_MIN_PCT", 0.004),
        spike_burst_mult=_env_float("VP_SPIKE_BURST_MULT", 1.8),
        increase_lookback=_env_int("VP_INCREASE_LOOKBACK", 7),
        increase_min_up=_env_int("VP_INCREASE_MIN_UP", 4),
        flat_lookback=_env_int("VP_FLAT_LOOKBACK", 12),
        flat_cv_max=_env_float("VP_FLAT_CV_MAX", 0.22),
    )


SETTINGS = load_vp_settings()
KLINE_INTERVAL = SETTINGS.interval
KLINE_LIMIT = SETTINGS.kline_limit
VOL_MA_PERIOD = SETTINGS.vol_ma_period
MIN_VOL_USD_MA = SETTINGS.min_vol_usd_ma
SPIKE_VOL_MULT = SETTINGS.spike_vol_mult
SPIKE_RANGE_MIN_PCT = SETTINGS.spike_range_min_pct
SPIKE_BURST_MULT = SETTINGS.spike_burst_mult
INCREASE_LOOKBACK = SETTINGS.increase_lookback
INCREASE_MIN_UP = SETTINGS.increase_min_up
FLAT_LOOKBACK = SETTINGS.flat_lookback
FLAT_CV_MAX = SETTINGS.flat_cv_max
INTER_SYMBOL_SLEEP_SEC = float(os.getenv("VP_INTER_SYMBOL_SLEEP_SEC", "0.05") or 0.05)
try:
    INTER_SYMBOL_JITTER_SEC = float(os.getenv("VP_INTER_SYMBOL_JITTER_SEC", "0.04") or 0.04)
except ValueError:
    INTER_SYMBOL_JITTER_SEC = 0.04
INTER_SYMBOL_JITTER_SEC = max(0.0, INTER_SYMBOL_JITTER_SEC)

_API_SHUFFLE_RAW = os.getenv("VP_API_ORDER_SHUFFLE", "1").strip().lower()
API_ORDER_SHUFFLE_ENABLED = _API_SHUFFLE_RAW not in ("0", "false", "no", "off", "disabled")
DB_TABLE = os.getenv("VP_DB_TABLE", "vp_regime_snapshots").strip() or "vp_regime_snapshots"
_DB_SKIP_RAW = os.getenv("VP_DB_SKIP_NO_TRADE", "").strip().lower()
DB_SKIP_NO_TRADE = _DB_SKIP_RAW in ("1", "true", "yes", "on")

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()


def _shuffle_symbol_list(symbols: List[str]) -> Tuple[List[str], bool]:
    """拷贝并可选随机打乱本轮 REST 调用顺序（不改变入选集合）。"""
    out = list(symbols)
    if not API_ORDER_SHUFFLE_ENABLED or len(out) < 2:
        return out, False
    seed_raw = os.getenv("VP_SHUFFLE_SEED", "").strip()
    if seed_raw:
        try:
            rng = random.Random(int(seed_raw))
            rng.shuffle(out)
        except ValueError:
            random.shuffle(out)
    else:
        random.shuffle(out)
    return out, True


def _inter_symbol_sleep_with_jitter() -> None:
    """基础间隔 + 均匀抖动，避免多实例/定时任务同相位打满权重。"""
    base = max(0.0, INTER_SYMBOL_SLEEP_SEC)
    j = INTER_SYMBOL_JITTER_SEC
    if base <= 0.0 and j <= 0.0:
        return
    time.sleep(base + random.uniform(0.0, j))


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


def klines_to_df(rows: List[List[Any]]) -> pd.DataFrame:
    """含 quote_asset_volume（列 7）时写入 vol_usd；否则用 close*volume。"""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    base = df[[0, 1, 2, 3, 4, 5]].copy()
    base.columns = ["open_time", "open", "high", "low", "close", "volume"]
    for c in ("open", "high", "low", "close", "volume"):
        base[c] = base[c].astype(float)
    base["open_time"] = base["open_time"].astype(np.int64)
    if df.shape[1] >= 8:
        qv = df.iloc[:, 7].astype(float)
        base["vol_usd"] = qv
    else:
        base["vol_usd"] = base["close"] * base["volume"]
    base["range_pct"] = np.where(
        base["close"] > 0, (base["high"] - base["low"]) / base["close"], 0.0
    )
    return base


@dataclass
class VPRegimeResult:
    symbol: str
    bar_open_ms: int
    close: float
    vol_usd: float
    vol_usd_ma: float
    liquidity_ok: bool
    vol_pattern: str
    scheme: str
    detail: Dict[str, Any]


def _classify_on_closed(
    df_closed: pd.DataFrame, settings: Optional[VPSettings] = None
) -> Optional[VPRegimeResult]:
    """
    df_closed：已去掉最后一根未完成 K；最后一行 = 信号根（已收盘）。
    """
    s = settings or SETTINGS
    if df_closed.empty or len(df_closed) < s.vol_ma_period + 5:
        return None

    v = df_closed["vol_usd"].astype(float)
    ma = v.rolling(s.vol_ma_period, min_periods=s.vol_ma_period).mean()
    sig_idx = len(df_closed) - 1
    vol_sig = float(v.iloc[sig_idx])
    vol_ma = float(ma.iloc[sig_idx])
    if not np.isfinite(vol_ma) or vol_ma <= 0:
        return None

    liquidity_ok = s.min_vol_usd_ma <= 0 or vol_ma >= s.min_vol_usd_ma

    hi = float(df_closed["high"].iloc[sig_idx])
    lo = float(df_closed["low"].iloc[sig_idx])
    cl = float(df_closed["close"].iloc[sig_idx])
    range_pct = float(df_closed["range_pct"].iloc[sig_idx])

    start_i = max(0, sig_idx - 4)
    prev_mean = float(v.iloc[start_i:sig_idx].mean()) if sig_idx > start_i else float(v.iloc[max(0, sig_idx - 1)])
    burst = (vol_sig / max(prev_mean, 1e-9)) if prev_mean > 0 else 0.0
    vol_vs_ma = vol_sig / vol_ma

    is_spike = (
        vol_vs_ma >= s.spike_vol_mult
        and range_pct >= s.spike_range_min_pct
        and burst >= s.spike_burst_mult
    )

    lo_inc = max(0, sig_idx - (s.increase_lookback - 1))
    seg = v.iloc[lo_inc : sig_idx + 1]
    up_count = 0
    if len(seg) >= 2:
        arr = seg.values
        for i in range(1, len(arr)):
            if arr[i] > arr[i - 1] * 0.98:
                up_count += 1
    is_increasing = up_count >= s.increase_min_up and not is_spike

    lo_f = max(0, sig_idx - (s.flat_lookback - 1))
    flat_seg = v.iloc[lo_f : sig_idx + 1]
    cv = 1.0
    if len(flat_seg) >= 3:
        m = float(flat_seg.mean())
        flat_std = float(flat_seg.std())
        cv = (flat_std / m) if m > 1e-9 else 1.0
    is_flat = cv <= s.flat_cv_max and not is_spike

    if is_spike:
        pattern = "spike_price_spike"
    elif is_increasing:
        pattern = "increasing"
    elif is_flat:
        pattern = "flat"
    else:
        pattern = "mixed"

    if not liquidity_ok:
        scheme = "NO_TRADE"
    elif pattern == "spike_price_spike":
        scheme = "REVERSAL_WATCH"
    elif pattern == "increasing":
        scheme = "MOMENTUM"
    elif pattern == "flat":
        scheme = "MEAN_REVERT"
    else:
        scheme = "WATCH"

    detail = {
        "vol_vs_ma": round(vol_vs_ma, 4),
        "range_pct": round(range_pct, 6),
        "burst_vs_prev5_mean": round(burst, 4),
        "increase_up_count": up_count,
        "flat_cv": round(cv, 6),
        "min_vol_usd_ma_threshold": s.min_vol_usd_ma,
        "kline_interval": s.interval,
    }

    return VPRegimeResult(
        symbol="",
        bar_open_ms=int(df_closed["open_time"].iloc[sig_idx]),
        close=cl,
        vol_usd=vol_sig,
        vol_usd_ma=vol_ma,
        liquidity_ok=liquidity_ok,
        vol_pattern=pattern,
        scheme=scheme,
        detail=detail,
    )


def analyze_symbol_vp(
    symbol: str, *, interval: Optional[str] = None
) -> Optional[VPRegimeResult]:
    s = load_vp_settings(interval=interval)
    rows = fetch_klines(
        symbol, s.interval, max(s.kline_limit, s.vol_ma_period + 30)
    )
    if len(rows) < 2:
        return None
    df = klines_to_df(rows)
    if df.empty:
        return None
    df_closed = df.iloc[:-1].copy()
    res = _classify_on_closed(df_closed, s)
    if res is None:
        return None
    res.symbol = symbol
    return res


def analyze_symbol(symbol: str) -> Optional[VPRegimeResult]:
    """使用 ``VP_KLINE_INTERVAL``（默认 5m）。"""
    return analyze_symbol_vp(symbol, interval=None)


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {DB_TABLE} (
            symbol TEXT PRIMARY KEY,
            updated_at_utc TEXT NOT NULL,
            bar_open_ms INTEGER NOT NULL,
            close REAL,
            vol_usd REAL,
            vol_usd_ma REAL,
            liquidity_ok INTEGER NOT NULL,
            vol_pattern TEXT NOT NULL,
            scheme TEXT NOT NULL,
            detail_json TEXT
        )
        """
    )
    conn.commit()


def upsert_row(conn: sqlite3.Connection, r: VPRegimeResult) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        f"""
        INSERT INTO {DB_TABLE} (
            symbol, updated_at_utc, bar_open_ms, close, vol_usd, vol_usd_ma,
            liquidity_ok, vol_pattern, scheme, detail_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(symbol) DO UPDATE SET
            updated_at_utc=excluded.updated_at_utc,
            bar_open_ms=excluded.bar_open_ms,
            close=excluded.close,
            vol_usd=excluded.vol_usd,
            vol_usd_ma=excluded.vol_usd_ma,
            liquidity_ok=excluded.liquidity_ok,
            vol_pattern=excluded.vol_pattern,
            scheme=excluded.scheme,
            detail_json=excluded.detail_json
        """,
        (
            r.symbol,
            now,
            r.bar_open_ms,
            r.close,
            r.vol_usd,
            r.vol_usd_ma,
            1 if r.liquidity_ok else 0,
            r.vol_pattern,
            r.scheme,
            json.dumps(r.detail, ensure_ascii=False),
        ),
    )


def maybe_send_tg(lines: List[str], *, max_message_chars: int = 4000) -> None:
    """发送 Telegram；过长时按行拆成多条（避免超 4096 上限）。"""
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    sep = 1  # newline between lines
    for line in lines:
        ln = len(line) + (sep if cur else 0)
        if cur and cur_len + ln > max_message_chars:
            chunks.append("\n".join(cur))
            cur = [line]
            cur_len = len(line)
        else:
            cur.append(line)
            cur_len += ln
    if cur:
        chunks.append("\n".join(cur))
    flat_chunks: List[str] = []
    for ch in chunks:
        c = ch
        while len(c) > max_message_chars:
            flat_chunks.append(c[:max_message_chars])
            c = c[max_message_chars:]
        if c.strip():
            flat_chunks.append(c)
    chunks = flat_chunks
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    for i, text in enumerate(chunks):
        if not text.strip():
            continue
        try:
            requests.post(
                url,
                json={
                    "chat_id": TG_CHAT_ID,
                    "text": text[:max_message_chars],
                    "disable_web_page_preview": True,
                },
                timeout=20,
            )
        except Exception:
            pass
        if i < len(chunks) - 1:
            time.sleep(0.35)


def resolve_scan_symbols(
    *,
    symbols_override: Optional[List[str]] = None,
    watchlist_request: Optional[bool] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """决定本轮扫描标的：显式列表 > watchlist 开关 > 环境变量默认。"""
    if symbols_override is not None:
        syms = [
            str(x).strip().upper()
            for x in symbols_override
            if x is not None and str(x).strip()
        ]
        syms = [s for s in syms if s.endswith("USDT")]
        return syms, {
            "universe": "symbols_override",
            "watchlist_pool_usdt": None,
            "watchlist_scanned": len(syms),
            "watchlist_max": 0,
        }
    if watchlist_request is True:
        bak = os.environ.get("VP_WATCHLIST_UNIVERSE")
        os.environ["VP_WATCHLIST_UNIVERSE"] = "1"
        try:
            return _symbols_from_env()
        finally:
            if bak is None:
                os.environ.pop("VP_WATCHLIST_UNIVERSE", None)
            else:
                os.environ["VP_WATCHLIST_UNIVERSE"] = bak
    if watchlist_request is False:
        bak = os.environ.get("VP_WATCHLIST_UNIVERSE")
        os.environ.pop("VP_WATCHLIST_UNIVERSE", None)
        try:
            return _symbols_from_env()
        finally:
            if bak is not None:
                os.environ["VP_WATCHLIST_UNIVERSE"] = bak
    return _symbols_from_env()


def run_scan(
    *,
    use_db: bool = True,
    use_tg: bool = True,
    symbols_override: Optional[List[str]] = None,
    watchlist_request: Optional[bool] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    syms, sym_meta = resolve_scan_symbols(
        symbols_override=symbols_override,
        watchlist_request=watchlist_request,
    )
    syms, api_shuffled = _shuffle_symbol_list(syms)
    sym_meta["api_order_shuffled"] = api_shuffled
    results: List[VPRegimeResult] = []
    errors: List[str] = []

    for i, sym in enumerate(syms):
        try:
            r = analyze_symbol(sym)
            if r:
                results.append(r)
        except Exception as e:
            errors.append(f"{sym}: {e}")
        if i < len(syms) - 1:
            _inter_symbol_sleep_with_jitter()

    if use_db:
        try:
            from accumulation_radar import init_db

            conn = init_db()
            ensure_table(conn)
            for r in results:
                if DB_SKIP_NO_TRADE and r.scheme == "NO_TRADE" and not r.liquidity_ok:
                    continue
                upsert_row(conn, r)
            conn.commit()
            conn.close()
        except Exception as e:
            errors.append(f"db: {e}")

    # stdout
    by_scheme: Dict[str, int] = {}
    for r in results:
        by_scheme[r.scheme] = by_scheme.get(r.scheme, 0) + 1
    summary: Dict[str, Any] = {
        "ok": True,
        "universe": sym_meta.get("universe", "default"),
        "symbols": len(syms),
        "written": len(results),
        "by_scheme": by_scheme,
        "errors": errors[:20],
    }
    summary.update({k: v for k, v in sym_meta.items() if k != "universe"})
    if (
        sym_meta.get("universe") == "watchlist"
        and sym_meta.get("watchlist_pool_usdt", 0) > 80
        and sym_meta.get("watchlist_max", 0) == 0
    ):
        summary["hint"] = (
            "收筹池 USDT 标的较多且未设上限，可设 VP_WATCHLIST_MAX_SYMBOLS 降低每轮 REST 次数"
        )
    if not quiet:
        print(json.dumps(summary, ensure_ascii=False))
        for r in sorted(results, key=lambda x: x.symbol):
            liq = "Y" if r.liquidity_ok else "N"
            print(
                f"{r.symbol}\t{r.scheme}\t{r.vol_pattern}\tliq={liq}\t"
                f"vol_ma={r.vol_usd_ma:.0f}\tclose={r.close:g}"
            )

    if use_tg and TG_BOT_TOKEN and TG_CHAT_ID:
        uni = str(sym_meta.get("universe") or "default")
        wl_pool = sym_meta.get("watchlist_pool_usdt")
        wl_line = f"标的 {len(syms)} · 有效行 {len(results)}"
        if wl_pool is not None:
            wl_line += f"（收筹池 USDT 共 {wl_pool}）"
        lines_tg = [
            "📊 VP Regime 扫描",
            f"universe={uni} · {wl_line}",
            "scheme 计数: " + ", ".join(f"{k}={v}" for k, v in sorted(by_scheme.items())) if by_scheme else "scheme 计数: (无)",
        ]
        if summary.get("hint"):
            lines_tg.append("hint: " + str(summary["hint"]))
        if errors:
            lines_tg.append("errors:")
            for e in errors[:20]:
                lines_tg.append("  " + str(e))
        lines_tg.append("---")
        if results:
            for r in sorted(results, key=lambda x: x.symbol):
                liq = "ok" if r.liquidity_ok else "low_liq"
                d = r.detail or {}
                vvm = d.get("vol_vs_ma", "")
                lines_tg.append(
                    f"{r.symbol} | {r.scheme} | {r.vol_pattern} | liq={liq} | "
                    f"vol_ma≈{r.vol_usd_ma:.0f} | px={r.close:g} | vol/ma={vvm}"
                )
        else:
            lines_tg.append("（无结果行：标的为空、K 线失败或数据不足）")
        maybe_send_tg(lines_tg)

    out: Dict[str, Any] = {
        "ok": True,
        "universe": sym_meta.get("universe", "default"),
        "symbols": len(syms),
        "written": len(results),
        "by_scheme": by_scheme,
        "results": [asdict(r) for r in results],
        "errors": errors,
    }
    out.update({k: v for k, v in sym_meta.items() if k != "universe"})
    if summary.get("hint"):
        out["hint"] = summary["hint"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="VP Regime Scanner (VolUSD proxy + volume patterns)")
    ap.add_argument(
        "--watchlist",
        action="store_true",
        help="从收筹池 watchlist 取标的（等价设置 VP_WATCHLIST_UNIVERSE=1）",
    )
    ap.add_argument(
        "--max-watchlist",
        type=int,
        default=0,
        metavar="N",
        help="收筹池模式下每轮最多扫 N 个（设置 VP_WATCHLIST_MAX_SYMBOLS；0 表示不覆盖）",
    )
    ap.add_argument("--no-db", action="store_true", help="不写入 accumulation.db")
    ap.add_argument("--no-tg", action="store_true", help="不发送 Telegram")
    args = ap.parse_args()
    if args.watchlist:
        os.environ["VP_WATCHLIST_UNIVERSE"] = "1"
    if args.max_watchlist > 0:
        os.environ["VP_WATCHLIST_MAX_SYMBOLS"] = str(args.max_watchlist)
    wl_req: Optional[bool] = True if args.watchlist else None
    run_scan(use_db=not args.no_db, use_tg=not args.no_tg, watchlist_request=wl_req, quiet=False)


if __name__ == "__main__":
    main()

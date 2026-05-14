"""
Next K API — accumulation / OI / ZCT; optional multi-asset anomaly radar (OHLCV, no ML).
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
# Static files removed - frontend is deployed separately on Vercel
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# 必须在读取下方可选开关（如 ZCT）之前执行：uvicorn 不会自动加载 .env
_env_oi = Path(__file__).resolve().parent / ".env.oi"
if _env_oi.is_file():
    with open(_env_oi, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# 临时关闭：s6 期货模拟盘定时任务（恢复时改为 True，并取消前端对应区块 hidden）
S6_FUTURES_ALPHA_SCHEDULER_ENABLED = False
# ZCT VWAP：设 ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED=1 开启定时（Asia/Shanghai 进程内 IntervalTrigger）
# 全量扫描（classify + 入库）与结算（resolve-only）拆成两档频率，见下方分钟数。
ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED = (
    os.getenv("ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
)
ZCT_VWAP_SCAN_INTERVAL_MINUTES = max(
    1, int(os.getenv("ZCT_VWAP_SCAN_INTERVAL_MINUTES", "30") or 30)
)
# 0 = 不注册独立结算任务（仍依赖全量扫描脚本内的 pre/post resolve）
ZCT_VWAP_RESOLVE_INTERVAL_MINUTES = max(
    0, int(os.getenv("ZCT_VWAP_RESOLVE_INTERVAL_MINUTES", "5") or 5)
)
# ZCT · 🔥⚡热度+OI：标的来自 worth_watch_hot_oi；库表 zct_hot_oi_*；定时与主 lane 错开（默认 35min / 7min）
# 默认开启；设 ZCT_HOT_OI_SIGNAL_SCHEDULER_ENABLED=0|false|off|disabled 可关闭
_zct_hot_oi_sched_raw = os.getenv("ZCT_HOT_OI_SIGNAL_SCHEDULER_ENABLED", "1").strip().lower()
ZCT_HOT_OI_SIGNAL_SCHEDULER_ENABLED = _zct_hot_oi_sched_raw not in (
    "0",
    "false",
    "no",
    "off",
    "disabled",
)
ZCT_HOT_OI_SCAN_INTERVAL_MINUTES = max(
    1, int(os.getenv("ZCT_HOT_OI_SCAN_INTERVAL_MINUTES", "35") or 35)
)
ZCT_HOT_OI_RESOLVE_INTERVAL_MINUTES = max(
    0, int(os.getenv("ZCT_HOT_OI_RESOLVE_INTERVAL_MINUTES", "7") or 7)
)


# ============== Asset Types & Symbols ==============

class AssetType(str, Enum):
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"


# Supported symbols by asset type
SYMBOLS = {
    AssetType.CRYPTO: [
        {"symbol": "BTC/USDT", "name": "Bitcoin", "icon": "₿"},
        {"symbol": "ETH/USDT", "name": "Ethereum", "icon": "Ξ"},
        {"symbol": "BNB/USDT", "name": "BNB", "icon": "B"},
        {"symbol": "SOL/USDT", "name": "Solana", "icon": "◎"},
        {"symbol": "PEPE/USDT", "name": "Pepe", "icon": "🐸"},
    ],
    AssetType.STOCK: [
        {"symbol": "AAPL", "name": "Apple Inc.", "icon": "🍎"},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "icon": "G"},
        {"symbol": "MSFT", "name": "Microsoft", "icon": "M"},
        {"symbol": "TSLA", "name": "Tesla Inc.", "icon": "T"},
        {"symbol": "NVDA", "name": "NVIDIA", "icon": "N"},
        {"symbol": "AMZN", "name": "Amazon", "icon": "A"},
        {"symbol": "META", "name": "Meta Platforms", "icon": "M"},
    ],
    AssetType.FOREX: [
        {"symbol": "EUR/USD", "name": "Euro/US Dollar", "icon": "€"},
        {"symbol": "GBP/USD", "name": "British Pound/US Dollar", "icon": "£"},
        {"symbol": "USD/JPY", "name": "US Dollar/Japanese Yen", "icon": "¥"},
        {"symbol": "AUD/USD", "name": "Australian Dollar/US Dollar", "icon": "A$"},
        {"symbol": "USD/CHF", "name": "US Dollar/Swiss Franc", "icon": "Fr"},
    ],
}


# ============== Global State ==============

class AppState:
    """Application state container."""
    ccxt_exchange = None  # For crypto
    yfinance_available = False  # For stocks/forex
    startup_time = None


state = AppState()


# ============== Data Fetchers ==============

# CoinGecko coin ID mapping
COINGECKO_IDS = {
    "BTC/USDT": "bitcoin", "ETH/USDT": "ethereum", "SOL/USDT": "solana",
    "BNB/USDT": "binancecoin", "PEPE/USDT": "pepe",
}


async def fetch_crypto_coingecko(symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
    """Fetch crypto OHLCV data from CoinGecko (fallback)."""
    import aiohttp

    coin_id = COINGECKO_IDS.get(symbol)
    if not coin_id:
        return None

    try:
        days = min(limit // 24 + 1, 90)  # CoinGecko max 90 days for hourly
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    logger.warning(f"CoinGecko returned {resp.status} for {symbol}")
                    return None
                data = await resp.json()

        if not data:
            return None

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['volume'] = 0  # CoinGecko OHLC doesn't include volume
        df = df.tail(limit)
        logger.info(f"Fetched {len(df)} bars from CoinGecko for {symbol}")
        return df

    except Exception as e:
        logger.warning(f"CoinGecko failed for {symbol}: {e}")
        return None


async def fetch_crypto_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 100) -> Optional[pd.DataFrame]:
    """Fetch crypto OHLCV data - tries Binance first, then CoinGecko."""
    # Try Binance first
    if state.ccxt_exchange:
        try:
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: state.ccxt_exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            )
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            logger.info(f"Fetched {len(df)} bars from Binance for {symbol}")
            return df
        except Exception as e:
            logger.warning(f"Binance failed for {symbol}: {e}")

    # Fallback to CoinGecko
    return await fetch_crypto_coingecko(symbol, limit)


async def fetch_stock_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 100) -> Optional[pd.DataFrame]:
    """Fetch stock OHLCV data from yfinance."""
    try:
        import yfinance as yf

        # Map timeframe to yfinance parameters
        # 对于1小时数据，需要至少11天才能获取250根（250/24 ≈ 11天）
        # yfinance最多支持730天（2年）的历史数据
        if timeframe == "1h":
            # 计算需要的天数：limit / 24，向上取整，最少11天，最多730天
            days_needed = max(11, min((limit + 23) // 24, 730))
            period = f"{days_needed}d"
            interval = "1h"
        else:
            tf_map = {
                "4h": ("60d", "1h"),  # yfinance doesn't support 4h, fetch 1h and resample
                "1d": (f"{limit}d", "1d"),
                "1w": (f"{limit * 7}d", "1wk"),
            }
            period, interval = tf_map.get(timeframe, ("7d", "1h"))

        def _fetch():
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty and interval == "1h":
                # Fallback to daily data if hourly not available
                df = ticker.history(period=f"{limit}d", interval="1d")
            return df

        df = await asyncio.get_event_loop().run_in_executor(None, _fetch)

        if df is not None and not df.empty:
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={'datetime': 'timestamp', 'date': 'timestamp'})
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # Resample for 4h if needed
            if timeframe == "4h" and interval == "1h":
                df = df.set_index('timestamp')
                df = df.resample('4h').agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum'
                }).dropna().reset_index()

            return df.tail(limit)

        return None
    except Exception as e:
        logger.warning(f"Failed to fetch stock OHLCV for {symbol}: {e}")
        return None


async def fetch_forex_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 100) -> Optional[pd.DataFrame]:
    """Fetch forex OHLCV data from yfinance."""
    try:
        import yfinance as yf

        # Convert forex symbol format: EUR/USD -> EURUSD=X
        yf_symbol = symbol.replace("/", "") + "=X"

        # Map timeframe to yfinance parameters
        # 对于1小时数据，需要至少11天才能获取250根（250/24 ≈ 11天）
        # yfinance最多支持730天（2年）的历史数据
        if timeframe == "1h":
            # 计算需要的天数：limit / 24，向上取整，最少11天，最多730天
            days_needed = max(11, min((limit + 23) // 24, 730))
            period = f"{days_needed}d"
            interval = "1h"
        else:
            tf_map = {
                "4h": ("60d", "1h"),  # Resample to 4h
                "1d": (f"{limit}d", "1d"),
                "1w": (f"{limit * 7}d", "1wk"),
            }
            period, interval = tf_map.get(timeframe, ("7d", "1h"))

        def _fetch():
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty and interval == "1h":
                df = ticker.history(period=f"{limit}d", interval="1d")
            return df

        df = await asyncio.get_event_loop().run_in_executor(None, _fetch)

        if df is not None and not df.empty:
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={'datetime': 'timestamp', 'date': 'timestamp'})
            if 'volume' not in df.columns:
                df['volume'] = 0
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # Resample for 4h if needed
            if timeframe == "4h" and interval == "1h":
                df = df.set_index('timestamp')
                df = df.resample('4h').agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum'
                }).dropna().reset_index()

            return df.tail(limit)

        return None
    except Exception as e:
        logger.warning(f"Failed to fetch forex OHLCV for {symbol}: {e}")
        return None


def generate_demo_ohlcv(symbol: str, asset_type: AssetType, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
    """Generate demo OHLCV data when real data is unavailable."""
    # Base prices for different assets
    base_prices = {
        "BTC/USDT": 95000, "ETH/USDT": 3200, "SOL/USDT": 180,
        "BNB/USDT": 650, "PEPE/USDT": 0.00002,
        "AAPL": 230, "MSFT": 420, "GOOGL": 175,
        "TSLA": 380, "NVDA": 140, "AMZN": 220, "META": 580,
        "EUR/USD": 1.08, "GBP/USD": 1.27, "USD/JPY": 155,
        "AUD/USD": 0.64, "USD/CHF": 0.90
    }
    base_price = base_prices.get(symbol, 100)
    volatility = 0.02 if asset_type == AssetType.CRYPTO else 0.01

    # Adjust volatility for different timeframes
    tf_hours = {"1h": 1, "4h": 4, "1d": 24, "1w": 168}
    hours_per_bar = tf_hours.get(timeframe, 1)
    volatility = volatility * np.sqrt(hours_per_bar)  # Scale volatility

    now = datetime.now(timezone.utc)
    data = []
    price = base_price

    for i in range(limit):
        timestamp = now - timedelta(hours=(limit - i) * hours_per_bar)
        change = np.random.randn() * volatility
        price = price * (1 + change)
        high = price * (1 + abs(np.random.randn() * volatility * 0.5))
        low = price * (1 - abs(np.random.randn() * volatility * 0.5))
        open_price = low + np.random.random() * (high - low)
        volume = np.random.randint(1000, 100000) * base_price * hours_per_bar

        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })

    return pd.DataFrame(data)


async def fetch_ohlcv(symbol: str, asset_type: AssetType, timeframe: str = "1h", limit: int = 100) -> Optional[pd.DataFrame]:
    """Unified OHLCV fetcher for all asset types. Falls back to demo data if unavailable."""
    df = None

    if asset_type == AssetType.CRYPTO:
        df = await fetch_crypto_ohlcv(symbol, timeframe, limit)
    elif asset_type == AssetType.STOCK:
        df = await fetch_stock_ohlcv(symbol, timeframe, limit)
    elif asset_type == AssetType.FOREX:
        df = await fetch_forex_ohlcv(symbol, timeframe, limit)

    # Fallback to demo data if real data unavailable
    if df is None or len(df) < 30:
        logger.info(f"Using demo data for {symbol} ({asset_type.value}) [{timeframe}]")
        df = generate_demo_ohlcv(symbol, asset_type, timeframe, limit)

    return df


def detect_asset_type(symbol: str) -> AssetType:
    """Auto-detect asset type from symbol."""
    symbol_upper = symbol.upper()

    # Check crypto
    for s in SYMBOLS[AssetType.CRYPTO]:
        if s["symbol"] == symbol_upper or symbol_upper.replace("-", "/") == s["symbol"]:
            return AssetType.CRYPTO

    # Check stocks
    for s in SYMBOLS[AssetType.STOCK]:
        if s["symbol"] == symbol_upper:
            return AssetType.STOCK

    # Check forex
    for s in SYMBOLS[AssetType.FOREX]:
        if s["symbol"] == symbol_upper or symbol_upper.replace("-", "/") == s["symbol"]:
            return AssetType.FOREX

    # Default to crypto if contains USDT/USD pair format
    if "USDT" in symbol_upper or (symbol_upper.count("/") == 1 and len(symbol_upper) < 10):
        return AssetType.CRYPTO

    # Default to stock for single tickers
    return AssetType.STOCK


# ============== Accumulation radar (APScheduler) ==============

_RADAR_SCRIPT = Path(__file__).resolve().parent / "accumulation_radar.py"


def _run_accumulation_radar_subprocess(mode: str) -> None:
    """Run accumulation_radar.py in a subprocess (pool / oi / full)."""
    logger.info("Starting accumulation_radar subprocess mode=%s", mode)
    try:
        subprocess.run(
            [sys.executable, str(_RADAR_SCRIPT), mode],
            cwd=str(_RADAR_SCRIPT.parent),
            check=False,
        )
    except Exception as e:
        logger.exception("accumulation_radar %s failed: %s", mode, e)


def run_pool_task() -> None:
    logger.info("开始执行每日收筹池扫描...")
    _run_accumulation_radar_subprocess("pool")


def run_oi_task() -> None:
    logger.info("开始执行每小时 OI 异动扫描...")
    _run_accumulation_radar_subprocess("oi")


_heat_watch_refresh_lock = threading.Lock()


def _refresh_heat_accum_watch_full_once() -> Dict[str, Any]:
    from accumulation_radar import init_db, refresh_all_heat_accum_watch_full, refresh_all_worth_watch_bpc_states

    conn = init_db()
    try:
        out = refresh_all_heat_accum_watch_full(conn)
        w = refresh_all_worth_watch_bpc_states(conn)
        out.update(w)
        return out
    finally:
        conn.close()


def run_heat_watch_refresh_task() -> None:
    """每小时：heat_accum_watch 现价/摘要（1h BPC 状态机已移除，worth/focus 的 bpc_json 不再重算）。"""
    if not _heat_watch_refresh_lock.acquire(blocking=False):
        logger.info("热度看盘整表刷新跳过：已有任务在执行")
        return
    try:
        logger.info("开始执行热度看盘整表刷新（现价/摘要；BPC 已关闭）...")
        data = _refresh_heat_accum_watch_full_once()
        logger.info(
            "热度看盘整表刷新完成: prices=%s bpc=%s bpc_failed_klines=%s worth_bpc=%s worth_bpc_fail_kl=%s worth_bpc_syms=%s bpc_disabled=%s",
            data.get("recalculated_prices"),
            data.get("bpc_recalculated"),
            data.get("bpc_failed_klines"),
            data.get("worth_watch_bpc_recalculated"),
            data.get("worth_watch_bpc_failed_klines"),
            data.get("worth_watch_bpc_symbols"),
            data.get("bpc_disabled"),
        )
    except Exception as e:
        logger.exception("heat watch refresh failed: %s", e)
    finally:
        _heat_watch_refresh_lock.release()


# ============== s2 OI + funding flip scanner (APScheduler) ==============

_S2_FUNDING_SCRIPT = Path(__file__).resolve().parent / "s2_oi_funding_rate_scanner.py"


def _run_s2_oi_funding_rate_scanner_subprocess() -> None:
    """Run s2_oi_funding_rate_scanner.py (OI + 费率刚转负, 与脚本内快照配合)."""
    logger.info("Starting s2_oi_funding_rate_scanner subprocess")
    try:
        subprocess.run(
            [sys.executable, str(_S2_FUNDING_SCRIPT)],
            cwd=str(_S2_FUNDING_SCRIPT.parent),
            check=False,
        )
    except Exception as e:
        logger.exception("s2_oi_funding_rate_scanner failed: %s", e)


def run_s2_oi_funding_task() -> None:
    logger.info("开始执行 s2 OI+费率转负扫描...")
    _run_s2_oi_funding_rate_scanner_subprocess()


# ============== s6 期货 Alpha 自主模拟交易 ==============

_S6_ALPHA_SCRIPT = Path(__file__).resolve().parent / "s6_futures_alpha_autonomous_trading_v1.py"


def _run_s6_futures_alpha_subprocess() -> None:
    """Run s6_futures_alpha_autonomous_trading_v1.py（虚拟开平仓 + 信号历史）。"""
    logger.info("Starting s6_futures_alpha_autonomous_trading_v1 subprocess")
    try:
        subprocess.run(
            [sys.executable, str(_S6_ALPHA_SCRIPT)],
            cwd=str(_S6_ALPHA_SCRIPT.parent),
            check=False,
        )
    except Exception as e:
        logger.exception("s6_futures_alpha_autonomous_trading_v1 failed: %s", e)


def run_s6_futures_alpha_task() -> None:
    logger.info("开始执行 s6 期货 Alpha 自主扫描...")
    _run_s6_futures_alpha_subprocess()


# ============== ZCT VWAP 信号扫描（VWAP 体制 + 关键位） =============

_ZCT_VWAP_SCRIPT = Path(__file__).resolve().parent / "zct_vwap_signal_scanner.py"


def _run_zct_vwap_signal_subprocess() -> None:
    """与 `_run_accumulation_radar_subprocess` 同范式：`cwd`=脚本目录；脚本内自载 `.env.oi`，TG 与 accumulation 共用变量。"""
    logger.info("Starting zct_vwap_signal_scanner subprocess")
    try:
        subprocess.run(
            [sys.executable, str(_ZCT_VWAP_SCRIPT)],
            cwd=str(_ZCT_VWAP_SCRIPT.parent),
            check=False,
        )
    except Exception as e:
        logger.exception("zct_vwap_signal_scanner failed: %s", e)


def run_zct_vwap_signal_task() -> None:
    logger.info("开始执行 ZCT VWAP 信号扫描...")
    _run_zct_vwap_signal_subprocess()


def _run_zct_vwap_resolve_only_subprocess() -> None:
    """仅 DB 纸面结算（--resolve-only），与全量扫描解耦。"""
    logger.info("Starting zct_vwap_signal_scanner --resolve-only subprocess")
    try:
        subprocess.run(
            [sys.executable, str(_ZCT_VWAP_SCRIPT), "--resolve-only"],
            cwd=str(_ZCT_VWAP_SCRIPT.parent),
            check=False,
        )
    except Exception as e:
        logger.exception("zct_vwap_signal_scanner --resolve-only failed: %s", e)


def run_zct_vwap_resolve_only_task() -> None:
    logger.info("开始执行 ZCT VWAP 结算(resolve-only)...")
    _run_zct_vwap_resolve_only_subprocess()


def _zct_hot_oi_child_env() -> dict:
    env = os.environ.copy()
    env["ZCT_HOT_OI_UNIVERSE"] = "1"
    env["ZCT_DB_SIGNALS_TABLE"] = "zct_hot_oi_signals"
    env["ZCT_DB_SETTLEMENTS_TABLE"] = "zct_hot_oi_settlements"
    return env


def _run_zct_hot_oi_signal_subprocess() -> None:
    logger.info("Starting zct_vwap_signal_scanner subprocess (hot_oi lane)")
    try:
        subprocess.run(
            [sys.executable, str(_ZCT_VWAP_SCRIPT)],
            cwd=str(_ZCT_VWAP_SCRIPT.parent),
            env=_zct_hot_oi_child_env(),
            check=False,
        )
    except Exception as e:
        logger.exception("zct hot_oi scan failed: %s", e)


def run_zct_hot_oi_signal_task() -> None:
    logger.info("开始执行 ZCT 🔥⚡热度+OI 扫描...")
    _run_zct_hot_oi_signal_subprocess()


def _run_zct_hot_oi_resolve_only_subprocess() -> None:
    logger.info("Starting zct_vwap_signal_scanner --resolve-only (hot_oi lane)")
    try:
        subprocess.run(
            [sys.executable, str(_ZCT_VWAP_SCRIPT), "--resolve-only"],
            cwd=str(_ZCT_VWAP_SCRIPT.parent),
            env=_zct_hot_oi_child_env(),
            check=False,
        )
    except Exception as e:
        logger.exception("zct hot_oi resolve failed: %s", e)


def run_zct_hot_oi_resolve_only_task() -> None:
    logger.info("开始执行 ZCT 🔥⚡热度+OI 结算...")
    _run_zct_hot_oi_resolve_only_subprocess()


# ============== Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("Starting Next K (Multi-Asset)...")
    state.startup_time = datetime.now(timezone.utc)

    # Initialize ccxt for crypto
    try:
        import ccxt
        state.ccxt_exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 15000,
        })
        await asyncio.get_event_loop().run_in_executor(
            None, state.ccxt_exchange.load_markets
        )
        logger.info("Binance (crypto) connection established")
    except Exception as e:
        logger.warning(f"Binance connection failed: {e}")
        state.ccxt_exchange = None

    # Check yfinance availability for stocks/forex
    try:
        import yfinance as yf
        state.yfinance_available = True
        logger.info("yfinance (stocks/forex) available")
    except ImportError:
        logger.warning("yfinance not available - stocks/forex will be limited")
        state.yfinance_available = False

    # Daily pool 10:00 CST; heat :07; OI :30; s2 :05 (Asia/Shanghai)
    tz = pytz.timezone("Asia/Shanghai")
    accumulation_scheduler = BackgroundScheduler(timezone=tz)
    accumulation_scheduler.add_job(run_pool_task, "cron", hour=10, minute=0)
    accumulation_scheduler.add_job(
        run_heat_watch_refresh_task,
        "cron",
        minute=7,
        id="heat_watch_refresh",
    )
    accumulation_scheduler.add_job(run_oi_task, "cron", minute=30)
    accumulation_scheduler.add_job(
        run_s2_oi_funding_task,
        "cron",
        minute=5,
        id="s2_oi_funding_rate_scanner",
    )
    if S6_FUTURES_ALPHA_SCHEDULER_ENABLED:
        accumulation_scheduler.add_job(
            run_s6_futures_alpha_task,
            "cron",
            minute=25,
            id="s6_futures_alpha_autonomous_trading",
        )
    if ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED:
        accumulation_scheduler.add_job(
            run_zct_vwap_signal_task,
            IntervalTrigger(minutes=ZCT_VWAP_SCAN_INTERVAL_MINUTES),
            id="zct_vwap_signal_scanner",
        )
        if ZCT_VWAP_RESOLVE_INTERVAL_MINUTES > 0:
            accumulation_scheduler.add_job(
                run_zct_vwap_resolve_only_task,
                IntervalTrigger(minutes=ZCT_VWAP_RESOLVE_INTERVAL_MINUTES),
                id="zct_vwap_resolve_only",
            )
    if ZCT_HOT_OI_SIGNAL_SCHEDULER_ENABLED:
        accumulation_scheduler.add_job(
            run_zct_hot_oi_signal_task,
            IntervalTrigger(minutes=ZCT_HOT_OI_SCAN_INTERVAL_MINUTES),
            id="zct_hot_oi_signal_scanner",
        )
        if ZCT_HOT_OI_RESOLVE_INTERVAL_MINUTES > 0:
            accumulation_scheduler.add_job(
                run_zct_hot_oi_resolve_only_task,
                IntervalTrigger(minutes=ZCT_HOT_OI_RESOLVE_INTERVAL_MINUTES),
                id="zct_hot_oi_resolve_only",
            )
    accumulation_scheduler.start()
    app.state.accumulation_scheduler = accumulation_scheduler
    logger.info("BPC（1H 突破—回踩—延续）与 OI 后 TG 延续推送：已随 breakout_pullback_fsm 移除")

    s6_cron_log = (
        "s6_futures_alpha 每整点后 25 分 (xx:25)"
        if S6_FUTURES_ALPHA_SCHEDULER_ENABLED
        else "s6_futures_alpha 定时已暂停"
    )
    if ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED:
        zct_vwap_log = (
            f"zct_vwap 全量扫描每 {ZCT_VWAP_SCAN_INTERVAL_MINUTES} 分钟 · "
            f"结算(resolve-only)每 {ZCT_VWAP_RESOLVE_INTERVAL_MINUTES} 分钟（Asia/Shanghai 进程内间隔触发）"
            if ZCT_VWAP_RESOLVE_INTERVAL_MINUTES > 0
            else (
                f"zct_vwap 全量扫描每 {ZCT_VWAP_SCAN_INTERVAL_MINUTES} 分钟 · "
                "独立结算定时已关闭（ZCT_VWAP_RESOLVE_INTERVAL_MINUTES=0）"
            )
        )
    else:
        zct_vwap_log = "zct_vwap 定时未启用（设 ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED=1）"
    if ZCT_HOT_OI_SIGNAL_SCHEDULER_ENABLED:
        zct_hot_oi_log = (
            f"zct_hot_oi 🔥⚡ 每 {ZCT_HOT_OI_SCAN_INTERVAL_MINUTES} 分钟 · "
            f"结算每 {ZCT_HOT_OI_RESOLVE_INTERVAL_MINUTES} 分钟"
            if ZCT_HOT_OI_RESOLVE_INTERVAL_MINUTES > 0
            else (
                f"zct_hot_oi 🔥⚡ 每 {ZCT_HOT_OI_SCAN_INTERVAL_MINUTES} 分钟 · "
                "独立结算已关闭（ZCT_HOT_OI_RESOLVE_INTERVAL_MINUTES=0）"
            )
        )
    else:
        zct_hot_oi_log = (
            "zct_hot_oi 定时已关闭（默认开启；当前由 ZCT_HOT_OI_SIGNAL_SCHEDULER_ENABLED=0|false|off 关闭）"
        )
    logger.info(
        "后台定时任务已启动: accumulation_radar pool 每日 10:00 CST, "
        "heat_watch 每小时 xx:07（现价/摘要 + 1h BPC）; "
        "oi 每小时 :30; "
        "s2_oi_funding_rate_scanner 每整点后 5 分 (xx:05); "
        + s6_cron_log
        + "; "
        + zct_vwap_log
        + "; "
        + zct_hot_oi_log
    )

    yield

    sch = getattr(app.state, "accumulation_scheduler", None)
    if sch is not None:
        sch.shutdown(wait=False)
        app.state.accumulation_scheduler = None

    logger.info("Shutting down...")


# ============== FastAPI App ==============

app = FastAPI(
    title="Next K",
    description="收筹 / OI / S2 / S6 / ZCT VWAP 与 ZCT 🔥⚡；可选 GET /api/radar 异动扫描（OHLCV，无 ML）。",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class SignalType(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class RadarItem(BaseModel):
    symbol: str
    name: str
    asset_type: str
    anomaly_score: float
    signal: SignalType
    signals: List[str]
    price: float
    price_change: float
    regime_hint: str


class HealthResponse(BaseModel):
    status: str
    crypto_connected: bool
    stocks_available: bool
    forex_available: bool
    version: str
    uptime: float


# ============== Helper Functions ==============

def smart_round(price: float, min_decimals: int = 4) -> float:
    """Dynamically round price based on magnitude (for small-price coins like PEPE)."""
    if price == 0:
        return 0.0
    abs_price = abs(price)
    if abs_price >= 1:
        return round(price, min_decimals)
    elif abs_price >= 0.01:
        return round(price, 6)
    elif abs_price >= 0.0001:
        return round(price, 8)
    else:
        return round(price, 10)


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "name": "Next K API",
        "version": "2.0.0",
        "description": "Accumulation / OI / S2 / S6 / ZCT APIs",
        "docs": "/docs",
        "health": "/api/health",
        "zct_vwap_dashboard": "/dashboard/zct-vwap",
    }


@app.get("/api/health", response_model=HealthResponse)
async def health():
    uptime = (datetime.now(timezone.utc) - state.startup_time).total_seconds() if state.startup_time else 0
    return HealthResponse(
        status="healthy",
        crypto_connected=state.ccxt_exchange is not None,
        stocks_available=state.yfinance_available,
        forex_available=state.yfinance_available,
        version="2.0.0",
        uptime=uptime,
    )


@app.get("/api/radar")
async def get_radar(asset_type: Optional[str] = None):
    """Anomaly Radar - Scan for unusual patterns across all asset types."""
    items = []

    # Determine which asset types to scan
    if asset_type:
        asset_types = [AssetType(asset_type)]
    else:
        asset_types = list(AssetType)

    for at in asset_types:
        for sym_info in SYMBOLS[at]:
            symbol = sym_info["symbol"]
            name = sym_info["name"]

            try:
                df = await fetch_ohlcv(symbol, at, "1h", 50)
                if df is None or len(df) < 24:
                    continue

                closes = df['close'].values.astype(float)
                current = float(closes[-1])

                # Calculate anomaly score
                if len(closes) > 48:
                    recent_vol = np.std(np.log(closes[-24:] / closes[-25:-1]))
                    hist_vol = np.std(np.log(closes[-48:-24] / closes[-49:-25]))
                    vol_ratio = recent_vol / hist_vol if hist_vol > 0 else 1.0
                else:
                    vol_ratio = 1.0

                price_24h_ago = closes[-24] if len(closes) >= 24 else closes[0]
                price_change = (current - price_24h_ago) / price_24h_ago * 100

                anomaly = min(100, max(0, (vol_ratio - 1) * 50 + abs(price_change) * 2))

                signals = []
                if vol_ratio > 1.5:
                    signals.append("波動率飆升")
                if abs(price_change) > 5:
                    signals.append(f"強勢動能 ({price_change:+.1f}%)")

                if price_change > 3 and anomaly > 50:
                    signal = SignalType.BULLISH
                elif price_change < -3 and anomaly > 50:
                    signal = SignalType.BEARISH
                else:
                    signal = SignalType.NEUTRAL

                regime_hint = "高度不確定 - 可能大波動" if anomaly > 70 else "正常波動" if anomaly < 30 else "值得關注"

                items.append(RadarItem(
                    symbol=symbol, name=name, asset_type=at.value,
                    anomaly_score=round(anomaly, 1), signal=signal,
                    signals=signals if signals else ["正常"],
                    price=smart_round(current), price_change=round(price_change, 2),
                    regime_hint=regime_hint,
                ))

            except Exception as e:
                logger.warning(f"Radar scan failed for {symbol}: {e}")

    items.sort(key=lambda x: x.anomaly_score, reverse=True)
    return items


def _oi_radar_snapshot_path() -> Path:
    """与 accumulation_radar 的 DATA_DIR / accumulation.db 同目录。"""
    db_dir = Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parent)))
    return db_dir / "oi_radar_snapshot.json"


_oi_radar_refresh_lock = threading.Lock()


@app.get("/api/accumulation/oi-radar")
async def get_accumulation_oi_radar():
    """
    返回磁盘上的最新 OI 雷达 JSON（由定时任务或 POST refresh 写入），响应极快，避免
    Railway/浏览器对长连接（完整扫描 1–2 分钟）超时导致「Failed to fetch」。

    本接口不触发 Telegram；每小时 :30 子进程仍会推送。
    """
    path = _oi_radar_snapshot_path()
    if not path.is_file():
        return {
            "ok": False,
            "error": "no_snapshot",
            "message": "尚无快照。请等待整点 :30 定时扫描写入，或点击前端「刷新」触发后台扫描（约 1–2 分钟后再次加载本接口）。",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("snapshot root must be object")
        data["snapshot_source"] = "disk"
        return data
    except Exception as e:
        logger.warning("OI radar snapshot read failed: %s", e)
        raise HTTPException(status_code=500, detail="snapshot_corrupt")


@app.get("/api/accumulation/heat-accum-watch")
async def get_heat_accum_watch():
    """热度+收筹独立看盘：读写 accumulation.db 表 heat_accum_watch；含生成日与 2 日保留。"""
    try:
        from accumulation_radar import init_db, load_heat_accum_watchlist_from_db

        conn = init_db()
        try:
            data = load_heat_accum_watchlist_from_db(conn)
        finally:
            conn.close()
        if not data.get("items"):
            data.setdefault(
                "message",
                "尚无归档，请等待整点 :30 扫描或点击「刷新」后重试。",
            )
        return data
    except Exception as e:
        logger.warning("heat_accum watchlist read failed: %s", e)
        raise HTTPException(status_code=500, detail="watchlist_db_error")


@app.get("/api/accumulation/ambush-watch")
async def get_ambush_watch():
    """埋伏榜内 🎯 暗流 / 💎 低市值+OI：表 ambush_watch；含生成日与 2 日保留。"""
    try:
        from accumulation_radar import init_db, load_ambush_watchlist_from_db

        conn = init_db()
        try:
            data = load_ambush_watchlist_from_db(conn)
        finally:
            conn.close()
        if not data.get("items"):
            data.setdefault(
                "message",
                "尚无归档，请等待整点 :30 扫描或点击「刷新」后重试。",
            )
        return data
    except Exception as e:
        logger.warning("ambush watchlist read failed: %s", e)
        raise HTTPException(status_code=500, detail="ambush_watch_db_error")


@app.get("/api/accumulation/focus-watch")
async def get_focus_watch():
    """👑 重点关注（逼空/天量/暗流 + 否决）：表 focus_watch；含生成日与 2 日保留。"""
    try:
        from accumulation_radar import init_db, load_focus_watchlist_from_db

        conn = init_db()
        try:
            data = load_focus_watchlist_from_db(conn)
        finally:
            conn.close()
        if not data.get("items"):
            data.setdefault(
                "message",
                "尚无归档，请等待整点 :30 扫描或点击「刷新」后重试。",
            )
        return data
    except Exception as e:
        logger.warning("focus_watch read failed: %s", e)
        raise HTTPException(status_code=500, detail="focus_watch_db_error")


@app.get("/api/accumulation/patrick-core-watch")
async def get_patrick_core_watch():
    """📍 Patrick 核心：收筹池 + OI 异动；表 patrick_core_watch；含生成日与 2 日保留。"""
    try:
        from accumulation_radar import init_db, load_patrick_core_watchlist_from_db

        conn = init_db()
        try:
            data = load_patrick_core_watchlist_from_db(conn)
        finally:
            conn.close()
        if not data.get("items"):
            data.setdefault(
                "message",
                "尚无归档，请等待整点 :30 扫描或点击「刷新」后重试。",
            )
        return data
    except Exception as e:
        logger.warning("patrick_core watchlist read failed: %s", e)
        raise HTTPException(status_code=500, detail="patrick_core_watch_db_error")


@app.get("/api/accumulation/worth-watch")
async def get_worth_watch(category: Optional[str] = Query(None, description="可选：heat_accum / patrick_core / …")):
    """值得关注七类归档：七张独立表 worth_watch_*；每类每轮动态门槛+至多 5 条入库；保留 2 日；各行含 bpc（每小时由定时任务写入）。响应含 tables / categories[].table、bpc_interval、bpc_snapshot_cst。可选 ?category=heat_accum。"""
    try:
        from accumulation_radar import (
            WORTH_HIGHLIGHT_CATEGORY_ORDER,
            init_db,
            load_worth_highlight_watchlist_from_db,
        )

        if category is not None and str(category).strip():
            cat = str(category).strip()
            if cat not in set(WORTH_HIGHLIGHT_CATEGORY_ORDER):
                raise HTTPException(status_code=400, detail=f"unknown category: {cat}")
        else:
            cat = None

        conn = init_db()
        try:
            data = load_worth_highlight_watchlist_from_db(conn, category=cat)
        finally:
            conn.close()
        if not data.get("items"):
            data.setdefault(
                "message",
                "尚无归档，请等待整点 :30 扫描或点击「刷新」后重试。",
            )
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("worth_highlight watchlist read failed: %s", e)
        raise HTTPException(status_code=500, detail="worth_watch_db_error")


@app.get("/api/zct-vwap/summary")
async def get_zct_vwap_summary():
    """ZCT VWAP 虚拟信号汇总：持仓笔数、已结算、累计 pnl_usdt、全局胜率、`per_symbol` 按标的胜率与笔数。"""
    try:
        from zct_vwap_api import load_zct_vwap_summary

        return load_zct_vwap_summary()
    except Exception as e:
        logger.warning("zct_vwap summary failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_vwap_summary_error")


@app.get("/api/zct-vwap/equity-curve")
async def get_zct_vwap_equity_curve():
    """按结算日（UTC 日历日）累计虚拟盈亏曲线，供看板折线图。"""
    try:
        from zct_vwap_api import load_zct_equity_curve

        return load_zct_equity_curve()
    except Exception as e:
        logger.warning("zct_vwap equity curve failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_vwap_equity_curve_error")


@app.get("/api/zct-vwap/signals")
async def get_zct_vwap_signals(
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    symbol: Optional[str] = Query(None, description="如 BTCUSDT"),
    status: Optional[str] = Query(
        None,
        description="all（默认）| open（持仓中）| settled（已结算）",
    ),
):
    """分页列出 ZCT VWAP 扫描入库的信号（含 SL/TP、虚拟名义与结算结果）。"""
    try:
        from zct_vwap_api import load_zct_vwap_signals

        return load_zct_vwap_signals(
            limit=limit,
            offset=offset,
            symbol=symbol,
            status=status or "all",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.warning("zct_vwap signals failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_vwap_signals_error")


class ZctVwapManualPatchBody(BaseModel):
    """实盘补充：部分更新；省略的字段不改写数据库。"""

    manual_entry_price: Optional[float] = Field(
        default=None,
        description="实盘成交价；可显式传 null 清空",
    )
    manual_exit_price: Optional[float] = Field(
        default=None,
        description="实盘平仓价；可显式传 null 清空",
    )
    manual_notes: Optional[str] = Field(
        default=None,
        description="实盘备注",
    )


@app.patch("/api/zct-vwap/signals/{signal_id}")
async def patch_zct_vwap_signal(signal_id: int, body: ZctVwapManualPatchBody):
    """更新 ZCT VWAP 信号的实盘入场/平仓价与备注（不影响脚本虚拟字段）。"""
    try:
        from zct_vwap_api import patch_zct_vwap_manual

        updates = body.model_dump(exclude_unset=True)
        if not updates:
            raise HTTPException(status_code=400, detail="no_fields_to_update")
        out = patch_zct_vwap_manual(signal_id, updates)
        if not out.get("ok"):
            if out.get("error") == "not_found":
                raise HTTPException(status_code=404, detail="signal_not_found")
            raise HTTPException(status_code=500, detail="zct_vwap_patch_failed")
        return out
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.warning("zct_vwap patch failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_vwap_patch_error")


@app.post("/api/zct-vwap/maintenance/clear-db")
async def post_zct_vwap_clear_db():
    """
    清空 ZCT VWAP：`zct_vwap_signals`（每标的快照）与 `zct_vwap_settlements`（已平仓历史）。
    无鉴权：请勿将 API 暴露在公网。
    """
    try:
        from accumulation_radar import init_db

        conn = init_db()
        try:
            cur = conn.cursor()
            n_settle = 0
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='zct_vwap_settlements'"
            )
            if cur.fetchone():
                cur.execute("SELECT COUNT(*) FROM zct_vwap_settlements")
                n_settle = int(cur.fetchone()[0] or 0)
                cur.execute("DELETE FROM zct_vwap_settlements")
            cur.execute("SELECT COUNT(*) FROM zct_vwap_signals")
            n_sig = int(cur.fetchone()[0] or 0)
            cur.execute("DELETE FROM zct_vwap_signals")
            conn.commit()
            logger.warning(
                "zct_vwap clear-db: deleted signals=%s settlements=%s",
                n_sig,
                n_settle,
            )
            return {
                "ok": True,
                "deleted_zct_vwap_signals": n_sig,
                "deleted_zct_vwap_settlements": n_settle,
            }
        finally:
            conn.close()
    except Exception as e:
        logger.exception("zct_vwap clear-db failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_vwap_clear_db_failed")


@app.get("/api/zct-hot-oi/summary")
async def get_zct_hot_oi_summary():
    """ZCT · 🔥⚡热度+OI lane：汇总（zct_hot_oi_*）。"""
    try:
        from zct_vwap_api import load_zct_vwap_summary

        return load_zct_vwap_summary(lane="hot_oi")
    except Exception as e:
        logger.warning("zct_hot_oi summary failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_hot_oi_summary_error")


@app.get("/api/zct-hot-oi/equity-curve")
async def get_zct_hot_oi_equity_curve():
    """热度+OI lane：按结算日累计虚拟盈亏曲线。"""
    try:
        from zct_vwap_api import load_zct_equity_curve

        return load_zct_equity_curve(lane="hot_oi")
    except Exception as e:
        logger.warning("zct_hot_oi equity curve failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_hot_oi_equity_curve_error")


@app.get("/api/zct-hot-oi/signals")
async def get_zct_hot_oi_signals(
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    symbol: Optional[str] = Query(None, description="如 BTCUSDT"),
    status: Optional[str] = Query(
        None,
        description="all（默认）| open（持仓中）| settled（已结算）",
    ),
):
    try:
        from zct_vwap_api import load_zct_vwap_signals

        return load_zct_vwap_signals(
            limit=limit,
            offset=offset,
            symbol=symbol,
            status=status or "all",
            lane="hot_oi",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.warning("zct_hot_oi signals failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_hot_oi_signals_error")


@app.patch("/api/zct-hot-oi/signals/{signal_id}")
async def patch_zct_hot_oi_signal(signal_id: int, body: ZctVwapManualPatchBody):
    try:
        from zct_vwap_api import patch_zct_vwap_manual

        updates = body.model_dump(exclude_unset=True)
        if not updates:
            raise HTTPException(status_code=400, detail="no_fields_to_update")
        out = patch_zct_vwap_manual(signal_id, updates, lane="hot_oi")
        if not out.get("ok"):
            if out.get("error") == "not_found":
                raise HTTPException(status_code=404, detail="signal_not_found")
            raise HTTPException(status_code=500, detail="zct_hot_oi_patch_failed")
        return out
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.warning("zct_hot_oi patch failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_hot_oi_patch_error")


@app.post("/api/zct-hot-oi/maintenance/clear-db")
async def post_zct_hot_oi_clear_db():
    """清空 zct_hot_oi_signals 与 zct_hot_oi_settlements。"""
    try:
        from accumulation_radar import init_db

        conn = init_db()
        try:
            cur = conn.cursor()
            n_settle = 0
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='zct_hot_oi_settlements'"
            )
            if cur.fetchone():
                cur.execute("SELECT COUNT(*) FROM zct_hot_oi_settlements")
                n_settle = int(cur.fetchone()[0] or 0)
                cur.execute("DELETE FROM zct_hot_oi_settlements")
            cur.execute("SELECT COUNT(*) FROM zct_hot_oi_signals")
            n_sig = int(cur.fetchone()[0] or 0)
            cur.execute("DELETE FROM zct_hot_oi_signals")
            conn.commit()
            logger.warning(
                "zct_hot_oi clear-db: deleted signals=%s settlements=%s",
                n_sig,
                n_settle,
            )
            return {
                "ok": True,
                "deleted_zct_hot_oi_signals": n_sig,
                "deleted_zct_hot_oi_settlements": n_settle,
            }
        finally:
            conn.close()
    except Exception as e:
        logger.exception("zct_hot_oi clear-db failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_hot_oi_clear_db_failed")


class VpRegimeScanBody(BaseModel):
    """POST /api/vp-regime/scan：同步跑一轮 vp_regime_scanner（可能数十秒，勿并发狂点）。"""

    symbols: Optional[str] = Field(
        default=None,
        description="逗号分隔 USDT 永续，如 BTCUSDT,ETHUSDT；指定时与 watchlist 互斥",
    )
    watchlist: bool = Field(default=False, description="为 True 且未传 symbols 时从收筹池 watchlist 取标的")
    persist: bool = Field(default=True, description="是否写入 vp_regime_snapshots")
    notify_tg: bool = Field(default=True, description="是否发 Telegram；关可避免推送")


@app.post("/api/vp-regime/scan")
async def post_vp_regime_scan(body: VpRegimeScanBody = Body(default_factory=VpRegimeScanBody)):
    """
    量价环境扫描（VolUSD 代理 + 三态量环境）；与定时无关，供维护面板手动试跑。
    """
    from starlette.concurrency import run_in_threadpool

    sy = (body.symbols or "").strip()
    if sy and body.watchlist:
        raise HTTPException(status_code=400, detail="symbols 与 watchlist 互斥")
    ov: Optional[List[str]] = None
    if sy:
        ov = [x.strip() for x in sy.split(",") if x.strip()]
        if not ov:
            raise HTTPException(status_code=400, detail="symbols 无效或为空")
    wl: Optional[bool] = True if (body.watchlist and not ov) else None

    def _work():
        import vp_regime_scanner as vp

        return vp.run_scan(
            use_db=body.persist,
            use_tg=body.notify_tg,
            symbols_override=ov,
            watchlist_request=wl,
            quiet=True,
        )

    try:
        return await run_in_threadpool(_work)
    except Exception as e:
        logger.exception("vp_regime scan failed: %s", e)


class ZctTouchPoolScanBody(BaseModel):
    """POST /api/zct-vwap/touch-pool-scan：近 N 天 walk-forward + 触轨池筛选（同步，可能数分钟）。"""

    symbols: str = Field(
        default="ZECUSDT,ONDOUSDT,1000SHIBUSDT",
        description="Comma-separated USDT perpetual symbols",
    )
    days: float = Field(default=3.0, ge=0.25, le=30.0)
    min_touch_trades: int = Field(default=130, ge=0, le=200_000)
    min_touch_win_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    strict_greater_touch: bool = Field(default=False)
    strict_greater_rate: bool = Field(default=False)
    signal_interval: str = Field(default="1m", description="1m or 5m")
    sleep_between_symbols: float = Field(default=0.25, ge=0.0, le=10.0)


@app.post("/api/zct-vwap/touch-pool-scan")
async def post_zct_touch_pool_scan(
    body: ZctTouchPoolScanBody = Body(default_factory=ZctTouchPoolScanBody),
):
    """ZCT walk-forward touch-pool filter; same as zct_vwap_asset_pool (no DB write)."""
    from starlette.concurrency import run_in_threadpool

    iv = str(body.signal_interval or "1m").strip().lower()
    if iv not in ("1m", "5m"):
        raise HTTPException(status_code=400, detail="signal_interval must be 1m or 5m")

    syms = [x.strip().upper() for x in (body.symbols or "").split(",") if x.strip()]
    if not syms:
        raise HTTPException(status_code=400, detail="empty_symbols")

    def _work():
        from zct_vwap_asset_pool import run_asset_pool_scan

        out, _summary = run_asset_pool_scan(
            days=float(body.days),
            symbols=syms,
            ignore_db_cooldown=True,
            sleep_between_symbols=float(body.sleep_between_symbols),
            signal_interval=iv,
            min_touch_trades=int(body.min_touch_trades),
            strict_greater_touch=bool(body.strict_greater_touch),
            min_touch_win_rate=float(body.min_touch_win_rate),
            strict_greater_rate=bool(body.strict_greater_rate),
            quiet=True,
        )
        return {"ok": True, "pool": out}

    try:
        return await run_in_threadpool(_work)
    except Exception as e:
        logger.exception("zct touch_pool scan failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_touch_pool_scan_failed")


@app.get("/dashboard/zct-vwap", response_class=HTMLResponse)
async def zct_vwap_dashboard_page():
    """ZCT VWAP 虚拟信号看板（静态页 + 调用上方 JSON API）。"""
    path = Path(__file__).resolve().parent / "static" / "zct_vwap_dashboard.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="dashboard_zct_vwap_not_found")
    return HTMLResponse(content=path.read_text(encoding="utf-8"))


class ClearWatchTablesBody(BaseModel):
    """清理看盘表（无鉴权：请勿将 API 长期暴露在公网）。"""

    tables: List[str] = Field(
        default_factory=lambda: ["ambush_watch"],
        description="watchlist（收筹池）/ focus_watch / ambush / heat / patrick；worth 侧可用 worth_watch_all 或单表 worth_watch_heat_accum 等",
    )


@app.post("/api/accumulation/maintenance/clear-watch-tables")
async def post_clear_watch_tables(body: ClearWatchTablesBody):
    """
    清空看盘 SQLite 表。

    清库后请再调一次「OI 刷新」或等整点扫描，以按新规则写回数据。
    """
    from accumulation_radar import WORTH_WATCH_TABLE_BY_CATEGORY

    _worth_tables = set(WORTH_WATCH_TABLE_BY_CATEGORY.values())
    allowed = {
        "watchlist",
        "focus_watch",
        "ambush_watch",
        "heat_accum_watch",
        "patrick_core_watch",
        "worth_highlight_watch",
        "worth_watch_all",
        *_worth_tables,
    }
    tables = [t.strip() for t in body.tables if t and str(t).strip()]
    if not tables:
        tables = ["ambush_watch"]
    unknown = [t for t in tables if t not in allowed]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown tables: {unknown}")

    try:
        from accumulation_radar import (
            clear_all_worth_watch_category_tables,
            clear_ambush_watch_table,
            clear_heat_accum_watch_table,
            clear_one_worth_watch_category_table,
            clear_patrick_core_watch_table,
            clear_focus_watch_table,
            clear_watchlist_table,
            init_db,
            patch_oi_radar_snapshot_after_watchlist_clear,
            patch_oi_radar_snapshot_watchlists_from_db,
        )

        conn = init_db()
        try:
            cleared: Dict[str, Any] = {}
            if "watchlist" in tables:
                cleared["watchlist"] = clear_watchlist_table(conn)
            if "focus_watch" in tables:
                cleared["focus_watch"] = clear_focus_watch_table(conn)
            if "ambush_watch" in tables:
                cleared["ambush_watch"] = clear_ambush_watch_table(conn)
            if "heat_accum_watch" in tables:
                cleared["heat_accum_watch"] = clear_heat_accum_watch_table(conn)
            if "patrick_core_watch" in tables:
                cleared["patrick_core_watch"] = clear_patrick_core_watch_table(conn)
            worth_tbls = {t for t in tables if t in set(WORTH_WATCH_TABLE_BY_CATEGORY.values())}
            worth_all = (
                "worth_watch_all" in tables
                or "worth_highlight_watch" in tables
            )
            if worth_all:
                cleared.update(clear_all_worth_watch_category_tables(conn))
            else:
                for t in sorted(worth_tbls):
                    cleared[t] = clear_one_worth_watch_category_table(conn, t)
            try:
                if "watchlist" in tables:
                    patch_oi_radar_snapshot_after_watchlist_clear(conn)
                else:
                    patch_oi_radar_snapshot_watchlists_from_db(conn)
            except Exception:
                logger.exception("patch oi_radar snapshot after clear failed")
            logger.warning(
                "maintenance clear-watch-tables tables=%s cleared=%s",
                tables,
                cleared,
            )
            return {"ok": True, "cleared_rows": cleared}
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("clear watch tables failed: %s", e)
        raise HTTPException(status_code=500, detail="clear_failed")


class TriggerCronBody(BaseModel):
    """手动触发与 APScheduler 注册项相同的子进程任务。"""

    task: str = Field(
        ...,
        description="pool | heat_watch | heat_zones | heat_bpc | oi | s2_funding | s6_alpha | zct_vwap | zct_vwap_resolve | zct_hot_oi | zct_hot_oi_resolve",
    )


_CRON_TASK_FUNCS: Dict[str, Any] = {
    "pool": run_pool_task,
    "heat_watch": run_heat_watch_refresh_task,
    "heat_zones": run_heat_watch_refresh_task,
    "heat_bpc": run_heat_watch_refresh_task,
    "oi": run_oi_task,
    "s2_funding": run_s2_oi_funding_task,
    "s6_alpha": run_s6_futures_alpha_task,
    "zct_vwap": run_zct_vwap_signal_task,
    "zct_vwap_resolve": run_zct_vwap_resolve_only_task,
    "zct_hot_oi": run_zct_hot_oi_signal_task,
    "zct_hot_oi_resolve": run_zct_hot_oi_resolve_only_task,
}


@app.post("/api/accumulation/maintenance/trigger-cron")
async def post_trigger_accumulation_cron(body: TriggerCronBody):
    """
    在后台线程执行与定时任务相同的逻辑（子进程跑脚本），HTTP 立即返回。

    - pool: accumulation_radar pool（定时每日 10:00 CST）
    - heat_watch: 热度看盘整表（现价/摘要 + 1h BPC，定时每小时 xx:07）
    - heat_zones / heat_bpc: 与 heat_watch 相同（兼容旧 task 名）
    - oi: accumulation_radar oi（定时每小时 :30）
    - s2_funding: s2_oi_funding_rate_scanner（定时每时 :05）
    - s6_alpha: s6 期货 Alpha（定时每时 :25，与 S6_FUTURES_ALPHA_SCHEDULER_ENABLED 无关可手动跑）
    - zct_vwap: ZCT VWAP 全量扫描（与定时同源子进程，间隔见 ZCT_VWAP_SCAN_INTERVAL_MINUTES）
    - zct_vwap_resolve: 仅纸面结算（--resolve-only，与定时 ZCT_VWAP_RESOLVE_INTERVAL_MINUTES 同源）
    - zct_hot_oi: 🔥⚡热度+OI 扫描（worth_watch_hot_oi → zct_hot_oi_*；间隔 ZCT_HOT_OI_SCAN_INTERVAL_MINUTES）
    - zct_hot_oi_resolve: 🔥⚡热度+OI 仅结算（ZCT_HOT_OI_RESOLVE_INTERVAL_MINUTES）
    """
    key = (body.task or "").strip()
    fn = _CRON_TASK_FUNCS.get(key)
    if fn is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown task {key!r}; allowed: {sorted(_CRON_TASK_FUNCS.keys())}",
        )

    def _work() -> None:
        try:
            fn()
        except Exception:
            logger.exception("manual trigger-cron task=%s failed", key)

    threading.Thread(target=_work, daemon=True).start()
    return {"accepted": True, "task": key}


@app.post("/api/accumulation/oi-radar/refresh")
async def post_accumulation_oi_radar_refresh():
    """
    在后台线程执行完整扫描并写入 `oi_radar_snapshot.json`，立即返回，避免 HTTP 超时。
    与 GET 快照配合：前端轮询 GET 直至 `ok` 为 true。
    """
    if not _oi_radar_refresh_lock.acquire(blocking=False):
        return {"accepted": False, "busy": True, "message": "已有扫描任务在执行中"}

    def _work():
        try:
            from accumulation_radar import init_db, run_oi_hourly_radar

            conn = init_db()
            try:
                run_oi_hourly_radar(conn, notify=False)
            finally:
                conn.close()
        except Exception:
            logger.exception("OI radar background refresh failed")
        finally:
            _oi_radar_refresh_lock.release()

    threading.Thread(target=_work, daemon=True).start()
    return {"accepted": True, "busy": False}


def _run_refresh_heat_watch_background() -> None:
    try:
        logger.info("manual refresh heat watch (full) accepted")
        data = _refresh_heat_accum_watch_full_once()
        logger.info(
            "manual refresh heat watch done: prices=%s bpc=%s failed_klines=%s worth_bpc=%s worth_bpc_fail_kl=%s",
            data.get("recalculated_prices"),
            data.get("bpc_recalculated"),
            data.get("bpc_failed_klines"),
            data.get("worth_watch_bpc_recalculated"),
            data.get("worth_watch_bpc_failed_klines"),
        )
    except Exception:
        logger.exception("manual refresh heat watch failed")
    finally:
        _heat_watch_refresh_lock.release()


@app.post("/api/accumulation/maintenance/refresh-heat-watch")
async def post_refresh_heat_watch():
    """热度看盘整表：现价/摘要 + 1h BPC；并刷新 worth_watch_* 七表各行 1H BPC（后台线程）。与定时 heat_watch_refresh 同源。"""
    if not _heat_watch_refresh_lock.acquire(blocking=False):
        return {"accepted": False, "busy": True, "message": "已有热度看盘刷新任务在执行中"}
    threading.Thread(target=_run_refresh_heat_watch_background, daemon=True).start()
    return {"accepted": True, "busy": False}


@app.post("/api/accumulation/maintenance/refresh-heat-zones")
async def post_refresh_heat_zones():
    """兼容旧路径：等同 refresh-heat-watch（现价 + BPC）。"""
    if not _heat_watch_refresh_lock.acquire(blocking=False):
        return {"accepted": False, "busy": True, "message": "已有热度看盘刷新任务在执行中"}
    threading.Thread(target=_run_refresh_heat_watch_background, daemon=True).start()
    return {"accepted": True, "busy": False}


@app.post("/api/accumulation/maintenance/refresh-heat-bpc")
async def post_refresh_heat_bpc():
    """兼容旧路径：等同 refresh-heat-watch（现价 + BPC + 值得关注七表 BPC）。"""
    if not _heat_watch_refresh_lock.acquire(blocking=False):
        return {"accepted": False, "busy": True, "message": "已有热度看盘刷新任务在执行中"}
    threading.Thread(target=_run_refresh_heat_watch_background, daemon=True).start()
    return {"accepted": True, "busy": False}


def _filter_s2_funding_signals_last_days(signals: List[Dict[str, Any]], days: int = 2) -> List[Dict[str, Any]]:
    """Keep entries with recorded_at within last `days` (Asia/Shanghai cutoff)."""
    cst = timezone(timedelta(hours=8))
    cutoff = datetime.now(cst) - timedelta(days=days)
    out: List[Dict[str, Any]] = []
    for row in signals:
        if not isinstance(row, dict):
            continue
        ts = row.get("recorded_at")
        if not ts or not isinstance(ts, str):
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=cst)
            if dt >= cutoff:
                out.append(row)
        except Exception:
            continue
    out.sort(key=lambda r: str(r.get("recorded_at", "")), reverse=True)
    return out


def _s6_candidates_s_only(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """归档里 candidates 仅展示 S；旧数据含 A/B 时在 API 层剥掉。"""
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        row = dict(r)
        c = row.get("candidates")
        if isinstance(c, list):
            row["candidates"] = [
                x for x in c if isinstance(x, dict) and x.get("strength") == "S"
            ]
            row["candidate_count"] = len(row["candidates"])
        out.append(row)
    return out


def _s6_signals_history_path() -> Path:
    return Path(__file__).resolve().parent / "s6_signals_history.json"


def _s6_trades_json_path() -> Path:
    return Path(__file__).resolve().parent / "trades.json"


def _s6_compute_balance_usd(trades_root: Dict[str, Any]) -> Tuple[float, float]:
    """(balance_after_closed, initial_balance) — 与 s6 get_balance 一致。"""
    initial = float(trades_root.get("initial_balance", 100.0))
    bal = initial
    trades = trades_root.get("trades")
    if not isinstance(trades, list):
        return bal, initial
    for t in trades:
        if isinstance(t, dict) and t.get("status") == "closed" and t.get("pnl_usd") is not None:
            try:
                bal += float(t["pnl_usd"])
            except (TypeError, ValueError):
                continue
    return bal, initial


@app.get("/api/s6/autonomous-alpha")
async def get_s6_autonomous_alpha():
    """
    s6 期货 Alpha：近 2 日每小时扫描归档 + 当前模拟持仓（trades.json）。
    """
    sig_path = _s6_signals_history_path()
    signals: List[Dict[str, Any]] = []
    if sig_path.is_file():
        try:
            raw = json.loads(sig_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("signals"), list):
                signals = raw["signals"]
        except Exception as e:
            logger.warning("s6 signals history read failed: %s", e)
            raise HTTPException(status_code=500, detail="s6_signals_corrupt")
    filtered = _filter_s2_funding_signals_last_days(signals, 2)
    filtered = _s6_candidates_s_only(filtered)

    trades_path = _s6_trades_json_path()
    open_positions: List[Dict[str, Any]] = []
    balance_usd = 100.0
    initial_balance = 100.0
    if trades_path.is_file():
        try:
            troot = json.loads(trades_path.read_text(encoding="utf-8"))
            if isinstance(troot, dict):
                balance_usd, initial_balance = _s6_compute_balance_usd(troot)
                for t in troot.get("trades") or []:
                    if isinstance(t, dict) and t.get("status") == "open":
                        open_positions.append(t)
        except Exception as e:
            logger.warning("s6 trades.json read failed: %s", e)

    return {
        "ok": True,
        "signals": filtered,
        "day_window": 2,
        "source": "disk",
        "count": len(filtered),
        "initial_balance": initial_balance,
        "balance_usd": round(balance_usd, 4),
        "open_positions": open_positions,
        "open_count": len(open_positions),
    }


@app.get("/api/s2/funding-signals")
async def get_s2_funding_signals():
    """
    返回近 2 日「费率刚转负 + OI 涨」强信号（与 TG 同源）。
    持久化：accumulation.db 表 s2_funding_signals（原 JSON 由脚本启动时迁移）。
    """
    try:
        from s2_oi_funding_rate_scanner import get_s2_funding_signals_for_api

        return get_s2_funding_signals_for_api(2)
    except Exception as e:
        logger.warning("s2 funding signals read failed: %s", e)
        raise HTTPException(status_code=500, detail="s2_signals_db_error")


# ============== Main ==============

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    # Use "::" for IPv6 support (required by Railway)
    uvicorn.run(app, host="0.0.0.0", port=port)

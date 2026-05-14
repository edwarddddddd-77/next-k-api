"""
Next K - Multi-Asset K-Line Weather Forecast API

A probabilistic K-line prediction tool powered by the Kronos foundation model.
Supports: Crypto, Stocks, Forex
"""

import asyncio
import base64
import io
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
from typing import Any, Dict, List, Literal, Optional, Tuple
import time as time_module

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
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
    1, int(os.getenv("ZCT_VWAP_SCAN_INTERVAL_MINUTES", "15") or 15)
)
# 0 = 不注册独立结算任务（仍依赖全量扫描脚本内的 pre/post resolve）
ZCT_VWAP_RESOLVE_INTERVAL_MINUTES = max(
    0, int(os.getenv("ZCT_VWAP_RESOLVE_INTERVAL_MINUTES", "5") or 5)
)


# ============== Asset Types & Symbols ==============

class AssetType(str, Enum):
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"


class TimeFrame(str, Enum):
    """Supported timeframes for prediction."""
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


# Timeframe configurations
TIMEFRAME_CONFIG = {
    TimeFrame.H1: {"label": "1 Hour", "hours": 1, "limit": 100, "horizon_default": 24},
    TimeFrame.H4: {"label": "4 Hours", "hours": 4, "limit": 100, "horizon_default": 24},
    TimeFrame.D1: {"label": "1 Day", "hours": 24, "limit": 100, "horizon_default": 14},
    TimeFrame.W1: {"label": "1 Week", "hours": 168, "limit": 52, "horizon_default": 8},
}


class TimeFrame(str, Enum):
    """Supported timeframes for prediction."""
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


# Timeframe configurations
TIMEFRAME_CONFIG = {
    TimeFrame.H1: {"label": "1 Hour", "hours": 1, "limit": 100, "horizon_default": 24},
    TimeFrame.H4: {"label": "4 Hours", "hours": 4, "limit": 100, "horizon_default": 24},
    TimeFrame.D1: {"label": "1 Day", "hours": 24, "limit": 100, "horizon_default": 14},
    TimeFrame.W1: {"label": "1 Week", "hours": 168, "limit": 52, "horizon_default": 8},
}


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
    predictor = None
    ccxt_exchange = None  # For crypto
    yfinance_available = False  # For stocks/forex
    is_ready = False
    startup_time = None
    # True when Kronos was not loaded at startup (default off; opt-in via ENABLE_KRONOS_MODEL=1)
    kronos_skipped = False


state = AppState()


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")

# HuggingFace model paths
TOKENIZER_PATH = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PATH = "NeoQuasar/Kronos-small"


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


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
    """每小时：heat_accum_watch 现价/摘要；若 BPC_FEATURE_ENABLED=1 则另含 1h BPC + worth/focus 各行 BPC。"""
    if not _heat_watch_refresh_lock.acquire(blocking=False):
        logger.info("热度看盘整表刷新跳过：已有任务在执行")
        return
    try:
        logger.info("开始执行热度看盘整表刷新（现价/摘要 + 可选 BPC，见 BPC_FEATURE_ENABLED）...")
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


def _zct_touch_pool_child_env() -> dict:
    """主 ZCT lane：全量扫描只跑触轨入选表中的标的（zct_vwap_touch_pool）。"""
    env = os.environ.copy()
    env["ZCT_TOUCH_POOL_UNIVERSE"] = "1"
    return env


def _run_zct_vwap_signal_subprocess() -> None:
    """与 `_run_accumulation_radar_subprocess` 同范式：`cwd`=脚本目录；脚本内自载 `.env.oi`，TG 与 accumulation 共用变量。"""
    logger.info("Starting zct_vwap_signal_scanner subprocess")
    try:
        subprocess.run(
            [sys.executable, str(_ZCT_VWAP_SCRIPT)],
            cwd=str(_ZCT_VWAP_SCRIPT.parent),
            env=_zct_touch_pool_child_env(),
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

    # Kronos + PyTorch are optional (heavy): load only when ENABLE_KRONOS_MODEL=1.
    # SKIP_KRONOS_MODEL=1 still forces skip even if ENABLE is set.
    load_kronos = _env_truthy("ENABLE_KRONOS_MODEL") and not _env_truthy(
        "SKIP_KRONOS_MODEL"
    )
    if not load_kronos:
        state.kronos_skipped = True
        if _env_truthy("SKIP_KRONOS_MODEL"):
            logger.info(
                "SKIP_KRONOS_MODEL is set — Kronos/PyTorch will not load "
                "(weather/chart/simulate/backtest disabled; radar & accumulation APIs unchanged)"
            )
        else:
            logger.info(
                "Kronos model startup skipped (set ENABLE_KRONOS_MODEL=1 to load; "
                "weather/chart/simulate/backtest disabled; radar & accumulation APIs unchanged)"
            )
    else:
        asyncio.create_task(initialize_model())

    # Kronos + PyTorch are optional (heavy): load only when ENABLE_KRONOS_MODEL=1.
    # SKIP_KRONOS_MODEL=1 still forces skip even if ENABLE is set.
    load_kronos = _env_truthy("ENABLE_KRONOS_MODEL") and not _env_truthy(
        "SKIP_KRONOS_MODEL"
    )
    if not load_kronos:
        state.kronos_skipped = True
        if _env_truthy("SKIP_KRONOS_MODEL"):
            logger.info(
                "SKIP_KRONOS_MODEL is set — Kronos/PyTorch will not load "
                "(weather/chart/simulate/backtest disabled; radar & accumulation APIs unchanged)"
            )
        else:
            logger.info(
                "Kronos model startup skipped (set ENABLE_KRONOS_MODEL=1 to load; "
                "weather/chart/simulate/backtest disabled; radar & accumulation APIs unchanged)"
            )
    else:
        asyncio.create_task(initialize_model())

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
    accumulation_scheduler.start()
    app.state.accumulation_scheduler = accumulation_scheduler
    try:
        from accumulation_radar import BPC_FEATURE_ENABLED as _bpc_feature

        logger.info(
            "BPC (1H结构) 定时重算与 TG 延续推送: %s",
            "开启 (BPC_FEATURE_ENABLED=1)" if _bpc_feature else "关闭（默认；不设或设为 0）",
        )
    except Exception:
        logger.warning("could not read BPC_FEATURE_ENABLED from accumulation_radar")

    s6_cron_log = (
        "s6_futures_alpha 每整点后 25 分 (xx:25)"
        if S6_FUTURES_ALPHA_SCHEDULER_ENABLED
        else "s6_futures_alpha 定时已暂停"
    )
    if ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED:
        zct_vwap_log = (
            f"zct_vwap 全量扫描每 {ZCT_VWAP_SCAN_INTERVAL_MINUTES} 分钟（标的=zct_vwap_touch_pool）· "
            f"结算(resolve-only)每 {ZCT_VWAP_RESOLVE_INTERVAL_MINUTES} 分钟（Asia/Shanghai 进程内间隔触发）"
            if ZCT_VWAP_RESOLVE_INTERVAL_MINUTES > 0
            else (
                f"zct_vwap 全量扫描每 {ZCT_VWAP_SCAN_INTERVAL_MINUTES} 分钟（标的=zct_vwap_touch_pool）· "
                "独立结算定时已关闭（ZCT_VWAP_RESOLVE_INTERVAL_MINUTES=0）"
            )
        )
    else:
        zct_vwap_log = "zct_vwap 定时未启用（设 ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED=1）"
    logger.info(
        "后台定时任务已启动: accumulation_radar pool 每日 10:00 CST, "
        "heat_watch 每小时 xx:07（现价/摘要 + 1h BPC）; "
        "oi 每小时 :30; "
        "s2_oi_funding_rate_scanner 每整点后 5 分 (xx:05); "
        + s6_cron_log
        + "; "
        + zct_vwap_log
    )

    yield

    sch = getattr(app.state, "accumulation_scheduler", None)
    if sch is not None:
        sch.shutdown(wait=False)
        app.state.accumulation_scheduler = None

    logger.info("Shutting down...")


async def initialize_model():
    """Load Kronos model from HuggingFace."""
    try:
        logger.info("Loading Kronos model...")
        from kronos import KronosTokenizer, Kronos, KronosPredictor

        tokenizer = await asyncio.get_event_loop().run_in_executor(
            None, lambda: KronosTokenizer.from_pretrained(TOKENIZER_PATH)
        )
        model = await asyncio.get_event_loop().run_in_executor(
            None, lambda: Kronos.from_pretrained(MODEL_PATH)
        )

        state.predictor = KronosPredictor(model, tokenizer)
        state.is_ready = True
        logger.info("Kronos model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load Kronos model: {e}")
        state.is_ready = False


# ============== FastAPI App ==============

app = FastAPI(
    title="Next K",
    description="""
    # Next K - 多資產 K線天氣預報

    基於 Kronos AI 的概率性價格預測工具。

    ## 支持資產類型
    - **加密貨幣** (Crypto): BTC, ETH, SOL, BNB, PEPE
    - **股票** (Stocks): AAPL, GOOGL, MSFT, TSLA, NVDA
    - **外匯** (Forex): EUR/USD, GBP/USD, USD/JPY

    ## 核心功能
    1. **K線天氣預報** - 概率分布預測
    2. **異動雷達** - 市場異常掃描
    3. **交易沙盒** - 模擬驗證交易計劃
    """,
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


class PriceLevel(BaseModel):
    price: float
    probability: float
    type: str


class ForecastBar(BaseModel):
    time: int
    mean: float
    min: float
    max: float
    p10: float
    p25: float
    p75: float
    p90: float


class HistoryBar(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class WeatherForecast(BaseModel):
    symbol: str
    asset_type: str
    timeframe: str = "1h"
    current_price: float
    generated_at: str
    horizon: int
    price_ranges: List[Dict[str, Any]]
    support_levels: List[PriceLevel]
    resistance_levels: List[PriceLevel]
    history: List[HistoryBar]  # Historical K-lines
    forecast: List[ForecastBar]
    upside_prob: float
    volatility: float
    confidence: float
    suggested_entry: Optional[float] = None
    suggested_stop_loss: Optional[float] = None
    suggested_take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    buy_point_score: Optional[float] = None
    buy_point_label: Optional[str] = None
    buy_point_reasons: Optional[List[str]] = None


class SimulationRequest(BaseModel):
    symbol: str
    asset_type: Optional[str] = None
    action: str
    entry_price: Optional[float] = None
    position_size: float = Field(gt=0)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: int = Field(default=1, ge=1, le=100)
    horizon: int = Field(default=24, ge=1, le=168)


class SimulationResult(BaseModel):
    symbol: str
    asset_type: str
    action: str
    entry_price: float
    position_size: float
    leverage: int
    win_rate: float
    expected_pnl: float
    expected_pnl_percent: float
    max_profit: float
    max_loss: float
    hit_take_profit_pct: float
    hit_stop_loss_pct: float
    expired_pct: float
    avg_bars_to_exit: float
    max_drawdown: float
    recommendation: str
    issues: List[str]
    optimizations: List[Dict[str, Any]]


class RadarItem(BaseModel):
    symbol: str
    name: str
    asset_type: str
    anomaly_score: float
    signal: SignalType
    signals: List[str]
    price: float
    price_change: float
    kronos_hint: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    kronos_skipped: bool = False  # True when Kronos not loaded — prediction routes unavailable by design
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


def get_price_scale_factor(price: float) -> float:
    """
    Calculate scale factor for very small price coins (like PEPE).
    Scales prices to ~100 range for better model performance.
    """
    if price <= 0:
        return 1.0
    if price >= 1.0:
        return 1.0  # No scaling needed for normal prices >= $1
    # Scale to bring price to ~100
    import math
    target = 100.0
    scale = target / price
    # Round to power of 10 for clean scaling
    log_scale = math.log10(scale)
    return 10 ** round(log_scale)


def scale_ohlcv_df(df: pd.DataFrame, scale_factor: float) -> pd.DataFrame:
    """Scale OHLCV dataframe prices by scale factor (volume unchanged)."""
    if scale_factor == 1.0:
        return df
    scaled = df.copy()
    for col in ['open', 'high', 'low', 'close']:
        if col in scaled.columns:
            scaled[col] = scaled[col] * scale_factor
    return scaled


def unscale_prediction(result: Dict, scale_factor: float) -> Dict:
    """Unscale prediction results back to original price range."""
    if scale_factor == 1.0:
        return result
    unscaled = result.copy()
    for key in ['mean', 'min', 'max', 'p10', 'p25', 'p75', 'p90']:
        if key in unscaled:
            unscaled[key] = unscaled[key] / scale_factor
    if 'all_samples' in unscaled:
        samples = unscaled['all_samples'].copy()
        # Unscale OHLC columns (0-3), keep volume (4) unchanged
        samples[:, :, 0:4] = samples[:, :, 0:4] / scale_factor
        unscaled['all_samples'] = samples
    return unscaled


def calculate_price_ranges(samples: np.ndarray, current_price: float, num_ranges: int = 6) -> List[Dict]:
    all_prices = samples.flatten()
    min_price, max_price = float(np.min(all_prices)), float(np.max(all_prices))
    range_size = (max_price - min_price) / num_ranges
    ranges = []

    for i in range(num_ranges):
        low = min_price + i * range_size
        high = min_price + (i + 1) * range_size
        count = int(np.sum((all_prices >= low) & (all_prices < high)))
        prob = count / len(all_prices) * 100
        ranges.append({
            "low": smart_round(float(low)),
            "high": smart_round(float(high)),
            "probability": round(float(prob), 1),
            "is_current": bool(low <= current_price <= high)
        })

    return sorted(ranges, key=lambda x: x["probability"], reverse=True)


def find_key_levels(samples: np.ndarray, current_price: float) -> Tuple[List[PriceLevel], List[PriceLevel]]:
    supports, resistances = [], []
    for pct in [10, 25]:
        price = float(np.percentile(samples[:, -1], pct))
        if price < current_price:
            prob = float(np.mean(samples[:, -1] > price) * 100)
            supports.append(PriceLevel(price=smart_round(price), probability=round(prob, 1), type="support"))
    for pct in [75, 90]:
        price = float(np.percentile(samples[:, -1], pct))
        if price > current_price:
            prob = float(np.mean(samples[:, -1] < price) * 100)
            resistances.append(PriceLevel(price=smart_round(price), probability=round(prob, 1), type="resistance"))
    return supports, resistances


def calculate_trend_buy_point_score(price_df: pd.DataFrame) -> Tuple[float, List[str]]:
    """
    基于三阶段策略的趋势买点评分函数（0-100）
    
    第一阶段：左侧定投区 - K线在EMA200下，RSI 24 < 50，向30靠近
    第二阶段：右侧加仓区 - K线穿过EMA200，RSI三线回归50，金叉
    第三阶段：趋势回踩区 - K线在EMA200上，RSI 24 > 50，RSI 6回踩到20
    
    Returns:
        (score, reasons)
    """
    reasons: List[str] = []
    data_len = len(price_df)
    
    # 必须至少200根K线来计算EMA200
    if data_len < 200:
        return 0.0, [f"历史数据不足（当前{data_len}根，需要至少200根K线计算EMA200）"]

    close = price_df["close"].astype(float)
    high = price_df["high"].astype(float)
    low = price_df["low"].astype(float)
    volume = price_df["volume"].astype(float)

    # 计算EMA200（周线级别长期均线）
    ema200 = close.ewm(span=200, adjust=False).mean()
    ema_name = "EMA200"
    
    # 计算RSI 6、RSI 12、RSI 24
    def calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    rsi6 = calculate_rsi(close, 6)
    rsi12 = calculate_rsi(close, 12)
    rsi24 = calculate_rsi(close, 24)

    # 获取最新值
    last_close = float(close.iloc[-1])
    last_ema200 = float(ema200.iloc[-1]) if not pd.isna(ema200.iloc[-1]) else last_close
    
    last_rsi6 = float(rsi6.iloc[-1]) if not pd.isna(rsi6.iloc[-1]) else 50.0
    last_rsi12 = float(rsi12.iloc[-1]) if not pd.isna(rsi12.iloc[-1]) else 50.0
    last_rsi24 = float(rsi24.iloc[-1]) if not pd.isna(rsi24.iloc[-1]) else 50.0
    
    # 获取前几期的值用于判断趋势
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
    prev_ema200 = float(ema200.iloc[-2]) if len(ema200) >= 2 and not pd.isna(ema200.iloc[-2]) else last_ema200
    
    prev_rsi6 = float(rsi6.iloc[-2]) if len(rsi6) >= 2 and not pd.isna(rsi6.iloc[-2]) else last_rsi6
    prev_rsi12 = float(rsi12.iloc[-2]) if len(rsi12) >= 2 and not pd.isna(rsi12.iloc[-2]) else last_rsi12
    prev_rsi24 = float(rsi24.iloc[-2]) if len(rsi24) >= 2 and not pd.isna(rsi24.iloc[-2]) else last_rsi24

    score = 0.0
    stage = None

    # 判断价格与EMA200的关系
    price_above_ema200 = last_close > last_ema200
    price_below_ema200 = last_close < last_ema200
    price_crossing_ema200 = (prev_close <= prev_ema200 and last_close > last_ema200) or \
                           (prev_close >= prev_ema200 and last_close < last_ema200)
    
    # 判断是否在EMA200附近纠缠（距离在2%以内）
    dist_to_ema200_pct = abs(last_close - last_ema200) / (last_ema200 + 1e-8) * 100
    entangled_with_ema200 = dist_to_ema200_pct <= 2.0

    # 检测底背离：价格创新低，但RSI不创新低
    def detect_bottom_divergence(prices: pd.Series, rsi: pd.Series, lookback: int = 20) -> bool:
        if len(prices) < lookback * 2:
            return False
        
        # 获取最近两段数据
        recent_prices = prices.iloc[-lookback:]
        recent_rsi = rsi.iloc[-lookback:]
        prev_prices = prices.iloc[-lookback*2:-lookback] if len(prices) >= lookback * 2 else pd.Series(dtype=float)
        
        if len(prev_prices) == 0:
            return False
        
        # 找到最近的最低点
        price_low = recent_prices.min()
        price_low_pos = recent_prices.values.argmin()
        rsi_at_low = recent_rsi.iloc[price_low_pos]
        
        # 检查之前是否有更低的价格
        prev_low = prev_prices.min()
        
        # 底背离：价格创新低，但RSI不创新低
        if price_low < prev_low:
            # 找到之前最低点对应的RSI
            prev_low_pos = prev_prices.values.argmin()
            prev_rsi = rsi.iloc[len(prices) - lookback*2 + prev_low_pos] if len(prices) >= lookback*2 else 50.0
            
            if rsi_at_low > prev_rsi:
                return True
        return False

    # 检测RSI金叉
    rsi6_cross_rsi24_up = prev_rsi6 <= prev_rsi24 and last_rsi6 > last_rsi24
    rsi12_cross_rsi24_up = prev_rsi12 <= prev_rsi24 and last_rsi12 > last_rsi24
    rsi_golden_cross = rsi6_cross_rsi24_up or rsi12_cross_rsi24_up

    # 判断RSI三线是否都在50附近（45-55区间）
    rsi_all_near_50 = (45 <= last_rsi6 <= 55) and (45 <= last_rsi12 <= 55) and (45 <= last_rsi24 <= 55)
    
    # 判断RSI三线是否都在50上方且发散向上
    rsi_all_above_50 = (last_rsi6 > 50) and (last_rsi12 > 50) and (last_rsi24 > 50)
    rsi_expanding = (last_rsi6 > last_rsi12 > last_rsi24) and (last_rsi6 > prev_rsi6)

    # ========== 第一阶段：左侧定投区 ==========
    if price_below_ema200 and last_rsi24 < 50:
        stage = "左侧定投区"
        base_score = 30  # 基础分
        
        # RSI 24向30靠近（超卖区）
        if 30 <= last_rsi24 < 40:
            score += 25
            reasons.append(f"RSI 24 处于超卖区 ({last_rsi24:.1f})，接近30支撑")
        elif 25 <= last_rsi24 < 30:
            score += 30
            reasons.append(f"RSI 24 深度超卖 ({last_rsi24:.1f})，接近底部")
        elif last_rsi24 < 25:
            score += 20
            reasons.append(f"RSI 24 极度超卖 ({last_rsi24:.1f})，可能超跌")
        else:
            score += 15
            reasons.append(f"RSI 24 在50下方 ({last_rsi24:.1f})，等待向30靠近")
        
        # 底背离检测（使用RSI 9/12，这里用RSI 12代替）
        if detect_bottom_divergence(close, rsi12, lookback=30):
            score += 20
            reasons.append("检测到底背离信号（价格创新低，RSI不创新低）")
        
        # RSI 6/12超卖
        if last_rsi6 < 30 or last_rsi12 < 30:
            score += 10
            reasons.append(f"RSI 6/12 超卖 (RSI6={last_rsi6:.1f}, RSI12={last_rsi12:.1f})")
        
        score += base_score
        reasons.append("价格在EMA200下方，适合定投布局")

    # ========== 第二阶段：右侧加仓区 ==========
    elif (price_crossing_ema200 or entangled_with_ema200) and last_rsi24 >= 45:
        stage = "右侧加仓区"
        base_score = 50  # 基础分
        
        # 价格突破EMA200
        if price_crossing_ema200 and last_close > last_ema200:
            score += 20
            reasons.append("价格由下往上突破EMA200")
        elif entangled_with_ema200:
            score += 15
            reasons.append("价格与EMA200反复纠缠")
        
        # RSI三线回归50
        if rsi_all_near_50:
            score += 15
            reasons.append("RSI 6/12/24 三线回归50分界线附近")
        
        # RSI金叉
        if rsi_golden_cross:
            score += 15
            if rsi6_cross_rsi24_up:
                reasons.append("RSI 6 向上金叉 RSI 24")
            if rsi12_cross_rsi24_up:
                reasons.append("RSI 12 向上金叉 RSI 24")
        
        # RSI三线在50上方发散
        if rsi_all_above_50 and rsi_expanding:
            score += 10
            reasons.append("RSI 三线在50上方呈发散状向上（多头扩张）")
        
        # 共振信号（最强买入信号）
        if price_crossing_ema200 and last_close > last_ema200 and \
           rsi_all_above_50 and rsi_golden_cross:
            score += 20
            reasons.append("【共振信号】价格突破EMA200 + RSI三线站上50 + RSI金叉")
        
        score += base_score
        reasons.append("右侧加仓区：趋势反转窗口，适合重仓布局")

    # ========== 第三阶段：趋势回踩区 ==========
    elif price_above_ema200 and last_rsi24 > 50:
        stage = "趋势回踩区"
        base_score = 40  # 基础分
        
        # RSI 24稳在50以上
        if last_rsi24 > 50:
            score += 15
            reasons.append(f"RSI 24 稳在50以上 ({last_rsi24:.1f})，多头趋势未破")
        
        # RSI 6快速下跌至20附近
        if 15 <= last_rsi6 <= 25:
            score += 25
            reasons.append(f"RSI 6 快速下跌至超卖区 ({last_rsi6:.1f})，短期情绪过冷")
        elif 25 < last_rsi6 <= 30:
            score += 15
            reasons.append(f"RSI 6 接近超卖区 ({last_rsi6:.1f})")
        
        # 价格回踩EMA200但未跌破
        if dist_to_ema200_pct <= 3.0:
            score += 10
            reasons.append(f"价格回踩EMA200附近（距离{dist_to_ema200_pct:.2f}%），但未跌破")
        
        # 回踩后止跌
        if len(close) >= 3 and close.iloc[-1] > close.iloc[-2] and close.iloc[-2] <= close.iloc[-3]:
            score += 10
            reasons.append("回踩后出现止跌回升")
        
        score += base_score
        reasons.append("趋势回踩区：上升途中的入场机会")

    # ========== 其他情况 ==========
    else:
        stage = "观望区"
        if price_above_ema200 and last_rsi24 < 50:
            score = 20
            reasons.append("价格在EMA200上方但RSI 24未站稳50，趋势可能转弱")
        elif price_below_ema200 and last_rsi24 > 50:
            score = 15
            reasons.append("价格在EMA200下方但RSI 24在50上方，信号矛盾")
        else:
            score = 10
            reasons.append("当前不符合三阶段买入条件，建议观望")

    # 最终评分限制在0-100
    score = max(0.0, min(100.0, round(score, 1)))
    
    if not reasons:
        reasons.append("暂无明确趋势买点信号")
    
    # 添加阶段标识到原因中
    if stage:
        reasons.insert(0, f"【{stage}】")
    
    return score, reasons


def get_buy_point_label(score: float, reasons: Optional[List[str]] = None) -> str:
    """
    根据评分和阶段信息返回买入标签
    
    标签含义：
    - 强买点：右侧加仓区（共振信号）或趋势回踩区（RSI 6深度超卖）
    - 关注：右侧加仓区（一般信号）或左侧定投区（底背离）
    - 定投：左侧定投区（RSI 24向30靠近）
    - 观望：不符合三阶段条件
    - 不建议：当前不适合买入
    """
    if reasons and len(reasons) > 0:
        first_reason = reasons[0]
        # 根据阶段判断
        if "右侧加仓区" in first_reason:
            if score >= 90:
                return "强买点"
            elif score >= 70:
                return "关注"
            else:
                return "观望"
        elif "趋势回踩区" in first_reason:
            if score >= 80:
                return "强买点"
            elif score >= 60:
                return "关注"
            else:
                return "观望"
        elif "左侧定投区" in first_reason:
            if "底背离" in " ".join(reasons):
                return "关注"
            elif score >= 50:
                return "定投"
            else:
                return "观望"
        elif "观望区" in first_reason:
            return "观望"
    
    # 如果没有阶段信息，按分数判断
    if score >= 80:
        return "强买点"
    if score >= 65:
        return "关注"
    if score >= 45:
        return "观望"
    return "不建议"


def run_simulation_with_paths(paths: np.ndarray, entry: float, size: float, action: str,
                               sl: Optional[float], tp: Optional[float], leverage: int) -> Dict:
    """
    Run trade simulation using pre-generated price paths (from Kronos).
    paths: shape (num_samples, horizon) - price predictions from Kronos
    """
    is_long = action.lower() in ("buy", "long")
    num_samples, horizon = paths.shape

    # Add entry price as first column
    paths_with_entry = np.column_stack([np.full(num_samples, entry), paths])

    hit_tp, hit_sl, expired = 0, 0, 0
    exit_bars, pnls = [], []

    for i in range(num_samples):
        path = paths_with_entry[i]
        exit_bar, exit_price, outcome = horizon, path[-1], "expired"

        for t in range(1, horizon + 1):
            p = path[t]
            if sl and ((is_long and p <= sl) or (not is_long and p >= sl)):
                exit_bar, exit_price, outcome = t, sl, "sl"
                break
            if tp and ((is_long and p >= tp) or (not is_long and p <= tp)):
                exit_bar, exit_price, outcome = t, tp, "tp"
                break

        pnl = ((exit_price - entry) / entry if is_long else (entry - exit_price) / entry) * size * leverage
        pnls.append(pnl)
        exit_bars.append(exit_bar)
        if outcome == "tp": hit_tp += 1
        elif outcome == "sl": hit_sl += 1
        else: expired += 1

    pnls = np.array(pnls)
    drawdowns = [(entry - np.min(paths_with_entry[i])) / entry if is_long else
                 (np.max(paths_with_entry[i]) - entry) / entry for i in range(num_samples)]

    return {
        "win_rate": float(np.mean(pnls > 0) * 100),
        "expected_pnl": float(np.mean(pnls)),
        "expected_pnl_percent": float(np.mean(pnls) / size * 100),
        "max_profit": float(np.max(pnls)),
        "max_loss": float(np.min(pnls)),
        "hit_take_profit_pct": hit_tp / num_samples * 100,
        "hit_stop_loss_pct": hit_sl / num_samples * 100,
        "expired_pct": expired / num_samples * 100,
        "avg_bars_to_exit": float(np.mean(exit_bars)),
        "max_drawdown": float(np.max(drawdowns) * 100),
        "num_simulations": num_samples,
    }


def diagnose_trade(r: Dict, sl: Optional[float], tp: Optional[float]) -> Tuple[str, List[str], List[Dict]]:
    issues, opts = [], []
    if r["expected_pnl"] < 0:
        issues.append("負期望值 - 平均而言這筆交易會虧錢")
    if r["hit_stop_loss_pct"] > 40:
        issues.append(f"止損觸發太頻繁 ({r['hit_stop_loss_pct']:.1f}%)")
        opts.append({"type": "widen_sl", "description": "放寬止損 30%", "expected_improvement": "+15% 勝率"})
    if r["hit_take_profit_pct"] < 30:
        issues.append(f"止盈很少觸發 ({r['hit_take_profit_pct']:.1f}%)")
        opts.append({"type": "lower_tp", "description": "降低止盈目標 20%", "expected_improvement": "+20% 觸發率"})

    if r["expected_pnl"] > 0 and len(issues) == 0:
        rec = "✅ 建議執行 - 正期望值交易"
    elif r["expected_pnl"] > 0:
        rec = "⚠️ 謹慎執行 - 有風險但正期望"
    else:
        rec = "❌ 不建議 - 負期望值交易"
    return rec, issues, opts


def create_forecast_chart(
    symbol: str,
    hist_timestamps: pd.Series,
    hist_close: pd.Series,
    hist_volume: pd.Series,
    pred_timestamps: pd.Series,
    close_samples: np.ndarray,  # shape: (num_samples, horizon)
    volume_samples: np.ndarray,  # shape: (num_samples, horizon)
    horizon: int,
    timeframe: str = "1h"
) -> bytes:
    """
    Generate forecast chart using Matplotlib (same style as Kronos official demo).
    Returns PNG image as bytes.
    """
    # Timeframe labels for chart title
    tf_labels = {"1h": "1H", "4h": "4H", "1d": "1D", "1w": "1W"}
    tf_label = tf_labels.get(timeframe, "1H")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
        facecolor='#0b0e11'
    )

    # Dark theme styling
    for ax in [ax1, ax2]:
        ax.set_facecolor('#141922')
        ax.tick_params(colors='#8A9AAD')
        ax.spines['bottom'].set_color('#2a3441')
        ax.spines['top'].set_color('#2a3441')
        ax.spines['left'].set_color('#2a3441')
        ax.spines['right'].set_color('#2a3441')

    # Price chart (ax1)
    ax1.plot(hist_timestamps, hist_close, color='#4169E1', label='Historical Price', linewidth=1.5)

    mean_preds = close_samples.mean(axis=0)
    # Use p10-p90 (80% confidence) instead of min-max to avoid extreme outliers
    p10_preds = np.percentile(close_samples, 10, axis=0)
    p90_preds = np.percentile(close_samples, 90, axis=0)

    ax1.plot(pred_timestamps, mean_preds, color='#FF8C00', linestyle='-', label='Mean Forecast', linewidth=2)
    ax1.fill_between(pred_timestamps, p10_preds, p90_preds, color='#FF8C00', alpha=0.2, label='Forecast Range (80%)')

    # Set Y-axis to reasonable range based on combined data
    all_prices = np.concatenate([hist_close.values, mean_preds])
    price_min, price_max = all_prices.min(), all_prices.max()
    price_margin = (price_max - price_min) * 0.15
    ax1.set_ylim(price_min - price_margin, price_max + price_margin)

    ax1.set_title(f'{symbol} [{tf_label}] Probabilistic Forecast (Next {horizon} bars)', fontsize=14, weight='bold', color='white')
    ax1.set_ylabel('Price', color='#8A9AAD')
    ax1.legend(loc='upper left', facecolor='#1e2530', edgecolor='#2a3441', labelcolor='#8A9AAD')
    ax1.grid(True, linestyle='--', linewidth=0.3, color='#2a3441')

    # Volume chart (ax2)
    bar_width = 0.025
    ax2.bar(hist_timestamps, hist_volume, color='#4169E1', alpha=0.7, label='Historical Volume', width=bar_width)

    if volume_samples is not None and len(volume_samples) > 0:
        mean_vol = volume_samples.mean(axis=0)
        ax2.bar(pred_timestamps, mean_vol, color='#FF8C00', alpha=0.7, label='Forecast Volume', width=bar_width)

    ax2.set_ylabel('Volume', color='#8A9AAD')
    ax2.set_xlabel('Time (UTC)', color='#8A9AAD')
    ax2.legend(loc='upper left', facecolor='#1e2530', edgecolor='#2a3441', labelcolor='#8A9AAD')
    ax2.grid(True, linestyle='--', linewidth=0.3, color='#2a3441')

    # Separator line (red dashed)
    separator_time = hist_timestamps.iloc[-1] + timedelta(minutes=30)
    for ax in [ax1, ax2]:
        ax.axvline(x=separator_time, color='#FF4560', linestyle='--', linewidth=1.5)
        ax.tick_params(axis='x', rotation=30, colors='#8A9AAD')

    fig.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor='#0b0e11', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def get_price_scale_factor(price: float) -> float:
    """
    Calculate scale factor for very small price coins (like PEPE).
    Scales prices to ~100 range for better model performance.
    """
    if price <= 0:
        return 1.0
    if price >= 1.0:
        return 1.0  # No scaling needed for normal prices >= $1
    # Scale to bring price to ~100
    import math
    target = 100.0
    scale = target / price
    # Round to power of 10 for clean scaling
    log_scale = math.log10(scale)
    return 10 ** round(log_scale)


def scale_ohlcv_df(df: pd.DataFrame, scale_factor: float) -> pd.DataFrame:
    """Scale OHLCV dataframe prices by scale factor (volume unchanged)."""
    if scale_factor == 1.0:
        return df
    scaled = df.copy()
    for col in ['open', 'high', 'low', 'close']:
        if col in scaled.columns:
            scaled[col] = scaled[col] * scale_factor
    return scaled


def unscale_prediction(result: Dict, scale_factor: float) -> Dict:
    """Unscale prediction results back to original price range."""
    if scale_factor == 1.0:
        return result
    unscaled = result.copy()
    for key in ['mean', 'min', 'max', 'p10', 'p25', 'p75', 'p90']:
        if key in unscaled:
            unscaled[key] = unscaled[key] / scale_factor
    if 'all_samples' in unscaled:
        samples = unscaled['all_samples'].copy()
        # Unscale OHLC columns (0-3), keep volume (4) unchanged
        samples[:, :, 0:4] = samples[:, :, 0:4] / scale_factor
        unscaled['all_samples'] = samples
    return unscaled


def calculate_price_ranges(samples: np.ndarray, current_price: float, num_ranges: int = 6) -> List[Dict]:
    all_prices = samples.flatten()
    min_price, max_price = float(np.min(all_prices)), float(np.max(all_prices))
    range_size = (max_price - min_price) / num_ranges
    ranges = []

    for i in range(num_ranges):
        low = min_price + i * range_size
        high = min_price + (i + 1) * range_size
        count = int(np.sum((all_prices >= low) & (all_prices < high)))
        prob = count / len(all_prices) * 100
        ranges.append({
            "low": smart_round(float(low)),
            "high": smart_round(float(high)),
            "probability": round(float(prob), 1),
            "is_current": bool(low <= current_price <= high)
        })

    return sorted(ranges, key=lambda x: x["probability"], reverse=True)


def find_key_levels(samples: np.ndarray, current_price: float) -> Tuple[List[PriceLevel], List[PriceLevel]]:
    supports, resistances = [], []
    for pct in [10, 25]:
        price = float(np.percentile(samples[:, -1], pct))
        if price < current_price:
            prob = float(np.mean(samples[:, -1] > price) * 100)
            supports.append(PriceLevel(price=smart_round(price), probability=round(prob, 1), type="support"))
    for pct in [75, 90]:
        price = float(np.percentile(samples[:, -1], pct))
        if price > current_price:
            prob = float(np.mean(samples[:, -1] < price) * 100)
            resistances.append(PriceLevel(price=smart_round(price), probability=round(prob, 1), type="resistance"))
    return supports, resistances


def calculate_trend_buy_point_score(price_df: pd.DataFrame) -> Tuple[float, List[str]]:
    """
    基于三阶段策略的趋势买点评分函数（0-100）
    
    第一阶段：左侧定投区 - K线在EMA200下，RSI 24 < 50，向30靠近
    第二阶段：右侧加仓区 - K线穿过EMA200，RSI三线回归50，金叉
    第三阶段：趋势回踩区 - K线在EMA200上，RSI 24 > 50，RSI 6回踩到20
    
    Returns:
        (score, reasons)
    """
    reasons: List[str] = []
    data_len = len(price_df)
    
    # 必须至少200根K线来计算EMA200
    if data_len < 200:
        return 0.0, [f"历史数据不足（当前{data_len}根，需要至少200根K线计算EMA200）"]

    close = price_df["close"].astype(float)
    high = price_df["high"].astype(float)
    low = price_df["low"].astype(float)
    volume = price_df["volume"].astype(float)

    # 计算EMA200（周线级别长期均线）
    ema200 = close.ewm(span=200, adjust=False).mean()
    ema_name = "EMA200"
    
    # 计算RSI 6、RSI 12、RSI 24
    def calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    rsi6 = calculate_rsi(close, 6)
    rsi12 = calculate_rsi(close, 12)
    rsi24 = calculate_rsi(close, 24)

    # 获取最新值
    last_close = float(close.iloc[-1])
    last_ema200 = float(ema200.iloc[-1]) if not pd.isna(ema200.iloc[-1]) else last_close
    
    last_rsi6 = float(rsi6.iloc[-1]) if not pd.isna(rsi6.iloc[-1]) else 50.0
    last_rsi12 = float(rsi12.iloc[-1]) if not pd.isna(rsi12.iloc[-1]) else 50.0
    last_rsi24 = float(rsi24.iloc[-1]) if not pd.isna(rsi24.iloc[-1]) else 50.0
    
    # 获取前几期的值用于判断趋势
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
    prev_ema200 = float(ema200.iloc[-2]) if len(ema200) >= 2 and not pd.isna(ema200.iloc[-2]) else last_ema200
    
    prev_rsi6 = float(rsi6.iloc[-2]) if len(rsi6) >= 2 and not pd.isna(rsi6.iloc[-2]) else last_rsi6
    prev_rsi12 = float(rsi12.iloc[-2]) if len(rsi12) >= 2 and not pd.isna(rsi12.iloc[-2]) else last_rsi12
    prev_rsi24 = float(rsi24.iloc[-2]) if len(rsi24) >= 2 and not pd.isna(rsi24.iloc[-2]) else last_rsi24

    score = 0.0
    stage = None

    # 判断价格与EMA200的关系
    price_above_ema200 = last_close > last_ema200
    price_below_ema200 = last_close < last_ema200
    price_crossing_ema200 = (prev_close <= prev_ema200 and last_close > last_ema200) or \
                           (prev_close >= prev_ema200 and last_close < last_ema200)
    
    # 判断是否在EMA200附近纠缠（距离在2%以内）
    dist_to_ema200_pct = abs(last_close - last_ema200) / (last_ema200 + 1e-8) * 100
    entangled_with_ema200 = dist_to_ema200_pct <= 2.0

    # 检测底背离：价格创新低，但RSI不创新低
    def detect_bottom_divergence(prices: pd.Series, rsi: pd.Series, lookback: int = 20) -> bool:
        if len(prices) < lookback * 2:
            return False
        
        # 获取最近两段数据
        recent_prices = prices.iloc[-lookback:]
        recent_rsi = rsi.iloc[-lookback:]
        prev_prices = prices.iloc[-lookback*2:-lookback] if len(prices) >= lookback * 2 else pd.Series(dtype=float)
        
        if len(prev_prices) == 0:
            return False
        
        # 找到最近的最低点
        price_low = recent_prices.min()
        price_low_pos = recent_prices.values.argmin()
        rsi_at_low = recent_rsi.iloc[price_low_pos]
        
        # 检查之前是否有更低的价格
        prev_low = prev_prices.min()
        
        # 底背离：价格创新低，但RSI不创新低
        if price_low < prev_low:
            # 找到之前最低点对应的RSI
            prev_low_pos = prev_prices.values.argmin()
            prev_rsi = rsi.iloc[len(prices) - lookback*2 + prev_low_pos] if len(prices) >= lookback*2 else 50.0
            
            if rsi_at_low > prev_rsi:
                return True
        return False

    # 检测RSI金叉
    rsi6_cross_rsi24_up = prev_rsi6 <= prev_rsi24 and last_rsi6 > last_rsi24
    rsi12_cross_rsi24_up = prev_rsi12 <= prev_rsi24 and last_rsi12 > last_rsi24
    rsi_golden_cross = rsi6_cross_rsi24_up or rsi12_cross_rsi24_up

    # 判断RSI三线是否都在50附近（45-55区间）
    rsi_all_near_50 = (45 <= last_rsi6 <= 55) and (45 <= last_rsi12 <= 55) and (45 <= last_rsi24 <= 55)
    
    # 判断RSI三线是否都在50上方且发散向上
    rsi_all_above_50 = (last_rsi6 > 50) and (last_rsi12 > 50) and (last_rsi24 > 50)
    rsi_expanding = (last_rsi6 > last_rsi12 > last_rsi24) and (last_rsi6 > prev_rsi6)

    # ========== 第一阶段：左侧定投区 ==========
    if price_below_ema200 and last_rsi24 < 50:
        stage = "左侧定投区"
        base_score = 30  # 基础分
        
        # RSI 24向30靠近（超卖区）
        if 30 <= last_rsi24 < 40:
            score += 25
            reasons.append(f"RSI 24 处于超卖区 ({last_rsi24:.1f})，接近30支撑")
        elif 25 <= last_rsi24 < 30:
            score += 30
            reasons.append(f"RSI 24 深度超卖 ({last_rsi24:.1f})，接近底部")
        elif last_rsi24 < 25:
            score += 20
            reasons.append(f"RSI 24 极度超卖 ({last_rsi24:.1f})，可能超跌")
        else:
            score += 15
            reasons.append(f"RSI 24 在50下方 ({last_rsi24:.1f})，等待向30靠近")
        
        # 底背离检测（使用RSI 9/12，这里用RSI 12代替）
        if detect_bottom_divergence(close, rsi12, lookback=30):
            score += 20
            reasons.append("检测到底背离信号（价格创新低，RSI不创新低）")
        
        # RSI 6/12超卖
        if last_rsi6 < 30 or last_rsi12 < 30:
            score += 10
            reasons.append(f"RSI 6/12 超卖 (RSI6={last_rsi6:.1f}, RSI12={last_rsi12:.1f})")
        
        score += base_score
        reasons.append("价格在EMA200下方，适合定投布局")

    # ========== 第二阶段：右侧加仓区 ==========
    elif (price_crossing_ema200 or entangled_with_ema200) and last_rsi24 >= 45:
        stage = "右侧加仓区"
        base_score = 50  # 基础分
        
        # 价格突破EMA200
        if price_crossing_ema200 and last_close > last_ema200:
            score += 20
            reasons.append("价格由下往上突破EMA200")
        elif entangled_with_ema200:
            score += 15
            reasons.append("价格与EMA200反复纠缠")
        
        # RSI三线回归50
        if rsi_all_near_50:
            score += 15
            reasons.append("RSI 6/12/24 三线回归50分界线附近")
        
        # RSI金叉
        if rsi_golden_cross:
            score += 15
            if rsi6_cross_rsi24_up:
                reasons.append("RSI 6 向上金叉 RSI 24")
            if rsi12_cross_rsi24_up:
                reasons.append("RSI 12 向上金叉 RSI 24")
        
        # RSI三线在50上方发散
        if rsi_all_above_50 and rsi_expanding:
            score += 10
            reasons.append("RSI 三线在50上方呈发散状向上（多头扩张）")
        
        # 共振信号（最强买入信号）
        if price_crossing_ema200 and last_close > last_ema200 and \
           rsi_all_above_50 and rsi_golden_cross:
            score += 20
            reasons.append("【共振信号】价格突破EMA200 + RSI三线站上50 + RSI金叉")
        
        score += base_score
        reasons.append("右侧加仓区：趋势反转窗口，适合重仓布局")

    # ========== 第三阶段：趋势回踩区 ==========
    elif price_above_ema200 and last_rsi24 > 50:
        stage = "趋势回踩区"
        base_score = 40  # 基础分
        
        # RSI 24稳在50以上
        if last_rsi24 > 50:
            score += 15
            reasons.append(f"RSI 24 稳在50以上 ({last_rsi24:.1f})，多头趋势未破")
        
        # RSI 6快速下跌至20附近
        if 15 <= last_rsi6 <= 25:
            score += 25
            reasons.append(f"RSI 6 快速下跌至超卖区 ({last_rsi6:.1f})，短期情绪过冷")
        elif 25 < last_rsi6 <= 30:
            score += 15
            reasons.append(f"RSI 6 接近超卖区 ({last_rsi6:.1f})")
        
        # 价格回踩EMA200但未跌破
        if dist_to_ema200_pct <= 3.0:
            score += 10
            reasons.append(f"价格回踩EMA200附近（距离{dist_to_ema200_pct:.2f}%），但未跌破")
        
        # 回踩后止跌
        if len(close) >= 3 and close.iloc[-1] > close.iloc[-2] and close.iloc[-2] <= close.iloc[-3]:
            score += 10
            reasons.append("回踩后出现止跌回升")
        
        score += base_score
        reasons.append("趋势回踩区：上升途中的入场机会")

    # ========== 其他情况 ==========
    else:
        stage = "观望区"
        if price_above_ema200 and last_rsi24 < 50:
            score = 20
            reasons.append("价格在EMA200上方但RSI 24未站稳50，趋势可能转弱")
        elif price_below_ema200 and last_rsi24 > 50:
            score = 15
            reasons.append("价格在EMA200下方但RSI 24在50上方，信号矛盾")
        else:
            score = 10
            reasons.append("当前不符合三阶段买入条件，建议观望")

    # 最终评分限制在0-100
    score = max(0.0, min(100.0, round(score, 1)))
    
    if not reasons:
        reasons.append("暂无明确趋势买点信号")
    
    # 添加阶段标识到原因中
    if stage:
        reasons.insert(0, f"【{stage}】")
    
    return score, reasons


def get_buy_point_label(score: float, reasons: Optional[List[str]] = None) -> str:
    """
    根据评分和阶段信息返回买入标签
    
    标签含义：
    - 强买点：右侧加仓区（共振信号）或趋势回踩区（RSI 6深度超卖）
    - 关注：右侧加仓区（一般信号）或左侧定投区（底背离）
    - 定投：左侧定投区（RSI 24向30靠近）
    - 观望：不符合三阶段条件
    - 不建议：当前不适合买入
    """
    if reasons and len(reasons) > 0:
        first_reason = reasons[0]
        # 根据阶段判断
        if "右侧加仓区" in first_reason:
            if score >= 90:
                return "强买点"
            elif score >= 70:
                return "关注"
            else:
                return "观望"
        elif "趋势回踩区" in first_reason:
            if score >= 80:
                return "强买点"
            elif score >= 60:
                return "关注"
            else:
                return "观望"
        elif "左侧定投区" in first_reason:
            if "底背离" in " ".join(reasons):
                return "关注"
            elif score >= 50:
                return "定投"
            else:
                return "观望"
        elif "观望区" in first_reason:
            return "观望"
    
    # 如果没有阶段信息，按分数判断
    if score >= 80:
        return "强买点"
    if score >= 65:
        return "关注"
    if score >= 45:
        return "观望"
    return "不建议"


def run_simulation_with_paths(paths: np.ndarray, entry: float, size: float, action: str,
                               sl: Optional[float], tp: Optional[float], leverage: int) -> Dict:
    """
    Run trade simulation using pre-generated price paths (from Kronos).
    paths: shape (num_samples, horizon) - price predictions from Kronos
    """
    is_long = action.lower() in ("buy", "long")
    num_samples, horizon = paths.shape

    # Add entry price as first column
    paths_with_entry = np.column_stack([np.full(num_samples, entry), paths])

    hit_tp, hit_sl, expired = 0, 0, 0
    exit_bars, pnls = [], []

    for i in range(num_samples):
        path = paths_with_entry[i]
        exit_bar, exit_price, outcome = horizon, path[-1], "expired"

        for t in range(1, horizon + 1):
            p = path[t]
            if sl and ((is_long and p <= sl) or (not is_long and p >= sl)):
                exit_bar, exit_price, outcome = t, sl, "sl"
                break
            if tp and ((is_long and p >= tp) or (not is_long and p <= tp)):
                exit_bar, exit_price, outcome = t, tp, "tp"
                break

        pnl = ((exit_price - entry) / entry if is_long else (entry - exit_price) / entry) * size * leverage
        pnls.append(pnl)
        exit_bars.append(exit_bar)
        if outcome == "tp": hit_tp += 1
        elif outcome == "sl": hit_sl += 1
        else: expired += 1

    pnls = np.array(pnls)
    drawdowns = [(entry - np.min(paths_with_entry[i])) / entry if is_long else
                 (np.max(paths_with_entry[i]) - entry) / entry for i in range(num_samples)]

    return {
        "win_rate": float(np.mean(pnls > 0) * 100),
        "expected_pnl": float(np.mean(pnls)),
        "expected_pnl_percent": float(np.mean(pnls) / size * 100),
        "max_profit": float(np.max(pnls)),
        "max_loss": float(np.min(pnls)),
        "hit_take_profit_pct": hit_tp / num_samples * 100,
        "hit_stop_loss_pct": hit_sl / num_samples * 100,
        "expired_pct": expired / num_samples * 100,
        "avg_bars_to_exit": float(np.mean(exit_bars)),
        "max_drawdown": float(np.max(drawdowns) * 100),
        "num_simulations": num_samples,
    }


def diagnose_trade(r: Dict, sl: Optional[float], tp: Optional[float]) -> Tuple[str, List[str], List[Dict]]:
    issues, opts = [], []
    if r["expected_pnl"] < 0:
        issues.append("負期望值 - 平均而言這筆交易會虧錢")
    if r["hit_stop_loss_pct"] > 40:
        issues.append(f"止損觸發太頻繁 ({r['hit_stop_loss_pct']:.1f}%)")
        opts.append({"type": "widen_sl", "description": "放寬止損 30%", "expected_improvement": "+15% 勝率"})
    if r["hit_take_profit_pct"] < 30:
        issues.append(f"止盈很少觸發 ({r['hit_take_profit_pct']:.1f}%)")
        opts.append({"type": "lower_tp", "description": "降低止盈目標 20%", "expected_improvement": "+20% 觸發率"})

    if r["expected_pnl"] > 0 and len(issues) == 0:
        rec = "✅ 建議執行 - 正期望值交易"
    elif r["expected_pnl"] > 0:
        rec = "⚠️ 謹慎執行 - 有風險但正期望"
    else:
        rec = "❌ 不建議 - 負期望值交易"
    return rec, issues, opts


def create_forecast_chart(
    symbol: str,
    hist_timestamps: pd.Series,
    hist_close: pd.Series,
    hist_volume: pd.Series,
    pred_timestamps: pd.Series,
    close_samples: np.ndarray,  # shape: (num_samples, horizon)
    volume_samples: np.ndarray,  # shape: (num_samples, horizon)
    horizon: int,
    timeframe: str = "1h"
) -> bytes:
    """
    Generate forecast chart using Matplotlib (same style as Kronos official demo).
    Returns PNG image as bytes.
    """
    # Timeframe labels for chart title
    tf_labels = {"1h": "1H", "4h": "4H", "1d": "1D", "1w": "1W"}
    tf_label = tf_labels.get(timeframe, "1H")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
        facecolor='#0b0e11'
    )

    # Dark theme styling
    for ax in [ax1, ax2]:
        ax.set_facecolor('#141922')
        ax.tick_params(colors='#8A9AAD')
        ax.spines['bottom'].set_color('#2a3441')
        ax.spines['top'].set_color('#2a3441')
        ax.spines['left'].set_color('#2a3441')
        ax.spines['right'].set_color('#2a3441')

    # Price chart (ax1)
    ax1.plot(hist_timestamps, hist_close, color='#4169E1', label='Historical Price', linewidth=1.5)

    mean_preds = close_samples.mean(axis=0)
    # Use p10-p90 (80% confidence) instead of min-max to avoid extreme outliers
    p10_preds = np.percentile(close_samples, 10, axis=0)
    p90_preds = np.percentile(close_samples, 90, axis=0)

    ax1.plot(pred_timestamps, mean_preds, color='#FF8C00', linestyle='-', label='Mean Forecast', linewidth=2)
    ax1.fill_between(pred_timestamps, p10_preds, p90_preds, color='#FF8C00', alpha=0.2, label='Forecast Range (80%)')

    # Set Y-axis to reasonable range based on combined data
    all_prices = np.concatenate([hist_close.values, mean_preds])
    price_min, price_max = all_prices.min(), all_prices.max()
    price_margin = (price_max - price_min) * 0.15
    ax1.set_ylim(price_min - price_margin, price_max + price_margin)

    ax1.set_title(f'{symbol} [{tf_label}] Probabilistic Forecast (Next {horizon} bars)', fontsize=14, weight='bold', color='white')
    ax1.set_ylabel('Price', color='#8A9AAD')
    ax1.legend(loc='upper left', facecolor='#1e2530', edgecolor='#2a3441', labelcolor='#8A9AAD')
    ax1.grid(True, linestyle='--', linewidth=0.3, color='#2a3441')

    # Volume chart (ax2)
    bar_width = 0.025
    ax2.bar(hist_timestamps, hist_volume, color='#4169E1', alpha=0.7, label='Historical Volume', width=bar_width)

    if volume_samples is not None and len(volume_samples) > 0:
        mean_vol = volume_samples.mean(axis=0)
        ax2.bar(pred_timestamps, mean_vol, color='#FF8C00', alpha=0.7, label='Forecast Volume', width=bar_width)

    ax2.set_ylabel('Volume', color='#8A9AAD')
    ax2.set_xlabel('Time (UTC)', color='#8A9AAD')
    ax2.legend(loc='upper left', facecolor='#1e2530', edgecolor='#2a3441', labelcolor='#8A9AAD')
    ax2.grid(True, linestyle='--', linewidth=0.3, color='#2a3441')

    # Separator line (red dashed)
    separator_time = hist_timestamps.iloc[-1] + timedelta(minutes=30)
    for ax in [ax1, ax2]:
        ax.axvline(x=separator_time, color='#FF4560', linestyle='--', linewidth=1.5)
        ax.tick_params(axis='x', rotation=30, colors='#8A9AAD')

    fig.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor='#0b0e11', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "name": "Next K API",
        "version": "2.0.0",
        "description": "AI-Powered K-Line Weather Forecast API",
        "docs": "/docs",
        "health": "/api/health",
        "zct_vwap_dashboard": "/dashboard/zct-vwap",
    }


@app.get("/api/health", response_model=HealthResponse)
async def health():
    uptime = (datetime.now(timezone.utc) - state.startup_time).total_seconds() if state.startup_time else 0
    up = state.is_ready or state.kronos_skipped
    return HealthResponse(
        status="healthy" if up else "initializing",
        model_ready=state.is_ready,
        kronos_skipped=state.kronos_skipped,
        crypto_connected=state.ccxt_exchange is not None,
        stocks_available=state.yfinance_available,
        forex_available=state.yfinance_available,
        version="2.0.0",
        uptime=uptime
    )


@app.get("/api/symbols")
async def get_symbols(asset_type: Optional[str] = None):
    """Get supported symbols, optionally filtered by asset type."""
    if asset_type:
        try:
            at = AssetType(asset_type)
            return {"asset_type": at.value, "symbols": SYMBOLS[at]}
        except ValueError:
            raise HTTPException(400, f"Invalid asset type: {asset_type}")
    return {
        "crypto": SYMBOLS[AssetType.CRYPTO],
        "stock": SYMBOLS[AssetType.STOCK],
        "forex": SYMBOLS[AssetType.FOREX],
    }


@app.get("/api/weather/{symbol:path}", response_model=WeatherForecast)
async def get_weather_forecast(
    symbol: str,
    asset_type: Optional[str] = None,
    timeframe: str = Query(default="1h", regex="^(1h|4h|1d|1w)$"),
    horizon: Optional[int] = None,
    sample_count: int = Query(default=5, ge=3, le=50)
):
    """K-Line Weather Forecast - Probabilistic price prediction with multiple timeframes."""
    symbol = symbol.upper().replace("-", "/")

    # Detect or use provided asset type
    at = AssetType(asset_type) if asset_type else detect_asset_type(symbol)

    # Get timeframe config
    tf = TimeFrame(timeframe)
    tf_config = TIMEFRAME_CONFIG[tf]
    hours_per_bar = tf_config["hours"]

    # Use default horizon for timeframe if not specified
    if horizon is None:
        horizon = tf_config["horizon_default"]
    horizon = max(1, min(horizon, 168))

    if not state.is_ready:
        raise HTTPException(503, "Model not ready. Please wait.")

    # Fetch data with appropriate timeframe
    # 为了计算EMA200，需要至少200根K线
    min_limit = max(tf_config["limit"], 250)  # 获取250根以确保有足够数据计算EMA200
    df = await fetch_ohlcv(symbol, at, timeframe, min_limit)
    if df is None or len(df) < 30:
        raise HTTPException(400, f"Insufficient data for {symbol} ({at.value})")
    
    # 记录实际获取的数据量
    actual_data_len = len(df)
    logger.info(f"Fetched {actual_data_len} bars for {symbol} ({at.value}) [{timeframe}], requested {min_limit}")
    
    # 如果数据不足200根，记录警告（但继续处理，让评分函数返回错误）
    if actual_data_len < 200:
        logger.warning(f"Warning: Only {actual_data_len} bars available for {symbol}, EMA200 calculation may fail")

    timestamps = pd.to_datetime(df['timestamp'])
    price_df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    current_price = float(price_df['close'].iloc[-1])

    # Scale prices for small-price coins (like PEPE)
    scale_factor = get_price_scale_factor(current_price)
    scaled_price_df = scale_ohlcv_df(price_df, scale_factor)
    if scale_factor != 1.0:
        logger.info(f"Scaling {symbol} prices by {scale_factor}x for prediction")

    # Generate future timestamps based on timeframe
    y_timestamps = pd.Series([timestamps.iloc[-1] + timedelta(hours=hours_per_bar * i) for i in range(1, horizon + 1)])

    # Run Kronos prediction with scaled prices
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: state.predictor.predict_multi_sample(
            scaled_price_df, timestamps, y_timestamps, horizon,
            T=1.0, top_p=0.9, sample_count=sample_count
        )
    )

    # Unscale prediction results back to original price range
    result = unscale_prediction(result, scale_factor)

    # Calculate metrics
    close_samples = result['all_samples'][:, :, 3]
    upside_prob = float(np.mean(close_samples[:, -1] > current_price) * 100)

    returns = np.log(price_df['close'].values[1:] / price_df['close'].values[:-1])
    # Annualize volatility based on timeframe
    bars_per_day = 24 // hours_per_bar
    volatility = float(np.std(returns[-bars_per_day:]) * np.sqrt(bars_per_day) * 100) if len(returns) >= bars_per_day else 0

    price_ranges = calculate_price_ranges(close_samples, current_price)
    supports, resistances = find_key_levels(close_samples, current_price)

    spread = (result['p90'] - result['p10']) / result['mean']
    confidence = float(max(0.3, min(0.95, 1.0 - float(np.mean(spread)) * 2)))
    buy_point_score, buy_point_reasons = calculate_trend_buy_point_score(price_df)
    buy_point_label = get_buy_point_label(buy_point_score, buy_point_reasons)

    # Build history data (last 72 bars = 3 days)
    history_len = min(72, len(df))
    history = []
    for i in range(len(df) - history_len, len(df)):
        ts = int(timestamps.iloc[i].timestamp())
        history.append(HistoryBar(
            time=ts,
            open=smart_round(float(price_df['open'].iloc[i])),
            high=smart_round(float(price_df['high'].iloc[i])),
            low=smart_round(float(price_df['low'].iloc[i])),
            close=smart_round(float(price_df['close'].iloc[i])),
            volume=round(float(price_df['volume'].iloc[i]), 2),
        ))

    # Build forecast data
    forecast = []
    last_ts = int(timestamps.iloc[-1].timestamp())
    seconds_per_bar = hours_per_bar * 3600
    for i in range(horizon):
        forecast.append(ForecastBar(
            time=last_ts + (i + 1) * seconds_per_bar,
            mean=smart_round(float(result['mean'][i])),
            min=smart_round(float(result['min'][i])),
            max=smart_round(float(result['max'][i])),
            p10=smart_round(float(result['p10'][i])),
            p25=smart_round(float(result['p25'][i])),
            p75=smart_round(float(result['p75'][i])),
            p90=smart_round(float(result['p90'][i])),
        ))

    # Trading suggestions
    entry = smart_round(float(result['p25'][0]))
    sl = smart_round(float(result['p10'][horizon // 2]))
    tp = smart_round(float(result['p75'][horizon - 1]))
    risk = entry - sl
    rr = round((tp - entry) / risk, 2) if risk > 0 else None

    return WeatherForecast(
        symbol=symbol,
        asset_type=at.value,
        timeframe=timeframe,
        current_price=smart_round(current_price),
        generated_at=datetime.now(timezone.utc).isoformat(),
        horizon=horizon,
        price_ranges=price_ranges,
        support_levels=supports,
        resistance_levels=resistances,
        history=history,
        forecast=forecast,
        upside_prob=round(upside_prob, 1),
        volatility=round(volatility, 2),
        confidence=round(confidence, 2),
        suggested_entry=entry,
        suggested_stop_loss=sl,
        suggested_take_profit=tp,
        risk_reward_ratio=rr,
        buy_point_score=buy_point_score,
        buy_point_label=buy_point_label,
        buy_point_reasons=buy_point_reasons[:5],
    )


@app.get("/api/chart/{symbol:path}")
async def get_chart_image(
    symbol: str,
    asset_type: Optional[str] = None,
    timeframe: str = Query(default="1h", regex="^(1h|4h|1d|1w)$"),
    horizon: Optional[int] = None,
    sample_count: int = Query(default=5, ge=3, le=50)
):
    """Generate forecast chart image (PNG) - same style as Kronos official demo."""
    symbol = symbol.upper().replace("-", "/")
    at = AssetType(asset_type) if asset_type else detect_asset_type(symbol)

    # Get timeframe config
    tf = TimeFrame(timeframe)
    tf_config = TIMEFRAME_CONFIG[tf]
    hours_per_bar = tf_config["hours"]

    # Use default horizon for timeframe if not specified
    if horizon is None:
        horizon = tf_config["horizon_default"]
    horizon = max(1, min(horizon, 168))

    if not state.is_ready:
        raise HTTPException(503, "Model not ready. Please wait.")

    # Fetch data with appropriate timeframe
    df = await fetch_ohlcv(symbol, at, timeframe, tf_config["limit"])
    if df is None or len(df) < 30:
        raise HTTPException(400, f"Insufficient data for {symbol}")

    timestamps = pd.to_datetime(df['timestamp'])
    price_df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    current_price = float(price_df['close'].iloc[-1])

    # Scale prices for small-price coins (like PEPE)
    scale_factor = get_price_scale_factor(current_price)
    scaled_price_df = scale_ohlcv_df(price_df, scale_factor)

    y_timestamps = pd.Series([timestamps.iloc[-1] + timedelta(hours=hours_per_bar * i) for i in range(1, horizon + 1)])

    # Run Kronos prediction with scaled prices
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: state.predictor.predict_multi_sample(
            scaled_price_df, timestamps, y_timestamps, horizon,
            T=1.0, top_p=0.9, sample_count=sample_count
        )
    )

    # Unscale prediction results
    result = unscale_prediction(result, scale_factor)

    # Prepare data for chart (use original unscaled prices for history)
    hist_len = min(72, len(df))
    hist_timestamps = timestamps.iloc[-hist_len:]
    hist_close = price_df['close'].iloc[-hist_len:]
    hist_volume = price_df['volume'].iloc[-hist_len:]

    # Close samples: shape (sample_count, horizon)
    close_samples = result['all_samples'][:, :, 3]
    # Volume samples (if available)
    volume_samples = result['all_samples'][:, :, 4] if result['all_samples'].shape[2] > 4 else None

    # Generate chart
    chart_bytes = create_forecast_chart(
        symbol=symbol,
        hist_timestamps=hist_timestamps,
        hist_close=hist_close,
        hist_volume=hist_volume,
        pred_timestamps=y_timestamps,
        close_samples=close_samples,
        volume_samples=volume_samples,
        horizon=horizon,
        timeframe=timeframe
    )

    return Response(content=chart_bytes, media_type="image/png")


@app.post("/api/simulate", response_model=SimulationResult)
async def simulate_trade(request: SimulationRequest):
    """Trade Sandbox - Simulate trade using Kronos AI predictions."""
    symbol = request.symbol.upper().replace("-", "/")
    at = AssetType(request.asset_type) if request.asset_type else detect_asset_type(symbol)

    if not state.is_ready:
        raise HTTPException(503, "Model not ready. Please wait.")

    # Fetch historical data
    df = await fetch_ohlcv(symbol, at, "1h", 100)
    if df is None or len(df) < 50:
        raise HTTPException(400, f"Cannot fetch data for {symbol}")

    timestamps = pd.to_datetime(df['timestamp'])
    price_df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    current = float(price_df['close'].iloc[-1])
    entry = request.entry_price or current

    # Scale prices for small-price coins
    scale_factor = get_price_scale_factor(current)
    scaled_price_df = scale_ohlcv_df(price_df, scale_factor)

    # Generate future timestamps
    y_timestamps = pd.Series([timestamps.iloc[-1] + timedelta(hours=i) for i in range(1, request.horizon + 1)])

    # Run Kronos prediction with more samples for better simulation
    sample_count = 30  # More samples for reliable simulation
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: state.predictor.predict_multi_sample(
            scaled_price_df, timestamps, y_timestamps, request.horizon,
            T=1.0, top_p=0.9, sample_count=sample_count
        )
    )

    # Unscale prediction results
    result = unscale_prediction(result, scale_factor)

    # Extract close price paths from Kronos predictions
    # all_samples shape: (sample_count, horizon, 5) where 5 = [open, high, low, close, volume]
    close_paths = result['all_samples'][:, :, 3]  # shape: (sample_count, horizon)

    # Run simulation with Kronos-predicted paths
    sim = run_simulation_with_paths(
        paths=close_paths,
        entry=entry,
        size=request.position_size,
        action=request.action,
        sl=request.stop_loss,
        tp=request.take_profit,
        leverage=request.leverage
    )

    rec, issues, opts = diagnose_trade(sim, request.stop_loss, request.take_profit)

    return SimulationResult(
        symbol=symbol, asset_type=at.value, action=request.action,
        entry_price=smart_round(entry), position_size=request.position_size, leverage=request.leverage,
        win_rate=round(sim["win_rate"], 1),
        expected_pnl=round(sim["expected_pnl"], 2),
        expected_pnl_percent=round(sim["expected_pnl_percent"], 2),
        max_profit=round(sim["max_profit"], 2),
        max_loss=round(sim["max_loss"], 2),
        hit_take_profit_pct=round(sim["hit_take_profit_pct"], 1),
        hit_stop_loss_pct=round(sim["hit_stop_loss_pct"], 1),
        expired_pct=round(sim["expired_pct"], 1),
        avg_bars_to_exit=round(sim["avg_bars_to_exit"], 1),
        max_drawdown=round(sim["max_drawdown"], 1),
        recommendation=rec, issues=issues, optimizations=opts,
    )


@app.get("/api/symbols")
async def get_symbols(asset_type: Optional[str] = None):
    """Get supported symbols, optionally filtered by asset type."""
    if asset_type:
        try:
            at = AssetType(asset_type)
            return {"asset_type": at.value, "symbols": SYMBOLS[at]}
        except ValueError:
            raise HTTPException(400, f"Invalid asset type: {asset_type}")
    return {
        "crypto": SYMBOLS[AssetType.CRYPTO],
        "stock": SYMBOLS[AssetType.STOCK],
        "forex": SYMBOLS[AssetType.FOREX],
    }


@app.get("/api/weather/{symbol:path}", response_model=WeatherForecast)
async def get_weather_forecast(
    symbol: str,
    asset_type: Optional[str] = None,
    timeframe: str = Query(default="1h", regex="^(1h|4h|1d|1w)$"),
    horizon: Optional[int] = None,
    sample_count: int = Query(default=5, ge=3, le=50)
):
    """K-Line Weather Forecast - Probabilistic price prediction with multiple timeframes."""
    symbol = symbol.upper().replace("-", "/")

    # Detect or use provided asset type
    at = AssetType(asset_type) if asset_type else detect_asset_type(symbol)

    # Get timeframe config
    tf = TimeFrame(timeframe)
    tf_config = TIMEFRAME_CONFIG[tf]
    hours_per_bar = tf_config["hours"]

    # Use default horizon for timeframe if not specified
    if horizon is None:
        horizon = tf_config["horizon_default"]
    horizon = max(1, min(horizon, 168))

    if not state.is_ready:
        raise HTTPException(503, "Model not ready. Please wait.")

    # Fetch data with appropriate timeframe
    # 为了计算EMA200，需要至少200根K线
    min_limit = max(tf_config["limit"], 250)  # 获取250根以确保有足够数据计算EMA200
    df = await fetch_ohlcv(symbol, at, timeframe, min_limit)
    if df is None or len(df) < 30:
        raise HTTPException(400, f"Insufficient data for {symbol} ({at.value})")
    
    # 记录实际获取的数据量
    actual_data_len = len(df)
    logger.info(f"Fetched {actual_data_len} bars for {symbol} ({at.value}) [{timeframe}], requested {min_limit}")
    
    # 如果数据不足200根，记录警告（但继续处理，让评分函数返回错误）
    if actual_data_len < 200:
        logger.warning(f"Warning: Only {actual_data_len} bars available for {symbol}, EMA200 calculation may fail")

    timestamps = pd.to_datetime(df['timestamp'])
    price_df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    current_price = float(price_df['close'].iloc[-1])

    # Scale prices for small-price coins (like PEPE)
    scale_factor = get_price_scale_factor(current_price)
    scaled_price_df = scale_ohlcv_df(price_df, scale_factor)
    if scale_factor != 1.0:
        logger.info(f"Scaling {symbol} prices by {scale_factor}x for prediction")

    # Generate future timestamps based on timeframe
    y_timestamps = pd.Series([timestamps.iloc[-1] + timedelta(hours=hours_per_bar * i) for i in range(1, horizon + 1)])

    # Run Kronos prediction with scaled prices
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: state.predictor.predict_multi_sample(
            scaled_price_df, timestamps, y_timestamps, horizon,
            T=1.0, top_p=0.9, sample_count=sample_count
        )
    )

    # Unscale prediction results back to original price range
    result = unscale_prediction(result, scale_factor)

    # Calculate metrics
    close_samples = result['all_samples'][:, :, 3]
    upside_prob = float(np.mean(close_samples[:, -1] > current_price) * 100)

    returns = np.log(price_df['close'].values[1:] / price_df['close'].values[:-1])
    # Annualize volatility based on timeframe
    bars_per_day = 24 // hours_per_bar
    volatility = float(np.std(returns[-bars_per_day:]) * np.sqrt(bars_per_day) * 100) if len(returns) >= bars_per_day else 0

    price_ranges = calculate_price_ranges(close_samples, current_price)
    supports, resistances = find_key_levels(close_samples, current_price)

    spread = (result['p90'] - result['p10']) / result['mean']
    confidence = float(max(0.3, min(0.95, 1.0 - float(np.mean(spread)) * 2)))
    buy_point_score, buy_point_reasons = calculate_trend_buy_point_score(price_df)
    buy_point_label = get_buy_point_label(buy_point_score, buy_point_reasons)

    # Build history data (last 72 bars = 3 days)
    history_len = min(72, len(df))
    history = []
    for i in range(len(df) - history_len, len(df)):
        ts = int(timestamps.iloc[i].timestamp())
        history.append(HistoryBar(
            time=ts,
            open=smart_round(float(price_df['open'].iloc[i])),
            high=smart_round(float(price_df['high'].iloc[i])),
            low=smart_round(float(price_df['low'].iloc[i])),
            close=smart_round(float(price_df['close'].iloc[i])),
            volume=round(float(price_df['volume'].iloc[i]), 2),
        ))

    # Build forecast data
    forecast = []
    last_ts = int(timestamps.iloc[-1].timestamp())
    seconds_per_bar = hours_per_bar * 3600
    for i in range(horizon):
        forecast.append(ForecastBar(
            time=last_ts + (i + 1) * seconds_per_bar,
            mean=smart_round(float(result['mean'][i])),
            min=smart_round(float(result['min'][i])),
            max=smart_round(float(result['max'][i])),
            p10=smart_round(float(result['p10'][i])),
            p25=smart_round(float(result['p25'][i])),
            p75=smart_round(float(result['p75'][i])),
            p90=smart_round(float(result['p90'][i])),
        ))

    # Trading suggestions
    entry = smart_round(float(result['p25'][0]))
    sl = smart_round(float(result['p10'][horizon // 2]))
    tp = smart_round(float(result['p75'][horizon - 1]))
    risk = entry - sl
    rr = round((tp - entry) / risk, 2) if risk > 0 else None

    return WeatherForecast(
        symbol=symbol,
        asset_type=at.value,
        timeframe=timeframe,
        current_price=smart_round(current_price),
        generated_at=datetime.now(timezone.utc).isoformat(),
        horizon=horizon,
        price_ranges=price_ranges,
        support_levels=supports,
        resistance_levels=resistances,
        history=history,
        forecast=forecast,
        upside_prob=round(upside_prob, 1),
        volatility=round(volatility, 2),
        confidence=round(confidence, 2),
        suggested_entry=entry,
        suggested_stop_loss=sl,
        suggested_take_profit=tp,
        risk_reward_ratio=rr,
        buy_point_score=buy_point_score,
        buy_point_label=buy_point_label,
        buy_point_reasons=buy_point_reasons[:5],
    )


@app.get("/api/chart/{symbol:path}")
async def get_chart_image(
    symbol: str,
    asset_type: Optional[str] = None,
    timeframe: str = Query(default="1h", regex="^(1h|4h|1d|1w)$"),
    horizon: Optional[int] = None,
    sample_count: int = Query(default=5, ge=3, le=50)
):
    """Generate forecast chart image (PNG) - same style as Kronos official demo."""
    symbol = symbol.upper().replace("-", "/")
    at = AssetType(asset_type) if asset_type else detect_asset_type(symbol)

    # Get timeframe config
    tf = TimeFrame(timeframe)
    tf_config = TIMEFRAME_CONFIG[tf]
    hours_per_bar = tf_config["hours"]

    # Use default horizon for timeframe if not specified
    if horizon is None:
        horizon = tf_config["horizon_default"]
    horizon = max(1, min(horizon, 168))

    if not state.is_ready:
        raise HTTPException(503, "Model not ready. Please wait.")

    # Fetch data with appropriate timeframe
    df = await fetch_ohlcv(symbol, at, timeframe, tf_config["limit"])
    if df is None or len(df) < 30:
        raise HTTPException(400, f"Insufficient data for {symbol}")

    timestamps = pd.to_datetime(df['timestamp'])
    price_df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    current_price = float(price_df['close'].iloc[-1])

    # Scale prices for small-price coins (like PEPE)
    scale_factor = get_price_scale_factor(current_price)
    scaled_price_df = scale_ohlcv_df(price_df, scale_factor)

    y_timestamps = pd.Series([timestamps.iloc[-1] + timedelta(hours=hours_per_bar * i) for i in range(1, horizon + 1)])

    # Run Kronos prediction with scaled prices
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: state.predictor.predict_multi_sample(
            scaled_price_df, timestamps, y_timestamps, horizon,
            T=1.0, top_p=0.9, sample_count=sample_count
        )
    )

    # Unscale prediction results
    result = unscale_prediction(result, scale_factor)

    # Prepare data for chart (use original unscaled prices for history)
    hist_len = min(72, len(df))
    hist_timestamps = timestamps.iloc[-hist_len:]
    hist_close = price_df['close'].iloc[-hist_len:]
    hist_volume = price_df['volume'].iloc[-hist_len:]

    # Close samples: shape (sample_count, horizon)
    close_samples = result['all_samples'][:, :, 3]
    # Volume samples (if available)
    volume_samples = result['all_samples'][:, :, 4] if result['all_samples'].shape[2] > 4 else None

    # Generate chart
    chart_bytes = create_forecast_chart(
        symbol=symbol,
        hist_timestamps=hist_timestamps,
        hist_close=hist_close,
        hist_volume=hist_volume,
        pred_timestamps=y_timestamps,
        close_samples=close_samples,
        volume_samples=volume_samples,
        horizon=horizon,
        timeframe=timeframe
    )

    return Response(content=chart_bytes, media_type="image/png")


@app.post("/api/simulate", response_model=SimulationResult)
async def simulate_trade(request: SimulationRequest):
    """Trade Sandbox - Simulate trade using Kronos AI predictions."""
    symbol = request.symbol.upper().replace("-", "/")
    at = AssetType(request.asset_type) if request.asset_type else detect_asset_type(symbol)

    if not state.is_ready:
        raise HTTPException(503, "Model not ready. Please wait.")

    # Fetch historical data
    df = await fetch_ohlcv(symbol, at, "1h", 100)
    if df is None or len(df) < 50:
        raise HTTPException(400, f"Cannot fetch data for {symbol}")

    timestamps = pd.to_datetime(df['timestamp'])
    price_df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    current = float(price_df['close'].iloc[-1])
    entry = request.entry_price or current

    # Scale prices for small-price coins
    scale_factor = get_price_scale_factor(current)
    scaled_price_df = scale_ohlcv_df(price_df, scale_factor)

    # Generate future timestamps
    y_timestamps = pd.Series([timestamps.iloc[-1] + timedelta(hours=i) for i in range(1, request.horizon + 1)])

    # Run Kronos prediction with more samples for better simulation
    sample_count = 30  # More samples for reliable simulation
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: state.predictor.predict_multi_sample(
            scaled_price_df, timestamps, y_timestamps, request.horizon,
            T=1.0, top_p=0.9, sample_count=sample_count
        )
    )

    # Unscale prediction results
    result = unscale_prediction(result, scale_factor)

    # Extract close price paths from Kronos predictions
    # all_samples shape: (sample_count, horizon, 5) where 5 = [open, high, low, close, volume]
    close_paths = result['all_samples'][:, :, 3]  # shape: (sample_count, horizon)

    # Run simulation with Kronos-predicted paths
    sim = run_simulation_with_paths(
        paths=close_paths,
        entry=entry,
        size=request.position_size,
        action=request.action,
        sl=request.stop_loss,
        tp=request.take_profit,
        leverage=request.leverage
    )

    rec, issues, opts = diagnose_trade(sim, request.stop_loss, request.take_profit)

    return SimulationResult(
        symbol=symbol, asset_type=at.value, action=request.action,
        entry_price=smart_round(entry), position_size=request.position_size, leverage=request.leverage,
        win_rate=round(sim["win_rate"], 1),
        expected_pnl=round(sim["expected_pnl"], 2),
        expected_pnl_percent=round(sim["expected_pnl_percent"], 2),
        max_profit=round(sim["max_profit"], 2),
        max_loss=round(sim["max_loss"], 2),
        hit_take_profit_pct=round(sim["hit_take_profit_pct"], 1),
        hit_stop_loss_pct=round(sim["hit_stop_loss_pct"], 1),
        expired_pct=round(sim["expired_pct"], 1),
        avg_bars_to_exit=round(sim["avg_bars_to_exit"], 1),
        max_drawdown=round(sim["max_drawdown"], 1),
        recommendation=rec, issues=issues, optimizations=opts,
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

                kronos_hint = "高度不確定 - 可能大波動" if anomaly > 70 else "正常波動" if anomaly < 30 else "值得關注"

                items.append(RadarItem(
                    symbol=symbol, name=name, asset_type=at.value,
                    anomaly_score=round(anomaly, 1), signal=signal,
                    signals=signals if signals else ["正常"],
                    price=smart_round(current), price_change=round(price_change, 2),
                    kronos_hint=kronos_hint,
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


class ZctTouchPoolScanBody(BaseModel):
    """POST /api/zct-vwap/touch-pool-scan：近 N 天 walk-forward + 触轨池筛选（同步，可能数分钟）。"""

    symbols: str = Field(
        default="ZECUSDT,ONDOUSDT,1000SHIBUSDT",
        description="Comma-separated USDT perpetual symbols（仅 symbols_source=request 时使用）",
    )
    symbols_source: Literal["request", "hot_oi_plus_default_22"] = Field(
        default="hot_oi_plus_default_22",
        description="hot_oi_plus_default_22=生产默认：worth_watch_hot_oi ∪ 扫描器默认 22 永续；request=使用 symbols",
    )
    days: float = Field(default=3.0, ge=0.25, le=30.0)
    min_touch_trades: int = Field(default=100, ge=0, le=200_000)
    min_touch_win_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    strict_greater_touch: bool = Field(default=False)
    strict_greater_rate: bool = Field(default=False)
    signal_interval: str = Field(default="1m", description="1m or 5m")
    sleep_between_symbols: float = Field(default=0.25, ge=0.0, le=10.0)
    persist_db: bool = Field(
        default=True,
        description="walk-forward 完成后：先清空 zct_vwap_touch_pool 再写入本轮入选，并追加 zct_vwap_touch_pool_runs",
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


@app.post("/api/zct-vwap/touch-pool-scan")
async def post_zct_touch_pool_scan(body: ZctTouchPoolScanBody = Body(...)):
    """ZCT walk-forward 触轨池；可选落库 accumulation.db（先清空入选表再写入）。"""
    from starlette.concurrency import run_in_threadpool

    iv = str(body.signal_interval or "1m").strip().lower()
    if iv not in ("1m", "5m"):
        raise HTTPException(status_code=400, detail="signal_interval must be 1m or 5m")

    src = str(body.symbols_source or "hot_oi_plus_default_22").strip().lower()
    if src not in ("request", "hot_oi_plus_default_22"):
        raise HTTPException(status_code=400, detail="invalid_symbols_source")

    if src == "hot_oi_plus_default_22":
        from zct_vwap_asset_pool import touch_pool_symbols_hot_oi_plus_default_22

        syms = touch_pool_symbols_hot_oi_plus_default_22()
        scan_src = "hot_oi_plus_default_22"
    else:
        syms = [x.strip().upper() for x in (body.symbols or "").split(",") if x.strip()]
        scan_src = None

    if not syms:
        raise HTTPException(status_code=400, detail="empty_symbols")

    persist = bool(body.persist_db)

    def _work():
        from accumulation_radar import init_db
        from zct_vwap_asset_pool import run_asset_pool_scan
        from zct_vwap_touch_pool_db import touch_pool_ensure_schema, touch_pool_write_db

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
            symbols_source=scan_src,
        )
        if persist:
            conn = init_db()
            try:
                touch_pool_ensure_schema(conn)
                touch_pool_write_db(conn, out)
            finally:
                conn.close()
        return {"ok": True, "pool": out, "persisted_db": persist}

    try:
        return await run_in_threadpool(_work)
    except Exception as e:
        logger.exception("zct touch_pool scan failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_touch_pool_scan_failed")


@app.get("/api/zct-hot-oi/summary")
async def get_zct_hot_oi_summary():
    """兼容旧路径：与 GET /api/zct-vwap/summary 相同（统一 zct_vwap_*）。"""
    try:
        from zct_vwap_api import load_zct_vwap_summary

        return load_zct_vwap_summary()
    except Exception as e:
        logger.warning("zct_hot_oi summary failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_hot_oi_summary_error")


@app.get("/api/zct-hot-oi/equity-curve")
async def get_zct_hot_oi_equity_curve():
    """兼容旧路径：与 GET /api/zct-vwap/equity-curve 相同。"""
    try:
        from zct_vwap_api import load_zct_equity_curve

        return load_zct_equity_curve()
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
        out = patch_zct_vwap_manual(signal_id, updates)
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
    """兼容旧路径：与 POST /api/zct-vwap/maintenance/clear-db 相同（清空统一 zct_vwap_*）。"""
    return await post_zct_vwap_clear_db()


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
        description="pool | heat_watch | heat_zones | heat_bpc | oi | s2_funding | s6_alpha | zct_vwap | zct_vwap_resolve | zct_hot_oi（=zct_vwap）| zct_hot_oi_resolve（=zct_vwap_resolve）",
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
    "zct_hot_oi": run_zct_vwap_signal_task,
    "zct_hot_oi_resolve": run_zct_vwap_resolve_only_task,
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
    - zct_hot_oi / zct_hot_oi_resolve: 与 zct_vwap / zct_vwap_resolve 相同（兼容旧 task 名；已统一到 zct_vwap_* 表）
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


# ============== Backtesting ==============

class BacktestResult(BaseModel):
    symbol: str
    asset_type: str
    timeframe: str
    test_periods: int
    horizon: int
    direction_accuracy: float  # % of correct up/down predictions
    mean_absolute_error: float  # Average absolute % error
    within_10pct_accuracy: float  # % predictions within 10% of actual
    within_5pct_accuracy: float  # % predictions within 5% of actual
    profitable_trades_pct: float  # If traded based on prediction
    avg_prediction_confidence: float
    details: List[Dict[str, Any]]  # Individual backtest results


# Simple backtest cache (symbol -> result, expires after 10 minutes)
_backtest_cache: Dict[str, Tuple[float, BacktestResult]] = {}
BACKTEST_CACHE_TTL = 600  # 10 minutes


@app.get("/api/backtest/{symbol:path}", response_model=BacktestResult)
async def run_backtest(
    symbol: str,
    asset_type: Optional[str] = None,
    timeframe: str = Query(default="1h", regex="^(1h|4h|1d|1w)$"),
    test_periods: int = Query(default=3, ge=1, le=10),  # Reduced default and max
    horizon: int = Query(default=6, ge=1, le=24),  # Reduced default and max for speed
):
    """
    Run backtesting on historical data.

    Tests Kronos predictions against actual historical outcomes.
    Note: Reduced parameters for faster execution on CPU.
    """
    symbol = symbol.upper().replace("-", "/")
    at = AssetType(asset_type) if asset_type else detect_asset_type(symbol)

    # Check cache first
    cache_key = f"{symbol}_{at.value}_{timeframe}_{test_periods}_{horizon}"
    if cache_key in _backtest_cache:
        cached_time, cached_result = _backtest_cache[cache_key]
        if time_module.time() - cached_time < BACKTEST_CACHE_TTL:
            logger.info(f"Returning cached backtest for {symbol}")
            return cached_result

    # Get timeframe config
    tf = TimeFrame(timeframe)
    tf_config = TIMEFRAME_CONFIG[tf]
    hours_per_bar = tf_config["hours"]

    if not state.is_ready:
        raise HTTPException(503, "Model not ready. Please wait.")

    # Fetch more historical data for backtesting
    # Need: test_periods * horizon + context_window
    data_needed = test_periods * horizon + 100
    df = await fetch_ohlcv(symbol, at, timeframe, data_needed)
    if df is None or len(df) < data_needed:
        raise HTTPException(400, f"Insufficient historical data for backtesting {symbol}")

    timestamps = pd.to_datetime(df['timestamp'])
    price_df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    # Scale prices for small-price coins (calculate once based on latest price)
    current_price = float(price_df['close'].iloc[-1])
    scale_factor = get_price_scale_factor(current_price)
    scaled_price_df = scale_ohlcv_df(price_df, scale_factor)

    # Run backtests
    details = []
    direction_correct = 0
    errors = []
    within_10 = 0
    within_5 = 0
    profitable = 0
    confidences = []

    for i in range(test_periods):
        # Calculate indices for this test period
        # We predict from end_idx for the next 'horizon' bars
        end_idx = len(df) - (test_periods - i) * horizon
        if end_idx < 80:  # Need at least 80 bars of context
            continue

        # Historical data up to this point (use scaled data for prediction)
        hist_df = scaled_price_df.iloc[:end_idx]
        hist_ts = timestamps.iloc[:end_idx]
        start_price = float(price_df['close'].iloc[end_idx - 1])  # Original unscaled price

        # Actual future prices (what actually happened) - use original prices
        actual_future = price_df['close'].iloc[end_idx:end_idx + horizon].values
        if len(actual_future) < horizon:
            continue
        actual_end_price = float(actual_future[-1])

        # Generate prediction timestamps
        y_ts = pd.Series([hist_ts.iloc[-1] + timedelta(hours=hours_per_bar * j) for j in range(1, horizon + 1)])

        try:
            # Run Kronos prediction with scaled data
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda h_df=hist_df, h_ts=hist_ts, y=y_ts, hor=horizon: state.predictor.predict_multi_sample(
                    h_df, h_ts, y, hor,
                    T=1.0, top_p=0.9, sample_count=3  # Reduced from 5 to 3 for faster backtest
                )
            )

            # Unscale prediction results
            result = unscale_prediction(result, scale_factor)

            pred_mean_end = float(result['mean'][-1])
            pred_upside_prob = float(np.mean(result['all_samples'][:, -1, 3] > start_price) * 100)

            # Calculate metrics
            actual_direction = 1 if actual_end_price > start_price else -1
            pred_direction = 1 if pred_mean_end > start_price else -1
            direction_match = actual_direction == pred_direction
            if direction_match:
                direction_correct += 1

            # Percentage error
            pct_error = abs(pred_mean_end - actual_end_price) / start_price * 100
            errors.append(pct_error)

            if pct_error <= 10:
                within_10 += 1
            if pct_error <= 5:
                within_5 += 1

            # Simulated trade profitability
            if pred_upside_prob > 50:  # Would go long
                trade_pnl = (actual_end_price - start_price) / start_price * 100
            else:  # Would go short
                trade_pnl = (start_price - actual_end_price) / start_price * 100

            if trade_pnl > 0:
                profitable += 1

            # Confidence from spread
            spread = (result['p90'] - result['p10']) / result['mean']
            confidence = float(max(0.3, min(0.95, 1.0 - float(np.mean(spread)) * 2)))
            confidences.append(confidence)

            details.append({
                "test_date": hist_ts.iloc[-1].isoformat(),
                "start_price": smart_round(start_price),
                "predicted_end": smart_round(pred_mean_end),
                "actual_end": smart_round(actual_end_price),
                "direction_correct": direction_match,
                "pct_error": round(pct_error, 2),
                "upside_prob": round(pred_upside_prob, 1),
                "trade_pnl_pct": round(trade_pnl, 2),
                "confidence": round(confidence, 2),
            })

        except Exception as e:
            logger.warning(f"Backtest failed for period {i}: {e}")
            continue

    # Calculate overall metrics
    n = len(details)
    if n == 0:
        raise HTTPException(500, "Backtesting failed - no valid test periods")

    result = BacktestResult(
        symbol=symbol,
        asset_type=at.value,
        timeframe=timeframe,
        test_periods=n,
        horizon=horizon,
        direction_accuracy=round(direction_correct / n * 100, 1),
        mean_absolute_error=round(sum(errors) / n, 2),
        within_10pct_accuracy=round(within_10 / n * 100, 1),
        within_5pct_accuracy=round(within_5 / n * 100, 1),
        profitable_trades_pct=round(profitable / n * 100, 1),
        avg_prediction_confidence=round(sum(confidences) / n, 2) if confidences else 0,
        details=details,
    )

    # Cache the result
    _backtest_cache[cache_key] = (time_module.time(), result)
    return result


# ============== Backtesting ==============

class BacktestResult(BaseModel):
    symbol: str
    asset_type: str
    timeframe: str
    test_periods: int
    horizon: int
    direction_accuracy: float  # % of correct up/down predictions
    mean_absolute_error: float  # Average absolute % error
    within_10pct_accuracy: float  # % predictions within 10% of actual
    within_5pct_accuracy: float  # % predictions within 5% of actual
    profitable_trades_pct: float  # If traded based on prediction
    avg_prediction_confidence: float
    details: List[Dict[str, Any]]  # Individual backtest results


# Simple backtest cache (symbol -> result, expires after 10 minutes)
_backtest_cache: Dict[str, Tuple[float, BacktestResult]] = {}
BACKTEST_CACHE_TTL = 600  # 10 minutes


@app.get("/api/backtest/{symbol:path}", response_model=BacktestResult)
async def run_backtest(
    symbol: str,
    asset_type: Optional[str] = None,
    timeframe: str = Query(default="1h", regex="^(1h|4h|1d|1w)$"),
    test_periods: int = Query(default=3, ge=1, le=10),  # Reduced default and max
    horizon: int = Query(default=6, ge=1, le=24),  # Reduced default and max for speed
):
    """
    Run backtesting on historical data.

    Tests Kronos predictions against actual historical outcomes.
    Note: Reduced parameters for faster execution on CPU.
    """
    symbol = symbol.upper().replace("-", "/")
    at = AssetType(asset_type) if asset_type else detect_asset_type(symbol)

    # Check cache first
    cache_key = f"{symbol}_{at.value}_{timeframe}_{test_periods}_{horizon}"
    if cache_key in _backtest_cache:
        cached_time, cached_result = _backtest_cache[cache_key]
        if time_module.time() - cached_time < BACKTEST_CACHE_TTL:
            logger.info(f"Returning cached backtest for {symbol}")
            return cached_result

    # Get timeframe config
    tf = TimeFrame(timeframe)
    tf_config = TIMEFRAME_CONFIG[tf]
    hours_per_bar = tf_config["hours"]

    if not state.is_ready:
        raise HTTPException(503, "Model not ready. Please wait.")

    # Fetch more historical data for backtesting
    # Need: test_periods * horizon + context_window
    data_needed = test_periods * horizon + 100
    df = await fetch_ohlcv(symbol, at, timeframe, data_needed)
    if df is None or len(df) < data_needed:
        raise HTTPException(400, f"Insufficient historical data for backtesting {symbol}")

    timestamps = pd.to_datetime(df['timestamp'])
    price_df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    # Scale prices for small-price coins (calculate once based on latest price)
    current_price = float(price_df['close'].iloc[-1])
    scale_factor = get_price_scale_factor(current_price)
    scaled_price_df = scale_ohlcv_df(price_df, scale_factor)

    # Run backtests
    details = []
    direction_correct = 0
    errors = []
    within_10 = 0
    within_5 = 0
    profitable = 0
    confidences = []

    for i in range(test_periods):
        # Calculate indices for this test period
        # We predict from end_idx for the next 'horizon' bars
        end_idx = len(df) - (test_periods - i) * horizon
        if end_idx < 80:  # Need at least 80 bars of context
            continue

        # Historical data up to this point (use scaled data for prediction)
        hist_df = scaled_price_df.iloc[:end_idx]
        hist_ts = timestamps.iloc[:end_idx]
        start_price = float(price_df['close'].iloc[end_idx - 1])  # Original unscaled price

        # Actual future prices (what actually happened) - use original prices
        actual_future = price_df['close'].iloc[end_idx:end_idx + horizon].values
        if len(actual_future) < horizon:
            continue
        actual_end_price = float(actual_future[-1])

        # Generate prediction timestamps
        y_ts = pd.Series([hist_ts.iloc[-1] + timedelta(hours=hours_per_bar * j) for j in range(1, horizon + 1)])

        try:
            # Run Kronos prediction with scaled data
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda h_df=hist_df, h_ts=hist_ts, y=y_ts, hor=horizon: state.predictor.predict_multi_sample(
                    h_df, h_ts, y, hor,
                    T=1.0, top_p=0.9, sample_count=3  # Reduced from 5 to 3 for faster backtest
                )
            )

            # Unscale prediction results
            result = unscale_prediction(result, scale_factor)

            pred_mean_end = float(result['mean'][-1])
            pred_upside_prob = float(np.mean(result['all_samples'][:, -1, 3] > start_price) * 100)

            # Calculate metrics
            actual_direction = 1 if actual_end_price > start_price else -1
            pred_direction = 1 if pred_mean_end > start_price else -1
            direction_match = actual_direction == pred_direction
            if direction_match:
                direction_correct += 1

            # Percentage error
            pct_error = abs(pred_mean_end - actual_end_price) / start_price * 100
            errors.append(pct_error)

            if pct_error <= 10:
                within_10 += 1
            if pct_error <= 5:
                within_5 += 1

            # Simulated trade profitability
            if pred_upside_prob > 50:  # Would go long
                trade_pnl = (actual_end_price - start_price) / start_price * 100
            else:  # Would go short
                trade_pnl = (start_price - actual_end_price) / start_price * 100

            if trade_pnl > 0:
                profitable += 1

            # Confidence from spread
            spread = (result['p90'] - result['p10']) / result['mean']
            confidence = float(max(0.3, min(0.95, 1.0 - float(np.mean(spread)) * 2)))
            confidences.append(confidence)

            details.append({
                "test_date": hist_ts.iloc[-1].isoformat(),
                "start_price": smart_round(start_price),
                "predicted_end": smart_round(pred_mean_end),
                "actual_end": smart_round(actual_end_price),
                "direction_correct": direction_match,
                "pct_error": round(pct_error, 2),
                "upside_prob": round(pred_upside_prob, 1),
                "trade_pnl_pct": round(trade_pnl, 2),
                "confidence": round(confidence, 2),
            })

        except Exception as e:
            logger.warning(f"Backtest failed for period {i}: {e}")
            continue

    # Calculate overall metrics
    n = len(details)
    if n == 0:
        raise HTTPException(500, "Backtesting failed - no valid test periods")

    result = BacktestResult(
        symbol=symbol,
        asset_type=at.value,
        timeframe=timeframe,
        test_periods=n,
        horizon=horizon,
        direction_accuracy=round(direction_correct / n * 100, 1),
        mean_absolute_error=round(sum(errors) / n, 2),
        within_10pct_accuracy=round(within_10 / n * 100, 1),
        within_5pct_accuracy=round(within_5 / n * 100, 1),
        profitable_trades_pct=round(profitable / n * 100, 1),
        avg_prediction_confidence=round(sum(confidences) / n, 2) if confidences else 0,
        details=details,
    )

    # Cache the result
    _backtest_cache[cache_key] = (time_module.time(), result)
    return result


# ============== Main ==============

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    # Use "::" for IPv6 support (required by Railway)
    uvicorn.run(app, host="0.0.0.0", port=port)

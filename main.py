"""
Next K - Multi-Asset K-Line Weather Forecast API

A probabilistic K-line prediction tool powered by the Kronos foundation model.
Supports: Crypto, Stocks, Forex
"""

import asyncio
import base64
import io
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import time as time_module

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
# Static files removed - frontend is deployed separately on Vercel
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


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


state = AppState()

# HuggingFace model paths
TOKENIZER_PATH = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PATH = "NeoQuasar/Kronos-small"


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
        tf_map = {
            "1h": ("7d", "1h"),
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
        tf_map = {
            "1h": ("7d", "1h"),
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
    else:
        # 对真实数据进行质量检查和清洗
        try:
            df, issues = validate_and_clean_ohlcv(df, symbol, strict=True)
            if issues:
                logger.info(f"[{symbol}] 数据清洗完成: {len(issues)} 个问题已处理")
        except Exception as e:
            logger.error(f"[{symbol}] 数据质量检查失败: {e}，使用原始数据")
            # 如果清洗失败，继续使用原始数据（但可能影响预测质量）

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

    # Initialize Kronos model (background)
    asyncio.create_task(initialize_model())

    yield
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
    allow_credentials=True,
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
    crypto_connected: bool
    stocks_available: bool
    forex_available: bool
    version: str
    uptime: float


# ============== Helper Functions ==============

def calculate_optimal_data_limit(
    timeframe: str,
    horizon: int,
    use_case: str = "forecast"  # "forecast", "backtest", "simulation"
) -> int:
    """
    智能计算最优的历史数据量。
    
    策略：
    - 根据时间周期调整：短期周期需要更多数据，长期周期可以少一些
    - 根据预测步数调整：更长的预测需要更多历史上下文
    - 考虑模型限制：max_context=512，但实际使用可能更少
    - 平衡性能和准确性：更多数据通常更好，但也要考虑API限制和性能
    
    Args:
        timeframe: 时间周期 ("1h", "4h", "1d", "1w")
        horizon: 预测步数
        use_case: 使用场景
    
    Returns:
        最优的数据量（K线数量）
    """
    tf_hours = {"1h": 1, "4h": 4, "1d": 24, "1w": 168}
    hours_per_bar = tf_hours.get(timeframe, 1)
    
    # 基础数据量：根据时间周期调整
    # 短期周期（1h）需要更多数据以捕捉短期模式
    # 长期周期（1w）可以少一些，因为每根K线包含更多信息
    if hours_per_bar <= 1:
        # 1h: 需要更多数据（200-400）
        base_limit = 300
    elif hours_per_bar <= 4:
        # 4h: 中等数据量（150-300）
        base_limit = 250
    elif hours_per_bar <= 24:
        # 1d: 中等数据量（100-200）
        base_limit = 200
    else:
        # 1w: 较少数据量（52-100）
        base_limit = 80
    
    # 根据预测步数调整
    # 更长的预测需要更多历史上下文
    # horizon 24 为基准
    horizon_factor = 1.0 + (horizon - 24) / 24 * 0.3
    horizon_factor = max(0.8, min(1.5, horizon_factor))  # 限制在 0.8-1.5 倍
    
    # 根据使用场景调整
    if use_case == "backtest":
        # 回测需要更多数据以确保有足够的历史
        use_case_factor = 1.2
    elif use_case == "simulation":
        # 模拟需要足够的数据以准确计算波动率等指标
        use_case_factor = 1.1
    else:
        # 普通预测
        use_case_factor = 1.0
    
    optimal = int(base_limit * horizon_factor * use_case_factor)
    
    # 考虑模型限制和API限制
    # 模型 max_context=512，但实际使用可能更少（通常200-300）
    # 为了安全，限制在 500 以内
    # 同时考虑API限制（Binance通常支持1000，但为了性能限制在500）
    max_limit = 500
    
    # 最小数据量：至少需要足够的数据进行归一化和特征计算
    min_limit = 50
    
    return max(min_limit, min(optimal, max_limit))


def calculate_optimal_sample_count(
    horizon: int,
    timeframe: str,
    use_case: str = "forecast",  # "forecast", "backtest", "simulation"
    user_provided: Optional[int] = None
) -> int:
    """
    智能计算最优的 sample_count。
    
    策略：
    - forecast: 根据 horizon 和 timeframe 动态调整，确保统计可靠性
    - simulation: 使用更多样本（30-50）以获得更准确的交易模拟
    - backtest: 平衡准确性和速度，使用中等样本数（10-20）
    
    Args:
        horizon: 预测步数
        timeframe: 时间周期 ("1h", "4h", "1d", "1w")
        use_case: 使用场景
        user_provided: 用户提供的值（如果提供则优先使用，但会应用最小限制）
    
    Returns:
        最优的 sample_count
    """
    if user_provided is not None:
        # 用户提供了值，但根据场景应用最小限制
        if use_case == "simulation":
            return max(user_provided, 20)  # 模拟至少需要20个样本
        elif use_case == "backtest":
            return max(user_provided, 10)  # 回测至少需要10个样本
        else:
            return max(user_provided, 15)  # 普通预测至少需要15个样本
    
    # 根据使用场景选择策略
    if use_case == "simulation":
        # 交易模拟需要更多样本以获得可靠的胜率计算
        # 根据 horizon 调整：更长的预测需要更多样本
        base_samples = 30
        horizon_factor = min(horizon / 24, 1.5)  # horizon 24 为基准
        return int(base_samples * horizon_factor)
    
    elif use_case == "backtest":
        # 回测需要平衡准确性和速度
        # 使用中等样本数，但根据 horizon 调整
        # 至少使用 10 个样本以确保统计可靠性
        base_samples = 15
        horizon_factor = min(horizon / 12, 1.3)
        optimal = int(base_samples * horizon_factor)
        return max(10, optimal)  # 至少 10 个样本
    
    else:  # forecast
        # 普通预测：根据 horizon 和 timeframe 动态调整
        # 更长的预测和更短的时间周期需要更多样本
        tf_hours = {"1h": 1, "4h": 4, "1d": 24, "1w": 168}
        hours_per_bar = tf_hours.get(timeframe, 1)
        
        # 基础样本数：根据时间周期调整
        # 短期周期（1h）需要更多样本，长期周期（1w）可以少一些
        if hours_per_bar <= 1:
            base_samples = 25  # 1h: 25个样本
        elif hours_per_bar <= 4:
            base_samples = 20  # 4h: 20个样本
        elif hours_per_bar <= 24:
            base_samples = 20  # 1d: 20个样本
        else:
            base_samples = 15  # 1w: 15个样本
        
        # 根据 horizon 调整：更长的预测需要更多样本
        # horizon 24 为基准，每增加 24 步增加 5 个样本
        horizon_factor = 1.0 + (horizon - 24) / 24 * 0.2
        horizon_factor = max(0.8, min(1.5, horizon_factor))  # 限制在 0.8-1.5 倍
        
        optimal = int(base_samples * horizon_factor)
        
        # 确保在合理范围内
        return max(15, min(optimal, 100))


def calculate_technical_indicators(
    df: pd.DataFrame,
    include_trend: bool = True,
    include_momentum: bool = True,
    include_volatility: bool = True,
    include_volume: bool = True
) -> pd.DataFrame:
    """
    计算技术指标并添加到DataFrame。
    
    技术指标用于辅助决策，不直接输入Kronos模型（模型输入维度固定）。
    指标可用于：
    - 调整采样参数（T, top_p）
    - 调整置信度
    - 生成交易信号
    - 市场状态判断
    
    Args:
        df: OHLCV DataFrame，必须包含 'open', 'high', 'low', 'close', 'volume'
        include_trend: 是否计算趋势指标（MA, EMA, BIAS）
        include_momentum: 是否计算动量指标（RSI, MACD）
        include_volatility: 是否计算波动率指标（布林带, ATR）
        include_volume: 是否计算成交量指标（VMA, VOL_RATIO）
    
    Returns:
        包含技术指标的DataFrame（原始列 + 技术指标列）
    """
    df = df.copy()
    
    # 确保数据类型正确
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
    
    # ============== 趋势指标 ==============
    if include_trend:
        # 移动平均线（MA）
        for period in [5, 10, 20, 60]:
            if len(df) >= period:
                ma = pd.Series(closes).rolling(window=period, min_periods=1).mean()
                df[f'ma_{period}'] = ma.values
                # 乖离率（BIAS）
                df[f'bias_{period}'] = ((closes - ma.values) / (ma.values + 1e-8)) * 100
        
        # 指数移动平均（EMA）
        if len(df) >= 12:
            df['ema_12'] = pd.Series(closes).ewm(span=12, adjust=False, min_periods=1).mean().values
        if len(df) >= 26:
            df['ema_26'] = pd.Series(closes).ewm(span=26, adjust=False, min_periods=1).mean().values
    
    # ============== 动量指标 ==============
    if include_momentum:
        # RSI（相对强弱指数）
        if len(df) >= 14:
            delta = pd.Series(closes).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            df['rsi'] = (100 - (100 / (1 + rs))).values
        
        # MACD（移动平均收敛散度）
        if len(df) >= 26 and include_trend:
            if 'ema_12' in df.columns and 'ema_26' in df.columns:
                df['macd_dif'] = (df['ema_12'] - df['ema_26']).values
                df['macd_dea'] = pd.Series(df['macd_dif']).ewm(span=9, adjust=False, min_periods=1).mean().values
                df['macd_hist'] = ((df['macd_dif'] - df['macd_dea']) * 2).values
        
        # 价格变化率（ROC）
        if len(df) >= 10:
            df['roc_10'] = ((closes - pd.Series(closes).shift(10)) / (pd.Series(closes).shift(10) + 1e-8) * 100).values
    
    # ============== 波动率指标 ==============
    if include_volatility:
        # 布林带（Bollinger Bands）
        if len(df) >= 20:
            bb_period = 20
            bb_std = 2
            bb_middle = pd.Series(closes).rolling(window=bb_period, min_periods=1).mean()
            bb_std_val = pd.Series(closes).rolling(window=bb_period, min_periods=1).std()
            df['bb_middle'] = bb_middle.values
            df['bb_upper'] = (bb_middle + bb_std * bb_std_val).values
            df['bb_lower'] = (bb_middle - bb_std * bb_std_val).values
            df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / (bb_middle.values + 1e-8) * 100).values
            # 布林带位置（0-1，0=下轨，1=上轨）
            df['bb_position'] = ((closes - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)).values
        
        # ATR（平均真实波幅）
        if len(df) >= 14:
            tr1 = highs - lows
            tr2 = np.abs(highs - np.roll(closes, 1))
            tr3 = np.abs(lows - np.roll(closes, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            df['atr'] = pd.Series(tr).rolling(window=14, min_periods=1).mean().values
            # ATR百分比（相对于价格）
            df['atr_pct'] = (df['atr'] / (closes + 1e-8) * 100).values
    
    # ============== 成交量指标 ==============
    if include_volume and len(volumes) > 0:
        # 成交量移动平均
        for period in [5, 10, 20]:
            if len(df) >= period:
                vma = pd.Series(volumes).rolling(window=period, min_periods=1).mean()
                df[f'vma_{period}'] = vma.values
                # 成交量比率
                df[f'vol_ratio_{period}'] = volumes / (vma.values + 1e-8)
        
        # OBV（能量潮）
        if len(df) >= 2:
            price_change = np.diff(closes)
            volume_direction = np.where(price_change > 0, volumes[1:], 
                                       np.where(price_change < 0, -volumes[1:], 0))
            obv = np.concatenate([[volumes[0]], volumes[0] + np.cumsum(volume_direction)])
            df['obv'] = obv
    
    # 填充NaN值（使用前向填充）
    df = df.ffill().bfill()
    
    return df


def get_market_signals(tech_indicators: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """
    从技术指标中提取市场信号。
    
    Args:
        tech_indicators: 包含技术指标的DataFrame
        current_price: 当前价格
    
    Returns:
        包含各种市场信号的字典
    """
    signals = {
        'trend': 'neutral',  # 'bullish', 'bearish', 'neutral'
        'momentum': 'neutral',
        'volatility': 'normal',  # 'low', 'normal', 'high'
        'overbought_oversold': 'neutral',  # 'overbought', 'oversold', 'neutral'
        'signals': [],
        'confidence_boost': 0.0,  # 置信度提升因子
    }
    
    if tech_indicators.empty or len(tech_indicators) == 0:
        return signals
    
    last_row = tech_indicators.iloc[-1]
    
    # RSI信号
    if 'rsi' in last_row and not pd.isna(last_row['rsi']):
        rsi = last_row['rsi']
        if rsi > 70:
            signals['overbought_oversold'] = 'overbought'
            signals['signals'].append(f"RSI超买 ({rsi:.1f})")
            signals['confidence_boost'] -= 0.05
        elif rsi < 30:
            signals['overbought_oversold'] = 'oversold'
            signals['signals'].append(f"RSI超卖 ({rsi:.1f})")
            signals['confidence_boost'] -= 0.05
        elif 40 <= rsi <= 60:
            signals['confidence_boost'] += 0.03  # RSI中性时置信度提升
    
    # MACD信号
    if 'macd_hist' in last_row and not pd.isna(last_row['macd_hist']):
        macd_hist = last_row['macd_hist']
        if macd_hist > 0.5:
            signals['momentum'] = 'bullish'
            signals['signals'].append("MACD金叉 - 看涨")
            signals['confidence_boost'] += 0.02
        elif macd_hist < -0.5:
            signals['momentum'] = 'bearish'
            signals['signals'].append("MACD死叉 - 看跌")
            signals['confidence_boost'] += 0.02
    
    # 移动平均线趋势
    if 'ma_20' in last_row and not pd.isna(last_row['ma_20']):
        ma20 = last_row['ma_20']
        if current_price > ma20 * 1.02:
            signals['trend'] = 'bullish'
            signals['signals'].append("价格在20日均线上方 - 趋势向上")
        elif current_price < ma20 * 0.98:
            signals['trend'] = 'bearish'
            signals['signals'].append("价格在20日均线下方 - 趋势向下")
    
    # 布林带信号
    if 'bb_position' in last_row and not pd.isna(last_row['bb_position']):
        bb_pos = last_row['bb_position']
        if bb_pos > 0.8:
            signals['signals'].append("价格接近布林带上轨 - 可能回调")
            signals['confidence_boost'] -= 0.03
        elif bb_pos < 0.2:
            signals['signals'].append("价格接近布林带下轨 - 可能反弹")
            signals['confidence_boost'] -= 0.03
        elif 0.3 <= bb_pos <= 0.7:
            signals['confidence_boost'] += 0.02  # 价格在布林带中间，置信度提升
    
    # 波动率信号
    if 'bb_width' in last_row and not pd.isna(last_row['bb_width']):
        bb_width = last_row['bb_width']
        if bb_width > 5:
            signals['volatility'] = 'high'
            signals['signals'].append(f"高波动率 (带宽={bb_width:.2f}%)")
        elif bb_width < 2:
            signals['volatility'] = 'low'
            signals['signals'].append(f"低波动率 (带宽={bb_width:.2f}%)")
    
    # 成交量信号
    if 'vol_ratio_20' in last_row and not pd.isna(last_row['vol_ratio_20']):
        vol_ratio = last_row['vol_ratio_20']
        if vol_ratio > 1.5:
            signals['signals'].append(f"成交量放大 ({vol_ratio:.2f}x)")
            signals['confidence_boost'] += 0.01
        elif vol_ratio < 0.5:
            signals['signals'].append(f"成交量萎缩 ({vol_ratio:.2f}x)")
    
    return signals


def validate_and_clean_ohlcv(df: pd.DataFrame, symbol: str, strict: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    验证和清洗OHLCV数据。
    
    Args:
        df: 原始OHLCV DataFrame
        symbol: 交易对符号（用于日志）
        strict: 是否严格模式（True: 修复错误，False: 仅警告）
    
    Returns:
        Tuple[清洗后的DataFrame, 问题列表]
    """
    issues = []
    df = df.copy()
    original_len = len(df)
    
    # 1. 检查必需的列
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # 2. 处理缺失值
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0].to_dict()
        issues.append(f"发现缺失值: {null_cols}")
        
        # 前向填充，然后后向填充
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 如果仍有缺失值，使用插值
        if df.isnull().any().any():
            for col in ['open', 'high', 'low', 'close']:
                if df[col].isnull().any():
                    df[col] = df[col].interpolate(method='linear')
            if df['volume'].isnull().any():
                df['volume'] = df['volume'].fillna(0)
        
        # 如果还有缺失值，删除这些行
        if df.isnull().any().any():
            df = df.dropna()
            issues.append(f"删除了包含缺失值的行，剩余 {len(df)} 条")
    
    # 3. 确保数据类型正确
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除转换失败的行
    df = df.dropna()
    
    # 4. 检查OHLC逻辑关系
    # high >= max(open, close)
    invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
    # low <= min(open, close)
    invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
    # high >= low
    invalid_range = df['high'] < df['low']
    
    invalid_count = (invalid_high | invalid_low | invalid_range).sum()
    if invalid_count > 0:
        issues.append(f"发现 {invalid_count} 条OHLC逻辑错误")
        
        if strict:
            # 修复逻辑错误
            # 确保数据类型一致（float）
            for col in ['open', 'high', 'low', 'close']:
                if df[col].dtype != 'float64':
                    df[col] = df[col].astype(float)
            
            # 确保 high >= max(open, close)
            if invalid_high.sum() > 0:
                max_oc = df.loc[invalid_high, ['open', 'close']].max(axis=1)
                df.loc[invalid_high, 'high'] = max_oc.astype(float)
            # 确保 low <= min(open, close)
            if invalid_low.sum() > 0:
                min_oc = df.loc[invalid_low, ['open', 'close']].min(axis=1)
                df.loc[invalid_low, 'low'] = min_oc.astype(float)
            # 确保 high >= low
            if invalid_range.sum() > 0:
                max_oc_range = df.loc[invalid_range, ['open', 'close']].max(axis=1)
                min_oc_range = df.loc[invalid_range, ['open', 'close']].min(axis=1)
                df.loc[invalid_range, 'high'] = max_oc_range.astype(float)
                df.loc[invalid_range, 'low'] = min_oc_range.astype(float)
            issues.append("已自动修复OHLC逻辑错误")
        else:
            # 删除无效行
            df = df[~(invalid_high | invalid_low | invalid_range)]
            issues.append(f"已删除 {original_len - len(df)} 条无效数据")
    
    # 5. 检查价格和成交量是否为非负数
    negative_prices = (df[['open', 'high', 'low', 'close']] < 0).any(axis=1)
    negative_volume = df['volume'] < 0
    
    if negative_prices.sum() > 0:
        issues.append(f"发现 {negative_prices.sum()} 条负价格数据")
        if strict:
            # 删除负价格的行
            df = df[~negative_prices]
        else:
            # 将负价格设为0（通常不应该发生）
            df.loc[negative_prices, ['open', 'high', 'low', 'close']] = 0
    
    if negative_volume.sum() > 0:
        issues.append(f"发现 {negative_volume.sum()} 条负成交量数据")
        df.loc[negative_volume, 'volume'] = 0
    
    # 6. 异常值检测（使用IQR方法）
    for col in ['open', 'high', 'low', 'close']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # 使用3倍IQR，更宽松
        upper_bound = Q3 + 3 * IQR
        
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            issues.append(f"{col} 列发现 {outlier_count} 个异常值（超出3倍IQR）")
            
            if strict and outlier_count < len(df) * 0.1:  # 如果异常值少于10%
                # 使用中位数替换异常值
                median_val = df[col].median()
                df.loc[outliers, col] = median_val
                issues.append(f"已用中位数替换 {col} 的异常值")
            elif outlier_count >= len(df) * 0.1:
                # 异常值太多，可能是数据源问题，仅警告
                issues.append(f"警告: {col} 异常值过多（{outlier_count}/{len(df)}），可能影响预测准确性")
    
    # 7. 检查价格跳跃（单根K线内价格变化过大）
    # 计算每根K线的价格变化率
    price_change_pct = abs((df['close'] - df['open']) / df['open']) * 100
    extreme_changes = price_change_pct > 50  # 单根K线内变化超过50%
    
    if extreme_changes.sum() > 0:
        issues.append(f"发现 {extreme_changes.sum()} 根K线价格变化超过50%")
        # 仅记录，不自动修复（可能是真实的市场波动）
    
    # 8. 检查时间戳连续性
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 检查是否有重复时间戳
        duplicates = df['timestamp'].duplicated()
        if duplicates.sum() > 0:
            issues.append(f"发现 {duplicates.sum()} 个重复时间戳")
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
            issues.append("已删除重复时间戳，保留最后一条")
    
    # 9. 确保数据按时间排序
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 10. 最终检查：确保数据量足够
    if len(df) < 10:
        raise ValueError(f"数据清洗后仅剩 {len(df)} 条，不足以进行预测（需要至少10条）")
    
    if len(df) < original_len * 0.5:
        issues.append(f"警告: 数据清洗后仅剩 {len(df)}/{original_len} 条（{len(df)/original_len*100:.1f}%）")
    
    # 记录问题
    if issues:
        logger.warning(f"[{symbol}] 数据质量检查发现问题: {'; '.join(issues)}")
    else:
        logger.info(f"[{symbol}] 数据质量检查通过，共 {len(df)} 条数据")
    
    return df, issues


def calculate_adaptive_sampling_params(
    price_df: pd.DataFrame,
    volatility: float,
    timeframe: str,
    horizon: int,
    user_temperature: Optional[float] = None,
    user_top_p: Optional[float] = None,
    tech_indicators: Optional[pd.DataFrame] = None
) -> Tuple[float, float]:
    """
    根据市场状态动态计算采样参数 T 和 top_p。
    
    策略：
    - T (temperature): 控制随机性
      * 高波动率 → 提高 T (1.0-1.5)，增加多样性
      * 低波动率 → 降低 T (0.7-1.0)，更保守
      * 强趋势 → 适度提高 T
      * 震荡市场 → 降低 T
    
    - top_p (nucleus sampling): 控制采样范围
      * 高波动率 → 提高 top_p (0.9-0.99)，覆盖更多可能性
      * 低波动率 → 降低 top_p (0.8-0.9)，更聚焦
      * 强趋势 → 降低 top_p，更确定
      * 震荡市场 → 提高 top_p
    
    Args:
        price_df: 价格数据 DataFrame (包含 open, high, low, close, volume)
        volatility: 历史波动率（年化百分比）
        timeframe: 时间周期
        horizon: 预测步数
        user_temperature: 用户指定的温度（如果提供则优先使用，但会应用最小/最大限制）
        user_top_p: 用户指定的 top_p（如果提供则优先使用，但会应用最小/最大限制）
    
    Returns:
        Tuple[T, top_p]
    """
    # 如果用户提供了参数，应用限制后返回
    if user_temperature is not None:
        T = max(0.5, min(2.0, user_temperature))
    if user_top_p is not None:
        top_p = max(0.7, min(0.99, user_top_p))
        if user_temperature is not None:
            return T, top_p
    
    # 计算市场状态指标
    closes = price_df['close'].values
    
    # 1. 计算趋势强度（使用线性回归斜率）
    if len(closes) >= 20:
        x = np.arange(len(closes[-20:]))
        y = closes[-20:]
        slope = np.polyfit(x, y, 1)[0]
        trend_strength = abs(slope) / closes[-1] * 100  # 归一化趋势强度
    else:
        trend_strength = 0.0
    
    # 2. 计算近期波动率变化（波动率加速度）
    if len(closes) >= 30:
        recent_returns = np.log(closes[-15:] / closes[-16:-1])
        hist_returns = np.log(closes[-30:-15] / closes[-31:-16])
        recent_vol = np.std(recent_returns)
        hist_vol = np.std(hist_returns)
        vol_acceleration = (recent_vol - hist_vol) / (hist_vol + 1e-8)  # 波动率变化率
    else:
        vol_acceleration = 0.0
    
    # 3. 计算价格变化率（动量）
    if len(closes) >= 10:
        price_change_pct = abs((closes[-1] - closes[-10]) / closes[-10]) * 100
    else:
        price_change_pct = 0.0
    
    # 4. 归一化波动率（相对于时间周期）
    tf_hours = {"1h": 1, "4h": 4, "1d": 24, "1w": 168}
    hours_per_bar = tf_hours.get(timeframe, 1)
    
    # 波动率阈值（根据时间周期调整）
    # 1h: 高波动 > 50%, 低波动 < 20%
    # 1d: 高波动 > 30%, 低波动 < 10%
    vol_threshold_high = 50.0 / np.sqrt(hours_per_bar)
    vol_threshold_low = 20.0 / np.sqrt(hours_per_bar)
    
    # 计算 T (temperature)
    base_T = 1.0
    
    # 根据波动率调整
    if volatility > vol_threshold_high:
        # 高波动率：提高 T
        vol_factor = min(1.5, 1.0 + (volatility - vol_threshold_high) / vol_threshold_high * 0.3)
    elif volatility < vol_threshold_low:
        # 低波动率：降低 T
        vol_factor = max(0.7, 1.0 - (vol_threshold_low - volatility) / vol_threshold_low * 0.3)
    else:
        vol_factor = 1.0
    
    # 根据波动率加速度调整
    if vol_acceleration > 0.2:
        # 波动率上升：提高 T
        vol_acc_factor = 1.0 + min(0.2, vol_acceleration * 0.5)
    elif vol_acceleration < -0.2:
        # 波动率下降：降低 T
        vol_acc_factor = 1.0 + max(-0.2, vol_acceleration * 0.5)
    else:
        vol_acc_factor = 1.0
    
    # 根据趋势强度调整
    if trend_strength > 1.0:
        # 强趋势：适度提高 T（增加多样性）
        trend_factor = 1.0 + min(0.15, trend_strength / 10.0)
    else:
        # 弱趋势/震荡：降低 T（更保守）
        trend_factor = max(0.9, 1.0 - (1.0 - trend_strength) * 0.1)
    
    # 根据预测步数调整（更长的预测需要更多多样性）
    horizon_factor = 1.0 + (horizon - 24) / 24 * 0.1
    horizon_factor = max(0.95, min(1.15, horizon_factor))
    
    T = base_T * vol_factor * vol_acc_factor * trend_factor * horizon_factor
    T = max(0.5, min(2.0, T))  # 限制在合理范围
    
    # 计算 top_p (nucleus sampling)
    base_top_p = 0.9
    
    # 根据波动率调整
    if volatility > vol_threshold_high:
        # 高波动率：提高 top_p（覆盖更多可能性）
        vol_top_p_factor = min(0.99, 0.9 + (volatility - vol_threshold_high) / vol_threshold_high * 0.05)
    elif volatility < vol_threshold_low:
        # 低波动率：降低 top_p（更聚焦）
        vol_top_p_factor = max(0.8, 0.9 - (vol_threshold_low - volatility) / vol_threshold_low * 0.1)
    else:
        vol_top_p_factor = 0.9
    
    # 根据趋势强度调整
    if trend_strength > 1.0:
        # 强趋势：降低 top_p（更确定）
        trend_top_p_factor = max(0.85, 0.9 - trend_strength / 20.0)
    else:
        # 震荡市场：提高 top_p（覆盖更多可能性）
        trend_top_p_factor = min(0.95, 0.9 + (1.0 - trend_strength) * 0.05)
    
    # 根据价格变化率调整
    if price_change_pct > 10:
        # 大幅波动：提高 top_p
        momentum_factor = min(0.99, 0.9 + price_change_pct / 100.0 * 0.05)
    else:
        momentum_factor = 0.9
    
    top_p = (vol_top_p_factor + trend_top_p_factor + momentum_factor) / 3.0
    top_p = max(0.7, min(0.99, top_p))  # 限制在合理范围
    
    # ============== 使用技术指标进一步调整 ==============
    if tech_indicators is not None and not tech_indicators.empty:
        last_row = tech_indicators.iloc[-1]
        
        # RSI调整T
        if 'rsi' in last_row and not pd.isna(last_row['rsi']):
            rsi = last_row['rsi']
            if rsi > 70 or rsi < 30:
                # RSI极端值（超买/超卖）：提高T增加多样性
                T *= 1.08
            elif 40 <= rsi <= 60:
                # RSI中性：降低T更保守
                T *= 0.97
        
        # MACD调整top_p
        if 'macd_hist' in last_row and not pd.isna(last_row['macd_hist']):
            macd_hist = last_row['macd_hist']
            if abs(macd_hist) > 1.0:
                # MACD强烈信号：降低top_p更确定
                top_p *= 0.96
            elif abs(macd_hist) < 0.3:
                # MACD信号弱：提高top_p覆盖更多可能性
                top_p *= 1.02
        
        # 布林带调整
        if 'bb_width' in last_row and not pd.isna(last_row['bb_width']):
            bb_width = last_row['bb_width']
            if bb_width > 5:
                # 高波动：提高T和top_p
                T *= 1.05
                top_p *= 1.02
            elif bb_width < 2:
                # 低波动：降低T和top_p
                T *= 0.95
                top_p *= 0.98
        
        # 移动平均线趋势调整
        if 'ma_20' in last_row and 'ma_60' in last_row:
            ma20 = last_row['ma_20']
            ma60 = last_row['ma_60']
            if not (pd.isna(ma20) or pd.isna(ma60)):
                if ma20 > ma60 * 1.02:
                    # 上升趋势：适度提高T
                    T *= 1.03
                elif ma20 < ma60 * 0.98:
                    # 下降趋势：适度提高T
                    T *= 1.03
    
    # 最终限制
    T = max(0.5, min(2.0, T))
    top_p = max(0.7, min(0.99, top_p))
    
    return T, top_p


def calculate_technical_indicators(
    df: pd.DataFrame,
    include_trend: bool = True,
    include_momentum: bool = True,
    include_volatility: bool = True,
    include_volume: bool = True
) -> pd.DataFrame:
    """
    计算技术指标并添加到DataFrame。
    
    技术指标用于辅助决策，不直接输入Kronos模型（模型输入维度固定）。
    指标可用于：
    - 调整采样参数（T, top_p）
    - 调整置信度
    - 生成交易信号
    - 市场状态判断
    
    Args:
        df: OHLCV DataFrame，必须包含 'open', 'high', 'low', 'close', 'volume'
        include_trend: 是否计算趋势指标（MA, EMA, BIAS）
        include_momentum: 是否计算动量指标（RSI, MACD）
        include_volatility: 是否计算波动率指标（布林带, ATR）
        include_volume: 是否计算成交量指标（VMA, VOL_RATIO）
    
    Returns:
        包含技术指标的DataFrame（原始列 + 技术指标列）
    """
    df = df.copy()
    
    # 确保数据类型正确
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
    
    # ============== 趋势指标 ==============
    if include_trend:
        # 移动平均线（MA）
        for period in [5, 10, 20, 60]:
            if len(df) >= period:
                ma = pd.Series(closes).rolling(window=period, min_periods=1).mean()
                df[f'ma_{period}'] = ma.values
                # 乖离率（BIAS）
                df[f'bias_{period}'] = ((closes - ma.values) / (ma.values + 1e-8)) * 100
        
        # 指数移动平均（EMA）
        if len(df) >= 12:
            df['ema_12'] = pd.Series(closes).ewm(span=12, adjust=False, min_periods=1).mean().values
        if len(df) >= 26:
            df['ema_26'] = pd.Series(closes).ewm(span=26, adjust=False, min_periods=1).mean().values
    
    # ============== 动量指标 ==============
    if include_momentum:
        # RSI（相对强弱指数）
        if len(df) >= 14:
            delta = pd.Series(closes).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            df['rsi'] = (100 - (100 / (1 + rs))).values
        
        # MACD（移动平均收敛散度）
        if len(df) >= 26 and include_trend:
            if 'ema_12' in df.columns and 'ema_26' in df.columns:
                df['macd_dif'] = (df['ema_12'] - df['ema_26']).values
                df['macd_dea'] = pd.Series(df['macd_dif']).ewm(span=9, adjust=False, min_periods=1).mean().values
                df['macd_hist'] = ((df['macd_dif'] - df['macd_dea']) * 2).values
        
        # 价格变化率（ROC）
        if len(df) >= 10:
            df['roc_10'] = ((closes - pd.Series(closes).shift(10)) / (pd.Series(closes).shift(10) + 1e-8) * 100).values
    
    # ============== 波动率指标 ==============
    if include_volatility:
        # 布林带（Bollinger Bands）
        if len(df) >= 20:
            bb_period = 20
            bb_std = 2
            bb_middle = pd.Series(closes).rolling(window=bb_period, min_periods=1).mean()
            bb_std_val = pd.Series(closes).rolling(window=bb_period, min_periods=1).std()
            df['bb_middle'] = bb_middle.values
            df['bb_upper'] = (bb_middle + bb_std * bb_std_val).values
            df['bb_lower'] = (bb_middle - bb_std * bb_std_val).values
            df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / (bb_middle.values + 1e-8) * 100).values
            # 布林带位置（0-1，0=下轨，1=上轨）
            df['bb_position'] = ((closes - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)).values
        
        # ATR（平均真实波幅）
        if len(df) >= 14:
            tr1 = highs - lows
            tr2 = np.abs(highs - np.roll(closes, 1))
            tr3 = np.abs(lows - np.roll(closes, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            df['atr'] = pd.Series(tr).rolling(window=14, min_periods=1).mean().values
            # ATR百分比（相对于价格）
            df['atr_pct'] = (df['atr'] / (closes + 1e-8) * 100).values
    
    # ============== 成交量指标 ==============
    if include_volume and len(volumes) > 0:
        # 成交量移动平均
        for period in [5, 10, 20]:
            if len(df) >= period:
                vma = pd.Series(volumes).rolling(window=period, min_periods=1).mean()
                df[f'vma_{period}'] = vma.values
                # 成交量比率
                df[f'vol_ratio_{period}'] = volumes / (vma.values + 1e-8)
        
        # OBV（能量潮）
        if len(df) >= 2:
            price_change = np.diff(closes)
            volume_direction = np.where(price_change > 0, volumes[1:], 
                                       np.where(price_change < 0, -volumes[1:], 0))
            obv = np.concatenate([[volumes[0]], volumes[0] + np.cumsum(volume_direction)])
            df['obv'] = obv
    
    # 填充NaN值（使用前向填充）
    df = df.ffill().bfill()
    
    return df


def get_market_signals(tech_indicators: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """
    从技术指标中提取市场信号。
    
    Returns:
        包含各种市场信号的字典
    """
    signals = {
        'trend': 'neutral',  # 'bullish', 'bearish', 'neutral'
        'momentum': 'neutral',
        'volatility': 'normal',  # 'low', 'normal', 'high'
        'overbought_oversold': 'neutral',  # 'overbought', 'oversold', 'neutral'
        'signals': [],
        'confidence_boost': 0.0,  # 置信度提升因子
    }
    
    if tech_indicators.empty or len(tech_indicators) == 0:
        return signals
    
    last_row = tech_indicators.iloc[-1]
    
    # RSI信号
    if 'rsi' in last_row and not pd.isna(last_row['rsi']):
        rsi = last_row['rsi']
        if rsi > 70:
            signals['overbought_oversold'] = 'overbought'
            signals['signals'].append(f"RSI超买 ({rsi:.1f})")
            signals['confidence_boost'] -= 0.05
        elif rsi < 30:
            signals['overbought_oversold'] = 'oversold'
            signals['signals'].append(f"RSI超卖 ({rsi:.1f})")
            signals['confidence_boost'] -= 0.05
        elif 40 <= rsi <= 60:
            signals['confidence_boost'] += 0.03  # RSI中性时置信度提升
    
    # MACD信号
    if 'macd_hist' in last_row and not pd.isna(last_row['macd_hist']):
        macd_hist = last_row['macd_hist']
        if macd_hist > 0.5:
            signals['momentum'] = 'bullish'
            signals['signals'].append("MACD金叉 - 看涨")
            signals['confidence_boost'] += 0.02
        elif macd_hist < -0.5:
            signals['momentum'] = 'bearish'
            signals['signals'].append("MACD死叉 - 看跌")
            signals['confidence_boost'] += 0.02
    
    # 移动平均线趋势
    if 'ma_20' in last_row and not pd.isna(last_row['ma_20']):
        ma20 = last_row['ma_20']
        if current_price > ma20 * 1.02:
            signals['trend'] = 'bullish'
            signals['signals'].append("价格在20日均线上方 - 趋势向上")
        elif current_price < ma20 * 0.98:
            signals['trend'] = 'bearish'
            signals['signals'].append("价格在20日均线下方 - 趋势向下")
    
    # 布林带信号
    if 'bb_position' in last_row and not pd.isna(last_row['bb_position']):
        bb_pos = last_row['bb_position']
        if bb_pos > 0.8:
            signals['signals'].append("价格接近布林带上轨 - 可能回调")
            signals['confidence_boost'] -= 0.03
        elif bb_pos < 0.2:
            signals['signals'].append("价格接近布林带下轨 - 可能反弹")
            signals['confidence_boost'] -= 0.03
        elif 0.3 <= bb_pos <= 0.7:
            signals['confidence_boost'] += 0.02  # 价格在布林带中间，置信度提升
    
    # 波动率信号
    if 'bb_width' in last_row and not pd.isna(last_row['bb_width']):
        bb_width = last_row['bb_width']
        if bb_width > 5:
            signals['volatility'] = 'high'
            signals['signals'].append(f"高波动率 (带宽={bb_width:.2f}%)")
        elif bb_width < 2:
            signals['volatility'] = 'low'
            signals['signals'].append(f"低波动率 (带宽={bb_width:.2f}%)")
    
    # 成交量信号
    if 'vol_ratio_20' in last_row and not pd.isna(last_row['vol_ratio_20']):
        vol_ratio = last_row['vol_ratio_20']
        if vol_ratio > 1.5:
            signals['signals'].append(f"成交量放大 ({vol_ratio:.2f}x)")
            signals['confidence_boost'] += 0.01
        elif vol_ratio < 0.5:
            signals['signals'].append(f"成交量萎缩 ({vol_ratio:.2f}x)")
    
    return signals


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
        "health": "/api/health"
    }


@app.get("/api/health", response_model=HealthResponse)
async def health():
    uptime = (datetime.now(timezone.utc) - state.startup_time).total_seconds() if state.startup_time else 0
    return HealthResponse(
        status="healthy" if state.is_ready else "initializing",
        model_ready=state.is_ready,
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
    timeframe: str = Query(default="1h", pattern="^(1h|4h|1d|1w)$"),
    horizon: Optional[int] = None,
    sample_count: Optional[int] = Query(default=None, ge=10, le=100)
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

    # Calculate optimal sample_count if not provided
    optimal_sample_count = calculate_optimal_sample_count(
        horizon=horizon,
        timeframe=timeframe,
        use_case="forecast",
        user_provided=sample_count
    )
    if sample_count is None:
        sample_count = optimal_sample_count
        logger.info(f"Using optimal sample_count={sample_count} for {symbol} [{timeframe}] horizon={horizon}")
    else:
        logger.info(f"Using user-provided sample_count={sample_count} for {symbol}")

    if not state.is_ready:
        raise HTTPException(503, "Model not ready. Please wait.")

    # Calculate optimal data limit based on timeframe and horizon
    optimal_limit = calculate_optimal_data_limit(
        timeframe=timeframe,
        horizon=horizon,
        use_case="forecast"
    )
    data_limit = max(tf_config["limit"], optimal_limit)  # 至少使用配置的最小值
    logger.info(f"[{symbol}] 使用历史数据量: {data_limit} 条 (时间周期={timeframe}, 预测步数={horizon})")

    # Fetch data with appropriate timeframe
    df = await fetch_ohlcv(symbol, at, timeframe, data_limit)
    if df is None or len(df) < 30:
        raise HTTPException(400, f"Insufficient data for {symbol} ({at.value})")

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

    # Calculate historical volatility for adaptive sampling
    returns = np.log(price_df['close'].values[1:] / price_df['close'].values[:-1])
    bars_per_day = 24 // hours_per_bar
    volatility = float(np.std(returns[-bars_per_day:]) * np.sqrt(bars_per_day) * 100) if len(returns) >= bars_per_day else 0
    
    # Calculate technical indicators for enhanced decision making
    tech_indicators = calculate_technical_indicators(
        price_df,
        include_trend=True,
        include_momentum=True,
        include_volatility=True,
        include_volume=True
    )
    
    # Get market signals from technical indicators
    market_signals = get_market_signals(tech_indicators, current_price)
    if market_signals['signals']:
        logger.info(f"[{symbol}] 技术指标信号: {', '.join(market_signals['signals'][:3])}")
    
    # Calculate adaptive sampling parameters (now with technical indicators)
    T, top_p = calculate_adaptive_sampling_params(
        price_df=price_df,
        volatility=volatility,
        timeframe=timeframe,
        horizon=horizon,
        user_temperature=None,
        user_top_p=None,
        tech_indicators=tech_indicators
    )

    # Run Kronos prediction with scaled prices and adaptive parameters
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: state.predictor.predict_multi_sample(
            scaled_price_df, timestamps, y_timestamps, horizon,
            T=T, top_p=top_p, sample_count=sample_count
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
    base_confidence = float(max(0.3, min(0.95, 1.0 - float(np.mean(spread)) * 2)))
    
    # Enhance confidence using technical indicators
    confidence = base_confidence + market_signals['confidence_boost']
    confidence = float(max(0.3, min(0.95, confidence)))

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

    # Trading suggestions (enhanced with technical indicators)
    entry = smart_round(float(result['p25'][0]))
    sl = smart_round(float(result['p10'][horizon // 2]))
    tp = smart_round(float(result['p75'][horizon - 1]))
    
    # Adjust suggestions based on technical indicators
    if market_signals['trend'] == 'bullish' and market_signals['momentum'] == 'bullish':
        # 强烈看涨信号：调整入场点更激进
        entry = smart_round(entry * 0.99)
        tp = smart_round(tp * 1.02)
    elif market_signals['trend'] == 'bearish' and market_signals['momentum'] == 'bearish':
        # 强烈看跌信号：调整入场点更保守
        entry = smart_round(entry * 1.01)
        tp = smart_round(tp * 0.98)
    
    # RSI超买/超卖调整
    if market_signals['overbought_oversold'] == 'oversold':
        # 超卖：可能反弹，调整止损更紧
        sl = smart_round(sl * 0.98)
    elif market_signals['overbought_oversold'] == 'overbought':
        # 超买：可能回调，调整止盈更保守
        tp = smart_round(tp * 0.98)
    
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
    )


@app.get("/api/chart/{symbol:path}")
async def get_chart_image(
    symbol: str,
    asset_type: Optional[str] = None,
    timeframe: str = Query(default="1h", pattern="^(1h|4h|1d|1w)$"),
    horizon: Optional[int] = None,
    sample_count: Optional[int] = Query(default=None, ge=10, le=100)
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

    # Calculate optimal sample_count if not provided
    optimal_sample_count = calculate_optimal_sample_count(
        horizon=horizon,
        timeframe=timeframe,
        use_case="forecast",
        user_provided=sample_count
    )
    if sample_count is None:
        sample_count = optimal_sample_count

    if not state.is_ready:
        raise HTTPException(503, "Model not ready. Please wait.")

    # Calculate optimal data limit for chart generation
    optimal_limit = calculate_optimal_data_limit(
        timeframe=timeframe,
        horizon=horizon,
        use_case="forecast"
    )
    data_limit = max(tf_config["limit"], optimal_limit)

    # Fetch data with appropriate timeframe
    df = await fetch_ohlcv(symbol, at, timeframe, data_limit)
    if df is None or len(df) < 30:
        raise HTTPException(400, f"Insufficient data for {symbol}")

    timestamps = pd.to_datetime(df['timestamp'])
    price_df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    current_price = float(price_df['close'].iloc[-1])

    # Scale prices for small-price coins (like PEPE)
    scale_factor = get_price_scale_factor(current_price)
    scaled_price_df = scale_ohlcv_df(price_df, scale_factor)

    y_timestamps = pd.Series([timestamps.iloc[-1] + timedelta(hours=hours_per_bar * i) for i in range(1, horizon + 1)])

    # Calculate historical volatility for adaptive sampling
    returns = np.log(price_df['close'].values[1:] / price_df['close'].values[:-1])
    bars_per_day = 24 // hours_per_bar
    volatility = float(np.std(returns[-bars_per_day:]) * np.sqrt(bars_per_day) * 100) if len(returns) >= bars_per_day else 0
    
    # Calculate technical indicators for enhanced decision making
    tech_indicators = calculate_technical_indicators(
        price_df,
        include_trend=True,
        include_momentum=True,
        include_volatility=True,
        include_volume=True
    )
    
    # Get market signals from technical indicators
    market_signals = get_market_signals(tech_indicators, current_price)
    if market_signals['signals']:
        logger.info(f"[{symbol}] 技术指标信号: {', '.join(market_signals['signals'][:3])}")
    
    # Calculate adaptive sampling parameters (now with technical indicators)
    T, top_p = calculate_adaptive_sampling_params(
        price_df=price_df,
        volatility=volatility,
        timeframe=timeframe,
        horizon=horizon,
        user_temperature=None,
        user_top_p=None,
        tech_indicators=tech_indicators
    )

    # Run Kronos prediction with scaled prices and adaptive parameters
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: state.predictor.predict_multi_sample(
            scaled_price_df, timestamps, y_timestamps, horizon,
            T=T, top_p=top_p, sample_count=sample_count
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

    # Calculate optimal sample_count for simulation (需要更多样本以获得可靠的交易模拟)
    sample_count = calculate_optimal_sample_count(
        horizon=request.horizon,
        timeframe="1h",  # 模拟默认使用1h
        use_case="simulation",
        user_provided=None
    )
    logger.info(f"Using sample_count={sample_count} for simulation of {symbol}")
    
    # Run Kronos prediction with more samples for better simulation
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
    timeframe: str = Query(default="1h", pattern="^(1h|4h|1d|1w)$"),
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

    # Calculate optimal data limit for backtesting
    # Backtest needs: test_periods * horizon (for testing) + context_window (for prediction)
    base_needed = test_periods * horizon + 100
    optimal_limit = calculate_optimal_data_limit(
        timeframe=timeframe,
        horizon=horizon,
        use_case="backtest"
    )
    data_needed = max(base_needed, optimal_limit)
    logger.info(f"[{symbol}] 回测使用历史数据量: {data_needed} 条 (测试周期={test_periods}, 预测步数={horizon})")
    
    # Fetch more historical data for backtesting
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
            # Calculate optimal sample_count for backtest (平衡准确性和速度)
            backtest_sample_count = calculate_optimal_sample_count(
                horizon=horizon,
                timeframe=timeframe,
                use_case="backtest",
                user_provided=None
            )
            
            # Calculate historical volatility for adaptive sampling (使用历史数据)
            hist_returns = np.log(hist_df['close'].values[1:] / hist_df['close'].values[:-1])
            bars_per_day = 24 // hours_per_bar
            hist_volatility = float(np.std(hist_returns[-bars_per_day:]) * np.sqrt(bars_per_day) * 100) if len(hist_returns) >= bars_per_day else 0
            
            # Calculate technical indicators for backtest (使用历史数据)
            hist_tech_indicators = calculate_technical_indicators(
                hist_df,
                include_trend=True,
                include_momentum=True,
                include_volatility=True,
                include_volume=True
            )
            
            # Calculate adaptive sampling parameters for backtest (with technical indicators)
            T, top_p = calculate_adaptive_sampling_params(
                price_df=hist_df,
                volatility=hist_volatility,
                timeframe=timeframe,
                horizon=horizon,
                user_temperature=None,
                user_top_p=None,
                tech_indicators=hist_tech_indicators
            )
            
            # Run Kronos prediction with scaled data and adaptive parameters
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda h_df=hist_df, h_ts=hist_ts, y=y_ts, hor=horizon, sc=backtest_sample_count, t=T, tp=top_p: state.predictor.predict_multi_sample(
                    h_df, h_ts, y, hor,
                    T=t, top_p=tp, sample_count=sc
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
            base_confidence = float(max(0.3, min(0.95, 1.0 - float(np.mean(spread)) * 2)))
            confidences.append(base_confidence)

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

"""
Next K - Multi-Asset K-Line Weather Forecast API

A probabilistic K-line prediction tool powered by the Kronos foundation model.
Supports: Crypto, Stocks, Forex
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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


async def fetch_stock_ohlcv(symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
    """Fetch stock OHLCV data from yfinance."""
    try:
        import yfinance as yf

        def _fetch():
            ticker = yf.Ticker(symbol)
            # Use 1h interval for consistency with crypto
            df = ticker.history(period="7d", interval="1h")
            if df.empty:
                # Fallback to daily data if hourly not available
                df = ticker.history(period=f"{limit}d", interval="1d")
            return df

        df = await asyncio.get_event_loop().run_in_executor(None, _fetch)

        if df is not None and not df.empty:
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={'datetime': 'timestamp', 'date': 'timestamp'})
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(limit)
            return df

        return None
    except Exception as e:
        logger.warning(f"Failed to fetch stock OHLCV for {symbol}: {e}")
        return None


async def fetch_forex_ohlcv(symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
    """Fetch forex OHLCV data from yfinance."""
    try:
        import yfinance as yf

        # Convert forex symbol format: EUR/USD -> EURUSD=X
        yf_symbol = symbol.replace("/", "") + "=X"

        def _fetch():
            ticker = yf.Ticker(yf_symbol)
            # Use 1h interval for consistency
            df = ticker.history(period="7d", interval="1h")
            if df.empty:
                # Fallback to daily data
                df = ticker.history(period=f"{limit}d", interval="1d")
            return df

        df = await asyncio.get_event_loop().run_in_executor(None, _fetch)

        if df is not None and not df.empty:
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={'datetime': 'timestamp', 'date': 'timestamp'})
            if 'volume' not in df.columns:
                df['volume'] = 0
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(limit)
            return df

        return None
    except Exception as e:
        logger.warning(f"Failed to fetch forex OHLCV for {symbol}: {e}")
        return None


def generate_demo_ohlcv(symbol: str, asset_type: AssetType, limit: int = 100) -> pd.DataFrame:
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

    now = datetime.now(timezone.utc)
    data = []
    price = base_price

    for i in range(limit):
        timestamp = now - timedelta(hours=limit - i)
        change = np.random.randn() * volatility
        price = price * (1 + change)
        high = price * (1 + abs(np.random.randn() * volatility * 0.5))
        low = price * (1 - abs(np.random.randn() * volatility * 0.5))
        open_price = low + np.random.random() * (high - low)
        volume = np.random.randint(1000, 100000) * base_price

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
        df = await fetch_stock_ohlcv(symbol, limit)
    elif asset_type == AssetType.FOREX:
        df = await fetch_forex_ohlcv(symbol, limit)

    # Fallback to demo data if real data unavailable
    if df is None or len(df) < 50:
        logger.info(f"Using demo data for {symbol} ({asset_type.value})")
        df = generate_demo_ohlcv(symbol, asset_type, limit)

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
            "low": round(float(low), 4),
            "high": round(float(high), 4),
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
            supports.append(PriceLevel(price=round(price, 4), probability=round(prob, 1), type="support"))
    for pct in [75, 90]:
        price = float(np.percentile(samples[:, -1], pct))
        if price > current_price:
            prob = float(np.mean(samples[:, -1] < price) * 100)
            resistances.append(PriceLevel(price=round(price, 4), probability=round(prob, 1), type="resistance"))
    return supports, resistances


def run_monte_carlo(entry: float, size: float, action: str, sl: Optional[float],
                    tp: Optional[float], leverage: int, vol: float, horizon: int) -> Dict:
    is_long = action.lower() in ("buy", "long")
    hourly_vol = vol / np.sqrt(24)

    np.random.seed(int(datetime.now(timezone.utc).timestamp()) % 10000)
    returns = np.random.normal(0, hourly_vol, (1000, horizon))
    paths = entry * np.exp(np.cumsum(returns, axis=1))
    paths = np.column_stack([np.full(1000, entry), paths])

    hit_tp, hit_sl, expired = 0, 0, 0
    exit_bars, pnls = [], []

    for i in range(1000):
        path = paths[i]
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
    drawdowns = [(entry - np.min(paths[i])) / entry if is_long else (np.max(paths[i]) - entry) / entry for i in range(1000)]

    return {
        "win_rate": float(np.mean(pnls > 0) * 100),
        "expected_pnl": float(np.mean(pnls)),
        "expected_pnl_percent": float(np.mean(pnls) / size * 100),
        "max_profit": float(np.max(pnls)),
        "max_loss": float(np.min(pnls)),
        "hit_take_profit_pct": hit_tp / 10,
        "hit_stop_loss_pct": hit_sl / 10,
        "expired_pct": expired / 10,
        "avg_bars_to_exit": float(np.mean(exit_bars)),
        "max_drawdown": float(np.max(drawdowns) * 100),
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
    horizon: int = Query(default=24, ge=1, le=168),
    sample_count: int = Query(default=5, ge=3, le=50)
):
    """K-Line Weather Forecast - Probabilistic price prediction."""
    symbol = symbol.upper().replace("-", "/")

    # Detect or use provided asset type
    at = AssetType(asset_type) if asset_type else detect_asset_type(symbol)

    if not state.is_ready:
        raise HTTPException(503, "Model not ready. Please wait.")

    # Fetch data
    df = await fetch_ohlcv(symbol, at, "1h", 100)
    if df is None or len(df) < 50:
        raise HTTPException(400, f"Insufficient data for {symbol} ({at.value})")

    timestamps = pd.to_datetime(df['timestamp'])
    price_df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    current_price = float(price_df['close'].iloc[-1])
    y_timestamps = pd.Series([timestamps.iloc[-1] + timedelta(hours=i) for i in range(1, horizon + 1)])

    # Run Kronos prediction
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: state.predictor.predict_multi_sample(
            price_df, timestamps, y_timestamps, horizon,
            T=1.0, top_p=0.9, sample_count=sample_count
        )
    )

    # Calculate metrics
    close_samples = result['all_samples'][:, :, 3]
    upside_prob = float(np.mean(close_samples[:, -1] > current_price) * 100)

    returns = np.log(price_df['close'].values[1:] / price_df['close'].values[:-1])
    volatility = float(np.std(returns[-24:]) * np.sqrt(24) * 100)

    price_ranges = calculate_price_ranges(close_samples, current_price)
    supports, resistances = find_key_levels(close_samples, current_price)

    spread = (result['p90'] - result['p10']) / result['mean']
    confidence = float(max(0.3, min(0.95, 1.0 - float(np.mean(spread)) * 2)))

    # Build history data (last 72 bars = 3 days)
    history_len = min(72, len(df))
    history = []
    for i in range(len(df) - history_len, len(df)):
        ts = int(timestamps.iloc[i].timestamp())
        history.append(HistoryBar(
            time=ts,
            open=round(float(price_df['open'].iloc[i]), 4),
            high=round(float(price_df['high'].iloc[i]), 4),
            low=round(float(price_df['low'].iloc[i]), 4),
            close=round(float(price_df['close'].iloc[i]), 4),
            volume=round(float(price_df['volume'].iloc[i]), 2),
        ))

    # Build forecast data
    forecast = []
    last_ts = int(timestamps.iloc[-1].timestamp())
    for i in range(horizon):
        forecast.append(ForecastBar(
            time=last_ts + (i + 1) * 3600,
            mean=round(float(result['mean'][i]), 4),
            min=round(float(result['min'][i]), 4),
            max=round(float(result['max'][i]), 4),
            p10=round(float(result['p10'][i]), 4),
            p25=round(float(result['p25'][i]), 4),
            p75=round(float(result['p75'][i]), 4),
            p90=round(float(result['p90'][i]), 4),
        ))

    # Trading suggestions
    entry = round(float(result['p25'][0]), 4)
    sl = round(float(result['p10'][horizon // 2]), 4)
    tp = round(float(result['p75'][horizon - 1]), 4)
    risk = entry - sl
    rr = round((tp - entry) / risk, 2) if risk > 0 else None

    return WeatherForecast(
        symbol=symbol,
        asset_type=at.value,
        current_price=round(current_price, 4),
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


@app.post("/api/simulate", response_model=SimulationResult)
async def simulate_trade(request: SimulationRequest):
    """Trade Sandbox - Simulate trade 1000 times."""
    symbol = request.symbol.upper().replace("-", "/")
    at = AssetType(request.asset_type) if request.asset_type else detect_asset_type(symbol)

    df = await fetch_ohlcv(symbol, at, "1h", 50)
    if df is None or len(df) < 24:
        raise HTTPException(400, f"Cannot fetch data for {symbol}")

    current = float(df['close'].iloc[-1])
    entry = request.entry_price or current

    returns = np.log(df['close'].values[1:] / df['close'].values[:-1])
    vol = float(np.std(returns[-24:]))

    sim = run_monte_carlo(entry, request.position_size, request.action,
                          request.stop_loss, request.take_profit,
                          request.leverage, vol, request.horizon)

    rec, issues, opts = diagnose_trade(sim, request.stop_loss, request.take_profit)

    return SimulationResult(
        symbol=symbol, asset_type=at.value, action=request.action,
        entry_price=entry, position_size=request.position_size, leverage=request.leverage,
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
                    price=round(current, 4), price_change=round(price_change, 2),
                    kronos_hint=kronos_hint,
                ))

            except Exception as e:
                logger.warning(f"Radar scan failed for {symbol}: {e}")

    items.sort(key=lambda x: x.anomaly_score, reverse=True)
    return items


# ============== Main ==============

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    # Use "::" for IPv6 support (required by Railway)
    uvicorn.run(app, host="0.0.0.0", port=port)

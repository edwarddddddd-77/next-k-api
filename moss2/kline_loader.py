"""Moss2 K 线：实盘优先 ccxt/币安缓存，factory CSV 兜底。"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from moss2 import config as cfg
from moss2.config import FactoryVariant
from moss2.dataset import load_ohlcv

logger = logging.getLogger(__name__)


def load_market_df(
    symbol: str,
    variant: FactoryVariant,
    *,
    limit: Optional[int] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """纸面/实盘扫描用 K 线。"""
    bar_limit = int(limit or cfg.MOSS2_KLINE_LIMIT)
    sym = str(symbol).strip().upper()
    if not cfg.MOSS2_HL_ENABLED:
        v = cfg.MOSS2_OPS_VARIANT
    else:
        v = "en" if str(variant).lower() == "en" else "hl"

    if cfg.MOSS2_LIVE_KLINES_ENABLED:
        try:
            if v == "hl":
                from moss_quant.hyperliquid_klines import load_hyperliquid_cached

                df = load_hyperliquid_cached(
                    sym, refresh=refresh, bar_limit=bar_limit
                )
            else:
                from moss_quant.kline_cache import load_cached

                df = load_cached(sym, refresh=refresh, research=False)
            if df is not None and not df.empty:
                return df.reset_index(drop=True)
        except Exception as e:
            logger.warning("[moss2] live klines failed %s %s: %s", sym, v, e)

    return load_ohlcv(sym, v, limit=bar_limit)

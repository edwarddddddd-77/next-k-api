"""Factory 数据目录：HL hyperliquid_* / EN binanceusdm_* CSV。"""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from moss2.config import FactoryVariant, en_data_cache_dir, hl_data_cache_dir

HL_CSV_RE = re.compile(
    r"^hyperliquid_([A-Z0-9]+?)(USDC|USDT|BUSD)_([0-9a-z]+)(?:_(.+))?\.csv$",
    re.IGNORECASE,
)
EN_CSV_RE = re.compile(
    r"^binanceusdm_(.+)_([0-9a-z]+)_.+\.csv$",
    re.IGNORECASE,
)


def normalize_symbol(raw: str, *, variant: FactoryVariant) -> str:
    s = (raw or "").strip().upper().replace("/", "").replace("-", "").replace("_", "")
    if not s:
        raise ValueError("empty symbol")
    for q in ("USDC", "USDT", "BUSD"):
        if s.endswith(q) and len(s) > len(q):
            return s
    if variant == "hl":
        return f"{s}USDC"
    return f"{s}USDT"


def _hl_csv_for_base(base: str) -> Optional[Path]:
    root = hl_data_cache_dir()
    if not root.is_dir():
        return None
    base = base.upper().replace("USDC", "").replace("USDT", "")
    pattern = str(root / f"hyperliquid_{base}USDC_15m*.csv")
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def _en_legacy_csv(base: str) -> Optional[Path]:
    from moss2.config import _skills_root

    skills = _skills_root()
    if not skills:
        return None
    legacy = (
        skills
        / "moss-trade-bot-factory-en-1.0.3"
        / "scripts"
        / f"data_{base}_USDT_15m_148d.csv"
    )
    return legacy if legacy.is_file() else None


def _en_csv_for_symbol(sym: str) -> Optional[Path]:
    root = en_data_cache_dir()
    if not root.is_dir():
        return None
    base = sym.upper().replace("USDT", "")
    pattern = str(root / f"binanceusdm_{base}_USDT_USDT_15m*.csv")
    matches = sorted(glob.glob(pattern))
    if matches:
        return Path(matches[-1])
    return _en_legacy_csv(base)


def resolve_csv_path(symbol: str, variant: FactoryVariant) -> Optional[Path]:
    sym = normalize_symbol(symbol, variant=variant)
    if variant == "hl":
        return _hl_csv_for_base(sym.replace("USDC", ""))
    return _en_csv_for_symbol(sym)


def load_ohlcv(
    symbol: str,
    variant: FactoryVariant,
    *,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    path = resolve_csv_path(symbol, variant)
    if not path or not path.is_file():
        cache_root = (
            hl_data_cache_dir() if variant == "hl" else en_data_cache_dir()
        )
        raise FileNotFoundError(
            f"Moss2 {variant}: no factory CSV for {symbol} "
            f"(data_cache={cache_root})"
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if limit and len(df) > limit:
        df = df.iloc[-limit:].copy()
    return df.reset_index(drop=True)


def list_hl_catalog() -> List[Dict[str, Any]]:
    root = hl_data_cache_dir()
    out: List[Dict[str, Any]] = []
    if not root.is_dir():
        return out
    for path in sorted(root.glob("hyperliquid_*.csv")):
        m = HL_CSV_RE.match(path.name)
        if not m:
            continue
        base, quote, tf, _ver = m.groups()
        out.append(
            {
                "variant": "hl",
                "path": str(path),
                "symbol": f"{base.upper()}{quote.upper()}",
                "pair": f"{base.upper()}/{quote.upper()}",
                "timeframe": tf,
            }
        )
    return out


def list_en_catalog() -> List[Dict[str, Any]]:
    root = en_data_cache_dir()
    out: List[Dict[str, Any]] = []
    if not root.is_dir():
        return out
    for path in sorted(root.glob("binanceusdm_*.csv")):
        sym = None
        parts = path.stem.split("_")
        if len(parts) >= 2 and parts[0] == "binanceusdm":
            sym = normalize_symbol(parts[1], variant="en")
        out.append(
            {
                "variant": "en",
                "path": str(path),
                "name": path.name,
                "symbol": sym,
            }
        )
    return out

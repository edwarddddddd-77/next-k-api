"""IndicatorEdge public asset-page lookup (no official API).

Fetches https://indicatoredge.io/assets/{slug} and parses meta/signal.
Personal research proxy only — not investment advice.
"""

from __future__ import annotations

import html as html_lib
import re
import urllib.error
import urllib.request
from typing import Any

BASE = "https://indicatoredge.io"
UA = "NextK-IndicatorEdge/1.0 (research proxy; +https://indicatoredge.io/)"

_ALIASES = {
    "SOL": "sol",
    "SOLUSDT": "sol",
    "SOL-USD": "sol",
    "BTC": "btc",
    "BTCUSDT": "btc",
    "BTC-USD": "btc",
    "ETH": "eth",
    "ETHUSDT": "eth",
    "ETH-USD": "eth",
    "XRP": "xrp",
    "XRPUSDT": "xrp",
    "DOGE": "doge",
    "DOGEUSDT": "doge",
    "ADA": "ada",
    "ADAUSDT": "ada",
    "AVAX": "avax",
    "AVAXUSDT": "avax",
    "DOT": "dot",
    "DOTUSDT": "dot",
    "LTC": "ltc",
    "LTCUSDT": "ltc",
    "LINK": "link",
    "LINKUSDT": "link",
}


def slugify(symbol: str) -> str:
    raw = (symbol or "").strip()
    if not raw:
        raise ValueError("symbol_required")
    s = raw.upper().replace("/", "").replace("_", "")
    if s in _ALIASES:
        return _ALIASES[s]
    # Strip common quote suffixes: BTCUSDT → btc, ETHUSDC → eth
    for suf in ("USDT", "USDC", "USD", "PERP"):
        if len(s) > len(suf) and s.endswith(suf):
            base = s[: -len(suf)]
            if base in _ALIASES:
                return _ALIASES[base]
            return re.sub(r"[^A-Za-z0-9-]", "", base).lower()
    return re.sub(r"[^A-Za-z0-9-]", "", raw).lower()


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_meta(page: str) -> dict[str, Any] | None:
    m = re.search(
        r'<meta\s+name="description"\s+content="([^"]+)"',
        page,
        flags=re.I,
    )
    if not m:
        return None
    desc = html_lib.unescape(m.group(1))
    m2 = re.search(
        r"best indicator for (.+?) \(([^)]+)\):\s*(.+?)\s+on the\s+(\w+)\s+chart\s*"
        r"\(CAGR\s*([^,]+),\s*Sharpe\s*([^,]+),\s*([+\-]?[\d.]+%)\s*vs buy-and-hold\)",
        desc,
        flags=re.I,
    )
    if not m2:
        return None
    name, symbol, indicator, tf, cagr, sharpe, alpha = m2.groups()
    return {
        "name": name.strip(),
        "symbol": symbol.upper(),
        "indicator": indicator.strip(),
        "timeframe": tf.capitalize(),
        "cagr": cagr.strip(),
        "sharpe": sharpe.strip(),
        "vs_buy_hold": alpha.strip(),
    }


def _parse_signal(page: str) -> str | None:
    m = re.search(
        r"Signaling\s+(LONG|FLAT|SHORT)\s+right\s+now",
        page,
        flags=re.I,
    )
    return m.group(1).upper() if m else None


def lookup_best_indicator(symbol: str) -> dict[str, Any]:
    slug = slugify(symbol)
    url = f"{BASE}/assets/{slug}"
    try:
        page = _fetch(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise LookupError(f"asset_not_found:{slug}") from e
        raise RuntimeError(f"upstream_http_{e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"upstream_network:{e.reason}") from e

    parsed = _parse_meta(page)
    if not parsed:
        raise RuntimeError("parse_failed")

    parsed["signal"] = _parse_signal(page)
    parsed["slug"] = slug
    parsed["url"] = url
    parsed["source"] = "indicatoredge"
    parsed["disclaimer"] = (
        "Hypothetical backtest from IndicatorEdge — educational only, not investment advice."
    )
    return parsed

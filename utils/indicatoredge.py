"""IndicatorEdge public asset-page lookup (no official API).

Fetches https://indicatoredge.io/assets/{slug} and parses meta/signal.
Also polls /screener 「Just flipped」 for recent signal changes.
Personal research proxy only — not investment advice.
"""

from __future__ import annotations

import html as html_lib
import json
import logging
import re
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant.common.paths import resolve_data_dir

logger = logging.getLogger(__name__)

BASE = "https://indicatoredge.io"
UA = "NextK-IndicatorEdge/1.0 (research proxy; +https://indicatoredge.io/)"
FLIPS_SNAP = "indicatoredge_flips_snapshot.json"
CST = timezone(timedelta(hours=8))
_flips_lock = threading.Lock()

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
    for suf in ("USDT", "USDC", "USD", "PERP"):
        if len(s) > len(suf) and s.endswith(suf):
            base = s[: -len(suf)]
            if base in _ALIASES:
                return _ALIASES[base]
            return re.sub(r"[^A-Za-z0-9-]", "", base).lower()
    return re.sub(r"[^A-Za-z0-9-]", "", raw).lower()


def _now_cst() -> datetime:
    return datetime.now(CST)


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


def _flips_path() -> Path:
    return resolve_data_dir() / FLIPS_SNAP


def _load_flips_snap() -> dict[str, Any]:
    path = _flips_path()
    if not path.is_file():
        return {"ok": True, "flips": [], "new_flips": [], "history": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"ok": True, "flips": [], "new_flips": [], "history": []}
        data.setdefault("flips", [])
        data.setdefault("new_flips", [])
        data.setdefault("history", [])
        data["ok"] = True
        return data
    except Exception as e:
        logger.warning("ie flips snapshot read failed: %s", e)
        return {"ok": True, "flips": [], "new_flips": [], "history": [], "error": str(e)}


def _save_flips_snap(data: dict[str, Any]) -> None:
    path = _flips_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data["ok"] = True
    data["updated_at_cst"] = _now_cst().isoformat()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_flip_label(raw: str) -> tuple[str, str | None, str | None]:
    """返回 (display_name, ticker_or_None, signal_or_None)。"""
    text = html_lib.unescape(re.sub(r"\s+", " ", raw or "")).strip()
    sig = None
    m = re.search(r"(?:→|->|—|-)\s*(LONG|FLAT|SHORT)\s*$", text, flags=re.I)
    if m:
        sig = m.group(1).upper()
        text = text[: m.start()].strip()
    ticker = None
    m2 = re.search(r"\(([A-Za-z0-9.\-]+)\)\s*$", text)
    if m2:
        ticker = m2.group(1).upper()
        name = text[: m2.start()].strip() or ticker
    else:
        name = text
        if re.fullmatch(r"[A-Za-z0-9.\-]{1,12}", name):
            ticker = name.upper()
    return name, ticker, sig


def parse_screener_flips(page: str) -> dict[str, Any]:
    """从 Screener HTML 解析 Just flipped 列表。"""
    updated = None
    m = re.search(r"updated\s+(\d{4}-\d{2}-\d{2})", page or "", flags=re.I)
    if m:
        updated = m.group(1)

    pat = re.compile(
        r'<a\s+class="flip-chip\s+(to-\w+)"\s+href="(/assets/([^"]+))"\s*>(.*?)</a>',
        flags=re.I | re.S,
    )
    flips: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cls, href, slug, label in pat.findall(page or ""):
        slug_l = str(slug or "").strip().lower()
        if not slug_l or slug_l in seen:
            continue
        seen.add(slug_l)
        name, ticker, sig = _parse_flip_label(label)
        if not sig:
            tail = str(cls or "").lower().replace("to-", "")
            if tail in ("long", "flat", "short"):
                sig = tail.upper()
        flips.append(
            {
                "slug": slug_l,
                "name": name,
                "ticker": ticker,
                "signal": sig,
                "url": f"{BASE}{href}" if href.startswith("/") else f"{BASE}/assets/{slug_l}",
                "chip_class": str(cls or "").lower(),
                "key": f"{slug_l}:{sig or '?'}",
            }
        )

    counts = {"LONG": 0, "FLAT": 0, "SHORT": 0}
    for f in flips:
        s = str(f.get("signal") or "").upper()
        if s in counts:
            counts[s] += 1

    return {
        "ok": True,
        "source_updated": updated,
        "flips": flips,
        "counts": counts,
        "count": len(flips),
        "source": "indicatoredge_screener",
        "url": f"{BASE}/screener",
        "disclaimer": (
            "Just flipped from IndicatorEdge screener — hypothetical signals, not investment advice."
        ),
    }


def fetch_screener_flips() -> dict[str, Any]:
    try:
        page = _fetch(f"{BASE}/screener")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"upstream_http_{e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"upstream_network:{e.reason}") from e
    out = parse_screener_flips(page)
    out["fetched_at_cst"] = _now_cst().isoformat()
    return out


def refresh_screener_flips(*, force: bool = True) -> dict[str, Any]:
    """
    拉取 Screener Just flipped，对比上次快照标出新增翻转。
    force=False 时若已有快照则只读缓存。
    """
    with _flips_lock:
        prev = _load_flips_snap()
        if not force and prev.get("flips"):
            return {
                **prev,
                "ok": True,
                "cached": True,
                "url": f"{BASE}/screener",
                "disclaimer": (
                    "Just flipped from IndicatorEdge screener — hypothetical signals, not investment advice."
                ),
            }

        live = fetch_screener_flips()
        prev_keys = {str(x.get("key")) for x in (prev.get("flips") or []) if x.get("key")}
        cur = list(live.get("flips") or [])
        if not prev_keys:
            new_flips: list[dict[str, Any]] = []
        else:
            new_flips = [x for x in cur if str(x.get("key")) not in prev_keys]

        history = list(prev.get("history") or [])
        if new_flips:
            history.insert(
                0,
                {
                    "detected_at_cst": live.get("fetched_at_cst"),
                    "items": new_flips,
                },
            )
            history = history[:50]

        snap = {
            "ok": True,
            "cached": False,
            "flips": cur,
            "new_flips": new_flips,
            "new_count": len(new_flips),
            "counts": live.get("counts") or {},
            "count": live.get("count") or 0,
            "source_updated": live.get("source_updated"),
            "fetched_at_cst": live.get("fetched_at_cst"),
            "history": history,
            "source": "indicatoredge_screener",
            "url": live.get("url"),
            "disclaimer": live.get("disclaimer"),
        }
        _save_flips_snap(snap)
        return snap


def load_screener_flips() -> dict[str, Any]:
    """读快照；无则拉一次。"""
    with _flips_lock:
        snap = _load_flips_snap()
    if snap.get("flips"):
        snap["cached"] = True
        snap["url"] = f"{BASE}/screener"
        snap.setdefault(
            "disclaimer",
            "Just flipped from IndicatorEdge screener — hypothetical signals, not investment advice.",
        )
        return snap
    return refresh_screener_flips(force=True)

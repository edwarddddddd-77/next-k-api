"""Trading OS automation — CVDD / taker / orderbook snapshot (research only).

Pulls public Binance + BM Pro Dash CVDD, scores bottom-zone signals, persists snapshot.
Not investment advice.
"""

from __future__ import annotations

import http.cookiejar
import json
import logging
import os
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant.common.paths import resolve_data_dir

logger = logging.getLogger(__name__)

UA = "NextK-TradingOS/1.0 (research; local desk)"
SNAP = "trading_os_snapshot.json"
JOURNAL = "trading_os_journal.jsonl"
CVDD_OVERRIDE = "trading_os_cvdd_override.json"
CST = timezone(timedelta(hours=8))
_lock = threading.Lock()
_override_lock = threading.Lock()

BINANCE = "https://api.binance.com"
BITBO_CVDD = "https://charts.bitbo.io/api/v1/cvdd/"
BTC_DATA = "https://bitcoin-data.com/v1"
BMP_CVDD_PAGE = "https://www.bitcoinmagazinepro.com/charts/cvdd/"
BMP_CVDD_DASH = "https://www.bitcoinmagazinepro.com/django_plotly_dash/app/cvdd"

# Soft TTL: cached GET older than this triggers background-ish refresh on read
SNAP_TTL_SEC = int(os.getenv("TRADING_OS_SNAP_TTL_SEC", str(25 * 60)) or str(25 * 60))
JOURNAL_MAX_LINES = int(os.getenv("TRADING_OS_JOURNAL_MAX_LINES", "500") or "500")
SCORE_MAX = 4

PHASE_COPY = {
    "wait": {
        "title": "等待确认",
        "body": "底未确认。主仓轻仓或空仓，现金为王。练盯数据，不练梭哈。",
    },
    "approach": {
        "title": "接近底区",
        "body": "进入框定区间。现货分批试探，合约最多极低倍且强平价留足缓冲。",
    },
    "confirmed": {
        "title": "底确认倾向",
        "body": "多信号共振。主仓可按计划加仓；卫星仓才开始找不对称。仍需人工确认。",
    },
    "bull": {
        "title": "偏牛 / 远离底区",
        "body": "价格相对 CVDD 偏高。减卫星、锁纪律；勿当抄底日。",
    },
}


def _now_cst() -> datetime:
    return datetime.now(CST)


def _snap_path() -> Path:
    return resolve_data_dir() / SNAP


def _journal_path() -> Path:
    return resolve_data_dir() / JOURNAL


def _override_path() -> Path:
    return resolve_data_dir() / CVDD_OVERRIDE


def _valid_cvdd(val: float) -> bool:
    return 1_000.0 < float(val) <= 1_000_000.0


def load_cvdd_override() -> dict[str, Any] | None:
    path = _override_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        val = float(data.get("cvdd") or 0)
    except (TypeError, ValueError):
        return None
    if not _valid_cvdd(val):
        return None
    return {
        "cvdd": val,
        "date": str(data.get("date") or ""),
        "note": str(data.get("note") or ""),
        "set_at_cst": str(data.get("set_at_cst") or ""),
        "source_label": str(data.get("source_label") or "manual"),
    }


def set_cvdd_override(
    cvdd: float,
    *,
    date: str = "",
    note: str = "",
    source_label: str = "manual",
) -> dict[str, Any]:
    val = float(cvdd)
    if not _valid_cvdd(val):
        raise ValueError("cvdd_out_of_range")
    payload = {
        "cvdd": round(val, 2),
        "date": (date or "").strip() or _now_cst().date().isoformat(),
        "note": (note or "").strip() or "手动粘贴的 CVDD",
        "source_label": (source_label or "manual").strip() or "manual",
        "set_at_cst": _now_cst().isoformat(),
    }
    with _override_lock:
        path = _override_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def clear_cvdd_override() -> dict[str, Any]:
    with _override_lock:
        path = _override_path()
        if path.is_file():
            path.unlink()
    return {"ok": True, "cleared": True}


def _http_json(url: str, *, timeout: float = 20.0, headers: dict[str, str] | None = None) -> Any:
    h = {"User-Agent": UA, "Accept": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _series_last(path: str, value_key: str) -> dict[str, Any]:
    rows = _http_json(f"{BTC_DATA}/{path}", timeout=45.0)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{path}_empty")
    row = rows[-1]
    return {
        "date": str(row.get("d") or ""),
        "value": float(row.get(value_key) or 0),
        "source": "bitcoin-data.com",
    }


def fetch_btc_price() -> float:
    data = _http_json(f"{BINANCE}/api/v3/ticker/price?symbol=BTCUSDT")
    return float(data["price"])


def fetch_side_floors() -> dict[str, Any]:
    """Optional enrichment: realized / balanced price (not used as primary when CVDD ok)."""
    out: dict[str, Any] = {}
    try:
        rp = _series_last("realized-price", "realizedPrice")
        out["realized_price"] = rp["value"]
        out["realized_date"] = rp["date"]
    except Exception as e:
        out["realized_error"] = str(e)
    try:
        bp = _series_last("balanced-price", "balancedPrice")
        out["balanced_price"] = bp["value"]
    except Exception as e:
        out["balanced_error"] = str(e)
    return out


def parse_cvdd_from_figure(fig: dict[str, Any]) -> dict[str, Any]:
    """Extract CVDD trace from a Plotly figure dict (unit-testable)."""
    traces = fig.get("data") or []
    for tr in traces:
        name = str(tr.get("name") or "").strip().upper()
        if name != "CVDD":
            continue
        xs = tr.get("x") or []
        ys = tr.get("y") or []
        if not ys:
            continue
        val = float(ys[-1])
        if not _valid_cvdd(val):
            raise RuntimeError(f"bmp_cvdd_out_of_range:{val}")
        date = str(xs[-1])[:10] if xs else ""
        return {
            "date": date,
            "cvdd": val,
            "primary": "bmp_dash_cvdd",
            "note": "Willy Woo CVDD via BM Pro chart (free Dash endpoint)",
            "source_url": BMP_CVDD_PAGE,
        }
    raise RuntimeError("bmp_cvdd_trace_missing")


def fetch_cvdd_from_bmp() -> dict[str, Any]:
    """
    Free true CVDD via BM Pro Plotly Dash callback
    (https://www.bitcoinmagazinepro.com/charts/cvdd/).
    """
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    def _open(url: str, data: bytes | None = None, extra: dict[str, str] | None = None) -> bytes:
        h = {
            "User-Agent": UA,
            "Accept": "application/json, text/html, */*",
            "Origin": "https://www.bitcoinmagazinepro.com",
            "Referer": BMP_CVDD_PAGE,
        }
        if extra:
            h.update(extra)
        req = urllib.request.Request(url, data=data, headers=h)
        with opener.open(req, timeout=90) as resp:
            return resp.read()

    _open(BMP_CVDD_PAGE)
    payload = {
        "output": "chart.figure",
        "outputs": {"id": "chart", "property": "figure"},
        "inputs": [
            {"id": "url", "property": "pathname", "value": "/charts/cvdd/"},
            {"id": "display", "property": "children", "value": "lg 1200px"},
        ],
        "changedPropIds": ["url.pathname"],
    }
    raw = _open(
        BMP_CVDD_DASH + "/_dash-update-component",
        data=json.dumps(payload).encode("utf-8"),
        extra={"Content-Type": "application/json"},
    )
    out = json.loads(raw.decode("utf-8", errors="replace"))
    fig = ((out.get("response") or {}).get("chart") or {}).get("figure") or {}
    return parse_cvdd_from_figure(fig)


def fetch_cvdd_from_bitbo(api_key: str) -> dict[str, Any]:
    key = api_key.strip()
    # Official docs use api_key query param; also try headers for compatibility
    url = f"{BITBO_CVDD}?latest=true&api_key={urllib.parse.quote(key)}"
    try:
        raw = _http_json(url, timeout=25.0)
    except Exception:
        raw = _http_json(
            f"{BITBO_CVDD}?latest=true",
            timeout=25.0,
            headers={"Authorization": f"Bearer {key}", "X-API-Key": key},
        )
    rows = raw.get("data") if isinstance(raw, dict) else raw
    row = rows[-1] if isinstance(rows, list) else rows
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        val = float(row[1])
        if not _valid_cvdd(val):
            raise RuntimeError(f"bitbo_cvdd_out_of_range:{val}")
        return {
            "date": str(row[0])[:10],
            "cvdd": val,
            "primary": "bitbo_cvdd",
            "note": "Willy Woo CVDD via Bitbo API",
        }
    if isinstance(row, dict):
        val = float(row.get("cvdd") or 0)
        if not _valid_cvdd(val):
            raise RuntimeError(f"bitbo_cvdd_out_of_range:{val}")
        return {
            "date": str(row.get("date") or "")[:10],
            "cvdd": val,
            "primary": "bitbo_cvdd",
            "note": "Willy Woo CVDD via Bitbo API",
        }
    raise RuntimeError("bitbo_cvdd_parse_failed")


def fetch_cvdd() -> dict[str, Any]:
    """Priority: manual > BM Pro Dash > Bitbo key > realized-price proxy."""
    out: dict[str, Any] = {
        "date": "",
        "cvdd": 0.0,
        "realized_price": None,
        "balanced_price": None,
        "primary": "none",
        "note": "",
        "override": None,
    }

    override = load_cvdd_override()
    if override:
        out["cvdd"] = float(override["cvdd"])
        out["date"] = override.get("date") or ""
        out["primary"] = "manual_override"
        out["note"] = override.get("note") or "手动覆盖 CVDD"
        out["override"] = override
        return out

    try:
        bmp = fetch_cvdd_from_bmp()
        out.update(bmp)
        return out
    except Exception as e:
        out["bmp_error"] = str(e)
        logger.warning("BMP CVDD fetch failed: %s", e)

    key = (os.getenv("NEXT_K_BITBO_API_KEY") or os.getenv("BITBO_API_KEY") or "").strip()
    if key:
        try:
            bitbo = fetch_cvdd_from_bitbo(key)
            out.update(bitbo)
            return out
        except Exception as e:
            out["bitbo_error"] = str(e)
            logger.warning("Bitbo CVDD fetch failed: %s", e)

    # Last resort soft floor
    side = fetch_side_floors()
    out["realized_price"] = side.get("realized_price")
    out["balanced_price"] = side.get("balanced_price")
    if side.get("realized_error"):
        out["realized_error"] = side["realized_error"]
    if side.get("balanced_error"):
        out["balanced_error"] = side["balanced_error"]
    if out.get("realized_price"):
        out["cvdd"] = float(out["realized_price"])
        out["date"] = str(side.get("realized_date") or "")
        out["primary"] = "realized_price_proxy"
        out["note"] = "BMP/Bitbo CVDD unavailable. Using realized price as soft floor proxy."
        return out

    raise RuntimeError("floor_metrics_unavailable")


def fetch_taker_stats() -> dict[str, Any]:
    """Binance spot klines: taker buy base volume / volume (last closed bars)."""
    out: dict[str, Any] = {}
    for tf, limit in (("1h", 48), ("1d", 14)):
        url = f"{BINANCE}/api/v3/klines?symbol=BTCUSDT&interval={tf}&limit={limit}"
        bars = _http_json(url)
        ratios: list[float] = []
        volumes: list[float] = []
        for b in bars[:-1]:  # drop unfinished bar
            vol = float(b[5])
            taker = float(b[9])
            if vol <= 0:
                continue
            ratios.append(taker / vol)
            volumes.append(vol)
        if not ratios:
            continue
        recent = ratios[-6:] if tf == "1h" else ratios[-5:]
        out[tf] = {
            "taker_buy_ratio_last": round(ratios[-1], 4),
            "taker_buy_ratio_avg": round(sum(recent) / len(recent), 4),
            "volume_last": round(volumes[-1], 2),
            "bars": len(ratios),
        }
    return out


def fetch_orderbook() -> dict[str, Any]:
    depth = _http_json(f"{BINANCE}/api/v3/depth?symbol=BTCUSDT&limit=20")
    bids = depth.get("bids") or []
    asks = depth.get("asks") or []
    bid_qty = sum(float(x[1]) for x in bids)
    ask_qty = sum(float(x[1]) for x in asks)
    total = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / total if total > 0 else 0.0
    best_bid = float(bids[0][0]) if bids else 0.0
    best_ask = float(asks[0][0]) if asks else 0.0
    mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
    spread_bps = ((best_ask - best_bid) / mid * 10000) if mid else 0.0
    return {
        "bid_qty_top20": round(bid_qty, 4),
        "ask_qty_top20": round(ask_qty, 4),
        "imbalance": round(imbalance, 4),
        "spread_bps": round(spread_bps, 2),
        "best_bid": best_bid,
        "best_ask": best_ask,
    }


def fetch_funding() -> dict[str, Any]:
    """USDT-M perpetual funding (fapi). Soft-fail if blocked."""
    try:
        data = _http_json(
            "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT",
            timeout=15.0,
        )
        return {
            "last_funding_rate": float(data.get("lastFundingRate") or 0),
            "mark_price": float(data.get("markPrice") or 0),
        }
    except Exception as e:
        logger.warning("funding fetch skipped: %s", e)
        return {"last_funding_rate": None, "mark_price": None, "error": str(e)}


def _score(price: float, cvdd: float, taker: dict[str, Any], book: dict[str, Any]) -> dict[str, Any]:
    distance_pct = ((price - cvdd) / cvdd * 100.0) if cvdd > 0 else 999.0
    signals = {
        "cvdd_near": distance_pct <= 25.0,
        "cvdd_very_near": distance_pct <= 12.0,
        "taker_weak": False,
        "book_bid_heavy": float(book.get("imbalance") or 0) >= 0.08,
        "price_above_cvdd": price > cvdd,
    }
    h = taker.get("1h") or {}
    d = taker.get("1d") or {}
    if h:
        signals["taker_weak"] = float(h.get("taker_buy_ratio_last") or 1) < float(
            h.get("taker_buy_ratio_avg") or 0
        )
    if d and not signals["taker_weak"]:
        signals["taker_weak"] = float(d.get("taker_buy_ratio_last") or 1) < (
            float(d.get("taker_buy_ratio_avg") or 0) - 0.02
        )

    score = 0
    score += 2 if signals["cvdd_very_near"] else (1 if signals["cvdd_near"] else 0)
    score += 1 if signals["taker_weak"] and distance_pct <= 40 else 0
    score += 1 if signals["book_bid_heavy"] and distance_pct <= 40 else 0

    if distance_pct > 55:
        phase = "bull"
    elif score >= 4:
        phase = "confirmed"
    elif score >= 2:
        phase = "approach"
    else:
        phase = "wait"

    return {
        "score": score,
        "score_max": SCORE_MAX,
        "distance_pct": round(distance_pct, 2),
        "signals": signals,
        "phase": phase,
        "phase_meta": PHASE_COPY[phase],
    }


def build_snapshot() -> dict[str, Any]:
    errors: list[str] = []
    price = fetch_btc_price()

    try:
        cvdd_info = fetch_cvdd()
    except Exception as e:
        errors.append(f"cvdd:{e}")
        cvdd_info = {"date": "", "cvdd": 0.0, "primary": "none", "error": str(e)}

    # Enrich with side floors when primary is real CVDD (best-effort, non-fatal)
    if cvdd_info.get("primary") in ("bmp_dash_cvdd", "bitbo_cvdd", "manual_override"):
        if cvdd_info.get("realized_price") is None:
            try:
                side = fetch_side_floors()
                if side.get("realized_price") is not None:
                    cvdd_info["realized_price"] = side["realized_price"]
                if side.get("balanced_price") is not None:
                    cvdd_info["balanced_price"] = side["balanced_price"]
            except Exception as e:
                errors.append(f"side_floors:{e}")

    taker: dict[str, Any] = {}
    book: dict[str, Any] = {}
    funding: dict[str, Any] = {}
    try:
        taker = fetch_taker_stats()
    except Exception as e:
        errors.append(f"taker:{e}")
    try:
        book = fetch_orderbook()
    except Exception as e:
        errors.append(f"orderbook:{e}")
    try:
        funding = fetch_funding()
    except Exception as e:
        errors.append(f"funding:{e}")

    cvdd_val = float(cvdd_info.get("cvdd") or 0)
    scored = (
        _score(price, cvdd_val, taker, book)
        if cvdd_val > 0
        else {
            "score": 0,
            "score_max": SCORE_MAX,
            "distance_pct": None,
            "signals": {},
            "phase": "wait",
            "phase_meta": PHASE_COPY["wait"],
        }
    )

    return {
        "ok": True,
        "fetched_at_cst": _now_cst().isoformat(),
        "symbol": "BTCUSDT",
        "price": price,
        "cvdd": cvdd_info,
        "taker": taker,
        "orderbook": book,
        "funding": funding,
        "score": scored,
        "rules": {
            "satellite_pct_max": 5,
            "single_risk_pct_max": 1,
            "leverage_max": 3,
            "note": "主仓现货分批；卫星仓归零不心疼；不跟课不梭哈",
        },
        "actions": PHASE_COPY[scored["phase"]]["body"],
        "errors": errors,
        "source": {
            "price": "binance_spot",
            "cvdd": (cvdd_info.get("primary") or "unknown"),
            "taker": "binance_klines",
            "orderbook": "binance_depth",
            "funding": "binance_fapi",
        },
        "ttl_sec": SNAP_TTL_SEC,
        "disclaimer": "Automated research snapshot — not investment advice.",
    }


def _load_snap() -> dict[str, Any]:
    path = _snap_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_snap(data: dict[str, Any]) -> None:
    path = _snap_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _snap_age_sec(snap: dict[str, Any]) -> float | None:
    raw = snap.get("fetched_at_cst")
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=CST)
        return max(0.0, (_now_cst() - ts).total_seconds())
    except Exception:
        return None


def _append_journal(data: dict[str, Any]) -> None:
    path = _journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "fetched_at_cst": data.get("fetched_at_cst"),
        "price": data.get("price"),
        "cvdd": (data.get("cvdd") or {}).get("cvdd"),
        "cvdd_primary": (data.get("cvdd") or {}).get("primary"),
        "distance_pct": (data.get("score") or {}).get("distance_pct"),
        "score": (data.get("score") or {}).get("score"),
        "phase": (data.get("score") or {}).get("phase"),
        "taker_1h": ((data.get("taker") or {}).get("1h") or {}).get("taker_buy_ratio_last"),
        "book_imbalance": (data.get("orderbook") or {}).get("imbalance"),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # Trim growth
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > JOURNAL_MAX_LINES:
            path.write_text("\n".join(lines[-JOURNAL_MAX_LINES:]) + "\n", encoding="utf-8")
    except Exception as e:
        logger.warning("journal trim failed: %s", e)


def load_snapshot() -> dict[str, Any]:
    snap = _load_snap()
    if not snap.get("ok"):
        return refresh_snapshot(force=True)

    age = _snap_age_sec(snap)
    stale = age is None or age > SNAP_TTL_SEC
    if stale:
        try:
            return refresh_snapshot(force=True)
        except Exception as e:
            logger.warning("stale snapshot refresh failed, serving cache: %s", e)
            return {**snap, "cached": True, "stale": True, "age_sec": age, "refresh_error": str(e)}

    return {**snap, "cached": True, "stale": False, "age_sec": age}


def refresh_snapshot(*, force: bool = True) -> dict[str, Any]:
    # Build outside lock so concurrent GETs are not blocked for ~90s
    if not force:
        with _lock:
            prev = _load_snap()
        if prev.get("ok"):
            age = _snap_age_sec(prev)
            if age is not None and age <= SNAP_TTL_SEC:
                return {**prev, "cached": True, "stale": False, "age_sec": age}

    live = build_snapshot()

    # Desk + full-auto monitor (TG); no browser required
    try:
        from utils.trading_os_desk import load_desk_bundle, refresh_desk_bundle, run_auto_monitor

        try:
            desk = refresh_desk_bundle()
        except Exception as e:
            logger.warning("desk refresh failed, loading cache: %s", e)
            desk = load_desk_bundle()
        live["desk"] = desk
        try:
            live["monitor"] = run_auto_monitor(live)
            if live.get("desk") and isinstance(live["desk"], dict):
                live["desk"]["alerts"] = live["monitor"]
        except Exception as e:
            logger.warning("auto monitor failed: %s", e)
            live["monitor"] = {"ok": False, "error": str(e)}
    except Exception as e:
        logger.warning("desk attach failed: %s", e)
        live["desk"] = {"ok": False, "error": str(e)}

    with _lock:
        _save_snap(live)
        try:
            _append_journal(live)
        except Exception as e:
            logger.warning("journal append failed: %s", e)
    return {**live, "cached": False, "stale": False, "age_sec": 0}


def load_journal(limit: int = 30) -> dict[str, Any]:
    path = _journal_path()
    if not path.is_file():
        return {"ok": True, "items": [], "count": 0}
    lines = path.read_text(encoding="utf-8").splitlines()
    items: list[dict[str, Any]] = []
    for raw in lines[-max(1, limit) :]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            items.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    items.reverse()
    return {"ok": True, "items": items, "count": len(items)}

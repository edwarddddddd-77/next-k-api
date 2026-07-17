#!/usr/bin/env python3
"""
跨所费率 / 标记价差警报（纸面警报表，不下单）。

默认对比：Hyperliquid · Backpack（主）· Binance · OKX（辅）。
费率统一到约 8h 口径：HL / Backpack 小时费率 × 8；BN / OKX 用当期费率。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from quant.common.paths import resolve_data_dir

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))
SNAPSHOT_NAME = "xarb_board_snapshot.json"
_lock = threading.Lock()

# 默认关注的基础资产（多所有交集时才出对比）
DEFAULT_BASES = (
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "DOGE",
    "AVAX",
    "LINK",
    "ARB",
    "OP",
    "SUI",
    "APT",
    "WIF",
    "PEPE",
    "NEAR",
    "TON",
    "ADA",
    "DOT",
    "LTC",
    "UNI",
    "AAVE",
    "FIL",
    "INJ",
    "TIA",
    "SEI",
    "JUP",
    "WLD",
    "ORDI",
    "1000PEPE",
    "KMNO",
    "HYPE",
)

EXCHANGES = ("hyperliquid", "backpack", "binance", "okx")

# 主对比优先（CJ 典型 HL↔BP），其余为辅
PRIMARY_PAIRS = (("hyperliquid", "backpack"),)
SECONDARY_PAIRS = (
    ("binance", "hyperliquid"),
    ("binance", "backpack"),
    ("binance", "okx"),
    ("hyperliquid", "okx"),
    ("backpack", "okx"),
)


def _now_cst() -> datetime:
    return datetime.now(CST)


def _snap_path() -> Path:
    return resolve_data_dir() / SNAPSHOT_NAME


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "NextK-XArb/1.0", "Accept": "application/json"})
    return s


def _funding_alert_8h() -> float:
    # 绝对值差 ≥ 该阈值（费率小数，如 0.0003 = 0.03%/8h）触发
    try:
        return max(1e-6, float(os.getenv("XARB_FUNDING_ALERT_8H", "0.00025")))
    except Exception:
        return 0.00025


def _price_alert_pct() -> float:
    # 标记价差相对中间价 ≥ 该百分比触发（0.08 = 0.08%）
    try:
        return max(0.01, float(os.getenv("XARB_PRICE_ALERT_PCT", "0.08")))
    except Exception:
        return 0.08


def _bases() -> List[str]:
    raw = (os.getenv("XARB_BASES") or "").strip()
    if not raw:
        return list(DEFAULT_BASES)
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


def _fetch_binance(sess: requests.Session) -> Dict[str, Dict[str, Any]]:
    r = sess.get("https://fapi.binance.com/fapi/v1/premiumIndex", timeout=20)
    r.raise_for_status()
    out: Dict[str, Dict[str, Any]] = {}
    for row in r.json() or []:
        sym = str(row.get("symbol") or "")
        if not sym.endswith("USDT"):
            continue
        base = sym[:-4]
        # 统一 1000PEPE 等
        out[base] = {
            "exchange": "binance",
            "symbol": sym,
            "mark": float(row.get("markPrice") or 0),
            "funding_raw": float(row.get("lastFundingRate") or 0),
            "funding_8h": float(row.get("lastFundingRate") or 0),
            "funding_period": "8h",
            "next_funding_ms": int(row.get("nextFundingTime") or 0) or None,
        }
    return out


def _fetch_hyperliquid(sess: requests.Session) -> Dict[str, Dict[str, Any]]:
    r = sess.post(
        "https://api.hyperliquid.xyz/info",
        json={"type": "metaAndAssetCtxs"},
        timeout=25,
    )
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("hyperliquid_meta_unexpected")
    meta, ctxs = payload[0], payload[1]
    universe = meta.get("universe") if isinstance(meta, dict) else None
    if not isinstance(universe, list) or not isinstance(ctxs, list):
        raise ValueError("hyperliquid_universe_unexpected")
    out: Dict[str, Dict[str, Any]] = {}
    for i, u in enumerate(universe):
        if i >= len(ctxs):
            break
        name = str((u or {}).get("name") or "").upper()
        if not name or name.startswith("@"):
            continue
        ctx = ctxs[i] or {}
        fr_1h = float(ctx.get("funding") or 0)
        out[name] = {
            "exchange": "hyperliquid",
            "symbol": name,
            "mark": float(ctx.get("markPx") or 0),
            "funding_raw": fr_1h,
            "funding_8h": fr_1h * 8.0,  # 对齐 BN/OKX 约 8h 口径
            "funding_period": "1h×8",
            "next_funding_ms": None,
        }
    return out


def _fetch_backpack(sess: requests.Session) -> Dict[str, Dict[str, Any]]:
    """Backpack 公开 markPrices：含 fundingRate；fundingInterval=1h → ×8 对齐。"""
    r = sess.get("https://api.backpack.exchange/api/v1/markPrices", timeout=20)
    r.raise_for_status()
    out: Dict[str, Dict[str, Any]] = {}
    for row in r.json() or []:
        sym = str(row.get("symbol") or "")
        if not sym.endswith("_PERP"):
            continue
        # BTC_USDC_PERP → BTC
        parts = sym.split("_")
        if len(parts) < 3:
            continue
        base = parts[0].upper()
        fr_1h = float(row.get("fundingRate") or 0)
        out[base] = {
            "exchange": "backpack",
            "symbol": sym,
            "mark": float(row.get("markPrice") or 0),
            "funding_raw": fr_1h,
            "funding_8h": fr_1h * 8.0,
            "funding_period": "1h×8",
            "next_funding_ms": int(row.get("nextFundingTimestamp") or 0) or None,
        }
    return out


def _okx_inst_id(base: str) -> str:
    # OKX 部分千倍合约
    if base.startswith("1000"):
        return f"{base}-USDT-SWAP"
    return f"{base}-USDT-SWAP"


def _fetch_okx(sess: requests.Session, bases: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    def one(base: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        inst = _okx_inst_id(base)
        try:
            fr_r = sess.get(
                "https://www.okx.com/api/v5/public/funding-rate",
                params={"instId": inst},
                timeout=12,
            )
            px_r = sess.get(
                "https://www.okx.com/api/v5/public/mark-price",
                params={"instType": "SWAP", "instId": inst},
                timeout=12,
            )
            fr_j = fr_r.json()
            px_j = px_r.json()
            if str(fr_j.get("code")) != "0" or not fr_j.get("data"):
                return None
            if str(px_j.get("code")) != "0" or not px_j.get("data"):
                return None
            fr_row = fr_j["data"][0]
            px_row = px_j["data"][0]
            return base, {
                "exchange": "okx",
                "symbol": inst,
                "mark": float(px_row.get("markPx") or 0),
                "funding_raw": float(fr_row.get("fundingRate") or 0),
                "funding_8h": float(fr_row.get("fundingRate") or 0),
                "funding_period": "8h",
                "next_funding_ms": int(fr_row.get("fundingTime") or 0) or None,
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(one, b) for b in bases]
        for fut in as_completed(futs):
            hit = fut.result()
            if hit:
                out[hit[0]] = hit[1]
    return out


_EX_LABEL = {
    "binance": "币安",
    "hyperliquid": "HL",
    "backpack": "BP",
    "okx": "OKX",
}


def _pair_advice(ex_hi: str, ex_lo: str, fr_hi: float, fr_lo: float) -> str:
    # 费率高的所：多头付费 → 纸面偏空；费率低的所：偏多对冲
    diff = fr_hi - fr_lo
    return (
        f"纸面：{_EX_LABEL.get(ex_hi, ex_hi)} 开空收更高费率，"
        f"{_EX_LABEL.get(ex_lo, ex_lo)} 开多对冲"
        f"（约 8h 费率差 {diff * 100:.4f}%）"
    )


def _pair_rank(a: str, b: str) -> int:
    """越小越优先展示（主对 HL/BP = 0）。"""
    pair = tuple(sorted((a, b)))
    primary = {tuple(sorted(p)) for p in PRIMARY_PAIRS}
    if pair in primary:
        return 0
    return 1


def _build_comparisons(
    books: Dict[str, Dict[str, Dict[str, Any]]],
    bases: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    fr_th = _funding_alert_8h()
    px_th = _price_alert_pct()
    pairs = list(PRIMARY_PAIRS) + list(SECONDARY_PAIRS)
    rows: List[Dict[str, Any]] = []
    funding_alerts: List[Dict[str, Any]] = []
    price_alerts: List[Dict[str, Any]] = []

    for base in bases:
        venues = {ex: books.get(ex, {}).get(base) for ex in EXCHANGES}
        present = {ex: v for ex, v in venues.items() if v and float(v.get("mark") or 0) > 0}
        if len(present) < 2:
            continue

        for a, b in pairs:
            if a not in present or b not in present:
                continue
            va, vb = present[a], present[b]
            fr_a = float(va["funding_8h"])
            fr_b = float(vb["funding_8h"])
            fr_diff = fr_a - fr_b
            mark_a = float(va["mark"])
            mark_b = float(vb["mark"])
            mid = (mark_a + mark_b) / 2.0
            px_diff_pct = ((mark_a - mark_b) / mid * 100.0) if mid else 0.0

            hi_ex, lo_ex = (a, b) if fr_a >= fr_b else (b, a)
            hi_fr = max(fr_a, fr_b)
            lo_fr = min(fr_a, fr_b)
            advice = _pair_advice(hi_ex, lo_ex, hi_fr, lo_fr)
            primary = _pair_rank(a, b) == 0

            row = {
                "base": base,
                "pair": f"{a}/{b}",
                "ex_a": a,
                "ex_b": b,
                "primary": primary,
                "symbol_a": va.get("symbol"),
                "symbol_b": vb.get("symbol"),
                "mark_a": mark_a,
                "mark_b": mark_b,
                "funding_8h_a": fr_a,
                "funding_8h_b": fr_b,
                "funding_diff_8h": fr_diff,
                "funding_diff_8h_pct": fr_diff * 100.0,
                "price_diff_pct": px_diff_pct,
                "advice": advice,
                "funding_alert": abs(fr_diff) >= fr_th,
                "price_alert": abs(px_diff_pct) >= px_th,
            }
            rows.append(row)
            if row["funding_alert"]:
                funding_alerts.append(row)
            if row["price_alert"]:
                price_alerts.append(row)

    def _sort_key_fr(x: Dict[str, Any]) -> Tuple[int, float]:
        return (_pair_rank(x["ex_a"], x["ex_b"]), -abs(float(x["funding_diff_8h"])))

    def _sort_key_px(x: Dict[str, Any]) -> Tuple[int, float]:
        return (_pair_rank(x["ex_a"], x["ex_b"]), -abs(float(x["price_diff_pct"])))

    funding_alerts.sort(key=_sort_key_fr)
    price_alerts.sort(key=_sort_key_px)
    rows.sort(key=_sort_key_fr)
    return rows, funding_alerts, price_alerts


def build_board(*, force_refresh: bool = True) -> Dict[str, Any]:
    bases = _bases()
    sess = _session()
    errors: List[str] = []
    books: Dict[str, Dict[str, Dict[str, Any]]] = {}

    try:
        books["binance"] = _fetch_binance(sess)
    except Exception as e:
        logger.exception("xarb binance failed")
        errors.append(f"binance:{e}")
        books["binance"] = {}

    try:
        books["hyperliquid"] = _fetch_hyperliquid(sess)
    except Exception as e:
        logger.exception("xarb hyperliquid failed")
        errors.append(f"hyperliquid:{e}")
        books["hyperliquid"] = {}

    try:
        books["backpack"] = _fetch_backpack(sess)
    except Exception as e:
        logger.exception("xarb backpack failed")
        errors.append(f"backpack:{e}")
        books["backpack"] = {}

    try:
        # OKX 只拉与 HL∩BP 或 BN 有交集的 base，减少请求
        hl = set(books.get("hyperliquid") or {})
        bp = set(books.get("backpack") or {})
        bn = set(books.get("binance") or {})
        core = (hl & bp) | (bn & hl) | (bn & bp)
        want = [b for b in bases if b in core] or list(bases)[:15]
        books["okx"] = _fetch_okx(sess, want)
    except Exception as e:
        logger.exception("xarb okx failed")
        errors.append(f"okx:{e}")
        books["okx"] = {}

    rows, fr_alerts, px_alerts = _build_comparisons(books, bases)

    # 焦点：最强费率警报，否则最强价差
    focus = None
    if fr_alerts:
        top = fr_alerts[0]
        focus = {
            "kind": "funding",
            "base": top["base"],
            "pair": top["pair"],
            "stance": "费率差可做纸面对锁",
            "detail": top["advice"],
            "funding_diff_8h_pct": top["funding_diff_8h_pct"],
            "price_diff_pct": top["price_diff_pct"],
        }
    elif px_alerts:
        top = px_alerts[0]
        hi = top["ex_a"] if top["mark_a"] >= top["mark_b"] else top["ex_b"]
        lo = top["ex_b"] if hi == top["ex_a"] else top["ex_a"]
        focus = {
            "kind": "price",
            "base": top["base"],
            "pair": top["pair"],
            "stance": "标记价差偏大",
            "detail": f"纸面：{_EX_LABEL.get(lo, lo)} 偏多 / {_EX_LABEL.get(hi, hi)} 偏空，等价差收敛",
            "funding_diff_8h_pct": top["funding_diff_8h_pct"],
            "price_diff_pct": top["price_diff_pct"],
        }
    else:
        focus = {
            "kind": "quiet",
            "base": None,
            "pair": None,
            "stance": "暂无触发阈值的机会",
            "detail": "继续扫描；阈值可用 XARB_FUNDING_ALERT_8H / XARB_PRICE_ALERT_PCT 调。",
            "funding_diff_8h_pct": None,
            "price_diff_pct": None,
        }

    coverage = {
        ex: len(books.get(ex) or {}) for ex in EXCHANGES
    }

    payload = {
        "ok": True,
        "generated_at_cst": _now_cst().isoformat(),
        "exchanges": list(EXCHANGES),
        "coverage": coverage,
        "thresholds": {
            "funding_alert_8h": _funding_alert_8h(),
            "funding_alert_8h_pct": _funding_alert_8h() * 100,
            "price_alert_pct": _price_alert_pct(),
        },
        "focus": focus,
        "funding_alerts": fr_alerts[:40],
        "price_alerts": px_alerts[:40],
        "rows": rows[:120],
        "errors": errors,
        "note": "主对比 HL↔Backpack（CJ 典型）；辅对比币安/OKX。纸面：费率高所开空、低所开多。非投资建议。",
    }

    if force_refresh:
        try:
            _snap_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("xarb snapshot write failed")
    return payload


def load_snapshot() -> Dict[str, Any]:
    path = _snap_path()
    if not path.is_file():
        return {"ok": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"ok": False}
    except Exception as e:
        return {"ok": False, "error": str(e)}

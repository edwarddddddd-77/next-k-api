#!/usr/bin/env python3
"""
Alpha 早期窗口 · 历史事件回测（价量层）

能验证的：
- 上新后 5m/10m/30m/1h/4h/24h 收益（币安 U 本位 5m K 线）
- 「早期窗口偏多」是否在样本上成立

不能用免费接口回溯验证的：
- 巨鲸/空投地址当时有没有同动（需要历史持仓快照；靠上线后持续落盘做前瞻验证）

用法：
  python alpha_backtest.py
  GET /api/alpha/backtest
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from quant.common.paths import resolve_data_dir

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))
FAPI = "https://fapi.binance.com"
SNAPSHOT_NAME = "alpha_backtest_snapshot.json"
CACHE_TTL_SEC = 3600.0

# 历史 Alpha / 相关合约上新样本（时间尽量贴近官方公告；合约以期货可取 K 为准）
DEFAULT_EVENTS: List[Dict[str, Any]] = [
    {
        "symbol": "EVAA",
        "name": "EVAA Protocol",
        "coingecko_id": "evaa-protocol",
        "perp": "EVAAUSDT",
        "start_at_cst": "2025-10-03T16:00:00+08:00",  # Alpha 08:00 UTC
        "note": "Binance Alpha + 随后 Futures",
    },
    {
        "symbol": "CLO",
        "name": "Yei Finance",
        "coingecko_id": "yei-finance",
        "perp": "CLOUSDT",
        "start_at_cst": "2025-10-14T19:00:00+08:00",
        "note": "Binance Alpha CLO",
    },
    {
        "symbol": "ON",
        "name": "Orochi Network",
        "coingecko_id": "orochi-network",
        "perp": "ONUSDT",
        "start_at_cst": "2025-10-24T16:30:00+08:00",
        "note": "Binance Alpha / Futures 样本",
    },
    {
        "symbol": "ERA",
        "name": "Caldera",
        "coingecko_id": "caldera",
        "perp": "ERAUSDT",
        "start_at_cst": "2026-07-17T21:30:00+08:00",
        "note": "日历上新（若合约已开则纳入）",
    },
    {
        "symbol": "BSB",
        "name": "Block Street",
        "coingecko_id": "block-street",
        "perp": "BSBUSDT",
        "start_at_cst": "2026-07-16T19:00:00+08:00",
        "note": "空投轮次样本",
    },
]

HORIZONS_MIN = (5, 10, 30, 60, 240, 1440)


def _now_cst() -> datetime:
    return datetime.now(CST)


def _parse_iso(s: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=CST)
        return dt.astimezone(CST)
    except Exception:
        return None


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "NextK-AlphaBacktest/1.0", "Accept": "application/json"})
    return s


def _load_events() -> List[Dict[str, Any]]:
    items = [dict(x) for x in DEFAULT_EVENTS]
    raw = (os.getenv("ALPHA_BACKTEST_EVENTS_JSON") or "").strip()
    if raw:
        try:
            extra = json.loads(raw)
            if isinstance(extra, list):
                by = {(str(x.get("symbol") or "").upper(), str(x.get("start_at_cst") or "")): x for x in items}
                for row in extra:
                    if not isinstance(row, dict):
                        continue
                    key = (str(row.get("symbol") or "").upper(), str(row.get("start_at_cst") or ""))
                    by[key] = {**by.get(key, {}), **row}
                items = list(by.values())
        except Exception as e:
            logger.warning("ALPHA_BACKTEST_EVENTS_JSON parse failed: %s", e)
    items.sort(key=lambda x: str(x.get("start_at_cst") or ""))
    return items


def fetch_klines_5m(perp: str, start: datetime, end: datetime) -> List[List[Any]]:
    sess = _session()
    out: List[List[Any]] = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor < end_ms and len(out) < 2000:
        r = sess.get(
            f"{FAPI}/fapi/v1/klines",
            params={
                "symbol": perp,
                "interval": "5m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            },
            timeout=25,
        )
        if r.status_code == 400:
            # 合约不存在 / 尚未上线
            return []
        r.raise_for_status()
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break
        out.extend(batch)
        last_open = int(batch[-1][0])
        nxt = last_open + 5 * 60 * 1000
        if nxt <= cursor:
            break
        cursor = nxt
        if len(batch) < 1500:
            break
        time.sleep(0.15)
    return out


def _price_at_or_after(klines: List[List[Any]], t: datetime) -> Optional[Tuple[datetime, float]]:
    t_ms = int(t.timestamp() * 1000)
    for k in klines:
        open_ms = int(k[0])
        if open_ms >= t_ms:
            return datetime.fromtimestamp(open_ms / 1000, tz=CST), float(k[1])  # open
    return None


def _price_at_or_before(klines: List[List[Any]], t: datetime) -> Optional[Tuple[datetime, float]]:
    t_ms = int(t.timestamp() * 1000)
    best = None
    for k in klines:
        open_ms = int(k[0])
        if open_ms <= t_ms:
            best = (datetime.fromtimestamp(open_ms / 1000, tz=CST), float(k[4]))  # close
        else:
            break
    return best


def _max_drawdown(klines: List[List[Any]], t0: datetime, minutes: int, px0: float) -> Optional[float]:
    if px0 <= 0:
        return None
    end = t0 + timedelta(minutes=minutes)
    t0_ms = int(t0.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    peak = px0
    max_dd = 0.0
    for k in klines:
        open_ms = int(k[0])
        if open_ms < t0_ms:
            continue
        if open_ms > end_ms:
            break
        high = float(k[2])
        low = float(k[3])
        peak = max(peak, high)
        dd = (low - peak) / peak * 100.0
        max_dd = min(max_dd, dd)
    return round(max_dd, 3)


def evaluate_event(event: Dict[str, Any]) -> Dict[str, Any]:
    sym = str(event.get("symbol") or "").upper()
    perp = str(event.get("perp") or f"{sym}USDT").upper()
    t0 = _parse_iso(str(event.get("start_at_cst") or ""))
    if t0 is None:
        return {**event, "ok": False, "error": "bad_start_at"}

    if t0 > _now_cst() + timedelta(minutes=5):
        return {
            **event,
            "ok": False,
            "error": "not_started",
            "message": "尚未开盘，无法回测",
        }

    start = t0 - timedelta(minutes=30)
    end = t0 + timedelta(hours=26)
    try:
        klines = fetch_klines_5m(perp, start, end)
    except Exception as e:
        return {**event, "ok": False, "error": f"klines:{e}"}

    if not klines:
        return {
            **event,
            "ok": False,
            "error": "no_klines",
            "message": f"{perp} 无 5m K 线（合约可能未上或代码不对）",
        }

    anchor = _price_at_or_after(klines, t0)
    if not anchor:
        return {**event, "ok": False, "error": "no_anchor_price"}

    t_anchor, px0 = anchor
    rets: Dict[str, Optional[float]] = {}
    for m in HORIZONS_MIN:
        tgt = t_anchor + timedelta(minutes=m)
        hit = _price_at_or_before(klines, tgt)
        if hit and px0 > 0:
            rets[f"ret_{m}m"] = round((hit[1] / px0 - 1.0) * 100.0, 3)
        else:
            rets[f"ret_{m}m"] = None

    dd30 = _max_drawdown(klines, t_anchor, 30, px0)

    # 早期窗口启发式判定（价量层，非筹码层）
    r10 = rets.get("ret_10m")
    r60 = rets.get("ret_60m")
    early_ok = r10 is not None and r10 > 0
    hour_ok = r60 is not None and r60 > 0

    return {
        "ok": True,
        "symbol": sym,
        "name": event.get("name"),
        "perp": perp,
        "coingecko_id": event.get("coingecko_id"),
        "start_at_cst": event.get("start_at_cst"),
        "note": event.get("note"),
        "anchor_at_cst": t_anchor.isoformat(),
        "anchor_price": px0,
        "n_bars": len(klines),
        **rets,
        "max_dd_30m_pct": dd30,
        "early_window_green": early_ok,  # 10m > 0
        "first_hour_green": hour_ok,
        "verdict": (
            "早期窗口上涨"
            if early_ok
            else ("首小时仍红" if (r60 is not None and r60 <= 0) else "早期偏弱/数据不足")
        ),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [r for r in rows if r.get("ok")]
    n = len(ok_rows)
    if n == 0:
        return {
            "n_events": 0,
            "n_ok": 0,
            "early_window_hit_rate": None,
            "first_hour_hit_rate": None,
            "avg_ret_10m": None,
            "avg_ret_60m": None,
            "avg_ret_1440m": None,
        }

    def _avg(key: str) -> Optional[float]:
        vals = [float(r[key]) for r in ok_rows if r.get(key) is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 3)

    early_hits = sum(1 for r in ok_rows if r.get("early_window_green"))
    hour_hits = sum(1 for r in ok_rows if r.get("first_hour_green"))
    return {
        "n_events": len(rows),
        "n_ok": n,
        "early_window_hit_rate": round(early_hits / n * 100.0, 1),
        "first_hour_hit_rate": round(hour_hits / n * 100.0, 1),
        "avg_ret_10m": _avg("ret_10m"),
        "avg_ret_60m": _avg("ret_60m"),
        "avg_ret_240m": _avg("ret_240m"),
        "avg_ret_1440m": _avg("ret_1440m"),
        "thesis": "原文『早期买盘主导』的价量层验证：看上新后 10m/1h 上涨占比",
        "limitation": "无法用免费接口回溯当时巨鲸/空投地址变动；筹码信号需上线后持续落盘做前瞻验证",
    }


def run_backtest(force_refresh: bool = False) -> Dict[str, Any]:
    path = resolve_data_dir() / SNAPSHOT_NAME
    if not force_refresh and path.is_file():
        try:
            age = time.time() - path.stat().st_mtime
            if age < CACHE_TTL_SEC:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("ok"):
                    data["snapshot_source"] = "disk_cache"
                    data["cache_age_sec"] = round(age, 1)
                    return data
        except Exception as e:
            logger.warning("backtest cache read failed: %s", e)

    events = _load_events()
    rows: List[Dict[str, Any]] = []
    for ev in events:
        rows.append(evaluate_event(ev))
        time.sleep(0.2)

    summary = summarize(rows)
    payload = {
        "ok": True,
        "generated_at_cst": _now_cst().isoformat(),
        "source": "binance_futures_5m",
        "summary": summary,
        "events": rows,
        "snapshot_source": "live",
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("backtest snapshot write failed: %s", e)
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = run_backtest(force_refresh=True)
    print(json.dumps({"summary": out.get("summary"), "events": [
        {k: e.get(k) for k in ("symbol", "ok", "perp", "ret_10m", "ret_60m", "ret_1440m", "verdict", "error", "message")}
        for e in (out.get("events") or [])
    ]}, ensure_ascii=False, indent=2))

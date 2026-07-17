#!/usr/bin/env python3
"""
币安 Alpha 筹码策略雷达

逻辑（CJ / 链上筹码启发）：
1. Alpha 开盘早于空投 → 早期窗口卖盘薄、买盘主导（约前 5–10 分钟）
2. 多类巨鲸同时动 → 流通放大 → 抛压（偏空）
3. 仅 Alpha/早期地址动 → 流通受限 → 偏多
4. 第二类（空投地址）开始放量流出 → 抛压确认，离场
5. 看筹码分布 + 解锁时间线；并非所有币适用

数据源：
- CoinGecko category=binance-alpha-spotlight（公开市场）
- 本地日历种子（上新 / 空投时间窗）
- 多链 Top 持仓：alpha_holders（ETH/BSC/Base/Arbitrum/Solana）
- 市场价量仍作辅助；主信号以地址余额变动为准
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
from alpha_coingecko import alpha_providers_status, coingecko_base_url, coingecko_session
from alpha_playbook import STRATEGY_RULES

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))
ALPHA_CATEGORY = "binance-alpha-spotlight"
SNAPSHOT_NAME = "alpha_board_snapshot.json"
CACHE_TTL_SEC = 90.0

# 地址类型（标签语义；可用 ALPHA_ADDRESS_LABELS_JSON 覆盖）
ADDRESS_TYPES = (
    {"id": "alpha", "label": "Alpha / 早期", "bias": "restricted_float"},
    {"id": "airdrop", "label": "空投地址", "bias": "sell_pressure"},
    {"id": "exchange", "label": "交易所", "bias": "listing_flow"},
    {"id": "mm", "label": "做市商", "bias": "liquidity"},
    {"id": "whale", "label": "巨鲸", "bias": "watch"},
    {"id": "burn", "label": "销毁地址", "bias": "ignore"},
)

# 上新 / 空投日历（可被环境变量 ALPHA_CALENDAR_JSON 覆盖合并）
DEFAULT_CALENDAR: List[Dict[str, Any]] = [
    {
        "symbol": "ERA",
        "name": "Caldera",
        "coingecko_id": "caldera",
        "event": "alpha_list",
        "start_at_cst": "2026-07-17T21:30:00+08:00",
        "note": "币安 Alpha 上线交易 / 空投领取",
        "points_threshold": None,
        # 写死合约，避免 CoinGecko 429 导致持仓监控失败
        "contracts": {
            "ethereum": "0xE2AD0BF751834f2fbdC62A41014f84d67cA1de2A",
            "binance-smart-chain": "0x00312400303d02c323295f6E8b7309bc30FB6BcE",
        },
    },
    {
        "symbol": "BSB",
        "name": "Block Street",
        "coingecko_id": "block-street",
        "event": "airdrop_round",
        "start_at_cst": "2026-07-16T19:00:00+08:00",
        "note": "Alpha 第二轮空投（约 250 积分门槛）",
        "points_threshold": 250,
        "contracts": {
            "ethereum": "0xdb6ba5d510f114f9b2ea08bea7d30e32eee33411",
        },
    },
    {
        "symbol": "ASP",
        "name": "Aspecta",
        "coingecko_id": "aspecta",
        "event": "alpha_list",
        "start_at_cst": "2026-07-24T00:00:00+08:00",
        "note": "预定上线币安 Alpha（具体时刻待官方确认）",
        "points_threshold": None,
        "contracts": {
            "binance-smart-chain": "0xad8c787992428cd158e451aab109f724b6bc36de",
        },
    },
]


def _snapshot_path() -> Path:
    return resolve_data_dir() / SNAPSHOT_NAME


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


def _load_calendar() -> List[Dict[str, Any]]:
    items = [dict(x) for x in DEFAULT_CALENDAR]
    raw = (os.getenv("ALPHA_CALENDAR_JSON") or "").strip()
    if raw:
        try:
            extra = json.loads(raw)
            if isinstance(extra, list):
                by_key = {
                    (str(x.get("symbol") or "").upper(), str(x.get("start_at_cst") or "")): x
                    for x in items
                }
                for row in extra:
                    if not isinstance(row, dict):
                        continue
                    key = (
                        str(row.get("symbol") or "").upper(),
                        str(row.get("start_at_cst") or ""),
                    )
                    by_key[key] = {**by_key.get(key, {}), **row}
                items = list(by_key.values())
        except Exception as e:
            logger.warning("ALPHA_CALENDAR_JSON parse failed: %s", e)
    items.sort(key=lambda x: str(x.get("start_at_cst") or ""))
    return items


def _session() -> requests.Session:
    return coingecko_session()


def fetch_alpha_markets(limit: int = 40) -> List[Dict[str, Any]]:
    """CoinGecko Binance Alpha Spotlight 市场列表。"""
    sess = _session()
    r = sess.get(
        f"{coingecko_base_url()}/coins/markets",
        params={
            "vs_currency": "usd",
            "category": ALPHA_CATEGORY,
            "order": "volume_desc",
            "per_page": max(1, min(int(limit), 100)),
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "1h,24h",
        },
        timeout=25,
    )
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise ValueError(f"unexpected coingecko payload: {type(data)}")
    return data


def fetch_focus_markets(ids: List[str]) -> List[Dict[str, Any]]:
    clean = [x.strip() for x in ids if x and str(x).strip()]
    if not clean:
        return []
    sess = _session()
    r = sess.get(
        f"{coingecko_base_url()}/coins/markets",
        params={
            "vs_currency": "usd",
            "ids": ",".join(clean[:40]),
            "sparkline": "false",
            "price_change_percentage": "1h,24h",
        },
        timeout=25,
    )
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def score_market_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    用价量结构近似筹码博弈信号（非链上标签）。
    - sell_pressure: 高位回落 + 放量下跌
    - float_tight: 上涨且成交/市值偏低（流通受限代理）
    - early_heat: 1h 波动大（开盘窗口代理）
    """
    price = _safe_float(row.get("current_price"))
    high = _safe_float(row.get("high_24h"), price)
    vol = _safe_float(row.get("total_volume"))
    mcap = _safe_float(row.get("market_cap"))
    chg_1h = _safe_float(row.get("price_change_percentage_1h_in_currency"))
    chg_24h = _safe_float(
        row.get("price_change_percentage_24h_in_currency")
        or row.get("price_change_percentage_24h")
    )

    drop_from_high = 0.0
    if high > 0 and price > 0:
        drop_from_high = max(0.0, (high - price) / high * 100.0)

    turnover = (vol / mcap * 100.0) if mcap > 0 else 0.0

    sell = 0.0
    if chg_1h < 0:
        sell += min(35.0, abs(chg_1h) * 4.0)
    if chg_24h < 0:
        sell += min(25.0, abs(chg_24h) * 1.2)
    sell += min(25.0, drop_from_high * 1.5)
    if turnover > 25:
        sell += min(15.0, (turnover - 25) * 0.35)
    sell = round(min(100.0, sell), 1)

    float_tight = 0.0
    if chg_1h > 0:
        float_tight += min(40.0, chg_1h * 5.0)
    if chg_24h > 0:
        float_tight += min(25.0, chg_24h * 1.5)
    if 0 < turnover < 12:
        float_tight += 20.0
    elif 12 <= turnover < 25:
        float_tight += 10.0
    if drop_from_high < 3:
        float_tight += 10.0
    float_tight = round(min(100.0, float_tight), 1)

    early_heat = round(min(100.0, abs(chg_1h) * 8.0 + min(30.0, turnover * 0.6)), 1)

    if sell >= 55 and sell >= float_tight + 12:
        bias = "bearish"
        bias_label = "抛压偏强"
        action = "减仓 / 观望空投流出"
    elif float_tight >= 55 and float_tight >= sell + 12:
        bias = "bullish"
        bias_label = "流通受限偏多"
        action = "波段做多 / 关注早期窗口"
    elif early_heat >= 50 and abs(chg_1h) >= 3:
        bias = "volatile"
        bias_label = "窗口波动"
        action = "短线波段，严控止损"
    else:
        bias = "neutral"
        bias_label = "中性"
        action = "等筹码结构更清晰"

    return {
        "sell_pressure": sell,
        "float_tight": float_tight,
        "early_heat": early_heat,
        "turnover_pct": round(turnover, 2),
        "drop_from_high_pct": round(drop_from_high, 2),
        "chg_1h": round(chg_1h, 2),
        "chg_24h": round(chg_24h, 2),
        "bias": bias,
        "bias_label": bias_label,
        "action": action,
    }


def _enrich_market(row: Dict[str, Any]) -> Dict[str, Any]:
    scores = score_market_row(row)
    return {
        "id": row.get("id"),
        "symbol": str(row.get("symbol") or "").upper(),
        "name": row.get("name"),
        "image": row.get("image"),
        "price": row.get("current_price"),
        "market_cap": row.get("market_cap"),
        "volume_24h": row.get("total_volume"),
        "ath_change_pct": row.get("ath_change_percentage"),
        "last_updated": row.get("last_updated"),
        **scores,
    }


def enrich_calendar(
    calendar: List[Dict[str, Any]],
    markets_by_id: Dict[str, Dict[str, Any]],
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    now = now or _now_cst()
    out: List[Dict[str, Any]] = []
    for item in calendar:
        start = _parse_iso(str(item.get("start_at_cst") or ""))
        minutes_to = None
        phase = "scheduled"
        if start is not None:
            delta_min = (start - now).total_seconds() / 60.0
            minutes_to = round(delta_min, 1)
            if -10 <= delta_min <= 0:
                phase = "early_window"
            elif 0 < delta_min <= 24 * 60:
                phase = "upcoming"
            elif delta_min < -10:
                phase = "live"
            else:
                phase = "scheduled"

        cg = str(item.get("coingecko_id") or "")
        mkt = markets_by_id.get(cg)
        row = {
            **item,
            "phase": phase,
            "minutes_to_start": minutes_to,
            "market": mkt,
        }
        if phase == "early_window":
            row["window_hint"] = "当前处于开盘早期窗口（约 5–10 分钟逻辑）"
        elif phase == "upcoming" and minutes_to is not None and minutes_to <= 180:
            row["window_hint"] = "临近开盘，准备盯盘筹码与卖压"
        elif phase == "live":
            row["window_hint"] = "已开盘：重点看空投地址是否开始放量流出"
        else:
            row["window_hint"] = "日程已排，等待官方精确时刻"
        out.append(row)
    return out


def pick_focus(calendar_enriched: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """优先：early_window > 今日即将上新 > 已开盘最近事件 > 下一档上新。"""
    if not calendar_enriched:
        return None
    for phase in ("early_window", "upcoming", "live", "scheduled"):
        cands = [x for x in calendar_enriched if x.get("phase") == phase]
        if not cands:
            continue
        if phase in ("upcoming", "scheduled"):
            cands = sorted(
                cands,
                key=lambda x: (
                    x.get("minutes_to_start") is None,
                    x.get("minutes_to_start") if x.get("minutes_to_start") is not None else 1e18,
                ),
            )
        else:
            cands = sorted(
                cands,
                key=lambda x: abs(x.get("minutes_to_start") or 0),
            )
        return cands[0]
    return calendar_enriched[0]


def attach_chip_watch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """把最新链上持仓监控挂到看板 payload（读盘/缓存时也要刷新，避免焦点信号过期）。"""
    if not isinstance(payload, dict) or not payload.get("ok"):
        return payload
    try:
        from alpha_holders import load_watch_snapshot

        chip = load_watch_snapshot()
        if not chip.get("ok"):
            return payload
        payload = dict(payload)
        payload["chip_watch"] = chip
        focus = payload.get("focus")
        if isinstance(focus, dict) and isinstance(chip.get("watches"), list):
            fid = str(focus.get("coingecko_id") or "")
            for w in chip["watches"]:
                if str(w.get("coingecko_id") or "") == fid:
                    payload["focus"] = {**focus, "chip": w}
                    break
    except Exception as e:
        logger.warning("attach chip_watch failed: %s", e)
    return payload


def patch_board_snapshot_chip_watch() -> None:
    """持仓刷新后，就地更新看板快照里的 chip_watch / focus.chip。"""
    path = _snapshot_path()
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data.get("ok"):
            return
        data = attach_chip_watch(data)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("patch board chip_watch failed: %s", e)


def build_board(limit: int = 40, force_refresh: bool = False) -> Dict[str, Any]:
    path = _snapshot_path()
    if not force_refresh and path.is_file():
        try:
            age = time.time() - path.stat().st_mtime
            if age < CACHE_TTL_SEC:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and cached.get("ok"):
                    cached["snapshot_source"] = "disk_cache"
                    cached["cache_age_sec"] = round(age, 1)
                    return attach_chip_watch(cached)
        except Exception as e:
            logger.warning("alpha board cache read failed: %s", e)

    calendar = _load_calendar()
    # 原文 = 筹码监控；默认不再请求 CoinGecko 行情（易 429，且非主信号）
    include_markets = str(os.getenv("ALPHA_BOARD_INCLUDE_MARKETS") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    markets_raw: List[Dict[str, Any]] = []
    focus_raw: List[Dict[str, Any]] = []
    errors: List[str] = []
    by_id: Dict[str, Dict[str, Any]] = {}

    if include_markets:
        focus_ids = [
            str(x.get("coingecko_id") or "").strip()
            for x in calendar
            if str(x.get("coingecko_id") or "").strip()
        ]
        try:
            markets_raw = fetch_alpha_markets(limit=limit)
        except Exception as e:
            logger.warning("fetch_alpha_markets failed: %s", e)
            errors.append(f"markets:{e}")
        try:
            focus_raw = fetch_focus_markets(focus_ids)
        except Exception as e:
            logger.warning("fetch_focus_markets failed: %s", e)
            errors.append(f"focus:{e}")
        for row in markets_raw + focus_raw:
            cid = str(row.get("id") or "")
            if cid:
                by_id[cid] = _enrich_market(row)

    board_rows = sorted(
        list(by_id.values()),
        key=lambda x: (
            0 if x.get("bias") == "bearish" else 1 if x.get("bias") == "bullish" else 2,
            -_safe_float(x.get("sell_pressure")),
            -_safe_float(x.get("volume_24h")),
        ),
    )

    calendar_enriched = enrich_calendar(calendar, by_id)
    focus = pick_focus(calendar_enriched)

    sell_rank = sorted(board_rows, key=lambda x: -_safe_float(x.get("sell_pressure")))[:8]
    tight_rank = sorted(board_rows, key=lambda x: -_safe_float(x.get("float_tight")))[:8]

    now = _now_cst()
    payload: Dict[str, Any] = {
        "ok": True,
        "generated_at_cst": now.isoformat(),
        "source": "calendar+chip_watch" if not include_markets else "coingecko:binance-alpha-spotlight",
        "markets_enabled": include_markets,
        "strategy": {
            "name": "Alpha 筹码窗口",
            "summary": "用 Alpha 早期上市 + 筹码地址类别，判断短线抛压与流通受限。",
            "rules": STRATEGY_RULES,
            "address_types": list(ADDRESS_TYPES),
            "disclaimer": "对齐 CJ 原文：盯头部地址余额变化；多鲸同动偏空、仅 Alpha 交易偏多、空投第二类抛压大就跑。地址请用 ALPHA_ADDRESS_LABELS_JSON 标成 alpha/airdrop/mm。非投资建议。行情非原文主信号，默认关闭。",
            "source_logic": "gch_enbsbxbs / Binance Alpha 筹码窗口",
            "chains": [
                "ethereum",
                "binance-smart-chain",
                "base",
                "arbitrum-one",
                "solana",
            ],
        },
        "focus": focus,
        "calendar": calendar_enriched,
        "board": board_rows[:limit],
        "rank_sell_pressure": sell_rank,
        "rank_float_tight": tight_rank,
        "chip_watch": None,
        "providers": alpha_providers_status(),
        "counts": {
            "board": len(board_rows),
            "calendar": len(calendar_enriched),
            "bearish": sum(1 for x in board_rows if x.get("bias") == "bearish"),
            "bullish": sum(1 for x in board_rows if x.get("bias") == "bullish"),
            "volatile": sum(1 for x in board_rows if x.get("bias") == "volatile"),
            "neutral": sum(1 for x in board_rows if x.get("bias") == "neutral"),
        },
        "errors": errors,
        "snapshot_source": "live",
    }

    payload = attach_chip_watch(payload)
    if payload.get("chip_watch") is None and errors:
        # attach 失败已记日志；保留 errors
        pass

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("alpha board snapshot write failed: %s", e)

    return payload


def load_snapshot() -> Dict[str, Any]:
    path = _snapshot_path()
    if not path.is_file():
        return {
            "ok": False,
            "error": "no_snapshot",
            "message": "尚无 Alpha 看板快照，请点击刷新生成。",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("snapshot root must be object")
        data["snapshot_source"] = "disk"
        return attach_chip_watch(data)
    except Exception as e:
        logger.warning("alpha snapshot corrupt: %s", e)
        return {"ok": False, "error": "snapshot_corrupt", "message": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = build_board(force_refresh=True)
    print(json.dumps({"ok": out.get("ok"), "focus": out.get("focus"), "counts": out.get("counts")}, ensure_ascii=False, indent=2))

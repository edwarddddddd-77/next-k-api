#!/usr/bin/env python3
"""
期货版 Skew 中性策略 — 借鉴期权 Skew / 波动率曲面思路，用币安永续数据实现。

核心逻辑：
1. 截面 Skew 评分：OI 变化、资金费率、价格-OI 背离 → z-score 合成
2. 三类微观信号：轧空做多 / 隐蔽建仓 / 弱势做空；否决空头平仓假涨
3. 宏观 Vol Regime：BTC 暴跌或去杠杆 → 防御模式，暂停新开配对
4. 配对中性：做多 Skew 最强 leg + 做空 Skew 最弱 leg（Beta 近似中性）

数据源：币安合约公开 API（与 accumulation_radar 同源）
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from accumulation_radar import FAPI, api_get, init_db

# === 配置（可通过环境变量覆盖）===
db_dir = os.getenv("DATA_DIR", str(Path(__file__).parent))
SKEW_SNAPSHOT_PATH = Path(db_dir) / "skew_strategy_snapshot.json"
SKEW_RETENTION_DAYS = max(1, int(os.getenv("SKEW_RETENTION_DAYS", "7").strip() or "7"))
SKEW_UNIVERSE_TOP_N = max(20, int(os.getenv("SKEW_UNIVERSE_TOP_N", "80").strip() or "80"))
SKEW_MIN_VOL_USD = float(os.getenv("SKEW_MIN_VOL_USD", "2000000").strip() or "2000000")
SKEW_PAIR_ENABLED = os.getenv("SKEW_PAIR_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
SKEW_RECORD_SIGNALS = os.getenv("SKEW_RECORD_SIGNALS", "1").strip().lower() in ("1", "true", "yes", "on")

# 微观信号阈值（z-score）
SQUEEZE_FR_Z_MAX = float(os.getenv("SKEW_SQUEEZE_FR_Z_MAX", "-1.0"))
SQUEEZE_OI_Z_MIN = float(os.getenv("SKEW_SQUEEZE_OI_Z_MIN", "0.5"))
SQUEEZE_PX_CHG_MAX = float(os.getenv("SKEW_SQUEEZE_PX_CHG_MAX", "15.0"))

DARK_OI_Z_MIN = float(os.getenv("SKEW_DARK_OI_Z_MIN", "1.0"))
DARK_FR_Z_MIN = float(os.getenv("SKEW_DARK_FR_Z_MIN", "0.0"))
DARK_PX_ABS_MAX = float(os.getenv("SKEW_DARK_PX_ABS_MAX", "3.0"))

WEAK_OI_Z_MAX = float(os.getenv("SKEW_WEAK_OI_Z_MAX", "-0.5"))
WEAK_FR_Z_MIN = float(os.getenv("SKEW_WEAK_FR_Z_MIN", "1.0"))

COVER_TRAP_PX_MIN = float(os.getenv("SKEW_COVER_TRAP_PX_MIN", "3.0"))
COVER_TRAP_OI_Z_MAX = float(os.getenv("SKEW_COVER_TRAP_OI_Z_MAX", "-1.0"))

# 宏观 Vol Regime
REGIME_BTC_DROP_PCT = float(os.getenv("SKEW_REGIME_BTC_DROP_PCT", "8.0"))
REGIME_BTC_FR_Z_MAX = float(os.getenv("SKEW_REGIME_BTC_FR_Z_MAX", "-2.0"))
REGIME_BTC_OI_Z_MAX = float(os.getenv("SKEW_REGIME_BTC_OI_Z_MAX", "-1.5"))

SIGNAL_TYPE_ZH: Dict[str, str] = {
    "squeeze_long": "轧空 Skew 做多",
    "dark_long": "隐蔽建仓做多",
    "weak_short": "弱势 Skew 做空",
    "cover_trap": "空头平仓陷阱（否决）",
    "neutral": "中性观望",
}


@dataclass(frozen=True)
class SkewConfig:
    """Skew 策略阈值；回测/研究可覆写，实盘默认 from_env()。"""

    squeeze_fr_z_max: float = -1.0
    squeeze_oi_z_min: float = 0.5
    squeeze_px_chg_max: float = 15.0
    dark_oi_z_min: float = 1.0
    dark_fr_z_min: float = 0.0
    dark_px_abs_max: float = 3.0
    weak_oi_z_max: float = -0.5
    weak_fr_z_min: float = 1.0
    cover_trap_px_min: float = 3.0
    cover_trap_oi_z_max: float = -1.0
    regime_btc_drop_pct: float = 5.0
    regime_btc_fr_z_max: float = -2.0
    regime_btc_oi_z_max: float = -1.5
    regime_cautious_drop_pct: float = 4.0
    regime_cautious_fr_z_max: float = -1.5
    enable_dark_long: bool = False
    long_signal_types: Tuple[str, ...] = ("squeeze_long",)
    pair_enabled: bool = True
    pair_risk_pct: float = 0.25
    pair_leverage: float = 2.0
    suggested_hold_hours: int = 48
    min_skew_gap: float = 1.0

    @classmethod
    def from_env(cls) -> "SkewConfig":
        def _f(name: str, default: float) -> float:
            return float(os.getenv(name, str(default)).strip() or default)

        profile = os.getenv("SKEW_PROFILE", "aggressive").strip().lower()
        if profile == "balanced":
            risk_default, lev_default, gap_default = 0.10, 1.0, 0.0
        else:
            risk_default, lev_default, gap_default = 0.25, 2.0, 1.0

        enable_dark = os.getenv("SKEW_ENABLE_DARK_LONG", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        long_types: Tuple[str, ...] = ("squeeze_long", "dark_long") if enable_dark else ("squeeze_long",)
        hold_h = int(os.getenv("SKEW_SUGGESTED_HOLD_HOURS", "48").strip() or "48")
        return cls(
            squeeze_fr_z_max=_f("SKEW_SQUEEZE_FR_Z_MAX", -0.8),
            squeeze_oi_z_min=_f("SKEW_SQUEEZE_OI_Z_MIN", 0.4),
            squeeze_px_chg_max=_f("SKEW_SQUEEZE_PX_CHG_MAX", 15.0),
            dark_oi_z_min=_f("SKEW_DARK_OI_Z_MIN", 1.0),
            dark_fr_z_min=_f("SKEW_DARK_FR_Z_MIN", 0.0),
            dark_px_abs_max=_f("SKEW_DARK_PX_ABS_MAX", 3.0),
            weak_oi_z_max=_f("SKEW_WEAK_OI_Z_MAX", -0.5),
            weak_fr_z_min=_f("SKEW_WEAK_FR_Z_MIN", 1.0),
            cover_trap_px_min=_f("SKEW_COVER_TRAP_PX_MIN", 3.0),
            cover_trap_oi_z_max=_f("SKEW_COVER_TRAP_OI_Z_MAX", -1.0),
            regime_btc_drop_pct=_f("SKEW_REGIME_BTC_DROP_PCT", 5.0),
            regime_btc_fr_z_max=_f("SKEW_REGIME_BTC_FR_Z_MAX", -2.0),
            regime_btc_oi_z_max=_f("SKEW_REGIME_BTC_OI_Z_MAX", -1.5),
            regime_cautious_drop_pct=_f("SKEW_REGIME_CAUTIOUS_DROP_PCT", 4.0),
            regime_cautious_fr_z_max=_f("SKEW_REGIME_CAUTIOUS_FR_Z_MAX", -1.5),
            enable_dark_long=enable_dark,
            long_signal_types=long_types,
            pair_enabled=os.getenv("SKEW_PAIR_ENABLED", "1").strip().lower()
            in ("1", "true", "yes", "on"),
            pair_risk_pct=_f("SKEW_PAIR_RISK_PCT", risk_default),
            pair_leverage=_f("SKEW_PAIR_LEVERAGE", lev_default),
            suggested_hold_hours=max(12, hold_h),
            min_skew_gap=_f("SKEW_MIN_SKEW_GAP", gap_default),
        )


_DEFAULT_CFG = SkewConfig.from_env()


def _cst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=8)))


def _z_scores(values: List[float]) -> List[float]:
    if not values:
        return []
    n = len(values)
    if n == 1:
        return [0.0]
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var) if var > 1e-12 else 1.0
    return [(v - mean) / std for v in values]


def ensure_skew_tables(conn: sqlite3.Connection) -> None:
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS skew_signals (
            symbol TEXT NOT NULL,
            coin TEXT,
            generated_date TEXT NOT NULL,
            last_seen_cst TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            side TEXT NOT NULL,
            skew_score REAL,
            fr_z REAL,
            oi_z REAL,
            px_z REAL,
            px_chg REAL,
            fr_pct REAL,
            d6h REAL,
            vol_24h REAL,
            rank_in_scan INTEGER,
            summary_line TEXT,
            detail_json TEXT,
            PRIMARY KEY (symbol, generated_date)
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS skew_pairs (
            pair_id TEXT PRIMARY KEY,
            generated_date TEXT NOT NULL,
            last_seen_cst TEXT NOT NULL,
            long_symbol TEXT NOT NULL,
            short_symbol TEXT NOT NULL,
            long_skew REAL,
            short_skew REAL,
            regime TEXT NOT NULL,
            summary_line TEXT,
            detail_json TEXT
        )
        """
    )
    conn.commit()


def _skew_prune(conn: sqlite3.Connection, now: datetime) -> int:
    today = _cst_now().date() if now.tzinfo is None else now.astimezone(timezone(timedelta(hours=8))).date()
    cutoff = (today - timedelta(days=SKEW_RETENTION_DAYS - 1)).isoformat()
    cur = conn.cursor()
    cur.execute("DELETE FROM skew_signals WHERE generated_date < ?", (cutoff,))
    n1 = cur.rowcount
    cur.execute("DELETE FROM skew_pairs WHERE generated_date < ?", (cutoff,))
    n2 = cur.rowcount
    conn.commit()
    return int(n1 or 0) + int(n2 or 0)


def _fetch_universe() -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """返回 ticker_map, funding_map。"""
    tickers_raw = api_get("/fapi/v1/ticker/24hr")
    premiums_raw = api_get("/fapi/v1/premiumIndex")
    if not tickers_raw or not premiums_raw:
        return {}, {}

    funding_map = {
        p["symbol"]: float(p["lastFundingRate"])
        for p in premiums_raw
        if str(p.get("symbol", "")).endswith("USDT")
    }

    ticker_map: Dict[str, Dict[str, float]] = {}
    for t in tickers_raw:
        sym = str(t.get("symbol") or "")
        if not sym.endswith("USDT"):
            continue
        vol = float(t.get("quoteVolume") or 0)
        if vol < SKEW_MIN_VOL_USD:
            continue
        ticker_map[sym] = {
            "px_chg": float(t.get("priceChangePercent") or 0),
            "vol": vol,
            "price": float(t.get("lastPrice") or 0),
            "funding": funding_map.get(sym, 0.0),
        }

    top = sorted(ticker_map.items(), key=lambda x: x[1]["vol"], reverse=True)[:SKEW_UNIVERSE_TOP_N]
    return dict(top), funding_map


def _fetch_oi_deltas(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i, sym in enumerate(symbols):
        hist = api_get(
            "/futures/data/openInterestHist",
            {"symbol": sym, "period": "1h", "limit": 6},
        )
        if hist and len(hist) >= 2:
            curr = float(hist[-1].get("sumOpenInterestValue") or 0)
            prev_6h = float(hist[0].get("sumOpenInterestValue") or 0)
            d6h = ((curr - prev_6h) / prev_6h * 100.0) if prev_6h > 0 else 0.0
            out[sym] = {"oi_usd": curr, "d6h": d6h}
        if (i + 1) % 15 == 0:
            time.sleep(0.3)
    return out


def _classify_signal(
    *,
    fr_z: float,
    oi_z: float,
    px_chg: float,
    cfg: Optional[SkewConfig] = None,
) -> Tuple[str, str, float]:
    """返回 (signal_type, side, skew_score)。"""
    c = cfg or _DEFAULT_CFG
    skew_score = oi_z * 1.2 - fr_z * 0.8 + (oi_z - px_chg / 10.0) * 0.3

    if px_chg > c.cover_trap_px_min and oi_z < c.cover_trap_oi_z_max:
        return "cover_trap", "SKIP", skew_score

    if (
        fr_z <= c.squeeze_fr_z_max
        and oi_z >= c.squeeze_oi_z_min
        and px_chg < c.squeeze_px_chg_max
    ):
        return "squeeze_long", "LONG", skew_score + abs(fr_z)

    if (
        c.enable_dark_long
        and oi_z >= c.dark_oi_z_min
        and fr_z >= c.dark_fr_z_min
        and abs(px_chg) <= c.dark_px_abs_max
    ):
        return "dark_long", "LONG", skew_score + oi_z * 0.5

    if oi_z <= c.weak_oi_z_max and fr_z >= c.weak_fr_z_min and px_chg < 0:
        return "weak_short", "SHORT", -skew_score + fr_z

    return "neutral", "SKIP", skew_score


def _btc_regime(
    btc_row: Optional[Dict[str, Any]],
    cfg: Optional[SkewConfig] = None,
) -> Dict[str, Any]:
    c = cfg or _DEFAULT_CFG
    if not btc_row:
        return {"regime": "normal", "halt_new_pairs": False, "reason": "no_btc_data"}
    px_chg = float(btc_row.get("px_chg") or 0)
    fr_z = float(btc_row.get("fr_z") or 0)
    oi_z = float(btc_row.get("oi_z") or 0)
    if px_chg <= -c.regime_btc_drop_pct:
        return {
            "regime": "defensive",
            "halt_new_pairs": True,
            "reason": f"btc_drop_{px_chg:.1f}pct",
        }
    if fr_z <= c.regime_btc_fr_z_max and oi_z <= c.regime_btc_oi_z_max:
        return {
            "regime": "defensive",
            "halt_new_pairs": True,
            "reason": "btc_deleveraging",
        }
    if px_chg <= -c.regime_cautious_drop_pct or fr_z <= c.regime_cautious_fr_z_max:
        return {
            "regime": "cautious",
            "halt_new_pairs": False,
            "reason": "elevated_vol",
        }
    return {"regime": "normal", "halt_new_pairs": False, "reason": "ok"}


def _build_summary(coin: str, signal_type: str, px_chg: float, fr_pct: float, d6h: float) -> str:
    label = SIGNAL_TYPE_ZH.get(signal_type, signal_type)
    return (
        f"{label} · {coin} | 24h {px_chg:+.1f}% | "
        f"费率 {fr_pct:+.4f}% | OI 6h {d6h:+.1f}%"
    )


def scan_skew_universe() -> List[Dict[str, Any]]:
    ticker_map, _ = _fetch_universe()
    if not ticker_map:
        return []

    oi_map = _fetch_oi_deltas(list(ticker_map.keys()))
    fr_vals = [t["funding"] for t in ticker_map.values()]
    oi_vals = [oi_map.get(s, {}).get("d6h", 0.0) for s in ticker_map]
    px_vals = [t["px_chg"] for t in ticker_map.values()]

    fr_zs = _z_scores(fr_vals)
    oi_zs = _z_scores(oi_vals)
    px_zs = _z_scores(px_vals)

    rows: List[Dict[str, Any]] = []
    for i, (sym, tk) in enumerate(ticker_map.items()):
        coin = sym.replace("USDT", "")
        oi = oi_map.get(sym, {})
        d6h = float(oi.get("d6h") or 0)
        fr_z = fr_zs[i]
        oi_z = oi_zs[i]
        px_z = px_zs[i]
        px_chg = tk["px_chg"]
        fr_pct = tk["funding"] * 100.0

        signal_type, side, skew_score = _classify_signal(
            fr_z=fr_z, oi_z=oi_z, px_chg=px_chg
        )
        rows.append(
            {
                "symbol": sym,
                "coin": coin,
                "signal_type": signal_type,
                "side": side,
                "skew_score": round(skew_score, 4),
                "fr_z": round(fr_z, 4),
                "oi_z": round(oi_z, 4),
                "px_z": round(px_z, 4),
                "px_chg": round(px_chg, 2),
                "fr_pct": round(fr_pct, 4),
                "d6h": round(d6h, 2),
                "vol_24h": tk["vol"],
                "price": tk["price"],
                "summary_line": _build_summary(coin, signal_type, px_chg, fr_pct, d6h),
            }
        )

    rows.sort(key=lambda r: abs(float(r.get("skew_score") or 0)), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_in_scan"] = rank
    return rows


def pick_pair_trade(
    rows: List[Dict[str, Any]],
    regime: Dict[str, Any],
    cfg: Optional[SkewConfig] = None,
) -> Optional[Dict[str, Any]]:
    c = cfg or _DEFAULT_CFG
    if not c.pair_enabled or regime.get("halt_new_pairs"):
        return None

    long_types = set(c.long_signal_types)
    long_candidates = [
        r
        for r in rows
        if r.get("signal_type") in long_types and r.get("side") == "LONG"
    ]
    short_candidates = [
        r for r in rows if r.get("signal_type") == "weak_short" and r.get("side") == "SHORT"
    ]
    if not long_candidates or not short_candidates:
        return None

    long_candidates.sort(key=lambda r: float(r.get("skew_score") or 0), reverse=True)
    short_candidates.sort(key=lambda r: float(r.get("skew_score") or 0), reverse=True)

    long_leg = long_candidates[0]
    short_leg = next(
        (r for r in short_candidates if r["symbol"] != long_leg["symbol"]),
        None,
    )
    if not short_leg:
        return None

    skew_gap = float(long_leg["skew_score"]) - float(short_leg["skew_score"])
    if skew_gap < c.min_skew_gap:
        return None

    pair_id = uuid.uuid4().hex[:12]
    leg_notional_pct = c.pair_risk_pct / 2.0
    summary = (
        f"Skew 配对 · 多 {long_leg['coin']} / 空 {short_leg['coin']} | "
        f"regime={regime.get('regime')} | "
        f"skew {float(long_leg['skew_score']):.2f} vs {float(short_leg['skew_score']):.2f} | "
        f"仓位 {c.pair_risk_pct*100:.0f}% × {c.pair_leverage:.0f}x"
    )
    return {
        "pair_id": pair_id,
        "long_symbol": long_leg["symbol"],
        "short_symbol": short_leg["symbol"],
        "long_coin": long_leg["coin"],
        "short_coin": short_leg["coin"],
        "long_skew": long_leg["skew_score"],
        "short_skew": short_leg["skew_score"],
        "skew_gap": round(skew_gap, 4),
        "long_signal": long_leg["signal_type"],
        "short_signal": short_leg["signal_type"],
        "regime": str(regime.get("regime") or "normal"),
        "summary_line": summary,
        "position": {
            "pair_risk_pct": c.pair_risk_pct,
            "pair_leverage": c.pair_leverage,
            "leg_notional_pct": leg_notional_pct,
            "suggested_hold_hours": c.suggested_hold_hours,
            "profile": os.getenv("SKEW_PROFILE", "aggressive"),
        },
        "detail": {
            "long": long_leg,
            "short": short_leg,
            "regime": regime,
        },
    }


def _persist_skew_signals(conn: sqlite3.Connection, rows: List[Dict[str, Any]], now_cst: datetime) -> None:
    generated_date = now_cst.date().isoformat()
    now_label = now_cst.strftime("%Y-%m-%d %H:%M")
    cur = conn.cursor()
    actionable = [r for r in rows if r.get("signal_type") != "neutral"]
    for r in actionable[:40]:
        cur.execute(
            """
            INSERT OR REPLACE INTO skew_signals (
                symbol, coin, generated_date, last_seen_cst, signal_type, side,
                skew_score, fr_z, oi_z, px_z, px_chg, fr_pct, d6h, vol_24h,
                rank_in_scan, summary_line, detail_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["symbol"],
                r["coin"],
                generated_date,
                now_label,
                r["signal_type"],
                r["side"],
                r["skew_score"],
                r["fr_z"],
                r["oi_z"],
                r["px_z"],
                r["px_chg"],
                r["fr_pct"],
                r["d6h"],
                r["vol_24h"],
                r.get("rank_in_scan"),
                r.get("summary_line"),
                json.dumps(r, ensure_ascii=False),
            ),
        )
    conn.commit()


def _persist_pair(conn: sqlite3.Connection, pair: Optional[Dict[str, Any]], now_cst: datetime) -> None:
    if not pair:
        return
    generated_date = now_cst.date().isoformat()
    now_label = now_cst.strftime("%Y-%m-%d %H:%M")
    conn.execute(
        """
        INSERT OR REPLACE INTO skew_pairs (
            pair_id, generated_date, last_seen_cst,
            long_symbol, short_symbol, long_skew, short_skew,
            regime, summary_line, detail_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pair["pair_id"],
            generated_date,
            now_label,
            pair["long_symbol"],
            pair["short_symbol"],
            pair["long_skew"],
            pair["short_skew"],
            pair["regime"],
            pair["summary_line"],
            json.dumps(pair.get("detail") or {}, ensure_ascii=False),
        ),
    )
    conn.commit()


def _record_pair_signals(pair: Dict[str, Any]) -> None:
    if not SKEW_RECORD_SIGNALS:
        return
    try:
        from orb.vnpy.strategy_signals import LANE_SKEW_NEUTRAL, record_strategy_signal

        detail_base = {
            "pair_id": pair["pair_id"],
            "regime": pair["regime"],
            "strategy": "skew_neutral_pair",
        }
        long_det = pair.get("detail", {}).get("long") or {}
        short_det = pair.get("detail", {}).get("short") or {}
        record_strategy_signal(
            lane=LANE_SKEW_NEUTRAL,
            symbol=pair["long_symbol"],
            side="LONG",
            status="emitted",
            detail={**detail_base, "leg": "long", "signal_type": pair.get("long_signal"), "peer": pair["short_symbol"], **long_det},
        )
        record_strategy_signal(
            lane=LANE_SKEW_NEUTRAL,
            symbol=pair["short_symbol"],
            side="SHORT",
            status="emitted",
            detail={**detail_base, "leg": "short", "signal_type": pair.get("short_signal"), "peer": pair["long_symbol"], **short_det},
        )
    except Exception as e:
        print(f"⚠️ skew 信号写入 strategy_signals 失败: {e}")


def _persist_snapshot(payload: Dict[str, Any]) -> None:
    if not payload.get("ok"):
        return
    tmp = SKEW_SNAPSHOT_PATH.with_suffix(".json.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp.replace(SKEW_SNAPSHOT_PATH)
        print(f"  💾 Skew 策略快照已写入 {SKEW_SNAPSHOT_PATH}")
    except Exception as e:
        print(f"⚠️ Skew 快照写入失败: {e}")


def load_skew_snapshot() -> Optional[Dict[str, Any]]:
    if not SKEW_SNAPSHOT_PATH.is_file():
        return None
    try:
        return json.loads(SKEW_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_skew_scan(conn: sqlite3.Connection, *, notify: bool = False) -> Dict[str, Any]:
    """主扫描入口。"""
    del notify  # 预留 Telegram；当前与 OI 雷达一致走 API 快照
    now_cst = _cst_now()
    ensure_skew_tables(conn)
    pruned = _skew_prune(conn, now_cst)
    if pruned:
        print(f"  🧹 Skew 表修剪：删除 {pruned} 行")

    print("📐 扫描期货 Skew 截面...")
    rows = scan_skew_universe()
    if not rows:
        return {"ok": False, "error": "scan_empty", "message": "截面扫描无数据"}

    btc_row = next((r for r in rows if r.get("symbol") == "BTCUSDT"), None)
    regime = _btc_regime(btc_row)
    pair = pick_pair_trade(rows, regime)
    if pair:
        _record_pair_signals(pair)

    _persist_skew_signals(conn, rows, now_cst)
    _persist_pair(conn, pair, now_cst)

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("signal_type") == "neutral":
            continue
        by_type.setdefault(str(r["signal_type"]), []).append(r)

    top_signals = [r for r in rows if r.get("signal_type") != "neutral"][:20]
    payload: Dict[str, Any] = {
        "ok": True,
        "generated_at_cst": now_cst.strftime("%Y-%m-%d %H:%M") + " CST",
        "regime": regime,
        "pair_trade": pair,
        "top_signals": top_signals,
        "signals_by_type": {k: v[:8] for k, v in by_type.items()},
        "universe_size": len(rows),
        "config": {
            "profile": os.getenv("SKEW_PROFILE", "aggressive"),
            "universe_top_n": SKEW_UNIVERSE_TOP_N,
            "pair_enabled": SKEW_PAIR_ENABLED,
            "min_vol_usd": SKEW_MIN_VOL_USD,
            "enable_dark_long": _DEFAULT_CFG.enable_dark_long,
            "pair_risk_pct": _DEFAULT_CFG.pair_risk_pct,
            "pair_leverage": _DEFAULT_CFG.pair_leverage,
            "suggested_hold_hours": _DEFAULT_CFG.suggested_hold_hours,
            "min_skew_gap": _DEFAULT_CFG.min_skew_gap,
        },
    }
    _persist_snapshot(payload)
    print(
        f"  ✅ Skew 扫描完成: {len(rows)} 标的, "
        f"有效信号 {len(top_signals)}, regime={regime.get('regime')}, "
        f"配对={'有' if pair else '无'}"
    )
    return payload


def main() -> None:
    print(f"📐 期货 Skew 中性策略 — {_cst_now().strftime('%Y-%m-%d %H:%M:%S')} CST\n")
    conn = init_db()
    try:
        payload = run_skew_scan(conn)
        if not payload.get("ok"):
            print(f"❌ {payload.get('message') or payload.get('error')}")
            sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

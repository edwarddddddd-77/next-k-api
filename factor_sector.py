"""Barra-style crypto sector factor board (HangukQuant-inspired).

Daily constrained WLS cross-section:
  R_t = B_t f_t + e_t
with market + sector dummies, liquidity weights ~ sqrt(30d quote volume),
and sum(s_c * f_c) = 0 so category factors are market-relative.

Signal layer (Sparkline-style crypto momentum, 1–2 weeks on factor returns):
  rank sectors by cumulative factor return; emit long-top / short-bottom
  factor-mimicking basket weights (paper signal only).

Refs:
  - HangukQuant Research, Trading in factor space (Jul 2026)
  - Sparkline Capital, Crypto Factor Investing (short-horizon momentum)
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

from quant.common.paths import resolve_data_dir

log = logging.getLogger("factor_sector")

SNAPSHOT_NAME = "factor_sector_snapshot.json"
FAPI = "https://fapi.binance.com"

# Liquid USDT-M perps with sector tags (categorical Barra exposures).
SECTOR_MAP: dict[str, str] = {
    "BTCUSDT": "L1",
    "ETHUSDT": "L1",
    "SOLUSDT": "L1",
    "BNBUSDT": "L1",
    "ADAUSDT": "L1",
    "AVAXUSDT": "L1",
    "DOTUSDT": "L1",
    "NEARUSDT": "L1",
    "APTUSDT": "L1",
    "SUIUSDT": "L1",
    "TONUSDT": "L1",
    "ARBUSDT": "L2",
    "OPUSDT": "L2",
    "POLUSDT": "L2",
    "STRKUSDT": "L2",
    "DOGEUSDT": "Meme",
    "1000PEPEUSDT": "Meme",
    "WIFUSDT": "Meme",
    "1000SHIBUSDT": "Meme",
    "ORDIUSDT": "Meme",
    "UNIUSDT": "DeFi",
    "AAVEUSDT": "DeFi",
    "INJUSDT": "DeFi",
    "LINKUSDT": "Oracle",
    "FILUSDT": "Storage",
    "ATOMUSDT": "Cosmos",
    "SEIUSDT": "L1",
    "TIAUSDT": "Modular",
    "RENDERUSDT": "AI",
    "FETUSDT": "AI",
}

LOOKBACK_DAYS = 90
VOL_WINDOW = 30
MOM_WINDOWS = (7, 14)  # trading days of factor returns
REQUEST_PAUSE = 0.08


def _snapshot_path() -> Path:
    return resolve_data_dir() / SNAPSHOT_NAME


def _now_cst_iso() -> str:
    # Asia/Shanghai = UTC+8
    return datetime.now(timezone.utc).astimezone().isoformat()


def fetch_daily_klines(symbol: str, limit: int = LOOKBACK_DAYS + VOL_WINDOW + 5) -> list[dict]:
    url = f"{FAPI}/fapi/v1/klines"
    resp = requests.get(
        url,
        params={"symbol": symbol, "interval": "1d", "limit": int(limit)},
        timeout=25,
    )
    resp.raise_for_status()
    rows = []
    for k in resp.json():
        # open_time, o, h, l, c, vol, close_time, quote_vol, ...
        rows.append({
            "ts": int(k[0]) // 1000,
            "close": float(k[4]),
            "quote_vol": float(k[7]),
        })
    # Drop incomplete current UTC day if still open (last bar close_time in future)
    if rows and rows[-1]["ts"] + 86400 > int(time.time()):
        rows = rows[:-1]
    return rows


def load_universe_history(symbols: list[str] | None = None) -> dict[str, list[dict]]:
    syms = symbols or list(SECTOR_MAP.keys())
    out: dict[str, list[dict]] = {}
    for i, sym in enumerate(syms):
        try:
            out[sym] = fetch_daily_klines(sym)
        except Exception as exc:  # noqa: BLE001
            log.warning("klines %s failed: %s", sym, exc)
        if i + 1 < len(syms):
            time.sleep(REQUEST_PAUSE)
    return out


def _align_dates(history: dict[str, list[dict]]) -> list[int]:
    sets = []
    for rows in history.values():
        if len(rows) < VOL_WINDOW + 5:
            continue
        sets.append({r["ts"] for r in rows})
    if not sets:
        return []
    common = set.intersection(*sets)
    return sorted(common)


def constrained_wls(y: np.ndarray, w: np.ndarray, X: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Solve min 1/2 (y-Xf)'W(y-Xf) s.t. c'f = 0. Returns f (K,)."""
    # W as diagonal via sqrt weights for numerical simplicity
    sw = np.sqrt(np.maximum(w, 1e-12))
    Xw = X * sw[:, None]
    yw = y * sw
    xtwx = Xw.T @ Xw
    xtwy = Xw.T @ yw
    k = X.shape[1]
    kkt = np.zeros((k + 1, k + 1), dtype=float)
    kkt[:k, :k] = xtwx
    kkt[:k, k] = c
    kkt[k, :k] = c
    rhs = np.zeros(k + 1, dtype=float)
    rhs[:k] = xtwy
    try:
        sol = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    return sol[:k]


def build_exposure(sectors: list[str], sector_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """X columns: market + sector dummies. Constraint c on sector factors only."""
    n = len(sectors)
    k = 1 + len(sector_names)
    X = np.zeros((n, k), dtype=float)
    X[:, 0] = 1.0
    for i, sec in enumerate(sectors):
        j = sector_names.index(sec)
        X[i, 1 + j] = 1.0
    c = np.zeros(k, dtype=float)
    # filled later with sector aggregate weights
    return X, c


def factor_mimicking_weights(w: np.ndarray, sectors: list[str], sector: str) -> np.ndarray:
    """p_i = 1{i in c} * w_i/s_c - w_i  (long sector basket, short market)."""
    mask = np.array([s == sector for s in sectors], dtype=float)
    s_c = float((w * mask).sum())
    if s_c <= 1e-12:
        return np.zeros_like(w)
    return mask * (w / s_c) - w


def run_barra(history: dict[str, list[dict]]) -> dict[str, Any]:
    dates = _align_dates(history)
    if len(dates) < VOL_WINDOW + 10:
        raise RuntimeError("insufficient_aligned_history")

    # index by ts
    by_sym = {sym: {r["ts"]: r for r in rows} for sym, rows in history.items()}
    symbols = [s for s in SECTOR_MAP if s in by_sym and len(by_sym[s]) >= VOL_WINDOW + 5]
    if len(symbols) < 8:
        raise RuntimeError("too_few_symbols")

    sector_names = sorted({SECTOR_MAP[s] for s in symbols})
    # Skip dates that don't have prior VOL_WINDOW for volume
    usable = [d for d in dates if dates.index(d) >= VOL_WINDOW]
    if not usable:
        raise RuntimeError("no_usable_dates")

    factor_hist: list[dict[str, Any]] = []
    last_pack: dict[str, Any] | None = None

    for d in usable:
        rets = []
        vols = []
        secs = []
        syms = []
        for sym in symbols:
            cur = by_sym[sym].get(d)
            prev_ts = d - 86400
            # find previous calendar day in series (may skip weekends? crypto has all days)
            # use previous available bar before d
            prev = None
            for back in range(1, 4):
                prev = by_sym[sym].get(d - back * 86400)
                if prev:
                    break
            if not cur or not prev or prev["close"] <= 0:
                continue
            # 30d quote volume ending yesterday
            q = 0.0
            n_v = 0
            for back in range(1, VOL_WINDOW + 1):
                row = by_sym[sym].get(d - back * 86400)
                if row:
                    q += row["quote_vol"]
                    n_v += 1
            if n_v < VOL_WINDOW // 2:
                continue
            rets.append(cur["close"] / prev["close"] - 1.0)
            vols.append(math.sqrt(max(q, 1.0)))
            secs.append(SECTOR_MAP[sym])
            syms.append(sym)

        if len(rets) < 8:
            continue

        y = np.asarray(rets, dtype=float)
        qv = np.asarray(vols, dtype=float)
        w = qv / qv.sum()
        X, c = build_exposure(secs, sector_names)
        # liquidity-weighted sector mass for constraint
        for j, name in enumerate(sector_names):
            mask = np.array([s == name for s in secs], dtype=bool)
            c[1 + j] = float(w[mask].sum())
        f = constrained_wls(y, w, X, c)
        fitted = X @ f
        resid = y - fitted

        day = {
            "ts": d,
            "date": datetime.fromtimestamp(d, tz=timezone.utc).strftime("%Y-%m-%d"),
            "f_market": float(f[0]),
            "factors": {sector_names[j]: float(f[1 + j]) for j in range(len(sector_names))},
            "n": len(syms),
        }
        factor_hist.append(day)
        last_pack = {
            "symbols": syms,
            "sectors": secs,
            "weights": w.tolist(),
            "returns": y.tolist(),
            "residuals": resid.tolist(),
            "f": f.tolist(),
            "sector_names": sector_names,
        }

    if not factor_hist or not last_pack:
        raise RuntimeError("barra_empty")

    # Momentum on factor returns
    sector_series = {name: [] for name in sector_names}
    market_series = []
    for row in factor_hist:
        market_series.append(row["f_market"])
        for name in sector_names:
            sector_series[name].append(row["factors"].get(name, 0.0))

    rankings = []
    for name in sector_names:
        series = sector_series[name]
        entry = {"sector": name, "latest": series[-1] if series else 0.0}
        for win in MOM_WINDOWS:
            window = series[-win:] if len(series) >= win else series
            # cumulative relative factor return
            cum = float(np.prod(1.0 + np.asarray(window)) - 1.0) if window else 0.0
            entry[f"mom_{win}d"] = cum
        rankings.append(entry)

    # Primary signal: 14d mom, tie-break 7d
    rankings.sort(key=lambda r: (r.get("mom_14d", 0.0), r.get("mom_7d", 0.0)), reverse=True)
    for i, r in enumerate(rankings):
        r["rank"] = i + 1

    long_sec = rankings[0]["sector"] if rankings else None
    short_sec = rankings[-1]["sector"] if rankings else None

    # Mimicking portfolio on latest day
    w = np.asarray(last_pack["weights"], dtype=float)
    secs = last_pack["sectors"]
    syms = last_pack["symbols"]
    basket = []
    if long_sec and short_sec and long_sec != short_sec:
        p_long = factor_mimicking_weights(w, secs, long_sec)
        p_short = factor_mimicking_weights(w, secs, short_sec)
        # Long strong sector factor, short weak sector factor
        p = p_long - p_short
        # normalize gross exposure to ~1
        gross = float(np.abs(p).sum()) or 1.0
        p = p / (gross / 2.0)  # target ~1 long + ~1 short notionally
        for sym, wi, sec, ret, eps in zip(
            syms, p, secs, last_pack["returns"], last_pack["residuals"]
        ):
            if abs(wi) < 1e-4:
                continue
            basket.append({
                "symbol": sym,
                "sector": sec,
                "weight": round(float(wi), 6),
                "side": "long" if wi > 0 else "short",
                "ret_1d": round(float(ret), 6),
                "residual_1d": round(float(eps), 6),
            })
        basket.sort(key=lambda x: abs(x["weight"]), reverse=True)

    # Idiosyncratic leaders (yesterday residual)
    idio = []
    for sym, sec, ret, eps in zip(
        last_pack["symbols"], last_pack["sectors"], last_pack["returns"], last_pack["residuals"]
    ):
        idio.append({
            "symbol": sym,
            "sector": sec,
            "ret_1d": round(float(ret), 6),
            "residual_1d": round(float(eps), 6),
        })
    idio_pos = sorted(idio, key=lambda x: x["residual_1d"], reverse=True)[:8]
    idio_neg = sorted(idio, key=lambda x: x["residual_1d"])[:8]

    # Compounded synthetic sector index from 1
    synth = {}
    for name in sector_names:
        lvl = 1.0
        curve = []
        for row in factor_hist[-60:]:
            lvl *= 1.0 + float(row["factors"].get(name, 0.0))
            curve.append({"date": row["date"], "level": round(lvl, 6)})
        synth[name] = curve

    mkt_lvl = 1.0
    mkt_curve = []
    for row in factor_hist[-60:]:
        mkt_lvl *= 1.0 + float(row["f_market"])
        mkt_curve.append({"date": row["date"], "level": round(mkt_lvl, 6)})

    thesis = ""
    if long_sec and short_sec:
        thesis = (
            f"因子动量信号：做多板块 {long_sec}（14d cum {rankings[0].get('mom_14d', 0)*100:.2f}%），"
            f"做空板块 {short_sec}（14d cum {rankings[-1].get('mom_14d', 0)*100:.2f}%）。"
            f"权重来自 Barra 因子模拟组合（板块篮子 − 市场篮子）。纸面信号，未自动下单。"
        )

    return {
        "ok": True,
        "strategy": "S_BARRA_SECTOR",
        "name": "板块因子轮动 (Barra)",
        "fetched_at": int(time.time()),
        "fetched_at_cst": _now_cst_iso(),
        "universe_n": len(symbols),
        "sectors": sector_names,
        "history_days": len(factor_hist),
        "latest": factor_hist[-1],
        "factor_history": factor_hist[-60:],
        "rankings": rankings,
        "signal": {
            "long_sector": long_sec,
            "short_sector": short_sec,
            "mom_window": 14,
            "thesis": thesis,
            "basket": basket[:24],
        },
        "idio_leaders": idio_pos,
        "idio_laggards": idio_neg,
        "synthetic_index": synth,
        "market_index": mkt_curve,
        "method": {
            "model": "Barra categorical constrained WLS",
            "weights": "sqrt(30d quote volume)",
            "constraint": "liquidity-weighted sector factor returns sum to 0",
            "signal": "14d cumulative sector factor momentum (7d tie-break)",
            "refs": [
                "HangukQuant — Trading in factor space",
                "Sparkline — Crypto factor momentum 1–4 weeks",
            ],
        },
    }


def build_board(*, force_refresh: bool = True) -> dict[str, Any]:
    history = load_universe_history()
    board = run_barra(history)
    path = _snapshot_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.warning("factor sector snapshot write failed: %s", exc)
    board["snapshot_source"] = "live"
    board["cached"] = False
    return board


def load_snapshot() -> dict[str, Any]:
    path = _snapshot_path()
    if not path.is_file():
        return {
            "ok": False,
            "error": "no_snapshot",
            "message": "尚无板块因子快照，请点击刷新生成（约需 30–60 秒拉币安日线）。",
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    data["snapshot_source"] = "disk"
    data["cached"] = True
    age = int(time.time()) - int(data.get("fetched_at") or 0)
    data["age_sec"] = age
    data["stale"] = age > 36 * 3600
    return data


def refresh_snapshot(*, force: bool = True) -> dict[str, Any]:
    return build_board(force_refresh=force)

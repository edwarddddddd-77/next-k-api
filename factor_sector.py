"""HangukQuant-aligned Barra sector factor board (part 2) + trend cascade.

Aligned with the public HangukQuant articles:
  1) Daily constrained WLS: R = X f + e
     - exposures: market + categorical dummies (L1 / Meme / Other)
     - weights w_i ∝ sqrt(prior 30d dollar volume)  [= sqrt(sum quote_vol)]
     - constraint: sum_c s_c * f_c = 0  (liquidity-weighted category returns)
  2) Synthetic sector index by compounding f_c from 1
  3) Factor-mimicking weights:
       p_market = w
       p_c,i = 1{i in c} * w_i/s_c - w_i
  4) Trade in factor space with cascade trend:
       pass sign(trail_20d) on synthetic index, then same-sign(trail_7d)
       → long confirmed bulls / short confirmed bears

Paper signal only. Paid HangukQuant source is not copied.

Refs:
  - https://www.research.hangukquant.com/p/quantitative-trading-strategies-how-3df
  - https://www.research.hangukquant.com/p/quantitative-trading-strategies-how-c9a
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

# Article demo uses L1 vs Meme; expand universe but keep the same 3-way taxonomy
# (market + L1 + Meme + Other) so the constrained Barra matches the write-up.
SECTOR_MAP: dict[str, str] = {
    # L1
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
    "SEIUSDT": "L1",
    # Meme
    "DOGEUSDT": "Meme",
    "1000PEPEUSDT": "Meme",
    "WIFUSDT": "Meme",
    "1000SHIBUSDT": "Meme",
    "ORDIUSDT": "Meme",
    # Other (L2 / DeFi / infra / AI …) — residual category
    "ARBUSDT": "Other",
    "OPUSDT": "Other",
    "POLUSDT": "Other",
    "STRKUSDT": "Other",
    "UNIUSDT": "Other",
    "AAVEUSDT": "Other",
    "INJUSDT": "Other",
    "LINKUSDT": "Other",
    "FILUSDT": "Other",
    "ATOMUSDT": "Other",
    "TIAUSDT": "Other",
    "RENDERUSDT": "Other",
    "FETUSDT": "Other",
}

LOOKBACK_DAYS = 120
VOL_WINDOW = 30          # article: prior 30d dollar volume
TREND_LOOKBACK = 20      # gate: sign of trailing return on synthetic index
MOM_CONFIRM_LOOKBACK = 7 # confirm: same-direction trail over shorter window
REQUEST_PAUSE = 0.08
SWITCH_COST_BPS = 10.0


def _snapshot_path() -> Path:
    return resolve_data_dir() / SNAPSHOT_NAME


def _now_cst_iso() -> str:
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
        rows.append({
            "ts": int(k[0]) // 1000,
            "close": float(k[4]),
            "quote_vol": float(k[7]),  # ≈ Σ P·V in USDT
        })
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
    return sorted(set.intersection(*sets))


def constrained_wls(y: np.ndarray, w: np.ndarray, X: np.ndarray, c: np.ndarray) -> np.ndarray:
    """HangukQuant KKT: min 1/2 (y-Xf)'W(y-Xf) s.t. c'f = 0."""
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


def build_exposure(sectors: list[str], sector_names: list[str]) -> np.ndarray:
    """X columns: [market, cat1, cat2, ...] — article exposure matrix."""
    n = len(sectors)
    k = 1 + len(sector_names)
    X = np.zeros((n, k), dtype=float)
    X[:, 0] = 1.0
    for i, sec in enumerate(sectors):
        X[i, 1 + sector_names.index(sec)] = 1.0
    return X


def factor_mimicking_weights(w: np.ndarray, sectors: list[str], sector: str) -> np.ndarray:
    """Article §9: p_c,i = 1{i∈c}·(w_i/s_c) − w_i."""
    mask = np.array([s == sector for s in sectors], dtype=float)
    s_c = float((w * mask).sum())
    if s_c <= 1e-12:
        return np.zeros_like(w)
    return mask * (w / s_c) - w


def compound_levels(factor_rets: list[float]) -> list[float]:
    """Article §8: synthetic sector index from compounding factor returns at 1."""
    lvl = 1.0
    out = []
    for r in factor_rets:
        lvl *= 1.0 + float(r)
        out.append(lvl)
    return out


def trend_sign_trailing(levels: list[float], lookback: int = TREND_LOOKBACK) -> int:
    """Part 1 simplest discrete trend: sign of trailing return."""
    if len(levels) <= lookback:
        return 0
    r = levels[-1] / levels[-1 - lookback] - 1.0
    if r > 0:
        return 1
    if r < 0:
        return -1
    return 0


def _sector_series(factor_hist: list[dict[str, Any]], name: str) -> list[float]:
    return [float(row["factors"].get(name, 0.0)) for row in factor_hist]


def _trail_ret(levels: list[float], lookback: int) -> float:
    if len(levels) <= lookback:
        return 0.0
    return levels[-1] / levels[-1 - lookback] - 1.0


def classify_sectors(
    factor_hist: list[dict[str, Any]],
    sector_names: list[str],
    *,
    lookback: int = TREND_LOOKBACK,
    confirm_lookback: int = MOM_CONFIRM_LOOKBACK,
) -> dict[str, dict[str, Any]]:
    """Cascade: pass 20d trail sign, then same-direction 7d trail sign.

    Long only if trail_20 > 0 and trail_7 > 0.
    Short only if trail_20 < 0 and trail_7 < 0.
    Otherwise flat (sit out on that sector).
    """
    out = {}
    for name in sector_names:
        levels = compound_levels(_sector_series(factor_hist, name))
        sign20 = trend_sign_trailing(levels, lookback)
        sign7 = trend_sign_trailing(levels, confirm_lookback)
        trail20 = _trail_ret(levels, lookback)
        trail7 = _trail_ret(levels, confirm_lookback)
        # Gate on longer trend, confirm with shorter momentum window
        if sign20 > 0 and sign7 > 0:
            sign = 1
        elif sign20 < 0 and sign7 < 0:
            sign = -1
        else:
            sign = 0
        out[name] = {
            "sector": name,
            "trend": sign,
            "trend_20": sign20,
            "trend_7": sign7,
            "label": "bullish" if sign > 0 else ("bearish" if sign < 0 else "flat"),
            "trail_ret": trail20,
            "trail_ret_20": trail20,
            "trail_ret_7": trail7,
            "level": levels[-1] if levels else 1.0,
        }
    return out


def ls_factor_return(
    nxt_factors: dict[str, float],
    bulls: list[str],
    bears: list[str],
) -> float:
    """Equal-weight long bullish factor(s) vs short bearish factor(s)."""
    if not bulls or not bears:
        return 0.0
    long_leg = float(np.mean([nxt_factors.get(s, 0.0) for s in bulls]))
    short_leg = float(np.mean([nxt_factors.get(s, 0.0) for s in bears]))
    return long_leg - short_leg


def walk_forward_factor_trend(
    factor_hist: list[dict[str, Any]],
    *,
    lookback: int = TREND_LOOKBACK,
    confirm_lookback: int = MOM_CONFIRM_LOOKBACK,
    cost_bps: float = SWITCH_COST_BPS,
) -> dict[str, Any]:
    """Signal at t: 20d gate then 7d confirm; earn LS factor return on t+1."""
    need = max(lookback, confirm_lookback) + 5
    if len(factor_hist) < need:
        return {"ok": False, "error": "too_short"}

    sector_names = sorted(factor_hist[0]["factors"].keys())
    net_rets: list[float] = []
    gross_rets: list[float] = []
    mkt_rets: list[float] = []
    curve = []
    equity = 1.0
    mkt_eq = 1.0
    prev_key = None
    switches = 0
    wins = 0
    start_i = max(lookback, confirm_lookback)

    for i in range(start_i, len(factor_hist) - 1):
        hist_i = factor_hist[: i + 1]
        state = classify_sectors(
            hist_i, sector_names, lookback=lookback, confirm_lookback=confirm_lookback,
        )
        bulls = [n for n, s in state.items() if s["trend"] > 0]
        bears = [n for n, s in state.items() if s["trend"] < 0]
        # If one side empty after cascade: sit out
        nxt = factor_hist[i + 1]
        if not bulls or not bears:
            gross = 0.0
            key = ("flat",)
        else:
            gross = ls_factor_return(nxt["factors"], bulls, bears)
            key = (tuple(sorted(bulls)), tuple(sorted(bears)))
        cost = 0.0
        if prev_key is not None and key != prev_key and key != ("flat",):
            cost = cost_bps / 10000.0
            switches += 1
        net = gross - cost
        net_rets.append(net)
        gross_rets.append(gross)
        mkt_r = float(nxt["f_market"])
        mkt_rets.append(mkt_r)
        equity *= 1.0 + net
        mkt_eq *= 1.0 + mkt_r
        if net > 0:
            wins += 1
        curve.append({
            "date": nxt["date"],
            "bulls": bulls,
            "bears": bears,
            "ret": round(net, 6),
            "equity": round(equity, 6),
            "mkt_equity": round(mkt_eq, 6),
        })
        if key != ("flat",):
            prev_key = key

    if not net_rets:
        return {"ok": False, "error": "no_trades"}

    arr = np.asarray(net_rets, dtype=float)
    mkt_arr = np.asarray(mkt_rets, dtype=float)
    eq = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    max_dd = float(dd.min())
    mu = float(arr.mean())
    sig = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sharpe = (mu / sig * math.sqrt(365)) if sig > 1e-12 else 0.0
    total = float(eq[-1] - 1.0)
    mkt_total = float(np.prod(1.0 + mkt_arr) - 1.0)
    days = len(net_rets)
    years = days / 365.0
    cagr = float(eq[-1] ** (1.0 / years) - 1.0) if years > 0.05 else total
    active = sum(1 for r in net_rets if abs(r) > 1e-12)

    return {
        "ok": True,
        "rule": (
            f"cascade: sign(trail_{lookback}d) then same-sign(trail_{confirm_lookback}d); "
            "long bulls / short bears"
        ),
        "lookback": lookback,
        "confirm_lookback": confirm_lookback,
        "cost_bps": cost_bps,
        "days": days,
        "active_days": active,
        "start": curve[0]["date"],
        "end": curve[-1]["date"],
        "total_return": round(total, 6),
        "cagr": round(cagr, 6),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 6),
        "win_rate": round(wins / days, 4) if days else 0.0,
        "switches": switches,
        "avg_daily": round(mu, 6),
        "market_total": round(mkt_total, 6),
        "excess_vs_market": round(total - mkt_total, 6),
        "gross_total": round(float(np.prod(1.0 + np.asarray(gross_rets)) - 1.0), 6),
        "curve": curve[-90:],
        "note": (
            f"先过 {lookback}d 趋势符号，再过 {confirm_lookback}d 同向确认；"
            "两侧都有才开多空；换仓扣 cost_bps。未计资金费/冲击。"
        ),
    }


def build_mimicking_basket(
    w: np.ndarray,
    sectors: list[str],
    symbols: list[str],
    returns: list[float],
    residuals: list[float],
    bulls: list[str],
    bears: list[str],
) -> list[dict[str, Any]]:
    if not bulls or not bears:
        return []
    p = np.zeros_like(w)
    for s in bulls:
        p = p + factor_mimicking_weights(w, sectors, s)
    p = p / len(bulls)
    for s in bears:
        p = p - factor_mimicking_weights(w, sectors, s) / len(bears)
    gross = float(np.abs(p).sum()) or 1.0
    p = p / (gross / 2.0)
    basket = []
    for sym, wi, sec, ret, eps in zip(symbols, p, sectors, returns, residuals):
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
    return basket


def run_barra(history: dict[str, list[dict]]) -> dict[str, Any]:
    dates = _align_dates(history)
    if len(dates) < VOL_WINDOW + 10:
        raise RuntimeError("insufficient_aligned_history")

    by_sym = {sym: {r["ts"]: r for r in rows} for sym, rows in history.items()}
    symbols = [s for s in SECTOR_MAP if s in by_sym and len(by_sym[s]) >= VOL_WINDOW + 5]
    if len(symbols) < 8:
        raise RuntimeError("too_few_symbols")

    # Stable category order: L1, Meme, Other (article-style)
    preferred = ["L1", "Meme", "Other"]
    present = {SECTOR_MAP[s] for s in symbols}
    sector_names = [c for c in preferred if c in present]
    sector_names += sorted(present - set(sector_names))

    usable = [d for d in dates if dates.index(d) >= VOL_WINDOW]
    if not usable:
        raise RuntimeError("no_usable_dates")

    factor_hist: list[dict[str, Any]] = []
    last_pack: dict[str, Any] | None = None

    for d in usable:
        rets, vols, secs, syms = [], [], [], []
        for sym in symbols:
            cur = by_sym[sym].get(d)
            prev = None
            for back in range(1, 4):
                prev = by_sym[sym].get(d - back * 86400)
                if prev:
                    break
            if not cur or not prev or prev["close"] <= 0:
                continue
            # Article: q_i = sqrt(Σ_{τ=t-30}^{t-1} P_τ V_τ)
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
        X = build_exposure(secs, sector_names)
        c = np.zeros(1 + len(sector_names), dtype=float)
        for j, name in enumerate(sector_names):
            mask = np.array([s == name for s in secs], dtype=bool)
            c[1 + j] = float(w[mask].sum())
        f = constrained_wls(y, w, X, c)
        resid = y - (X @ f)

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
            "sector_names": sector_names,
        }

    if not factor_hist or not last_pack:
        raise RuntimeError("barra_empty")

    backtest = walk_forward_factor_trend(
        factor_hist,
        lookback=TREND_LOOKBACK,
        confirm_lookback=MOM_CONFIRM_LOOKBACK,
        cost_bps=SWITCH_COST_BPS,
    )

    state = classify_sectors(
        factor_hist,
        sector_names,
        lookback=TREND_LOOKBACK,
        confirm_lookback=MOM_CONFIRM_LOOKBACK,
    )
    bulls = [n for n, s in state.items() if s["trend"] > 0]
    bears = [n for n, s in state.items() if s["trend"] < 0]
    rankings = sorted(state.values(), key=lambda r: r["trail_ret"], reverse=True)
    for i, r in enumerate(rankings):
        r["rank"] = i + 1

    w = np.asarray(last_pack["weights"], dtype=float)
    basket = build_mimicking_basket(
        w,
        last_pack["sectors"],
        last_pack["symbols"],
        last_pack["returns"],
        last_pack["residuals"],
        bulls,
        bears,
    )

    # Market mimicking = regression weights (article §9)
    p_mkt = [
        {"symbol": sym, "sector": sec, "weight": round(float(wi), 6)}
        for sym, sec, wi in zip(last_pack["symbols"], last_pack["sectors"], w)
    ]
    p_mkt.sort(key=lambda x: -x["weight"])

    idio = [
        {
            "symbol": sym,
            "sector": sec,
            "ret_1d": round(float(ret), 6),
            "residual_1d": round(float(eps), 6),
        }
        for sym, sec, ret, eps in zip(
            last_pack["symbols"], last_pack["sectors"],
            last_pack["returns"], last_pack["residuals"],
        )
    ]
    idio_pos = sorted(idio, key=lambda x: x["residual_1d"], reverse=True)[:8]
    idio_neg = sorted(idio, key=lambda x: x["residual_1d"])[:8]

    synth = {}
    for name in sector_names:
        levels = compound_levels(_sector_series(factor_hist, name))
        synth[name] = [
            {"date": factor_hist[i]["date"], "level": round(levels[i], 6)}
            for i in range(max(0, len(levels) - 60), len(levels))
        ]

    mkt_levels = compound_levels([row["f_market"] for row in factor_hist])
    mkt_curve = [
        {"date": factor_hist[i]["date"], "level": round(mkt_levels[i], 6)}
        for i in range(max(0, len(mkt_levels) - 60), len(mkt_levels))
    ]

    if bulls and bears:
        thesis = (
            f"串联规则：先过 {TREND_LOOKBACK}d 趋势，再过 {MOM_CONFIRM_LOOKBACK}d 同向确认；"
            f"看涨 {','.join(bulls)}，看跌 {','.join(bears)}；"
            f"持有因子模拟组合（多看涨 − 空看跌）。纸面信号，未自动下单。"
        )
    else:
        thesis = (
            f"当前无完整多空对（bulls={bulls or '—'}, bears={bears or '—'}）："
            f"需同时满足 {TREND_LOOKBACK}d 与 {MOM_CONFIRM_LOOKBACK}d 同向后才开仓。"
        )

    return {
        "ok": True,
        "strategy": "S_BARRA_SECTOR",
        "name": "板块因子 (HangukQuant Barra)",
        "alignment": {
            "barra_wls": True,
            "vol_weight_sqrt_30d": True,
            "category_constraint": True,
            "categories": sector_names,
            "synthetic_index": True,
            "factor_mimicking": True,
            "trend_rule": (
                f"cascade sign(trail_{TREND_LOOKBACK}d) → "
                f"same-sign(trail_{MOM_CONFIRM_LOOKBACK}d)"
            ),
            "trade_rule": "long confirmed bulls / short confirmed bears",
        },
        "fetched_at": int(time.time()),
        "fetched_at_cst": _now_cst_iso(),
        "universe_n": len(symbols),
        "sectors": sector_names,
        "history_days": len(factor_hist),
        "latest": factor_hist[-1],
        "factor_history": factor_hist[-60:],
        "rankings": rankings,
        "signal": {
            "bulls": bulls,
            "bears": bears,
            "long_sector": ",".join(bulls) if bulls else "",
            "short_sector": ",".join(bears) if bears else "",
            "trend_lookback": TREND_LOOKBACK,
            "confirm_lookback": MOM_CONFIRM_LOOKBACK,
            "thesis": thesis,
            "basket": basket[:24],
            "market_mimicking": p_mkt[:12],
        },
        "idio_leaders": idio_pos,
        "idio_laggards": idio_neg,
        "synthetic_index": synth,
        "market_index": mkt_curve,
        "method": {
            "model": "Barra categorical constrained WLS (HangukQuant)",
            "weights": "w_i ∝ sqrt(Σ prior 30d quote volume)",
            "constraint": "Σ s_c f_c = 0",
            "signal": (
                f"cascade trail{TREND_LOOKBACK}d then trail{MOM_CONFIRM_LOOKBACK}d "
                "same sign; L/S confirmed sectors"
            ),
            "refs": [
                "HangukQuant — Trading in factor space",
                "HangukQuant — Trend my friend (sign of trailing return)",
            ],
        },
        "backtest": backtest,
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

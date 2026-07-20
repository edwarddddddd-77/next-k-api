"""Grid search with anti-overfit selection.

- IS (first 21d) grid -> score = net / max(drawdown, $10) per combo
- plateau ranking: a combo's rank uses the MEDIAN score of its grid
  neighborhood (each param at most 1 level away) — picks plateaus, not spikes
- combos with too few IS trades are not eligible (S08/S09 lowered thresholds)
- OOS (last 9d) runs ONLY for the picked combo + the yaml-default baseline
- S11 runs two stages: trigger shape first, then tp/stop refinement
"""
import asyncio
import itertools
import json
import logging
import statistics
from multiprocessing import get_context

from . import data
from .runner import run_backtest

log = logging.getLogger("bt.opt")

S11 = "S11_CRASH_SCALP"

# trimmed 2026-07-18 (S02/S06/S11), extended with the new HF candidates
GRIDS = {
    "S02_LIQ_REBOUND": {
        "liq_spike_mult": [4, 6, 8],
        "drop_atr": [2.0, 2.5, 3.0],
        "tp_retrace": [0.4, 0.5, 0.6],
    },
    "S06_CVD_FADE": {
        "div_gap_pct": [10, 15, 20],
    },
    "S14_VWAP_REVERT": {
        "dev_pct": [1.0, 1.5, 2.0],
        "taker_conf": [0.95, 99.0],
        "exit_frac": [0.3, 0.5],
    },
}
# drop_pct floor raised to 1.5%: sub-1.5% "crashes" on minor coins are routine
# noise, not liquidation cascades, and the candidate-window scan is sized to
# this floor (两级窗口条件对 1.5% 网格下限零漏)
S11_STAGE1 = {"drop_pct": [1.5, 2.0, 2.5],
              "window_sec": [20, 30, 45],
              "decel_ratio": [0.3, 0.4, 0.5]}
S11_STAGE2 = {"tp_pct": [0.4, 0.6, 0.8],
              "stop_pct": [0.6, 0.8, 1.0]}

MIN_TRADES = {"S02_LIQ_REBOUND": 8}
DEFAULT_MIN_TRADES = 15

ORDER = ["S02_LIQ_REBOUND", "S06_CVD_FADE", S11, "S14_VWAP_REVERT"]

RAW_DIR = data.DATA_DIR.parent / "reports" / "raw"


def _link(sid, p):
    """Derived params kept in a sane relation to the searched ones."""
    if sid == "S01_FUNDING_FADE":
        p["funding_long_th"] = round(-0.6 * p["funding_short_th"], 6)
        p["lsr_long_th"] = round(1.65 / p["lsr_short_th"], 3)
    return p


def expand(sid, grid):
    return [_link(sid, dict(zip(grid, vals)))
            for vals in itertools.product(*grid.values())]


def score_of(m):
    return m["net_usd"] / max(m["max_dd_usd"], 10.0)


# -- multiprocessing workers -------------------------------------------------

_DS = None


def _init_worker(symbols):
    global _DS
    from .sim import Dataset
    _DS = Dataset(symbols)


def _eval_one(args):
    sid, params, t0, t1 = args
    res = asyncio.run(run_backtest(sid, params, _DS, t0=t0, t1=t1))
    by_sym = {}
    for t in res.trades:
        d = by_sym.setdefault(t["symbol"], {"pnl": 0.0, "n": 0})
        d["pnl"] = round(d["pnl"] + t["pnl"], 2)
        d["n"] += 1
    return params, res.metrics(), by_sym


def evaluate_grid(sid, combos, t0, t1, symbols, workers=6):
    jobs = [(sid, p, t0, t1) for p in combos]
    if workers <= 1:
        _init_worker(symbols)
        return [_eval_one(j) for j in jobs]
    ctx = get_context("spawn")
    out = []
    with ctx.Pool(workers, initializer=_init_worker, initargs=(symbols,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_one, jobs)):
            out.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                log.info("%s grid %d/%d", sid, i + 1, len(jobs))
    return out


# -- plateau ranking ---------------------------------------------------------

def plateau_rank(grid, results):
    names = list(grid)

    def ivec(p):
        return tuple(grid[n].index(p[n]) for n in names)

    pts = {ivec(p): score_of(m) for p, m, _ in results}
    ranked = []
    for p, m, bs in results:
        iv = ivec(p)
        neigh = [pts[tuple(i + d for i, d in zip(iv, delta))]
                 for delta in itertools.product((-1, 0, 1), repeat=len(names))
                 if tuple(i + d for i, d in zip(iv, delta)) in pts]
        ranked.append({"params": p, "metrics": m, "by_symbol": bs,
                       "score": round(pts[iv], 3),
                       "plateau": round(statistics.median(neigh), 3)})
    ranked.sort(key=lambda r: (r["plateau"], r["metrics"]["sharpe"]),
                reverse=True)
    return ranked


def _pick(sid, ranked):
    need = MIN_TRADES.get(sid, DEFAULT_MIN_TRADES)
    eligible = [r for r in ranked if r["metrics"]["trades"] >= need]
    if eligible:
        return eligible[0], False
    return ranked[0], True         # insufficient sample — flagged, not trusted


async def _single(sid, params, ds, t0, t1):
    res = await run_backtest(sid, params, ds, t0=t0, t1=t1)
    return res


# -- search ------------------------------------------------------------------

def search(sid, symbols=None, workers=6):
    from .sim import Dataset
    symbols = symbols or data.valid_symbols()
    t0, is_end = data.T0, data.IS_END

    if sid == S11:
        r1 = evaluate_grid(sid, expand(sid, S11_STAGE1), t0, is_end, symbols, workers)
        rank1 = plateau_rank(S11_STAGE1, r1)
        base, insufficient1 = _pick(sid, rank1)
        combos2 = [{**base["params"], **dict(zip(S11_STAGE2, vals))}
                   for vals in itertools.product(*S11_STAGE2.values())]
        r2 = evaluate_grid(sid, combos2, t0, is_end, symbols, workers)
        rank2 = plateau_rank(S11_STAGE2, r2)
        picked, insufficient = _pick(sid, rank2)
        ranked = rank2 + rank1
        grid_desc = {"stage1": S11_STAGE1, "stage2": S11_STAGE2}
        insufficient = insufficient or insufficient1
    else:
        grid = GRIDS[sid]
        results = evaluate_grid(sid, expand(sid, grid), t0, is_end, symbols, workers)
        ranked = plateau_rank(grid, results)
        picked, insufficient = _pick(sid, ranked)
        grid_desc = grid

    # baseline + OOS runs (cheap, sequential in-process)
    ds = Dataset(symbols)

    async def _tail():
        d_is = await _single(sid, {}, ds, t0, is_end)
        d_oos = await _single(sid, {}, ds, is_end, data.T1)
        p_oos = await _single(sid, picked["params"], ds, is_end, data.T1)
        p_full = await _single(sid, picked["params"], ds, t0, data.T1)
        return d_is, d_oos, p_oos, p_full

    d_is, d_oos, p_oos, p_full = asyncio.run(_tail())

    by_sym = picked.get("by_symbol") or {}
    active = [s for s, v in by_sym.items() if v["n"] > 0]
    positive = [s for s in active if by_sym[s]["pnl"] > 0]
    consistency = round(len(positive) / len(active) * 100, 0) if active else 0.0

    is_net = picked["metrics"]["net_usd"]
    oos_m = p_oos.metrics()
    decay = None
    if is_net > 0:
        is_daily = is_net / 21
        oos_daily = oos_m["net_usd"] / 9
        decay = round((1 - oos_daily / is_daily) * 100, 1) if is_daily else None

    out = {
        "sid": sid,
        "window": {"t0": t0, "is_end": is_end, "t1": data.T1,
                   "data_end": ds.t_end},
        "grid": grid_desc,
        "symbols": symbols,
        "insufficient_sample": insufficient,
        "min_trades_required": MIN_TRADES.get(sid, DEFAULT_MIN_TRADES),
        "picked": {"params": picked["params"], "is": picked["metrics"],
                   "oos": oos_m, "score": picked["score"],
                   "plateau": picked["plateau"], "by_symbol": by_sym},
        "default": {"is": d_is.metrics(), "oos": d_oos.metrics()},
        "full_window": {"metrics": p_full.metrics(),
                        "equity": p_full.equity[::15],
                        "trades": p_full.trades},
        "consistency_pct": consistency,
        "oos_decay_pct": decay,
        "ranked": [{k: r[k] for k in ("params", "metrics", "score", "plateau")}
                   for r in ranked[:50]],
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{sid}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    log.info("%s done: picked=%s IS=%s OOS=%s", sid, picked["params"],
             picked["metrics"]["net_usd"], oos_m["net_usd"])
    return out


def search_all(sids=None, workers=6):
    out = {}
    for sid in (sids or ORDER):
        try:
            out[sid] = search(sid, workers=workers)
        except Exception:  # noqa: BLE001
            log.exception("search %s failed", sid)
    return out

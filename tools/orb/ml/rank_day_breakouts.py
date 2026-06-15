#!/usr/bin/env python3
"""指定交易日：从标的池选 Top-K，目标 ≥min_hits 个真突破。

默认 mode=pick6：当日各 symbol 首次突破 → 过滤早突破陷阱 → 按 P(true) 取 Top-6。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.core.backtest import _daily_df_asof, _iter_scan_ms, _resolve_open, _SimOpen  # noqa: E402
from orb.ml.samples import parse_symbol_list  # noqa: E402
from orb.core.config import OrbConfig  # noqa: E402
from orb.ml.ranker import BreakoutRanker  # noqa: E402
from orb.ml.features import (
    extract_features,
    label_is_true_breakout,
)
from orb.core.kline_cache import has_kline_cache, load_klines  # noqa: E402
from orb.core.paper import analyze_at_ms, in_regular_session, is_actionable  # noqa: E402
from orb.core.session import extended_fetch_anchor_ms  # noqa: E402
from orb.core.session import session_anchor_ms, session_close_ms, session_day_str  # noqa: E402

DEFAULT_PICK_K = 6
DEFAULT_MIN_HITS = 4
EARLY_TRAP_MINUTES = 20.0
EARLY_TRAP_MIN_SYNC = 3
EARLY_TRAP_MAX_SYNC = 14  # 仅过滤早+中等sync；mass sync(15+) 保留
DAY_GATE_MIN_EST = 5.0  # eligible 池 sum(P true) 低于此值 → skip 日


def _ml_cfg() -> OrbConfig:
    os.environ["ORB_MACRO_FILTER"] = "0"
    cfg = OrbConfig.from_env()
    cfg.macro_filter = False
    cfg.max_open_positions = 999
    return cfg


def _cached_symbols(symbols: List[str]) -> List[str]:
    return [s for s in symbols if has_kline_cache(s, "5m")]


def _day_scan_range(session_date: str, cfg: OrbConfig) -> Tuple[int, int, List[int]]:
    tz = cfg.session_tz
    open_time = cfg.session_open_time
    ts = pd.Timestamp(f"{session_date} 12:00:00", tz=tz)
    anchor = session_anchor_ms(int(ts.value // 1_000_000), tz=tz, session_open_time=open_time)
    close = session_close_ms(anchor, tz=tz, session_close_time=cfg.session_close_time)
    if close is None:
        close = anchor + 6 * 60 * 60 * 1000
    bar = cfg.bar_step_ms()
    scans = [s for s in _iter_scan_ms(anchor, close, bar_step_ms=bar) if session_day_str(s, tz=tz, session_open_time=open_time) == session_date]
    return anchor, close, scans


def _load_day_data(
    session_date: str,
    symbols: List[str],
    cfg: OrbConfig,
) -> Tuple[int, int, List[int], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    anchor, close, scans = _day_scan_range(session_date, cfg)
    warmup = cfg.daily_atr_warmup_ms() + cfg.bar_step_ms() * 96
    fetch_start = extended_fetch_anchor_ms(anchor, cfg) - warmup
    end_ms = close + cfg.bar_step_ms() * 4

    dfs5: Dict[str, pd.DataFrame] = {}
    dfs1: Dict[str, pd.DataFrame] = {}
    dfs_daily: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        dfs5[sym] = load_klines(sym, cfg.signal_interval, start_ms=fetch_start, end_ms=end_ms)
        dfs1[sym] = load_klines(sym, "1m", start_ms=fetch_start, end_ms=end_ms)
        if (cfg.sl_mode or "").strip().lower() == "atr_pct":
            dfs_daily[sym] = load_klines(sym, "1d", start_ms=fetch_start - cfg.daily_atr_warmup_ms(), end_ms=end_ms)
    return anchor, close, scans, dfs5, dfs1, dfs_daily


def _resolve_candidate(
    sym: str,
    sig: Any,
    *,
    session_date: str,
    scan_ms: int,
    close: int,
    cfg: OrbConfig,
    dfs1: Dict[str, pd.DataFrame],
) -> Optional[Dict[str, Any]]:
    entry_bo = int(sig.entry_bar_open_ms or 0)
    if entry_bo <= 0:
        return None
    df1 = dfs1.get(sym)
    if df1 is None or df1.empty:
        return None
    pos = _SimOpen(
        symbol=sym,
        side=str(sig.side),
        play=str(sig.play),
        entry=float(sig.price),
        sl=float(sig.sl_price),
        tp=float(sig.tp_price) if sig.tp_price is not None else None,
        entry_bar_open_ms=entry_bo,
        notional=float(sig.paper_notional_usdt or cfg.default_paper_notional()),
        session_date=session_date,
        scan_open_ms=int(scan_ms),
    )
    out, ex_px, note, exit_bo = _resolve_open(pos, df1, scan_ms=close + cfg.bar_step_ms(), cfg=cfg)
    if out is None or out == "supersede":
        return None
    from orb.core.resolve import pnl_usdt

    pnl = float(pnl_usdt(pos.side, pos.entry, ex_px, pos.notional))
    return {
        "session_date": session_date,
        "symbol": sym,
        "side": pos.side,
        "scan_open_ms": int(scan_ms),
        "entry": pos.entry,
        "sl": pos.sl,
        "exit_price": ex_px,
        "outcome": out,
        "pnl_usdt": round(pnl, 4),
        "true_breakout": label_is_true_breakout(out, pnl),
        "sig": sig,
    }


def evaluate_day_scans(session_date: str, symbols: List[str], cfg: OrbConfig) -> List[Dict[str, Any]]:
    """按 scan 收集同日 actionable 候选（每 symbol 仅首次突破）。"""
    _, close, scans, dfs5, dfs1, dfs_daily = _load_day_data(session_date, symbols, cfg)
    if not scans:
        return []

    session_traded: Dict[str, bool] = {}
    scan_events: List[Dict[str, Any]] = []

    for scan_ms in scans:
        if not in_regular_session(cfg, now_ms=scan_ms):
            continue
        candidates: List[Tuple[str, Any]] = []
        for sym in symbols:
            if session_traded.get(sym):
                continue
            df5 = dfs5.get(sym)
            if df5 is None or df5.empty:
                continue
            ddf = _daily_df_asof(dfs_daily.get(sym, pd.DataFrame()), scan_ms)
            sig = analyze_at_ms(
                sym,
                cfg=cfg,
                now_ms=scan_ms,
                session_traded=False,
                daily_df=ddf if not ddf.empty else None,
                bot_equity_usdt=cfg.per_symbol_bot_equity(),
                df5=df5,
            )
            if not is_actionable(sig, cfg):
                continue
            candidates.append((sym, sig))

        if not candidates:
            continue

        sync_by_sym: Dict[str, int] = {}
        for sym, sig in candidates:
            side = str(sig.side)
            sync_by_sym[sym] = sum(1 for s2, g2 in candidates if s2 != sym and str(g2.side) == side)

        rows: List[Dict[str, Any]] = []
        for sym, sig in candidates:
            if session_traded.get(sym):
                continue
            sync_n = int(sync_by_sym.get(sym, 0))
            base = _resolve_candidate(sym, sig, session_date=session_date, scan_ms=scan_ms, close=close, cfg=cfg, dfs1=dfs1)
            if base is None:
                continue
            session_traded[sym] = True
            feat = extract_features(sig, cfg, sync_same_side=sync_n)
            rows.append({**base, "sync_same_side": sync_n, "features": feat})

        if rows:
            max_sync = max(int(r["sync_same_side"]) for r in rows)
            scan_events.append(
                {
                    "scan_open_ms": int(scan_ms),
                    "n_candidates": len(rows),
                    "max_sync": max_sync,
                    "rows": rows,
                }
            )
    return scan_events


def pick_mass_scan(
    scan_events: List[Dict[str, Any]],
    *,
    min_candidates: int = 3,
    prefer_sync: int = 5,
) -> Optional[Dict[str, Any]]:
    """取当日第一个「足够多候选」的 scan（同刻横截面），而非 sync 最大的 scan。"""
    if not scan_events:
        return None
    ordered = sorted(scan_events, key=lambda s: int(s["scan_open_ms"]))
    eligible = [s for s in ordered if int(s["n_candidates"]) >= min_candidates]
    if not eligible:
        return ordered[0]
    if prefer_sync > 0:
        for s in eligible:
            if int(s["max_sync"]) >= prefer_sync:
                return s
    return eligible[0]


def day_risk_from_scan(
    scan: Optional[Dict[str, Any]],
    *,
    fake_m,
    true_m,
    sync_skip: int = 12,
    fake_median_skip: float = 0.88,
    rank_only: bool = True,
) -> Dict[str, Any]:
    """日级风险：高 sync + 模型普遍判假 → 建议 skip（不参与 symbol 排序特征）。"""
    if scan is None:
        return {"skip_day": False, "reason": "no_scan", "max_sync": 0, "median_p_fake": 0.0}
    rows = list(scan.get("rows") or [])
    max_sync = int(scan.get("max_sync") or 0)
    p_fakes = [fake_m.predict_proba(r["features"], symbol=str(r["symbol"]), rank_only=rank_only) for r in rows]
    med_fake = float(sorted(p_fakes)[len(p_fakes) // 2]) if p_fakes else 0.0
    skip = max_sync >= sync_skip and med_fake >= fake_median_skip
    return {
        "skip_day": skip,
        "reason": f"sync>={sync_skip}_med_p_fake>={fake_median_skip}" if skip else "ok",
        "max_sync": max_sync,
        "median_p_fake": round(med_fake, 4),
        "n_candidates": len(rows),
    }


def evaluate_day(session_date: str, symbols: List[str], cfg: OrbConfig) -> List[Dict[str, Any]]:
    """Legacy: flatten all first-breakouts (不同 scan 混在一起排序)。"""
    out: List[Dict[str, Any]] = []
    for ev in evaluate_day_scans(session_date, symbols, cfg):
        for r in ev["rows"]:
            out.append({k: v for k, v in r.items() if k != "sig"})
    return out


def evaluate_day_cross_section(
    session_date: str,
    symbols: List[str],
    cfg: OrbConfig,
    *,
    min_candidates: int = 3,
    prefer_sync: int = 5,
) -> Dict[str, Any]:
    scans = evaluate_day_scans(session_date, symbols, cfg)
    pick = pick_mass_scan(scans, min_candidates=min_candidates, prefer_sync=prefer_sync)
    pick_rows = [{k: v for k, v in r.items() if k != "sig"} for r in (pick.get("rows") or [])] if pick else []
    all_rows = [{k: v for k, v in r.items() if k != "sig"} for ev in scans for r in ev["rows"]]
    return {
        "session_date": session_date,
        "n_scans_with_breakouts": len(scans),
        "pick_scan_ms": int(pick["scan_open_ms"]) if pick else None,
        "pick_n": len(pick_rows),
        "pick_max_sync": int(pick["max_sync"]) if pick else 0,
        "pick_rows": pick_rows,
        "all_rows": all_rows,
        "scans": [
            {
                "scan_open_ms": ev["scan_open_ms"],
                "n_candidates": ev["n_candidates"],
                "max_sync": ev["max_sync"],
            }
            for ev in scans
        ],
    }


def _model_rank_only(ranker: BreakoutRanker) -> bool:
    return True


def rank_rows(
    rows: List[Dict[str, Any]],
    ranker: BreakoutRanker,
    *,
    rank_only: bool = True,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        feat = dict(r["features"])
        sym = str(r["symbol"])
        p_true = ranker.predict_true(feat, symbol=sym)
        p_fake = ranker.predict_fake(feat, symbol=sym)
        out.append({**r, "p_fake": p_fake, "p_true": p_true, "p_model": round(p_true, 4)})
    return sorted(out, key=lambda x: x["p_true"], reverse=True)


def day_gate(ranked: List[Dict[str, Any]], *, min_hits: int = DEFAULT_MIN_HITS, min_est: float = DAY_GATE_MIN_EST) -> Dict[str, Any]:
    est = round(sum(float(r.get("p_true") or 0) for r in ranked), 2)
    need = max(float(min_hits), float(min_est))
    skip = est < need
    return {
        "skip_day": skip,
        "reason": f"est_true={est}<{need}" if skip else "ok",
        "est_true_sum": est,
        "eligible_n": len(ranked),
    }


def is_early_sync_trap(
    row: Dict[str, Any],
    *,
    early_minutes: float = EARLY_TRAP_MINUTES,
    min_sync: int = EARLY_TRAP_MIN_SYNC,
    max_sync: int = EARLY_TRAP_MAX_SYNC,
) -> bool:
    """早突破 + 中等 sync(3~14)：假突破高发；mass sync(15+) 不在此列。"""
    feat = row.get("features") or {}
    mins = float(feat.get("minutes_after_or", 0) or 0)
    sync = int(row.get("sync_same_side", feat.get("sync_same_side", 0)) or 0)
    return mins < early_minutes and min_sync <= sync <= max_sync


def filter_pick_pool(
    rows: List[Dict[str, Any]],
    *,
    early_minutes: float = EARLY_TRAP_MINUTES,
    min_sync: int = EARLY_TRAP_MIN_SYNC,
    max_sync: int = EARLY_TRAP_MAX_SYNC,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eligible: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for r in rows:
        if is_early_sync_trap(r, early_minutes=early_minutes, min_sync=min_sync, max_sync=max_sync):
            excluded.append(r)
        else:
            eligible.append(r)
    return eligible, excluded


def summarize_pick(
    ranked: List[Dict[str, Any]],
    *,
    k: int = DEFAULT_PICK_K,
    min_hits: int = DEFAULT_MIN_HITS,
    pool_n: int = 0,
    true_in_pool: int = 0,
    excluded_n: int = 0,
) -> Dict[str, Any]:
    pick = ranked[:k]
    hits = sum(1 for r in pick if r["true_breakout"])
    pick_syms = {str(r["symbol"]) for r in pick}
    missed = [
        {
            "symbol": str(r["symbol"]).replace("USDT", ""),
            "p_true": r["p_true"],
            "pnl_usdt": r["pnl_usdt"],
            "rank": i + 1,
            "sync": int(r["sync_same_side"]),
        }
        for i, r in enumerate(ranked)
        if r["true_breakout"] and r["symbol"] not in pick_syms
    ]
    return {
        "pick_k": k,
        "min_hits": min_hits,
        "pool_n": pool_n or len(ranked),
        "true_in_pool": true_in_pool or sum(1 for r in ranked if r["true_breakout"]),
        "excluded_n": excluded_n,
        "pick_hits": hits,
        "pick_hit_rate": round(hits / k, 3) if k else 0,
        "goal_met": hits >= min_hits,
        "picked": [
            {
                "rank": i + 1,
                "symbol": str(r["symbol"]).replace("USDT", ""),
                "side": r["side"],
                "p_true": r["p_true"],
                "p_fake": r["p_fake"],
                "true": bool(r["true_breakout"]),
                "pnl_usdt": r["pnl_usdt"],
                "sync": int(r["sync_same_side"]),
                "minutes_after_or": round(float((r.get("features") or {}).get("minutes_after_or", 0) or 0), 1),
            }
            for i, r in enumerate(pick)
        ],
        "missed_true": missed,
    }


def evaluate_day_pick6(
    session_date: str,
    symbols: List[str],
    cfg: OrbConfig,
    ranker: BreakoutRanker,
    *,
    k: int = DEFAULT_PICK_K,
    min_hits: int = DEFAULT_MIN_HITS,
    early_minutes: float = EARLY_TRAP_MINUTES,
    min_sync: int = EARLY_TRAP_MIN_SYNC,
    max_sync: int = EARLY_TRAP_MAX_SYNC,
    min_est: float = DAY_GATE_MIN_EST,
) -> Dict[str, Any]:
    """某日：全池首次突破 → 过滤早突破陷阱 → 日级 gate → Top-K。"""
    raw = evaluate_day(session_date, symbols, cfg)
    ranked_all = rank_rows(raw, ranker)
    eligible, excluded = filter_pick_pool(
        ranked_all, early_minutes=early_minutes, min_sync=min_sync, max_sync=max_sync
    )
    ranked = eligible
    gate = day_gate(ranked, min_hits=min_hits, min_est=min_est)
    pick_summary = summarize_pick(
        ranked,
        k=k,
        min_hits=min_hits,
        pool_n=len(ranked_all),
        true_in_pool=sum(1 for r in ranked_all if r["true_breakout"]),
        excluded_n=len(excluded),
    )
    if gate.get("skip_day"):
        pick_summary["goal_met"] = False
        pick_summary["skipped_by_gate"] = True
    else:
        pick_summary["skipped_by_gate"] = False
    return {
        "session_date": session_date,
        "breakouts_that_day": len(ranked_all),
        "eligible_n": len(eligible),
        "day_gate": gate,
        "ranker_kind": ranker.kind,
        "excluded": [
            {
                "symbol": str(r["symbol"]).replace("USDT", ""),
                "p_true": r.get("p_true"),
                "sync": int(r["sync_same_side"]),
                "minutes_after_or": round(float((r.get("features") or {}).get("minutes_after_or", 0) or 0), 1),
                "true": bool(r["true_breakout"]),
            }
            for r in excluded
        ],
        "ranked_all": ranked_all,
        "ranked": ranked,
        **pick_summary,
    }


def main() -> int:
    load_env_oi()
    ap = argparse.ArgumentParser(description="Pick top-K true breakouts for one day from symbol pool")
    ap.add_argument("--date", required=True, help="交易日 YYYY-MM-DD")
    ap.add_argument("--symbols-file", default=str(ROOT / "config" / "orb" / "v2" / "symbols.txt"))
    ap.add_argument(
        "--mode",
        choices=("pick6", "legacy", "cross"),
        default="pick6",
        help="pick6=某日Top-K(默认); legacy=不过滤; cross=同scan横截面(旧)",
    )
    ap.add_argument("--pick-k", type=int, default=DEFAULT_PICK_K)
    ap.add_argument("--min-hits", type=int, default=DEFAULT_MIN_HITS)
    ap.add_argument("--early-minutes", type=float, default=EARLY_TRAP_MINUTES)
    ap.add_argument("--early-min-sync", type=int, default=EARLY_TRAP_MIN_SYNC)
    ap.add_argument("--early-max-sync", type=int, default=EARLY_TRAP_MAX_SYNC)
    ap.add_argument("--min-candidates", type=int, default=3, help="cross 模式用")
    ap.add_argument("--prefer-sync", type=int, default=5, help="cross 模式用")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    syms = _cached_symbols(parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8")))
    cfg = _ml_cfg()
    ranker = BreakoutRanker.load(use_prior=True)
    if ranker.gbm is None and ranker.logistic is None:
        print("Missing model (run train_breakout_gbm.py or train_shared_breakout_model.py)")
        return 1

    k = max(1, int(args.pick_k))
    min_hits = max(0, int(args.min_hits))

    if args.mode == "pick6":
        result = evaluate_day_pick6(
            args.date.strip(),
            syms,
            cfg,
            ranker,
            k=k,
            min_hits=min_hits,
            early_minutes=float(args.early_minutes),
            min_sync=int(args.early_min_sync),
            max_sync=int(args.early_max_sync),
        )
        ranked = result["ranked"]
        gate = result.get("day_gate") or {}
        summary = {
            "date": args.date,
            "mode": "pick6",
            "ranker": ranker.kind,
            "symbols_cached": len(syms),
            "breakouts_that_day": result["breakouts_that_day"],
            "eligible_n": result["eligible_n"],
            "excluded_n": result["excluded_n"],
            "true_in_pool": result["true_in_pool"],
            "day_gate": gate,
            "skipped_by_gate": bool(result.get("skipped_by_gate")),
            "pick_k": k,
            "min_hits": min_hits,
            "pick_hits": result["pick_hits"],
            "goal_met": result["goal_met"] and not result.get("skipped_by_gate"),
            "early_filter": f"{args.early_minutes}min & sync={args.early_min_sync}-{args.early_max_sync}",
            "macro_filter": False,
        }
        display = result["picked"]
    elif args.mode == "cross":
        day = evaluate_day_cross_section(
            args.date.strip(),
            syms,
            cfg,
            min_candidates=max(1, int(args.min_candidates)),
            prefer_sync=max(0, int(args.prefer_sync)),
        )
        ranked = rank_rows(day["pick_rows"], ranker)
        summary = {
            "date": args.date,
            "mode": "cross",
            "ranker": ranker.kind,
            "symbols_cached": len(syms),
            "pick_scan_ms": day["pick_scan_ms"],
            "breakouts_that_day": len(day["all_rows"]),
        }
        display = [_brief_row(r, i + 1) for i, r in enumerate(ranked[:k])]
    else:
        raw = evaluate_day(args.date.strip(), syms, cfg)
        ranked = rank_rows(raw, ranker)
        pick = summarize_pick(ranked, k=k, min_hits=min_hits)
        summary = {
            "date": args.date,
            "mode": "legacy",
            "ranker": ranker.kind,
            "symbols_cached": len(syms),
            "breakouts_that_day": len(ranked),
            "pick_hits": pick["pick_hits"],
            "goal_met": pick["goal_met"],
        }
        display = pick["picked"]

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.mode == "pick6":
        gate_note = " SKIP" if summary.get("skipped_by_gate") else ""
        print(f"\n=== Top-{k} pick ({result['pick_hits']}/{k} true, goal {min_hits}+: {'OK' if summary['goal_met'] else 'MISS'}{gate_note}) ===")
        if gate.get("skip_day"):
            print(f"day_gate: {gate.get('reason')} (est sum P(true)={gate.get('est_true_sum')})")
        print(f"{'rk':>3} {'sym':<6} {'side':<5} {'P(true)':>7} {'sync':>4} {'min':>5} {'pnl':>8} {'label'}")
        print("-" * 52)
        for p in display:
            tag = "TRUE" if p["true"] else "FAKE"
            print(
                f"{p['rank']:3d} {p['symbol']:<6} {p['side']:<5} {p['p_true']:7.3f} "
                f"{p['sync']:4d} {p['minutes_after_or']:5.0f} {float(p['pnl_usdt']):+8.1f}U {tag}"
            )
        if result["excluded"]:
            ex = ", ".join(f"{x['symbol']}(sync={x['sync']},min={x['minutes_after_or']})" for x in result["excluded"][:8])
            print(f"\nfiltered early+sync ({result['excluded_n']}): {ex}")
        if result["missed_true"]:
            missed = ", ".join(f"{m['symbol']}@#{m['rank']}" for m in result["missed_true"][:6])
            print(f"missed true: {missed}")
    else:
        print(f"\n{'rank':>4} {'sym':<6} {'side':<5} {'P(true)':>7} {'P(fake)':>7} {'sync':>4} {'pnl':>8} {'label'}")
        print("-" * 58)
        for i, r in enumerate(ranked[:k], 1):
            sym = str(r["symbol"]).replace("USDT", "")
            tag = "TRUE" if r["true_breakout"] else "FAKE"
            print(
                f"{i:4d} {sym:<6} {r['side']:<5} {r['p_true']:7.3f} {r['p_fake']:7.3f} "
                f"{int(r['sync_same_side']):4d} {float(r['pnl_usdt']):+8.1f}U {tag}"
            )

    if args.json_out:
        payload = {"summary": summary, "ranked": ranked[:k] if args.mode != "pick6" else result}
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def _brief_row(r: dict, rank: int) -> dict:
    return {
        "rank": rank,
        "symbol": str(r["symbol"]).replace("USDT", ""),
        "side": r["side"],
        "p_true": r["p_true"],
        "true": bool(r["true_breakout"]),
        "pnl_usdt": r["pnl_usdt"],
        "sync": int(r["sync_same_side"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())

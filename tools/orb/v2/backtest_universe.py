#!/usr/bin/env python3
"""ORB V2 全池 live_gate 回测（43 标，scan-by-scan，按日输出明细）。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.core.config import OrbConfig  # noqa: E402
from orb.core.kline_cache import has_kline_cache, kline_path, symbol_cache_dir  # noqa: E402
from orb.core.macro_calendar import is_macro_skip_day, macro_events_for_day  # noqa: E402
from orb.ml.gate import LiveGateConfig  # noqa: E402
from orb.ml.model import BreakoutModelBundle  # noqa: E402
from orb.ml.ranker import BreakoutRanker  # noqa: E402
from orb.ml.samples import parse_symbol_list  # noqa: E402
from orb.v2.paths import resolve_gate_config_path, resolve_symbols_path  # noqa: E402

from tools.orb.ml.eval_live_gate import (  # noqa: E402
    _ml_cfg,
    init_robot_wallets,
    init_symbol_wallets,
    simulate_live_gate_day,
)
from tools.orb.v2.backtest_symbol import session_dates_from_cache  # noqa: E402


def universe_session_dates(symbols: List[str], cfg: OrbConfig) -> List[str]:
    dates: set[str] = set()
    for sym in symbols:
        dates.update(session_dates_from_cache(sym, cfg))
    return sorted(dates)


def cached_symbols(symbols: List[str]) -> tuple[List[str], List[str]]:
    ok, missing = [], []
    for sym in symbols:
        if has_kline_cache(sym, "5m"):
            ok.append(sym)
        else:
            missing.append(sym)
    return ok, missing


def _day_pnl(day: dict) -> float:
    total = 0.0
    for r in day.get("opened") or []:
        if r.get("pnl_usdt") is not None:
            total += float(r.get("pnl_usdt") or 0)
        elif r.get("wallet_before") is not None and r.get("wallet_after") is not None:
            total += float(r.get("wallet_after") or 0) - float(r.get("wallet_before") or 0)
    return round(total, 4)


def _trade_pnl(r: dict) -> float:
    if r.get("pnl_usdt") is not None:
        return float(r.get("pnl_usdt") or 0)
    if r.get("wallet_before") is not None and r.get("wallet_after") is not None:
        return float(r.get("wallet_after") or 0) - float(r.get("wallet_before") or 0)
    return 0.0


def _macro_counterfactual_days(
    dates: List[str],
    syms: List[str],
    cfg: OrbConfig,
    ranker: BreakoutRanker,
    gate: LiveGateConfig,
    *,
    robot_count: int,
    robot_equity: float,
) -> List[dict]:
    """宏观过滤日：关闭 macro 重跑，估算被挡掉的交易。"""
    cf_cfg = OrbConfig.from_env()
    cf_cfg.macro_filter = False
    cf_cfg.max_open_positions = cfg.max_open_positions
    cf_cfg.fixed_notional_usdt = cfg.fixed_notional_usdt

    blocked: List[dict] = []
    for d in dates:
        if not (cfg.macro_filter and is_macro_skip_day(d)):
            continue
        robots = init_robot_wallets(count=robot_count, equity_usdt=robot_equity)
        alt = simulate_live_gate_day(d, syms, cf_cfg, ranker, gate, robot_wallets=robots)
        alt_pnl = _day_pnl(alt)
        blocked.append(
            {
                "session_date": d,
                "macro_events": list(macro_events_for_day(d)),
                "actual_opens": 0,
                "counterfactual_opens": int(alt.get("opens") or 0),
                "counterfactual_pnl_usdt": alt_pnl,
                "counterfactual_trades": [
                    {
                        "symbol": r.get("symbol"),
                        "side": r.get("side"),
                        "entry": r.get("entry"),
                        "p_true": r.get("p_true"),
                        "pnl_usdt": r.get("pnl_usdt"),
                        "true_breakout": r.get("true_breakout"),
                        "scan_et": r.get("scan_et"),
                        "robot_id": r.get("robot_id"),
                        "outcome": r.get("outcome"),
                    }
                    for r in alt.get("opened") or []
                ],
            }
        )
    return blocked


def _print_daily_detail(days: List[dict]) -> None:
    print("\n=== 每日交易明细 ===\n")
    for day in days:
        sd = day["session_date"]
        pnl = _day_pnl(day)
        opens = day.get("opened") or []
        ok = "OK" if day.get("goal_met_min") else "—"
        macro_tag = ""
        if day.get("macro_skip_day"):
            ev = ",".join(day.get("macro_events") or []) or "macro"
            macro_tag = f"  [宏观过滤日: {ev}]"
        print(f"## {sd}  PnL={pnl:+.1f}U  opens={day.get('opens', 0)}  true={day.get('true_opens', 0)}  goal={ok}{macro_tag}")
        if day.get("macro_skip_day"):
            print("  (全日无开单 — 宏观事件日信号层拦截 macro_event_day)\n")
            continue
        if not opens:
            print("  (无开单)\n")
            continue
        for r in opens:
            sym = str(r.get("symbol", "")).replace("USDT", "")
            tb = "T" if r.get("true_breakout") else "F"
            notion = r.get("notional_usdt")
            n_txt = f" n={notion:.0f}U" if notion is not None else ""
            bot = f" R{r.get('robot_id')}" if r.get("robot_id") is not None else ""
            tp = _trade_pnl(r)
            print(
                f"  {sym:6}{bot} {str(r.get('side', '')):5} entry={r.get('entry')} "
                f"P={float(r.get('p_true', 0)):.3f} pnl={tp:+.1f}U{n_txt} "
                f"{tb} @ {r.get('scan_et', '')}  {r.get('outcome', r.get('reason', ''))}"
            )
        print()


def _print_macro_impact(macro_blocked: List[dict]) -> None:
    if not macro_blocked:
        return
    print("\n=== 宏观过滤影响（关闭 macro 的 counterfactual）===\n")
    total_missed_pnl = 0.0
    total_missed_opens = 0
    for row in macro_blocked:
        sd = row["session_date"]
        ev = ",".join(row.get("macro_events") or []) or "macro"
        cf_pnl = float(row.get("counterfactual_pnl_usdt") or 0)
        cf_opens = int(row.get("counterfactual_opens") or 0)
        total_missed_pnl += cf_pnl
        total_missed_opens += cf_opens
        print(f"## {sd}  [{ev}]  实际=0笔  若无宏观={cf_opens}笔  估算PnL={cf_pnl:+.1f}U")
        for r in row.get("counterfactual_trades") or []:
            sym = str(r.get("symbol", "")).replace("USDT", "")
            tb = "T" if r.get("true_breakout") else "F"
            print(
                f"  [被挡] {sym:6} R{r.get('robot_id')} {str(r.get('side', '')):5} "
                f"P={float(r.get('p_true', 0)):.3f} pnl={float(r.get('pnl_usdt') or 0):+.1f}U "
                f"{tb} @ {r.get('scan_et', '')}  {r.get('outcome', '')}"
            )
        print()
    print(
        f"宏观过滤日合计: {len(macro_blocked)} 天 | "
        f"挡掉约 {total_missed_opens} 笔 | 估算 missed PnL={total_missed_pnl:+.1f}U\n"
    )


def _write_trades_csv(days: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "session_date",
        "macro_skip_day",
        "macro_events",
        "robot_id",
        "symbol",
        "side",
        "entry",
        "notional_usdt",
        "wallet_before",
        "wallet_after",
        "p_true",
        "p_fake",
        "pnl_usdt",
        "true_breakout",
        "scan_et",
        "scan_open_ms",
        "minutes_after_or",
        "sync_same_side",
        "outcome",
        "exit_ms",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for day in days:
            macro_skip = bool(day.get("macro_skip_day"))
            macro_ev = ",".join(day.get("macro_events") or [])
            for r in day.get("opened") or []:
                row = dict(r)
                row["session_date"] = day["session_date"]
                row["macro_skip_day"] = macro_skip
                row["macro_events"] = macro_ev
                if row.get("pnl_usdt") is None:
                    row["pnl_usdt"] = round(_trade_pnl(row), 4)
                w.writerow(row)


def _write_macro_blocked_csv(macro_blocked: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "session_date",
        "macro_events",
        "symbol",
        "robot_id",
        "side",
        "entry",
        "p_true",
        "pnl_usdt",
        "true_breakout",
        "scan_et",
        "outcome",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in macro_blocked:
            sd = row["session_date"]
            ev = ",".join(row.get("macro_events") or [])
            trades = row.get("counterfactual_trades") or []
            if not trades:
                w.writerow(
                    {
                        "session_date": sd,
                        "macro_events": ev,
                        "note": "macro_skip_day_no_counterfactual_opens",
                    }
                )
                continue
            for t in trades:
                w.writerow(
                    {
                        "session_date": sd,
                        "macro_events": ev,
                        "symbol": t.get("symbol"),
                        "robot_id": t.get("robot_id"),
                        "side": t.get("side"),
                        "entry": t.get("entry"),
                        "p_true": t.get("p_true"),
                        "pnl_usdt": t.get("pnl_usdt"),
                        "true_breakout": t.get("true_breakout"),
                        "scan_et": t.get("scan_et"),
                        "outcome": t.get("outcome"),
                        "note": "blocked_by_macro_filter",
                    }
                )


def _wallet_summary(wallets: Dict[str, float], initial: float) -> dict:
    vals = list(wallets.values())
    total = round(sum(vals), 2)
    init_total = round(initial * len(wallets), 2)
    pnl = round(total - init_total, 2)
    return {
        "initial_bot_equity_usdt": initial,
        "initial_total_equity_usdt": init_total,
        "final_total_equity_usdt": total,
        "total_pnl_usdt": pnl,
        "return_pct": round(100.0 * pnl / init_total, 2) if init_total else 0.0,
        "depleted_bots": sum(1 for v in vals if v <= 0),
        "max_wallet_usdt": round(max(vals), 2) if vals else 0,
        "min_wallet_usdt": round(min(vals), 2) if vals else 0,
    }


def _robot_summary(wallets: List[float], initial: float) -> dict:
    total = round(sum(wallets), 2)
    init_total = round(initial * len(wallets), 2)
    pnl = round(total - init_total, 2)
    return {
        "robot_count": len(wallets),
        "initial_robot_equity_usdt": initial,
        "initial_total_equity_usdt": init_total,
        "final_total_equity_usdt": total,
        "total_pnl_usdt": pnl,
        "return_pct": round(100.0 * pnl / init_total, 2) if init_total else 0.0,
        "depleted_robots": sum(1 for v in wallets if v <= 0),
        "max_robot_wallet_usdt": round(max(wallets), 2) if wallets else 0,
        "min_robot_wallet_usdt": round(min(wallets), 2) if wallets else 0,
        "robots": {f"R{i + 1}": round(w, 2) for i, w in enumerate(wallets)},
    }


def main() -> int:
    load_env_oi()
    ap = argparse.ArgumentParser(description="ORB V2 universe live_gate backtest")
    ap.add_argument("--symbols-file", default=str(resolve_symbols_path()))
    ap.add_argument("--last-sessions", type=int, default=60, help="最近 N 个 NYSE 交易日")
    ap.add_argument("--from-date", default="")
    ap.add_argument("--to-date", default="")
    ap.add_argument("--gate-config", default=str(resolve_gate_config_path()))
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument(
        "--sizing",
        choices=("eight_robots", "per_symbol", "fixed"),
        default="eight_robots",
        help="eight_robots=8槽复利 | per_symbol=每标bot | fixed=固定名义",
    )
    ap.add_argument("--robot-count", type=int, default=8)
    ap.add_argument("--robot-equity", type=float, default=10_000.0)
    ap.add_argument("--fixed-notional", type=float, default=1000.0, help="仅 sizing=fixed")
    ap.add_argument(
        "--no-live-filters",
        action="store_true",
        help="关闭 env 宏观过滤（旧回测口径）",
    )
    ap.add_argument("--json-out", default="")
    ap.add_argument("--csv-out", default="")
    ap.add_argument("--quiet-detail", action="store_true")
    args = ap.parse_args()

    all_syms = parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8"))
    syms, missing = cached_symbols(all_syms)
    if not syms:
        print("No cached symbols. Run fetch_orb_kline_cache.py first.")
        return 1

    sizing = args.sizing.strip().lower()
    live_filters = not bool(args.no_live_filters)
    if sizing == "fixed":
        cfg = _ml_cfg(
            compound_per_symbol=False,
            fixed_notional=float(args.fixed_notional),
            respect_env_filters=live_filters,
        )
    else:
        cfg = _ml_cfg(
            compound_per_symbol=True,
            fixed_notional=0.0,
            respect_env_filters=live_filters,
        )

    gate = LiveGateConfig.from_json(Path(args.gate_config))
    if args.min_p is not None:
        gate.min_p_true = float(args.min_p)

    model = BreakoutModelBundle.load_production()
    if not model.is_ready:
        print("ML model missing — run tools/orb/v2/monthly_train.py --bootstrap-only")
        return 1
    ranker = model.ranker

    all_dates = universe_session_dates(syms, cfg)
    if not all_dates:
        print("No session dates in cache")
        return 1

    d0 = args.from_date.strip() or ""
    d1 = args.to_date.strip() or ""
    if d0 or d1:
        lo = d0 or all_dates[0]
        hi = d1 or all_dates[-1]
        dates = [d for d in all_dates if lo <= d <= hi]
    else:
        dates = all_dates[-max(1, int(args.last_sessions)) :]

    if not dates:
        print("Empty date range")
        return 1

    if sizing == "eight_robots":
        sizing_desc = (
            f"8-robot x {args.robot_equity:.0f}U risk={cfg.risk_pct} "
            f"(p-ranked; {'reuse after SL/TP' if gate.robot_reuse_after_exit else '1 trade/robot/day'})"
        )
    elif sizing == "per_symbol":
        sizing_desc = f"per_symbol compound bot={cfg.per_symbol_bot_equity()}U risk={cfg.risk_pct}"
    else:
        sizing_desc = f"fixed {cfg.fixed_notional_usdt}U/trade"

    print(
        f"[v2 universe] {len(syms)}/{len(all_syms)} syms | "
        f"{dates[0]} .. {dates[-1]} | {len(dates)} NYSE | "
        f"gate p>={gate.min_p_true} max={gate.max_opens_per_day} | {sizing_desc} | "
        f"filters macro={cfg.macro_filter}",
        flush=True,
    )
    if missing:
        print(f"missing cache ({len(missing)}): {', '.join(s.replace('USDT', '') for s in missing)}", flush=True)

    symbol_wallets: Optional[Dict[str, float]] = None
    robot_wallets: Optional[List[float]] = None
    if sizing == "per_symbol":
        symbol_wallets = init_symbol_wallets(syms, cfg)
    elif sizing == "eight_robots":
        robot_wallets = init_robot_wallets(count=int(args.robot_count), equity_usdt=float(args.robot_equity))

    t0 = time.time()
    days = []
    for i, d in enumerate(dates, 1):
        print(f"[{i}/{len(dates)}] {d} ...", flush=True)
        days.append(
            simulate_live_gate_day(
                d,
                syms,
                cfg,
                ranker,
                gate,
                wallets=symbol_wallets,
                robot_wallets=robot_wallets,
            )
        )

    pnls = [_day_pnl(d) for d in days]
    total_pnl = round(sum(pnls), 2)
    opens_total = sum(int(d.get("opens") or 0) for d in days)
    true_total = sum(int(d.get("true_opens") or 0) for d in days)
    macro_skip_days = [d["session_date"] for d in days if d.get("macro_skip_day")]
    macro_blocked: List[dict] = []
    if live_filters and cfg.macro_filter and macro_skip_days:
        print(f"\nmacro counterfactual for {len(macro_skip_days)} skip days ...", flush=True)
        macro_blocked = _macro_counterfactual_days(
            macro_skip_days,
            syms,
            cfg,
            ranker,
            gate,
            robot_count=int(args.robot_count),
            robot_equity=float(args.robot_equity),
        )

    init_total = 0.0
    if sizing == "eight_robots" and robot_wallets is not None:
        init_total = float(args.robot_equity) * len(robot_wallets)
    elif sizing == "per_symbol" and symbol_wallets is not None:
        init_total = float(cfg.per_symbol_bot_equity()) * len(symbol_wallets)

    out = {
        "kind": "orb_v2_universe_backtest",
        "sizing": sizing,
        "symbols_file": str(args.symbols_file),
        "symbols_cached": syms,
        "symbols_missing_cache": missing,
        "date_range": {"from": dates[0], "to": dates[-1], "sessions": len(dates)},
        "gate": gate.__dict__,
        "ranker": ranker.kind,
        "risk_pct": cfg.risk_pct,
        "fixed_notional_usdt": cfg.fixed_notional_usdt,
        "summary": {
            "total_pnl_usdt": total_pnl,
            "return_pct": round(100.0 * total_pnl / init_total, 2) if init_total else None,
            "initial_total_equity_usdt": round(init_total, 2) if init_total else None,
            "avg_daily_pnl_usdt": round(total_pnl / len(days), 2) if days else 0,
            "win_days": sum(1 for p in pnls if p > 0),
            "loss_days": sum(1 for p in pnls if p < 0),
            "flat_days": sum(1 for p in pnls if p == 0),
            "total_opens": opens_total,
            "total_true_opens": true_total,
            "avg_opens_per_day": round(opens_total / len(days), 2) if days else 0,
            "goal_min_hit_days": sum(1 for d in days if d.get("goal_met_min")),
            "macro_skip_days": macro_skip_days,
            "macro_skip_day_count": len(macro_skip_days),
            "macro_blocked_opens_est": sum(int(x.get("counterfactual_opens") or 0) for x in macro_blocked),
            "macro_blocked_pnl_est_usdt": round(
                sum(float(x.get("counterfactual_pnl_usdt") or 0) for x in macro_blocked), 2
            ),
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "macro_impact": macro_blocked,
        "days": days,
    }
    if sizing == "per_symbol" and symbol_wallets is not None:
        out["wallets"] = {k: round(v, 2) for k, v in sorted(symbol_wallets.items())}
        out["wallet_summary"] = _wallet_summary(symbol_wallets, float(cfg.per_symbol_bot_equity()))
    if sizing == "eight_robots" and robot_wallets is not None:
        out["robot_summary"] = _robot_summary(robot_wallets, float(args.robot_equity))

    tag = sizing.replace("_", "-")
    json_out = (
        Path(args.json_out)
        if args.json_out.strip()
        else ROOT / "output" / "orb" / "v2" / "eval" / f"universe_{len(dates)}d_{tag}_backtest.json"
    )
    csv_out = Path(args.csv_out) if args.csv_out.strip() else json_out.with_suffix(".trades.csv")
    macro_csv = json_out.with_suffix(".macro_blocked.csv")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_trades_csv(days, csv_out)
    if macro_blocked:
        _write_macro_blocked_csv(macro_blocked, macro_csv)

    print(json.dumps(out["summary"], indent=2, ensure_ascii=False))
    if out.get("robot_summary"):
        print(json.dumps(out["robot_summary"], indent=2, ensure_ascii=False))
    elif out.get("wallet_summary"):
        print(json.dumps(out["wallet_summary"], indent=2, ensure_ascii=False))
    _print_macro_impact(macro_blocked)
    if not args.quiet_detail:
        _print_daily_detail(days)
    print(f"\njson -> {json_out}")
    print(f"csv  -> {csv_out}")
    if macro_blocked:
        print(f"macro blocked -> {macro_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

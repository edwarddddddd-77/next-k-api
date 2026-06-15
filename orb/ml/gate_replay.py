"""Live Gate timeline 重放与 min_p 扫描。"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from orb.ml.gate import LiveGateConfig, is_early_trap


def replay_day(
    timeline: list[dict],
    *,
    min_p: float,
    max_opens: int,
    gate: LiveGateConfig | None = None,
    early_minutes: float | None = None,
    sync_min: int | None = None,
    sync_max: int | None = None,
    day_abort: bool | None = None,
    abort_after: int | None = None,
    abort_med_max: float | None = None,
) -> dict:
    g = gate or LiveGateConfig()
    em = float(early_minutes if early_minutes is not None else g.early_trap_minutes)
    smin = int(sync_min if sync_min is not None else g.early_trap_sync_min)
    smax = int(sync_max if sync_max is not None else g.early_trap_sync_max)
    dab = bool(g.day_abort_enabled if day_abort is None else day_abort)
    aafter = int(abort_after if abort_after is not None else g.day_abort_after_signals)
    amed = float(abort_med_max if abort_med_max is not None else g.day_abort_median_p_max)

    opens: list[dict] = []
    recent_p: list[float] = []
    aborted = False

    for ev in timeline:
        p = float(ev.get("p_true") or 0)
        sync = int(ev.get("sync_same_side") or 0)
        mins = float(ev.get("minutes_after_or") or 0)
        recent_p.append(p)

        if dab and not aborted and len(recent_p) == aafter and not opens:
            med = sorted(recent_p)[len(recent_p) // 2]
            if med < amed:
                aborted = True

        if aborted:
            continue
        if len(opens) >= max_opens:
            continue
        if is_early_trap(
            {"minutes_after_or": mins},
            sync,
            early_minutes=em,
            sync_min=smin,
            sync_max=smax,
        ):
            continue
        if p < min_p:
            continue
        if ev.get("pnl_usdt") is None:
            continue
        opens.append(ev)

    pnl = sum(float(o.get("pnl_usdt") or 0) for o in opens)
    true_n = sum(1 for o in opens if o.get("true_breakout"))
    return {
        "opens": len(opens),
        "true_n": true_n,
        "pnl": round(pnl, 2),
        "aborted": aborted,
    }


def replay_sessions(
    days: Sequence[dict],
    *,
    min_p: float,
    gate: LiveGateConfig,
) -> Dict[str, Any]:
    max_o = int(gate.max_opens_per_day)
    total_pnl = 0.0
    opens_total = 0
    win = loss = 0
    for d in days:
        r = replay_day(d.get("timeline") or [], min_p=min_p, max_opens=max_o, gate=gate)
        total_pnl += r["pnl"]
        opens_total += r["opens"]
        if r["pnl"] > 0:
            win += 1
        elif r["pnl"] < 0:
            loss += 1
    n = max(len(days), 1)
    return {
        "min_p": round(float(min_p), 4),
        "total_pnl": round(total_pnl, 1),
        "avg_daily_pnl": round(total_pnl / n, 1),
        "avg_opens": round(opens_total / n, 2),
        "win_days": win,
        "loss_days": loss,
        "days": len(days),
    }


def sweep_min_p_grid(
    days: Sequence[dict],
    *,
    gate: LiveGateConfig,
    grid: Sequence[float],
) -> List[Dict[str, Any]]:
    rows = [replay_sessions(days, min_p=float(p), gate=gate) for p in grid]
    rows.sort(key=lambda x: x["total_pnl"], reverse=True)
    return rows


def timeline_score_stats(days: Sequence[dict]) -> Dict[str, float]:
    ps: list[float] = []
    for d in days:
        for ev in d.get("timeline") or []:
            if ev.get("p_true") is not None:
                ps.append(float(ev["p_true"]))
    if not ps:
        return {"median_p": 0.0, "mean_p": 0.0, "n": 0.0}
    ps.sort()
    return {
        "median_p": round(ps[len(ps) // 2], 4),
        "mean_p": round(sum(ps) / len(ps), 4),
        "n": float(len(ps)),
    }

#!/usr/bin/env python3
"""
RSI+ADX 1H 轮换策略 — 强多/强空各最多 5 个。

规则：
- 每 1H 扫描市值 Top50（除稳定币）
- 出列表 → 平仓
- 已有持仓后：同步至候选 Top5（不足 5 个则扩仓，超过则只持候选内）
- 平仓后若无后补候选 → 不开仓，空位保留
- 冷启动（从未持仓）→ 不主动建仓，除非 RSI_ADX_BOOTSTRAP=1
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from accumulation_radar import init_db

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

# tools/ on path for rsi_adx_core
_TOOLS = Path(__file__).resolve().parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from rsi_adx_core import ScanRow, scan_universe  # noqa: E402

db_dir = os.getenv("DATA_DIR", str(Path(__file__).parent))
STATE_PATH = Path(db_dir) / "rsi_adx_rotation_state.json"
SNAPSHOT_PATH = Path(db_dir) / "rsi_adx_rotation_snapshot.json"

MAX_SLOTS = max(1, int(os.getenv("RSI_ADX_MAX_SLOTS", "5").strip() or "5"))
TOP_N = max(10, int(os.getenv("RSI_ADX_TOP_N", "50").strip() or "50"))
INTERVAL = os.getenv("RSI_ADX_INTERVAL", "1h").strip() or "1h"
LIVE_ENABLED = os.getenv("RSI_ADX_LIVE_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
LEG_NOTIONAL_USD = float(os.getenv("RSI_ADX_LEG_NOTIONAL_USD", "100").strip() or "100")
LEVERAGE = max(1, int(os.getenv("RSI_ADX_LEVERAGE", "3").strip() or "3"))
RECORD_SIGNALS = os.getenv("RSI_ADX_RECORD_SIGNALS", "1").strip().lower() in ("1", "true", "yes", "on")
BOOTSTRAP = os.getenv("RSI_ADX_BOOTSTRAP", "0").strip().lower() in ("1", "true", "yes", "on")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_label() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.is_file():
        return {"long_slots": [], "short_slots": [], "version": 1}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("long_slots", [])
            data.setdefault("short_slots", [])
            return data
    except Exception:
        pass
    return {"long_slots": [], "short_slots": [], "version": 1}


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def _slot_symbols(slots: List[Dict[str, Any]]) -> List[str]:
    return [str(s.get("symbol") or "") for s in slots if s.get("symbol")]


def _row_to_slot(row: ScanRow) -> Dict[str, Any]:
    return {
        "symbol": row.symbol,
        "coin": row.coin,
        "score": round(float(row.score), 4),
        "adx": round(float(row.adx), 2),
        "plus_di": round(float(row.plus_di), 2),
        "minus_di": round(float(row.minus_di), 2),
        "rsi": round(float(row.rsi), 2),
        "price": float(row.price),
        "opened_at_utc": _utc_label(),
    }


def _pick_targets(
    bulls: List[ScanRow],
    bears: List[ScanRow],
    max_slots: int,
) -> Tuple[List[ScanRow], List[ScanRow]]:
    bulls = sorted(bulls, key=lambda r: r.score, reverse=True)
    bears = sorted(bears, key=lambda r: r.score, reverse=True)

    target_long: List[ScanRow] = []
    for row in bulls:
        target_long.append(row)
        if len(target_long) >= max_slots:
            break

    long_syms = {r.symbol for r in target_long}
    target_short: List[ScanRow] = []
    for row in bears:
        if row.symbol in long_syms:
            continue
        target_short.append(row)
        if len(target_short) >= max_slots:
            break

    return target_long, target_short


def compute_rotation(
    prev_state: Dict[str, Any],
    scan_rows: List[ScanRow],
    *,
    max_slots: int = MAX_SLOTS,
    bootstrap: Optional[bool] = None,
) -> Dict[str, Any]:
    bulls = [r for r in scan_rows if r.filter_key == "strong_bull"]
    bears = [r for r in scan_rows if r.filter_key == "strong_bear"]
    target_long, target_short = _pick_targets(bulls, bears, max_slots)
    row_by_sym = {r.symbol: r for r in scan_rows}

    prev_long = _slot_symbols(prev_state.get("long_slots") or [])
    prev_short = _slot_symbols(prev_state.get("short_slots") or [])
    target_long_syms = {r.symbol for r in target_long}
    target_short_syms = {r.symbol for r in target_short}

    old_long = {s["symbol"]: s for s in (prev_state.get("long_slots") or []) if s.get("symbol")}
    old_short = {s["symbol"]: s for s in (prev_state.get("short_slots") or []) if s.get("symbol")}

    actions: List[Dict[str, Any]] = []

    close_long_syms = [sym for sym in prev_long if sym not in target_long_syms]
    close_short_syms = [sym for sym in prev_short if sym not in target_short_syms]

    for sym in close_long_syms:
        actions.append(
            {
                "action": "close",
                "side": "LONG",
                "symbol": sym,
                "reason": "out_of_bull_list",
            }
        )
    for sym in close_short_syms:
        actions.append(
            {
                "action": "close",
                "side": "SHORT",
                "symbol": sym,
                "reason": "out_of_bear_list",
            }
        )

    stayed_long_syms = [sym for sym in prev_long if sym in target_long_syms]
    stayed_short_syms = [sym for sym in prev_short if sym in target_short_syms]

    ever_held = bool(prev_state.get("ever_held")) or bool(prev_long) or bool(prev_short)
    boot = BOOTSTRAP if bootstrap is None else bootstrap
    allow_opens = ever_held or boot

    # 同步至目标列表：在 target 内且未持有的 → 开仓（含 3→5 扩仓）
    # 冷启动且未开 BOOTSTRAP → 不开首仓
    open_candidates_long = [r for r in target_long if r.symbol not in stayed_long_syms]
    open_candidates_short = [r for r in target_short if r.symbol not in stayed_short_syms]
    opens_long = open_candidates_long if allow_opens else []
    opens_short = open_candidates_short if allow_opens else []

    # 平仓后无后补：目标列表因候选不足变短时，不开“假后补”
    # （opens 已限制在 target 内，候选不够则 target 本身更短，自然少开）

    for row in opens_long:
        actions.append(
            {
                "action": "open",
                "side": "LONG",
                "symbol": row.symbol,
                "coin": row.coin,
                "score": row.score,
                "reason": "sync_bull_target",
            }
        )
    for row in opens_short:
        actions.append(
            {
                "action": "open",
                "side": "SHORT",
                "symbol": row.symbol,
                "coin": row.coin,
                "score": row.score,
                "reason": "sync_bear_target",
            }
        )

    def _build_slots(
        target_rows: List[ScanRow],
        opened_rows: List[ScanRow],
        stayed_syms: List[str],
        old_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """持仓 = 目标 Top 列表（仍在候选内）。"""
        slots: List[Dict[str, Any]] = []
        for row in target_rows:
            sym = row.symbol
            if sym in stayed_syms and sym in old_map:
                slots.append(dict(old_map[sym]))
            elif sym in stayed_syms or any(r.symbol == sym for r in opened_rows):
                slots.append(_row_to_slot(row))
        return slots[:max_slots]

    new_long_slots = _build_slots(target_long, opens_long, stayed_long_syms, old_long)
    new_short_slots = _build_slots(target_short, opens_short, stayed_short_syms, old_short)

    return {
        "target_long": [asdict(r) for r in target_long],
        "target_short": [asdict(r) for r in target_short],
        "long_slots": new_long_slots,
        "short_slots": new_short_slots,
        "actions": actions,
        "candidate_bulls": len(bulls),
        "candidate_bears": len(bears),
        "ever_held": ever_held or bool(new_long_slots) or bool(new_short_slots),
        "allow_opens": allow_opens,
        "skipped_opens_cold_start": {
            "long": len(open_candidates_long) if not allow_opens else 0,
            "short": len(open_candidates_short) if not allow_opens else 0,
        },
    }


def _record_signal(action: Dict[str, Any], *, price: Optional[float] = None) -> None:
    if not RECORD_SIGNALS:
        return
    try:
        from orb.vnpy.strategy_signals import LANE_RSI_ADX_ROTATION, record_strategy_signal

        record_strategy_signal(
            lane=LANE_RSI_ADX_ROTATION,
            symbol=action["symbol"],
            side=action["side"],
            action=action["action"],
            entry_price=price,
            status="executed" if action.get("executed") else "emitted",
            skip_reason=action.get("error"),
            detail={
                "reason": action.get("reason"),
                "score": action.get("score"),
                "strategy": "rsi_adx_rotation",
                "live": LIVE_ENABLED,
            },
        )
    except Exception as e:
        print(f"  ⚠️ 信号写入失败 {action.get('symbol')}: {e}")


def _execute_actions(actions: List[Dict[str, Any]], scan_by_sym: Dict[str, ScanRow]) -> None:
    if not LIVE_ENABLED:
        for act in actions:
            row = scan_by_sym.get(act["symbol"])
            _record_signal(act, price=float(row.price) if row else None)
        return

    try:
        from orb.vnpy.binance_account import (
            ensure_one_way_mode,
            fetch_position_amounts,
            set_symbol_leverage,
            set_symbol_margin_isolated,
        )
    except Exception as e:
        print(f"  ❌ 实盘模块加载失败，跳过下单: {e}")
        for act in actions:
            act["error"] = str(e)
            _record_signal(act)
        return

    try:
        ensure_one_way_mode()
    except Exception as e:
        print(f"  ⚠️ one-way 模式: {e}")

    symbols = list({a["symbol"] for a in actions})
    positions = fetch_position_amounts(symbols)

    for act in actions:
        sym = act["symbol"]
        side = act["side"]
        row = scan_by_sym.get(sym)
        price = float(row.price) if row else None
        try:
            set_symbol_margin_isolated(sym)
            set_symbol_leverage(sym, LEVERAGE)
            if act["action"] == "close":
                _market_close(sym, side, positions.get(sym, 0.0))
            else:
                _market_open(sym, side, price or 0.0)
            act["executed"] = True
            print(f"  ✅ {act['action'].upper()} {side} {sym}")
        except Exception as e:
            act["error"] = str(e)
            print(f"  ❌ {act['action']} {side} {sym}: {e}")
        _record_signal(act, price=price)


def _signed_post(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    from orb.vnpy.binance_account import _signed_post as _sp

    return _sp(path, params)


def _market_open(symbol: str, side: str, price: float) -> None:
    if price <= 0 or LEG_NOTIONAL_USD <= 0:
        raise ValueError("invalid price or notional")
    qty = _round_qty(symbol, LEG_NOTIONAL_USD / price)
    order_side = "BUY" if side == "LONG" else "SELL"
    _signed_post(
        "/fapi/v1/order",
        {
            "symbol": symbol,
            "side": order_side,
            "type": "MARKET",
            "quantity": qty,
        },
    )


def _market_close(symbol: str, side: str, position_amt: float) -> None:
    amt = float(position_amt or 0.0)
    if side == "LONG":
        if amt <= 0:
            return
        _signed_post(
            "/fapi/v1/order",
            {
                "symbol": symbol,
                "side": "SELL",
                "type": "MARKET",
                "quantity": _round_qty(symbol, abs(amt)),
                "reduceOnly": "true",
            },
        )
    else:
        if amt >= 0:
            return
        _signed_post(
            "/fapi/v1/order",
            {
                "symbol": symbol,
                "side": "BUY",
                "type": "MARKET",
                "quantity": _round_qty(symbol, abs(amt)),
                "reduceOnly": "true",
            },
        )


def _round_qty(symbol: str, qty: float) -> str:
    """粗精度：3 位小数；小价币 0 位。"""
    if qty <= 0:
        raise ValueError("qty must be positive")
    if qty >= 100:
        return f"{qty:.3f}".rstrip("0").rstrip(".")
    if qty >= 1:
        return f"{qty:.2f}".rstrip("0").rstrip(".")
    return f"{max(qty, 1e-8):.0f}" if qty > 1000 else f"{qty:.3f}".rstrip("0").rstrip(".")


def _persist_snapshot(payload: Dict[str, Any]) -> None:
    tmp = SNAPSHOT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(SNAPSHOT_PATH)
    print(f"  💾 快照 → {SNAPSHOT_PATH}")


def run_rotation_scan(*, notify: bool = False) -> Dict[str, Any]:
    del notify
    run_id = uuid.uuid4().hex[:12]
    print(f"🔄 RSI+ADX 1H 轮换扫描 [{run_id}]")
    print(f"   Top{TOP_N} | 每侧最多 {MAX_SLOTS} | live={'ON' if LIVE_ENABLED else 'OFF'}")

    prev_state = load_state()
    scan_rows, errors = scan_universe(interval=INTERVAL, top_n=TOP_N)
    if not scan_rows:
        return {"ok": False, "error": "scan_empty", "errors": errors}

    rotation = compute_rotation(prev_state, scan_rows, max_slots=MAX_SLOTS)
    scan_by_sym = {r.symbol: r for r in scan_rows}

    closes = [a for a in rotation["actions"] if a["action"] == "close"]
    opens = [a for a in rotation["actions"] if a["action"] == "open"]
    print(f"   候选 强多 {rotation['candidate_bulls']} / 强空 {rotation['candidate_bears']}")
    print(f"   动作 平仓 {len(closes)} / 同步开仓 {len(opens)}")

    # 先平后开
    ordered_actions = closes + opens
    _execute_actions(ordered_actions, scan_by_sym)

    new_state = {
        "version": 1,
        "last_scan_at_utc": _utc_label(),
        "long_slots": rotation["long_slots"],
        "short_slots": rotation["short_slots"],
        "ever_held": rotation.get("ever_held", False),
        "last_run_id": run_id,
    }
    save_state(new_state)

    payload: Dict[str, Any] = {
        "ok": True,
        "run_id": run_id,
        "generated_at_utc": _utc_label(),
        "interval": INTERVAL,
        "top_n": TOP_N,
        "max_slots": MAX_SLOTS,
        "live_enabled": LIVE_ENABLED,
        "long_list": rotation["long_slots"],
        "short_list": rotation["short_slots"],
        "target_long": rotation["target_long"],
        "target_short": rotation["target_short"],
        "actions": ordered_actions,
        "candidate_bulls": rotation["candidate_bulls"],
        "candidate_bears": rotation["candidate_bears"],
        "skipped_opens_cold_start": rotation.get("skipped_opens_cold_start"),
        "ever_held": rotation.get("ever_held"),
        "errors": errors,
        "config": {
            "leg_notional_usd": LEG_NOTIONAL_USD,
            "leverage": LEVERAGE,
        },
    }
    _persist_snapshot(payload)

    print("\n📈 当前持有多仓:")
    for s in rotation["long_slots"]:
        print(f"   {s['coin']:<8} ADX={s['adx']} score={s['score']}")
    if not rotation["long_slots"]:
        print("   (无)")
    print("📉 当前持有空仓:")
    for s in rotation["short_slots"]:
        print(f"   {s['coin']:<8} ADX={s['adx']} score={s['score']}")
    if not rotation["short_slots"]:
        print("   (无)")
    print("\n🎯 本周期目标 Top5:")
    for side, key in (("强多", "target_long"), ("强空", "target_short")):
        items = rotation.get(key) or []
        coins = ", ".join(x.get("coin", "?") for x in items[:MAX_SLOTS]) or "(无)"
        print(f"   {side}: {coins}")

    return payload


def main() -> None:
    conn = init_db()
    try:
        payload = run_rotation_scan()
        if not payload.get("ok"):
            print(f"❌ {payload.get('error')}")
            sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

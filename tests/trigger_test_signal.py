"""模拟触发开仓信号测试脚本。

绕过 signal_bridge，直接调用 trader.execute_trade() 测试完整开仓流程。

─────────────────────────────────────────────────────────────
模式
─────────────────────────────────────────────────────────────
  --dry-run (默认)  只做前置检查 + 展示预估参数，不写 DB，不交易
  --live            写 signals_log → 调 execute_trade() → 下单

─────────────────────────────────────────────────────────────
用法
─────────────────────────────────────────────────────────────
  cd next-k-api

  # 安全检查（不开仓）
  python tests/trigger_test_signal.py

  # 自定义参数安全检查
  python tests/trigger_test_signal.py --symbol ETHUSDT --side SHORT --entry 3200 --sl 3250 --tp 3100

  # 真正开仓（LONG 示例）
  python tests/trigger_test_signal.py \
    --symbol BTCUSDT --side LONG \
    --entry 95000 --sl 94000 --tp 98000 \
    --live

  # 真正开仓（SHORT 示例）
  python tests/trigger_test_signal.py \
    --symbol ETHUSDT --side SHORT \
    --entry 3200 --sl 3250 --tp 3100 \
    --notional 200 --leverage 5 \
    --live

─────────────────────────────────────────────────────────────
参数
─────────────────────────────────────────────────────────────
  --symbol    交易对，默认 BTCUSDT
  --side      LONG 或 SHORT
  --entry     入场价
  --sl        止损价（LONG 须 < entry，SHORT 须 > entry）
  --tp        止盈价（LONG 须 > entry，SHORT 须 < entry）
  --notional  保证金 USDT，默认 100
  --leverage  杠杆倍数（仅 dry-run 展示用，实际读取 binance.db 配置）
  --live      真正执行开仓

─────────────────────────────────────────────────────────────
前置条件（--live 模式）
─────────────────────────────────────────────────────────────
  1. binance.db 中 enabled = true
  2. enabled_sources 包含 zct_vwap
  3. 同 symbol 无开仓中的 position
  4. binance.db 中 margin_usdt / leverage 配置正确
  5. BINANCE_API_KEY / BINANCE_API_SECRET 已配置（env 或 DB）
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

# 与 binance_bridge/db.py 保持一致：默认 DATA_DIR = binance_bridge/ 目录
_DB_MODULE_DIR = API_DIR / "binance_bridge"
DATA_DIR = Path(os.getenv("DATA_DIR", _DB_MODULE_DIR))
BINANCE_DB_PATH = DATA_DIR / "binance.db"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_prerequisites() -> list[str]:
    issues = []
    if not BINANCE_DB_PATH.exists():
        issues.append(f"binance.db 不存在: {BINANCE_DB_PATH}")
        return issues

    conn = sqlite3.connect(str(BINANCE_DB_PATH))
    conn.row_factory = sqlite3.Row

    row = conn.execute("SELECT value FROM config WHERE key='enabled'").fetchone()
    if not row or row["value"].lower() != "true":
        issues.append("交易未启用 (enabled != true)")

    row = conn.execute("SELECT value FROM config WHERE key='enabled_sources'").fetchone()
    enabled_sources = (row["value"] if row else "").split(",")
    if "zct_vwap" not in [s.strip() for s in enabled_sources]:
        issues.append("zct_vwap 不在 enabled_sources 中")

    conn.close()
    return issues


def run_dry_run(args):
    print("[DRY RUN] 模拟触发开仓信号")
    print(f"  symbol={args.symbol}  side={args.side}")
    print(f"  entry={args.entry}  sl={args.sl}  tp={args.tp}")
    print(f"  notional={args.notional}  leverage={args.leverage}")
    print()

    issues = check_prerequisites()
    if issues:
        print("前置检查失败 (开仓会被阻止):")
        for i in issues:
            print(f"  ! {i}")
    else:
        print("前置检查通过")

    qty = args.notional * args.leverage / args.entry
    print(f"  预估: margin={args.notional}U leverage={args.leverage}x "
          f"notional={args.notional * args.leverage}U qty≈{qty:.4f}")


def run_live(args):
    print("[LIVE] 直接调用 trader.execute_trade()")
    print(f"  symbol={args.symbol}  side={args.side}")
    print(f"  entry={args.entry}  sl={args.sl}  tp={args.tp}")
    print()

    issues = check_prerequisites()
    if issues:
        print("前置检查失败:")
        for i in issues:
            print(f"  ! {i}")
        sys.exit(1)

    # 检查同 symbol 持仓
    conn = sqlite3.connect(str(BINANCE_DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM positions WHERE symbol=? AND status='open'",
        (args.symbol,),
    ).fetchone()
    if row and row["cnt"] > 0:
        print(f"! {args.symbol} 已有开仓，跳过")
        conn.close()
        sys.exit(1)
    conn.close()

    # 构造信号，手动插入 signals_log
    from binance_bridge import db as _db
    from binance_bridge.trader import execute_trade

    sig_id = _db.insert_signal(
        source="MANUAL_TEST",
        api_signal_id=f"test_{now_utc()}",
        symbol=args.symbol,
        side=args.side,
        entry_price=args.entry,
        sl_price=args.sl,
        tp_price=args.tp,
        confidence=None,
        regime=None,
        notional_usdt=args.notional,
        received_at=now_utc(),
    )
    if sig_id is None:
        print("! insert_signal 返回 None (duplicate?)")
        sys.exit(1)

    print(f"  1. signals_log 写入 (id={sig_id})")

    ok = execute_trade({
        "signal_log_id": sig_id,
        "symbol": args.symbol,
        "side": args.side,
        "sl_price": args.sl,
        "tp_price": args.tp,
        "notional_usdt": args.notional,
    })

    if ok:
        print(f"  2. execute_trade 成功")
    else:
        print(f"  2. execute_trade 失败 (查看 signals_log status)")
        sig_rows = _db.list_signals(limit=1)
        if sig_rows:
            s = sig_rows[0]
            print(f"     status={s['status']} skip_reason={s.get('skip_reason', '')}")


def main():
    parser = argparse.ArgumentParser(description="模拟触发开仓信号")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--side", default="LONG", choices=["LONG", "SHORT"])
    parser.add_argument("--entry", type=float, default=95000.0)
    parser.add_argument("--sl", type=float, default=94000.0)
    parser.add_argument("--tp", type=float, default=98000.0)
    parser.add_argument("--notional", type=float, default=100.0,
                        help="保证金 USDT (default: 100)")
    parser.add_argument("--leverage", type=int, default=10,
                        help="杠杆倍数 (仅 dry-run 展示用)")
    parser.add_argument("--live", action="store_true",
                        help="真正开仓。绕过 signal_bridge，直接调 execute_trade()")
    args = parser.parse_args()

    # 校验 SL/TP 方向
    if args.side == "LONG":
        if args.sl >= args.entry or args.tp <= args.entry:
            print(f"! LONG: 需要 sl({args.sl}) < entry({args.entry}) < tp({args.tp})")
            sys.exit(1)
    else:
        if args.sl <= args.entry or args.tp >= args.entry:
            print(f"! SHORT: 需要 tp({args.tp}) < entry({args.entry}) < sl({args.sl})")
            sys.exit(1)

    if args.live:
        run_live(args)
    else:
        run_dry_run(args)


if __name__ == "__main__":
    main()

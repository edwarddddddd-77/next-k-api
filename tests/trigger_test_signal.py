"""模拟触发开仓信号测试脚本。

通过 HTTP 调用 Next-k-protocol 的 /api/binance/signals/ingest 端点触发开仓。

─────────────────────────────────────────────────────────────
模式
─────────────────────────────────────────────────────────────
  --dry-run (默认)  只做前置检查 + 展示预估参数，不发送信号
  --live            发送信号到 Next-k-protocol 触发开仓

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
  --leverage  杠杆倍数（仅 dry-run 展示用）
  --live      发送信号到 Next-k-protocol 触发开仓

─────────────────────────────────────────────────────────────
前置条件（--live 模式）
─────────────────────────────────────────────────────────────
  1. Next-k-protocol 服务已启动
  2. binance.db 中 enabled = true
  3. enabled_sources 包含 zct_vwap
  4. 同 symbol 无开仓中的 position
  5. BINANCE_API_KEY / BINANCE_API_SECRET 已配置（env 或 DB）
  6. PROTOCOL_API_URL 和 PROTOCOL_MAINTENANCE_TOKEN 已设置
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

_PROTOCOL_DIR = API_DIR.parent / "Next-k-protocol"
DATA_DIR = Path(os.getenv("DATA_DIR", _PROTOCOL_DIR))
BINANCE_DB_PATH = DATA_DIR / "binance.db"

PROTOCOL_API_URL = os.getenv("PROTOCOL_API_URL", "http://localhost:8001").rstrip("/")
PROTOCOL_MAINTENANCE_TOKEN = os.getenv("PROTOCOL_MAINTENANCE_TOKEN", "")


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
    print("[LIVE] 通过 HTTP POST 发送信号到 Next-k-protocol")
    print(f"  url={PROTOCOL_API_URL}/api/binance/signals/ingest")
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

    signal = {
        "source": "MANUAL_TEST",
        "api_signal_id": f"test_{now_utc()}",
        "symbol": args.symbol,
        "side": args.side,
        "entry_price": args.entry,
        "sl_price": args.sl,
        "tp_price": args.tp,
        "confidence": None,
        "regime": None,
        "margin_usdt": args.notional,
    }

    body = json.dumps({"signals": [signal]}).encode("utf-8")
    url = f"{PROTOCOL_API_URL}/api/binance/signals/ingest"
    headers = {"Content-Type": "application/json"}
    if PROTOCOL_MAINTENANCE_TOKEN:
        headers["X-Maintenance-Token"] = PROTOCOL_MAINTENANCE_TOKEN

    try:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        print(f"  响应: scanned={result.get('scanned')} traded={result.get('traded')} "
              f"skipped={result.get('skipped')} errors={result.get('errors')}")
        for d in result.get("details", []):
            print(f"    {d.get('api_signal_id')}: {d.get('action')}")
    except Exception as e:
        print(f"! HTTP 请求失败: {e}")
        sys.exit(1)


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

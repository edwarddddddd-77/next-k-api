"""Trade journal: one JSONL entry per closed trade with the full context an
AI needs to optimize strategies later — entry/exit factor snapshots, MFE/MAE,
signal reason, hold time. Files: journal/trades-YYYY-MM-DD.jsonl (UTC days).
"""
import json
import logging
import time
from pathlib import Path

from . import config

log = logging.getLogger("journal")

JOURNAL_DIR = config.ROOT / "journal"

_FIELDS_DOC = """# 交易日记(机器可读,供 AI 策略优化)

每行一个 JSON(JSONL),按 UTC 日期分文件:`trades-YYYY-MM-DD.jsonl`。

| 字段 | 含义 |
|---|---|
| closed_at / opened_at | 平仓/开仓时间(UTC ISO) |
| strategy / symbol / side | 策略ID / 交易对 / long·short |
| qty / entry_price / close_price | 数量 / 开仓价 / 平仓价 |
| pnl / pnl_pct | 已实现盈亏(USDT,含手续费)/ 相对开仓名义的百分比 |
| hold_minutes | 持仓时长(分钟) |
| signal_reason | 开仓时的信号描述(入场依据) |
| close_reason | 平仓原因(止损/止盈/移动止损/时限/策略出场/手动) |
| mfe_pct / mae_pct | 持仓期间最大有利偏移 / 最大不利偏移(%,0.5s 采样)——校准止盈止损的核心数据 |
| entry_factors / exit_factors | 开仓/平仓时刻的关键因子快照(费率、多空比、OI变化、taker、爆仓倍数、基差、恐贪等) |
| mode | paper / live |

AI 优化建议用法:按 strategy 分组,对比盈利/亏损样本的 entry_factors 分布差异;
用 MFE/MAE 分布评估当前止盈止损倍数是否留利润/砍太早。
"""


def _ensure_dir():
    JOURNAL_DIR.mkdir(exist_ok=True)
    readme = JOURNAL_DIR / "README.md"
    if not readme.exists():
        readme.write_text(_FIELDS_DOC, encoding="utf-8")


def _iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)) if ts else None


def record_close(pos, close_price, pnl, fee, close_reason, exit_factors, mode):
    """Append one journal line for a closed position. Never raises."""
    try:
        _ensure_dir()
        notional = abs(pos["entry_price"] * pos["qty"]) or 1
        entry_factors = None
        if pos.get("entry_factors"):
            try:
                entry_factors = json.loads(pos["entry_factors"])
            except (TypeError, ValueError):
                entry_factors = None
        entry = {
            "closed_at": _iso(time.time()),
            "opened_at": _iso(pos["opened_at"]),
            "strategy": pos["strategy"], "symbol": pos["symbol"], "side": pos["side"],
            "qty": pos["qty"], "entry_price": pos["entry_price"],
            "close_price": close_price,
            "pnl": round(pnl, 4), "pnl_pct": round(pnl / notional * 100, 4),
            "fee": round(fee, 4),
            "hold_minutes": round((time.time() - pos["opened_at"]) / 60, 1),
            "signal_reason": pos.get("reason") or "",
            "close_reason": close_reason,
            "mfe_pct": round(pos.get("mfe_pct") or 0, 4),
            "mae_pct": round(pos.get("mae_pct") or 0, 4),
            "entry_factors": entry_factors,
            "exit_factors": exit_factors,
            "mode": mode,
        }
        day = time.strftime("%Y-%m-%d", time.gmtime())
        path = JOURNAL_DIR / f"trades-{day}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 - journaling must never break trading
        log.warning("journal write failed: %s", exc)

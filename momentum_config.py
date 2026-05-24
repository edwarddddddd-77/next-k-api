"""动量多一空一 — topMovers 纸面仓位 · env 参数。"""

from __future__ import annotations

import os


def env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


MOM_SCHEDULER_ENABLED = env_truthy("MOM_SCHEDULER_ENABLED", default=True)
MOM_SCAN_INTERVAL_MINUTES = max(
    1, int(os.getenv("MOM_SCAN_INTERVAL_MINUTES", "15") or 15)
)

# 纸面名义 = 权益 × 杠杆系数（与原脚本 LEVERAGE 语义一致）
MOM_ACCOUNT_EQUITY_USDT = max(
    1.0, float(os.getenv("MOM_ACCOUNT_EQUITY_USDT", "10000") or 10000)
)
MOM_LEVERAGE = max(0.0, float(os.getenv("MOM_LEVERAGE", "0.1") or 0.1))
MOM_NOTIONAL_USDT = MOM_ACCOUNT_EQUITY_USDT * MOM_LEVERAGE

MOM_LONG_EVENT = (os.getenv("MOM_LONG_EVENT", "PULLBACK") or "PULLBACK").strip()
MOM_SHORT_EVENT = (os.getenv("MOM_SHORT_EVENT", "RALLY") or "RALLY").strip()

_raw_bl = (os.getenv("MOM_BLACKLIST", "XNY") or "XNY").strip()
MOM_BLACKLIST = tuple(
    x.strip().upper() for x in _raw_bl.split(",") if x.strip()
)

MOM_ALLOW_USDC = os.getenv("MOM_ALLOW_USDC", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

MOM_COOLDOWN_SEC = max(0, int(os.getenv("MOM_COOLDOWN_SEC", "10") or 10))

MOM_TG_NOTIFY = os.getenv("MOM_TG_NOTIFY", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def momentum_scheduler_enabled() -> bool:
    return MOM_SCHEDULER_ENABLED

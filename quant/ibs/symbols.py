"""IBS 标的解析 — 现货 tokenized ETF 与永续映射。"""

from __future__ import annotations

from quant.common.kline_cache import norm_symbol

# Bitget 现货 Ondo / Reality token（2026-07 在线）
IBS_SPOT_BASE_MAP: dict[str, str] = {
    "SPY": "SPYONUSDT",
    "QQQ": "QQQONUSDT",
    "TQQQ": "RTQQQUSDT",
    "SPYON": "SPYONUSDT",
    "QQQON": "QQQONUSDT",
    "RTQQQ": "RTQQQUSDT",
    "RSPY": "RSPYUSDT",
    "RQQQ": "RQQQUSDT",
}

KNOWN_SPOT_PAIRS = frozenset(IBS_SPOT_BASE_MAP.values())


def resolve_ibs_trading_symbol(raw: str, product_type: str = "spot") -> str:
    """将 symbols.txt 中的 SPY/QQQ/TQQQ 解析为交易所交易对。"""
    s = str(raw or "").strip().upper()
    if not s:
        return ""
    if str(product_type).strip().lower() != "spot":
        return norm_symbol(s)
    if s in KNOWN_SPOT_PAIRS:
        return s
    base = s[:-4] if s.endswith("USDT") else s
    mapped = IBS_SPOT_BASE_MAP.get(base)
    if mapped:
        return mapped
    return norm_symbol(s)

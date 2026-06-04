"""Moss2 Profile  onboarding：用近期 K 线 + regime + discipline 回测，给出创建建议。

不依赖外部寻优目录；标的以 factory-en catalog / 可拉 K 线为准。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from moss2 import config as cfg
from moss2.dataset import list_en_catalog, normalize_symbol, resolve_csv_path
from moss2.discipline.report import regime_distribution
from moss2.kline_loader import load_market_df
from moss2.selection import compete_templates


def list_tradeable_symbols() -> List[str]:
    """当前机器上「有 EN 回测 CSV 或至少能解析」的 compact symbol 列表。"""
    seen: set[str] = set()
    out: List[str] = []
    for row in list_en_catalog():
        name = str(row.get("name") or "")
        parts = name.replace(".csv", "").split("_")
        if len(parts) >= 2 and parts[0] == "binanceusdm":
            base = parts[1].upper()
            sym = normalize_symbol(base, variant="en")
            if sym not in seen:
                seen.add(sym)
                out.append(sym)
    return sorted(out)


def _regime_template_hint(regime_mix: Dict[str, float]) -> str:
    """Moss2 风格：用近期 regime 结构推荐性格模板（非全市场寻优）。"""
    if not regime_mix:
        return cfg.MOSS2_DEFAULT_TEMPLATE
    dominant = max(regime_mix, key=regime_mix.get)
    bull = float(regime_mix.get("BULL", 0) or 0)
    bear = float(regime_mix.get("BEAR", 0) or 0)
    side = float(regime_mix.get("SIDEWAYS", 0) or 0)
    if dominant == "BULL" and bull >= 0.45:
        return "trend"
    if dominant == "SIDEWAYS" and side >= 0.5:
        return "mean_revert"
    if dominant == "BEAR" and bear >= 0.35:
        return "balanced"
    if bull >= 0.35:
        return "momentum"
    return cfg.MOSS2_DEFAULT_TEMPLATE


def suggest_profile(
    symbol: str,
    *,
    lookback_bars: Optional[int] = None,
    backtest_bars: Optional[int] = None,
    capital: Optional[float] = None,
    min_trades: Optional[int] = None,
) -> Dict[str, Any]:
    """
    为「当下」创建 Profile 提供依据：
    1) 近期 live/缓存 K 线的 regime 结构 → 模板倾向
    2) 若有 factory CSV，对 4 模板做短窗回测 + discipline → 数据驱动推荐
    """
    sym = normalize_symbol(symbol, variant=cfg.MOSS2_OPS_VARIANT)
    lookback = int(lookback_bars or min(cfg.MOSS2_KLINE_LIMIT, 2880))
    bt_bars = int(backtest_bars or min(cfg.MOSS2_EVOLVE_LIMIT_BARS, 4500))
    cap = float(capital or cfg.MOSS2_PROFILE_CAPITAL)

    out: Dict[str, Any] = {
        "lane": "moss2",
        "symbol": sym,
        "variant": cfg.MOSS2_OPS_VARIANT,
        "protocol_venue": cfg.MOSS2_PROTOCOL_VENUE,
        "data_csv": None,
        "regime_recent": {},
        "regime_hint_template": cfg.MOSS2_DEFAULT_TEMPLATE,
        "template_scores": [],
        "recommended_template": cfg.MOSS2_DEFAULT_TEMPLATE,
        "recommended_params": None,
        "recommended_enabled": False,
        "reason": "",
        "notes": [],
    }

    try:
        df = load_market_df(sym, cfg.MOSS2_OPS_VARIANT, limit=lookback)
    except Exception as e:
        out["ok"] = False
        out["reason"] = f"klines_unavailable:{e}"
        out["notes"].append(
            "无法读取近期 K 线；请确认币安可访问或 factory-en data_cache 有该币 CSV。"
        )
        return out

    from moss2.variants.en.core.regime import classify_regime

    regime = classify_regime(df, version=cfg.MOSS2_REGIME_VERSION)
    tail = regime.iloc[-min(len(regime), lookback) :]
    regime_mix = regime_distribution(tail)
    out["regime_recent"] = regime_mix
    hint = _regime_template_hint(regime_mix)
    out["regime_hint_template"] = hint
    out["bars_analyzed"] = int(len(df))
    out["last_close"] = float(df["close"].iloc[-1])

    csv_path = resolve_csv_path(sym, cfg.MOSS2_OPS_VARIANT)
    out["data_csv"] = str(csv_path) if csv_path else None

    if csv_path and csv_path.is_file():
        comp = compete_templates(
            sym,
            capital=cap,
            limit_bars=bt_bars,
            optimize_tactical=cfg.MOSS2_SELECTION_TACTICAL_NARROW,
            min_trades=min_trades,
        )
        scores = comp.get("rows") or []
        out["template_scores"] = [
            {
                "template": r.get("template"),
                "score": r.get("score"),
                "sharpe": r.get("sharpe"),
                "max_drawdown": r.get("max_drawdown"),
                "total_trades": r.get("total_trades"),
                "ev_per_trade_pct": r.get("ev_per_trade_pct"),
                "passes_discipline": r.get("passes_gates"),
                "discipline": r.get("discipline"),
                "summary": r.get("summary"),
            }
            for r in scores
        ]
        best = comp.get("best")
        if best:
            out["recommended_template"] = best["template"]
            out["recommended_params"] = best.get("params")
            out["selection_best"] = best
            out["reason"] = "backtest_selection_pass"
            out["notes"].append(
                f"回测窗 {bt_bars} 根：{best['template']} 过关且综合分最高"
                f"（含战术窄搜={cfg.MOSS2_SELECTION_TACTICAL_NARROW}）。"
            )
            gates_note = (
                f"闸门：成交≥{cfg.MOSS2_SELECTION_MIN_TRADES}、"
                f"EV≥{cfg.MOSS2_SELECTION_MIN_EV_PCT}、"
                f"Sharpe≥{cfg.MOSS2_SELECTION_MIN_SHARPE}、"
                f"MDD≤{cfg.MOSS2_SELECTION_MAX_MDD:.0%}"
            )
            out["notes"].append(gates_note)
        else:
            out["recommended_template"] = hint
            out["reason"] = "regime_hint_only"
            out["notes"].append(
                "四模板+窄搜均无候选过关；退回 regime 倾向，不建议启用。"
            )
    else:
        out["recommended_template"] = hint
        out["reason"] = "regime_hint_only"
        out["notes"].append(
            "无 factory-en CSV，未跑四模板回测；推荐仅来自近期 regime 结构。"
        )
        out["notes"].append(
            "上线前建议为该币准备 data_cache CSV，再执行 evolve + approve。"
        )

    out["ok"] = True
    out["recommended_name"] = f"{sym.lower().replace('usdt', '')}-en-{out['recommended_template']}"
    out["recommended_enabled"] = out["reason"] == "backtest_selection_pass"
    if cfg.MOSS2_AUTO_PROVISION_ENABLED:
        out["workflow"] = [
            "调度器/POST maintenance/auto-provision 将自动执行下列步骤",
            "suggest → create/update → evolve → approve → enable（过关时）",
            "纸面 15m 扫描对已启用 Profile 发 Protocol 信号",
        ]
    else:
        out["workflow"] = [
            "POST /api/moss2/profiles 使用 recommended_* 字段",
            "enabled 建议先 false，除非 reason=backtest_selection_pass",
            "POST /profiles/{id}/evolve?force=true 后 approve-candidate",
            "再 PATCH enabled=true 接 Protocol",
        ]
    return out

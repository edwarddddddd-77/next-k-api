#!/usr/bin/env python3
"""
Groq（OpenAI 兼容接口）生成右侧交易计划：买入区间 / 止损 / 止盈。
密钥通过环境变量配置，勿写入代码。

环境变量：
  GROQ_API_KEY      必填（Groq Cloud）
  GROQ_BASE_URL     默认 https://api.groq.com/openai/v1
  GROQ_MODEL        默认 llama-3.3-70b-versatile（可按 Groq 控制台可用模型调整）
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

# 与 accumulation_radar 一致：可选加载 .env.oi
_env_file = os.path.join(os.path.dirname(__file__), ".env.oi")
if os.path.isfile(_env_file):
    with open(_env_file, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

try:
    from openai import OpenAI
except ImportError as _e:
    OpenAI = None  # type: ignore
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None

REQUIRED_KEYS = (
    "buy_zone_bottom",
    "buy_zone_top",
    "stop_loss",
    "take_profit_1",
    "reasoning",
)


def _parse_json_object(text: str) -> Dict[str, Any]:
    """兼容模型偶发包裹 ```json ... ``` 的情况。"""
    t = text.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", t)
    if fence:
        t = fence.group(1).strip()
    return json.loads(t)


def _coerce_plan(raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in REQUIRED_KEYS:
        if k not in raw:
            raise ValueError(f"missing key: {k}")
        if k == "reasoning":
            out[k] = str(raw[k]).strip()
        else:
            out[k] = float(raw[k])
    return out


def _sanity_check_long(symbol: str, current_price: float, plan: Dict[str, Any]) -> None:
    """做多直觉校验：仅打印警告，不拦截（避免过于教条）。"""
    lo, hi = plan["buy_zone_bottom"], plan["buy_zone_top"]
    sl, tp = plan["stop_loss"], plan["take_profit_1"]
    if lo > hi:
        print(f"⚠️ [{symbol}] buy_zone_bottom > buy_zone_top，请人工复核")
    if current_price > 0 and sl >= hi:
        print(f"⚠️ [{symbol}] 止损不低于区间上沿，请复核是否做多逻辑")
    if current_price > 0 and tp <= hi:
        print(f"⚠️ [{symbol}] 止盈不高于区间上沿，请复核")


def get_ai_trading_plan_groq(
    symbol: str,
    current_price: float,
    kline_summary_text: str,
    phase: str,
    *,
    client: Optional[Any] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    调用 Groq Chat Completions，返回买入区间、止损、止盈与简短理由。
    未配置 GROQ_API_KEY 或缺少 openai 包时返回 {"ok": False, ...}。
    """
    if _IMPORT_ERROR is not None:
        return {"ok": False, "error": "openai_not_installed", "detail": str(_IMPORT_ERROR)}
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return {"ok": False, "error": "GROQ_API_KEY_missing"}

    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
    model_name = (model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")).strip()

    system_prompt = """
你是一个加密货币衍生品方向的量化助手。根据提供的盘面摘要，给出做多语境下的右侧交易计划参考。
必须严格输出一个 JSON 对象，且仅包含下列键（数值为 float，reasoning 为简短中文）：
{
  "buy_zone_bottom": float,
  "buy_zone_top": float,
  "stop_loss": float,
  "take_profit_1": float,
  "reasoning": "简短逻辑说明"
}
价格单位与现价一致；做多时止损应低于买入区间下沿，止盈应高于区间上沿。
"""

    user_prompt = (
        f"标的: {symbol}\n"
        f"现价: {current_price}\n"
        f"BPC/结构相位: {phase}\n"
        f"盘面摘要:\n{kline_summary_text}\n"
    )

    cli = client or OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = cli.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content
        if not raw_text:
            return {"ok": False, "error": "empty_response"}
        parsed = _parse_json_object(raw_text)
        plan = _coerce_plan(parsed)
        _sanity_check_long(symbol, current_price, plan)
        return {"ok": True, **plan, "model": model_name}
    except Exception as e:
        return {"ok": False, "error": "groq_request_failed", "detail": str(e)}


if __name__ == "__main__":
    # 示例：需 export GROQ_API_KEY=gsk_...
    demo = get_ai_trading_plan_groq(
        "CATIUSDT",
        0.07065,
        "突破阻力0.064，缩量回踩，OI上涨14%",
        "continuation",
    )
    print(json.dumps(demo, indent=2, ensure_ascii=False))

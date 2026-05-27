"""LLM 进化反思 — 由 evolution_log 生成 evolution_schedule。"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from moss_quant import config as cfg
from moss_quant.params import validate_schedule_round

logger = logging.getLogger(__name__)

_EVOLUTION_GUIDE = """
反思 7 原则（摘要）：
1. 先看累计大局，单段亏损勿过度反应
2. 分析盈利单原因
3. 分析亏损单原因（止损过密？阈值过低？）
4. 指出具体参数问题
5. 微调，单参数单次约不超过初始值 10%
6. 近 2 段内刚大调则保持或微调
7. 不可连续 3 段完全不调整

硬性约束（你必须遵守）：
- 不得修改任何 signal_weight、long_bias、leverage、risk_per_trade、rolling_* 等性格参数
- 只能调整战术参数：entry_threshold, exit_threshold, sl_atr_mult, tp_rr_ratio, trailing_*, regime_sensitivity, supertrend_mult, trend_strength_min, fast_ma_period, slow_ma_period, rsi_*
- 输出 JSON 数组，每项 {"round": N, "params": {完整参数对象}}，round 从 1 连续递增
- params 必须包含初始 params 的全部键，仅在战术字段上微调
"""


def _anthropic_messages(prompt: str) -> str:
    import requests

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    base = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    resp = requests.post(
        f"{base}/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"anthropic HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    parts = data.get("content") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return text.strip()


def _extract_json_array(text: str) -> List[dict]:
    text = text.strip()
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        raise ValueError("LLM response missing JSON array")
    return json.loads(m.group(0))


def generate_evolution_schedule(
    *,
    initial_params: dict,
    evolution_log: List[dict],
    n_segments: int,
) -> List[dict]:
    if not cfg.MOSS_QUANT_LLM_ENABLED:
        raise RuntimeError("MOSS_QUANT_LLM_ENABLED=0")
    prompt = (
        _EVOLUTION_GUIDE
        + "\n\n初始参数（性格字段不得在输出中改变）：\n"
        + json.dumps(initial_params, ensure_ascii=False, indent=2)
        + "\n\n分段回测 evolution_log：\n"
        + json.dumps(evolution_log, ensure_ascii=False, indent=2)
        + f"\n\n请输出长度为 {n_segments} 的 evolution_schedule JSON 数组。"
    )
    raw = _anthropic_messages(prompt)
    schedule = _extract_json_array(raw)
    if len(schedule) < n_segments:
        # pad last params
        last = schedule[-1]["params"] if schedule else initial_params
        for i in range(len(schedule) + 1, n_segments + 1):
            schedule.append({"round": i, "params": last})
    schedule = schedule[:n_segments]
    out: List[dict] = []
    for i, item in enumerate(schedule):
        params = validate_schedule_round(
            item.get("params") or initial_params,
            initial_params,
        )
        out.append({"round": i + 1, "params": params})
    return out

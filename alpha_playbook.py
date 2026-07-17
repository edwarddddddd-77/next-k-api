#!/usr/bin/env python3
"""
CJ 原文决策树（币安 Alpha 筹码）

原文要点：
1. 开盘后直接观察持仓靠前地址的余额变化
2. 地址动得越少越好分析
3. 很多巨鲸地址同时动 → 流通增加很多 → 跌
4. 只有 alpha 地址在交易 → 流通受限 → 涨
5. 第二类（空投）开始动，幅度=抛压；大就跑
6. Alpha 更早上市：初期几乎没有卖盘只有买盘；前 5–10 分钟卖压多来自工作室/个人
7. 关键：筹码分布 + 代币释放时间线（并非所有币适用）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# 第二类（空投）抛压达到该百分点合计 → 「跑」
AIRDROP_PRESSURE_PP = 0.8
# 「很多巨鲸同时动」
MULTI_WHALE_MIN_COUNT = 3
MULTI_TYPE_MIN_COUNT = 2

# 看板展示用：与原文一一对应
STRATEGY_RULES: List[Dict[str, str]] = [
    {
        "id": "observe",
        "title": "盯地址余额变化",
        "body": "开盘后直接观察持仓靠前地址的余额变化（Alpha / 交易所 / 做市商 / 空投）。",
        "quote": "直接观察一些地址的余额变化",
    },
    {
        "id": "fewer_better",
        "title": "动得越少越好分析",
        "body": "同时变动的头部地址越少，盘面越清晰。",
        "quote": "地址动的越少越好分析",
    },
    {
        "id": "multi_whale",
        "title": "多鲸同动 → 跌",
        "body": "很多巨鲸地址同时开始动 → 流通会增加很多 → 偏空。",
        "quote": "当很多巨鲸地址开始动，就意味着流通的会增加很多，基本上就会跌",
    },
    {
        "id": "alpha_only",
        "title": "仅 Alpha 交易 → 涨",
        "body": "只有 alpha 地址在交易 → 流通受限 → 偏多。",
        "quote": "如果只有 alpha 地址在交易，则意味着流通受限，基本上就会涨",
    },
    {
        "id": "airdrop_run",
        "title": "第二类空投抛压 → 跑",
        "body": "看空投地址何时开始动；动的幅度就是抛压，大就跑。",
        "quote": "看第二类地址（空投）什么时候开始动，看动的幅度就是抛压，大就跑",
    },
    {
        "id": "early_window",
        "title": "早期 5–10 分钟窗口",
        "body": "Alpha 更早：初期几乎没有卖盘只有买盘；前 5–10 分钟卖压多来自工作室/个人。",
        "quote": "因为早，所以初期几乎是没有卖盘的，只有买盘",
    },
    {
        "id": "structure",
        "title": "筹码分布 + 释放时间线",
        "body": "把筹码分布和代币释放时间线对齐；并非所有币适用。",
        "quote": "筹码分布和代币释放的时间线",
    },
]


def evaluate_cj_playbook(
    *,
    has_baseline: bool,
    outflow_addrs: List[Dict[str, Any]],
    outflow_share: float,
    inflow_share: float,
    phase: Optional[str] = None,
) -> Dict[str, Any]:
    type_set = {str(m.get("type") or "other") for m in outflow_addrs}
    n = len(outflow_addrs)
    airdrop_out = [m for m in outflow_addrs if str(m.get("type")) == "airdrop"]
    airdrop_pp = round(sum(-float(m.get("delta_share_pct") or 0) for m in airdrop_out), 4)
    alpha_out = [m for m in outflow_addrs if str(m.get("type")) == "alpha"]
    exchange_mm_out = [m for m in outflow_addrs if str(m.get("type")) in ("exchange", "mm")]

    pressure_pp = round(
        sum(
            -float(m.get("delta_share_pct") or 0)
            for m in outflow_addrs
            if str(m.get("type")) in ("airdrop", "exchange", "mm", "whale", "other", "mid")
        ),
        4,
    )

    steps: List[Dict[str, Any]] = [
        {"id": "watch_top", "text": "盯持仓靠前地址的余额变化", "hit": True},
        {
            "id": "simultaneous",
            "text": f"同时流出的头部地址数 = {n}（动得越少越好分析）",
            "hit": True,
            "value": n,
        },
    ]

    if not has_baseline:
        return {
            "signal": "baseline",
            "signal_label": "已建持仓基线",
            "bias": "neutral",
            "action": "再刷一次持仓，对比余额变动后才能出原文信号",
            "quote": "直接观察一些地址的余额变化",
            "pressure_pp": 0.0,
            "simultaneous_movers": 0,
            "steps": steps + [{"id": "baseline", "text": "首次快照，尚无对比", "hit": True}],
        }

    # 第二类空投 · 大就跑
    if airdrop_out and airdrop_pp >= AIRDROP_PRESSURE_PP:
        steps.append(
            {
                "id": "airdrop",
                "text": f"第二类（空投）开始动，抛压≈{airdrop_pp:.2f}pp → 跑",
                "hit": True,
            }
        )
        return {
            "signal": "airdrop_dump",
            "signal_label": "空投抛压 · 跑",
            "bias": "bearish",
            "action": "第二类地址开始动且幅度偏大 → 优先进场离场/减仓",
            "quote": "看第二类地址（空投）什么时候开始动，看动的幅度就是抛压，大就跑",
            "pressure_pp": pressure_pp,
            "simultaneous_movers": n,
            "airdrop_pressure_pp": airdrop_pp,
            "steps": steps,
        }

    # 很多巨鲸同动 → 跌
    multi = n >= MULTI_WHALE_MIN_COUNT or (
        n >= MULTI_TYPE_MIN_COUNT and len(type_set - {"burn"}) >= 2
    )
    if multi:
        steps.append(
            {
                "id": "multi_whale",
                "text": f"多地址同动（{n} 个 / 类型 {sorted(type_set)}）→ 流通↑ → 偏空",
                "hit": True,
            }
        )
        return {
            "signal": "multi_whale_outflow",
            "signal_label": "多鲸同动 · 流通放大",
            "bias": "bearish",
            "action": "很多巨鲸地址同时动 → 流通增加很多 → 减仓/观望",
            "quote": "当很多巨鲸地址开始动，就意味着流通的会增加很多，基本上就会跌",
            "pressure_pp": pressure_pp,
            "simultaneous_movers": n,
            "steps": steps,
        }

    # 只有 alpha 在交易 → 涨
    only_alpha = n >= 1 and len(alpha_out) == n and not airdrop_out and not exchange_mm_out
    if only_alpha:
        steps.append(
            {
                "id": "alpha_only",
                "text": "只有 Alpha/头部候选在动，空投与交易所未放量 → 流通受限 → 偏多",
                "hit": True,
            }
        )
        return {
            "signal": "alpha_only_move",
            "signal_label": "仅 Alpha 在交易 · 流通受限",
            "bias": "bullish",
            "action": "只有 alpha 地址在交易 → 流通受限 → 可做波短/偏多",
            "quote": "如果只有 alpha 地址在交易，则意味着流通受限，基本上就会涨",
            "pressure_pp": pressure_pp,
            "simultaneous_movers": n,
            "steps": steps,
        }

    if airdrop_out:
        steps.append(
            {
                "id": "airdrop_warn",
                "text": f"第二类（空投）已开始动，抛压≈{airdrop_pp:.2f}pp（未达跑出门槛）",
                "hit": True,
            }
        )
        return {
            "signal": "airdrop_watch",
            "signal_label": "空投开始动 · 盯抛压",
            "bias": "volatile",
            "action": "空投地址已动；继续刷，幅度放大就跑",
            "quote": "看第二类地址（空投）什么时候开始动，看动的幅度就是抛压",
            "pressure_pp": pressure_pp,
            "simultaneous_movers": n,
            "airdrop_pressure_pp": airdrop_pp,
            "steps": steps,
        }

    if phase == "early_window" and n == 0:
        steps.append(
            {
                "id": "early_window",
                "text": "早期 5–10 分钟窗口且头部大户未动 → 卖盘偏散户/工作室",
                "hit": True,
            }
        )
        return {
            "signal": "early_buy_window",
            "signal_label": "早期窗口 · 大户未动",
            "bias": "bullish",
            "action": "初期几乎没有卖盘只有买盘；前 5–10 分钟卖压多来自工作室/个人",
            "quote": "因为早，所以初期几乎是没有卖盘的，只有买盘",
            "pressure_pp": 0.0,
            "simultaneous_movers": 0,
            "steps": steps,
        }

    if n >= 1:
        steps.append(
            {
                "id": "light_flow",
                "text": f"头部有流出但未达多鲸/空投抛压阈值（类型 {sorted(type_set)}）",
                "hit": True,
            }
        )
        bias = "bearish" if pressure_pp >= 1.0 else "neutral"
        return {
            "signal": "light_outflow",
            "signal_label": "轻度流出 · 继续观察",
            "bias": bias,
            "action": "筹码开始松动但结构未失控；对照释放时间线",
            "quote": "地址动的越少越好分析",
            "pressure_pp": pressure_pp,
            "simultaneous_movers": n,
            "steps": steps,
        }

    steps.append({"id": "quiet", "text": "头部筹码相对稳定", "hit": True})
    action = "继续盯：开盘早期大户是否动、空投是否进场"
    if phase == "upcoming":
        action = "临近 Alpha 开盘：准备刷持仓，开盘后对比余额变化"
    return {
        "signal": "quiet",
        "signal_label": "头部稳定 · 流通未放大",
        "bias": "neutral",
        "action": action,
        "quote": "地址动的越少越好分析",
        "pressure_pp": pressure_pp,
        "simultaneous_movers": 0,
        "inflow_share_pct": round(inflow_share, 4),
        "steps": steps,
    }

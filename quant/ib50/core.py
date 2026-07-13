"""IB50 信号核心 — Initial Balance 50% 中点延续（MrZinc / SpookyQuant 机械骨架）。

规则摘要：
- Initial Balance = 会话开盘前 N 分钟（默认 60）的高/低区间
- 方向：IB 期间先形成的极值 — 先 low → 做多；先 high → 做空（continuation）
- 入场：IB 结束后市价（避免 limit 在已穿越中点时的虚假 fill）
- 止损：区间对侧边缘；目标：bias 方向对侧边缘（1:1 以 IB 半宽为基准）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

DirectionMode = Literal["continuation", "inverse"]
FirstExtreme = Literal["high", "low"]

_WEEKDAY_ALIASES: dict[str, int] = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


@dataclass(frozen=True)
class IntrabarOhlc:
    open_ms: int
    open: float
    high: float
    low: float
    close: float


@dataclass
class Ib50SessionState:
    ib_high: float = 0.0
    ib_low: float = 0.0
    first_extreme: FirstExtreme | None = None
    ib_complete: bool = False


@dataclass(frozen=True)
class InitialBalance:
    high: float
    low: float
    midpoint: float
    range: float
    first_extreme: FirstExtreme


@dataclass(frozen=True)
class Ib50Setup:
    side: int
    entry_price: float
    stop: float
    target: float
    ib: InitialBalance
    direction_mode: DirectionMode


def compute_midpoint(high: float, low: float) -> float:
    return (float(high) + float(low)) / 2.0


def first_extreme_on_bar(*, open_: float, high: float, low: float) -> FirstExtreme:
    """首根 IB K 无 tick 数据时：距 open 更远的一侧视为先形成。"""
    o, h, l = float(open_), float(high), float(low)
    if (o - l) >= (h - o):
        return "low"
    return "high"


def update_ib_range(
    *,
    ib_high: float,
    ib_low: float,
    first_extreme: FirstExtreme | None,
    open_: float,
    high: float,
    low: float,
) -> tuple[float, float, FirstExtreme | None]:
    """累积 IB 区间；记录首个被突破的极值方向。"""
    hi = float(high)
    lo = float(low)
    if ib_high <= 0 or ib_low <= 0:
        return hi, lo, first_extreme_on_bar(open_=open_, high=hi, low=lo)

    new_first = first_extreme
    if lo < ib_low and new_first is None:
        new_first = "low"
    elif hi > ib_high and new_first is None:
        new_first = "high"
    return max(ib_high, hi), min(ib_low, lo), new_first


def finalize_initial_balance(
    *,
    ib_high: float,
    ib_low: float,
    first_extreme: FirstExtreme | None,
) -> InitialBalance | None:
    if ib_high <= 0 or ib_low <= 0 or ib_high <= ib_low:
        return None
    ext: FirstExtreme = first_extreme or "low"
    rng = ib_high - ib_low
    return InitialBalance(
        high=ib_high,
        low=ib_low,
        midpoint=compute_midpoint(ib_high, ib_low),
        range=rng,
        first_extreme=ext,
    )


def normalize_direction_mode(raw: str) -> DirectionMode:
    value = str(raw or "continuation").strip().lower().replace(" ", "_").replace("-", "_")
    if value in ("inverse", "fade", "revert", "reversal"):
        return "inverse"
    return "continuation"


def continuation_side(first_extreme: FirstExtreme) -> int:
    return 1 if first_extreme == "low" else -1


def trade_side(first_extreme: FirstExtreme, *, direction_mode: DirectionMode) -> int:
    side = continuation_side(first_extreme)
    if direction_mode == "inverse":
        return -side
    return side


def ib50_stop_target(ib: InitialBalance, side: int) -> tuple[float, float]:
    if side > 0:
        return ib.low, ib.high
    return ib.high, ib.low


def build_ib50_setup(
    ib: InitialBalance,
    entry_price: float,
    *,
    direction_mode: DirectionMode = "continuation",
) -> Ib50Setup:
    side = trade_side(ib.first_extreme, direction_mode=direction_mode)
    stop, target = ib50_stop_target(ib, side)
    return Ib50Setup(
        side=side,
        entry_price=float(entry_price),
        stop=float(stop),
        target=float(target),
        ib=ib,
        direction_mode=direction_mode,
    )


def parse_weekday_filter(raw: str | None) -> frozenset[int] | None:
    text = str(raw or "").strip().lower()
    if not text or text in ("all", "*", "none", "off"):
        return None
    days: set[int] = set()
    for part in text.replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            days.add(int(token) % 7)
            continue
        key = token[:3] if len(token) > 3 else token
        if key in _WEEKDAY_ALIASES:
            days.add(_WEEKDAY_ALIASES[key])
        elif token in _WEEKDAY_ALIASES:
            days.add(_WEEKDAY_ALIASES[token])
    return frozenset(days) if days else None


def weekday_allowed(weekday: int, allowed: frozenset[int] | None) -> bool:
    if allowed is None:
        return True
    return int(weekday) in allowed


def ib_window_end_ms(anchor_ms: int, *, ib_minutes: int) -> int:
    return int(anchor_ms) + max(1, int(ib_minutes)) * 60_000


def in_ib_window(bar_ms: int, *, anchor_ms: int, ib_minutes: int) -> bool:
    end_ms = ib_window_end_ms(anchor_ms, ib_minutes=ib_minutes)
    return int(anchor_ms) <= int(bar_ms) < end_ms


def ib_complete_at_bar(bar_ms: int, *, anchor_ms: int, ib_minutes: int) -> bool:
    return int(bar_ms) >= ib_window_end_ms(anchor_ms, ib_minutes=ib_minutes)


def bar_exit_reason(
    *,
    side: int,
    high: float,
    low: float,
    stop: float,
    target: float,
    prev_high: float,
    prev_low: float,
) -> str | None:
    """SL / TP 首次触碰；同 bar 双触 → 记 stop。"""
    if side > 0:
        sl_hit = low <= stop and prev_low > stop
        tp_hit = high >= target and prev_high < target
    else:
        sl_hit = high >= stop and prev_high < stop
        tp_hit = low <= target and prev_low > target
    if sl_hit and tp_hit:
        return "stop_loss"
    if sl_hit:
        return "stop_loss"
    if tp_hit:
        return "target_hit"
    return None


def replay_ib_from_bars(
    bars: Sequence[IntrabarOhlc],
    *,
    anchor_ms: int,
    ib_minutes: int,
) -> InitialBalance | None:
    """从分钟 K 序列重放 IB 区间（回测 / 测试用）。"""
    state = Ib50SessionState()
    for bar in bars:
        if not in_ib_window(bar.open_ms, anchor_ms=anchor_ms, ib_minutes=ib_minutes):
            continue
        state.ib_high, state.ib_low, state.first_extreme = update_ib_range(
            ib_high=state.ib_high,
            ib_low=state.ib_low,
            first_extreme=state.first_extreme,
            open_=bar.open,
            high=bar.high,
            low=bar.low,
        )
    return finalize_initial_balance(
        ib_high=state.ib_high,
        ib_low=state.ib_low,
        first_extreme=state.first_extreme,
    )

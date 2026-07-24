# HL 跟单候选筛选说明

本文说明短线桌（`hl-short`）**跟单地址**如何从 Hyperliquid 全站筛到「能跟」名单，以及线上候选池如何与本地口径对齐。

---

## 目标

从公开排行榜挖出：

- **日内 / 波段**风格（不要剥头皮狂刷）
- **真·一开一合**高胜率（仓位归零回合，不是碎单配对）
- **Bitget 能映射**的品种（含可映的美股 `xyz:`）
- **地址上还有钱**（不要求永续账户必须有余额）

最终本地标定的 **能跟 10 地址**见文末；线上 `ready` 按同一套规则，并用 seed 强制扫这 10 个。

---

## 漏斗总览

```
HL 排行榜 ~4万
    │  粗筛：AV / 周ROI / 周盈亏
    ▼
~1000+ 粗筛池
    │  拉 7d fills · Bitget 映射 · flat 回合
    ▼
指标全集（dump）
    │  风格 day/swing · WR≥80% · 回合 10–100
    ▼
~40 日内/波段表
    │  钱包资金 · 静默 · 盈亏 · 反刷单
    ▼
能跟 ~10  →  线上 ready
```

---

## 第 0 步 · 榜单粗筛

数据源：`https://stats-data.hyperliquid.xyz/Mainnet/leaderboard`

| 条件 | 阈值 |
|------|------|
| 账户权益 `av` | 8,000 ～ 2,000,000 |
| 周收益率 `week_roi` | ≥ 10% |
| 周盈亏 `week_pnl` | ≥ $3,000 |
| 周成交等 | 见 `_leaderboard_candidates` |

说明：粗筛只看榜单账本表现（可含浮盈），**不看**真实开平节奏。

---

## 第 1 步 · 深挖成交与回合

对粗筛地址拉 `userFillsByTime`（近 7 日），并：

1. 只保留 **Bitget 可映射**币种的成交  
2. 合成 **flat 回合**（与 hyper-track 同思路）：

| 概念 | 定义 |
|------|------|
| 开 | 某币净仓从 **0 → 非 0** |
| 合 | 净仓 **再回到 0**（或翻向，先结一轮） |
| 加仓/减仓 | 仍算**同一轮**，直到归零 |
| 盈亏 | 该轮 `closedPnl` 合计 − `fee` |
| 胜 | 净盈亏 > 0；平盘不计分母 |

> 旧口径「60 秒开平碎片配对」已弃用（会虚增回合）。代码：`utils/hl_wr_screen.py` → `_round_trips_flat`。

同时计算：`wr7`、`trips7`、`pnl7_closed`、`pnl_per_trip`、`raw_fph24`、`lph24`、`quiet_h`、`bitget_share` 等。

### 本地下载（可选全量）

```bash
python scripts/download_desk_screen_data.py          # → hl_desk_screen_dump.json
python scripts/rescore_desk_screen_dump.py           # 用 flat 重算 wr/trips
```

---

## 第 2 步 · 风格（hyper-track）+ 胜率表

风格只看 **90 日逻辑浓缩到本窗口 flat 回合**（持仓时长 + 日频）：

| 风格 | 条件 |
|------|------|
| `scalper` | 均持仓 &lt; 1h 且 每天 &gt; 10 回合 |
| `day_trader`（日内） | 均持仓 &lt; 24h 且 每天 &gt; 2 回合 |
| `swing_trader`（波段） | 均持仓 &lt; 7 天 |
| `position_trader` | 更长 |

**本产品只要日内 + 波段。**

本地导出：

```bash
python scripts/export_ht_day_swing.py
# → hl_ht_day_swing.csv / .json
```

门槛：

| 条件 | 阈值 |
|------|------|
| 风格 | `day_trader` 或 `swing_trader` |
| flat 回合 | 10 ～ 100 |
| 胜率 `wr7` | ≥ 80% |

---

## 第 3 步 · 「能跟」硬门槛

在第 2 步结果上再砍（`scripts/regrade_day_swing_wallet_funds.py`）：

| 条件 | 阈值 | 说明 |
|------|------|------|
| 钱包总资金 | ≥ $8,000 | **永续 AV + Core 现货 USDC + HyperEVM USDC**；永续=0 但现货有钱 → **可通过** |
| 静默 `quiet_h` | ≤ 24h | 距最近一笔 fill |
| 平仓净盈 `pnl7_closed` | ≥ $3,000 | flat 回合合计 |
| 单回合均盈 | ≥ $80 | 反薄利刷单 |
| `raw_fph24` | ≤ 12 | 反碎单 |
| `lph24` | ≤ 6 | 合并腿频率 |
| Bitget | fills 可映占比够低阈值也可；或主币多数可映 | **美股可映则允许** |
| 胜率 / 风格 | 仍满足第 2 步 | |

输出：`hl_ht_day_swing_followable.json`（能跟 / 可试 / 不能跟）。

---

## 线上候选池（与上面对齐）

| 模块 | 路径 |
|------|------|
| 核心 | `utils/hl_desk_candidates.py` |
| 周任务 | 周一 09:30 · `build_desk_candidates` |
| 手动 | `python scripts/build_desk_candidates.py` |
| API | `GET /api/hl-short/candidates` |
| UI | `hl-short.html` 候选池表 |
| Seed | `hl_ready_seed_addrs.json`（强制深挖「能跟」地址） |

### ready（可绑）默认

- 风格：日内 / 波段  
- flat WR ≥ **80%**，回合 **10–100**  
- 钱包合计 ≥ **$8k**（非仅永续）  
- 静默 ≤ **24h**，平仓盈亏 / 反刷单同上  
- `DEEP_N=0`：深挖**整段粗筛池**（勿再设成 120，会漏地址）

### 重要环境变量

| 变量 | 建议 | 含义 |
|------|------|------|
| `HL_CANDIDATE_DEEP_N` | `0` 或不设 | 0 = 全粗筛池；`120` 会漏人 |
| `HL_CANDIDATE_READY_MIN_WR` | `0.80` | ready 最低胜率 |
| `HL_CANDIDATE_READY_MIN_WALLET_USD` | `8000` | 钱包合计 |
| `HL_CANDIDATE_READY_STYLES` | `day_trader,swing_trader` | ready 风格 |
| `HL_CANDIDATE_READY_MAX_QUIET_H` | `24` | 静默上限 |

---

## 标定「能跟」10 地址（样例快照）

> 快照日期约 2026-07-24；资金/静默会变，seed 用于强制复扫，ready 仍以当时门槛为准。

| # | 风格 | WR | 回合 | 钱包约 | 地址 |
|---|------|-----|------|--------|------|
| 1 | 日内 | 100% | 43 | 2.5万 | `0x6dbbefad3d24da625fa233c070678ab1938fcd38` |
| 2 | 日内 | 100% | 21 | 2.7万 | `0x8ae835143debfe89b50d8913114eafda33347a34` |
| 3 | 日内 | 100% | 15 | 9.2万 | `0x677fa53cd2acb8e87f65ed1d0353683b82618d02` |
| 4 | 日内 | 95% | 19 | 2.7万 | `0xa7405ff2687cb83b8a8a08eeaa4e4bc249344d23` |
| 5 | 日内 | 93% | 15 | 12.6万 | `0xb1169c6daac76bacaf0d8f87641fc38fbabe569f` |
| 6 | 日内 | 93% | 15 | 2.0万 | `0x9e16d69bafe80636c5be57f6ded44f45a6ef625e` |
| 7 | 日内 | 93% | 15 | 2.8万 | `0xa567027c7a7a5f3b2e2e58de823cef0c3d5e1e1f` |
| 8 | 日内 | 92% | 13 | 58.6万 | `0xa7aae418ee1714642f46b577c5b6f6c8bcb561dc` |
| 9 | 波段 | 92% | 12 | 163万 | `0x212abcf1a07986aedb244fc4c6f5fe037913a8c6` |
| 10 | 日内 | 90% | 29 | 2.4万 | `0x7d6785068887e9a76fa84a830ca92873fdbbcaa6` |

Seed 文件：`hl_ready_seed_addrs.json`（与上表一致）。

---

## 常用命令速查

```bash
# 全量下载粗筛池（慢，可断点续跑）
python scripts/download_desk_screen_data.py

# flat 重算 dump
python scripts/rescore_desk_screen_dump.py

# 日内/波段表
python scripts/export_ht_day_swing.py

# 钱包资金重评能跟
python scripts/regrade_day_swing_wallet_funds.py

# 线上同款候选池
python scripts/build_desk_candidates.py
```

---

## 设计取舍（备忘）

| 点 | 结论 |
|----|------|
| 回合 | 用 **仓位归零**，不用 60s 碎单配对 |
| 资金 | 看 **地址总资金**，永续空仓不一票否决 |
| 美股 | Bitget **有映射就可以跟** |
| 榜周 ROI | 只作粗筛；跟单质量看 flat 胜率与回合 |
| 自动换绑 | 尚未启用；目前只维护 `ready/watch/bound` 池 |

# ORB 配置（V2）

- `v2/symbols.txt` — 默认标的池（**17 标**）
- 实盘 Gate + 模型：**`orb_live/`**（唯一参数目录）
- 标的：**`config/orb/v2/symbols.txt`**（17 标，随 git 部署；勿放 `data/` Volume）

## 实盘模式（默认）

| 项 | 值 |
|----|-----|
| OR 窗口 | 全池统一 **15m**（`ORB_OR_MINUTES=15`） |
| 信号 K 线 | 5m 突破 |
| 标的 | 17 标共享扫描 |
| Robot | **8 台共享池**（`live_gate.json` `robot_pool_size` / `max_opens_per_day`） |
| 绑定 | `ORB_V2_ROBOT_BOUND=0`（17≠8，一标一绑不生效） |
| 单台本金 | `ORB_V2_ROBOT_EQUITY=14`（与 `live_gate.json` 备注一致） |
| Gate | ML p≥0.35 + BS≥45 + early-trap 15m |
| 入场 | `ORB_ENTRY_FILL=fvg_prox` → Protocol LIMIT |

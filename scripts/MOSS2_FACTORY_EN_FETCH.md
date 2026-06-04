# Moss2 回测 CSV（线上自动，无需 skills 目录）

## 默认行为（已内置）

- **目录**：`next-k-api/data/moss2_en_data_cache/`（或 `DATA_DIR/moss2_en_data_cache`）
- **不依赖** `moss-trade-bot-skills-main`
- **启动后约 90 秒**：仅在「有缺失 seed CSV」或「从未成功 bootstrap」时拉取；若 25 个核心 CSV 已齐且存在 `.moss2_bootstrap_last.json`，则跳过（避免每次重启全量扫币安）
- **K 线窗**：默认 `MOSS2_FETCH_SINCE_ROLLING=True`，拉取**最近 148 天至当前**；文件名 `..._15m_148d.csv`（非固定 2025-10-06 终点）
- **每周日 04:00 UTC**：过期文件刷新
- **启动后约 5 分钟 / 每周日 04:45 UTC**：全自动 Profile（25 核心 suggest→创建→进化→启用）
- **实盘 15m 扫描**：仍用币安实时 K 线，不等待 CSV

覆盖路径：`moss2/config.py` → `MOSS2_DATA_BOOTSTRAP_*`

## 手动触发

```http
POST /api/moss2/maintenance/bootstrap-data?force=false
X-Maintenance-Token: <PROTOCOL_MAINTENANCE_TOKEN>
```

`force=true` 强制重拉全部。

## 全自动 Profile

```http
POST /api/moss2/maintenance/auto-provision?force_evolve=false
X-Maintenance-Token: <PROTOCOL_MAINTENANCE_TOKEN>
```

开关：`moss2/config.py` → `MOSS2_AUTO_PROVISION_*`、`MOSS2_EVOLVE_AUTO_APPROVE`、`MOSS2_AUTO_ENABLE_PROFILES`。

## 本地开发（可选读旧 skills 目录）

若本机有 `moss-trade-bot-skills-main` 且想沿用其中 CSV：

```env
MOSS2_PREFER_SKILLS_DATA_CACHE=1
```

## 自定义目录

```env
MOSS2_EN_DATA_CACHE=/data/moss2_csv
```

## 离线脚本（可选）

仍可使用 `scripts/fetch_factory_en_moss_universe.ps1`；线上一般不需要。

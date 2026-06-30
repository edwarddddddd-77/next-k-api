# ORB 实盘参数包

**实盘与回测统一从此目录加载 Gate + 大模型。** 改参数时直接覆盖文件，**无需重启 API**（下一档 scan 自动重载）。

## 目录位置

```
next-k-api/orb_live/
```

在 `data/` 外面，不受 `DATA_DIR` Volume 影响；随 **git 提交 → 部署** 进入镜像。

## 需要维护的文件

| 文件 | 说明 |
|------|------|
| `live_gate.json` | Gate 参数（min_p_true、BS、robot 数、early trap、`ml_gate_enabled` 等） |
| `breakout_gbm.pkl` | GBM 模型（必须） |
| `breakout_gbm.json` | 模型 meta / 指标 |
| `symbol_breakout_profiles.json` | 标的先验画像（必须） |
| `breakout_gbm_train_report.json` | 可选，训练报告参考 |

## 更新流程

1. 本地训练 / 调 Gate 完成后，将文件复制到 `orb_live/`
2. 确认 JSON 合法、`breakout_gbm.pkl` 存在
3. **git commit 并部署**
4. 下一档 scan 生效

## 首次初始化

若目录为空，可从训练产物复制：

```bash
python tools/orb/bootstrap_live_bundle.py
```

## 其他目录（不要混用）

| 目录 | 用途 |
|------|------|
| `data/orb/ml/models/` | 训练中间产物；`monthly_train.py --skip-promote` 写这里 |
| `data/orb/kline/` | K 线缓存（程序每月自动拉取） |
| `config/orb/v2/symbols.txt` | 扫描标的池（33 标，git 部署） |
| `output/` | 回测报告（只写不读） |

训练 promote 成功时会自动同步模型文件到 `orb_live/`（**不含** `live_gate.json`，Gate 仍须人工维护）。

## 环境变量

| 变量 | 说明 |
|------|------|
| `ORB_LIVE_BUNDLE_ROOT` | 留空即可；默认 `orb_live/` |
| `ORB_V2_GATE_CONFIG` | 留空；覆盖 Gate 路径时用 |
| `ORB_V2_GBM_PATH` | 留空；覆盖模型路径时用 |

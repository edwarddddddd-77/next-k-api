# ORB Live 人工替换包

Live 与回测从此目录加载 **Gate + 大模型**，直接覆盖文件即可，**无需重启 API**（下次 scan 自动重载）。

## 目录

`data/orb/live/`（可用 `ORB_LIVE_BUNDLE_ROOT` 覆盖）

参数随 **git 提交 → 部署** 进入镜像，**不需要**拷贝到 Volume。

## 需要维护的文件

| 文件 | 说明 |
|------|------|
| `live_gate.json` | Gate 参数（min_p_true、8 robot、early trap 等） |
| `breakout_gbm.pkl` | GBM 模型 |
| `breakout_gbm.json` | 模型 meta / 指标 |
| `symbol_breakout_profiles.json` | 43 标先验画像 |
| `breakout_gbm_train_report.json` | 可选，训练报告参考 |

## 首次初始化

```bash
python tools/orb/bootstrap_live_bundle.py
```

## 人工更新流程

1. 本地训练 / 调 Gate 完成后，将上述文件复制到 `data/orb/live/`
2. 确认 JSON 合法、pkl 存在
3. **提交 git 并部署**
4. Live 下一档 scan 生效

## 自动化（程序负责）

- **仅 K 线**：每月 1 日 02:00 UTC 刷新 `data/orb/kline/`（`ORB_ML_KLINE_REFRESH_ENABLED=1`）
- **不自动**：月度训练、模型 promote、Gate 调参（需人工跑 `monthly_train.py` 或自行训练）

## 标的池

仍在 `data/orb/ml/symbols/universe.txt`（`ORB_V2_SYMBOLS_FILE`），与模型包分开。

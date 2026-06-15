# ORB（仅 V2）

ML Live Gate + 8-robot 纸面扫描。

## 入口

| 用途 | 路径 |
|------|------|
| 定时/CLI 扫描 | `orb_scanner.py` → `orb/v2/paper.py` |
| Live 人工包 | `orb_live/` |
| 标的池配置 | `config/orb/v2/symbols.txt` |
| 训练产物 | `data/orb/ml/` |
| K 线缓存 | `data/orb/kline/` |
| 回测报告 | `output/orb/v2/eval/` |

## 共享模块（`orb/core/`）

- `paper.py` — 信号分析、入库、持仓结算
- `backtest.py` — 回放回测引擎
- `config.py` — 策略参数（`OrbConfig`）
- `db.py` — `orb_signals` / settlements

## 环境变量

- `ORB_V2_ENABLED=1` — 开启扫描
- `ORB_V2_SCHEDULER_ENABLED=1` — 定时任务（默认开）
- `ORB_V2_SYMBOLS_FILE` — 可选覆盖；默认 `config/orb/v2/symbols.txt`（勿指向 `data/`）

## 常用命令

```bash
python orb_scanner.py
python tools/orb/v2/monthly_train.py
python tools/orb/bootstrap_live_bundle.py
```

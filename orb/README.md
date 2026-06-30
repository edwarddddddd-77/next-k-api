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
- `ORB_V2_SYMBOLS_FILE` — 可选覆盖；默认 `config/orb/v2/symbols.txt`（17 标，勿指向 `data/`）
- `ORB_OR_MINUTES=15` — 全池统一 15m OR
- `ORB_V2_ROBOT_BOUND=0` — 17 标共享 8 robot（见 `live_gate.json`）
- `ORB_V2_ROBOT_EQUITY=14` — 单台 robot 本金（U）

## 实盘流程（17 标 + 8 Robot 共享池）

1. **配置**：`symbols.txt`（17 标）+ `orb_live/live_gate.json`（8 并发、ML Gate）+ `.env.oi`（OR15、fvg_prox、`ORB_LIVE_ENABLED` + `PROTOCOL_API_URL`）
2. **盘前**：09:25 ET 刷新 K 线缓存
3. **盘中**：每 5m UTC+5s → `orb_scanner.py` → 扫 17 标 OR15 突破 → ML/BS Gate → 占空 robot 槽 → FVG LIMIT 发 Protocol
4. **平仓**：每档 scan `resolve` SL/TP/EOD → Protocol close；`robot_reuse` 释放槽位
5. **改参**：覆盖 `orb_live/` 或 env → 部署 → 下一档 scan 生效（无需重启）

## 常用命令

```bash
python orb_scanner.py
python tools/orb/v2/monthly_train.py
python tools/orb/bootstrap_live_bundle.py
```

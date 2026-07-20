"""Markdown reports + strategies.suggested.yaml from optimize raw JSON."""
import datetime
import json

import yaml

from .. import config
from . import data
from .optimize import ORDER, RAW_DIR

REPORT_DIR = data.DATA_DIR.parent / "reports"

NAMES = {
    "S01_FUNDING_FADE": "极端费率反转", "S02_LIQ_REBOUND": "爆仓瀑布接针",
    "S03_SQUEEZE_RIDE": "轧空/轧多动量", "S04_SMART_DUMB_DIV": "大户散户背离",
    "S05_OI_TREND_RIDE": "OI确认趋势", "S06_CVD_FADE": "CVD背离反转",
    "S07_EXCH_FLOW_ALERT": "交易所大额资金流", "S08_CB_PREMIUM_US": "Coinbase溢价美盘",
    "S09_EVENT_BREAKOUT": "宏观数据突破", "S10_XSEC_ROTATE": "横截面轮动",
    "S11_CRASH_SCALP": "高频接针",
    "S12_SPIKE_RIDE": "秒级动量延续", "S13_SQUEEZE_POP": "波动收缩突破",
    "S14_VWAP_REVERT": "VWAP偏离回归",
}

RELAXATIONS = {
    "S01_FUNDING_FADE": ["费率/多空比数据与实盘同源,无放宽。"],
    "S02_LIQ_REBOUND": [
        "爆仓单流 10 分钟衰减条件恒视为通过(liq_orders 无历史)。",
        "订单簿买盘回补条件恒视为通过(深度无历史)。",
        "聚合爆仓为已完结 1h 桶:信号最多延迟 1 小时,实盘(滚动桶)触发更及时 → 回测偏保守、触发偏少。"],
    "S03_SQUEEZE_RIDE": ["聚合爆仓为已完结 1h 桶(同 S02,回测偏保守)。"],
    "S04_SMART_DUMB_DIV": [
        "Binance 多空比仅保留 30 天,无法 warmup → 回测从窗口第 8 天开始(前 7 天 z 分数窗口不足)。"],
    "S05_OI_TREND_RIDE": [
        "CVD 使用修正后的字段(实盘 f_cvd 有字段名 bug,见 SUMMARY 重要发现)。",
        "基差用 premiumIndexKlines 真实历史,无放宽。"],
    "S06_CVD_FADE": [
        "订单簿大额挂单墙恒视为存在(ob_wall 无历史)。",
        "盘口深度比用 taker 买卖比代理(方向语义一致,数值近似)。",
        "CVD 使用修正后的字段(实盘 bug)。"],
    "S08_CB_PREMIUM_US": [
        "Coinbase 溢价为已完结 1h 桶:转正/转负信号最多延迟 1 小时。",
        "ETF 流按 T+1 可见(数据日次日生效)。"],
    "S09_EVENT_BREAKOUT": [
        "经济日历使用修正后的字段(实盘 f_econ_cal 有字段名 bug)。",
        "30 天内高重要性事件极少,本策略仅做参数敏感性观察,不构成优化结论。"],
    "S10_XSEC_ROTATE": [
        "资金净流因子无历史 → 该项得分恒 0(其余三因子权重相对比例不变,排序不受影响)。",
        "代币解锁过滤跳过(无历史)。",
        "注意:代码中不存在 META 所述『下次调仓整体换仓』逻辑,持仓仅靠 24h 时限/止损退出(实盘行为相同,建议修复)。"],
    "S11_CRASH_SCALP": [
        "秒级价格由 aggTrades 重建(1s 采样),触发与出场均按秒级回放。",
        "接针场景滑点按 2 倍计。",
        "drop_pct 网格下限抬至 1.5%(更低的『暴跌』在小币属常态噪声,非清算瀑布)。"],
    "S14_VWAP_REVERT": [
        "VWAP 由 1m K线滚动重建(60 根典型价×量),与实盘同源同算法。",
        "taker 衰竭确认用 Binance 5m taker 比(完结桶),无放宽。"],
}


def _verdict(r):
    if r.get("skipped"):
        return "❌ 无法回测", r.get("skip_reason", "")
    p = r["picked"]
    if r["insufficient_sample"]:
        return "⚠️ 样本不足", f"IS 成交 {p['is']['trades']} 笔 < {r['min_trades_required']},结论不可信"
    if p["is"]["net_usd"] <= 0:
        return "❌ 不建议启用", "样本内即为负收益"
    if p["oos"]["net_usd"] <= 0:
        return "❌ 不建议启用", "样本外验证为负收益(疑似过拟合或市况依赖)"
    reasons = []
    if r["oos_decay_pct"] is not None and r["oos_decay_pct"] > 60:
        reasons.append(f"OOS 日均收益衰减 {r['oos_decay_pct']}%")
    if r["consistency_pct"] < 50:
        reasons.append(f"多币一致性仅 {r['consistency_pct']:.0f}%")
    if p["oos"]["trades"] < 5:
        reasons.append(f"OOS 仅 {p['oos']['trades']} 笔")
    if reasons:
        return "⚠️ 谨慎观察", ";".join(reasons)
    return "✅ 可小仓启用", "IS/OOS 双正、参数高原稳健"


def _mtable(rows):
    head = ("| 场景 | 交易数 | 净利($) | 收益率 | 最大回撤($) | 胜率 | 盈亏比 | Sharpe | 均持仓(h) | 费用($) | 资金费($) |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    out = head
    for label, m in rows:
        out += (f"| {label} | {m['trades']} | {m['net_usd']} | {m['net_return_pct']}% "
                f"| {m['max_dd_usd']} | {m['win_rate']}% | {m['profit_factor']} "
                f"| {m['sharpe']} | {m['avg_hold_h']} | {m['fees_usd']} | {m['funding_usd']} |\n")
    return out


def _fmt_ts(ts):
    return datetime.datetime.utcfromtimestamp(ts).strftime("%m-%d %H:%M")


def strategy_report(r):
    sid = r["sid"]
    verdict, why = _verdict(r)
    md = [f"# {sid} {NAMES.get(sid, '')} — 回测报告",
          f"\n**结论:{verdict}** — {why}\n"]
    if r.get("skipped"):
        md.append(r.get("skip_reason", ""))
        return "\n".join(md)

    p = r["picked"]
    md.append("## 指标对比(IS = 前 21 天,OOS = 后 9 天,每笔名义 $1000)\n")
    md.append(_mtable([("默认参数 IS", r["default"]["is"]),
                       ("默认参数 OOS", r["default"]["oos"]),
                       ("优化参数 IS", p["is"]),
                       ("优化参数 OOS", p["oos"])]))
    if r["oos_decay_pct"] is not None:
        md.append(f"\nIS→OOS 日均收益衰减:**{r['oos_decay_pct']}%**"
                  "(>60% 视为过拟合警示)\n")
    md.append(f"多币一致性(有成交币种中正收益占比):**{r['consistency_pct']:.0f}%**\n")

    md.append("## 优化参数(vs 默认)\n\n```yaml")
    md.append(yaml.safe_dump(p["params"], allow_unicode=True, sort_keys=False).strip())
    md.append("```\n")
    md.append(f"选择依据:参数高原得分 {p['plateau']}(邻域中位数),单点得分 {p['score']};"
              "拒绝孤立尖峰,取稳健高原。\n")

    md.append("## 参数敏感性(高原得分 Top 10)\n")
    md.append("| params | IS净利($) | 回撤($) | 交易数 | 单点分 | 高原分 |\n|---|---|---|---|---|---|")
    for row in r["ranked"][:10]:
        ps = ", ".join(f"{k}={v}" for k, v in row["params"].items()
                       if k in str(r["grid"]))
        m = row["metrics"]
        md.append(f"| {ps} | {m['net_usd']} | {m['max_dd_usd']} | {m['trades']} "
                  f"| {row['score']} | {row['plateau']} |")
    md.append("")

    by = p.get("by_symbol") or {}
    if by:
        md.append("## 逐币分解(优化参数,IS)\n")
        md.append("| 币种 | 交易数 | 净利($) |\n|---|---|---|")
        for s, v in sorted(by.items(), key=lambda kv: -kv[1]["pnl"]):
            md.append(f"| {s} | {v['n']} | {v['pnl']} |")
        md.append("")

    trades = r["full_window"]["trades"]
    if trades:
        md.append(f"## 交易明细(优化参数,全窗口,共 {len(trades)} 笔,展示前 50)\n")
        md.append("| 开仓(UTC) | 币种 | 方向 | 开仓价 | 平仓价 | 净利($) | 持仓 | 出场原因 |\n"
                  "|---|---|---|---|---|---|---|---|")
        for t in trades[:50]:
            md.append(f"| {_fmt_ts(t['entry_ts'])} | {t['symbol']} | {t['side']} "
                      f"| {t['entry_price']:.6g} | {t['exit_price']:.6g} "
                      f"| {t['pnl']:.2f} | {t['hold_sec']//3600}h{t['hold_sec']%3600//60}m "
                      f"| {t['reason_exit']} |")
        md.append("")

    md.append("## 数据与放宽条件(可信度标注)\n")
    for line in RELAXATIONS.get(sid, ["无放宽。"]):
        md.append(f"- {line}")
    md.append("\n> 本回测窗口仅 30 天、单一市场状态;结论为『当前市况校准』,"
              "非普适参数。启用前建议 paper 观察 3-7 天对照回测预期。")
    return "\n".join(md)


def summary_report(results, outdir):
    md = ["# 量化策略回测总览(30 天)",
          f"\n窗口:2026-06-18 → 2026-07-18(IS 前 21 天 / OOS 后 9 天),"
          f"每笔固定名义 $1000,taker 费 0.05%/边 + 滑点(大币 1bp/小币 2.5bp/S11×2)"
          f"+ 真实资金费率。\n",
          "## 横向对比\n",
          "| 策略 | 结论 | IS净利($) | OOS净利($) | OOS交易 | 衰减 | 一致性 |",
          "|---|---|---|---|---|---|---|"]
    for sid in ORDER:
        r = results.get(sid)
        if not r:
            continue
        verdict, _ = _verdict(r)
        if r.get("skipped"):
            md.append(f"| {sid} {NAMES.get(sid,'')} | {verdict} | – | – | – | – | – |")
            continue
        p = r["picked"]
        decay = f"{r['oos_decay_pct']}%" if r["oos_decay_pct"] is not None else "–"
        md.append(f"| {sid} {NAMES.get(sid,'')} | {verdict} | {p['is']['net_usd']} "
                  f"| {p['oos']['net_usd']} | {p['oos']['trades']} | {decay} "
                  f"| {r['consistency_pct']:.0f}% |")
    md += ["", "## ⚠️ 回测过程中发现的实盘 bug(建议尽快修复)\n",
           "1. **`factors.f_cvd` 字段名不匹配**:Clawby CVD 接口返回 `cum_vol_delta`,"
           "代码读取 `close`/`cvd` → 实盘 CVD 因子恒为 0 → **S06 永远不会触发、"
           "S05 的 CVD 确认恒为真**。修复:字段列表加入 `cum_vol_delta`。",
           "2. **`factors.f_econ_cal` 字段名不匹配**:接口返回 `publish_timestamp`/"
           "`importance_level`,代码读取 `time`/`importance` → 经济日历恒为空 → "
           "**S09 永不触发、全局事件静默风控失效**。",
           "3. **universe 僵尸符号**:`PEPEUSDT`、`MATICUSDT` 在 Binance 合约不存在"
           "(应为 `1000PEPEUSDT`;MATIC 已更名 POL)→ 实盘对这两个币的行情/因子请求"
           "持续 400,策略层静默跳过。建议在配置页修正。",
           "4. **S10 缺换仓平仓逻辑**:META 描述『下次调仓整体换仓』,但代码没有平旧仓"
           "动作,持仓只能靠 24h 时限或止损退出,与设计意图不符。",
           "5. **`1000SHIBUSDT` 的 Clawby 因子缺失**:`factors._coin` 生成 `1000SHIB`,"
           "Clawby 需要 `SHIB` → 该币的聚合费率/爆仓/CVD 因子实盘恒为空(回测保持同样行为)。\n"]

    md += ["## 100 USDT 实盘落地换算\n",
           "回测每笔名义 $1000;你的合约钱包 100 USDT、3x 杠杆下单笔名义上限约 $300。"
           "启用策略时建议在风控页设置:`capital_usd: 100`、`max_position_usd: 250-300`、"
           "`risk_per_trade_pct: 0.5-1`;预期绝对收益按名义比例约为报告数值的 25-30%,"
           "手续费占比不变。多策略同时启用时注意 `max_concurrent_coins` 与总杠杆约束。\n",
           "## 总体声明\n",
           "- 30 天窗口只覆盖一种市场状态,所有结论为当前市况校准,不代表长期期望。",
           "- 出场按 1 分钟 bar 悲观规则(同 bar 止损优先)、S11 按秒级回放;"
           "策略自定义退出按 5 分钟粒度检查。",
           "- 组合层风控(并发上限、总杠杆、事件静默、恐贪缩放)未参与单策略回测;"
           "多策略叠加的相互作用需另行评估。"]
    (outdir / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")


def suggested_yaml(results, outdir):
    cfg = yaml.safe_load(config.STRATEGIES_PATH.read_text(encoding="utf-8"))
    header = ["# strategies.suggested.yaml — 回测优化建议(2026-07-18)",
              "# 由 backend.backtest 生成;enabled 全部保持 false,由你审阅后自行启用。",
              "# 结论速览:"]
    for sid in ORDER:
        r = results.get(sid)
        if not r:
            continue
        verdict, why = _verdict(r)
        header.append(f"#   {sid}: {verdict} — {why}")
        if not r.get("skipped"):
            cfg["strategies"][sid]["params"].update(r["picked"]["params"])
        cfg["strategies"][sid]["enabled"] = False
        cfg["strategies"][sid]["risk"].update(
            {"capital_usd": 100, "max_position_usd": 300})
    body = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    (outdir / "strategies.suggested.yaml").write_text(
        "\n".join(header) + "\n" + body, encoding="utf-8")


def write_all():
    results = {}
    for sid in ORDER:
        p = RAW_DIR / f"{sid}.json"
        if p.exists():
            results[sid] = json.loads(p.read_text())
    day = datetime.date.today().isoformat()
    outdir = REPORT_DIR / day
    outdir.mkdir(parents=True, exist_ok=True)
    for sid in ORDER:
        if sid in results:
            (outdir / f"{sid}.md").write_text(strategy_report(results[sid]),
                                              encoding="utf-8")
    summary_report(results, outdir)
    suggested_yaml(results, outdir)
    return outdir

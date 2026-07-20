# 量化因子库 — Binance 全自动交易 Bot

> 数据双通道:**Clawby 聚合 API**(`POST https://api.openclawby.com/api/relay`,`X-API-Key` 认证,admin plan / 360 次/分钟)+ **Binance 官方接口**(经 Clawby Binance 执行器 `binance-cli` 本地调用,自带 key,不消耗 Clawby 额度)。
>
> **通道分工原则**:高频行情类因子(K线/订单簿/最新费率)走 Binance 官方——免费、延迟最低、就是执行所在交易所;跨所聚合、链上、情绪、周期类因子走 Clawby——Binance 自己没有这些数据。
>
> 每个因子标注:代号 · 名称 · 说明(衡量什么/怎么用)· 数据来源(接口级)· 建议扫描频率。
> 频率设计依据:上游数据更新粒度 + 因子时效价值;已核算总调用量远低于限速(见文末预算)。

---

## A. 价格与技术面(执行盘口的基础行情)

| 代号 | 因子名 | 说明 | 数据来源 | 扫描频率 |
|---|---|---|---|---|
| `PX_SPOT_KLINE` | 现货K线 | 多周期 OHLCV,一切技术因子的原料(动量/波动率/突破在本地由它衍生) | **Binance 官方** `binance-cli spot klines --symbol BTCUSDT --interval 1m/1h/…` | 1m(随最小交易周期) |
| `PX_FUT_KLINE` | 合约K线 | U 本位合约 OHLCV,合约策略的主行情 | **Binance 官方** `futures-usds kline-candlestick-data` | 1m |
| `PX_MARK` | 标记价格+实时费率 | 标记价、指数价、下一期预测资金费率,爆仓风控与费率抢跑用 | **Binance 官方** `futures-usds mark-price` | 30s |
| `PX_PREMIUM` | 溢价指数K线 | 合约相对指数的溢价路径,短线多空情绪的高频代理 | **Binance 官方** `futures-usds premium-index-kline-data` | 1m |
| `PX_BASIS` | 期现基差 | 基差率(合约-现货),正基差扩大=杠杆做多拥挤,负基差=恐慌贴水 | **Binance 官方** `futures-usds basis`;跨所版 Clawby `futures_basis_history` | 5m |
| `TECH_RSI` | RSI(单币) | 超买超卖振荡器,现成计算免去本地实现 | **Clawby** `futures_indicators_rsi`(exchange=Binance, symbol=BTCUSDT, interval) | 15m |
| `TECH_MA_EMA_BOLL_MACD_ATR` | 均线族/布林/MACD/ATR | 趋势、通道、动能、波动率的标准技术指标族 | **Clawby** `futures_indicators_ma` / `_ema` / `_boll` / `_macd` / `_avg_true_range` | 15m |
| `TECH_RSI_XSEC` | 全市场RSI横截面 | 一次拉全币种 RSI 快照,横截面选出极端超买/超卖币,择币用 | **Clawby** `futures_rsi_list`(免参数) | 15m |
| `TECH_ATR_XSEC` | 全市场ATR横截面 | 全币种波动率快照,用于仓位归一化与高波动筛选 | **Clawby** `futures_avg_true_range_list` | 1h |

## B. 衍生品结构(资金费率 / 持仓 / 爆仓 / 期权)

| 代号 | 因子名 | 说明 | 数据来源 | 扫描频率 |
|---|---|---|---|---|
| `FUND_BINANCE` | 币安资金费率历史 | 结算费率序列,持续高正费率=多头拥挤(反向信号/套费机会) | **Binance 官方** `futures-usds get-funding-rate-history` | 15m |
| `FUND_AGG_OIW` | 全网OI加权费率 | 跨所按持仓加权的聚合费率,比单所更抗操纵,拥挤度主指标 | **Clawby** `futures_funding_rate_oi_weight_history`(symbol=BTC ⚠️币名格式) | 15m |
| `FUND_XSEC` | 全网费率快照 | 所有币在各所的当前费率,横截面找极端费率币(费率反转策略) | **Clawby** `futures_funding_rate_exchange_list`(免参数) | 15m |
| `FUND_ACCUM` | 累计费率排行 | 1d/7d/30d 累计费率排名,识别长期单边拥挤标的 | **Clawby** `futures_funding_rate_accumulated_exchange_list`(range=7d) | 1h |
| `FUND_ARB` | 跨所费率套利 | 现成的跨所费率价差套利机会列表(含本金测算) | **Clawby** `futures_funding_rate_arbitrage`(usd=10000 ⚠️数字不加引号) | 30m |
| `OI_BINANCE` | 币安持仓量 | 单所 OI 即时值+历史统计;OI 升价升=趋势健康,OI 升价横=蓄势 | **Binance 官方** `futures-usds open-interest` / `open-interest-statistics` | 1m / 5m |
| `OI_AGG` | 全网聚合持仓 | 跨所 OI 总量历史,全市场杠杆水位计 | **Clawby** `futures_open_interest_aggregated_history`(symbol=BTC) | 15m |
| `OI_XSEC` | OI交易所分布 | 单币 OI 在各所分布快照,识别主力所与迁移 | **Clawby** `futures_open_interest_exchange_list` | 1h |
| `LIQ_AGG` | 聚合爆仓量 | 跨所多空爆仓金额序列;巨量空爆=轧空燃料耗尽,巨量多爆=瀑布 | **Clawby** `futures_liquidation_aggregated_history`(exchange_list+symbol=BTC) | 5m |
| `LIQ_XSEC` | 全币爆仓快照 | 24h 内各币爆仓排行,横截面找刚被清洗过的标的(反弹备选) | **Clawby** `futures_liquidation_exchange_list`(range=24h) | 15m |
| `LIQ_MAP` | 爆仓地图/热力图 | 价格×清算密度分布,上下方"磁吸位"=短线目标位与止损参考 | **Clawby** `futures_liquidation_aggregated_map`(range=7d)/ `futures_liquidation_aggregated_heatmap_model1` | 1h |
| `LIQ_ORDERS` | 大额爆仓单流 | ≥10万U 的单笔爆仓实录,瀑布/轧空的实时确认 | **Clawby** `futures_liquidation_order`(min_liquidation_amount=100000) | 5m |
| `LSR_GLOBAL` | 全局多空人数比 | 散户情绪代理(人数比),极端值反向用 | **Binance 官方** `futures-usds long-short-ratio`;跨所 Clawby `futures_global_long_short_account_ratio_history` | 15m |
| `LSR_TOP_ACC` | 大户多空人数比 | 头部账户人数倾斜 | **Binance 官方** `futures-usds top-trader-long-short-ratio-accounts` | 15m |
| `LSR_TOP_POS` | 大户多空持仓比 | 头部账户真金白银的持仓倾斜,比人数比含金量高,大户-散户背离是经典信号 | **Binance 官方** `futures-usds top-trader-long-short-ratio-positions`;Clawby `futures_top_long_short_position_ratio_history` | 15m |
| `LSR_HL` | Hyperliquid多空比 | 链上永续人群的多空比,与 CEX 人群互补 | **Clawby** `hyperliquid_global_long_short_account_ratio_history`(symbol=BTC ⚠️无exchange) | 30m |
| `OPT_MAXPAIN` | 期权最大痛点 | 期权卖方利益最大化价位,到期日前的价格"引力位" | **Clawby** `option_max_pain`(symbol=BTC, exchange=Deribit) | 1h |
| `OPT_IV_PCR` | 期权IV与PCR | 隐含波动率+看跌看涨比,恐慌定价与尾部风险预警 | **Clawby** `option_info`(symbol=BTC) | 1h |
| `OPT_FUT_RATIO` | 期权/期货OI比 | 期权市场相对期货的体量变化,机构对冲需求代理 | **Clawby** `index_option_vs_futures_oi_ratio`(免参数) | 1d |
| `BORROW_RATE` | 借贷利率 | 杠杆现货做多成本,利率飙升=现货杠杆拥挤 | **Clawby** `borrow_interest_rate_history`(symbol=BTC ⚠️币名格式例外) | 1h |

## C. 订单流与微观结构

| 代号 | 因子名 | 说明 | 数据来源 | 扫描频率 |
|---|---|---|---|---|
| `OB_DEPTH` | 订单簿深度失衡 | bid/ask 量比与斜率,短线方向与滑点预估(执行前必查) | **Binance 官方** `spot depth` / `futures-usds depth` | 30s(交易时)|
| `OB_WALL` | 大额挂单墙 | 当前巨额限价单位置=人造支撑阻力;历史版可回看撤单行为 | **Clawby** `futures_orderbook_large_limit_order`(exchange=Binance, symbol=BTCUSDT)+ `spot_orderbook_large_limit_order` | 5m |
| `OB_HIST` | 盘口挂单历史 | ±2% 范围内 bid/ask 量的时间序列,盘口厚度趋势 | **Clawby** `futures_orderbook_ask_bids_history` | 30m |
| `TAKER_FLOW` | 主动买卖比 | taker buy/sell 量比,进攻性资金方向(现货+合约两套) | **Binance 官方** `futures-usds taker-buy-sell-volume`;跨所 Clawby `futures_aggregated_taker_buy_sell_volume_history` | 5m |
| `CVD` | 累积成交量差 | 主动买卖差的累积路径;价新高而 CVD 不新高=背离顶信号 | **Clawby** `futures_cvd_history` / `spot_cvd_history`(exchange=Binance, symbol=BTCUSDT) | 15m |
| `NETFLOW` | 合约资金净流 | 单币在各所的合约资金净流入/出明细 | **Clawby** `futures_coin_netflow`(symbol=BTC, exchange_list)| 15m |
| `NETFLOW_XSEC` | 净流排行榜 | 全币种合约/现货净流排行,横截面资金轮动信号 | **Clawby** `futures_netflow_list` + `spot_netflow_list` | 15m |
| `SPOT_FUT_VOL` | 期现成交比 | 合约/现货成交比,投机热度计;比值极端=杠杆泡沫 | **Clawby** `futures_spot_volume_ratio`(exchange_list+symbol=BTC) | 1h |
| `WHALE_IDX` | 鲸鱼行为指数 | 交易所大单行为合成指数(CoinGlass 模型) | **Clawby** `futures_whale_index_history`(exchange=Binance, symbol=BTCUSDT) | 1h |
| `AGG_TRADES` | 大单成交流 | 逐笔聚合成交,本地过滤大单占比/方向(自建大单因子) | **Binance 官方** `spot agg-trades` / `futures-usds compressed-aggregate-trades-list` | 1m(交易时)|

## D. 链上与资金流

| 代号 | 因子名 | 说明 | 数据来源 | 扫描频率 |
|---|---|---|---|---|
| `EXCH_BALANCE` | 交易所链上余额 | 各所 BTC/ETH 链上储备变化;持续流出=囤币(看多),大额流入=潜在抛压 | **Clawby** `exchange_balance_chart`(symbol=BTC)/ `exchange_balance_list` | 1h |
| `WHALE_TX` | 链上鲸鱼转账 | 大额链上转账实录(含交易所方向标注) | **Clawby** `chain_v2_whale_transfer` | 15m |
| `EXCH_CHAIN_TX` | 交易所大额出入金 | ERC-20 ≥100万U 的交易所出入金单据流 | **Clawby** `exchange_chain_tx_list`(min_usd=1000000) | 15m |
| `STABLE_MCAP` | 稳定币总市值 | 场内"弹药"总量;稳定币扩张=购买力积蓄 | **Clawby** `index_stablecoin_marketcap_history`(免参数) | 1d |
| `UNLOCK` | 代币解锁日程 | 未来解锁时间与规模,解锁前规避/做空候选(供给冲击) | **Clawby** `coin_unlock_list` + `coin_vesting`(symbol);辅以 `rootdata_events`(type=13) | 1d |
| `HL_WHALE` | HL鲸鱼开平仓 | Hyperliquid 链上透明大户实时开平仓警报,聪明钱方向参考 | **Clawby** `hyperliquid_whale_alert` / `hyperliquid_whale_position`(免参数) | 5m |
| `HL_POS_DIST` | HL持仓/盈亏分布 | 全网钱包仓位与浮盈浮亏分布,链上版多空温度计 | **Clawby** `hyperliquid_wallet_position_distribution` / `_pnl_distribution` | 1h |

## E. 机构与 ETF 资金

| 代号 | 因子名 | 说明 | 数据来源 | 扫描频率 |
|---|---|---|---|---|
| `ETF_FLOW_BTC` | BTC ETF净流 | 美股现货 ETF 逐日净申赎(分 ticker),机构边际买卖力 | **Clawby** `etf_bitcoin_flow_history`(免参数) | 1d(美股收盘后)|
| `ETF_FLOW_ETH/SOL/XRP` | 其他ETF净流 | ETH/SOL/XRP ETF 逐日净流 | **Clawby** `etf_ethereum_flow_history` / `etf_solana_flow_history` / `etf_xrp_flow_history` | 1d |
| `ETF_PREMIUM` | ETF溢价折价 | ETF 价格对 NAV 溢/折价,机构需求过热/冷却信号 | **Clawby** `etf_bitcoin_premium_discount_history` | 1d |
| `COINBASE_PREM` | Coinbase溢价 | Coinbase 对其他所价差,美国机构/散户净买压代理 | **Clawby** `coinbase_premium_index`(仅 interval ⚠️不传symbol) | 15m |

## F. 情绪与社交

| 代号 | 因子名 | 说明 | 数据来源 | 扫描频率 |
|---|---|---|---|---|
| `FEAR_GREED` | 恐惧贪婪指数 | 市场综合情绪 0-100,极端区反向配仓 | **Clawby** `index_fear_greed_history`(免参数) | 1d |
| `X_BUZZ` | X话题热度 | 按币种关键词搜推文,统计量/情绪斜率;突发放量=事件驱动 | **Clawby** `x_search`(query="$BTC" 等) | 30m |
| `X_KOL` | 关键账号监控 | 监控项目方/头部 KOL 最新发言(公告抢跑) | **Clawby** `x_user_posts`(username) | 15m |
| `REDDIT_HEAT` | Reddit热度 | 按币每日提及计数的时间序列,散户注意力代理 | **Clawby** `Daily Reddit mention counts for a given crypto base`(base=BTC ⚠️参数名base) | 1d |
| `MENTION_RANK` | 提及排行 | 窗口内被讨论最多的标的排行,注意力轮动选币 | **Clawby** `Top mentioned tickers for a given timeframe`(start=unix秒) | 1h |
| `NEWS_FLOW` | 新闻事件流 | 聚合行业新闻(可中文),做事件触发器与黑天鹅监测 | **Clawby** `article_list`(language=zh-CN) | 15m |
| `ROOT_HOT` | RootData热度榜 | 项目研究热度 Top100/增长 Top300,叙事轮动先行指标 | **Clawby** `rootdata_hot_index`(days=1)/ `rootdata_rd_top300` | 1d |

## G. 宏观与周期(低频,做仓位水位调节)

| 代号 | 因子名 | 说明 | 数据来源 | 扫描频率 |
|---|---|---|---|---|
| `CYCLE_AHR999` | AHR999定投指数 | 经典 BTC 抄底/逃顶参考区间 | **Clawby** `index_ahr999` | 1d |
| `CYCLE_PI` | Pi Cycle顶部 | 双均线顶部探测器,历史顶部命中率高 | **Clawby** `index_pi_cycle_indicator` | 1d |
| `CYCLE_PUELL` | Puell Multiple | 矿工收入比率,底部/顶部区域判定 | **Clawby** `index_puell_multiple` | 1d |
| `CYCLE_RAINBOW` | 彩虹图 | 长周期估值色带 | **Clawby** `index_bitcoin_rainbow_chart` | 1d |
| `CYCLE_PEAK` | 牛市顶部综合 | 多指标合成的逃顶仪表盘 | **Clawby** `bull_market_peak_indicator` | 1d |
| `ONCHAIN_SOPR` | 短期/长期SOPR | 链上已实现盈亏比;STH-SOPR 跌破1后收复=洗盘结束经典信号 | **Clawby** `index_bitcoin_sth_sopr` / `index_bitcoin_lth_sopr` | 1d |
| `ONCHAIN_NUPL` | NUPL | 全网未实现盈亏,贪婪/投降区间划分 | **Clawby** `index_bitcoin_net_unrealized_profit_loss` | 1d |
| `ONCHAIN_RHODL` | RHODL比率 | 新老筹码热度比,周期顶部探测 | **Clawby** `index_bitcoin_rhodl_ratio` | 1d |
| `ONCHAIN_STH_COST` | 短期持有者成本 | STH 已实现价格=牛市回调强支撑/熊市反弹阻力 | **Clawby** `index_bitcoin_sth_realized_price` | 1d |
| `ONCHAIN_ADDR` | 活跃/新增地址 | 链上使用热度,基本面动量 | **Clawby** `index_bitcoin_active_addresses` / `index_bitcoin_new_addresses` | 1d |
| `BTC_DOM` | BTC市占率 | 大盘/山寨风格轮动开关 | **Clawby** `index_bitcoin_dominance` | 1d |
| `ALT_SEASON` | 山寨季指数 | 是否处于山寨普涨窗口,决定选币池宽度 | **Clawby** `index_altcoin_season` | 1d |
| `MACRO_M2` | BTC vs M2 | 全球流动性与 BTC 的背离/共振 | **Clawby** `index_bitcoin_vs_global_m2_growth` | 1d |
| `ECON_CAL` | 经济日历 | CPI/FOMC 等事件时刻表;事件前自动降杠杆/停开新仓 | **Clawby** `calendar_economic_data`(language=zh-CN) | 1d(事件前1h警戒)|

## H. 横截面市场扫描(选币池维护)

| 代号 | 因子名 | 说明 | 数据来源 | 扫描频率 |
|---|---|---|---|---|
| `MKT_FUT_SNAP` | 全币合约快照 | 全币种价格/OI/涨跌一张表,横截面因子的底表 | **Clawby** `futures_coins_markets` | 15m |
| `MKT_PRICE_CHG` | 全币涨跌快照 | 多周期涨跌幅横截面,动量/反转选币 | **Clawby** `futures_coins_price_change`(免参数) | 15m |
| `MKT_SPOT_SNAP` | 全币现货快照 | 现货侧全币行情表 | **Clawby** `spot_coins_markets` | 15m |
| `MKT_24H_XSEC` | 币安24h横截面 | 全交易对 24h 统计(量/涨跌/高低),币安本所的选币底表 | **Binance 官方** `spot ticker24hr` / `futures-usds ticker24hr-price-change-statistics` | 15m |
| `DEX_TREND` | 链上热币排行 | DEX 成交/涨幅排行,新叙事币先在链上爆量,对币安新上币有前瞻性 | **Clawby** `dex_trending`(chain=sol, interval=1h) | 1h |
| `DEX_SMART` | 链上聪明钱 | 高胜率钱包近期买入标的(跟踪其转向 CEX 已上币种的信号) | **Clawby** `dex_token_signals` / `dex_smart_money` | 1h |

---

## 调用预算核算(全因子开启)

- **Binance 官方通道**(免费,独立 weight 限额 6000/min,用量 ≪ 1%):30s-1m 级行情因子全部走此通道,约 20-40 请求/分钟。
- **Clawby 通道**(admin plan,360 次/分钟):
  - 5m 级 4 个接口 ≈ 0.8 次/分钟
  - 15m 级 ~18 个接口 ≈ 1.2 次/分钟
  - 30m-1h 级 ~15 个接口 ≈ 0.4 次/分钟
  - 1d 级 ~25 个接口 ≈ 忽略不计
  - **合计 < 3 次/分钟**,监控多币种(如 10 个)时按币数线性放大,仍 < 30 次/分钟,余量充足。

## 参数格式速查(高频踩坑点)

1. **币名 vs 交易对**:聚合类接口(`*_aggregated_*`、Hyperliquid、期权)用 `BTC`;单所接口用 `BTCUSDT`。传错报 "pair does not exist"。
2. `exchange_list` 是逗号分隔字符串 `"Binance,OKX"`,不是数组。
3. `futures_price_history` / `futures_orderbook_history` 的 `limit` 是**必填**。
4. 时间戳:Clawby 用 Unix **毫秒**;binance-cli `--start-time/--end-time` 也是**毫秒**;Reddit/OHLC bars 接口的 `start` 是**秒**。
5. Clawby 响应外层 `{source, data, credits}`,衍生品类内层还有 `{code, data}`,`code=="0"` 才是成功。
6. `borrow_interest_rate_history` 与 `coinbase_premium_index` 是格式例外(见表内标注)。

## 执行通道(交易下单,非因子)

- 现货:`binance-cli spot new-order --symbol BTCUSDT --side BUY --type MARKET --quantity 0.001`
- U 本位合约:`binance-cli futures-usds new-order ...`
- 撤单/查单:`spot delete-order` / `spot get-open-orders`
- 凭据经环境变量注入(见 `.env`),仅本地使用,不经过 Clawby 服务器。

"""Factor collection: declarative registry + due-based scheduler.

Each factor = (name, interval_sec, per_symbol, fetch coroutine). The engine
calls collect_due() every tick; only stale factors are refreshed. Values are
stored in SQLite (latest + history) for strategies and the dashboard.

Factor codes follow FACTORS.md.
"""
import asyncio
import logging
import time

from . import binance, clawby, config, db

log = logging.getLogger("factors")


def _coin(symbol):
    s = symbol.replace("USDT", "").replace("BUSD", "")
    # 1000-multiplied contracts (1000SHIB, 1000PEPE) -> aggregator coin name
    return s[4:] if s.startswith("1000") and len(s) > 4 else s


# -- fetchers (each returns the value to store, or None to skip) -------------

async def f_mark_all(_):
    """All-symbol mark price / index / funding — position pricing + basis."""
    rows = await binance.mark_price()
    out = {}
    for r in rows if isinstance(rows, list) else [rows]:
        sym = r.get("symbol", "")
        if sym in config.UNIVERSE:
            mark, idx = float(r.get("markPrice") or 0), float(r.get("indexPrice") or 0)
            out[sym] = {"mark": mark, "index": idx,
                        "basis_pct": (mark - idx) / idx * 100 if idx else 0,
                        "funding_next": float(r.get("lastFundingRate") or 0)}
    return out


async def f_funding_agg(symbol):
    data = await clawby.relay_safe("futures_funding_rate_oi_weight_history",
                                   {"symbol": _coin(symbol), "interval": "1h", "limit": 24})
    if not data:
        return None
    closes = [float(d.get("close") or 0) for d in data]
    return {"latest": closes[-1], "series": closes[-24:]}


async def f_funding_xsec(_):
    data = await clawby.relay_safe("futures_funding_rate_exchange_list")
    return data if data else None


async def f_lsr_global(symbol):
    rows = await binance.long_short_global(symbol, "1h", 168)
    if not rows:
        return None
    series = [float(r["longShortRatio"]) for r in rows]
    return {"latest": series[-1], "series": series}


async def f_lsr_top_pos(symbol):
    rows = await binance.long_short_top_positions(symbol, "1h", 168)
    if not rows:
        return None
    series = [float(r["longShortRatio"]) for r in rows]
    return {"latest": series[-1], "series": series}


async def f_taker_flow(symbol):
    rows = await binance.taker_ratio(symbol, "5m", 48)
    if not rows:
        return None
    series = [float(r["buySellRatio"]) for r in rows]
    return {"latest": series[-1], "series": series}


async def f_oi_binance(symbol):
    rows = await binance.open_interest_hist(symbol, "1h", 48)
    if not rows:
        return None
    series = [float(r["sumOpenInterestValue"]) for r in rows]
    chg_1h = (series[-1] / series[-2] - 1) * 100 if len(series) >= 2 and series[-2] else 0
    return {"latest": series[-1], "chg_1h_pct": chg_1h, "series": series}


async def f_oi_agg(symbol):
    data = await clawby.relay_safe("futures_open_interest_aggregated_history",
                                   {"symbol": _coin(symbol), "interval": "1h", "limit": 24})
    if not data:
        return None
    closes = [float(d.get("close") or 0) for d in data]
    slope = (closes[-1] / closes[-4] - 1) * 100 if len(closes) >= 4 and closes[-4] else 0
    return {"latest": closes[-1], "slope_3h_pct": slope}


async def f_liq_agg(symbol):
    data = await clawby.relay_safe(
        "futures_liquidation_aggregated_history",
        {"exchange_list": "Binance,OKX,Bybit", "symbol": _coin(symbol),
         "interval": "1h", "limit": 168})
    if not data:
        return None
    longs = [float(d.get("aggregated_long_liquidation_usd")
                   or d.get("longLiquidationUsd") or d.get("long") or 0) for d in data]
    shorts = [float(d.get("aggregated_short_liquidation_usd")
                    or d.get("shortLiquidationUsd") or d.get("short") or 0) for d in data]
    avg_l = sum(longs[:-1]) / max(len(longs) - 1, 1)
    avg_s = sum(shorts[:-1]) / max(len(shorts) - 1, 1)
    return {"long_1h": longs[-1], "short_1h": shorts[-1],
            "long_mult": longs[-1] / avg_l if avg_l else 0,
            "short_mult": shorts[-1] / avg_s if avg_s else 0}


async def f_liq_orders(symbol):
    data = await clawby.relay_safe(
        "futures_liquidation_order",
        {"exchange": "Binance", "symbol": symbol, "min_liquidation_amount": 100000})
    if data is None:
        return None
    rows = data if isinstance(data, list) else []
    now_ms = time.time() * 1000
    recent = [r for r in rows if now_ms - float(r.get("time") or r.get("ts") or 0) < 600_000]
    older = [r for r in rows if 600_000 <= now_ms - float(r.get("time") or r.get("ts") or 0) < 1_800_000]
    return {"n_10m": len(recent), "n_prev_20m": len(older)}


async def f_liq_map(symbol):
    data = await clawby.relay_safe("futures_liquidation_aggregated_map",
                                   {"symbol": _coin(symbol), "range": "7d"})
    return {"raw_present": bool(data)} if data is None else {"data": data}


async def f_ob_wall(symbol):
    data = await clawby.relay_safe("futures_orderbook_large_limit_order",
                                   {"exchange": "Binance", "symbol": symbol})
    if data is None:
        return None
    return {"orders": data if isinstance(data, list) else data}


async def f_cvd(symbol):
    data = await clawby.relay_safe("futures_cvd_history",
                                   {"exchange": "Binance", "symbol": symbol,
                                    "interval": "1h", "limit": 48})
    if not data:
        return None
    # upstream field is cum_vol_delta (close/cvd kept for compatibility)
    vals = [float(d.get("cum_vol_delta") or d.get("close") or d.get("cvd") or 0)
            for d in data]
    return {"latest": vals[-1], "series": vals}


async def f_coinbase_prem(_):
    data = await clawby.relay_safe("coinbase_premium_index", {"interval": "1h", "limit": 48})
    if not data:
        return None
    vals = [float(d.get("premium_rate") or d.get("premium") or d.get("close") or 0) for d in data]
    return {"latest": vals[-1], "series": vals}


async def f_etf_flow_btc(_):
    data = await clawby.relay_safe("etf_bitcoin_flow_history")
    if not data:
        return None
    rows = data if isinstance(data, list) else []
    last = rows[-1] if rows else {}
    total = last.get("flow_usd") or last.get("total") or last.get("changeUsd") or 0
    try:
        total = float(total)
    except (TypeError, ValueError):
        total = 0
    return {"last_day_flow_usd": total}


async def f_fear_greed(_):
    data = await clawby.relay_safe("index_fear_greed_history")
    if not data:
        return None
    # shape: {"data_list": [values...], "time_list": [...]} (ascending)
    if isinstance(data, dict) and isinstance(data.get("data_list"), list) and data["data_list"]:
        try:
            return {"latest": float(data["data_list"][-1]),
                    "series": [float(x) for x in data["data_list"][-30:]]}
        except (TypeError, ValueError):
            return None
    return None


async def f_econ_cal(_):
    now_ms = int(time.time() * 1000)
    data = await clawby.relay_safe("calendar_economic_data",
                                   {"start_time": now_ms - 86400_000,
                                    "end_time": now_ms + 3 * 86400_000, "language": "zh-CN"})
    if data is None:
        return None
    rows = data if isinstance(data, list) else []
    events = []
    for r in rows:
        # upstream fields: publish_timestamp (ms) / calendar_name / importance_level
        ts = (r.get("publish_timestamp") or r.get("time")
              or r.get("timestamp") or r.get("date"))
        title = str(r.get("calendar_name") or r.get("title") or r.get("name") or "")
        imp = str(r.get("importance_level") or r.get("importance") or r.get("star") or "")
        if ts:
            events.append({"ts": int(float(ts) / (1000 if float(ts) > 4e10 else 1)),
                           "title": title[:80], "importance": imp})
    return {"events": events[:50]}


async def f_exch_chain_tx(_):
    data = await clawby.relay_safe("exchange_chain_tx_list", {"min_usd": 50_000_000})
    if data is None:
        return None
    rows = data if isinstance(data, list) else []
    return {"txs": rows[:30]}


async def f_whale_tx(_):
    data = await clawby.relay_safe("chain_v2_whale_transfer")
    if data is None:
        return None
    rows = data if isinstance(data, list) else []
    return {"txs": rows[:30]}


async def f_mkt_price_chg(_):
    data = await clawby.relay_safe("futures_coins_price_change")
    return {"rows": data} if data else None


async def f_netflow_xsec(_):
    data = await clawby.relay_safe("futures_netflow_list", {"per_page": 60})
    return {"rows": data} if data else None


async def f_unlock(_):
    data = await clawby.relay_safe("coin_unlock_list", {"per_page": 50})
    return {"rows": data} if data else None


async def f_alt_season(_):
    data = await clawby.relay_safe("index_altcoin_season")
    if not data:
        return None
    rows = data.get("data") if isinstance(data, dict) else data
    if isinstance(rows, list) and rows:
        try:
            return {"latest": float(rows[-1].get("altcoin_index") or rows[-1].get("value") or 0)}
        except (TypeError, ValueError, AttributeError):
            return None
    if isinstance(rows, dict):
        try:
            return {"latest": float(rows.get("altcoin_index") or rows.get("value") or 0)}
        except (TypeError, ValueError):
            return None
    return None


# -- expanded factor fetchers (2026-07-18 full Clawby sweep) ------------------

_EX3 = {"exchange_list": "Binance,OKX,Bybit"}
_MAJORS = {"BTCUSDT", "ETHUSDT"}


def _series_fetcher(api, field, per_coin=True, n=48, extra=None, majors_only=False):
    """Factory: history endpoint -> {latest, series[-n:]}."""
    async def f(symbol):
        if majors_only and symbol not in _MAJORS:
            return None
        p = {"interval": "1h", "limit": max(n, 50), **(extra or {})}
        if per_coin:
            p["symbol"] = _coin(symbol)
        data = await clawby.relay_safe(api, p)
        if not isinstance(data, list) or not data:
            return None
        vals = []
        for d in data:
            try:
                vals.append(float(d.get(field) or 0))
            except (TypeError, ValueError):
                vals.append(0.0)
        return {"latest": vals[-1], "series": vals[-n:]}
    return f


f_funding_vol_weight = _series_fetcher("futures_funding_rate_vol_weight_history", "close", n=24)
f_whale_index = _series_fetcher("futures_whale_index_history", "whale_index_value",
                                per_coin=False, extra={"exchange": "Binance"})
f_oi_stable = _series_fetcher("futures_open_interest_aggregated_stablecoin_history",
                              "close", extra=_EX3)
f_fut_spot_ratio = _series_fetcher("futures_spot_volume_ratio",
                                   "futures_spot_vol_ratio", extra=_EX3, n=24)
f_spot_cvd = _series_fetcher("spot_aggregated_cvd_history", "cum_vol_delta", extra=_EX3)
f_cgdi = _series_fetcher("futures_cgdi_index_history", "cgdi_index_value", per_coin=False)
f_cdri = _series_fetcher("futures_cdri_index_history", "cdri_index_value", per_coin=False)
f_borrow_rate = _series_fetcher("borrow_interest_rate_history", "interest_rate",
                                extra={"exchange": "Binance"}, majors_only=True, n=24)
f_hl_lsr = _series_fetcher("hyperliquid_global_long_short_account_ratio_history",
                           "global_account_long_short_ratio", n=24)


async def _whale_index_pair(symbol):
    """whale_index needs the trading pair, not the coin."""
    data = await clawby.relay_safe("futures_whale_index_history",
                                   {"exchange": "Binance", "symbol": symbol,
                                    "interval": "1h", "limit": 50})
    if not isinstance(data, list) or not data:
        return None
    vals = [float(d.get("whale_index_value") or 0) for d in data]
    return {"latest": vals[-1], "series": vals[-48:]}


async def f_basis_agg(symbol):
    data = await clawby.relay_safe("futures_basis_history",
                                   {"exchange": "Binance", "symbol": symbol,
                                    "interval": "1h", "limit": 50})
    if not isinstance(data, list) or not data:
        return None
    vals = [float(d.get("close_basis") or 0) for d in data]
    return {"latest": vals[-1], "series": vals[-24:]}


async def f_net_position(symbol):
    data = await clawby.relay_safe("futures_v2_net_position_history",
                                   {"exchange": "Binance", "symbol": symbol,
                                    "interval": "1h", "limit": 50})
    if not isinstance(data, list) or not data:
        return None
    last = data[-1]
    return {"latest": float(last.get("net_position_change_cum") or 0),
            "long_chg": float(last.get("net_long_change") or 0),
            "short_chg": float(last.get("net_short_change") or 0),
            "series": [float(d.get("net_position_change_cum") or 0) for d in data][-48:]}


async def f_ob_imbalance(symbol):
    data = await clawby.relay_safe("futures_orderbook_aggregated_ask_bids_history",
                                   {**_EX3, "symbol": _coin(symbol),
                                    "interval": "1h", "limit": 30, "range": "1"})
    if not isinstance(data, list) or not data:
        return None
    last = data[-1]
    bids = float(last.get("aggregated_bids_usd") or 0)
    asks = float(last.get("aggregated_asks_usd") or 0)
    return {"latest": bids / asks if asks else 1.0,
            "bids_usd": bids, "asks_usd": asks}


def _taker_ratio_fetcher(api):
    async def f(symbol):
        data = await clawby.relay_safe(api, {**_EX3, "symbol": _coin(symbol),
                                             "interval": "1h", "limit": 50})
        if not isinstance(data, list) or not data:
            return None
        series = []
        for d in data:
            buy = float(d.get("aggregated_buy_volume_usd") or 0)
            sell = float(d.get("aggregated_sell_volume_usd") or 0)
            series.append(buy / sell if sell else 1.0)
        return {"latest": series[-1], "series": series[-48:]}
    return f


f_taker_agg = _taker_ratio_fetcher("futures_aggregated_taker_buy_sell_volume_history")
f_spot_taker_agg = _taker_ratio_fetcher("spot_aggregated_taker_buy_sell_volume_history")


def _netflow_fetcher(api):
    async def f(symbol):
        data = await clawby.relay_safe(api, {"symbol": _coin(symbol)})
        row = data[0] if isinstance(data, list) and data else data
        if not isinstance(row, dict):
            return None
        out = {}
        for w in ("5m", "15m", "30m", "1h", "4h"):
            v = row.get(f"net_flow_usd_{w}")
            if v is not None:
                try:
                    out[f"flow_{w}"] = float(v)
                except (TypeError, ValueError):
                    pass
        return out or None
    return f


f_netflow_coin = _netflow_fetcher("futures_coin_netflow")
f_spot_netflow_coin = _netflow_fetcher("spot_coin_netflow")


async def f_option_pcr(symbol):
    if symbol not in _MAJORS:
        return None
    data = await clawby.relay_safe("option_max_pain",
                                   {"symbol": _coin(symbol), "exchange": "Deribit"})
    if not isinstance(data, list) or not data:
        return None
    calls = sum(float(d.get("call_open_interest") or 0) for d in data)
    puts = sum(float(d.get("put_open_interest") or 0) for d in data)
    near = data[0] if data else {}
    return {"pcr": puts / calls if calls else 0,
            "max_pain_near": float(near.get("max_pain_price") or 0)}


async def f_bitfinex_margin(symbol):
    if symbol not in _MAJORS:
        return None
    data = await clawby.relay_safe("bitfinex_margin_long_short",
                                   {"symbol": _coin(symbol), "interval": "1h", "limit": 30})
    if not isinstance(data, list) or not data:
        return None
    last = data[-1]
    lq = float(last.get("long_quantity") or 0)
    sq = float(last.get("short_quantity") or 0)
    return {"latest": lq / sq if sq else 0, "long_qty": lq, "short_qty": sq}


def _etf_flow_fetcher(api):
    async def f(_):
        data = await clawby.relay_safe(api)
        rows = data if isinstance(data, list) else []
        if not rows:
            return None
        last = rows[-1]
        try:
            return {"last_day_flow_usd": float(last.get("flow_usd") or 0)}
        except (TypeError, ValueError):
            return None
    return f


f_etf_eth_flow = _etf_flow_fetcher("etf_ethereum_flow_history")
f_etf_sol_flow = _etf_flow_fetcher("etf_solana_flow_history")
f_etf_xrp_flow = _etf_flow_fetcher("etf_xrp_flow_history")


async def f_etf_btc_premium(_):
    data = await clawby.relay_safe("etf_bitcoin_premium_discount_history")
    rows = data if isinstance(data, list) else []
    if not rows:
        return None
    last = rows[-1].get("list") or []
    vals = [float(x.get("premium_discount") or x.get("premium_discount_percent") or 0)
            for x in last if isinstance(x, dict)]
    return {"avg_premium_pct": sum(vals) / len(vals) if vals else 0}


async def f_stablecoin_mcap(_):
    data = await clawby.relay_safe("index_stablecoin_marketcap_history")
    if not isinstance(data, dict) or not data.get("data_list"):
        return None
    vals = []
    for x in data["data_list"]:
        if isinstance(x, dict):          # rows like {"USDT": 304712.76, ...}
            vals.append(sum(v for v in x.values() if isinstance(v, (int, float))))
        else:
            try:
                vals.append(float(x))
            except (TypeError, ValueError):
                continue
    if not vals:
        return None
    chg7 = (vals[-1] / vals[-8] - 1) * 100 if len(vals) >= 8 and vals[-8] else 0
    return {"latest": vals[-1], "chg_7d_pct": chg7}


def _ts_value_fetcher(api, field, n=30):
    async def f(_):
        data = await clawby.relay_safe(api)
        rows = data if isinstance(data, list) else []
        if not rows:
            return None
        vals = []
        for d in rows:
            try:
                vals.append(float(d.get(field) or 0))
            except (TypeError, ValueError):
                continue
        if not vals:
            return None
        return {"latest": vals[-1], "series": vals[-n:]}
    return f


f_btc_dominance = _ts_value_fetcher("index_bitcoin_dominance", "bitcoin_dominance")
f_puell = _ts_value_fetcher("index_puell_multiple", "puell_multiple")
f_sth_sopr = _ts_value_fetcher("index_bitcoin_sth_sopr", "sth_sopr")
f_nupl = _ts_value_fetcher("index_bitcoin_net_unrealized_profit_loss", "net_unpnl")
f_active_addr = _ts_value_fetcher("index_bitcoin_active_addresses", "active_address_count")


async def f_ahr999(_):
    data = await clawby.relay_safe("index_ahr999")
    rows = data if isinstance(data, list) else []
    if not rows:
        return None
    try:
        return {"latest": float(rows[-1].get("ahr999_value") or 0)}
    except (TypeError, ValueError):
        return None


async def f_option_fut_ratio(_):
    data = await clawby.relay_safe("index_option_vs_futures_oi_ratio")
    rows = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
    if not rows:
        return None
    last = rows[-1]
    return {"btc": float(last.get("btc_option_vs_futures_radio") or 0),
            "eth": float(last.get("eth_option_vs_futures_radio") or 0)}


async def f_funding_arb(_):
    data = await clawby.relay_safe("futures_funding_rate_arbitrage",
                                   {"usd_amount": 10000})
    rows = data if isinstance(data, list) else []
    return {"rows": rows[:20]} if rows else None


async def f_hl_whale_alert(_):
    data = await clawby.relay_safe("hyperliquid_whale_alert")
    rows = data if isinstance(data, list) else []
    return {"rows": rows[:30]} if rows else None


async def f_hl_whale_pos(_):
    data = await clawby.relay_safe("hyperliquid_whale_position")
    rows = data if isinstance(data, list) else []
    return {"rows": rows[:30]} if rows else None


# -- registry ---------------------------------------------------------------
# (name, default_interval_sec, per_symbol, fetcher, 中文名, 来源)

REGISTRY = [
    ("mark_all",      30,    False, f_mark_all,      "标记价/基差快照",   "Binance官方", "Mark price/basis snapshot"),
    ("taker_flow",    300,   True,  f_taker_flow,    "主动买卖比",        "Binance官方", "Taker buy/sell ratio"),
    ("oi_binance",    300,   True,  f_oi_binance,    "持仓量OI",          "Binance官方", "Open interest (OI)"),
    ("liq_agg",       300,   True,  f_liq_agg,       "聚合爆仓量",        "Clawby", "Aggregated liquidations"),
    ("liq_orders",    300,   True,  f_liq_orders,    "大额爆仓单流",      "Clawby", "Large liq order flow"),
    ("ob_wall",       300,   True,  f_ob_wall,       "大额挂单墙",        "Clawby", "Large limit-order walls"),
    ("funding_agg",   900,   True,  f_funding_agg,   "OI加权聚合费率",    "Clawby", "OI-weighted funding"),
    ("funding_xsec",  900,   False, f_funding_xsec,  "全网费率快照",      "Clawby", "Cross-exchange funding"),
    ("lsr_global",    900,   True,  f_lsr_global,    "全局多空人数比",    "Binance官方", "Global long/short accounts"),
    ("lsr_top_pos",   900,   True,  f_lsr_top_pos,   "大户多空持仓比",    "Binance官方", "Top traders position L/S"),
    ("oi_agg",        900,   True,  f_oi_agg,        "全网聚合OI",        "Clawby", "Aggregated OI (all venues)"),
    ("cvd",           900,   True,  f_cvd,           "累积成交量差CVD",   "Clawby", "Cumulative volume delta"),
    ("coinbase_prem", 900,   False, f_coinbase_prem, "Coinbase溢价",      "Clawby", "Coinbase premium"),
    ("exch_chain_tx", 900,   False, f_exch_chain_tx, "交易所大额出入金",  "Clawby", "Exchange on-chain flows"),
    ("whale_tx",      900,   False, f_whale_tx,      "链上鲸鱼转账",      "Clawby", "Whale transfers"),
    ("mkt_price_chg", 900,   False, f_mkt_price_chg, "全币涨跌快照",      "Clawby", "Market-wide price change"),
    ("netflow_xsec",  900,   False, f_netflow_xsec,  "资金净流排行",      "Clawby", "Netflow leaderboard"),
    ("liq_map",       3600,  True,  f_liq_map,       "爆仓地图",          "Clawby", "Liquidation map"),
    ("etf_flow_btc",  3600,  False, f_etf_flow_btc,  "BTC ETF净流",       "Clawby", "BTC ETF net flow"),
    ("fear_greed",    3600,  False, f_fear_greed,    "恐惧贪婪指数",      "Clawby", "Fear & Greed index"),
    ("econ_cal",      3600,  False, f_econ_cal,      "经济日历",          "Clawby", "Economic calendar"),
    ("unlock",        3600,  False, f_unlock,        "代币解锁日程",      "Clawby", "Token unlock schedule"),
    ("alt_season",    3600,  False, f_alt_season,    "山寨季指数",        "Clawby", "Altcoin season index"),
    # -- 2026-07-18 full-sweep expansion --------------------------------------
    ("funding_vw",    1800,  True,  f_funding_vol_weight, "量加权聚合费率", "Clawby", "Vol-weighted funding"),
    ("basis_agg",     1800,  True,  f_basis_agg,     "期现基差历史",      "Clawby", "Basis history"),
    ("whale_index",   1800,  True,  _whale_index_pair, "鲸鱼指数",        "Clawby", "Whale index"),
    ("net_position",  1800,  True,  f_net_position,  "净持仓变动",        "Clawby", "Net position change"),
    ("ob_imbalance",  900,   True,  f_ob_imbalance,  "盘口挂单失衡",      "Clawby", "Orderbook imbalance"),
    ("spot_cvd",      900,   True,  f_spot_cvd,      "现货聚合CVD",       "Clawby", "Spot aggregated CVD"),
    ("taker_agg",     900,   True,  f_taker_agg,     "全网期货taker比",   "Clawby", "Futures taker (all venues)"),
    ("spot_taker_agg", 900,  True,  f_spot_taker_agg, "全网现货taker比",  "Clawby", "Spot taker (all venues)"),
    ("fut_spot_ratio", 1800, True,  f_fut_spot_ratio, "期现成交比",       "Clawby", "Futures/spot volume ratio"),
    ("oi_stable",     1800,  True,  f_oi_stable,     "U本位聚合OI",       "Clawby", "USDT-margined agg OI"),
    ("netflow_coin",  900,   True,  f_netflow_coin,  "单币期货净流",      "Clawby", "Futures coin netflow"),
    ("spot_netflow_coin", 900, True, f_spot_netflow_coin, "单币现货净流", "Clawby", "Spot coin netflow"),
    ("hl_lsr",        1800,  True,  f_hl_lsr,        "HL多空人数比",      "Clawby", "Hyperliquid L/S accounts"),
    ("option_pcr",    3600,  True,  f_option_pcr,    "期权PCR/最大痛点",  "Clawby", "Options PCR / max pain"),
    ("borrow_rate",   3600,  True,  f_borrow_rate,   "借贷利率",          "Clawby", "Borrow interest rate"),
    ("bitfinex_margin", 3600, True, f_bitfinex_margin, "Bitfinex杠杆多空", "Clawby", "Bitfinex margin L/S"),
    ("cgdi",          3600,  False, f_cgdi,          "衍生品综合指数CGDI", "Clawby", "Derivatives index CGDI"),
    ("cdri",          3600,  False, f_cdri,          "衍生品风险指数CDRI", "Clawby", "Derivatives risk CDRI"),
    ("etf_eth_flow",  3600,  False, f_etf_eth_flow,  "ETH ETF净流",       "Clawby", "ETH ETF net flow"),
    ("etf_sol_flow",  14400, False, f_etf_sol_flow,  "SOL ETF净流",       "Clawby", "SOL ETF net flow"),
    ("etf_xrp_flow",  14400, False, f_etf_xrp_flow,  "XRP ETF净流",       "Clawby", "XRP ETF net flow"),
    ("etf_btc_premium", 3600, False, f_etf_btc_premium, "BTC ETF溢价折价", "Clawby", "BTC ETF premium/discount"),
    ("stablecoin_mcap", 14400, False, f_stablecoin_mcap, "稳定币总市值",   "Clawby", "Stablecoin market cap"),
    ("btc_dominance", 14400, False, f_btc_dominance, "BTC支配率",         "Clawby", "BTC dominance"),
    ("option_fut_ratio", 14400, False, f_option_fut_ratio, "期权/期货OI比", "Clawby", "Options/futures OI ratio"),
    ("ahr999",        14400, False, f_ahr999,        "AHR999抄底指标",    "Clawby", "AHR999 accumulation"),
    ("puell",         14400, False, f_puell,         "Puell矿工指标",     "Clawby", "Puell multiple"),
    ("sth_sopr",      14400, False, f_sth_sopr,      "短期持有者SOPR",    "Clawby", "Short-term holder SOPR"),
    ("nupl",          14400, False, f_nupl,          "未实现盈亏NUPL",    "Clawby", "Net unrealized P/L (NUPL)"),
    ("active_addr",   14400, False, f_active_addr,   "BTC活跃地址",       "Clawby", "BTC active addresses"),
    ("funding_arb",   1800,  False, f_funding_arb,   "费率套利机会",      "Clawby", "Funding arbitrage board"),
    ("hl_whale_alert", 900,  False, f_hl_whale_alert, "HL鲸鱼动向",       "Clawby", "HL whale alerts"),
    ("hl_whale_pos",  1800,  False, f_hl_whale_pos,  "HL鲸鱼持仓",        "Clawby", "HL whale positions"),
]


def get_interval(name, default):
    v = db.get_meta(f"factor_interval:{name}", "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


def is_enabled(name):
    return db.get_meta(f"factor_enabled:{name}", "1") == "1"


def set_config(name, interval_sec=None, enabled=None):
    if name not in {r[0] for r in REGISTRY}:
        raise KeyError(name)
    if interval_sec is not None:
        db.set_meta(f"factor_interval:{name}", max(1, int(interval_sec)))
    if enabled is not None:
        db.set_meta(f"factor_enabled:{name}", "1" if enabled else "0")


def _value_summary(name, per_symbol):
    """Compact current-value string for the factor library UI."""
    ref_sym = "BTCUSDT" if per_symbol else ""
    v = db.get_factor(name, ref_sym)
    if v is None and per_symbol:            # majors-only factors may miss BTC? try ETH
        v = db.get_factor(name, "ETHUSDT")
        ref_sym = "ETHUSDT" if v is not None else ref_sym
    if v is None:
        return None, ref_sym
    if isinstance(v, dict):
        for k in ("latest", "pcr", "avg_premium_pct", "last_day_flow_usd",
                  "chg_1h_pct", "flow_5m", "btc", "long_mult"):
            if isinstance(v.get(k), (int, float)):
                val = v[k]
                return (f"{k}={val:,.6g}" if k != "latest" else f"{val:,.6g}"), ref_sym
        for k, val in v.items():
            if isinstance(val, (int, float)):
                return f"{k}={val:,.6g}", ref_sym
            if isinstance(val, list):
                return f"{k}[{len(val)}]", ref_sym
        if all(isinstance(x, dict) for x in v.values()):
            return f"{len(v)} symbols", ref_sym
    return str(v)[:24], ref_sym


def config_snapshot():
    """Full factor-library view for the management panel."""
    out = []
    for name, default, per_symbol, _f, label, source, label_en in REGISTRY:
        symbols = config.UNIVERSE if per_symbol else [""]
        ages = []
        for sym in symbols:
            ts = db.get_factor_ts(name, sym)
            if ts:
                ages.append(int(time.time()) - ts)
        interval = get_interval(name, default)
        value, value_symbol = _value_summary(name, per_symbol)
        out.append({
            "name": name, "label": label, "source": source,
            "per_symbol": per_symbol, "default_interval": default,
            "interval": interval, "enabled": is_enabled(name),
            "last_age_sec": min(ages) if ages else None,
            "stale": (min(ages) > interval * 2 + 30) if ages else True,
            "symbols_collected": len(ages), "symbols_total": len(symbols),
            "value": value, "value_symbol": value_symbol,
            "label_en": label_en,
        })
    return out

# heavy history rows we don't need forever
_NO_HISTORY = {"mark_all", "liq_map", "ob_wall", "exch_chain_tx", "whale_tx",
               "mkt_price_chg", "netflow_xsec", "unlock", "econ_cal", "funding_xsec",
               "netflow_coin", "spot_netflow_coin", "funding_arb",
               "hl_whale_alert", "hl_whale_pos", "ob_imbalance"}


_IN_FLIGHT = set()          # (name, symbol) currently being fetched
_MAX_BATCH = 64             # per-tick cap so one tick can't flood the loop


async def _run_one(name, symbol, fetcher):
    key = (name, symbol or "")
    try:
        value = await fetcher(symbol)
        if value is not None:
            db.set_factor(name, symbol or "", value,
                          keep_history=name not in _NO_HISTORY)
    except Exception as exc:  # noqa: BLE001
        log.warning("factor %s/%s failed: %s", name, symbol, exc)
    finally:
        _IN_FLIGHT.discard(key)


async def collect_due():
    """Refresh enabled factors whose stored value is older than its interval.

    Anti-pileup: a (factor, symbol) already being fetched is never re-queued —
    when the user sets intervals faster than upstream throughput (rate-limit
    lock / slow upstream), collection degrades to best-effort instead of
    accumulating unbounded tasks (2026-07-18 incident). Fire-and-forget so a
    slow upstream never blocks the 0.5s engine tick.
    """
    now = time.time()
    n = 0
    for name, default, per_symbol, fetcher, _label, _src, _en in REGISTRY:
        if not is_enabled(name):
            continue
        interval = get_interval(name, default)
        symbols = config.UNIVERSE if per_symbol else [""]
        for sym in symbols:
            key = (name, sym or "")
            if key in _IN_FLIGHT:
                continue
            if now - db.get_factor_ts(name, sym) >= interval - 1:
                _IN_FLIGHT.add(key)
                asyncio.create_task(_run_one(name, sym, fetcher))
                n += 1
                if n >= _MAX_BATCH:
                    return n
    return n


# -- derived helpers (used by strategies) -----------------------------------

_kline_cache = {}   # (symbol, interval, limit) -> (fetched_at, data)
_KLINE_TTL = 5      # seconds; keeps 1s strategy scans within Binance weight limits


async def get_klines(symbol, interval="1h", limit=100):
    key = (symbol, interval, limit)
    hit = _kline_cache.get(key)
    if hit and time.time() - hit[0] < _KLINE_TTL:
        return hit[1]
    data = await binance.klines(symbol, interval, limit)
    _kline_cache[key] = (time.time(), data)
    return data


def atr_from_klines(ks, period=14):
    if len(ks) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(ks)):
        h, l, pc = ks[i]["high"], ks[i]["low"], ks[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-period:]) / period


def mark_price_of(symbol):
    """Real-time WebSocket price first; REST snapshot as fallback."""
    from . import ws
    live = ws.live_price(symbol)
    if live:
        return live
    snap = db.get_factor("mark_all", "", max_age_sec=120) or {}
    return (snap.get(symbol) or {}).get("mark", 0.0)


def basis_pct_of(symbol):
    snap = db.get_factor("mark_all", "", max_age_sec=120) or {}
    return (snap.get(symbol) or {}).get("basis_pct", 0.0)


def snapshot_for(symbol):
    """Compact factor snapshot for the trade journal (scalar values only)."""
    def g(name, sym, *path):
        v = db.get_factor(name, sym)
        for k in path:
            v = (v or {}).get(k) if isinstance(v, dict) else None
        return round(v, 6) if isinstance(v, (int, float)) else v

    return {
        "funding_agg": g("funding_agg", symbol, "latest"),
        "lsr_global": g("lsr_global", symbol, "latest"),
        "lsr_top_pos": g("lsr_top_pos", symbol, "latest"),
        "oi_chg_1h_pct": g("oi_binance", symbol, "chg_1h_pct"),
        "taker_flow": g("taker_flow", symbol, "latest"),
        "cvd": g("cvd", symbol, "latest"),
        "liq_long_mult": g("liq_agg", symbol, "long_mult"),
        "liq_short_mult": g("liq_agg", symbol, "short_mult"),
        "basis_pct": round(basis_pct_of(symbol), 6),
        "coinbase_prem": g("coinbase_prem", "", "latest"),
        "fear_greed": g("fear_greed", "", "latest"),
        "mark_price": mark_price_of(symbol),
    }

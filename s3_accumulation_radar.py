#!/usr/bin/env python3
"""
热度做多雷达 v2 — 热度+费率+OI 三维扫描

核心逻辑（拉哪模式）：
1. 热度先行 → CG热搜+放量=资金涌入信号
2. 负费率=空头燃料，庄家拉盘爆空单
3. OI暴涨=大资金建仓=即将拉盘

单策略：发现热度→小仓做多→严格止损→拿住赢家

数据源：币安合约API + CoinGecko Trending（零成本）
"""

import json
import os
import sys
import time
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from square_heat import get_square_heat

# === 加载 .env（与 accumulation_radar.py 一致：next-k-api/.env.oi）===
env_file = Path(__file__).parent / ".env.oi"
if env_file.exists():
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# === 配置 ===
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
FAPI = "https://fapi.binance.com"

# 热度历史记录（用于检测首次上榜）
HEAT_HISTORY_FILE = Path(__file__).parent / "heat_history.json"

# 热度参数
VOL_SURGE_MULT = 2.5     # 成交量放大2.5倍以上=放量
MIN_VOL_USD = 20_000_000  # 日均成交>$20M才检测放量

# OI异动参数
MIN_OI_DELTA_PCT = 3.0    # OI变化至少3%
MIN_OI_USD = 2_000_000    # 最低OI门槛 $2M


def api_get(endpoint, params=None):
    """币安API请求"""
    url = f"{FAPI}{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(2)
            else:
                return None
        except:
            time.sleep(1)
    return None


def format_usd(v):
    if v >= 1e9: return f"${v/1e9:.1f}B"
    if v >= 1e6: return f"${v/1e6:.1f}M"
    if v >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


def mcap_str(v):
    if v >= 1e9: return f"${v/1e9:.1f}B"
    if v >= 1e6: return f"${v/1e6:.0f}M"
    if v >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


def send_telegram(text):
    """发送 TG 消息（与 accumulation_radar.send_telegram 行为一致）。"""
    if not TG_BOT_TOKEN:
        print("\n[TG] No token, stdout:\n")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"

    # 分段发送（TG限制4096字）
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > 3800:
            chunks.append(current)
            current = line
        else:
            current += "\n" + line if current else line
    if current:
        chunks.append(current)

    for chunk in chunks:
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id": TG_CHAT_ID,
                    "text": chunk,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                print(f"[TG] Sent ✓ ({len(chunk)} chars)")
            else:
                resp2 = requests.post(
                    url,
                    json={
                        "chat_id": TG_CHAT_ID,
                        "text": chunk.replace("*", "").replace("_", ""),
                    },
                    timeout=10,
                )
                print(f"[TG] Sent plain ({'✓' if resp2.status_code == 200 else '✗'})")
        except Exception as e:
            print(f"[TG] Error: {e}")
        time.sleep(0.5)


def main():
    print(f"🔥 热度做多雷达 v2 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 全市场行情+费率
    tickers_raw = api_get("/fapi/v1/ticker/24hr")
    premiums_raw = api_get("/fapi/v1/premiumIndex")

    if not tickers_raw or not premiums_raw:
        print("❌ API失败")
        return

    ticker_map = {}
    for t in tickers_raw:
        if t["symbol"].endswith("USDT"):
            ticker_map[t["symbol"]] = {
                "px_chg": float(t["priceChangePercent"]),
                "vol": float(t["quoteVolume"]),
                "price": float(t["lastPrice"]),
            }

    funding_map = {}
    for p in premiums_raw:
        if p["symbol"].endswith("USDT"):
            funding_map[p["symbol"]] = float(p["lastFundingRate"])

    # 2. 真实流通市值（币安现货API）
    mcap_map = {}
    try:
        r = requests.get(
            "https://www.binance.com/bapi/composite/v1/public/marketing/symbol/list",
            timeout=10
        )
        if r.status_code == 200:
            for item in r.json().get("data", []):
                name = item.get("name", "")
                mc = item.get("marketCap", 0)
                if name and mc:
                    mcap_map[name] = float(mc)
            print(f"✅ 真实市值: {len(mcap_map)}个币")
    except Exception as e:
        print(f"⚠️ 市值API失败: {e}")

    # 3. 热度检测：币安广场热搜 + CoinGecko Trending + 成交量暴增
    heat_map = {}
    cg_trending = set()
    square_trending = set()

    # 3a. 币安广场热搜（6H）— 最重要！币安用户=交易用户
    sq_coins = get_square_heat()
    if sq_coins:
        for i, c in enumerate(sq_coins):
            coin = c["coin"]
            square_trending.add(coin)
            # 排名越靠前分越高，急速上升额外加分
            rank_score = max(50 - i * 4, 10)
            if c.get("rapidRiser"):
                rank_score += 15
            heat_map[coin] = heat_map.get(coin, 0) + rank_score
        print(f"🏦 广场热搜: {len(square_trending)}个币 {[c['coin'] for c in sq_coins[:5]]}")

    # 3b. CoinGecko Trending
    try:
        r = requests.get("https://api.coingecko.com/api/v3/search/trending", timeout=10)
        if r.status_code == 200:
            for item in r.json().get("coins", []):
                sym = item["item"]["symbol"].upper()
                rank = item["item"].get("score", 99)
                cg_trending.add(sym)
                heat_map[sym] = heat_map.get(sym, 0) + max(50 - rank * 3, 10)
            print(f"🌐 CG Trending: {len(cg_trending)}个币")
    except Exception as e:
        print(f"⚠️ CG Trending失败: {e}")

    # 成交量暴增检测
    vol_surge_coins = set()
    for sym, tk in ticker_map.items():
        coin = sym.replace("USDT", "")
        vol_24h = tk["vol"]
        if vol_24h > MIN_VOL_USD:
            kl = api_get("/fapi/v1/klines", {"symbol": sym, "interval": "1d", "limit": 8})
            if kl and len(kl) >= 5:
                avg_prev = sum(float(k[7]) for k in kl[:-1]) / (len(kl) - 1)
                if avg_prev > 0:
                    ratio = vol_24h / avg_prev
                    if ratio >= VOL_SURGE_MULT:
                        vol_surge_coins.add(coin)
                        heat_map[coin] = heat_map.get(coin, 0) + min(ratio * 10, 50)
            time.sleep(0.05)

    print(f"📈 放量(≥{VOL_SURGE_MULT}x): {len(vol_surge_coins)}个币")

    # 双重/三重热度
    dual_heat = cg_trending & vol_surge_coins
    square_vol = square_trending & vol_surge_coins
    triple_heat = cg_trending & vol_surge_coins & square_trending
    
    all_multi_heat = dual_heat | square_vol
    if all_multi_heat:
        for coin in all_multi_heat:
            heat_map[coin] = heat_map.get(coin, 0) + 20
        if triple_heat:
            for coin in triple_heat:
                heat_map[coin] = heat_map.get(coin, 0) + 30  # 三重热度超级加分
            print(f"🔥🔥🔥 三重热度: {triple_heat}")
        else:
            print(f"🔥🔥 双重热度: {all_multi_heat}")

    # 4. OI扫描（Top100成交量 + 热度币）
    scan_syms = set()
    # 热度币必扫
    for coin in heat_map:
        sym = coin + "USDT"
        if sym in ticker_map:
            scan_syms.add(sym)
    # Top100成交量
    top_by_vol = sorted(ticker_map.items(), key=lambda x: x[1]["vol"], reverse=True)[:100]
    for sym, _ in top_by_vol:
        scan_syms.add(sym)

    oi_map = {}
    for i, sym in enumerate(scan_syms):
        oi_hist = api_get("/futures/data/openInterestHist", {
            "symbol": sym, "period": "1h", "limit": 6
        })
        if oi_hist and len(oi_hist) >= 2:
            curr = float(oi_hist[-1]["sumOpenInterestValue"])
            prev_1h = float(oi_hist[-2]["sumOpenInterestValue"])
            prev_6h = float(oi_hist[0]["sumOpenInterestValue"])
            d1h = ((curr - prev_1h) / prev_1h * 100) if prev_1h > 0 else 0
            d6h = ((curr - prev_6h) / prev_6h * 100) if prev_6h > 0 else 0
            oi_map[sym] = {"oi_usd": curr, "d1h": d1h, "d6h": d6h}
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    print(f"📊 OI扫描: {len(oi_map)}个币")

    # 5. 整合所有数据
    all_syms = set(list(ticker_map.keys()))
    coin_data = {}
    for sym in all_syms:
        tk = ticker_map.get(sym, {})
        if not tk:
            continue
        oi = oi_map.get(sym, {})
        fr = funding_map.get(sym, 0)
        coin = sym.replace("USDT", "")

        d6h = oi.get("d6h", 0)
        fr_pct = fr * 100
        oi_usd = oi.get("oi_usd", 0)

        # 真实市值优先，fallback粗估
        if coin in mcap_map:
            est_mcap = mcap_map[coin]
        else:
            est_mcap = max(tk["vol"] * 0.3, oi_usd * 2) if oi_usd > 0 else tk["vol"] * 0.3

        heat = heat_map.get(coin, 0)

        coin_data[sym] = {
            "coin": coin, "sym": sym,
            "px_chg": tk["px_chg"], "vol": tk["vol"],
            "fr_pct": fr_pct, "d6h": d6h,
            "oi_usd": oi_usd, "est_mcap": est_mcap,
            "heat": heat,
            "in_cg": coin in cg_trending,
            "in_sq": coin in square_trending,
            "vol_surge": coin in vol_surge_coins,
        }

    # ═══════════════════════════════════════
    # 热度榜
    # ═══════════════════════════════════════
    hot_coins = sorted(
        [d for d in coin_data.values() if d["heat"] > 0],
        key=lambda x: x["heat"], reverse=True
    )

    # 检测首次上榜
    heat_history = {}
    if HEAT_HISTORY_FILE.exists():
        try:
            heat_history = json.loads(HEAT_HISTORY_FILE.read_text())
        except:
            pass

    now_ts = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    new_entries = []  # 首次上榜的币
    for s in hot_coins:
        coin = s["coin"]
        if coin not in heat_history:
            # 首次上榜！
            heat_history[coin] = {"first_seen": now_ts, "price": s.get("px_chg", 0)}
            sources = []
            if s["in_sq"]: sources.append("广场")
            if s["in_cg"]: sources.append("CG")
            if s["vol_surge"]: sources.append("放量")
            new_entries.append({"coin": coin, "sources": sources, "data": s})

    # 清理超过 2 天的历史（避免文件无限增长）
    cutoff = (datetime.now(timezone(timedelta(hours=8))) - timedelta(days=2)).strftime("%Y-%m-%d")
    heat_history = {k: v for k, v in heat_history.items()
                    if v.get("first_seen", "9999") >= cutoff}

    # 保存历史
    HEAT_HISTORY_FILE.write_text(json.dumps(heat_history, indent=2, ensure_ascii=False))

    # ═══════════════════════════════════════
    # 追多：负费率+在涨
    # ═══════════════════════════════════════
    chase = []
    for sym, d in coin_data.items():
        if d["px_chg"] > 3 and d["fr_pct"] < -0.005 and d["vol"] > 1_000_000:
            fr_hist = api_get("/fapi/v1/fundingRate", {"symbol": sym, "limit": 5})
            fr_rates = [float(f["fundingRate"]) * 100 for f in fr_hist] if fr_hist else [d["fr_pct"]]
            fr_prev = fr_rates[-2] if len(fr_rates) >= 2 else d["fr_pct"]
            fr_delta = d["fr_pct"] - fr_prev

            trend = "加速恶化" if fr_delta < -0.05 else "转负" if fr_delta < -0.01 else "持平" if abs(fr_delta) < 0.01 else "回升"

            chase.append({**d, "fr_delta": fr_delta, "trend": trend,
                          "rates": " → ".join([f"{x:.3f}" for x in fr_rates[-3:]])})
            time.sleep(0.2)

    chase.sort(key=lambda x: x["fr_pct"])

    # ═══════════════════════════════════════
    # 生成推送
    # ═══════════════════════════════════════
    now = datetime.now(timezone(timedelta(hours=8)))
    lines = [
        f"**热度做多雷达**",
        f"{now.strftime('%Y-%m-%d %H:%M')} CST",
    ]

    # 热度榜（表格）
    # 首次上榜放最前面
    if new_entries:
        lines.append(f"\n**[ 首次上榜 ]** 新出现的热度币，重点关注")
        tbl = ["```"]
        tbl.append(f"{'币种':<10} {'市值':>8} {'涨幅':>7} {'来源'}")
        tbl.append(f"{'-'*10} {'-'*8} {'-'*7} {'-'*20}")
        for e in new_entries:
            s = e["data"]
            src_str = "/".join(e["sources"])
            tbl.append(f"{s['coin']:<10} {mcap_str(s['est_mcap']):>8} {s['px_chg']:>+6.0f}%  {src_str}")
        tbl.append("```")
        lines.append("\n".join(tbl))

    if hot_coins:
        lines.append(f"\n**[ 热度榜 ]**")
        tbl = ["```"]
        tbl.append(f"{'币种':<10} {'市值':>8} {'涨幅':>7} {'来源'}")
        tbl.append(f"{'-'*10} {'-'*8} {'-'*7} {'-'*20}")
        for s in hot_coins[:10]:
            sources = []
            if s["in_sq"]: sources.append("广场")
            if s["in_cg"]: sources.append("CG")
            if s["vol_surge"]: sources.append("放量")
            extra = []
            if abs(s["d6h"]) >= 3: extra.append(f"OI{s['d6h']:+.0f}%")
            if s["fr_pct"] < -0.03: extra.append(f"费率{s['fr_pct']:.2f}%")
            src_str = "/".join(sources)
            if extra:
                src_str += " " + " ".join(extra)
            coin_name = s['coin']
            tbl.append(f"{coin_name:<10} {mcap_str(s['est_mcap']):>8} {s['px_chg']:>+6.0f}%  {src_str}")
        tbl.append("```")
        lines.append("\n".join(tbl))
    else:
        lines.append("\n**[ 热度榜 ]** 暂无热点")

    # 追多（表格）
    lines.append(f"\n**[ 追多 ]** 涨了+费率负=空头燃料")
    if chase:
        tbl = ["```"]
        tbl.append(f"{'币种':<10} {'费率':>10} {'趋势':>8} {'涨幅':>7} {'市值':>8}")
        tbl.append(f"{'-'*10} {'-'*10} {'-'*8} {'-'*7} {'-'*8}")
        for s in chase[:8]:
            tbl.append(
                f"{s['coin']:<10} {s['fr_pct']:>+9.3f}% {s['trend']:>8} {s['px_chg']:>+6.0f}%  {mcap_str(s['est_mcap']):>7}"
            )
        tbl.append("```")
        lines.append("\n".join(tbl))
    else:
        lines.append("  暂无符合条件的标的")

    # OI异动（表格）
    oi_alerts = []
    for sym, oi in oi_map.items():
        if abs(oi["d6h"]) >= 8:
            d = coin_data.get(sym)
            if d and d["heat"] == 0:
                oi_alerts.append(d)
    oi_alerts.sort(key=lambda x: abs(x["d6h"]), reverse=True)

    if oi_alerts:
        lines.append(f"\n**[ OI异动 ]** 6小时持仓变化>=8%")
        tbl = ["```"]
        tbl.append(f"{'币种':<10} {'方向':>4} {'OI变化':>8} {'涨幅':>7} {'市值':>8}")
        tbl.append(f"{'-'*10} {'-'*4} {'-'*8} {'-'*7} {'-'*8}")
        for s in oi_alerts[:6]:
            direction = "增仓" if s["d6h"] > 0 else "减仓"
            tbl.append(
                f"{s['coin']:<10} {direction:>4} {s['d6h']:>+7.1f}% {s['px_chg']:>+6.0f}%  {mcap_str(s['est_mcap']):>7}"
            )
        tbl.append("```")
        lines.append("\n".join(tbl))

    # 值得关注
    highlights = []

    hot_oi = [d for d in coin_data.values() if d["heat"] > 0 and d["d6h"] > 5]
    for s in sorted(hot_oi, key=lambda x: x["d6h"], reverse=True)[:3]:
        highlights.append(f"{s['coin']} — 热度高+OI涨{s['d6h']:+.0f}%，资金涌入")

    hot_fuel = [d for d in coin_data.values() if d["heat"] > 0 and d["fr_pct"] < -0.03]
    for s in sorted(hot_fuel, key=lambda x: x["fr_pct"])[:2]:
        if s["coin"] not in " ".join(highlights):
            highlights.append(f"{s['coin']} — 热度高+费率{s['fr_pct']:.2f}%，空头燃料足")

    chase_fire = [s for s in chase[:5] if "加速" in s.get("trend", "")]
    for s in chase_fire[:2]:
        if s["coin"] not in " ".join(highlights):
            highlights.append(f"{s['coin']} — 费率{s['fr_pct']:.3f}%持续恶化，逼空在即")

    if highlights:
        lines.append(f"\n**[ 值得关注 ]**")
        for h in highlights[:5]:
            lines.append(f"  {h}")

    lines.append(f"\n广场=币安站内搜索 / CG=CoinGecko全球热度")
    lines.append(f"费率负=做空多，庄家拉盘爆空头")

    report = "\n".join(lines)
    send_telegram(report)
    print("\n✅ 完成")


if __name__ == "__main__":
    main()

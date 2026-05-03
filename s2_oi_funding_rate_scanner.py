#!/usr/bin/env python3
"""
OI持续放大 + 费率由正转负 扫描器
- main.py 定时：每整点后第 5 分钟 (xx:05, Asia/Shanghai)；亦可手动高频跑测
- 检测: OI持续放大(4段递增, 总涨幅>8%) + 费率由正转负
- 去重: 同一币种24小时内只推一次
- 纯API零成本
"""

import requests
import json
import os
import time
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ============ 配置 ============
SCRIPT_DIR = Path(__file__).parent
ENV_FILE = SCRIPT_DIR / ".env.oi"
ALERT_HISTORY_FILE = SCRIPT_DIR / "oi_funding_alerts.json"
FR_SNAPSHOT_FILE = SCRIPT_DIR / "fr_snapshot.json"  # 上一次费率快照
SIGNALS_HISTORY_FILE = SCRIPT_DIR / "s2_signals_history.json"  # 供前端展示近 7 日
CST = timezone(timedelta(hours=8))
SIGNAL_HISTORY_DAYS = 7

# 信号参数
MIN_OI_CHANGE_PCT = 8       # OI总涨幅最低8%
MIN_VOLUME_USDT = 0  # 无门槛，全扫
MIN_FR_PERIODS_POSITIVE = 2  # 转负前至少2期为正
DEDUP_HOURS = 24             # 去重窗口24小时

# ============ 加载TG配置 ============
def load_env():
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().strip().split('\n'):
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip()
    return env

env = load_env()
TG_BOT_TOKEN = env.get('TG_BOT_TOKEN', '')
TG_CHAT_ID = env.get('TG_CHAT_ID', '')

# ============ TG推送 ============
def send_tg(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("[TG] 未配置, 仅打印:")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    # 分段发送(TG限制4096字)
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        try:
            resp = requests.post(url, json={
                'chat_id': TG_CHAT_ID,
                'text': chunk,
                'parse_mode': 'Markdown'
            }, timeout=10)
            if resp.status_code != 200:
                # fallback无格式
                requests.post(url, json={
                    'chat_id': TG_CHAT_ID,
                    'text': chunk
                }, timeout=10)
        except Exception as e:
            print(f"[TG] 发送失败: {e}")

# ============ 去重 ============
def load_alert_history():
    if ALERT_HISTORY_FILE.exists():
        try:
            return json.loads(ALERT_HISTORY_FILE.read_text())
        except:
            return {}
    return {}

def save_alert_history(history):
    ALERT_HISTORY_FILE.write_text(json.dumps(history))

def is_duplicate(symbol, history):
    if symbol not in history:
        return False
    last_alert = datetime.fromisoformat(history[symbol])
    return (datetime.now() - last_alert).total_seconds() < DEDUP_HOURS * 3600

def mark_alerted(symbol, history):
    history[symbol] = datetime.now().isoformat()
    # 清理过期记录
    cutoff = datetime.now() - timedelta(hours=DEDUP_HOURS * 2)
    history = {k: v for k, v in history.items() 
               if datetime.fromisoformat(v) > cutoff}
    return history

# ============ 费率快照 ============
def load_fr_snapshot():
    if FR_SNAPSHOT_FILE.exists():
        try:
            return json.loads(FR_SNAPSHOT_FILE.read_text())
        except:
            pass
    return {}

def save_fr_snapshot(snapshot):
    FR_SNAPSHOT_FILE.write_text(json.dumps(snapshot))

# ============ 核心扫描 ============
def scan():
    ts_start = time.time()
    
    # 1. 获取所有永续合约
    try:
        info = requests.get('https://fapi.binance.com/fapi/v1/exchangeInfo', timeout=10).json()
        symbols = [s['symbol'] for s in info['symbols'] 
                   if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
    except Exception as e:
        print(f"[ERROR] exchangeInfo: {e}")
        return []
    
    # 2. 批量获取24h行情(过滤低量币)
    try:
        tickers = requests.get('https://fapi.binance.com/fapi/v1/ticker/24hr', timeout=10).json()
        ticker_map = {t['symbol']: t for t in tickers}
    except Exception as e:
        print(f"[ERROR] ticker: {e}")
        return []
    
    active = [s for s in symbols if float(ticker_map.get(s, {}).get('quoteVolume', 0)) > MIN_VOLUME_USDT]
    
    # 3. 批量获取当前费率 (一次拿全部)
    try:
        fr_all = requests.get('https://fapi.binance.com/fapi/v1/premiumIndex', timeout=10).json()
        fr_current = {item['symbol']: float(item['lastFundingRate']) for item in fr_all}
    except:
        fr_current = {}
    
    # 4. 加载上次快照，对比找"刚转负"的
    prev_snapshot = load_fr_snapshot()
    
    # 保存本次快照(供下次对比)
    save_fr_snapshot(fr_current)
    
    if not prev_snapshot:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 首次运行，保存快照，下次开始对比")
        return []
    
    # 找出: 上次>=0, 这次<0 的币
    just_turned_negative = []
    for sym in active:
        prev_fr = prev_snapshot.get(sym)
        curr_fr = fr_current.get(sym)
        if prev_fr is None or curr_fr is None:
            continue
        if prev_fr >= 0 and curr_fr < 0:
            just_turned_negative.append(sym)
    
    if not just_turned_negative:
        elapsed = time.time() - ts_start
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 扫描完成: {len(active)}币/{elapsed:.1f}s, 无新转负")
        return []
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 发现 {len(just_turned_negative)} 个刚转负: {just_turned_negative}")
    
    # 5. 只对刚转负的币查OI
    signals = []
    for sym in just_turned_negative:
        try:
            # OI历史
            oi_hist = requests.get('https://fapi.binance.com/futures/data/openInterestHist', 
                params={'symbol': sym, 'period': '1h', 'limit': 48}, timeout=10).json()
            
            oi_chg = 0
            segs = []
            oi_rising = False
            if oi_hist and len(oi_hist) >= 12:
                oi_values = [float(x['sumOpenInterestValue']) for x in oi_hist]
                seg_len = len(oi_values) // 4
                if seg_len >= 3:
                    segs = [
                        sum(oi_values[:seg_len]) / seg_len,
                        sum(oi_values[seg_len:seg_len*2]) / seg_len,
                        sum(oi_values[seg_len*2:seg_len*3]) / seg_len,
                        sum(oi_values[seg_len*3:]) / max(1, len(oi_values[seg_len*3:]))
                    ]
                    oi_chg = (segs[3] - segs[0]) / segs[0] * 100 if segs[0] > 0 else 0
                    oi_rising = oi_chg > 0
            
            t = ticker_map.get(sym, {})
            signals.append({
                'symbol': sym,
                'price': float(t.get('lastPrice', 0)),
                'price_chg_24h': float(t.get('priceChangePercent', 0)),
                'volume': float(t.get('quoteVolume', 0)),
                'oi_change': oi_chg,
                'oi_segments': segs,
                'oi_rising': oi_rising,
                'current_fr': fr_current.get(sym, 0),
                'prev_fr': prev_snapshot.get(sym, 0),
            })
        except:
            continue
    
    elapsed = time.time() - ts_start
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 扫描完成: {len(active)}币/{elapsed:.1f}s, 信号: {len(signals)}")
    
    return signals

# ============ 附加信息 ============
def get_square_discussion(coin):
    """查询币安广场该币的帖子数和浏览量"""
    try:
        r = requests.get(
            "https://www.binance.com/bapi/composite/v4/friendly/pgc/content/queryByHashtag",
            params={"hashtag": f"#{coin.lower()}", "pageIndex": 1, "pageSize": 1, "orderBy": "HOT"},
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.binance.com/en/square"},
            timeout=8
        )
        if r.status_code == 200:
            ht = r.json().get("data", {}).get("hashtag", {})
            return ht.get("contentCount", 0), ht.get("viewCount", 0)
    except:
        pass
    return 0, 0

def get_market_caps():
    """获取币安流通市值"""
    mcap = {}
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
                    mcap[name] = float(mc)
    except:
        pass
    return mcap

def get_spot_symbols():
    """获取有现货的币种"""
    try:
        info = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=10).json()
        return {s["baseAsset"] for s in info["symbols"]
                if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"}
    except:
        return set()

def fmt_mcap(v):
    if v >= 1e9: return f"${v/1e9:.2f}B"
    if v >= 1e6: return f"${v/1e6:.1f}M"
    if v >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"

def fmt_views(v):
    if v >= 1e6: return f"{v/1e6:.1f}M"
    if v >= 1e3: return f"{v/1e3:.0f}K"
    return str(v)

# ============ 格式化推送 ============
def format_alert(signals):
    if not signals:
        return None
    
    # OI在涨的排前面，同组内按费率绝对值排序
    signals.sort(key=lambda x: (-int(x.get('oi_rising', False)), x['current_fr']))
    
    # 批量获取附加信息
    mcap_map = get_market_caps()
    spot_set = get_spot_symbols()
    
    now = datetime.now().strftime('%m-%d %H:%M')
    lines = [f"*[ 费率刚转负+OI涨 ]* {now}\n"]
    
    for s in signals:
        coin = s['symbol'].replace('USDT', '')
        
        # 费率: 上期→本期
        fr_change = f"{s['prev_fr']:+.4%} -> {s['current_fr']:+.4%}"
        
        # 附加信息
        mcap = mcap_map.get(coin, 0)
        has_spot = coin in spot_set
        sq_posts, sq_views = get_square_discussion(coin)
        
        lines.append(f"```")
        lines.append(f"{coin}")
        lines.append(f"  价格: {s['price']:.4f}  24h: {s['price_chg_24h']:+.1f}%")
        lines.append(f"  费率: {fr_change}")
        if s['oi_segments']:
            oi_segs = ' > '.join([f"{v/1e6:.1f}M" for v in s['oi_segments']])
            lines.append(f"  OI: +{s['oi_change']:.1f}%  ({oi_segs})")
        lines.append(f"  成交额: ${s['volume']/1e6:.1f}M")
        lines.append(f"  市值: {fmt_mcap(mcap) if mcap > 0 else '未知'}  现货: {'有' if has_spot else '仅合约'}")
        if sq_posts > 0:
            lines.append(f"  广场: {sq_posts}帖 / {fmt_views(sq_views)}浏览")
        else:
            lines.append(f"  广场: 无讨论")
        lines.append(f"```")
    
    return '\n'.join(lines)


def _parse_recorded_at(row):
    s = row.get("recorded_at") or ""
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=CST)
        return dt
    except Exception:
        return datetime.min.replace(tzinfo=CST)


def persist_strong_signals(strong):
    """写入与 Telegram 一致的强信号，供 main GET /api/s2/funding-signals 读取。"""
    if not strong:
        return
    try:
        mcap_map = get_market_caps()
        spot_set = get_spot_symbols()
        now_cst = datetime.now(CST)
        ts = now_cst.isoformat()
        new_rows = []
        for s in strong:
            coin = s["symbol"].replace("USDT", "")
            sq_posts, sq_views = get_square_discussion(coin)
            mcap = float(mcap_map.get(coin, 0) or 0)
            segs = s.get("oi_segments") or []
            new_rows.append({
                "recorded_at": ts,
                "symbol": s["symbol"],
                "coin": coin,
                "price": float(s.get("price", 0) or 0),
                "price_chg_24h": float(s.get("price_chg_24h", 0) or 0),
                "prev_fr": float(s.get("prev_fr", 0) or 0),
                "current_fr": float(s.get("current_fr", 0) or 0),
                "oi_change_pct": float(s.get("oi_change", 0) or 0),
                "oi_segment_avgs_usd": [float(x) for x in segs] if segs else [],
                "volume_usd": float(s.get("volume", 0) or 0),
                "est_mcap_usd": mcap,
                "has_spot": coin in spot_set,
                "square_posts": int(sq_posts or 0),
                "square_views": int(sq_views or 0),
            })
        payload = {"signals": []}
        if SIGNALS_HISTORY_FILE.exists():
            try:
                raw = json.loads(SIGNALS_HISTORY_FILE.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("signals"), list):
                    payload["signals"] = raw["signals"]
            except Exception:
                pass
        merged = new_rows + payload["signals"]
        cutoff = now_cst - timedelta(days=SIGNAL_HISTORY_DAYS)
        kept = [row for row in merged if _parse_recorded_at(row) >= cutoff]
        kept = kept[:2000]
        tmp = SIGNALS_HISTORY_FILE.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"signals": kept}, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp.replace(SIGNALS_HISTORY_FILE)
        print(
            f"  💾 信号历史 +{len(new_rows)} 条 (近{SIGNAL_HISTORY_DAYS}天共{len(kept)}条) -> {SIGNALS_HISTORY_FILE.name}"
        )
    except Exception as e:
        print(f"  ⚠️ 信号历史写入失败: {e}")


# ============ 主逻辑 ============
def main():
    signals = scan()
    
    if signals:
        # 只推: 费率当前为负 + OI在涨 (最强组合)
        strong = [s for s in signals if s['current_fr'] < 0 and s.get('oi_rising')]
        if strong:
            msg = format_alert(strong)
            persist_strong_signals(strong)
            if msg:
                send_tg(msg)
                print(f"  推送 {len(strong)} 个信号 (总{len(signals)}个转负, {len(strong)}个OI也涨)")
        else:
            print(f"  {len(signals)} 个转负但无OI在涨的, 跳过")
    else:
        print(f"  无信号")

if __name__ == '__main__':
    main()

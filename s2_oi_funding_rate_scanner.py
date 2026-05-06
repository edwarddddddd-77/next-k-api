#!/usr/bin/env python3
"""
OI持续放大 + 费率由正转负 扫描器
- main.py 定时：每整点后第 5 分钟 (xx:05, Asia/Shanghai)；亦可手动高频跑测
- 检测: OI 四段首尾抬升 + 费率由非负→足够负（见 MIN_CURR_FR_FOR_FLIP）
- 费率快照带时间戳；旧版扁平快照 / 快照间隔过长时跳过一期对比，避免部署或久置后一批假阳性
- 纯API零成本
"""

import requests
import json
import os
import sqlite3
import time
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

# ============ 配置 ============
SCRIPT_DIR = Path(__file__).parent
ENV_FILE = SCRIPT_DIR / ".env.oi"
ALERT_HISTORY_FILE = SCRIPT_DIR / "oi_funding_alerts.json"
FR_SNAPSHOT_FILE = SCRIPT_DIR / "fr_snapshot.json"  # 上一次费率快照
SIGNALS_HISTORY_FILE = SCRIPT_DIR / "s2_signals_history.json"  # 迁移后弃用，历史改存 accumulation.db
CST = timezone(timedelta(hours=8))
SIGNAL_HISTORY_DAYS = 7


def _accumulation_db_path() -> Path:
    return Path(os.getenv("DATA_DIR", str(SCRIPT_DIR))) / "accumulation.db"


def _ensure_s2_funding_table(conn: sqlite3.Connection) -> None:
    """与 accumulation_radar.init_db 中 s2_funding_signals 表结构一致。"""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS s2_funding_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recorded_at TEXT NOT NULL,
        symbol TEXT NOT NULL,
        coin TEXT,
        price REAL,
        price_chg_24h REAL,
        prev_fr REAL,
        current_fr REAL,
        oi_change_pct REAL,
        oi_segment_avgs_json TEXT,
        volume_usd REAL,
        est_mcap_usd REAL,
        has_spot INTEGER DEFAULT 0,
        square_posts INTEGER DEFAULT 0,
        square_views INTEGER DEFAULT 0
    )"""
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_s2_recorded_symbol "
        "ON s2_funding_signals(recorded_at, symbol)"
    )


def _insert_s2_signal(cur: sqlite3.Cursor, row: Dict[str, Any]) -> None:
    segs = row.get("oi_segment_avgs_usd")
    if not isinstance(segs, list):
        segs = []
    oj = json.dumps(segs, ensure_ascii=False)
    hs = row.get("has_spot")
    hs_i = 1 if (hs is True or hs == 1 or hs == "1") else 0
    cur.execute(
        """INSERT OR IGNORE INTO s2_funding_signals (
            recorded_at, symbol, coin, price, price_chg_24h, prev_fr, current_fr,
            oi_change_pct, oi_segment_avgs_json, volume_usd, est_mcap_usd,
            has_spot, square_posts, square_views
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            row.get("recorded_at"),
            row.get("symbol"),
            row.get("coin"),
            float(row.get("price") or 0),
            float(row.get("price_chg_24h") or 0),
            float(row.get("prev_fr") or 0),
            float(row.get("current_fr") or 0),
            float(row.get("oi_change_pct") or 0),
            oj,
            float(row.get("volume_usd") or 0),
            float(row.get("est_mcap_usd") or 0),
            hs_i,
            int(row.get("square_posts") or 0),
            int(row.get("square_views") or 0),
        ),
    )


def _migrate_s2_json_legacy(conn: sqlite3.Connection) -> None:
    """一次性：旧 s2_signals_history.json → DB，成功后改名为 .json.bak。"""
    if not SIGNALS_HISTORY_FILE.is_file():
        return
    try:
        raw = json.loads(SIGNALS_HISTORY_FILE.read_text(encoding="utf-8"))
        sigs = raw.get("signals") if isinstance(raw, dict) else None
        if not isinstance(sigs, list):
            return
        cur = conn.cursor()
        for row in sigs:
            if isinstance(row, dict):
                _insert_s2_signal(cur, row)
        conn.commit()
        bak = SIGNALS_HISTORY_FILE.with_suffix(".json.bak")
        SIGNALS_HISTORY_FILE.rename(bak)
        print(f"  📦 S2 信号历史 JSON → SQLite，原文件改为 {bak.name}")
    except Exception as e:
        print(f"  ⚠️ S2 JSON 迁移跳过: {e}")


def _prune_s2_funding_rows(
    conn: sqlite3.Connection,
    now_cst: datetime,
    *,
    days: int,
) -> None:
    cutoff = now_cst - timedelta(days=days)
    cur = conn.cursor()
    cur.execute("SELECT id, recorded_at FROM s2_funding_signals")
    kill: List[int] = []
    for rid, ra in cur.fetchall():
        if _parse_recorded_at({"recorded_at": ra}) < cutoff:
            kill.append(int(rid))
    for rid in kill:
        conn.execute("DELETE FROM s2_funding_signals WHERE id = ?", (rid,))


def _s2_db_row_to_signal(row: sqlite3.Row) -> Dict[str, Any]:
    raw_j = row["oi_segment_avgs_json"]
    segs: List[float] = []
    if raw_j:
        try:
            t = json.loads(raw_j)
            if isinstance(t, list):
                segs = [float(x) for x in t]
        except Exception:
            pass
    return {
        "recorded_at": row["recorded_at"],
        "symbol": row["symbol"],
        "coin": row["coin"],
        "price": float(row["price"] or 0),
        "price_chg_24h": float(row["price_chg_24h"] or 0),
        "prev_fr": float(row["prev_fr"] or 0),
        "current_fr": float(row["current_fr"] or 0),
        "oi_change_pct": float(row["oi_change_pct"] or 0),
        "oi_segment_avgs_usd": segs,
        "volume_usd": float(row["volume_usd"] or 0),
        "est_mcap_usd": float(row["est_mcap_usd"] or 0),
        "has_spot": bool(row["has_spot"]),
        "square_posts": int(row["square_posts"] or 0),
        "square_views": int(row["square_views"] or 0),
    }


def get_s2_funding_signals_for_api(days: int = 7) -> Dict[str, Any]:
    """
    供 FastAPI GET /api/s2/funding-signals。
    数据存 accumulation.db（s2_funding_signals），保留最近 ``days`` 天（与定时扫描裁剪一致）。
    """
    db_path = _accumulation_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        _ensure_s2_funding_table(conn)
        _migrate_s2_json_legacy(conn)
        now_cst = datetime.now(CST)
        _prune_s2_funding_rows(conn, now_cst, days=days)
        conn.commit()

        cur = conn.cursor()
        cur.execute(
            """
            SELECT recorded_at, symbol, coin, price, price_chg_24h, prev_fr, current_fr,
                   oi_change_pct, oi_segment_avgs_json, volume_usd, est_mcap_usd,
                   has_spot, square_posts, square_views
            FROM s2_funding_signals
            ORDER BY recorded_at DESC
            LIMIT 4000
            """
        )
        rows = [_s2_db_row_to_signal(r) for r in cur.fetchall()]
        return {
            "ok": True,
            "signals": rows,
            "day_window": days,
            "source": "sqlite",
            "count": len(rows),
        }
    finally:
        conn.close()

# 信号参数
MIN_OI_CHANGE_PCT = 8       # OI总涨幅最低8%
MIN_VOLUME_USDT = 0  # 无门槛，全扫
MIN_FR_PERIODS_POSITIVE = 2  # 转负前至少2期为正
DEDUP_HOURS = 24             # 去重窗口24小时

# 费率「刚转负」防噪：当前费率须明显为负（过滤 -0.00001 级微负）
MIN_CURR_FR_FOR_FLIP = -0.0001
# 距上次写入快照超过此时长则只做刷新、不做跨期对比（避免停机/旧文件后一次扫出几十上百条）
SNAPSHOT_MAX_GAP_HOURS = 8

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
    """
    返回 (rates_dict, _saved_at_iso_or_None, kind)
    kind: "v2" | "legacy" | "empty"
    legacy = 旧版纯 {SYMBOL: rate}，无时间戳，易在部署后与当前费率差一档导致满屏「转负」。
    """
    if not FR_SNAPSHOT_FILE.exists():
        return {}, None, "empty"
    try:
        raw = json.loads(FR_SNAPSHOT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}, None, "empty"
    if not isinstance(raw, dict):
        return {}, None, "empty"
    if "rates" in raw and isinstance(raw["rates"], dict):
        return raw["rates"], raw.get("_saved_at"), "v2"
    # 扁平：视为 legacy（全是合约名键）
    if raw:
        return raw, None, "legacy"
    return {}, None, "empty"


def save_fr_snapshot(rates: dict):
    """写入带 _saved_at 的快照，供下一轮对比与间隔判断。"""
    payload = {
        "_saved_at": datetime.now(CST).isoformat(),
        "rates": rates,
    }
    tmp = FR_SNAPSHOT_FILE.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp.replace(FR_SNAPSHOT_FILE)

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
    
    # 4. 加载上次快照，对比找"刚转负"的（先写本轮快照，再与内存中的 prev 对比）
    prev_snapshot, prev_saved_at, snap_kind = load_fr_snapshot()
    save_fr_snapshot(fr_current)

    if not prev_snapshot:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 首次运行，保存快照，下次开始对比")
        return []

    if snap_kind == "legacy" or prev_saved_at is None:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 费率快照为旧版或无时间戳，"
            f"跳过本期转负检测（避免部署/仓库旧快照导致一批假阳性），已写入新版快照"
        )
        return []

    try:
        prev_dt = datetime.fromisoformat(prev_saved_at.replace("Z", "+00:00"))
        if prev_dt.tzinfo is None:
            prev_dt = prev_dt.replace(tzinfo=CST)
        gap_sec = (datetime.now(CST) - prev_dt.astimezone(CST)).total_seconds()
    except Exception:
        gap_sec = SNAPSHOT_MAX_GAP_HOURS * 3600 + 1

    if gap_sec > SNAPSHOT_MAX_GAP_HOURS * 3600:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 距上次快照 {gap_sec / 3600:.1f}h "
            f"> {SNAPSHOT_MAX_GAP_HOURS}h，跳过本期转负检测，已刷新"
        )
        return []

    # 找出: 上次>=0, 本次明显为负（过滤结算后微负噪声）
    just_turned_negative = []
    for sym in active:
        prev_fr = prev_snapshot.get(sym)
        curr_fr = fr_current.get(sym)
        if prev_fr is None or curr_fr is None:
            continue
        if prev_fr >= 0 and curr_fr <= MIN_CURR_FR_FOR_FLIP:
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
    """写入 accumulation.db：与 Telegram 一致的强信号，供 GET /api/s2/funding-signals 读取。"""
    if not strong:
        return
    try:
        mcap_map = get_market_caps()
        spot_set = get_spot_symbols()
        now_cst = datetime.now(CST)
        ts = now_cst.isoformat()
        new_rows: List[Dict[str, Any]] = []
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
        db_path = _accumulation_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            _ensure_s2_funding_table(conn)
            _migrate_s2_json_legacy(conn)
            cur = conn.cursor()
            for row in new_rows:
                _insert_s2_signal(cur, row)
            _prune_s2_funding_rows(conn, now_cst, days=SIGNAL_HISTORY_DAYS)
            conn.commit()
            cur.execute("SELECT COUNT(*) FROM s2_funding_signals")
            total = cur.fetchone()[0]
            print(
                f"  💾 S2 信号 +{len(new_rows)} 条 -> SQLite ({db_path.name}，表内共 {total} 条，保留近{SIGNAL_HISTORY_DAYS}天)"
            )
        finally:
            conn.close()
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

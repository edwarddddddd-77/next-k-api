"""FastAPI app: dashboard API + engine background task + static frontend."""
import asyncio
import faulthandler
import logging
import os
import signal
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Unix only — Windows has no SIGUSR1
if hasattr(signal, "SIGUSR1"):
    try:
        faulthandler.register(signal.SIGUSR1)  # kill -USR1 <pid> dumps all stacks
    except Exception:
        pass


from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import binance, config, db, engine, executor, factors, risk, ws

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    tasks = [asyncio.create_task(ws.ws_loop()),
             asyncio.create_task(engine.scan_loop()),
             asyncio.create_task(engine.position_loop())]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="quant-bot", lifespan=lifespan)


@app.get("/api/status")
async def api_status():
    eq, bal, upnl = await executor.equity()
    cfg = config.load_strategies()
    day_pnl = risk.daily_realized_pnl()
    return {
        "mode": executor.mode(),
        "equity": eq, "balance": bal, "unrealized": upnl,
        "daily_realized_pnl": day_pnl,
        "halted": risk.halted(),
        "event_quiet": risk.event_quiet(cfg["global"]),
        "open_positions": len(db.open_positions()),
        "universe": config.UNIVERSE,
        "last_tick_ts": int(db.get_meta("last_tick_ts", "0") or 0),
        "last_manage_ms": int(db.get_meta("last_manage_ts_ms", "0") or 0),
        "ws_symbols": len(ws.PRICES),
        "started_at": int(db.get_meta("started_at", "0") or 0),
        "last_error": db.get_meta("last_error", ""),
        "now": int(time.time()),
        "now_ms": int(time.time() * 1000),
    }


@app.get("/api/positions")
async def api_positions():
    out = []
    for p in db.open_positions():
        price = factors.mark_price_of(p["symbol"]) or p["entry_price"]
        sign = 1 if p["side"] == "long" else -1
        upnl = (price - p["entry_price"]) * p["qty"] * sign
        out.append({**p, "mark_price": price, "unrealized_pnl": upnl,
                    "notional": abs(p["qty"] * price),
                    "age_sec": int(time.time()) - p["opened_at"]})
    return {"positions": out}


@app.post("/api/positions/{pid}/close")
async def api_close_position(pid: int):
    pos = db.get_position(pid)
    if not pos or pos["status"] != "open":
        raise HTTPException(404, "position not open")
    price = factors.mark_price_of(pos["symbol"])
    if not price:
        raise HTTPException(503, "no live price available")
    ok = await executor.close_position(pos, price, "手动市价平仓")
    return {"ok": ok}


@app.post("/api/positions/close-all")
async def api_close_all():
    closed, failed = 0, 0
    for pos in db.open_positions():
        price = factors.mark_price_of(pos["symbol"])
        if price and await executor.close_position(pos, price, "手动一键全平"):
            closed += 1
        else:
            failed += 1
    return {"ok": failed == 0, "closed": closed, "failed": failed}


@app.get("/api/trades")
async def api_trades(limit: int = 100, mode: str = "auto"):
    m = None if mode in ("all", "") else mode
    return {"trades": db.recent_trades(limit, mode=m), "mode": executor.mode()}


@app.get("/api/signals")
async def api_signals(limit: int = 50):
    return {"signals": db.recent_signals(limit)}


@app.get("/api/equity")
async def api_equity(limit: int = 2880, mode: str = "auto"):
    m = None if mode in ("all", "") else mode
    shown = executor.mode() if mode == "auto" else (m or "all")
    return {"series": db.equity_series(limit, mode=m), "mode": shown,
            "active_mode": executor.mode()}


@app.get("/api/factors")
async def api_factors():
    return {"factors": db.all_factors_snapshot(), "now": int(time.time())}


@app.get("/api/factor-history")
async def api_factor_history(factor: str, symbol: str = "", limit: int = 200):
    return {"history": db.factor_history(factor, symbol, limit)}


@app.get("/api/factor-config")
async def api_factor_config():
    return {"factors": factors.config_snapshot()}


@app.post("/api/factor-config/{name}")
async def api_factor_config_set(name: str, payload: dict):
    try:
        factors.set_config(name,
                           interval_sec=payload.get("interval_sec"),
                           enabled=payload.get("enabled"))
    except KeyError:
        raise HTTPException(404, f"unknown factor {name}")
    db.log("info", f"因子 {name} 配置更新: {payload}")
    return {"ok": True}


@app.get("/api/strategies")
async def api_strategies():
    from .strategies import REGISTRY, strategy_meta
    cfg = config.load_strategies()
    meta = strategy_meta()
    factor_labels = {r[0]: r[4] for r in factors.REGISTRY}

    def deps_of(base):
        cls = REGISTRY.get(base)
        return [{"name": f, "label": factor_labels.get(f, f),
                 "enabled": factors.is_enabled(f)}
                for f in getattr(cls, "FACTORS", [])] if cls else []

    trades = db.recent_trades(500, mode="auto")   # stats follow the active mode
    stats = {}
    for t in trades:
        if t["kind"] != "close":
            continue
        s = stats.setdefault(t["strategy"], {"closed": 0, "wins": 0, "pnl": 0.0})
        s["closed"] += 1
        s["pnl"] += t["pnl"] or 0
        if (t["pnl"] or 0) > 0:
            s["wins"] += 1
    return {"global": cfg["global"],
            "templates": [{"id": tid, "meta": m} for tid, m in meta.items()],
            "strategies": [{"id": iid, **scfg,
                            "base": scfg.get("base") or iid,
                            "stats": stats.get(iid),
                            "factors": deps_of(scfg.get("base") or iid),
                            "meta": meta.get(scfg.get("base") or iid, {})}
                           for iid, scfg in cfg.get("strategies", {}).items()]}


@app.post("/api/strategies/{sid}/config")
async def api_strategy_config(sid: str, payload: dict):
    try:
        if "scan_interval_sec" in payload:
            config.set_strategy_interval(sid, payload["scan_interval_sec"])
        if "symbols" in payload:
            config.set_strategy_symbols(sid, payload["symbols"])
        if "risk" in payload:
            config.set_strategy_risk(sid, payload["risk"])
        if "params" in payload:
            config.set_strategy_params(sid, payload["params"])
        if "display_name" in payload:
            config.set_strategy_display_name(sid, payload["display_name"])
    except KeyError:
        raise HTTPException(404, f"unknown strategy {sid}")
    except (TypeError, ValueError):
        raise HTTPException(400, "invalid config value")
    db.log("info", f"策略 {sid} 配置更新")
    return {"ok": True}


@app.post("/api/strategies/create")
async def api_strategy_create(payload: dict):
    base = str(payload.get("base") or "")
    try:
        iid = config.create_strategy_instance(base, payload.get("name") or "")
    except KeyError:
        raise HTTPException(404, f"unknown template {base}")
    db.log("info", f"新建策略实例 {iid}(模板 {base})")
    return {"ok": True, "id": iid}


@app.delete("/api/strategies/{sid}")
async def api_strategy_delete(sid: str):
    if db.open_positions(strategy=sid, mode=None):   # any mode blocks deletion
        raise HTTPException(409, "该策略实例仍有未平仓位,请先平仓")
    try:
        config.delete_strategy_instance(sid)
    except KeyError:
        raise HTTPException(404, f"unknown strategy {sid}")
    db.log("warn", f"策略实例 {sid} 已删除")
    return {"ok": True}


# -- risk / universe / credentials -------------------------------------------

@app.get("/api/risk")
async def api_risk_get():
    cfg = config.load_strategies()
    eq, _, _ = await executor.equity()
    strat = []
    for sid, scfg in cfg.get("strategies", {}).items():
        risk = scfg.get("risk", {})
        used = sum(abs(p["qty"] * factors.mark_price_of(p["symbol"]))
                   for p in db.open_positions(strategy=sid))
        strat.append({"id": sid, "meta_name": (scfg.get("params") and ""),
                      "enabled": scfg.get("enabled"), "risk": risk, "used_notional": used})
    return {"global": cfg["global"], "equity": eq, "strategies": strat}


@app.post("/api/risk/global")
async def api_risk_global(payload: dict):
    config.set_global_risk(payload)
    db.log("info", "全局风控参数更新")
    return {"ok": True}


@app.get("/api/universe")
async def api_universe_get():
    return {"universe": config.UNIVERSE, "known": config.KNOWN_SYMBOLS}


@app.post("/api/universe")
async def api_universe_set(payload: dict):
    try:
        syms = config.set_universe(payload.get("symbols") or [])
    except ValueError as e:
        raise HTTPException(400, str(e))
    db.log("info", f"监控币种更新为 {','.join(syms)}")
    return {"ok": True, "universe": syms}


@app.get("/api/credentials")
async def api_credentials_get():
    return config.credentials_masked()


@app.post("/api/credentials")
async def api_credentials_set(payload: dict):
    masked = config.save_credentials(payload)
    db.log("warn", "API 凭据已更新(热生效)")
    return {"ok": True, **masked}


@app.post("/api/credentials/test")
async def api_credentials_test(payload: dict):
    """Live connectivity check for the Config page."""
    which = payload.get("which", "both")
    result = {}
    if which in ("clawby", "both"):
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get(f"{config.CLAWBY_BASE}/api/account",
                                headers={"X-API-Key": config.CLAWBY_API_KEY})
                d = r.json()
            result["clawby"] = {"ok": r.status_code == 200,
                                "plan": d.get("plan"), "balance": d.get("payg_balance")}
        except Exception as e:  # noqa: BLE001
            result["clawby"] = {"ok": False, "error": str(e)[:120]}
    if which in ("binance", "both"):
        try:
            acc = await binance.account()
            result["binance"] = {"ok": True, "can_trade": acc.get("canTrade"),
                                 "wallet_usdt": acc.get("totalWalletBalance"),
                                 "available": acc.get("availableBalance")}
        except Exception as e:  # noqa: BLE001
            result["binance"] = {"ok": False, "error": str(e)[:120]}
    from . import exchanges
    for venue, tester in (("bitget", exchanges.bitget_test),
                          ("okx", exchanges.okx_test)):
        if which in (venue, "both") and exchanges.has_credentials(venue):
            try:
                result[venue] = await tester()
            except Exception as e:  # noqa: BLE001
                result[venue] = {"ok": False, "error": str(e)[:120]}
    return result


@app.post("/api/strategies/{sid}/toggle")
async def api_toggle(sid: str, payload: dict):
    try:
        config.set_strategy_enabled(sid, bool(payload.get("enabled")))
    except KeyError:
        raise HTTPException(404, f"unknown strategy {sid}")
    db.log("info", f"策略 {sid} {'启用' if payload.get('enabled') else '停用'}")
    return {"ok": True}


@app.get("/api/executor-exchange")
async def api_executor_exchange_get():
    from . import exchanges
    cred = config.credentials_masked()
    return {"exchange": executor.exchange(),
            "supported": [{"id": v, "has_credentials":
                           exchanges.has_credentials(v)} for v in exchanges.SUPPORTED],
            "has_bitget": cred["has_bitget"], "has_okx": cred["has_okx"]}


@app.post("/api/executor-exchange")
async def api_executor_exchange_set(payload: dict):
    venue = str(payload.get("exchange") or "")
    try:
        executor.set_exchange(venue)
    except ValueError as e:
        raise HTTPException(400, str(e))
    db.log("warn", f"实盘执行交易所切换为 {venue}")
    return {"ok": True, "exchange": venue}


@app.post("/api/mode")
async def api_mode(payload: dict):
    m = payload.get("mode")
    if m not in ("paper", "live"):
        raise HTTPException(400, "mode must be paper|live")
    if m == "live" and payload.get("confirm") != "LIVE":
        raise HTTPException(400, "switching to live requires confirm=LIVE")
    executor.set_mode(m)
    db.log("warn", f"交易模式切换为 {m}")
    return {"ok": True, "mode": m}


@app.get("/api/klines")
async def api_klines(symbol: str = "BTCUSDT", interval: str = "15m", limit: int = 200):
    ks = await binance.klines(symbol, interval, limit)
    marks = []
    for t in db.recent_trades(300):
        if t["symbol"] == symbol:
            marks.append({"ts": t["ts"], "price": t["price"], "kind": t["kind"],
                          "side": t["side"], "strategy": t["strategy"]})
    return {"klines": ks, "marks": marks}


@app.get("/api/logs")
async def api_logs(limit: int = 100):
    return {"logs": db.recent_logs(limit)}


# -- trade journal (JSONL files under journal/) -------------------------------

def _journal_rows(strategy="", symbol="", day="", limit=0, mode=""):
    import json as _json

    from .journal import JOURNAL_DIR
    rows = []
    if not JOURNAL_DIR.exists():
        return rows
    files = sorted(JOURNAL_DIR.glob("trades-*.jsonl"), reverse=True)
    if day:
        files = [f for f in files if f.stem == f"trades-{day}"]
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            try:
                r = _json.loads(line)
            except ValueError:
                continue
            if strategy and r.get("strategy") != strategy:
                continue
            if symbol and r.get("symbol") != symbol.upper():
                continue
            if mode and r.get("mode") != mode:
                continue
            rows.append(r)
            if limit and len(rows) >= limit:
                return rows
    return rows


@app.get("/api/journal")
async def api_journal(limit: int = 200, strategy: str = "", symbol: str = "",
                      day: str = "", mode: str = ""):
    from .journal import JOURNAL_DIR
    days = sorted((f.stem.replace("trades-", "")
                   for f in JOURNAL_DIR.glob("trades-*.jsonl")), reverse=True) \
        if JOURNAL_DIR.exists() else []
    return {"rows": _journal_rows(strategy, symbol, day, limit, mode), "days": days}


@app.get("/api/journal/export")
async def api_journal_export(strategy: str = "", symbol: str = "", day: str = "",
                             mode: str = ""):
    import csv
    import io
    import json as _json

    from fastapi.responses import StreamingResponse
    rows = _journal_rows(strategy, symbol, day, mode=mode)
    cols = ["closed_at", "opened_at", "strategy", "symbol", "side", "qty",
            "entry_price", "close_price", "pnl", "pnl_pct", "fee",
            "hold_minutes", "signal_reason", "close_reason", "mfe_pct",
            "mae_pct", "mode", "entry_factors", "exit_factors"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        r = dict(r)
        for k in ("entry_factors", "exit_factors"):
            if isinstance(r.get(k), dict):
                r[k] = _json.dumps(r[k], ensure_ascii=False)
        w.writerow(r)
    buf.seek(0)
    name = f"journal-{day or 'all'}{'-' + strategy if strategy else ''}.csv"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition":
                                      f'attachment; filename="{name}"'})


@app.get("/health")
async def health():
    return {"ok": True, "last_tick_ts": db.get_meta("last_tick_ts", "0")}


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        target = DIST / path
        if path and target.is_file():
            return FileResponse(target)
        index = DIST / "index.html"
        prefix = (os.environ.get("CLAWBY_API_PREFIX") or "").strip()
        if not prefix:
            return FileResponse(index)
        html = index.read_text(encoding="utf-8")
        inject = f"<script>window.__CLAWBY_API_PREFIX__={prefix!r};</script>"
        if "__CLAWBY_API_PREFIX__" not in html:
            html = html.replace("<head>", "<head>" + inject, 1)
        from fastapi.responses import HTMLResponse

        return HTMLResponse(html)

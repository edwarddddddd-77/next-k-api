#!/usr/bin/env python3
"""ORB 纸面扫描与结算。"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from binance_fapi import fetch_klines_forward, klines_to_df
from orb.config import OrbConfig
from orb.indicators import daily_atr_asof
from orb.macro_calendar import is_macro_skip_day, macro_calendar_status
from orb.db import (
    archive_settlement,
    count_open_positions,
    ensure_symbol_bots,
    fetch_open_for_resolve,
    fetch_open_hold,
    migrate_orb_tables,
    symbol_bot_enabled,
    symbol_bot_wallet_balance,
    symbol_session_traded,
)
from orb.premarket import extended_fetch_anchor_ms, uses_alpaca_premarket
from orb.providers.alpaca import load_premarket_history
from orb.resolve import pnl_r, pnl_usdt, resolve_forward
from orb.session import (
    is_trading_session,
    session_day_floor_ms,
    session_day_str,
)
from orb.tz import session_tz_abbrev, session_utc_offset_hours
from orb.signals import OrbSignal, classify_signal

logger = logging.getLogger(__name__)


def _live_open(sig: OrbSignal, cfg: OrbConfig) -> Optional[Dict[str, Any]]:
    if not cfg.live_enabled:
        return None
    try:
        from orb.live_exec import notify_open

        return notify_open(sig, cfg)
    except Exception as exc:
        logger.warning("[orb] live open %s failed: %s", sig.symbol, exc)
        return {"error": str(exc)}


def _live_close(
    cfg: OrbConfig,
    symbol: str,
    side: str,
    *,
    close_price: Optional[float] = None,
    play: Optional[str] = None,
    tag: str = "resolve",
) -> Optional[Dict[str, Any]]:
    if not cfg.live_enabled:
        return None
    try:
        from orb.live_exec import notify_close

        return notify_close(
            symbol,
            side,
            cfg,
            close_price=close_price,
            play=play,
            tag=tag,
        )
    except Exception as exc:
        logger.warning("[orb] live close %s failed: %s", symbol, exc)
        return {"error": str(exc)}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _session_date_now(cfg: OrbConfig) -> str:
    return session_day_str(
        int(time.time() * 1000), tz=cfg.session_tz, session_open_time=cfg.session_open_time
    )


def _drop_forming_bar(df: pd.DataFrame, cfg: OrbConfig, *, now_ms: Optional[int] = None) -> pd.DataFrame:
    """去掉尚未收盘的最后一根 K 线，避免在 forming bar 上误判信号。"""
    if df.empty:
        return df
    t = int(now_ms if now_ms is not None else time.time() * 1000)
    step = cfg.bar_step_ms()
    last_open = int(df["open_time"].iloc[-1])
    if last_open + step > t:
        return df.iloc[:-1].reset_index(drop=True)
    return df


def _load_signal_df(symbol: str, cfg: OrbConfig, *, now_ms: Optional[int] = None) -> pd.DataFrame:
    end_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    day0 = extended_fetch_anchor_ms(end_ms, cfg)
    rows = fetch_klines_forward(symbol, cfg.signal_interval, day0, end_ms)
    df = klines_to_df(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time").reset_index(drop=True)
    return _drop_forming_bar(df, cfg, now_ms=end_ms)


def _load_daily_df(symbol: str, cfg: OrbConfig, *, now_ms: Optional[int] = None) -> pd.DataFrame:
    end_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    warmup_ms = cfg.daily_atr_warmup_ms()
    rows = fetch_klines_forward(symbol, "1d", end_ms - warmup_ms, end_ms)
    df = klines_to_df(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time").reset_index(drop=True)


def _signal_df_from_bars(
    df5: pd.DataFrame,
    cfg: OrbConfig,
    *,
    now_ms: int,
) -> pd.DataFrame:
    """Binance RTH 信号窗口（Alpaca 盘前时不含 04:00–09:30 Binance  bars）。"""
    if df5.empty:
        return df5
    day0 = extended_fetch_anchor_ms(now_ms, cfg)
    df = df5[(df5["open_time"] >= day0) & (df5["open_time"] <= now_ms)].copy()
    if df.empty:
        return df
    df = (
        df.drop_duplicates(subset=["open_time"], keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    return _drop_forming_bar(df, cfg, now_ms=now_ms)


def daily_window_for_atr(
    daily_df: pd.DataFrame,
    now_ms: int,
    cfg: OrbConfig,
) -> pd.DataFrame:
    """与 _load_daily_df 相同窗口：now 往前 atr_period+20 日。"""
    if daily_df.empty:
        return daily_df
    warmup_ms = cfg.daily_atr_warmup_ms()
    start = int(now_ms) - warmup_ms
    sub = daily_df[(daily_df["open_time"] >= start) & (daily_df["open_time"] <= int(now_ms))]
    return (
        sub.drop_duplicates(subset=["open_time"], keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def analyze_at_ms(
    symbol: str,
    *,
    cfg: OrbConfig,
    now_ms: int,
    session_traded: bool = False,
    daily_df: Optional[pd.DataFrame] = None,
    bot_equity_usdt: Optional[float] = None,
    df5: Optional[pd.DataFrame] = None,
    df_alpaca: Optional[pd.DataFrame] = None,
    alpaca_daily: Optional[pd.DataFrame] = None,
) -> OrbSignal:
    """纸面 analyze_live 的可回放版本（now_ms = 扫描时刻）。"""
    sym = str(symbol).strip().upper()
    full_hist = df5
    if df5 is not None:
        df = _signal_df_from_bars(df5, cfg, now_ms=now_ms)
    else:
        df = _load_signal_df(sym, cfg, now_ms=now_ms)
        full_hist = df
    if df.empty:
        return OrbSignal(sym, 0.0, "FLAT", "ORB_NO_TRADE", "low", ["empty_klines"])
    asof = int(df["open_time"].iloc[-1])
    ddf = daily_df
    if ddf is None and (
        (cfg.sl_mode or "").strip().lower() == "atr_pct"
        or (cfg.premarket_filter and not uses_alpaca_premarket(cfg))
    ):
        ddf = _load_daily_df(sym, cfg, now_ms=now_ms)
    daily_atr = None
    if (cfg.sl_mode or "").strip().lower() == "atr_pct":
        if ddf is not None and not ddf.empty:
            ddf_atr = daily_window_for_atr(ddf, now_ms, cfg)
            daily_atr = daily_atr_asof(ddf_atr, asof, period=cfg.atr_period, tz=cfg.session_tz)

    pm_hist: Optional[pd.DataFrame] = None
    pm_daily: Optional[pd.DataFrame] = None
    if uses_alpaca_premarket(cfg):
        pm_hist, pm_daily = load_premarket_history(
            sym,
            cfg,
            asof_ms=now_ms,
            cached_bars=df_alpaca,
            cached_daily=alpaca_daily,
        )

    return classify_signal(
        sym,
        df,
        asof_open_ms=asof,
        cfg=cfg,
        session_traded=session_traded,
        daily_atr=daily_atr,
        bot_equity_usdt=bot_equity_usdt,
        full_df=full_hist,
        daily_df=ddf if (ddf is not None and not ddf.empty) else None,
        pm_history_df=pm_hist,
        pm_daily_df=pm_daily,
    )


def analyze_live(
    symbol: str,
    *,
    cfg: OrbConfig,
    session_traded: bool = False,
    daily_df: Optional[pd.DataFrame] = None,
    bot_equity_usdt: Optional[float] = None,
) -> OrbSignal:
    sym = str(symbol).strip().upper()
    now_ms = int(time.time() * 1000)
    return analyze_at_ms(
        sym,
        cfg=cfg,
        now_ms=now_ms,
        session_traded=session_traded,
        daily_df=daily_df,
        bot_equity_usdt=bot_equity_usdt,
    )


def _scan_params(cfg: OrbConfig) -> Dict[str, Any]:
    return {
        "strategy": "orb",
        "market": cfg.market,
        "session_tz": cfg.session_tz,
        "session_tz_abbrev": session_tz_abbrev(int(time.time() * 1000), cfg.session_tz),
        "session_utc_offset_h": session_utc_offset_hours(int(time.time() * 1000), cfg.session_tz),
        "session_open_time": cfg.session_open_time,
        "session_close_time": cfg.session_close_time,
        "regular_session_only": cfg.regular_session_only,
        "entry_mode": cfg.entry_mode,
        "vwap_filter": cfg.vwap_filter,
        "premarket_filter": cfg.premarket_filter,
        "premarket_source": cfg.premarket_source,
        "premarket_mode": cfg.premarket_mode,
        "premarket_rvol_min": cfg.premarket_rvol_min,
        "sl_mode": cfg.sl_mode,
        "atr_period": cfg.atr_period,
        "atr_sl_fraction": cfg.atr_sl_fraction,
        "exit_mode": cfg.exit_mode,
        "risk_pct": cfg.risk_pct,
        "symbol_bot_equity_usdt": cfg.per_symbol_bot_equity(),
        "account_equity_usdt": cfg.account_equity_usdt,
        "fixed_notional_usdt": cfg.fixed_notional_usdt,
        "position_safety_pct": cfg.position_safety_pct,
        "uses_risk_sizing": cfg.uses_risk_sizing(),
        "tp_r": cfg.tp_r_multiple,
        "confirm_bars": cfg.confirm_bars,
        "confirm_no_soften": cfg.confirm_no_soften,
        "vol_mult": cfg.vol_mult,
        "signal_interval": cfg.signal_interval,
        "or_minutes": cfg.or_minutes,
        "margin_usdt": cfg.margin_usdt,
        "leverage": cfg.leverage,
    }


def is_actionable(sig: OrbSignal, cfg: OrbConfig) -> bool:
    if sig.side not in ("LONG", "SHORT") or sig.sl_price is None:
        return False
    if (cfg.exit_mode or "").strip().lower() == "eod":
        return True
    return sig.tp_price is not None


def _upsert_signal(cur, *, ts: str, sig: OrbSignal, scan_params: dict, cfg: OrbConfig) -> None:
    actionable = is_actionable(sig, cfg)
    if not actionable:
        cur.execute(
            """
            SELECT outcome, side, sl_price FROM orb_signals
            WHERE symbol = ? LIMIT 1
            """,
            (sig.symbol,),
        )
        existing = cur.fetchone()
        if existing is not None:
            outcome, side, sl = existing
            if outcome is not None:
                return
            if str(side) in ("LONG", "SHORT") and sl is not None:
                return

    outcome_cols = (
        ", outcome, outcome_at_utc, exit_price, pnl_r, pnl_usdt"
        if actionable
        else ""
    )
    outcome_vals = ", NULL, NULL, NULL, NULL, NULL" if actionable else ""
    outcome_reset = (
        ", outcome=NULL, outcome_at_utc=NULL, exit_price=NULL, pnl_r=NULL, pnl_usdt=NULL"
        if actionable
        else ""
    )
    cur.execute(
        f"""
        INSERT INTO orb_signals (
            recorded_at_utc, updated_at_utc, symbol, play, side, confidence,
            entry_price, entry_bar_open_ms, sl_price, tp_price, r_unit,
            virtual_notional_usdt, or_high, or_low, or_width_pct, session_date,
            volume, vol_ma, reasons_json, scan_params_json{outcome_cols}
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?{outcome_vals})
        ON CONFLICT(symbol) DO UPDATE SET
            recorded_at_utc=excluded.recorded_at_utc,
            updated_at_utc=excluded.updated_at_utc,
            play=excluded.play, side=excluded.side, confidence=excluded.confidence,
            entry_price=excluded.entry_price, entry_bar_open_ms=excluded.entry_bar_open_ms,
            sl_price=excluded.sl_price, tp_price=excluded.tp_price, r_unit=excluded.r_unit,
            virtual_notional_usdt=excluded.virtual_notional_usdt,
            or_high=excluded.or_high, or_low=excluded.or_low, or_width_pct=excluded.or_width_pct,
            session_date=excluded.session_date, volume=excluded.volume, vol_ma=excluded.vol_ma,
            reasons_json=excluded.reasons_json, scan_params_json=excluded.scan_params_json{outcome_reset}
        """,
        (
            ts, ts, sig.symbol, sig.play, sig.side, sig.confidence,
            sig.price or None, sig.entry_bar_open_ms, sig.sl_price, sig.tp_price, sig.r_unit,
            sig.paper_notional_usdt, sig.or_high or None, sig.or_low or None, sig.or_width_pct or None,
            sig.session_date or None, sig.volume or None, sig.vol_ma or None,
            json.dumps(sig.reasons, ensure_ascii=False),
            json.dumps(scan_params, ensure_ascii=False),
        ),
    )


def _sync_symbol_bot_wallet(conn, symbol: str, cfg: OrbConfig) -> None:
    symbol_bot_wallet_balance(
        conn,
        symbol,
        initial_equity_usdt=cfg.per_symbol_bot_equity(),
        sync=True,
    )


def _settle_supersede(
    conn,
    cur,
    row,
    *,
    exit_price: float,
    now_utc: str,
    note: str,
    cfg: OrbConfig,
) -> None:
    sid, sym, side, play, entry, sl, tp, notion, sess = row
    pr = pnl_r(side, float(entry), float(exit_price), float(sl))
    pu = pnl_usdt(side, float(entry), float(exit_price), float(notion))
    cur.execute(
        """
        UPDATE orb_signals SET outcome='supersede', outcome_at_utc=?, exit_price=?,
            pnl_r=?, pnl_usdt=?, exit_rule=?, notes=? WHERE id=? AND outcome IS NULL
        """,
        (now_utc, float(exit_price), round(pr, 6), round(pu, 4), note, note, int(sid)),
    )
    archive_settlement(
        cur,
        signal_id=int(sid),
        symbol=str(sym),
        side=str(side),
        play=str(play) if play else None,
        outcome="supersede",
        entry_price=float(entry),
        exit_price=float(exit_price),
        pnl_r=round(pr, 6),
        pnl_usdt=round(pu, 4),
        notional=float(notion),
        exit_rule=note,
        settled_at_utc=now_utc,
        session_date=str(sess) if sess else None,
    )
    _sync_symbol_bot_wallet(conn, str(sym), cfg)


def resolve_open_positions(conn, *, cfg: Optional[OrbConfig] = None) -> Dict[str, Any]:
    c = cfg or OrbConfig.from_env()
    now_utc = _utc_now()
    stats = {"checked": 0, "resolved": 0, "skipped": 0, "live": []}
    conn.row_factory = __import__("sqlite3").Row
    migrate_orb_tables(conn.cursor())
    cur = conn.cursor()
    rows = fetch_open_for_resolve(cur, default_notional=c.default_paper_notional())
    end_ms = int(time.time() * 1000)
    for row in rows:
        stats["checked"] += 1
        sid, sym, side, play, entry, sl, tp, bar_open, notion = row
        if bar_open is None:
            stats["skipped"] += 1
            continue
        signal_step = c.bar_step_ms()
        resolve_start = int(bar_open) + signal_step
        kl = fetch_klines_forward(sym, "1m", resolve_start, end_ms)
        if not kl:
            stats["skipped"] += 1
            continue
        df = klines_to_df(kl)
        out, ex_px, note, _, exit_bo = resolve_forward(
            df,
            entry=float(entry),
            entry_bar_open_ms=int(bar_open),
            side=str(side),
            sl=float(sl),
            tp=float(tp) if tp is not None else None,
            hist_end_ms=end_ms,
            bar_step_ms=signal_step,
            cfg=c,
        )
        if out is None:
            stats["skipped"] += 1
            logger.debug("[orb] resolve skip %s id=%s note=%s", sym, sid, note)
            continue
        pr = pnl_r(side, float(entry), ex_px, float(sl))
        pu = pnl_usdt(side, float(entry), ex_px, float(notion))
        cur.execute("SELECT session_date FROM orb_signals WHERE id=?", (int(sid),))
        sr = cur.fetchone()
        sess_date = str(sr[0]) if sr and sr[0] else None
        cur.execute(
            """
            UPDATE orb_signals SET outcome=?, outcome_at_utc=?, exit_price=?,
                pnl_r=?, pnl_usdt=?, exit_rule=?, notes=? WHERE id=? AND outcome IS NULL
            """,
            (out, now_utc, ex_px, round(pr, 6), round(pu, 4), note, note, int(sid)),
        )
        if cur.rowcount:
            archive_settlement(
                cur,
                signal_id=int(sid),
                symbol=str(sym),
                side=str(side),
                play=str(play) if play else None,
                outcome=str(out),
                entry_price=float(entry),
                exit_price=ex_px,
                pnl_r=round(pr, 6),
                pnl_usdt=round(pu, 4),
                notional=float(notion),
                exit_rule=note,
                settled_at_utc=now_utc,
                session_date=sess_date,
            )
            _sync_symbol_bot_wallet(conn, str(sym), c)
            stats["resolved"] += 1
            live_close = _live_close(
                c,
                str(sym),
                str(side),
                close_price=ex_px,
                play=str(play) if play else None,
                tag=str(out),
            )
            if live_close is not None:
                stats["live"].append(
                    {"action": "close", "symbol": sym, "tag": str(out), "result": live_close}
                )
    conn.commit()
    return stats


def in_regular_session(cfg: OrbConfig, *, now_ms: Optional[int] = None) -> bool:
    if not (cfg.session_open_time or "").strip():
        return True
    t = int(now_ms if now_ms is not None else time.time() * 1000)
    return is_trading_session(
        t,
        tz=cfg.session_tz,
        session_open_time=cfg.session_open_time,
        session_close_time=cfg.session_close_time,
        market=cfg.market,
    )


def _idle_scan_skip_reason(cfg: OrbConfig, cur, *, now_ms: Optional[int] = None) -> Optional[str]:
    """美东 RTH 外且无持仓：跳过整轮（不拉 K 线、不写库）。"""
    if not cfg.regular_session_only:
        return None
    if in_regular_session(cfg, now_ms=now_ms):
        return None
    if count_open_positions(cur) > 0:
        return None
    return "outside_regular_session_no_open_positions"


def _record_orb_run(
    cur,
    *,
    now_utc: str,
    symbols_scanned: int,
    opens: int,
    resolves: int,
    detail: Dict[str, Any],
) -> None:
    cur.execute(
        "INSERT INTO orb_runs (ran_at_utc, symbols_scanned, opens, resolves, detail_json) VALUES (?,?,?,?,?)",
        (
            now_utc,
            symbols_scanned,
            opens,
            resolves,
            json.dumps(detail, default=str),
        ),
    )


def run_scan_conn(conn, *, do_resolve: bool = True, cfg: Optional[OrbConfig] = None) -> Dict[str, Any]:
    c = cfg or OrbConfig.from_env()
    if not c.enabled:
        return {"ok": True, "lane": "orb", "skipped": True, "reason": "orb_disabled"}
    now_utc = _utc_now()
    session_day = _session_date_now(c)
    syms = c.symbol_list()
    stats: Dict[str, Any] = {
        "ok": True,
        "lane": "orb",
        "ran_at_utc": now_utc,
        "symbols": syms,
        "written": 0,
        "skipped": [],
        "opens": [],
        "live": [],
    }
    conn.row_factory = __import__("sqlite3").Row
    migrate_orb_tables(conn.cursor())
    conn.commit()
    cur = conn.cursor()
    bot_equity = c.per_symbol_bot_equity()
    ensure_symbol_bots(cur, syms, initial_equity_usdt=bot_equity)
    conn.commit()

    if c.macro_filter:
        macro_meta = macro_calendar_status()
        stats["macro_calendar"] = macro_meta
        logger.info(
            "[orb] macro filter on: total=%s fomc_live=%s(%s) cpi_live=%s(%s) cache_age_s=%s",
            macro_meta["total_dates"],
            macro_meta["fomc_live"],
            macro_meta["fomc_live_count"],
            macro_meta["cpi_live"],
            macro_meta["cpi_live_count"],
            macro_meta["cache_age_seconds"],
        )
        is_macro_skip_day(session_day)

    idle_reason = _idle_scan_skip_reason(c, cur)
    if idle_reason:
        logger.info("[orb] idle skip: %s", idle_reason)
        return {
            **stats,
            "skipped": True,
            "reason": idle_reason,
            "opens": [],
        }

    if not in_regular_session(c):
        if not do_resolve:
            return {
                **stats,
                "skipped": True,
                "reason": "outside_regular_session_resolve_disabled",
            }
        resolve_pre = resolve_open_positions(conn, cfg=c)
        stats["live"] = list(resolve_pre.get("live") or [])
        stats["mode"] = "resolve_only"
        stats["reason"] = "outside_regular_session_has_open_positions"
        stats["resolve_pre"] = resolve_pre
        stats["resolve_post"] = {}
        _record_orb_run(
            cur,
            now_utc=now_utc,
            symbols_scanned=0,
            opens=0,
            resolves=int(resolve_pre.get("resolved", 0)),
            detail={"stats": stats, "resolve_pre": resolve_pre},
        )
        conn.commit()
        return stats

    scan_params = _scan_params(c)
    resolve_pre = resolve_open_positions(conn, cfg=c) if do_resolve else {}
    stats["live"].extend(resolve_pre.get("live") or [])
    daily_cache: Dict[str, pd.DataFrame] = {}
    for sym in syms:
        if (c.sl_mode or "").strip().lower() == "atr_pct":
            daily_cache[sym] = _load_daily_df(sym, c)
    for sym in syms:
        try:
            if not symbol_bot_enabled(cur, sym):
                stats["skipped"].append({"symbol": sym, "reason": "bot_disabled"})
                continue
            bot_wallet = symbol_bot_wallet_balance(
                conn, sym, initial_equity_usdt=bot_equity, sync=True
            )
            if bot_wallet <= 0:
                stats["skipped"].append({"symbol": sym, "reason": "bot_wallet_depleted"})
                continue
            hold = fetch_open_hold(cur, sym, default_notional=c.default_paper_notional())
            traded = symbol_session_traded(cur, sym, session_day) if c.one_trade_per_session else False
            sig = analyze_live(
                sym,
                cfg=c,
                session_traded=traded,
                daily_df=daily_cache.get(sym),
                bot_equity_usdt=bot_wallet,
            )
            actionable = is_actionable(sig, c)
            if not actionable:
                if hold is not None or c.db_skip_flat:
                    stats["skipped"].append({"symbol": sym, "reason": "flat", "detail": sig.reasons})
                    continue
                _upsert_signal(cur, ts=now_utc, sig=sig, scan_params=scan_params, cfg=c)
                stats["written"] += 1
                continue
            if hold is not None:
                if str(hold["side"]) == sig.side:
                    stats["skipped"].append({"symbol": sym, "reason": "same_side_open"})
                    continue
                live_close = _live_close(
                    c,
                    sym,
                    str(hold["side"]),
                    close_price=float(sig.price),
                    play=str(hold["play"]) if hold["play"] else None,
                    tag="supersede",
                )
                if live_close is not None:
                    stats["live"].append({"action": "close", "symbol": sym, "tag": "supersede", "result": live_close})
                _settle_supersede(
                    conn, cur, hold, exit_price=float(sig.price), now_utc=now_utc,
                    note="supersede:reverse", cfg=c,
                )
            if c.max_open_positions > 0 and count_open_positions(cur) >= c.max_open_positions:
                stats["skipped"].append({"symbol": sym, "reason": "max_open_cap"})
                continue
            _upsert_signal(cur, ts=now_utc, sig=sig, scan_params=scan_params, cfg=c)
            stats["written"] += 1
            stats["opens"].append(
                {"symbol": sym, "side": sig.side, "entry": sig.price, "sl": sig.sl_price, "tp": sig.tp_price}
            )
            live_open = _live_open(sig, c)
            if live_open is not None:
                stats["live"].append({"action": "open", "symbol": sym, "result": live_open})
        except Exception as exc:
            logger.warning("[orb] %s failed: %s", sym, exc)
            stats["skipped"].append({"symbol": sym, "reason": "error", "error": str(exc)})
    conn.commit()
    resolve_post = resolve_open_positions(conn, cfg=c) if do_resolve else {}
    stats["live"].extend(resolve_post.get("live") or [])
    _record_orb_run(
        cur,
        now_utc=now_utc,
        symbols_scanned=len(syms),
        opens=len(stats["opens"]),
        resolves=int(resolve_pre.get("resolved", 0)) + int(resolve_post.get("resolved", 0)),
        detail={"stats": stats, "resolve_pre": resolve_pre, "resolve_post": resolve_post},
    )
    conn.commit()
    stats["resolve_pre"] = resolve_pre
    stats["resolve_post"] = resolve_post
    return stats


def run_scan(*, do_resolve: bool = True) -> Dict[str, Any]:
    from accumulation_radar import init_db

    conn = init_db()
    try:
        return run_scan_conn(conn, do_resolve=do_resolve)
    finally:
        conn.close()


def run_resolve_only() -> Dict[str, Any]:
    from accumulation_radar import init_db

    conn = init_db()
    try:
        return resolve_open_positions(conn)
    finally:
        conn.close()

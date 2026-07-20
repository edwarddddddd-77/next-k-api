"""Unit tests for Trading OS desk helpers (no network)."""

from __future__ import annotations

from pathlib import Path

from utils.trading_os_desk import (
    add_wallet,
    compute_risk,
    list_alts,
    list_wallets,
    maybe_alert,
    remove_wallet,
    set_alts,
)


def test_wallet_crud(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from utils import trading_os_desk as desk

    monkeypatch.setattr(desk, "resolve_data_dir", lambda: Path(tmp_path))
    # stub discovery so tests stay offline
    monkeypatch.setattr(
        desk,
        "discover_accumulators",
        lambda **kw: {
            "ok": True,
            "candidates": [
                {
                    "address": "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
                    "chain": "btc",
                    "label": "吸筹·测·12BTC",
                    "withdraw_btc": 12.0,
                    "probe": "Bitfinex热钱包",
                    "source": "cex_outflow",
                },
                {
                    "address": "0x0000000000000000000000000000000000000002",
                    "chain": "eth",
                    "label": "吸筹·测·80ETH",
                    "withdraw_eth": 80.0,
                    "probe": "Bitfinex-ETH热",
                    "source": "cex_outflow",
                },
                {
                    "address": "3yJNtRhKbVSnTm9GZD3ubWbUhbzvX3P9KJjj87pWpkmH",
                    "chain": "sol",
                    "label": "吸筹·测·150SOL",
                    "withdraw_sol": 150.0,
                    "probe": "Bitfinex-SOL热",
                    "source": "cex_outflow",
                },
            ],
            "count": 3,
            "btc": {"count": 1},
            "eth": {"count": 1},
            "sol": {"count": 1},
            "errors": [],
        },
    )
    seeded = list_wallets()["wallets"]
    assert len(seeded) >= 3
    chains = {w["chain"] for w in seeded}
    assert "btc" in chains and "eth" in chains and "sol" in chains
    w = add_wallet("1BoatSLRHtKNngkdXEeobR76b53LETtpyT", label="manual", chain="btc")
    assert w["ok"]
    assert any(x.get("label") == "manual" for x in list_wallets()["wallets"])
    remove_wallet("1BoatSLRHtKNngkdXEeobR76b53LETtpyT")
    assert all(x.get("label") != "manual" for x in list_wallets()["wallets"])


def test_legacy_junk_detected():
    from utils.trading_os_desk import _is_legacy_junk_watchlist

    junk = [
        {"seeded": True, "source": "bitinfocharts", "address": "x"},
        {"seeded": True, "source": "ens/public", "address": "y"},
    ]
    assert _is_legacy_junk_watchlist(junk) is True
    good = [{"seeded": True, "source": "cex_outflow", "address": "z"}]
    assert _is_legacy_junk_watchlist(good) is False


def test_eth_detect(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from utils import trading_os_desk as desk

    monkeypatch.setattr(desk, "resolve_data_dir", lambda: Path(tmp_path))
    w = add_wallet("0x0000000000000000000000000000000000000001", label="e")
    assert w["wallet"]["chain"] == "eth"


def test_set_alts(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from utils import trading_os_desk as desk

    monkeypatch.setattr(desk, "resolve_data_dir", lambda: Path(tmp_path))
    out = set_alts(["pepe", "WIF/USDT", "BONK"], mode="manual")
    assert out["mode"] == "manual"
    assert out["symbols"] == ["PEPEUSDT", "WIFUSDT", "BONKUSDT"]
    assert list_alts()["symbols"] == out["symbols"]
    assert list_alts()["mode"] == "manual"


def test_discover_alt_universe_filters(monkeypatch):
    from utils import trading_os_desk as desk

    fake = [
        {"symbol": "BTCUSDT", "quoteVolume": "9e12", "priceChangePercent": "1", "lastPrice": "1"},
        {"symbol": "PEPEUSDT", "quoteVolume": "5e8", "priceChangePercent": "3", "lastPrice": "1"},
        {"symbol": "WIFUSDT", "quoteVolume": "4e8", "priceChangePercent": "2", "lastPrice": "1"},
        {"symbol": "DOGEUSDT", "quoteVolume": "3e8", "priceChangePercent": "1", "lastPrice": "1"},
        {"symbol": "SHIBUSDT", "quoteVolume": "1000", "priceChangePercent": "1", "lastPrice": "1"},  # below min
        {"symbol": "BTCUPUSDT", "quoteVolume": "9e9", "priceChangePercent": "1", "lastPrice": "1"},
    ]
    monkeypatch.setattr(desk, "_http_json", lambda *a, **k: fake)
    out = desk.discover_alt_universe(limit=10)
    assert "BTCUSDT" not in out["symbols"]
    assert "BTCUPUSDT" not in out["symbols"]
    assert out["symbols"][:3] == ["PEPEUSDT", "WIFUSDT", "DOGEUSDT"]


def test_compute_risk():
    r = compute_risk(20_000)
    assert r["equity_usd"] == 20000
    assert r["satellite_cap_usd"] == 1000.0
    assert r["single_risk_usd"] == 200.0
    assert r["position_examples"][0]["max_notional_usd"] == 10000.0  # 200 / 0.02


def test_maybe_alert_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TRADING_OS_ALERTS", "0")
    from utils import trading_os_desk as desk

    monkeypatch.setattr(desk, "resolve_data_dir", lambda: Path(tmp_path))
    out = maybe_alert(phase="approach")
    assert out["enabled"] is False
    assert out["sent"] == []


def test_auto_monitor_boot_and_phase(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TRADING_OS_ALERTS", "1")
    monkeypatch.setenv("TRADING_OS_DIGEST_HOURS", "0")  # skip digest
    monkeypatch.setenv("TRADING_OS_SCORE_ALERT_COOLDOWN_H", "99")
    from utils import trading_os_desk as desk

    monkeypatch.setattr(desk, "resolve_data_dir", lambda: Path(tmp_path))
    sent_msgs: list[str] = []
    monkeypatch.setattr(desk, "_send_tg", lambda t: sent_msgs.append(t) or True)

    snap = {
        "price": 50000,
        "cvdd": {"cvdd": 46000},
        "score": {"phase": "wait", "score": 1, "score_max": 4, "distance_pct": 8, "signals": {}},
        "desk": {"wallets": {"events": [], "balances": {}}, "alts": {"flagged": []}, "strategy": {"lanes": {}}},
    }
    out1 = desk.run_auto_monitor(snap)
    assert out1["enabled"] is True
    assert "boot" in out1["sent"]
    assert any("已启动" in m for m in sent_msgs)

    snap["score"]["phase"] = "approach"
    snap["score"]["score"] = 3
    snap["score"]["signals"] = {"cvdd_near": True, "taker_weak": True}
    monkeypatch.setenv("TRADING_OS_SCORE_ALERT_COOLDOWN_H", "0")
    # force cooldown clear by rewriting — score with 0 hours always ok
    out2 = desk.run_auto_monitor(snap)
    assert any(s.startswith("phase:") for s in out2["sent"])

    st = desk.monitor_status()
    assert st["booted"] is True
    assert st["last_phase"] == "approach"


def test_strategy_signal_alert(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TRADING_OS_ALERTS", "1")
    monkeypatch.setenv("TRADING_OS_BOOT_PING", "0")
    monkeypatch.setenv("TRADING_OS_DIGEST_HOURS", "0")
    monkeypatch.setenv("TRADING_OS_SCORE_ALERT_COOLDOWN_H", "99")
    from utils import trading_os_desk as desk

    monkeypatch.setattr(desk, "resolve_data_dir", lambda: Path(tmp_path))
    msgs: list[str] = []
    monkeypatch.setattr(desk, "_send_tg", lambda t: msgs.append(t) or True)

    base = {
        "score": {"phase": "bull", "score": 0, "score_max": 4, "signals": {}},
        "desk": {
            "wallets": {"events": [], "balances": {}},
            "alts": {"flagged": []},
            "strategy": {"lanes": {"donchian": {"signals": [{"id": 10, "symbol": "BTCUSDT", "side": "long"}]}}},
        },
    }
    desk.run_auto_monitor(base)  # seed
    msgs.clear()
    base["desk"]["strategy"]["lanes"]["donchian"]["signals"] = [
        {"id": 11, "symbol": "ETHUSDT", "side": "long"},
        {"id": 10, "symbol": "BTCUSDT", "side": "long"},
    ]
    out = desk.run_auto_monitor(base)
    assert any(s.startswith("strategy:donchian:11") for s in out["sent"])
    assert any("ETHUSDT" in m for m in msgs)

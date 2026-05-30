from __future__ import annotations

import pytest
import httpx


def test_live_notional_uses_protocol_balance_and_leverage():
    from moss_quant.paper_scanner import live_notional_from_account

    params = {"risk_per_trade": 0.10, "max_position_pct": 0.50}
    notional = live_notional_from_account(
        wallet_balance_usdt=1000,
        enabled_profile_count=5,
        protocol_leverage=8,
        params=params,
    )

    assert notional == 160.0


def test_live_notional_rejects_invalid_inputs():
    from moss_quant.paper_scanner import live_notional_from_account

    with pytest.raises(ValueError, match="enabled_profile_count"):
        live_notional_from_account(
            wallet_balance_usdt=1000,
            enabled_profile_count=0,
            protocol_leverage=8,
            params={"risk_per_trade": 0.1, "max_position_pct": 0.5},
        )


@pytest.mark.parametrize("risk", ["nan", -0.1])
def test_live_notional_rejects_invalid_risk(risk):
    from moss_quant.paper_scanner import live_notional_from_account

    with pytest.raises(ValueError, match="risk_per_trade"):
        live_notional_from_account(
            wallet_balance_usdt=1000,
            enabled_profile_count=5,
            protocol_leverage=8,
            params={"risk_per_trade": risk, "max_position_pct": 0.5},
        )


def test_protocol_client_builds_headers(monkeypatch):
    monkeypatch.setenv("PROTOCOL_API_URL", "http://protocol.test")
    monkeypatch.setenv("PROTOCOL_MAINTENANCE_TOKEN", "secret")

    from moss_quant.protocol_client import ProtocolClient

    c = ProtocolClient.from_env()
    assert c.base_url == "http://protocol.test"
    assert c.headers()["X-Maintenance-Token"] == "secret"


def test_protocol_client_surfaces_protocol_detail(monkeypatch):
    from moss_quant.protocol_client import ProtocolClient

    req = httpx.Request("GET", "http://protocol.test/api/binance/account/summary")
    resp = httpx.Response(
        502,
        request=req,
        json={"detail": "account_summary_failed_upstream_401"},
    )

    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: resp)

    with pytest.raises(RuntimeError, match="account_summary_failed_upstream_401"):
        ProtocolClient(base_url="http://protocol.test").get_account_summary()


def test_protocol_update_sl_omits_missing_profile_id(monkeypatch):
    from moss_quant.protocol_client import ProtocolClient

    captured = {}

    def fake_put(self, path, body):
        captured["path"] = path
        captured["body"] = body
        return {"ok": True}

    monkeypatch.setattr(ProtocolClient, "_put", fake_put)

    c = ProtocolClient(base_url="http://protocol.test")
    c.send_update_sl(position_id=12, new_sl_price=123.45)

    assert captured["path"] == "/api/binance/positions/12/sl"
    assert captured["body"]["new_sl_price"] == 123.45
    assert "profile_id" not in captured["body"]


def test_signal_sender_rolling_action_is_stable(monkeypatch):
    from moss_quant import signal_sender

    captured = {}

    class FakeClient:
        def send_open(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

    monkeypatch.setattr(signal_sender, "is_real_mode", lambda: True)
    monkeypatch.setattr(signal_sender, "_client", lambda: FakeClient())

    signal_sender.send_rolling(
        symbol="BTCUSDT",
        side="LONG",
        notional=100,
        profile_id=7,
        play="trend",
        sl_price=90,
        tp_price=120,
        rolling_count=3,
    )

    assert captured["action"] == "rolling"


def test_signal_sender_fetch_position_id_filters_profile(monkeypatch):
    from moss_quant import signal_sender

    class FakeClient:
        def get_moss_positions(self, status="open", limit=200):
            return [
                {"id": 91, "source": "moss_quant", "symbol": "BTCUSDT", "profile_id": 9},
                {"id": 71, "source": "moss_quant", "symbol": "BTCUSDT", "profile_id": 7},
            ]

    monkeypatch.setattr(signal_sender, "_client", lambda timeout=10: FakeClient())

    assert signal_sender.fetch_and_cache_position_id("BTCUSDT", 7) == 71

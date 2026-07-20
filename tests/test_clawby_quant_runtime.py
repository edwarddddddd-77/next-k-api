"""clawby-quant embed helpers (no network)."""

from __future__ import annotations

from utils.clawby_quant_runtime import clawby_base_url, clawby_port, embed_enabled, status, vendor_root


def test_vendor_root_exists():
    root = vendor_root()
    assert (root / "backend" / "main.py").is_file()
    assert (root / "strategies.yaml").is_file()


def test_defaults(monkeypatch):
    monkeypatch.delenv("NEXT_K_CLAWBY_EMBED", raising=False)
    monkeypatch.delenv("CLAWBY_QUANT_URL", raising=False)
    monkeypatch.delenv("CLAWBY_QUANT_PORT", raising=False)
    assert embed_enabled() is True
    assert clawby_port() == 8899
    assert clawby_base_url() == "http://127.0.0.1:8899"
    st = status()
    assert st["ok"] is True
    assert "vendor_root" in st

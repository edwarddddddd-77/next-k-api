"""Live 人工替换包路径测试。"""

from __future__ import annotations

from orb.ml.live_bundle import live_bundle_root, resolve_live_gate_path, resolve_live_gbm_path
from orb.ml.model import layout_status, resolve_gbm_path


def test_live_bundle_root():
    assert str(live_bundle_root()).replace("\\", "/").endswith("data/orb/live")


def test_resolve_prefers_live_bundle():
    assert "data/orb/live" in str(resolve_gbm_path()).replace("\\", "/")
    assert str(resolve_live_gate_path()).endswith(".json")


def test_layout_includes_live_bundle():
    st = layout_status()
    assert "live_bundle_root" in st
    assert "data/orb/live" in st["live_bundle_root"].replace("\\", "/")


def test_live_bundle_hint_ready():
    from orb.ml.live_bundle import live_bundle_hint

    hint = live_bundle_hint()
    assert hint["ok"] is True
    assert "message" in hint
    assert hint["severity"] in ("ok", "warn", "block")
    assert "data/orb/live" in hint.get("root", "").replace("\\", "/")

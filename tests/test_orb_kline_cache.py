"""K 线缓存路径测试。"""

from __future__ import annotations

from quant.common.kline_cache import default_cache_root, legacy_kline_path
from quant.common.paths import KLINE_ROOT


def test_kline_default_root_is_data_orb_kline():
    root = default_cache_root()
    norm = str(root).replace("\\", "/")
    assert norm.endswith("data/orb/kline")


def test_legacy_fallback_path_under_output():
    p = legacy_kline_path("TSLAUSDT", "5m")
    assert "output" in str(p).replace("\\", "/")
    assert p.name == "5m.csv"


def test_kline_root_constant():
    assert str(KLINE_ROOT).replace("\\", "/").endswith("data/orb/kline")

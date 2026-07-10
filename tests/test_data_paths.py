"""DATA_DIR / K 线缓存路径解析。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quant.common import paths


class TestDataPaths(unittest.TestCase):
    def test_kline_root_under_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {"DATA_DIR": tmp},
                clear=False,
            ):
                os.environ.pop("ORB_KLINE_CACHE_ROOT", None)
                self.assertEqual(
                    paths.resolve_kline_cache_root(),
                    Path(tmp) / "orb" / "kline",
                )

    def test_explicit_kline_root_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "custom_kline"
            with mock.patch.dict(
                os.environ,
                {"ORB_KLINE_CACHE_ROOT": str(custom)},
                clear=False,
            ):
                self.assertEqual(paths.resolve_kline_cache_root(), custom)

    def test_railway_volume_auto_detect(self):
        with tempfile.TemporaryDirectory() as tmp:
            app_data = Path(tmp) / "app" / "data"
            app_data.mkdir(parents=True)
            with mock.patch.object(paths, "_RAILWAY_DATA_CANDIDATES", (app_data,)):
                with mock.patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(paths.resolve_data_dir(), app_data)


if __name__ == "__main__":
    unittest.main()

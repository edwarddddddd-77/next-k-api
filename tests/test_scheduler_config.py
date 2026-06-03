"""调度开关默认行为。"""

from __future__ import annotations

import os
import unittest
from unittest import mock


class TestEmbedSchedulerDefault(unittest.TestCase):
    def test_embed_scheduler_default_on_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEXT_K_EMBED_SCHEDULER", None)
            import importlib

            import scheduler_config as sc

            importlib.reload(sc)
            self.assertTrue(sc.embed_scheduler_enabled())

    def test_embed_scheduler_off_when_zero(self):
        with mock.patch.dict(os.environ, {"NEXT_K_EMBED_SCHEDULER": "0"}):
            import importlib

            import scheduler_config as sc

            importlib.reload(sc)
            self.assertFalse(sc.embed_scheduler_enabled())


if __name__ == "__main__":
    unittest.main()

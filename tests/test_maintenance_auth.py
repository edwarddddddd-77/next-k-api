#!/usr/bin/env python3
"""维护鉴权单元测试（鉴权已禁用）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parent.parent
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

from utils.maintenance_token import verify_maintenance_token


class MaintenanceAuthTests(unittest.TestCase):
    def test_always_passes(self) -> None:
        verify_maintenance_token(None, None)
        verify_maintenance_token("any", None)
        verify_maintenance_token(None, "Bearer any")


if __name__ == "__main__":
    unittest.main()

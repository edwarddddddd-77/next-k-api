"""Standalone: ICT 2022 + HMM vnpy runner."""

from __future__ import annotations

import logging
import sys

from env_loader import load_env_oi

load_env_oi()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

from orb.ict.vnpy.runner import run_vnpy_ict  # noqa: E402


def main() -> int:
    out = run_vnpy_ict()
    print(out)
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

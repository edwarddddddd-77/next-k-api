#!/usr/bin/env python3
"""
按 Moss2 种子目录（moss2.config.MOSS2_SEED_BASES）批量拉 factory-en 币安 U 本位 15m CSV，
供 Moss2 回测 / 进化 / discipline / onboarding 使用。

输出目录（默认）:
  moss-trade-bot-skills-main/moss-trade-bot-factory-en-1.0.3/data_cache/
  文件名形如: binanceusdm_BTC_USDT_USDT_15m_2025-10-06_148d.csv

用法（在 next-k-api 目录）:
  python scripts/fetch_factory_en_moss_universe.py
  python scripts/fetch_factory_en_moss_universe.py --dry-run
  python scripts/fetch_factory_en_moss_universe.py --skip-existing
  python scripts/fetch_factory_en_moss_universe.py --bases BTC,ETH,SOL
  python scripts/fetch_factory_en_moss_universe.py --sleep 2.0
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

# next-k-api 根目录
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss2.config import MOSS2_SEED_BASES, base_to_fetch_slash


def _factory_en_scripts_dir() -> Path:
    skills = ROOT.parent / "moss-trade-bot-skills-main"
    en = skills / "moss-trade-bot-factory-en-1.0.3" / "scripts"
    if en.is_dir():
        return en
    raise FileNotFoundError(
        f"factory-en scripts not found under {skills}; "
        "clone moss-trade-bot-skills-main first."
    )


def _python_bin(scripts_dir: Path) -> str:
    venv = scripts_dir.parent / ".venv" / "Scripts" / "python.exe"
    if venv.is_file():
        return str(venv)
    venv_unix = scripts_dir.parent / ".venv" / "bin" / "python"
    if venv_unix.is_file():
        return str(venv_unix)
    return sys.executable


def _slash_symbol(base: str) -> str:
    return base_to_fetch_slash(base)


def _has_csv(data_cache: Path, base: str) -> bool:
    core = base.upper().replace("USDT", "")
    pattern = str(data_cache / f"binanceusdm_{core}_USDT_USDT_15m*.csv")
    return bool(glob.glob(pattern))


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch fetch factory-en CSV for Moss1 universe")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    parser.add_argument("--skip-existing", action="store_true", help="data_cache 已有则跳过")
    parser.add_argument("--bases", default="", help="逗号分隔 base，默认 MOSS2_SEED_BASES")
    parser.add_argument("--days", type=int, default=148)
    parser.add_argument("--since", default="2025-10-06")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--sleep", type=float, default=1.5, help="每个币之间的间隔秒")
    args = parser.parse_args()

    scripts_dir = _factory_en_scripts_dir()
    data_cache = scripts_dir.parent / "data_cache"
    data_cache.mkdir(parents=True, exist_ok=True)
    py = _python_bin(scripts_dir)
    fetch_py = scripts_dir / "fetch_data.py"
    if not fetch_py.is_file():
        print(f"Missing {fetch_py}", file=sys.stderr)
        return 1

    if args.bases.strip():
        bases = [b.strip().upper() for b in args.bases.split(",") if b.strip()]
    else:
        bases = list(MOSS2_SEED_BASES)

    print(f"factory-en: {scripts_dir.parent}")
    print(f"data_cache: {data_cache}")
    print(f"python: {py}")
    print(f"symbols: {len(bases)} bases\n")

    ok, skip, fail = [], [], []
    for i, base in enumerate(bases, 1):
        sym = _slash_symbol(base)
        if args.skip_existing and _has_csv(data_cache, base):
            skip.append(base)
            print(f"[{i}/{len(bases)}] SKIP {base} ({sym}) — CSV exists")
            continue

        cmd = [
            py,
            str(fetch_py),
            "--symbol",
            sym,
            "--timeframe",
            args.timeframe,
            "--days",
            str(args.days),
            "--since",
            args.since,
        ]
        print(f"[{i}/{len(bases)}] FETCH {base} -> {sym}")
        if args.dry_run:
            print("  ", " ".join(cmd))
            ok.append(base)
            continue

        try:
            subprocess.run(cmd, cwd=str(scripts_dir), check=True, timeout=600)
            ok.append(base)
        except subprocess.CalledProcessError as e:
            fail.append((base, str(e)))
            print(f"  FAIL {base}: {e}", file=sys.stderr)
        except Exception as e:
            fail.append((base, str(e)))
            print(f"  FAIL {base}: {e}", file=sys.stderr)

        if i < len(bases) and args.sleep > 0 and not args.dry_run:
            time.sleep(args.sleep)

    print("\n--- done ---")
    print(f"ok={len(ok)} skip={len(skip)} fail={len(fail)}")
    if fail:
        print("failed:", ", ".join(b for b, _ in fail))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

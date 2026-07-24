"""Build / refresh the desk follow candidate pool (hl_desk_candidates.json)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.hl_desk_candidates import build_candidate_pool  # noqa: E402


def main() -> None:
    out = build_candidate_pool()
    print(
        f"ok scanned={out.get('scanned')} hits={out.get('hit_count')} "
        f"ready={out.get('ready_count')} watch={out.get('watch_count')} "
        f"bound={out.get('bound_count')}",
        flush=True,
    )
    print("=== READY (bindable) ===", flush=True)
    for i, h in enumerate(out.get("ready") or [], 1):
        print(
            f"{i:02d} score={h.get('score')} wr={float(h.get('wr7') or 0):.0%} "
            f"n={h.get('trips7')} live={h.get('live_av')} quiet={h.get('quiet_h')}h "
            f"{h.get('coins')} {h.get('addr')}",
            flush=True,
        )
    print("=== WATCH (passed 7d, not live enough) ===", flush=True)
    for i, h in enumerate((out.get("watch") or [])[:15], 1):
        print(
            f"{i:02d} score={h.get('score')} wr={float(h.get('wr7') or 0):.0%} "
            f"quiet={h.get('quiet_h')}h live={h.get('live_av')} {h.get('addr')}",
            flush=True,
        )


if __name__ == "__main__":
    main()

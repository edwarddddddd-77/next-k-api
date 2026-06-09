from __future__ import annotations

from datetime import datetime, timezone


class AppState:
    startup_time: datetime | None = None


state = AppState()

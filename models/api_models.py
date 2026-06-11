from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    scheduler_embedded: bool = False
    scheduler_running: bool = False


class ClearWatchTablesBody(BaseModel):
    tables: List[str] = Field(default_factory=lambda: ["ambush_watch"])


class TriggerCronBody(BaseModel):
    task: str = Field(...)

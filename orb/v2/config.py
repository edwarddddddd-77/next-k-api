"""ORB 2.0 配置：继承 1.0 OrbConfig，附加 ML gate、模型路径与独立标的池。"""



from __future__ import annotations



import os

from dataclasses import dataclass, field

from pathlib import Path

from typing import List



from orb.core.config import OrbConfig

from orb.ml.gate import LiveGateConfig

from orb.ml.samples import parse_symbol_list

from orb.ml.live_bundle import resolve_live_gate_path, resolve_live_gbm_path, resolve_live_profiles_path
from orb.v2.paths import resolve_symbols_path


def _env_truthy(name: str, *, default: bool = False) -> bool:

    raw = os.getenv(name, "")

    if not str(raw).strip():

        return default

    return str(raw).strip().lower() in ("1", "true", "yes", "on")





def _load_symbols(*, env_override: str, symbols_file: Path) -> List[str]:

    if env_override.strip():

        return parse_symbol_list(env_override)

    if symbols_file.is_file():

        return parse_symbol_list(symbols_file.read_text(encoding="utf-8"))

    return []





@dataclass

class OrbV2Config:

    """ORB 2.0：1.0 策略参数 + ML gate + 独立扫描标的。"""



    base: OrbConfig

    symbols: List[str] = field(default_factory=list)

    symbols_file: Path = Path()

    enabled: bool = False

    shadow: bool = False

    gate_config_path: Path = Path()

    gbm_path: Path = Path()

    profiles_path: Path = Path()



    @classmethod

    def from_env(cls) -> "OrbV2Config":

        base = OrbConfig.from_env()

        gate_path = resolve_live_gate_path()

        gbm_path = resolve_live_gbm_path()

        profiles_path = resolve_live_profiles_path()

        symbols_file = resolve_symbols_path()

        symbols = _load_symbols(

            env_override=os.getenv("ORB_V2_SYMBOLS", ""),

            symbols_file=symbols_file,

        )

        enabled = _env_truthy("ORB_V2_ENABLED", default=True)

        shadow = _env_truthy("ORB_V2_SHADOW", default=False)

        return cls(

            base=base,

            symbols=symbols,

            symbols_file=symbols_file,

            enabled=enabled,

            shadow=shadow,

            gate_config_path=gate_path,

            gbm_path=gbm_path,

            profiles_path=profiles_path,

        )



    def symbol_list(self) -> List[str]:

        """V2 扫描标的（`ORB_V2_SYMBOLS` / `ORB_V2_SYMBOLS_FILE`）。"""
        return list(self.symbols)



    def load_gate(self) -> LiveGateConfig:
        return LiveGateConfig.from_json(resolve_live_gate_path())



    @property

    def lane(self) -> str:

        return "orb_v2"




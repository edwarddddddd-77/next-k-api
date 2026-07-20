"""Configuration: .env credentials + strategies.yaml (runtime-editable).

strategies.yaml is the single source of truth for strategy switches / params /
per-strategy symbols / per-strategy risk + the global monitored universe.
Credentials live in .env and are editable at runtime via the Config page.
"""
import os
import threading
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
STRATEGIES_PATH = ROOT / "strategies.yaml"
DB_PATH = os.environ.get("QB_DB_PATH", str(ROOT / "quantbot.db"))

_lock = threading.Lock()
_env_lock = threading.Lock()


def _load_env():
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


_load_env()

# credentials — module globals; read at call-time by clawby/binance so runtime
# edits (via save_credentials) take effect without restart
CLAWBY_API_KEY = os.environ.get("CLAWBY_API_KEY", "")
CLAWBY_BASE = os.environ.get("CLAWBY_BASE", "https://api.openclawby.com")
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY", "")
BITGET_API_KEY = os.environ.get("BITGET_API_KEY", "")
BITGET_SECRET_KEY = os.environ.get("BITGET_SECRET_KEY", "")
BITGET_PASSPHRASE = os.environ.get("BITGET_PASSPHRASE", "")
OKX_API_KEY = os.environ.get("OKX_API_KEY", "")
OKX_SECRET_KEY = os.environ.get("OKX_SECRET_KEY", "")
OKX_PASSPHRASE = os.environ.get("OKX_PASSPHRASE", "")

DEFAULT_UNIVERSE = [s.strip() for s in os.environ.get(
    "QB_UNIVERSE", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT").split(",") if s.strip()]
UNIVERSE = list(DEFAULT_UNIVERSE)   # reassigned from strategies.yaml on load

PAPER_START_BALANCE = float(os.environ.get("QB_PAPER_BALANCE", "10000"))
PAPER_FEE_RATE = 0.0005      # taker
PAPER_SLIPPAGE = 0.0003

ENGINE_TICK_SEC = 30  # legacy, loops now run at 0.5s (engine.LOOP_SEC)

# common futures symbols offered in the universe picker (extend freely)
KNOWN_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "TONUSDT", "TRXUSDT", "DOTUSDT",
    "POLUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT",
    "OPUSDT", "SUIUSDT", "INJUSDT", "SEIUSDT", "TIAUSDT", "1000PEPEUSDT",
    "WIFUSDT", "1000SHIBUSDT", "ORDIUSDT", "FILUSDT", "ATOMUSDT", "UNIUSDT",
]

DEFAULT_STRATEGY_RISK = {
    "capital_usd": 0,        # 0 = 用全账户净值参与计算
    "leverage": 3,           # 该策略下单杠杆
    "max_position_usd": 0,   # 单笔名义上限,0 = capital×leverage
    "risk_per_trade_pct": 0, # 单笔风险占资金 %,0 = 用全局
}


# -- strategies.yaml (with migration) ---------------------------------------

def _migrate(cfg):
    """Ensure global.universe + per-strategy symbols/risk exist. Returns
    (cfg, changed)."""
    changed = False
    g = cfg.setdefault("global", {})
    if "universe" not in g or not g["universe"]:
        g["universe"] = list(DEFAULT_UNIVERSE)
        changed = True
    for sid, scfg in cfg.get("strategies", {}).items():
        if "symbols" not in scfg:
            scfg["symbols"] = []        # empty = follow global universe
            changed = True
        risk = scfg.setdefault("risk", {})
        for k, v in DEFAULT_STRATEGY_RISK.items():
            if k not in risk:
                risk[k] = v
                changed = True
    return cfg, changed


def load_strategies():
    with _lock:
        cfg = yaml.safe_load(STRATEGIES_PATH.read_text(encoding="utf-8"))
    cfg, changed = _migrate(cfg)
    if changed:
        _write(cfg)
    # keep the runtime universe in sync with persisted config
    global UNIVERSE
    uni = cfg.get("global", {}).get("universe")
    if uni:
        UNIVERSE = list(uni)
    return cfg


def _write(cfg):
    with _lock:
        STRATEGIES_PATH.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def save_strategies(cfg):
    _write(cfg)


def _update_strategy(sid, mutate):
    cfg = load_strategies()
    if sid not in cfg.get("strategies", {}):
        raise KeyError(sid)
    mutate(cfg["strategies"][sid])
    save_strategies(cfg)
    return cfg


def set_strategy_enabled(sid, enabled):
    return _update_strategy(sid, lambda s: s.update(enabled=bool(enabled)))


def set_strategy_params(sid, params):
    """Merge param edits (numbers/bools/strings only) into the instance."""
    def mutate(s):
        blk = s.setdefault("params", {})
        for k, v in (params or {}).items():
            if isinstance(v, (int, float, bool, str)):
                blk[str(k)] = v
    return _update_strategy(sid, mutate)


def set_strategy_display_name(sid, name):
    return _update_strategy(sid, lambda s: s.update(display_name=str(name)[:60]))


def create_strategy_instance(base, name=""):
    """Clone a template's config into a new instance (enabled=false)."""
    cfg = load_strategies()
    strategies = cfg.setdefault("strategies", {})
    template = None
    for iid, scfg in strategies.items():
        if (scfg.get("base") or iid) == base:
            template = scfg
            break
    if template is None:
        raise KeyError(base)
    n = 2
    while f"{base}__{n}" in strategies:
        n += 1
    iid = f"{base}__{n}"
    import copy
    inst = copy.deepcopy(template)
    inst["base"] = base
    inst["enabled"] = False
    if name:
        inst["display_name"] = str(name)[:60]
    strategies[iid] = inst
    save_strategies(cfg)
    return iid


def delete_strategy_instance(iid):
    cfg = load_strategies()
    if iid not in cfg.get("strategies", {}):
        raise KeyError(iid)
    del cfg["strategies"][iid]
    save_strategies(cfg)


def set_strategy_interval(sid, interval_sec):
    return _update_strategy(sid, lambda s: s.update(scan_interval=max(1, int(interval_sec))))


def set_strategy_symbols(sid, symbols):
    syms = [str(x).upper() for x in (symbols or [])]
    allowed = set(UNIVERSE) | set(KNOWN_SYMBOLS)
    bad = [s for s in syms if s not in allowed]
    if bad:
        raise ValueError(f"未知交易币种: {','.join(bad[:5])}(需在全局监控或已知币列表内)")
    return _update_strategy(sid, lambda s: s.update(symbols=syms))


def set_strategy_risk(sid, risk):
    def mutate(s):
        blk = s.setdefault("risk", dict(DEFAULT_STRATEGY_RISK))
        for k in DEFAULT_STRATEGY_RISK:
            if k in risk and risk[k] is not None:
                blk[k] = float(risk[k]) if k != "leverage" else int(risk[k])
    return _update_strategy(sid, mutate)


def set_global_risk(updates):
    cfg = load_strategies()
    g = cfg.setdefault("global", {})
    for k in ("max_gross_leverage", "daily_loss_halt_pct", "risk_per_trade_pct",
              "max_concurrent_coins", "max_position_pct_per_coin", "event_quiet_minutes"):
        if k in updates and updates[k] is not None:
            g[k] = updates[k]
    save_strategies(cfg)
    return cfg


def set_universe(symbols):
    syms = [str(x).upper() for x in (symbols or []) if str(x).strip()]
    if not syms:
        raise ValueError("universe cannot be empty")
    cfg = load_strategies()
    cfg["global"]["universe"] = syms
    save_strategies(cfg)
    global UNIVERSE
    UNIVERSE = list(syms)
    return syms


# -- credentials (.env, runtime-editable) -----------------------------------

_CRED_KEYS = ("CLAWBY_API_KEY", "CLAWBY_BASE", "BINANCE_API_KEY",
              "BINANCE_SECRET_KEY", "BINANCE_API_ENV",
              "BITGET_API_KEY", "BITGET_SECRET_KEY", "BITGET_PASSPHRASE",
              "OKX_API_KEY", "OKX_SECRET_KEY", "OKX_PASSPHRASE")


def _mask(v):
    if not v:
        return ""
    if len(v) <= 10:
        return v[:2] + "…" + v[-2:]
    return v[:6] + "…" + v[-4:]


def credentials_masked():
    return {
        "clawby_api_key": _mask(CLAWBY_API_KEY),
        "clawby_base": CLAWBY_BASE,
        "binance_api_key": _mask(BINANCE_API_KEY),
        "binance_secret_key": _mask(BINANCE_SECRET_KEY) if BINANCE_SECRET_KEY else "",
        "binance_env": os.environ.get("BINANCE_API_ENV", "prod"),
        "bitget_api_key": _mask(BITGET_API_KEY),
        "okx_api_key": _mask(OKX_API_KEY),
        "has_clawby": bool(CLAWBY_API_KEY),
        "has_binance": bool(BINANCE_API_KEY and BINANCE_SECRET_KEY),
        "has_bitget": bool(BITGET_API_KEY and BITGET_SECRET_KEY and BITGET_PASSPHRASE),
        "has_okx": bool(OKX_API_KEY and OKX_SECRET_KEY and OKX_PASSPHRASE),
    }


def save_credentials(updates):
    """Update .env (preserving other lines) and hot-swap module globals.
    updates keys: clawby_api_key / binance_api_key / binance_secret_key / clawby_base."""
    global CLAWBY_API_KEY, CLAWBY_BASE, BINANCE_API_KEY, BINANCE_SECRET_KEY, \
        BITGET_API_KEY, BITGET_SECRET_KEY, BITGET_PASSPHRASE, \
        OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE
    mapping = {
        "clawby_api_key": "CLAWBY_API_KEY",
        "clawby_base": "CLAWBY_BASE",
        "binance_api_key": "BINANCE_API_KEY",
        "binance_secret_key": "BINANCE_SECRET_KEY",
        "binance_env": "BINANCE_API_ENV",
        "bitget_api_key": "BITGET_API_KEY",
        "bitget_secret_key": "BITGET_SECRET_KEY",
        "bitget_passphrase": "BITGET_PASSPHRASE",
        "okx_api_key": "OKX_API_KEY",
        "okx_secret_key": "OKX_SECRET_KEY",
        "okx_passphrase": "OKX_PASSPHRASE",
    }
    to_set = {}
    for k, envk in mapping.items():
        if k in updates and updates[k] not in (None, ""):
            to_set[envk] = str(updates[k]).strip()

    with _env_lock:
        lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
        seen = set()
        out = []
        for line in lines:
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                key = s.split("=", 1)[0].strip()
                if key in to_set:
                    out.append(f"{key}={to_set[key]}")
                    seen.add(key)
                    continue
            out.append(line)
        for key, val in to_set.items():
            if key not in seen:
                out.append(f"{key}={val}")
        ENV_PATH.write_text("\n".join(out) + "\n")
        ENV_PATH.chmod(0o600)

    for envk, val in to_set.items():
        os.environ[envk] = val
    CLAWBY_API_KEY = os.environ.get("CLAWBY_API_KEY", "")
    CLAWBY_BASE = os.environ.get("CLAWBY_BASE", "https://api.openclawby.com")
    BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY", "")
    BITGET_API_KEY = os.environ.get("BITGET_API_KEY", "")
    BITGET_SECRET_KEY = os.environ.get("BITGET_SECRET_KEY", "")
    BITGET_PASSPHRASE = os.environ.get("BITGET_PASSPHRASE", "")
    OKX_API_KEY = os.environ.get("OKX_API_KEY", "")
    OKX_SECRET_KEY = os.environ.get("OKX_SECRET_KEY", "")
    OKX_PASSPHRASE = os.environ.get("OKX_PASSPHRASE", "")
    return credentials_masked()

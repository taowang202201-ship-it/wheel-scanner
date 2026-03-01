from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path("config.json")

DEFAULT_CONFIG: Dict[str, Any] = {
    "watchlist": ["NVDA", "SOFI", "TSLA"],
    "include_put": True,
    "include_call": True,
    "target_delta": 0.20,
    "min_delta": 0.15,
    "max_delta": 0.25,
    "min_annualized": 0.15,
    "min_dte": 14,
    "max_dte": 60,
    "require_liquid": True,
    "global_top_n": 10,
    "per_ticker_top_k": 3,
    "put_otm_min_pct": 0.02,
    "put_otm_max_pct": 0.12,
    "put_otm_target_pct": 0.05,
    "cost_basis_map": {},
    "call_min_above_cost_pct": 0.02,
    "call_target_above_cost_pct": 0.06,
    "earnings_block_days_before": 7,
    "earnings_block_days_after": 1,
    "earnings_filter_mode": "strict",
    "max_per_sector": 2,
}


def load_config() -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if not CONFIG_PATH.exists():
        return cfg

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            cfg.update(data)
    except Exception:
        return cfg

    if not isinstance(cfg.get("watchlist"), list):
        cfg["watchlist"] = DEFAULT_CONFIG["watchlist"]
    if not isinstance(cfg.get("cost_basis_map"), dict):
        cfg["cost_basis_map"] = {}

    return cfg


def save_config(config: Dict[str, Any]) -> None:
    payload = DEFAULT_CONFIG.copy()
    payload.update(config or {})
    CONFIG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

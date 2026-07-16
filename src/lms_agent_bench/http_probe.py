"""http_probe.py — shared HTTP base-URL normalization + endpoint probing.

Consolidates the duplicated ``normalize_base_url`` / ``probe_endpoint`` /
``probe_endpoints`` logic that previously lived in lms_endpoint_registry,
fleet_orchestrator, and the benchmark runners (see docs/LLD_UNIFIED_FLEET.md
W-69). Every caller should import from here instead of re-implementing it.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence


def normalize_base_url(value: str) -> str:
    """Normalize an endpoint string to a trailing-slash ``.../v1`` base URL."""
    value = value.strip().rstrip("/")
    if not value.startswith("http://") and not value.startswith("https://"):
        value = "http://" + value
    if not value.endswith("/v1"):
        value += "/v1"
    return value


def probe_endpoint(base_url: str, timeout: int = 8) -> Dict[str, Any]:
    """Probe ``base_url`` for LM Studio reachability; return a result dict with
    ``reachable``, ``models``, ``model_count``, ``error``, etc."""
    url = normalize_base_url(base_url).rstrip("/") + "/models"
    started = dt.datetime.now(dt.timezone.utc)
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(5_000_000).decode("utf-8", errors="replace")
            data = json.loads(body)
            models: List[str] = []
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                models = [
                    str(m["id"])
                    for m in data["data"]
                    if isinstance(m, dict) and m.get("id")
                ]
            elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
            return {
                "reachable": True,
                "status": getattr(resp, "status", None),
                "elapsed_s": round(elapsed, 4),
                "models": models,
                "model_count": len(models),
                "error": None,
            }
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
        return {
            "reachable": False,
            "status": None,
            "elapsed_s": round(elapsed, 4),
            "models": [],
            "model_count": 0,
            "error": repr(exc),
        }


def probe_endpoints(
    endpoints: Sequence[Dict[str, Any]],
    url_key: str = "base_url",
    timeout: int = 8,
    max_workers: int = 8,
) -> List[Dict[str, Any]]:
    """Probe many endpoints concurrently; returns the same result dicts augmented
    with the original endpoint entry under ``endpoint``."""
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(endpoints)))) as pool:
        future_map = {
            pool.submit(probe_endpoint, ep[url_key], timeout): ep
            for ep in endpoints
        }
        for future in as_completed(future_map):
            ep = future_map[future]
            probe = future.result()
            results.append({**ep, "probe": probe})
    return results

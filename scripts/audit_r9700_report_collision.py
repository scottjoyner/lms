#!/usr/bin/env python3
"""GET-only Joyner node-report collision witness. No POST, credentials or model calls."""
from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import time
from typing import Any

ROUTER_IP = "100.64.43.123"
ROUTER_PORT = 8088
PATH = "/api/fleet/node/joyner"
MAX_BYTES = 262144
MAX_SAMPLES = 24


def get_report() -> dict[str, Any] | None:
    """Call a pinned read-only router endpoint without forwarding credentials."""
    conn = http.client.HTTPConnection(ROUTER_IP, ROUTER_PORT, timeout=3.0)
    try:
        conn.request("GET", PATH, headers={"Accept": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            return None
        payload = response.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            return None
        data = json.loads(payload)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, UnicodeError, http.client.HTTPException):
        return None
    finally:
        conn.close()


def summarize(raw: Any, *, now: float | None = None) -> dict[str, Any]:
    """Exclude arbitrary specs contents, credentials and prompt data."""
    now = time.time() if now is None else now
    if not isinstance(raw, dict) or raw.get("hostname") != "joyner":
        return {
            "status": "UNKNOWN",
            "hostname": None,
            "spec_keys": [],
            "library_ids": [],
            "received_at": None,
            "loaded_model_attestation": "NOT_ATTESTED",
        }
    specs = raw.get("specs")
    if not isinstance(specs, dict):
        specs = {}
    entries = raw.get("library")
    ids: list[str] = []
    if isinstance(entries, list):
        for item in entries[:100]:
            ident = item.get("id") if isinstance(item, dict) else item
            if isinstance(ident, str) and ident in ("joyner-toolcall", "joyner-ling"):
                ids.append(ident)
    timestamp = raw.get("received_at")
    ts = float(timestamp) if type(timestamp) in (int, float) and 0 < timestamp <= now + 30 else None
    # A model advertised by an unauthenticated report does not prove GPU residency.
    return {
        "status": "OBSERVED",
        "hostname": "joyner",
        "spec_keys": sorted(str(k) for k in specs if isinstance(k, str) and len(k) < 80)[:100],
        "library_ids": sorted(set(ids)),
        "received_at": ts,
        "age_seconds": round(max(0.0, now - ts), 2) if ts is not None else None,
        "power_profile": str(raw.get("power_profile") or "")[:20],
        "power_model_class": str(raw.get("power_model_class") or "")[:20],
        "loaded_model_attestation": "NOT_ATTESTED",
    }


def classify(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        item for item in samples
        if item.get("status") == "OBSERVED"
        and item.get("received_at") is not None
        and item.get("age_seconds") is not None
        and item["age_seconds"] < 180
    ]
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    for item in valid:
        signature = tuple(item.get("spec_keys") or []) + (
            "|POWER|" + str(item.get("power_profile") or ""),
            "|CLASS|" + str(item.get("power_model_class") or ""),
        )
        group = groups.setdefault(signature, {"observations": 0, "catalogs": set(), "updates": set()})
        group["observations"] += 1
        group["catalogs"].add(tuple(item.get("library_ids") or []))
        group["updates"].add(item["received_at"])
    disjoint_catalogs = {
        catalog for group in groups.values() for catalog in group["catalogs"]
    }
    divergent = len(groups) >= 2 and len(disjoint_catalogs) >= 2
    finding = "CONFLICTING_REPORT_FORMATS" if divergent else (
        "INSUFFICIENT_WITNESS" if len(valid) < 2 else
        "ONE_FORMAT_OBSERVED" if len(groups) == 1 else
        "MULTIPLE_FORMATS_SAME_CATALOG"
    )
    return {
        "finding": finding,
        "valid_observations": len(valid),
        "report_formats": len(groups),
        "catalog_variants": len(disjoint_catalogs),
        "distinct_update_timestamps": len({item["received_at"] for item in valid}),
        "source_pid_identified": False,
        "authenticated_publisher_binding": False,
        "model_gpu_residency_proven": False,
        "production_admission": "NO_GO",
        "note": "Different report formats from one hostname suggest multiple producers, not a proven source PID. No routing or resident-model attestation.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=13)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args(argv)
    if not 1 <= args.samples <= MAX_SAMPLES or not 0 <= args.interval <= 30:
        parser.error("samples must be 1..24 and interval 0..30 seconds")
    samples: list[dict[str, Any]] = []
    for i in range(args.samples):
        samples.append(summarize(get_report()))
        if i + 1 < args.samples:
            time.sleep(args.interval)
    report = {
        "schema": "r9700-fleet-node-report-collision-audit.v1",
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "read_only": True,
        "router_path": PATH,
        "samples": samples,
        "assessment": classify(samples),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
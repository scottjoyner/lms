#!/usr/bin/env python3
"""
Discover LM Studio OpenAI-compatible models on a set of Tailscale IPs
and store results in Neo4j.

Default assumes LM Studio server is on port 1234 and OpenAI-compatible at /v1.

Usage:
  pip install neo4j requests
  export NEO4J_URI="bolt://localhost:7687"
  export NEO4J_USER="neo4j"
  export NEO4J_PASSWORD="password"

  python3 ts_lmstudio_inventory_to_neo4j.py \
    --port 1234 \
    --scheme http \
    --neo4j-db neo4j
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from neo4j import GraphDatabase
from requests.exceptions import RequestException


# ----------------------------
# Your Tailscale device list
# ----------------------------
RAW_DEVICES = [
    {
        "ts_name": "deathstar-xps-8920",
        "user_email": "kipnerter@gmail.com",
        "ts_ip": "100.78.106.121",
        "ts_version": "1.94.1",
        "os": "Linux 6.14.0-37-generic",
    },
    {
        "ts_name": "demo",
        "user_email": "kipnerter@gmail.com",
        "ts_ip": "100.67.106.114",
        "ts_version": "1.94.1",
        "os": "Windows 11 25H2",
    },
    {
        "ts_name": "destroyer",
        "user_email": "kipnerter@gmail.com",
        "ts_ip": "100.81.57.77",
        "ts_version": "1.92.3",
        "os": "Linux 6.14.0-33-generic",
    },
    {
        "ts_name": "iphone-12-pro-max",
        "user_email": "kipnerter@gmail.com",
        "ts_ip": "100.96.196.106",
        "ts_version": "1.92.3",
        "os": "iOS 17.5.1",
    },
    {
        "ts_name": "mini-pc-22",
        "user_email": "kipnerter@gmail.com",
        "ts_ip": "100.96.121.98",
        "ts_version": "1.92.3",
        "os": "Windows 11 25H2",
    },
    {
        "ts_name": "r2d2",
        "user_email": "kipnerter@gmail.com",
        "ts_ip": "100.105.87.118",
        "ts_version": "1.92.3",
        "os": "Windows 11 25H2",
    },
    {
        "ts_name": "raspberrypi",
        "user_email": "kipnerter@gmail.com",
        "ts_ip": "100.114.88.89",
        "ts_version": "1.92.5",
        "os": "Linux 6.12.62+rpt-rpi-v8",
    },
    {
        "ts_name": "scotts-macbook-air",
        "user_email": "kipnerter@gmail.com",
        "ts_ip": "100.85.64.117",
        "ts_version": "1.92.3",
        "os": "macOS 14.5.0",
    },
    {
        "ts_name": "scotts-macbook-pro-1",
        "user_email": "kipnerter@gmail.com",
        "ts_ip": "100.111.79.107",
        "ts_version": "1.92.3",
        "os": "macOS 13.0.0",
    },
    {
        "ts_name": "tie",
        "user_email": "kipnerter@gmail.com",
        "ts_ip": "100.96.18.65",
        "ts_version": "1.92.3",
        "os": "Linux 6.8.0-41-generic",
    },
]


@dataclass
class EndpointProbeResult:
    reachable: bool
    status_code: Optional[int]
    error: Optional[str]
    models: List[Dict[str, Any]]
    raw_response: Optional[Dict[str, Any]]


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_base_url(scheme: str, ip: str, port: int, api_root: str) -> str:
    api_root = api_root.strip()
    if not api_root.startswith("/"):
        api_root = "/" + api_root
    api_root = api_root.rstrip("/")
    return f"{scheme}://{ip}:{port}{api_root}"


def _safe_json(resp: requests.Response) -> Optional[Dict[str, Any]]:
    try:
        return resp.json()
    except Exception:
        return None


def fetch_models_openai_compatible(
    base_url: str,
    timeout_s: float,
    api_key: Optional[str],
) -> EndpointProbeResult:
    """
    Attempt to list models from an OpenAI-compatible server.

    Primary: GET {base_url}/models
    Fallbacks:
      - GET {base_url}/v1/models (if user passed base_url without /v1)
      - Also tolerate non-OpenAI-shaped payloads by best-effort normalization.
    """
    headers: Dict[str, str] = {"Accept": "application/json"}
    # LM Studio often doesn't require a key, but some OpenAI-compatible stacks do.
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    candidates = [
        f"{base_url}/models",
    ]

    # If user passed ".../v1" already, candidates covers it.
    # If they passed "...", add a common alternate.
    if not base_url.endswith("/v1"):
        candidates.append(f"{base_url}/v1/models")

    last_err = None
    last_status = None
    last_payload = None

    for url in candidates:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout_s)
            last_status = resp.status_code
            payload = _safe_json(resp)
            last_payload = payload

            if resp.status_code >= 400:
                # Keep trying fallbacks if payload indicates wrong path.
                last_err = f"HTTP {resp.status_code}"
                continue

            if not isinstance(payload, dict):
                last_err = "Non-JSON or non-dict response"
                continue

            # OpenAI models list shape: {"object":"list","data":[{...},{...}]}
            models = []
            if isinstance(payload.get("data"), list):
                models = [m for m in payload["data"] if isinstance(m, dict)]
            # Some servers might return {"models":[...]}
            elif isinstance(payload.get("models"), list):
                models = [m for m in payload["models"] if isinstance(m, dict)]
            # Or a raw list-like object keyed differently
            elif isinstance(payload.get("result"), list):
                models = [m for m in payload["result"] if isinstance(m, dict)]

            if models:
                return EndpointProbeResult(
                    reachable=True,
                    status_code=resp.status_code,
                    error=None,
                    models=models,
                    raw_response=payload,
                )

            # If the server is reachable but returns no recognizable models, still record reachability.
            return EndpointProbeResult(
                reachable=True,
                status_code=resp.status_code,
                error="Reachable, but model list payload not recognized or empty",
                models=[],
                raw_response=payload,
            )

        except RequestException as e:
            last_err = str(e)

    return EndpointProbeResult(
        reachable=False,
        status_code=last_status,
        error=last_err,
        models=[],
        raw_response=last_payload,
    )


def ensure_schema(driver, neo4j_db: str) -> None:
    """
    Create constraints/indexes (Neo4j 5.x+ syntax). Safe to run repeatedly.
    """
    cypher_statements = [
        "CREATE CONSTRAINT host_ts_name IF NOT EXISTS FOR (h:Host) REQUIRE h.ts_name IS UNIQUE",
        "CREATE CONSTRAINT endpoint_base_url IF NOT EXISTS FOR (e:Endpoint) REQUIRE e.base_url IS UNIQUE",
        "CREATE CONSTRAINT model_id IF NOT EXISTS FOR (m:Model) REQUIRE m.id IS UNIQUE",
        "CREATE INDEX host_ts_ip IF NOT EXISTS FOR (h:Host) ON (h.ts_ip)",
    ]

    with driver.session(database=neo4j_db) as session:
        for stmt in cypher_statements:
            session.run(stmt)


def upsert_inventory(
    driver,
    neo4j_db: str,
    host: Dict[str, Any],
    endpoint_base_url: str,
    probe: EndpointProbeResult,
    collected_at: str,
) -> None:
    """
    Write Host, Endpoint, Models, and relationships.
    """
    host_props = {
        "ts_name": host["ts_name"],
        "ts_ip": host["ts_ip"],
        "user_email": host.get("user_email"),
        "ts_version": host.get("ts_version"),
        "os": host.get("os"),
        "updated_at": collected_at,
    }

    endpoint_props = {
        "base_url": endpoint_base_url,
        "kind": "lmstudio-openai-compatible",
        "reachable": bool(probe.reachable),
        "status_code": probe.status_code,
        "error": probe.error,
        "last_checked_at": collected_at,
        "raw_response": json.dumps(probe.raw_response) if probe.raw_response is not None else None,
    }

    # Normalize model entries to at least an id
    norm_models: List[Dict[str, Any]] = []
    for m in probe.models:
        mid = m.get("id") or m.get("model") or m.get("name")
        if not mid:
            continue
        norm_models.append(
            {
                "id": str(mid),
                "object": m.get("object"),
                "owned_by": m.get("owned_by"),
                "created": m.get("created"),
                "raw": json.dumps(m),
            }
        )

    cypher = """
    MERGE (h:Host {ts_name: $host.ts_name})
      ON CREATE SET h.created_at = $collected_at
    SET h.ts_ip = $host.ts_ip,
        h.user_email = $host.user_email,
        h.ts_version = $host.ts_version,
        h.os = $host.os,
        h.updated_at = $collected_at

    MERGE (e:Endpoint {base_url: $endpoint.base_url})
      ON CREATE SET e.created_at = $collected_at
    SET e.kind = $endpoint.kind,
        e.reachable = $endpoint.reachable,
        e.status_code = $endpoint.status_code,
        e.error = $endpoint.error,
        e.last_checked_at = $endpoint.last_checked_at,
        e.raw_response = $endpoint.raw_response

    MERGE (h)-[r:HAS_ENDPOINT]->(e)
    SET r.last_seen_at = $collected_at

    WITH e, $models AS models, $collected_at AS collected_at
    FOREACH (m IN models |
      MERGE (mm:Model {id: m.id})
        ON CREATE SET mm.created_at = collected_at
      SET mm.object = m.object,
          mm.owned_by = m.owned_by,
          mm.created = m.created,
          mm.raw = m.raw,
          mm.updated_at = collected_at
      MERGE (e)-[s:SERVES_MODEL]->(mm)
      SET s.last_seen_at = collected_at
    )
    """

    with driver.session(database=neo4j_db) as session:
        session.run(
            cypher,
            host=host_props,
            endpoint=endpoint_props,
            models=norm_models,
            collected_at=collected_at,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="http", choices=["http", "https"])
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--api-root", default="/v1", help="OpenAI-compatible root, usually /v1")
    ap.add_argument("--timeout", type=float, default=3.5)

    ap.add_argument("--lmstudio-api-key", default=None, help="Optional bearer token")

    ap.add_argument("--neo4j-uri", default=None)
    ap.add_argument("--neo4j-user", default=None)
    ap.add_argument("--neo4j-password", default=None)
    ap.add_argument("--neo4j-db", default="neo4j")
    ap.add_argument("--no-schema", action="store_true", help="Skip constraint/index creation")

    args = ap.parse_args()

    neo4j_uri = args.neo4j_uri or _get_env("NEO4J_URI")
    neo4j_user = args.neo4j_user or _get_env("NEO4J_USER")
    neo4j_password = args.neo4j_password or _get_env("NEO4J_PASSWORD")
    if not (neo4j_uri and neo4j_user and neo4j_password):
        print("ERROR: Neo4j creds missing. Set NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD or pass --neo4j-* flags.", file=sys.stderr)
        return 2

    collected_at = utc_now_iso()

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        if not args.no_schema:
            ensure_schema(driver, args.neo4j_db)

        ok = 0
        fail = 0

        for host in RAW_DEVICES:
            base_url = build_base_url(args.scheme, host["ts_ip"], args.port, args.api_root)
            probe = fetch_models_openai_compatible(
                base_url=base_url,
                timeout_s=args.timeout,
                api_key=args.lmstudio_api_key,
            )

            upsert_inventory(
                driver=driver,
                neo4j_db=args.neo4j_db,
                host=host,
                endpoint_base_url=base_url,
                probe=probe,
                collected_at=collected_at,
            )

            if probe.reachable:
                ok += 1
                print(f"[OK]   {host['ts_name']:20} {host['ts_ip']:15} models={len(probe.models)} url={base_url}")
            else:
                fail += 1
                print(f"[FAIL] {host['ts_name']:20} {host['ts_ip']:15} err={probe.error} url={base_url}")

        print(f"\nDone. Reachable: {ok}, Unreachable: {fail}. Timestamp: {collected_at}")
        return 0

    finally:
        driver.close()


def _get_env(name: str) -> Optional[str]:
    import os
    v = os.environ.get(name)
    return v if v and v.strip() else None


if __name__ == "__main__":
    raise SystemExit(main())

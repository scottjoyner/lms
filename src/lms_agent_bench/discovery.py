"""discovery.py — shared fleet/Tailscale node discovery for lms.

Unifies the previously-duplicated Tailscale discovery implementations
(``fleet_discover._discover_tailscale``, ``fleet_orchestrator._discover_tailscale``,
``lms_endpoint_registry.tailscale_status``) into one canonical source. See
docs/LLD_UNIFIED_FLEET.md W-69 / W-70.

Callers that need the benchmark-oriented ``Node`` list should still use
``lms_agent_bench.fleet_discover`` (which remains the canonical benchmark
discovery). This module owns the raw Tailscale status parsing + a generic
node dict builder used by the endpoint registry and orchestrator.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List, Optional


def tailscale_status(timeout: int = 8) -> Optional[Dict[str, Any]]:
    """Return parsed ``tailscale status --json`` output, or ``None`` if unavailable."""
    try:
        proc = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def iter_tailscale_nodes(
    status: Dict[str, Any], include_self: bool = True
) -> List[Dict[str, Any]]:
    """Yield the Self node (optionally) + every Peer as raw status dicts."""
    nodes: List[Dict[str, Any]] = []
    self_node = status.get("Self")
    if include_self and isinstance(self_node, dict):
        nodes.append(self_node)
    peers = status.get("Peer")
    if isinstance(peers, dict):
        for peer in peers.values():
            if isinstance(peer, dict):
                nodes.append(peer)
    return nodes


def discover_tailscale_nodes(
    port: int = 1234, include_self: bool = True, timeout: int = 8
) -> List[Dict[str, Any]]:
    """Return a normalized list of tailscale nodes as
    ``{name, ip, online, os, base_url, native_url}`` dicts."""
    status = tailscale_status(timeout=timeout)
    if not status:
        return []
    out: List[Dict[str, Any]] = []
    for node in iter_tailscale_nodes(status, include_self=include_self):
        name = str(node.get("HostName") or node.get("DNSName") or "").strip().rstrip(".")
        ips = [str(ip).strip() for ip in (node.get("TailscaleIPs") or []) if str(ip).strip()]
        if not ips:
            continue
        ip = ips[0]
        out.append(
            {
                "name": name,
                "ip": ip,
                "online": bool(node.get("Online", False)),
                "os": node.get("OS", ""),
                "base_url": f"http://{ip}:{port}/v1",
                "native_url": f"http://{ip}:{port}/api/v1/models",
            }
        )
    return out

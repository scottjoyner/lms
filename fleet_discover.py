#!/usr/bin/env python3
"""Single source of truth for fleet node discovery.

Replaces the duplicated ``NODES`` dicts in bench_fleet.py /
bench_concurrency_probe.py and the fragile ``ALIAS`` hacks in the analysis
scripts. Nodes come from (in priority order):

  1. ``fleet.toml`` next to this file (explicit, stable),
  2. auto-discovery: ``tailscale status --json`` + an LM Studio
     ``/v1/models`` reachability probe (with retry/backoff).

Every node has a canonical short name. Hardware/alias collisions are resolved
here once, so the rest of the tooling never re-derives them.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
FLEET_TOML = HERE / "fleet.toml"
PORT = 1234


@dataclass
class Node:
    name: str
    url: str
    via: str = "explicit"          # explicit | tailscale | manual
    aliases: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _reachable(url: str, retries: int = 3, backoff: float = 1.5) -> bool:
    """Probe /v1/models with retry + exponential backoff.

    A single probe is what the old tools did; transient Tailscale blips or a
    node mid-wake used to be reported as permanently down. We now retry.
    """
    last: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlopen(url + "/v1/models", timeout=8)
            return True
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries:
                time.sleep(backoff * (2 ** (attempt - 1)))
    return False


def _discover_tailscale() -> list[Node]:
    """Parse ``tailscale status --json``; any host exposing :PORT/v1 is a node."""
    try:
        out = subprocess.check_output(
            ["tailscale", "status", "--json"], stderr=subprocess.DEVNULL, timeout=15
        )
    except Exception:  # noqa: BLE001
        return []
    try:
        data = json.loads(out)
    except Exception:
        return []
    nodes: list[Node] = []
    peers = data.get("Peer", {})
    for _key, info in peers.items():
        name = info.get("HostName")
        if not name:
            continue
        grp = info.get("TailscaleIPs", [])
        flat = grp if isinstance(grp, list) else [grp]
        ip = flat[0] if flat else None
        if not ip:
            continue
        url = f"http://{ip}:{PORT}/v1"
        nodes.append(Node(name=name, url=url, via="tailscale"))
    self_info = data.get("Self", {})
    self_name = self_info.get("HostName")
    if self_name:
        nodes.append(Node(name=self_name, url=f"http://127.0.0.1:{PORT}/v1", via="tailscale"))
    return nodes


def _load_toml() -> list[Node]:
    if not FLEET_TOML.exists():
        return []
    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            return []
    data = tomllib.loads(FLEET_TOML.read_text(encoding="utf-8"))
    nodes = []
    for n in data.get("nodes", []):
        nodes.append(Node(
            name=n["name"],
            url=n["url"],
            via="explicit",
            aliases=list(n.get("aliases", [])),
            notes=n.get("notes", ""),
        ))
    return nodes


def discover(retries: int = 3) -> list[Node]:
    """Return the canonical node list.

    Explicit (fleet.toml) entries win on name AND on IP — a tailscale peer
    whose IP already matches an explicit entry is dropped so we don't end up
    with both ``scotts-macbook-air`` (explicit) and ``Scott's MacBook Air``
    (tailscale) as separate nodes.
    """
    toml_nodes = _load_toml()
    by_name: dict[str, Node] = {}
    explicit_ips = set()
    for n in toml_nodes:
        by_name[n.name] = n
        explicit_ips.add(n.url.rsplit(":", 1)[0])
    for n in _discover_tailscale():
        ip = n.url.rsplit(":", 1)[0]
        if n.name in by_name or ip in explicit_ips:
            continue
        by_name[n.name] = n
    return list(by_name.values())


def live_nodes(nodes: list[Node], retries: int = 3) -> list[Node]:
    return [n for n in nodes if _reachable(n.url, retries=retries)]


def retry(call, retries: int = 3, backoff: float = 1.5, what: str = "op"):
    """Run ``call`` up to ``retries`` times with exponential backoff.

    Used by the benchmark stage so a transient LM Studio hiccup (model still
    loading, brief CPU stall) doesn't fail a whole node's run. Returns the
    call's result, or raises the last exception after exhausting retries.
    """
    last: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return call()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries:
                time.sleep(backoff * (2 ** (attempt - 1)))
    raise last if last else RuntimeError(f"{what} failed")


def all_aliases(nodes: list[Node]) -> dict[str, str]:
    """Map every alias (and the canonical name) -> canonical name.

    Centralizes the macbook-air <-> scotts-macbook-air resolution so the
    analysis scripts can drop their own opposite-direction ALIAS hacks.
    """
    m: dict[str, str] = {}
    for n in nodes:
        m[n.name] = n.name
        for a in n.aliases:
            m[a] = n.name
    return m


if __name__ == "__main__":
    ns = discover()
    for n in ns:
        print(f"{n.name:28s} {n.url:48s} [{n.via}]")
    print(f"\n{len(ns)} nodes configured.")

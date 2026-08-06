"""lms_agent_bench — reliable exact-loadout fleet qualification.

The package provides endpoint discovery, repeated-throughput measurement,
census-complete fleet operation, strict controller and remote readiness checks,
immutable model/runtime loadout identity, one-run throughput and Hermes
qualification, transactional canary lifecycle and soak testing, verified rollback,
cryptographically linked evidence, record-only prompt-prefix/KV metadata,
heterogeneous tailnet capability routing, and verifiable non-admitted run
artifacts. Live admission and KV restoration remain external.
"""

from __future__ import annotations

from .fleet_routing_serialization_policy import install_json_safe_routing_matrix

install_json_safe_routing_matrix()

__version__ = "0.33.0"

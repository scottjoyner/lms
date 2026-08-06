from __future__ import annotations

import datetime as _datetime
from typing import Any, Iterable, Mapping, Sequence

# datetime.UTC was added in Python 3.11. The project supports Python 3.10, so
# provide the equivalent module attribute only while loading the routing module.
if not hasattr(_datetime, "UTC"):
    setattr(_datetime, "UTC", _datetime.timezone.utc)

from . import fleet_routing_matrix as _implementation
from .fleet_routing_serialization_policy import _json_safe

_raw_builder = _implementation.build_routing_matrix


def build_routing_matrix(
    tailnet_status: Mapping[str, Any],
    *,
    role_policy: Any | None = None,
    benchmark_documents: Iterable[Any] = (),
) -> dict[str, Any]:
    return _json_safe(
        _raw_builder(
            tailnet_status,
            role_policy=role_policy,
            benchmark_documents=benchmark_documents,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    original = _implementation.build_routing_matrix
    _implementation.build_routing_matrix = build_routing_matrix
    try:
        return _implementation.main(argv)
    finally:
        _implementation.build_routing_matrix = original


__all__ = ["build_routing_matrix", "main"]

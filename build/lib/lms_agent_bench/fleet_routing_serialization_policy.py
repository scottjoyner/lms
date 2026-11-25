from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_INSTALLED = False


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def install_json_safe_routing_matrix() -> None:
    """Ensure the public builder always returns a JSON-serializable document."""

    global _INSTALLED
    if _INSTALLED:
        return
    from . import fleet_routing_matrix as module

    original = module.build_routing_matrix

    def build_routing_matrix(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return _json_safe(original(*args, **kwargs))

    module.build_routing_matrix = build_routing_matrix
    _INSTALLED = True

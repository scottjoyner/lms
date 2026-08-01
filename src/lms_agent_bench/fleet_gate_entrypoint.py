"""Installed release-gate entrypoint with selected-model inventory semantics."""
from __future__ import annotations

import tarfile
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lms_agent_bench import fleet_gate as _base


def verify_sweep_artifacts(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    observation_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    missing = sorted(_base.SWEEP_REQUIRED - set(members))
    if missing:
        raise ValueError("sweep bundle is missing: " + ", ".join(missing))
    execution = _base.read_member_json(
        archive, members, "benchmark/execution_manifest.json"
    )
    selection = _base.read_member_json(
        archive, members, "selected_loadout.json"
    )
    selected_inventory = _base.read_member_json(
        archive, members, "model_inventory.selected.json"
    )
    if execution.get("loopback_only") is not True:
        raise ValueError("execution manifest does not prove loopback isolation")
    if str(execution.get("plan_fingerprint")) != str(
        observation_summary["plan_fingerprint"]
    ):
        raise ValueError("execution manifest references a different plan")
    execution_fp = str(execution.get("execution_fingerprint") or "")
    if not _base.SHA256_RE.fullmatch(execution_fp):
        raise ValueError("execution fingerprint is invalid")

    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("selection contains no selected candidate")
    if not _base._boolean(selected.get("eligible")):
        raise ValueError("selected candidate is not eligible")
    if selected.get("hard_failures"):
        raise ValueError("selected candidate contains hard failures")
    candidate_id = str(selected.get("candidate_id") or "")
    executed_ids = {
        str(item) for item in execution.get("candidate_ids", [])
    }
    if not candidate_id or candidate_id not in executed_ids:
        raise ValueError("selected candidate was not executed")
    candidate = selected.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("selected candidate configuration is missing")
    selected_model = candidate.get("model")
    if not isinstance(selected_model, Mapping) or not selected_model.get("id"):
        raise ValueError("selected candidate model is missing")
    selected_model_id = str(selected_model["id"])

    gates = selected.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("selected candidate contains no gate results")
    failed_gates = sorted(
        gate
        for gate in _base.REQUIRED_SELECTION_GATES
        if not _base._boolean(gates.get(gate))
    )
    if failed_gates:
        raise ValueError(
            "selected candidate failed required gates: "
            + ", ".join(failed_gates)
        )
    admission = selection.get("admission")
    if not isinstance(admission, Mapping) or admission.get("admitted") is not False:
        raise ValueError("selection artifact must remain non-admitted")

    inventory_models = selected_inventory.get("models")
    if not isinstance(inventory_models, list) or not inventory_models:
        raise ValueError("selected model inventory contains no models")
    full_models: List[Mapping[str, Any]] = [
        item
        for item in inventory_models
        if isinstance(item, Mapping)
        and str(item.get("fingerprint_mode")) == "full"
    ]
    if len(full_models) != 1:
        raise ValueError(
            "selected model inventory must contain exactly one full-hash record"
        )
    full_model = full_models[0]
    if str(full_model.get("id")) != selected_model_id:
        raise ValueError(
            "full-hash model record does not match the selected candidate"
        )
    model_hash = str(
        full_model.get("content_sha256")
        or full_model.get("artifact_fingerprint")
        or ""
    ).lower()
    if not _base.SHA256_RE.fullmatch(model_hash):
        raise ValueError("selected model content SHA-256 is invalid")
    return {
        "execution_fingerprint": execution_fp,
        "selection_fingerprint": selection.get("selection_fingerprint"),
        "candidate_id": candidate_id,
        "model_id": selected_model_id,
        "model_content_sha256": model_hash,
    }


def evaluate_rollout(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    _base.verify_sweep_artifacts = verify_sweep_artifacts
    return _base.evaluate_rollout(*args, **kwargs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _base.verify_sweep_artifacts = verify_sweep_artifacts
    return _base.main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())

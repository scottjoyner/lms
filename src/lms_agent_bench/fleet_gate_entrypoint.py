"""Installed release gate with provenance, reliability, and archive verification."""
from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lms_agent_bench import fleet_gate as _base
from lms_agent_bench.fleet_loadout import canonical_hash
from lms_agent_bench.fleet_provenance import verify_source_control

_SOURCE_ARTIFACT = "source_control.json"
_REQUIRED_RELIABILITY_GATES = set(_base.REQUIRED_SELECTION_GATES) | {
    "measurement_reliability"
}
_RELIABILITY_METRICS = (
    "reliability_score",
    "valid_trials",
    "required_trials",
    "trial_attempts",
    "trial_retry_rate",
    "sample_completeness",
    "success_wilson_lower_95",
    "trial_tps_cv",
    "trial_ttft_cv",
    "tps_relative_mad",
    "ttft_relative_mad",
    "tps_p10",
    "tps_p90",
    "ttft_p90",
    "tps_median_ci95_low",
    "tps_median_ci95_high",
    "ttft_median_ci95_low",
    "ttft_median_ci95_high",
    "warmup_cv",
    "warmup_stable",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_slug(value: str) -> str:
    slug = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in value
    ).strip("-")
    return slug or "candidate"


def _int_value(value: Any, label: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is missing or invalid") from exc


def _normalized(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value).strip().lower()


def verify_reliability_artifact(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    candidate_id: str,
    selected_model_id: str,
    selected_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    path = (
        f"benchmark/{_safe_slug(candidate_id)}/suite/reliability.json"
    )
    report = _base.read_member_json(archive, members, path)
    if report.get("schema_version") != "reliable_benchmark.v1":
        raise ValueError("unsupported benchmark reliability schema")
    if report.get("artifact_type") != "benchmark_reliability":
        raise ValueError("invalid benchmark reliability artifact type")
    if report.get("passed") is not True:
        raise ValueError("benchmark reliability report did not pass")
    admission = report.get("admission")
    if not isinstance(admission, Mapping) or admission.get("admitted") is not False:
        raise ValueError("benchmark reliability report must remain non-admitted")

    fingerprint = str(report.get("reliability_fingerprint") or "").lower()
    if not _base.SHA256_RE.fullmatch(fingerprint):
        raise ValueError("benchmark reliability fingerprint is invalid")
    core = {
        key: value
        for key, value in report.items()
        if key not in {"created_at_utc", "reliability_fingerprint"}
    }
    if fingerprint != canonical_hash(core):
        raise ValueError("benchmark reliability fingerprint mismatch")
    if str(selected_metrics.get("reliability_fingerprint") or "").lower() != fingerprint:
        raise ValueError(
            "selected metrics reference a different reliability report"
        )
    if not _base._boolean(selected_metrics.get("reliability_pass")):
        raise ValueError("selected metrics do not record reliability pass")

    valid_trials = _int_value(report.get("valid_trials"), "valid_trials")
    requested_trials = _int_value(
        report.get("requested_trials"), "requested_trials"
    )
    trial_attempts = _int_value(
        report.get("trial_attempts"), "trial_attempts"
    )
    if valid_trials < 3 or requested_trials < 3:
        raise ValueError("benchmark reliability requires at least three trials")
    if valid_trials > requested_trials or trial_attempts < valid_trials:
        raise ValueError("benchmark reliability trial counts are inconsistent")

    summaries = report.get("summaries")
    if not isinstance(summaries, list) or len(summaries) != 1:
        raise ValueError(
            "candidate reliability report must contain exactly one summary"
        )
    summary = summaries[0]
    if not isinstance(summary, Mapping):
        raise ValueError("benchmark reliability summary must be an object")
    if summary.get("reliability_pass") is not True:
        raise ValueError("benchmark reliability summary did not pass")
    if str(summary.get("model_key") or "") != selected_model_id:
        raise ValueError(
            "benchmark reliability model does not match selected model"
        )
    if _int_value(summary.get("valid_trials"), "summary valid_trials") != valid_trials:
        raise ValueError("benchmark reliability valid-trial count mismatch")
    if _int_value(summary.get("trial_attempts"), "summary trial_attempts") != trial_attempts:
        raise ValueError("benchmark reliability attempt count mismatch")

    for field in _RELIABILITY_METRICS:
        if field not in summary or field not in selected_metrics:
            raise ValueError(f"benchmark reliability metric is missing: {field}")
        if _normalized(summary[field]) != _normalized(selected_metrics[field]):
            raise ValueError(
                f"selected reliability metric does not match report: {field}"
            )
    failures = summary.get("reliability_failures")
    if failures not in (None, [], "", "[]"):
        raise ValueError("benchmark reliability summary contains failures")
    return {
        "artifact": path,
        "reliability_fingerprint": fingerprint,
        "model_id": selected_model_id,
        "valid_trials": valid_trials,
        "requested_trials": requested_trials,
        "trial_attempts": trial_attempts,
        "reliability_score": summary.get("reliability_score"),
        "sample_completeness": summary.get("sample_completeness"),
        "trial_tps_cv": summary.get("trial_tps_cv"),
        "trial_ttft_cv": summary.get("trial_ttft_cv"),
    }


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
        for gate in _REQUIRED_RELIABILITY_GATES
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
    selected_metrics = selected.get("metrics")
    if not isinstance(selected_metrics, Mapping):
        raise ValueError("selected candidate contains no benchmark metrics")
    reliability = verify_reliability_artifact(
        archive,
        members,
        candidate_id,
        selected_model_id,
        selected_metrics,
    )

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
        "reliability": reliability,
    }


def _verify_bundle_fingerprint(
    manifest: Mapping[str, Any], expected_run_id: str
) -> None:
    if str(manifest.get("run_id")) != expected_run_id:
        raise ValueError("bundle run_id does not match rollout results")
    source_fp = str(manifest.get("source_fingerprint") or "")
    if not _base.SHA256_RE.fullmatch(source_fp):
        raise ValueError("bundle source fingerprint is missing or invalid")
    core = {
        "schema_version": manifest.get("schema_version"),
        "node_id": manifest.get("node_id"),
        "run_id": manifest.get("run_id"),
        "remote_exit_code": manifest.get("remote_exit_code"),
        "source_fingerprint": manifest.get("source_fingerprint"),
        "files": manifest.get("files"),
    }
    if str(manifest.get("bundle_fingerprint") or "") != canonical_hash(core):
        raise ValueError("bundle fingerprint mismatch")


def verify_archive(
    path: Path,
    expected_node_id: str,
    expected_run_id: str,
    mode: str,
    expected_sha256: str,
    expected_size_bytes: int,
) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"collected archive does not exist: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size_bytes:
        raise ValueError("collected archive size does not match rollout results")
    actual_sha256 = _hash_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("collected archive SHA-256 does not match rollout results")

    with tarfile.open(path, mode="r:gz") as archive:
        members = _base.member_map(archive)
        manifest = _base.verify_manifest(archive, members, expected_node_id)
        _verify_bundle_fingerprint(manifest, expected_run_id)
        source_artifact = _base.read_member_json(
            archive, members, _SOURCE_ARTIFACT
        )
        source_summary = verify_source_control(
            source_artifact,
            expected_node_id=expected_node_id,
            expected_run_id=expected_run_id,
        )
        if manifest.get("source_fingerprint") != source_summary["source_fingerprint"]:
            raise ValueError(
                "bundle source fingerprint does not match source provenance"
            )
        observation = _base.verify_observation_artifacts(archive, members)
        sweep: Optional[Dict[str, Any]] = None
        if mode == "sweep":
            sweep = verify_sweep_artifacts(
                archive, members, observation
            )
    return {
        "archive": str(path),
        "archive_size_bytes": actual_size,
        "archive_sha256": actual_sha256,
        "bundle_fingerprint": manifest.get("bundle_fingerprint"),
        "manifest_file_count": len(manifest.get("files", [])),
        "source": source_summary,
        "observation": observation,
        "sweep": sweep,
    }


def evaluate_rollout(
    results_path: Path,
    required_nodes: Sequence[str],
    mode: str,
) -> Dict[str, Any]:
    payload = _base.load_json(results_path)
    raw_results = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(raw_results, list):
        raise ValueError("rollout results must contain a results array")
    run_id = str(payload.get("run_id") or "")
    if not run_id:
        raise ValueError("rollout results require a run_id")
    typed_results = [
        item for item in raw_results if isinstance(item, Mapping)
    ]
    by_node, duplicates = _base.unique_results(typed_results)
    node_ids = list(required_nodes) or sorted(by_node)
    if not node_ids:
        raise ValueError("release gate requires at least one node")

    node_reports: List[Dict[str, Any]] = []
    for node_id in node_ids:
        errors: List[str] = []
        archive_summary: Optional[Dict[str, Any]] = None
        result = by_node.get(node_id)
        if result is None:
            errors.append("node is missing from rollout results")
        else:
            try:
                returncode = int(result.get("returncode"))
            except (TypeError, ValueError):
                returncode = -1
            if returncode != 0:
                errors.append(f"remote rollout returned {returncode}")
            if result.get("timed_out") is True:
                errors.append("remote rollout timed out")
            try:
                scp_returncode = int(result.get("scp_returncode"))
            except (TypeError, ValueError):
                scp_returncode = -1
            if scp_returncode != 0:
                errors.append(
                    f"artifact collection returned {scp_returncode}"
                )
            if result.get("scp_timed_out") is True:
                errors.append("artifact collection timed out")

            artifact_value = str(result.get("collected_artifact") or "")
            archive_sha256 = str(
                result.get("collected_artifact_sha256") or ""
            ).lower()
            try:
                archive_size = int(result.get("collected_artifact_size_bytes"))
            except (TypeError, ValueError):
                archive_size = -1
            if not artifact_value:
                errors.append("no collected artifact was recorded")
            elif not _base.SHA256_RE.fullmatch(archive_sha256):
                errors.append("collected archive SHA-256 is missing or invalid")
            elif archive_size < 0:
                errors.append("collected archive size is missing or invalid")
            else:
                archive_path = _base.resolve_archive_path(
                    artifact_value, results_path
                )
                try:
                    archive_summary = verify_archive(
                        archive_path,
                        expected_node_id=node_id,
                        expected_run_id=run_id,
                        mode=mode,
                        expected_sha256=archive_sha256,
                        expected_size_bytes=archive_size,
                    )
                except (OSError, ValueError, tarfile.TarError) as exc:
                    errors.append(str(exc))
        node_reports.append(
            {
                "node_id": node_id,
                "passed": not errors,
                "errors": errors,
                "archive": archive_summary,
            }
        )

    if duplicates:
        node_reports.append(
            {
                "node_id": "<rollout-results>",
                "passed": False,
                "errors": [
                    "duplicate node results: " + ", ".join(duplicates)
                ],
                "archive": None,
            }
        )
    passed = all(item["passed"] for item in node_reports)
    report_core = {
        "mode": mode,
        "rollout_run_id": run_id,
        "required_node_ids": node_ids,
        "passed": passed,
        "nodes": node_reports,
        "next_stage": (
            "candidate_review" if mode == "observe" else "profile_import_review"
        ),
        "admission": {"admitted": False},
    }
    return {
        "schema_version": _base.SCHEMA_VERSION,
        "artifact_type": "fleet_release_gate",
        "created_at_utc": _base.utc_now_iso(),
        **report_core,
        "gate_fingerprint": canonical_hash(report_core),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    _base.OBSERVE_REQUIRED = set(_base.OBSERVE_REQUIRED) | {_SOURCE_ARTIFACT}
    _base.SWEEP_REQUIRED = set(_base.SWEEP_REQUIRED) | {_SOURCE_ARTIFACT}
    _base.verify_sweep_artifacts = verify_sweep_artifacts
    _base.evaluate_rollout = evaluate_rollout
    return _base.main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())

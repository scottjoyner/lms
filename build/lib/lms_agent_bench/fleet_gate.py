#!/usr/bin/env python3
"""Verify collected fleet rollout archives before the next release stage.

This command never deploys or admits a runtime. It verifies SSH results, archive
integrity, bundle hashes, required artifacts, and sweep eligibility so operators
have one deterministic gate between observation, execution, and profile import.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from lms_agent_bench.fleet_loadout import canonical_hash, utc_now_iso, write_json

SCHEMA_VERSION = "fleet_release_gate.v1"
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
OBSERVE_REQUIRED = {
    "machine_observation.json",
    "model_inventory.json",
    "benchmark_plan.json",
}
SWEEP_REQUIRED = OBSERVE_REQUIRED | {
    "benchmark/execution_manifest.json",
    "selected_loadout.json",
    "model_inventory.selected.json",
}
REQUIRED_SELECTION_GATES = {
    "completion",
    "streaming",
    "concurrency",
    "memory_headroom",
    "sustained_stability",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_member_name(name: str) -> str:
    normalized = name
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member path: {name!r}")
    return path.as_posix()


def member_map(archive: tarfile.TarFile) -> Dict[str, tarfile.TarInfo]:
    members: Dict[str, tarfile.TarInfo] = {}
    for info in archive.getmembers():
        if not info.isfile():
            continue
        name = normalize_member_name(info.name)
        if name in members:
            raise ValueError(f"duplicate archive member: {name}")
        members[name] = info
    return members


def read_member_bytes(
    archive: tarfile.TarFile, info: tarfile.TarInfo
) -> bytes:
    handle = archive.extractfile(info)
    if handle is None:
        raise ValueError(f"unable to read archive member: {info.name}")
    return handle.read()


def read_member_json(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    name: str,
) -> Mapping[str, Any]:
    if name not in members:
        raise ValueError(f"archive is missing required JSON artifact: {name}")
    try:
        value = json.loads(read_member_bytes(archive, members[name]))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {name}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact must be an object: {name}")
    return value


def hash_member(archive: tarfile.TarFile, info: tarfile.TarInfo) -> str:
    handle = archive.extractfile(info)
    if handle is None:
        raise ValueError(f"unable to hash archive member: {info.name}")
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(8 * 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def verify_manifest(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    expected_node_id: str,
) -> Mapping[str, Any]:
    manifest = read_member_json(archive, members, "bundle_manifest.json")
    if manifest.get("schema_version") != "fleet_artifact_bundle.v1":
        raise ValueError("unsupported bundle manifest schema")
    if str(manifest.get("node_id")) != expected_node_id:
        raise ValueError("bundle node_id does not match rollout result")
    try:
        remote_exit_code = int(manifest.get("remote_exit_code"))
    except (TypeError, ValueError) as exc:
        raise ValueError("bundle remote_exit_code is missing or invalid") from exc
    if remote_exit_code != 0:
        raise ValueError(
            f"bundle records failed remote execution: {remote_exit_code}"
        )
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("bundle manifest contains no file entries")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("bundle file entry must be an object")
        name = normalize_member_name(str(entry.get("path") or ""))
        if name in seen:
            raise ValueError(f"duplicate bundle manifest path: {name}")
        seen.add(name)
        if name == "bundle_manifest.json":
            raise ValueError("bundle manifest must not hash itself")
        if name not in members:
            raise ValueError(f"bundle member is missing: {name}")
        info = members[name]
        try:
            expected_size = int(entry.get("size_bytes"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid size for bundle member: {name}") from exc
        if info.size != expected_size:
            raise ValueError(f"bundle member size mismatch: {name}")
        expected_hash = str(entry.get("sha256") or "").lower()
        if not SHA256_RE.fullmatch(expected_hash):
            raise ValueError(f"invalid SHA-256 for bundle member: {name}")
        if hash_member(archive, info) != expected_hash:
            raise ValueError(f"bundle member hash mismatch: {name}")
    unlisted = sorted(
        set(members) - seen - {"bundle_manifest.json"}
    )
    if unlisted:
        raise ValueError(
            "archive contains files not listed in the bundle manifest: "
            + ", ".join(unlisted)
        )
    return manifest


def verify_observation_artifacts(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
) -> Dict[str, Any]:
    missing = sorted(OBSERVE_REQUIRED - set(members))
    if missing:
        raise ValueError(
            "observation bundle is missing: " + ", ".join(missing)
        )
    observation = read_member_json(
        archive, members, "machine_observation.json"
    )
    plan = read_member_json(archive, members, "benchmark_plan.json")
    inventory = read_member_json(
        archive, members, "model_inventory.json"
    )
    observation_fp = str(observation.get("observation_fingerprint") or "")
    plan_fp = str(plan.get("plan_fingerprint") or "")
    if not SHA256_RE.fullmatch(observation_fp):
        raise ValueError("machine observation fingerprint is invalid")
    if str(plan.get("observation_fingerprint")) != observation_fp:
        raise ValueError("benchmark plan references a different observation")
    if not SHA256_RE.fullmatch(plan_fp):
        raise ValueError("benchmark plan fingerprint is invalid")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("benchmark plan contains no candidates")
    models = inventory.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("model inventory contains no models")
    return {
        "observation_fingerprint": observation_fp,
        "plan_fingerprint": plan_fp,
        "candidate_count": len(candidates),
        "model_count": len(models),
    }


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "pass",
        "passed",
    }


def verify_sweep_artifacts(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    observation_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    missing = sorted(SWEEP_REQUIRED - set(members))
    if missing:
        raise ValueError("sweep bundle is missing: " + ", ".join(missing))
    execution = read_member_json(
        archive, members, "benchmark/execution_manifest.json"
    )
    selection = read_member_json(
        archive, members, "selected_loadout.json"
    )
    selected_inventory = read_member_json(
        archive, members, "model_inventory.selected.json"
    )
    if execution.get("loopback_only") is not True:
        raise ValueError("execution manifest does not prove loopback isolation")
    if str(execution.get("plan_fingerprint")) != str(
        observation_summary["plan_fingerprint"]
    ):
        raise ValueError("execution manifest references a different plan")
    execution_fp = str(execution.get("execution_fingerprint") or "")
    if not SHA256_RE.fullmatch(execution_fp):
        raise ValueError("execution fingerprint is invalid")
    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("selection contains no selected candidate")
    if not _boolean(selected.get("eligible")):
        raise ValueError("selected candidate is not eligible")
    if selected.get("hard_failures"):
        raise ValueError("selected candidate contains hard failures")
    candidate_id = str(selected.get("candidate_id") or "")
    executed_ids = {
        str(item) for item in execution.get("candidate_ids", [])
    }
    if not candidate_id or candidate_id not in executed_ids:
        raise ValueError("selected candidate was not executed")
    gates = selected.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("selected candidate contains no gate results")
    failed_gates = sorted(
        gate
        for gate in REQUIRED_SELECTION_GATES
        if not _boolean(gates.get(gate))
    )
    if failed_gates:
        raise ValueError(
            "selected candidate failed required gates: "
            + ", ".join(failed_gates)
        )
    if selection.get("admission", {}).get("admitted") is not False:
        raise ValueError("selection artifact must remain non-admitted")
    selected_models = selected_inventory.get("models")
    if not isinstance(selected_models, list) or len(selected_models) != 1:
        raise ValueError(
            "selected model inventory must contain exactly one model"
        )
    selected_model = selected_models[0]
    if not isinstance(selected_model, Mapping):
        raise ValueError("selected model record must be an object")
    if str(selected_model.get("fingerprint_mode")) != "full":
        raise ValueError("selected model fingerprint is not full")
    model_hash = str(
        selected_model.get("content_sha256")
        or selected_model.get("artifact_fingerprint")
        or ""
    ).lower()
    if not SHA256_RE.fullmatch(model_hash):
        raise ValueError("selected model content SHA-256 is invalid")
    return {
        "execution_fingerprint": execution_fp,
        "selection_fingerprint": selection.get("selection_fingerprint"),
        "candidate_id": candidate_id,
        "model_id": selected_model.get("id"),
        "model_content_sha256": model_hash,
    }


def verify_archive(
    path: Path, expected_node_id: str, mode: str
) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"collected archive does not exist: {path}")
    with tarfile.open(path, mode="r:gz") as archive:
        members = member_map(archive)
        manifest = verify_manifest(archive, members, expected_node_id)
        observation = verify_observation_artifacts(archive, members)
        sweep: Optional[Dict[str, Any]] = None
        if mode == "sweep":
            sweep = verify_sweep_artifacts(
                archive, members, observation
            )
    return {
        "archive": str(path),
        "archive_size_bytes": path.stat().st_size,
        "manifest_file_count": len(manifest.get("files", [])),
        "observation": observation,
        "sweep": sweep,
    }


def resolve_archive_path(value: str, results_path: Path) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw
    candidates = [Path.cwd() / raw, results_path.parent / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def unique_results(
    values: Iterable[Mapping[str, Any]],
) -> Tuple[Dict[str, Mapping[str, Any]], List[str]]:
    by_node: Dict[str, Mapping[str, Any]] = {}
    duplicates: List[str] = []
    for item in values:
        node_id = str(item.get("node_id") or "")
        if not node_id:
            continue
        if node_id in by_node:
            duplicates.append(node_id)
        by_node[node_id] = item
    return by_node, sorted(set(duplicates))


def evaluate_rollout(
    results_path: Path,
    required_nodes: Sequence[str],
    mode: str,
) -> Dict[str, Any]:
    payload = load_json(results_path)
    raw_results = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(raw_results, list):
        raise ValueError("rollout results must contain a results array")
    typed_results = [
        item for item in raw_results if isinstance(item, Mapping)
    ]
    by_node, duplicates = unique_results(typed_results)
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
            if "scp_returncode" in result:
                try:
                    scp_returncode = int(result.get("scp_returncode"))
                except (TypeError, ValueError):
                    scp_returncode = -1
                if scp_returncode != 0:
                    errors.append(
                        f"artifact collection returned {scp_returncode}"
                    )
            artifact_value = str(result.get("collected_artifact") or "")
            if not artifact_value:
                errors.append("no collected artifact was recorded")
            else:
                archive_path = resolve_archive_path(
                    artifact_value, results_path
                )
                try:
                    archive_summary = verify_archive(
                        archive_path, node_id, mode
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
        "rollout_run_id": payload.get("run_id"),
        "required_node_ids": node_ids,
        "passed": passed,
        "nodes": node_reports,
        "next_stage": (
            "candidate_review" if mode == "observe" else "profile_import_review"
        ),
        "admission": {"admitted": False},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "fleet_release_gate",
        "created_at_utc": utc_now_iso(),
        **report_core,
        "gate_fingerprint": canonical_hash(report_core),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify collected fleet rollout evidence"
    )
    parser.add_argument("--rollout-results", required=True)
    parser.add_argument(
        "--mode", choices=("observe", "sweep"), default="sweep"
    )
    parser.add_argument("--required-node", action="append", default=[])
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_rollout(
            Path(args.rollout_results), args.required_node, args.mode
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report_core = {
            "mode": args.mode,
            "rollout_run_id": None,
            "required_node_ids": list(args.required_node),
            "passed": False,
            "nodes": [],
            "next_stage": None,
            "error": str(exc),
            "admission": {"admitted": False},
        }
        report = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "fleet_release_gate",
            "created_at_utc": utc_now_iso(),
            **report_core,
            "gate_fingerprint": canonical_hash(report_core),
        }
    write_json(args.out, report)
    print(f"wrote {args.out}")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

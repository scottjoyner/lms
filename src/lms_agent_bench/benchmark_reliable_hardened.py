"""Harden reliable benchmark artifacts and resume against local tampering."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from lms_agent_bench import benchmark_reliable as _base

_ORIGINAL_EXECUTE_TRIAL_ATTEMPT = _base.execute_trial_attempt
_MANIFEST_FINGERPRINT_FIELD = "trial_manifest_fingerprint"


def _safe_relative(value: str, label: str) -> Path:
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe {label}: {value!r}")
    return relative


def _raw_run_dir(output_dir: Path) -> Path:
    config = _base.read_json(output_dir / "config.json")
    run_id = config.get("run_id")
    sidecar_value = str(config.get("sidecar_dir") or "")
    if run_id in (None, "") or not sidecar_value:
        raise ValueError("raw benchmark config lacks run_id or sidecar_dir")
    sidecar_dir = Path(sidecar_value).resolve()
    run_dir = (sidecar_dir / f"run_{run_id}").resolve()
    try:
        run_dir.relative_to(sidecar_dir)
    except ValueError as exc:
        raise ValueError("raw benchmark run directory escapes sidecar root") from exc
    if not run_dir.is_dir():
        raise ValueError(f"raw benchmark sidecar run is missing: {run_dir.name}")
    return run_dir


def sample_output_artifacts(
    output_dir: Path, rows: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, Any]]:
    if rows is None:
        rows = _base.read_csv(output_dir / "run_results.csv")
    run_dir = _raw_run_dir(output_dir)
    seen: set[str] = set()
    artifacts: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("phase") != "run" or not _base.bool_value(row.get("ok")):
            continue
        output_file = str(row.get("output_file") or "")
        relative = _safe_relative(output_file, "sample output path")
        normalized = relative.as_posix()
        if normalized in seen:
            raise ValueError(f"duplicate successful sample output path: {normalized}")
        seen.add(normalized)
        path = (run_dir / relative).resolve()
        try:
            path.relative_to(run_dir)
        except ValueError as exc:
            raise ValueError(
                f"sample output escapes raw run directory: {normalized}"
            ) from exc
        if not path.is_file():
            raise ValueError(f"successful sample output is missing: {normalized}")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"successful sample output is empty: {normalized}")
        artifacts.append(
            {
                "output_file": normalized,
                "size_bytes": size,
                "sha256": _base.file_sha256(path),
            }
        )
    return sorted(artifacts, key=lambda item: item["output_file"])


def validate_trial_artifacts(
    output_dir: Path,
    expected_keys: set[Tuple[str, str, str, int]],
) -> Tuple[bool, List[str], List[Dict[str, str]]]:
    valid, errors, rows = _base.validate_trial_artifacts(
        output_dir, expected_keys
    )
    if errors:
        return False, errors, rows
    try:
        sample_output_artifacts(output_dir, rows)
    except (OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
        errors.append(str(exc))
    return valid and not errors, errors, rows


def _manifest_core(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key != _MANIFEST_FINGERPRINT_FIELD
    }


def _seal_manifest(path: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    sealed = dict(manifest)
    sealed[_MANIFEST_FINGERPRINT_FIELD] = _base.canonical_hash(
        _manifest_core(sealed)
    )
    _base.write_json(path, sealed)
    return sealed


def execute_trial_attempt(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    result = _ORIGINAL_EXECUTE_TRIAL_ATTEMPT(*args, **kwargs)
    output_dir = Path(str(result["output_dir"]))
    attempt_dir = output_dir.parent
    manifest_path = attempt_dir / "trial_manifest.json"
    manifest = dict(result["manifest"])
    if manifest.get("valid") is True:
        manifest["sample_outputs"] = sample_output_artifacts(
            output_dir, result.get("rows")
        )
    else:
        manifest["sample_outputs"] = []
    sealed = _seal_manifest(manifest_path, manifest)
    result["manifest"] = sealed
    return result


def _verify_file_record(path: Path, record: Mapping[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"resumed {label} is missing")
    try:
        size = int(record.get("size_bytes"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"resumed {label} size is invalid") from exc
    if path.stat().st_size != size:
        raise ValueError(f"resumed {label} size mismatch")
    expected = str(record.get("sha256") or "")
    if _base.file_sha256(path) != expected:
        raise ValueError(f"resumed {label} hash mismatch")


def _verify_resumable_manifest(
    manifest_path: Path, manifest: Mapping[str, Any], input_fp: str
) -> List[Dict[str, str]]:
    fingerprint = str(manifest.get(_MANIFEST_FINGERPRINT_FIELD) or "")
    if fingerprint != _base.canonical_hash(_manifest_core(manifest)):
        raise ValueError("resumed trial manifest fingerprint mismatch")
    if manifest.get("input_fingerprint") != input_fp:
        raise ValueError("resumed trial input fingerprint mismatch")
    if manifest.get("valid") is not True:
        raise ValueError("resumed trial was not valid")
    if int(manifest.get("returncode")) != 0 or manifest.get("timed_out") is True:
        raise ValueError("resumed trial did not complete successfully")
    postflight = manifest.get("postflight")
    if not isinstance(postflight, Mapping) or not postflight:
        raise ValueError("resumed trial has no postflight evidence")
    if any(
        not isinstance(item, Mapping) or item.get("ok") is not True
        for item in postflight.values()
    ):
        raise ValueError("resumed trial postflight did not pass")

    attempt_dir = manifest_path.parent
    output_dir = attempt_dir / "output"
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("resumed trial artifact hashes are missing")
    for name in (
        "config.json",
        "run_results.csv",
        "run_summary.csv",
        "task_summary.csv",
    ):
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"resumed trial lacks artifact record: {name}")
        _verify_file_record(output_dir / name, record, name)

    log_path = attempt_dir / "runner.log"
    if _base.file_sha256(log_path) != str(
        manifest.get("runner_log_sha256") or ""
    ):
        raise ValueError("resumed runner log hash mismatch")
    rows = _base.read_csv(output_dir / "run_results.csv")
    actual_outputs = sample_output_artifacts(output_dir, rows)
    recorded_outputs = manifest.get("sample_outputs")
    if actual_outputs != recorded_outputs:
        raise ValueError("resumed sample output evidence mismatch")
    return rows


def existing_valid_trial(
    trial_root: Path, input_fp: str
) -> Optional[Dict[str, Any]]:
    for manifest_path in sorted(trial_root.glob("attempt_*/trial_manifest.json")):
        try:
            manifest = _base.read_json(manifest_path)
            rows = _verify_resumable_manifest(
                manifest_path, manifest, input_fp
            )
        except (
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
            csv.Error,
        ):
            continue
        output_dir = manifest_path.parent / "output"
        return {
            "trial_index": int(manifest.get("trial_index")),
            "attempt_index": int(manifest.get("attempt_index")),
            "manifest": manifest,
            "rows": rows,
            "output_dir": str(output_dir),
            "resumed": True,
        }
    return None


def main(argv: Optional[List[str]] = None) -> int:
    _base.validate_trial_artifacts = validate_trial_artifacts
    _base.execute_trial_attempt = execute_trial_attempt
    _base.existing_valid_trial = existing_valid_trial
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Bind throughput evidence to an exact loadout and qualify the full evidence chain."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import urlparse

from lms_agent_bench.hermes_agent_common import (
    canonical_hash,
    load_json,
    normalize_sha256,
    utc_now_iso,
    write_json,
)
from lms_agent_bench.model_loadout import validate_manifest

THROUGHPUT_SCHEMA_VERSION = "loadout_throughput_evidence.v1"
QUALIFICATION_SCHEMA_VERSION = "loadout_qualification.v1"
BASE_HERMES_SUITE_ID = "hermes_agent_intelligence.v1"
CONTEXT_HERMES_SUITE_ID = "hermes_agent_context_pressure.v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _mapping(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _non_admitted(value: Any, label: str) -> None:
    admission = _mapping(value, f"{label}.admission")
    if admission.get("admitted") is not False:
        raise ValueError(f"{label} must remain non-admitted")


def _loopback_url(value: Any, label: str) -> str:
    text = str(value or "").strip()
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{label} must use http or https")
    if (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"{label} must be loopback-local")
    return text


def _reliable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _identity(loadout: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "node_id": loadout["node_id"],
        "candidate_id": loadout["candidate_id"],
        "model_id": loadout["model"]["id"],
        "model_content_sha256": loadout["model"]["content_sha256"],
        "loadout_fingerprint": loadout["loadout_fingerprint"],
        "architecture_kind": loadout["architecture"]["kind"],
        "configured_context_tokens": loadout["context"]["configured_tokens"],
        "parallel_slots": loadout["concurrency"]["parallel_slots"],
    }


def verify_reliability_report(
    report: Mapping[str, Any], loadout: Mapping[str, Any]
) -> Dict[str, Any]:
    if report.get("schema_version") != "reliable_benchmark.v1":
        raise ValueError("unsupported reliability report schema")
    if report.get("artifact_type") != "benchmark_reliability":
        raise ValueError("invalid reliability artifact type")
    if report.get("passed") is not True:
        raise ValueError("reliability report did not pass")
    _non_admitted(report.get("admission"), "reliability report")
    fingerprint = normalize_sha256(str(report.get("reliability_fingerprint") or ""))
    core = {
        key: value
        for key, value in report.items()
        if key not in {"created_at_utc", "reliability_fingerprint"}
    }
    if fingerprint != _reliable_hash(core):
        raise ValueError("reliability report fingerprint mismatch")
    if int(report.get("valid_trials") or 0) < 3:
        raise ValueError("reliability report requires at least three valid trials")
    summaries = report.get("summaries")
    if not isinstance(summaries, list) or len(summaries) != 1:
        raise ValueError("loadout throughput evidence requires exactly one summary")
    summary = _mapping(summaries[0], "reliability summary")
    if summary.get("reliability_pass") is not True:
        raise ValueError("reliability summary did not pass")
    if summary.get("reliability_failures") not in (None, [], "", "[]"):
        raise ValueError("reliability summary contains failures")
    if str(summary.get("host_name") or "") != str(loadout["node_id"]):
        raise ValueError("reliability node does not match loadout")
    if str(summary.get("model_key") or "") != str(loadout["model"]["id"]):
        raise ValueError("reliability model does not match loadout")
    endpoint = _loopback_url(summary.get("base_url"), "reliability endpoint")
    return {
        "reliability_fingerprint": fingerprint,
        "summary": summary,
        "endpoint": endpoint,
    }


def build_throughput_evidence(
    loadout_raw: Mapping[str, Any],
    reliability_report: Mapping[str, Any],
    *,
    reliability_artifact_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    loadout = validate_manifest(
        loadout_raw,
        require_fingerprint=bool(loadout_raw.get("loadout_fingerprint")),
    )
    verified = verify_reliability_report(reliability_report, loadout)
    artifact_hash = (
        normalize_sha256(reliability_artifact_sha256)
        if reliability_artifact_sha256
        else None
    )
    core = {
        "schema_version": THROUGHPUT_SCHEMA_VERSION,
        "artifact_type": "exact_loadout_throughput_evidence",
        "identity": _identity(loadout),
        "loadout": loadout,
        "reliability_fingerprint": verified["reliability_fingerprint"],
        "reliability_artifact_sha256": artifact_hash,
        "reliability_summary": verified["summary"],
        "endpoint": verified["endpoint"],
        "loopback_only": True,
        "throughput_qualified": True,
        "admission": {"admitted": False},
    }
    return {
        **core,
        "created_at_utc": utc_now_iso(),
        "throughput_evidence_fingerprint": canonical_hash(core),
    }


def verify_throughput_evidence(
    evidence: Mapping[str, Any], expected_loadout: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    if evidence.get("schema_version") != THROUGHPUT_SCHEMA_VERSION:
        raise ValueError("unsupported throughput evidence schema")
    if evidence.get("artifact_type") != "exact_loadout_throughput_evidence":
        raise ValueError("invalid throughput evidence artifact type")
    if evidence.get("throughput_qualified") is not True:
        raise ValueError("throughput evidence is not qualified")
    if evidence.get("loopback_only") is not True:
        raise ValueError("throughput evidence is not loopback-only")
    _loopback_url(evidence.get("endpoint"), "throughput endpoint")
    _non_admitted(evidence.get("admission"), "throughput evidence")
    fingerprint = normalize_sha256(
        str(evidence.get("throughput_evidence_fingerprint") or "")
    )
    core = {
        key: value
        for key, value in evidence.items()
        if key not in {"created_at_utc", "throughput_evidence_fingerprint"}
    }
    if fingerprint != canonical_hash(core):
        raise ValueError("throughput evidence fingerprint mismatch")
    loadout_raw = _mapping(evidence.get("loadout"), "throughput loadout")
    loadout = validate_manifest(loadout_raw, require_fingerprint=True)
    identity = _mapping(evidence.get("identity"), "throughput identity")
    if identity != _identity(loadout):
        raise ValueError("throughput identity does not match embedded loadout")
    normalize_sha256(str(evidence.get("reliability_fingerprint") or ""))
    artifact_hash = evidence.get("reliability_artifact_sha256")
    if artifact_hash is not None:
        normalize_sha256(str(artifact_hash))
    if expected_loadout is not None:
        expected = validate_manifest(
            expected_loadout,
            require_fingerprint=bool(expected_loadout.get("loadout_fingerprint")),
        )
        if loadout["loadout_fingerprint"] != expected["loadout_fingerprint"]:
            raise ValueError("throughput evidence belongs to a different loadout")
    return {
        "fingerprint": fingerprint,
        "loadout": loadout,
        "identity": identity,
        "reliability_fingerprint": evidence["reliability_fingerprint"],
    }


def verify_hermes_report(
    report: Mapping[str, Any],
    *,
    expected_suite_id: str,
    expected_loadout: Mapping[str, Any],
) -> Dict[str, Any]:
    if report.get("schema_version") != "hermes_agent_benchmark.v1":
        raise ValueError("unsupported Hermes report schema")
    if report.get("artifact_type") != "hermes_agent_intelligence_benchmark":
        raise ValueError("invalid Hermes artifact type")
    if report.get("suite_id") != expected_suite_id:
        raise ValueError(f"unexpected Hermes suite: {report.get('suite_id')}")
    if report.get("dry_run") is not False:
        raise ValueError("dry-run Hermes evidence cannot qualify")
    _non_admitted(report.get("admission"), "Hermes report")
    gate = _mapping(report.get("gate"), "Hermes gate")
    if gate.get("passed") is not True or gate.get("intelligence_qualified") is not True:
        raise ValueError("Hermes gate did not pass")
    if gate.get("failures") not in (None, []):
        raise ValueError("Hermes gate contains failures")
    identity = _mapping(report.get("identity"), "Hermes identity")
    if identity.get("loopback_only") is not True:
        raise ValueError("Hermes evidence is not loopback-only")
    _loopback_url(identity.get("endpoint"), "Hermes endpoint")
    loadout_raw = _mapping(report.get("loadout"), "Hermes loadout")
    report_loadout = validate_manifest(loadout_raw, require_fingerprint=True)
    expected = validate_manifest(
        expected_loadout,
        require_fingerprint=bool(expected_loadout.get("loadout_fingerprint")),
    )
    if report_loadout["loadout_fingerprint"] != expected["loadout_fingerprint"]:
        raise ValueError("Hermes report belongs to a different loadout")
    if identity.get("loadout_fingerprint") != expected["loadout_fingerprint"]:
        raise ValueError("Hermes identity loadout fingerprint mismatch")
    if identity.get("node_id") != expected["node_id"]:
        raise ValueError("Hermes node does not match loadout")
    if identity.get("candidate_id") != expected["candidate_id"]:
        raise ValueError("Hermes candidate does not match loadout")
    if identity.get("model_id") != expected["model"]["id"]:
        raise ValueError("Hermes model does not match loadout")
    aggregate = _mapping(report.get("aggregate"), "Hermes aggregate")
    if int(aggregate.get("valid_trial_count") or 0) < 1:
        raise ValueError("Hermes report contains no valid trials")
    fingerprint = normalize_sha256(str(report.get("benchmark_fingerprint") or ""))
    benchmark_core = {
        "identity": report.get("identity"),
        "suite_id": report.get("suite_id"),
        "suite_fingerprint": report.get("suite_fingerprint"),
        "trials_per_case": report.get("trials_per_case"),
        "trials": report.get("trials"),
        "aggregate": report.get("aggregate"),
        "gate": report.get("gate"),
        "dry_run": report.get("dry_run"),
        "admission": report.get("admission"),
    }
    if fingerprint != canonical_hash(benchmark_core):
        raise ValueError("Hermes benchmark fingerprint mismatch")
    return {
        "fingerprint": fingerprint,
        "suite_id": expected_suite_id,
        "suite_fingerprint": normalize_sha256(
            str(report.get("suite_fingerprint") or "")
        ),
        "aggregate": aggregate,
    }


def build_qualification(
    loadout_raw: Mapping[str, Any],
    throughput_evidence: Mapping[str, Any],
    base_hermes_report: Mapping[str, Any],
    context_hermes_report: Mapping[str, Any],
    *,
    throughput_artifact_sha256: Optional[str] = None,
    base_hermes_artifact_sha256: Optional[str] = None,
    context_hermes_artifact_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    loadout = validate_manifest(
        loadout_raw,
        require_fingerprint=bool(loadout_raw.get("loadout_fingerprint")),
    )
    throughput = verify_throughput_evidence(throughput_evidence, loadout)
    base = verify_hermes_report(
        base_hermes_report,
        expected_suite_id=BASE_HERMES_SUITE_ID,
        expected_loadout=loadout,
    )
    context = verify_hermes_report(
        context_hermes_report,
        expected_suite_id=CONTEXT_HERMES_SUITE_ID,
        expected_loadout=loadout,
    )

    def normalized_optional(value: Optional[str]) -> Optional[str]:
        return normalize_sha256(value) if value else None

    core = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "artifact_type": "exact_loadout_qualification",
        "identity": _identity(loadout),
        "loadout": loadout,
        "evidence": {
            "throughput": {
                "throughput_evidence_fingerprint": throughput["fingerprint"],
                "reliability_fingerprint": throughput["reliability_fingerprint"],
                "artifact_sha256": normalized_optional(throughput_artifact_sha256),
            },
            "base_hermes": {
                "suite_id": base["suite_id"],
                "suite_fingerprint": base["suite_fingerprint"],
                "benchmark_fingerprint": base["fingerprint"],
                "artifact_sha256": normalized_optional(base_hermes_artifact_sha256),
            },
            "context_hermes": {
                "suite_id": context["suite_id"],
                "suite_fingerprint": context["suite_fingerprint"],
                "benchmark_fingerprint": context["fingerprint"],
                "artifact_sha256": normalized_optional(context_hermes_artifact_sha256),
            },
        },
        "gates": {
            "exact_loadout_identity": True,
            "throughput_reliability": True,
            "base_hermes_intelligence": True,
            "context_pressure_intelligence": True,
            "matching_loadout_fingerprint": True,
            "loopback_only": True,
            "non_admitted": True,
        },
        "qualified": True,
        "admission": {"admitted": False},
    }
    return {
        **core,
        "created_at_utc": utc_now_iso(),
        "qualification_fingerprint": canonical_hash(core),
    }


def verify_qualification(
    report: Mapping[str, Any], expected_loadout: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    if report.get("schema_version") != QUALIFICATION_SCHEMA_VERSION:
        raise ValueError("unsupported qualification schema")
    if report.get("artifact_type") != "exact_loadout_qualification":
        raise ValueError("invalid qualification artifact type")
    if report.get("qualified") is not True:
        raise ValueError("loadout is not qualified")
    _non_admitted(report.get("admission"), "qualification")
    gates = _mapping(report.get("gates"), "qualification gates")
    required = {
        "exact_loadout_identity",
        "throughput_reliability",
        "base_hermes_intelligence",
        "context_pressure_intelligence",
        "matching_loadout_fingerprint",
        "loopback_only",
        "non_admitted",
    }
    failed = sorted(name for name in required if gates.get(name) is not True)
    if failed:
        raise ValueError("qualification gates failed: " + ", ".join(failed))
    fingerprint = normalize_sha256(str(report.get("qualification_fingerprint") or ""))
    core = {
        key: value
        for key, value in report.items()
        if key not in {"created_at_utc", "qualification_fingerprint"}
    }
    if fingerprint != canonical_hash(core):
        raise ValueError("qualification fingerprint mismatch")
    loadout_raw = _mapping(report.get("loadout"), "qualification loadout")
    loadout = validate_manifest(loadout_raw, require_fingerprint=True)
    identity = _mapping(report.get("identity"), "qualification identity")
    if identity != _identity(loadout):
        raise ValueError("qualification identity does not match loadout")
    evidence = _mapping(report.get("evidence"), "qualification evidence")
    throughput = _mapping(evidence.get("throughput"), "throughput evidence reference")
    base = _mapping(evidence.get("base_hermes"), "base Hermes evidence reference")
    context = _mapping(evidence.get("context_hermes"), "context Hermes evidence reference")
    normalize_sha256(str(throughput.get("throughput_evidence_fingerprint") or ""))
    normalize_sha256(str(throughput.get("reliability_fingerprint") or ""))
    normalize_sha256(str(base.get("benchmark_fingerprint") or ""))
    normalize_sha256(str(context.get("benchmark_fingerprint") or ""))
    if base.get("suite_id") != BASE_HERMES_SUITE_ID:
        raise ValueError("qualification base Hermes suite mismatch")
    if context.get("suite_id") != CONTEXT_HERMES_SUITE_ID:
        raise ValueError("qualification context Hermes suite mismatch")
    if expected_loadout is not None:
        expected = validate_manifest(
            expected_loadout,
            require_fingerprint=bool(expected_loadout.get("loadout_fingerprint")),
        )
        if loadout["loadout_fingerprint"] != expected["loadout_fingerprint"]:
            raise ValueError("qualification belongs to a different loadout")
    return {
        "fingerprint": fingerprint,
        "identity": identity,
        "loadout": loadout,
        "evidence": evidence,
        "gates": gates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lms-loadout-qualify",
        description="Bind and qualify exact-loadout throughput and Hermes evidence",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    bind = commands.add_parser("bind-throughput")
    bind.add_argument("--loadout", type=Path, required=True)
    bind.add_argument("--reliability", type=Path, required=True)
    bind.add_argument("--out", type=Path, required=True)

    qualify = commands.add_parser("qualify")
    qualify.add_argument("--loadout", type=Path, required=True)
    qualify.add_argument("--throughput", type=Path, required=True)
    qualify.add_argument("--base-hermes", type=Path, required=True)
    qualify.add_argument("--context-hermes", type=Path, required=True)
    qualify.add_argument("--out", type=Path, required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--qualification", type=Path, required=True)
    verify.add_argument("--loadout", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "bind-throughput":
            report = build_throughput_evidence(
                _mapping(load_json(args.loadout), "loadout"),
                _mapping(load_json(args.reliability), "reliability report"),
                reliability_artifact_sha256=file_sha256(args.reliability),
            )
            write_json(args.out, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "qualify":
            report = build_qualification(
                _mapping(load_json(args.loadout), "loadout"),
                _mapping(load_json(args.throughput), "throughput evidence"),
                _mapping(load_json(args.base_hermes), "base Hermes report"),
                _mapping(load_json(args.context_hermes), "context Hermes report"),
                throughput_artifact_sha256=file_sha256(args.throughput),
                base_hermes_artifact_sha256=file_sha256(args.base_hermes),
                context_hermes_artifact_sha256=file_sha256(args.context_hermes),
            )
            write_json(args.out, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "verify":
            expected = (
                _mapping(load_json(args.loadout), "loadout")
                if args.loadout
                else None
            )
            result = verify_qualification(
                _mapping(load_json(args.qualification), "qualification"),
                expected,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        raise AssertionError(args.command)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"loadout qualification rejected: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

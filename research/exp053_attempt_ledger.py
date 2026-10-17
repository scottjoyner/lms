#!/usr/bin/env python3
"""Verify future private per-cell attempt ledger against signed, parsed scores.

Structural evidence only. No inference, keys, human labels or runtime probes.
"""
import hashlib
from pathlib import Path

from exp053_execution_attestation import (
    hex_digest, private_regular_file, unambiguous_json,
)
from exp053_metrics import MODELS, ARMS, ROTATIONS
from exp053_race_review import canonical, filehash, deny

SPEC = Path(__file__).resolve().parent / "EXP053-ATTEMPT-LEDGER-SPEC-2026-10-17.md"
SPEC_SHA256 = "5d9c94cf7efb094f1fcea5ef422d51907c64bad2c2438a24ddf619f8a1a1bba0"
MAX_LEDGER_BYTES = 8 * 1024 * 1024
SCHEMA = "exp053-attempt-ledger-v1"
ENTRY_FIELDS = {"model", "case_id", "arm", "rotation", "score_row_sha256", "attempts"}
ATTEMPT_FIELDS = {"sequence", "status", "reason"}
REASONS = {"timeout", "runtime_error", "tokenization_error", "resource_exhausted"}


def cell_key(value):
    if not isinstance(value, dict):
        deny("attempt ledger cell must be an object")
    model, cid, arm, rotation = (value.get(k) for k in
                                 ("model", "case_id", "arm", "rotation"))
    if (not isinstance(model, str) or model not in MODELS or
            not isinstance(cid, str) or not cid or len(cid) > 128 or
            not isinstance(arm, str) or arm not in ARMS or
            type(rotation) is not int or rotation not in ROTATIONS):
        deny("attempt ledger cell identity invalid")
    return model, cid, arm, rotation


def verify_attempt_ledger(log_path, private_dir, scored_rows, *,
                          execution_id_sha256, packet_sha256, attempt_log_sha256):
    """Strictly match every source-free signed ledger cell to one score row.

    A syntactically complete log is NOT proof of actual model execution or
    that unlogged native attempts did not happen.
    """
    if not SPEC.is_file() or filehash(SPEC) != SPEC_SHA256:
        deny("prospective attempt ledger protocol drift")
    if not all(hex_digest(v) for v in (
            execution_id_sha256, packet_sha256, attempt_log_sha256)):
        deny("attested attempt ledger evidence fingerprints missing")
    if not isinstance(scored_rows, list) or not scored_rows:
        deny("attempt ledger cannot match an empty score archive")

    score_cells = {}
    for row in scored_rows:
        key = cell_key(row)
        if key in score_cells:
            deny("duplicate model/case/arm/rotation in score archive")
        score_cells[key] = hashlib.sha256(canonical(row)).hexdigest()
    observed, raw = private_regular_file(
        log_path, private_dir, small_limit=MAX_LEDGER_BYTES)
    if observed != attempt_log_sha256:
        deny("parsed attempt-ledger bytes differ from signed execution receipt")
    ledger = unambiguous_json(raw)
    if not isinstance(ledger, dict) or set(ledger) != {
            "schema", "execution_id", "packet_sha256", "entries"}:
        deny("attempt-ledger top-level fields incomplete or leaking data")
    if ledger["schema"] != SCHEMA:
        deny("unknown attempt-ledger schema")
    run_id = ledger["execution_id"]
    if (not isinstance(run_id, str) or
            hashlib.sha256(run_id.encode()).hexdigest() != execution_id_sha256):
        deny("attempt ledger belongs to a different signer-approved run")
    if ledger["packet_sha256"] != packet_sha256:
        deny("attempt ledger not bound to approved candidate packet")
    entries = ledger["entries"]
    if not isinstance(entries, list) or len(entries) != len(score_cells):
        deny("attempt ledger lacks exact one-to-one score-cell coverage")
    seen = set()
    failed_attempts = 0
    total_attempts = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS:
            deny("attempt ledger entry contains missing or unexpected fields")
        key = cell_key(entry)
        if key in seen or key not in score_cells:
            deny("duplicate or unscored attempt ledger cell")
        seen.add(key)
        if (not hex_digest(entry["score_row_sha256"]) or
                entry["score_row_sha256"] != score_cells[key]):
            deny("attempt ledger row differs from exact private score record")
        attempts = entry["attempts"]
        if not isinstance(attempts, list) or not 1 <= len(attempts) <= 4:
            deny("attempt ledger has missing or unbounded retries")
        for j, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, dict) or set(attempt) != ATTEMPT_FIELDS:
                deny("attempt record contains missing or unexpected fields")
            if type(attempt["sequence"]) is not int or attempt["sequence"] != j:
                deny("attempt sequence is not contiguous from 1")
            expected_status = "success" if j == len(attempts) else "failure"
            if attempt["status"] != expected_status:
                deny("attempt ledger missing final success or has early success")
            reason = attempt["reason"]
            if not isinstance(reason, str) or (
                    (expected_status == "success" and reason != "none") or
                    (expected_status == "failure" and reason not in REASONS)):
                deny("invalid failure/retry reason or success reason")
            total_attempts += 1
            if expected_status == "failure":
                failed_attempts += 1
    if seen != set(score_cells):
        deny("attempt ledger omits scored cells")
    return {
        "attempt_ledger_sha256": observed,
        "attempt_cells": len(seen),
        "total_logged_attempts": total_attempts,
        "logged_failed_attempts": failed_attempts,
        "audit_limit": "source-free structural ledger only, not native execution proof",
    }

#!/usr/bin/env python3
"""Export ONLY aggregate machine-proxy Qwen statistics, after strict private replay.

Read-only. No source passages, questions, answers, prompts, per-case scores,
human review claims, licensed corpus access or model forwards are emitted.
"""
import hashlib
import json
import re
from pathlib import Path

import exp053_agent_qwen_summary as summary

SCHEMA = "exp053-qwen-agent-aggregate-public-v1"
COUNT = 64
SOURCE_SUMMARY = "summary-000-063-v2.json"
MODEL = "Qwen3.5-4B-Base"


def refuse(message):
    raise ValueError("HOLD EXP053 AGENT PUBLIC REPORT: " + message)


def hex_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def public_payload(private, metrics, *, report_sha):
    """Allowlist safe aggregate metadata; unknown private fields never escape."""
    if (private.get("schema") != "exp053-qwen-agent-only-summary-v2" or
            private.get("complete_frozen_cohort") is not True or
            private.get("human_adjudicated") is not False or
            private.get("machine_selected_foil") is not True or
            private.get("semantic_grounding_proven") is not False or
            type(private.get("case_count")) is not int or private["case_count"] != COUNT or
            type(private.get("condition_count")) is not int or
            private["condition_count"] != COUNT * 6):
        refuse("summary claims or cohort cardinality invalid")
    for field in ("model_checkpoint_receipt_sha256", "runner_sha256"):
        if not hex_sha(private.get(field)):
            refuse("invalid pinned fingerprint")
    if not hex_sha(report_sha):
        refuse("invalid immutable summary checksum")
    if not isinstance(metrics, dict) or set(metrics) != {"D", "N", "S"}:
        refuse("unexpected metric inventory")
    values = {}
    for name in ("D", "N", "S"):
        row = metrics[name]
        if not isinstance(row, dict) or set(row) != {"mean", "median", "positive", "n"}:
            refuse("invalid aggregate metric schema")
        if (type(row["positive"]) is not int or not 0 <= row["positive"] <= COUNT or
                type(row["n"]) is not int or row["n"] != COUNT or
                any(type(row[x]) is not float or not -10000 < row[x] < 10000
                    for x in ("mean", "median"))):
            refuse("nonfinite or inconsistent aggregate")
        values[name] = {key: row[key] for key in ("mean", "median", "positive", "n")}
    return {
        "schema": SCHEMA,
        "mode": "machine_only_exam_key_proxy",
        "model": MODEL,
        "case_count": COUNT,
        "source_conditions": COUNT * 6,
        "human_reviewers": 0,
        "human_adjudicated": False,
        "scientific_grounding_proven": False,
        "independent_semantic_foil": False,
        "actual_model_receipts_replayed": True,
        "source_text_or_private_labels_exposed": False,
        "model_checkpoint_receipt_sha256": private["model_checkpoint_receipt_sha256"],
        "runner_sha256": private["runner_sha256"],
        "private_summary_sha256": report_sha,
        "aggregates": values,
        "metric_definitions": {
            "D": "average across rotations: supported minus withheld gold/foil logprob margin",
            "N": "average across rotations: decoy minus withheld gold/foil logprob margin",
            "S": "average across rotations: supported minus decoy gold/foil logprob margin",
        },
        "limitations": "Machine-selected exam-option foil is NOT independent semantic adjudication; descriptive statistics only, no causal or contamination claim.",
    }


def audit():
    summary.private_dir(summary.RECEIPTS)
    cohort, cases, model = summary.source_data()
    model_fingerprint = model["checkpoint_receipt_sha256"]
    runner_sha = summary.sha(Path(__file__).with_name("exp053_agent_qwen_cpu.py").read_bytes())
    records, hashes = [], []
    expected_files = {"case-%03d.json" % i for i in range(COUNT)}
    actual_files = {p.name for p in summary.RECEIPTS.glob("case-*.json")}
    if actual_files != expected_files:
        refuse("missing, extra or misnamed case receipt")
    if any(summary.RECEIPTS.glob("failure-*.json")):
        refuse("incomplete or failed case receipt present")
    for idx, (cid, _article_hash) in enumerate(cases):
        file = summary.RECEIPTS / ("case-%03d.json" % idx)
        entry = summary.read_receipt(file)
        summary.verify_case(entry, idx, cid, model_fingerprint, cohort["packet_sha256"], runner_sha)
        if (entry.get("tokenizer_sha256") != model["tokenizer_sha256"] or
                entry.get("study_complete") is not False or
                entry.get("condition_count") != 6 or
                entry.get("runtime", {}).get("transformers") != "5.12.1" or
                entry.get("runtime", {}).get("torch") != "2.13.0+rocm7.1"):
            refuse("receipt provenance or claim drift")
        records.append(entry)
        hashes.append({"index": idx, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    private_path = summary.RECEIPTS / SOURCE_SUMMARY
    private = summary.read_receipt(private_path)
    metrics = summary.summarize(records, count=COUNT)
    if (private.get("descriptive_metrics") != metrics or
            private.get("receipts") != hashes or
            private.get("runner_sha256") != runner_sha or
            private.get("model_checkpoint_receipt_sha256") != model_fingerprint):
        refuse("private archive digest or aggregate did not replay")
    return public_payload(
        private, metrics, report_sha=hashlib.sha256(private_path.read_bytes()).hexdigest())


def main():
    print(json.dumps(audit(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Source-locked, machine-only EXP-053 exploratory labels; never impersonate people.

Dry run validates only the public manifest. Explicit private mode requires the
exact licensed dataset, pinned 64-case packet and sealed exam answer key. It
emits a private proxy receipt, NEVER a human review or a model evaluation.
"""
import argparse
import json
from pathlib import Path

import exp053_race_review as trust

SCHEMA = "exp053-agent-answer-key-proxy-v1"
MODE = "automated_exam_key_proxy_not_independent_human_judgement"


def preflight():
    manifest = trust._read_pinned_public_manifest()
    if not isinstance(manifest, dict) or manifest.get("schema") != "exp053-public-manifest-v1":
        trust.deny("invalid pinned source-free public cohort manifest")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != trust.TARGET:
        trust.deny("agent-only source-free cohort size differs from pinned source")
    ids = [row.get("case_id") if isinstance(row, dict) else None for row in rows]
    if (not all(isinstance(x, str) and x for x in ids) or
            len(set(ids)) != trust.TARGET):
        trust.deny("invalid or repeated public cohort case identity")
    if manifest.get("source_sha256") != trust.SOURCE_SHA or manifest.get("protocol_sha256") != trust.PROTOCOL_SHA:
        trust.deny("public manifest does not match pinned source and protocol")
    if manifest.get("evaluation_admitted") is not False:
        trust.deny("public manifest falsely asserts admission")
    return {
        "schema": SCHEMA,
        "mode": MODE,
        "source_sha256": trust.SOURCE_SHA,
        "protocol_sha256": trust.PROTOCOL_SHA,
        "public_manifest_sha256": trust.MANIFEST_SHA,
        "candidate_articles": len(rows),
        "human_review_required": False,
        "human_approval": False,
        "scientific_human_adjudication": False,
        "model_forward_passes": 0,
        "status": "SOURCE_ONLY_PREFLIGHT_VALID_NO_MODEL_OR_HUMAN",
    }


def machine_receipt(private_dir):
    """Validate pinned private packet/key without trusting spoofed review receipts."""
    evidence = preflight()
    if (not trust.SOURCE.is_file() or trust.filehash(trust.SOURCE) != trust.SOURCE_SHA or
            not trust.PROTOCOL.is_file() or trust.filehash(trust.PROTOCOL) != trust.PROTOCOL_SHA):
        trust.deny("pinned licensed source or protocol unavailable/changed")
    packet = trust._read_private_admission_json(Path(private_dir) / "packet.json")
    sealed = trust._read_private_admission_json(Path(private_dir) / "sealed_exam_key.json")
    manifest = trust._read_pinned_public_manifest()
    if (packet.get("source_sha256") != trust.SOURCE_SHA or
            packet.get("protocol_sha256") != trust.PROTOCOL_SHA or
            packet.get("license") != "RACE: non-commercial research only; do not redistribute" or
            trust.digest(packet) != manifest.get("packet_sha256")):
        trust.deny("private agent-only packet source, license or identity mismatch")
    answers = trust._validate_sealed_case_scope(packet, sealed, manifest)
    source_rows = {row["case_id"]: row for row in manifest["rows"]}
    if set(source_rows) != set(answers):
        trust.deny("public cohort case identifiers do not match sealed key")

    out = {}
    for case in packet["cases"]:
        cid = case["case_id"]
        pinned = source_rows[cid]
        options = case.get("options")
        if (not isinstance(options, dict) or set(options) != set(trust.LETTERS) or
                not all(isinstance(options[l], str) and options[l].strip() for l in trust.LETTERS) or
                len(set(options[l].strip().casefold() for l in trust.LETTERS)) != 4):
            trust.deny("agent-only packet has invalid or ambiguous options")
        if (not isinstance(case.get("article"), str) or
                not isinstance(case.get("question"), str) or
                trust.digest(case["article"].encode()) != pinned.get("article_sha256") or
                trust.digest(case["question"].encode()) != pinned.get("question_sha256") or
                trust.digest(trust.canonical([options[l] for l in trust.LETTERS])) != pinned.get("options_sha256")):
            trust.deny("private agent-only packet contents not bound to published digest")
        key = answers[cid]
        # Precommitted deterministic *incorrect option*, not a human-vetted
        # plausible semantic foil and never optimized using model logits.
        negatives = [l for l in trust.LETTERS if l != key]
        foil = min(negatives, key=lambda l: trust.digest((trust.SEED + "|machine-foil|" + cid + "|" + l).encode()))
        out[cid] = {"supported_exam_key": key, "other_exam_option_proxy": foil}
    if len(out) != trust.TARGET:
        trust.deny("missing machine-only answer key coverage")
    return {
        **evidence,
        "status": "MACHINE_PROXY_LABELS_PRIVATE_READY",
        "packet_sha256": trust.digest(packet),
        "sealed_answer_key_sha256": manifest["sealed_answer_key_sha256"],
        "accepted": out,
        "human_reviewers": 0,
        "human_approval": False,
        "agent_only_exploratory": True,
        "evaluation_admitted_as_human_validated": False,
        "model_forward_passes": 0,
        "claims_limitation": "Exam key and arbitrarily wrong option are NOT independent semantic-grounding adjudication",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path)
    parser.add_argument("--private-out", type=Path)
    args = parser.parse_args(argv)
    if (args.private_dir is None) != (args.private_out is None):
        parser.error("--private-dir and --private-out are required together")
    if args.private_out is None:
        print(json.dumps(preflight(), sort_keys=True))
        return 0
    if args.private_out.exists() or args.private_out.is_symlink():
        trust.deny("agent-only proxy output already exists; never overwrite")
    if args.private_out.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        trust.deny("agent-only private proxy receipt must not be inside public source tree")
    receipt = machine_receipt(args.private_dir)
    trust.save_private(args.private_out, receipt)
    print(json.dumps({
        "status": receipt["status"], "machine_labeled_cases": len(receipt["accepted"]),
        "human_approval": False, "model_forward_passes": 0,
        "private_receipt_saved": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

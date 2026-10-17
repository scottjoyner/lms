#!/usr/bin/env python3
"""Read-only BF16 checkpoint byte identity; DOES NOT import or execute models."""
import argparse
import hashlib
import json
from pathlib import Path

from exp053_race_review import deny, filehash, canonical
from exp053_tokenizer_preflight import PATHS, ROOT

# Frozen BEFORE hashing weight files; never silently accept a revised spec.
SPEC = ROOT / "EXP053-WEIGHT-ID-PREFLIGHT-SPEC-2026-10-16.md"
SPEC_SHA = "b89f573acc2567b114e65746b3026e36636aee9add6c6568c0fac6319f620a81"
TOKENS_REPORT = ROOT / "EXP053-TOKENIZER-PREFLIGHT-2026-10-16.json"
TOKENS_REPORT_SHA = "8e35a44e1ebb725c0b52469baa4ab5d41472e38fe55346352b96dd26697726be"
TWO_SHARD_MODELS = {"K2-Horizon-3.7B", "Qwen3.5-4B-Base"}

def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def allowed_resolved(path, model_path):
    canonical_path = path.resolve(strict=True)
    model_root = model_path.resolve(strict=True)
    if canonical_path.is_relative_to(model_root):
        return canonical_path
    # Hugging Face stores repo-specific immutable blobs adjacent to snapshots.
    # Some pre-existing snapshots also point to a hub-wide content-addressed
    # blob store. Neither rule applies to unrecognized storage layouts.
    if (model_root.parent.name == "snapshots" and
            model_root.parent.parent.name.startswith("models--") and
            model_root.parent.parent.parent.name == "hub"):
        repo_blobs = model_root.parent.parent / "blobs"
        if canonical_path.is_relative_to(repo_blobs):
            return canonical_path
        hub_blobs = model_root.parent.parent.parent / "blobs"
        try:
            rel = canonical_path.relative_to(hub_blobs)
        except ValueError:
            deny("unexpected model file symlink/path escape " + path.name)
        if (len(rel.parts) == 2 and len(rel.parts[0]) == 2 and
                len(rel.parts[1]) == 64 and
                all(c in "0123456789abcdef" for c in rel.parts[0]+rel.parts[1]) and
                rel.parts[1].startswith(rel.parts[0])):
            return canonical_path
    deny("unexpected model file symlink/path escape " + path.name)

def stream_digest(path, model_path):
    allowed_resolved(path, model_path)
    before = path.stat()
    if before.st_size < 4096:
        deny("incomplete checkpoint weight file " + path.name)
    h = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
            count += len(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ) or count != before.st_size:
        deny("checkpoint weight was modified during identity scan")
    return {"filename": path.name, "bytes": count, "sha256": h.hexdigest()}

def source_files(name, path):
    all_weight_files = {p.name: p for p in path.glob("*.safetensors")}
    if name in TWO_SHARD_MODELS:
        index = path / "model.safetensors.index.json"
        if not index.is_file():
            deny("missing frozen safetensors weight index " + name)
        allowed_resolved(index, path)
        data = json.loads(index.read_text())
        weights = data.get("weight_map")
        if not isinstance(weights, dict) or not weights:
            deny("invalid shard map " + name)
        expected = set(weights.values())
        if len(expected) < 2 or any(
            not isinstance(x, str) or "/" in x or "\\" in x
            or not x.endswith(".safetensors") for x in expected
        ):
            deny("bad weight-map filename or insufficient shard count " + name)
        if set(all_weight_files) != expected:
            deny("unexpected/missing weight shards " + name)
        return [all_weight_files[n] for n in sorted(expected)], filehash(index)
    if set(all_weight_files) != {"model.safetensors"} or (
        path / "model.safetensors.index.json"
    ).exists():
        deny("noncanonical MiniCPM full-checkpoint inventory")
    return [all_weight_files["model.safetensors"]], None

def scan():
    if not SPEC.exists() or filehash(SPEC) != SPEC_SHA:
        deny("preregistered weights-only protocol changed")
    if not TOKENS_REPORT.exists() or filehash(TOKENS_REPORT) != TOKENS_REPORT_SHA:
        deny("tokenizer-only preflight source changed")
    token_receipt = json.loads(TOKENS_REPORT.read_text())
    if set(token_receipt["models"]) != set(PATHS):
        deny("tokenizer checkpoint catalog mismatch")
    out = {}
    for name, path in PATHS.items():
        if not path.is_dir():
            deny("model directory missing " + name)
        checkpoint = token_receipt["models"][name]
        config = path / "config.json"
        tok = path / "tokenizer.json"
        for candidate in (config, tok):
            allowed_resolved(candidate, path)
        if filehash(config) != checkpoint["config_sha256"]:
            deny("configuration byte identity mismatch " + name)
        if filehash(tok) != checkpoint["tokenizer_json_sha256"]:
            deny("tokenizer byte identity mismatch " + name)
        paths, index_hash = source_files(name, path)
        receipts = []
        for shard in paths:
            receipt = stream_digest(shard, path)
            print("HASHED_WEIGHT_SHARD", name, shard.name, receipt["bytes"], flush=True)
            receipts.append(receipt)
        model_receipt = {
            "model": name,
            "checkpoint_revision_or_directory": checkpoint["checkpoint_directory"],
            "config_sha256": checkpoint["config_sha256"],
            "tokenizer_sha256": checkpoint["tokenizer_json_sha256"],
            "weight_index_sha256": index_hash,
            "shards": receipts,
            "shard_count": len(receipts),
            "total_weight_bytes": sum(r["bytes"] for r in receipts),
        }
        model_receipt["checkpoint_receipt_sha256"] = sha(model_receipt)
        out[name] = model_receipt
    result = {
        "schema": "exp053-source-only-weight-identity-v1",
        "spec_sha256": SPEC_SHA,
        "tokenizer_receipt_sha256": TOKENS_REPORT_SHA,
        "models": out,
        "neural_forward_passes": 0,
        "human_reviewed_articles": 0,
        "checkpoint_receipts_sha256": sha(out),
        "claim_limit": "File-byte identity only; no proof of runtime execution or grounding.",
    }
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute-read-only-hashes", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if filehash(SPEC) != SPEC_SHA or filehash(TOKENS_REPORT) != TOKENS_REPORT_SHA:
        deny("research sources drifted")
    if not a.execute_read_only_hashes:
        print("EXP053_WEIGHTS_DRY_RUN_ONLY; no weights read")
        return
    if a.out is None or a.out.exists() or a.out.parent.resolve() != ROOT.resolve():
        deny("weight receipt destination must be new research file")
    result = scan()
    with a.out.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("EXP053_WEIGHTS_READ_ONLY_OK", result["checkpoint_receipts_sha256"])

if __name__ == "__main__":
    main()

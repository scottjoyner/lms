# EXP-053 future scorer execution attestation — prospective contract (2026-10-17)

**Research preparation only. No human approvals or EXP-053 neural forwards exist.**

The currently implemented score-archive replay authenticates the private source
packet, signed human-admission claim, prompt/option/decoy receipts, pinned
tokenizer/config and exact model-weight inventory. That **does not establish**
whether a real model executed or whether log probabilities were fabricated.

## Future separately reviewed runner authority

Before score analysis, require a private Ed25519 execution-attestation record
signed by a research runner whose **public key fingerprint is pinned by an
independent human operator** in a code-reviewed change, and whose exact
**scorer implementation SHA256 is separately pinned**. Both pins intentionally
remain unconfigured. A synthetic test key or generated signature cannot
configure either pin.

## Execution receipt contract

The attestation is a single JSON object with exactly these fields:

- schema = `exp053-execution-attestation-v1`
- execution_id = unique opaque identifier, not a person name
- runner_public_key_b64 = raw Ed25519 public key
- runner_source_sha256 = pinned implementation source digest
- packet_sha256 = exact admitted private candidate packet
- signed_admission_sha256 = digest of the *verified* human-admission result
- weight_inventory_sha256 = pinned and approved weight-file receipt
- tokenizer_preflight_sha256 = immutable tokenizer-only receipt
- prompt_builder_sha256 = immutable source-bound prompt code
- score_archive_sha256 = exact byte SHA256 of the private score file
- attempt_log_sha256 = exact byte SHA256 of a separate, PRIVATE failure/retry
  log (including explicit no-failure records if there were no retries)
- checkpoint_receipts = exact map of all three model names to the independently
  approved per-model checkpoint digests
- score_row_count = `approved articles × 3 models × 3 arms × 2 rotations`
- execution_environment_sha256 = separately retained environment inventory
- runner_mode = `raw`
- simulation = false
- signature_hex = Ed25519 signature of canonical sorted JSON for **all fields
  except signature_hex**.

Do not accept unknown fields, hash substitutes, duplicate JSON keys, symlinked
private files, invalid signatures, wrong key roots, source-hash drift, or
simulated execution. The score archive and attempt log must be files inside
the selected private directory, not public-repository paths.

Validate this authority *before* opening or analyzing individual score rows;
later source-only and statistical checks remain independent and mandatory.

## What a signature cannot prove

A signature binds bytes to a key but **cannot prove a model actually executed**
those rows, that a machine was trusted, or that logs are exhaustive.
Independent operator review of runtime instrumentation, process/model-load
evidence, retry completeness, and physical-machine checkpoint receipt is
still necessary. A crypto-valid **test fixture** is not a human signoff.

The future runner, key authorization and runtime control protocol must be
separately reviewed, approved and frozen before any actual EXP-053 forward pass.
No private prompt, corpus passage, source-key, model weight or per-case score
shall be published. No production changes, training or deployment are permitted.

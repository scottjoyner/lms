# LMS Agent Benchmarking Toolkit High-Level Design

## Document status

- **System:** `lms-agent-bench`
- **Purpose:** Canonical high-level design for deterministic fleet observation, benchmarking, exact-loadout qualification, runtime canary, rollback, and signed evidence
- **Audience:** Fleet operators, benchmark maintainers, runtime owners, profile importers, and admission reviewers
- **Authority:** Describes the intended architecture of the current `main` branch. CLI implementations, schemas, and verification code remain authoritative when they differ.

## 1. Problem statement

A heterogeneous local inference fleet cannot be safely routed from ad hoc benchmark output, mutable node state, or a single successful completion. Operators need reproducible evidence that binds hardware, runtime, model artifact, quantization, source code, commands, benchmark results, intelligence results, sustained behavior, and rollback outcome.

The toolkit must generate this evidence without making routing or admission decisions itself.

## 2. Goals

1. Observe the complete operator-approved fleet using strict, bounded, reproducible procedures.
2. Produce deterministic throughput and intelligence evidence for one immutable loadout.
3. Bind model artifact, quantization, runtime, source commits, inventory, and test cases into exact fingerprints.
4. Execute one transactional candidate canary and restore the previous runtime even after failure.
5. Measure sustained latency, memory, temperature, throughput, TTFT, and error behavior during soak.
6. Seal evidence in manifests with path, size, and SHA-256 verification.
7. Authenticate evidence with OpenSSH signatures and external allowed-signers policy.
8. Preserve a strict separation between evidence production and live admission.
9. Fail closed on unsafe files, mutable source, shell commands, missing telemetry, stale locks, partial archives, or inconsistent loadouts.
10. Support downstream profile import and independent admission review.

## 3. Non-goals

The toolkit does not:

- assign tasks or route production requests;
- modify AssistX or router admission state;
- leave a canary candidate running after a successful test;
- autonomously load or unload models outside reviewed lifecycle wrappers;
- restore KV cache state or claim cache savings;
- treat repository CI as proof of physical node readiness;
- include Raspberry Pi or iPhone devices as inference nodes;
- benchmark powered-off nodes while pretending the fleet census is complete;
- store secrets in command arguments, logs, or manifests.

## 4. System context

```text
operator-reviewed fleet census and plans
                 |
                 v
        lms-agent-bench toolkit
        - preflight and observation
        - repeated throughput
        - Hermes intelligence
        - exact-loadout qualification
        - transactional canary and soak
        - rollback verification
        - manifest sealing and attestation
                 |
                 v
      signed non-admitting evidence bundles
                 |
       +---------+------------------+
       |                            |
       v                            v
fleet-llm-profiles importer    independent admission review
                                     |
                                     v
                           AssistX / auto-router policy
```

### Related repositories

| Repository | Relationship |
|---|---|
| `fleet-llm-profiles` | Imports and stores verified observation, qualification, and canary evidence in desired-state profiles. |
| `auto-assist` | Uses independently reviewed live evidence for fleet admission, allocation, and recovery decisions. |
| `auto-router` | Forwards only to AssistX-approved local runtimes; it does not consume benchmark success as autonomous admission. |

## 5. Architectural principles

### 5.1 Evidence is non-admitting

Every command produces artifacts and verification results only. A successful run cannot enable a profile, change routing, or modify a production runtime beyond the bounded candidate lifecycle that is always rolled back.

### 5.2 Exact-loadout identity

Throughput, base Hermes intelligence, context-pressure Hermes intelligence, and canary evidence must refer to the same immutable loadout fingerprint. Results from different model files, quantizations, runtime settings, source commits, or inventories cannot be combined.

### 5.3 Complete fleet accounting

Fleet workflows are driven by a reviewed census. Each expected node must end in a declared state such as completed, failed with evidence, or explicitly deferred under policy. `joyner` remains `benchmark_deferred` while powered off and must return to `benchmark_required` when online.

### 5.4 Deterministic and bounded execution

Commands use absolute executable paths, reviewed argv, bounded timeouts, process groups, fixed source commits, clean worktrees, and explicit input hashes. Shell interpreters and secret-bearing arguments are rejected.

### 5.5 Tamper-evident artifacts

Each run produces a normalized state file and a manifest binding all included artifacts by relative path, size, and SHA-256. Verification rejects missing, changed, duplicated, escaping, unsafe, or symlinked files.

### 5.6 Independent cryptographic verification

OpenSSH signatures bind the manifest and critical identity fields. Verification uses an external protected allowed-signers policy rather than trusting a key bundled inside the evidence.

### 5.7 Rollback is part of success criteria

A runtime canary is successful only when the candidate sequence passes and the previous runtime is restored and verified healthy. A failed candidate remains failed even if rollback succeeds.

## 6. Major subsystems

### 6.1 Agent-facing and benchmark CLIs

The package exposes focused commands for doctor/probe/profile/route inspection, reliability-first benchmarking, fleet observation, rollout, attestation, loadout matrices, Hermes suites, qualification, prompt-cache evidence, and runtime canary.

Each CLI has a narrow contract and emits machine-readable artifacts suitable for independent verification.

### 6.2 Fleet census and operator workflow

The fleet operator reads a reviewed node configuration and protected environment file, validates local controller safety, checks SSH host trust and node coverage, executes bounded remote observation, collects archives atomically, runs postflight, and seals one fleet run.

### 6.3 Throughput benchmarking

Benchmark cases define prompts or request shapes, concurrency, context, token limits, warmup, repetition, and acceptance rules. Results preserve raw case evidence and summarized throughput/latency statistics without mixing incompatible loadouts.

### 6.4 Hermes intelligence benchmarking

Hermes suites exercise base task intelligence and context-pressure behavior through reviewed local endpoints and source commits. API-key values are inherited from named environment variables and excluded from argv and artifacts.

### 6.5 Exact-loadout qualification

One qualification run orchestrates throughput plus both Hermes suites under a single lock and loadout fingerprint. The qualification gate verifies that every component references the same exact artifact, runtime, inventory, and source identities.

### 6.6 Runtime canary and rollback

The canary validates an immutable plan, acquires a local lock, snapshots the previous runtime, starts the candidate, checks health, runs exact-loadout qualification, performs a sustained soak, stops the candidate, restores the previous runtime, verifies rollback health, and seals evidence.

### 6.7 Prompt-cache evidence

The prompt-cache subsystem records opaque prefix identity, compatibility metadata, candidate observations, and engine-native payload references in a content-addressed store. It remains record-only and never restores state, skips tokens, alters routing, or claims measured savings without a separate physical experiment.

### 6.8 Attestation and release gates

Attestation commands sign and verify observation, qualification, and canary manifests under distinct namespaces and identities. Release gates enforce required success, rollback, manifest, and signer conditions before an artifact may be imported downstream.

## 7. Evidence chain

A production admission review should require four independent roots:

```text
signed fleet observation
        +
signed exact-loadout qualification
        +
signed runtime canary with sustained soak and verified rollback
        +
current live identity, reachability, capacity, freshness, and rollback checks
        |
        v
independent admission decision outside this repository
```

No individual root is sufficient by itself.

## 8. Fleet observation flow

```text
reviewed census + protected env
  -> controller preflight
  -> strict SSH host-key validation
  -> exact source and command validation
  -> per-node preflight
  -> bounded remote observation
  -> atomic archive transfer and promotion
  -> all-node postflight
  -> complete coverage gate
  -> manifest and optional attestation
```

Unexpected failure aborts rather than allowing an agent to improvise a new procedure.

## 9. Exact qualification flow

```text
reviewed loadout and inventory
  -> hash artifact/config/source/cases
  -> acquire qualification lock
  -> repeated throughput suite
  -> base Hermes intelligence suite
  -> context-pressure Hermes suite
  -> cross-check one loadout fingerprint
  -> compute gate result
  -> seal manifest
  -> sign and independently verify
```

## 10. Runtime canary flow

```text
validate plan and executable hashes
  -> acquire host/boot/PID-aware lock
  -> snapshot previous runtime
  -> start candidate process group
  -> candidate health check
  -> exact-loadout qualification
  -> sustained soak and metric gates
  -> stop candidate
  -> restore previous runtime
  -> rollback health check
  -> seal state and manifest
```

Any failure after candidate startup enters best-effort candidate stop, rollback, and rollback health verification. The run remains failed regardless of rollback success.

## 11. Data and artifact model

| Artifact class | Purpose |
|---|---|
| Normalized plan/config | Canonical reviewed inputs with unsafe or secret values excluded |
| Run state | Step status, timestamps, identifiers, success/failure, rollback status |
| Raw benchmark/case evidence | Reproducible measurements and bounded logs |
| Soak samples | JSONL metrics: success, latency, RSS, temperature, TPS, TTFT |
| Command records | Absolute executable path, executable hash, argv metadata, timeout, exit status |
| Manifest | Relative path, size, SHA-256, run/loadout/plan identity |
| Signature record | OpenSSH namespace, signer identity, key fingerprint, signature path |
| Content-addressed cache payload | Opaque engine-native data outside graph/control-plane state |

## 12. Security boundaries

- SSH uses strict host-key verification and reviewed host aliases.
- Secret files are mode-restricted and external to the repository.
- API secrets are inherited through named environment variables only.
- Shell interpreters and `*-c` command execution are rejected in secure entrypoints.
- Command executables must be absolute, existing, nonsymlinked where required, and hashed.
- Plans and evidence cannot be group/world writable.
- Archive extraction rejects path traversal, symlinks, and containment violations.
- Allowed signers are supplied from protected policy outside the evidence bundle.
- Raw prompts, private content, and secret-bearing logs are not persisted in route or cache evidence.

## 13. Reliability and failure strategy

| Failure | Behavior |
|---|---|
| Node unavailable | Record bounded failure or policy-approved deferment; do not claim complete success |
| SSH trust mismatch | Abort before remote execution |
| Dirty or wrong source commit | Abort |
| Disk, path, hostname, or clock preflight failure | Abort node/run according to complete coverage policy |
| Command timeout | Kill bounded process group and record failure |
| Partial transfer | Keep temporary file; do not promote final archive |
| Archive containment violation | Reject archive |
| Loadout fingerprint mismatch | Reject qualification |
| Missing soak metric required by a gate | Fail gate |
| Candidate health or qualification failure | Stop candidate and rollback |
| Soak failure | Stop candidate and rollback |
| Rollback health failure | Mark run failed with rollback failure evidence |
| Controller interruption | Recover or reject stale lock using host/boot/PID policy |
| Signature or manifest mismatch | Reject evidence |

## 14. Deployment and operating model

The toolkit runs from a trusted operator/controller environment and may invoke reviewed remote procedures over SSH. It does not require a central service. Workspaces are local protected directories containing one run per immutable run ID. Operators should separate keys, allowed-signers files, plans, environment files, run workspaces, and imported evidence destinations.

## 15. Architectural decisions

1. Evidence production and admission remain separate.
2. Observation, throughput, intelligence, qualification, canary, and profile import are distinct stages.
3. Exact-loadout identity is mandatory across qualification components.
4. Successful canary always restores the previous runtime.
5. Signatures bind verified manifests, not loose summaries.
6. SSH trust, source identity, file safety, and timeouts are fail-closed.
7. Prompt/KV support remains record-only until a separate restore experiment proves behavior.
8. Physical failure injection remains required beyond repository CI.

## 16. Related documents

- [`LOW_LEVEL_DESIGN.md`](LOW_LEVEL_DESIGN.md)
- [`DETERMINISTIC_FLEET_OPERATOR.md`](DETERMINISTIC_FLEET_OPERATOR.md)
- [`FLEET_OPERATIONAL_RELIABILITY.md`](FLEET_OPERATIONAL_RELIABILITY.md)
- [`EXACT_LOADOUT_QUALIFICATION.md`](EXACT_LOADOUT_QUALIFICATION.md)
- [`LOADOUT_QUALIFICATION_OPERATOR.md`](LOADOUT_QUALIFICATION_OPERATOR.md)
- [`RUNTIME_CANARY_AND_ROLLBACK.md`](RUNTIME_CANARY_AND_ROLLBACK.md)
- [`OPERATIONAL_GAP_AUDIT.md`](OPERATIONAL_GAP_AUDIT.md)

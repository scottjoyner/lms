# LMS Agent Benchmarking Toolkit Low-Level Design

## Document status

- **System:** `lms-agent-bench`
- **Scope:** CLI structure, run directories, immutable identities, remote execution, qualification, canary, soak, rollback, manifests, and attestation
- **Companion:** [`HIGH_LEVEL_DESIGN.md`](HIGH_LEVEL_DESIGN.md)
- **Implementation authority:** Python modules, CLI parsers, JSON schemas, and verification code override this document when they differ.

## 1. Package and command structure

The package exposes multiple installed commands with deliberately separate responsibilities:

| Command family | Responsibility |
|---|---|
| `lms-agent` | Agent-facing doctor, probe, profile, and route inspection |
| `lms-bench` | Reliability-first single-endpoint benchmark execution |
| `lms-fleet` | Hardware observation, planning, and candidate selection |
| `lms-fleet-bench` | Guarded loopback candidate benchmarking |
| `lms-fleet-models` | Runtime model inventory and selected model hashing |
| `lms-fleet-rollout` | Low-level census-validated SSH execution |
| `lms-fleet-operator` | Hardened complete-fleet observation workflow |
| `lms-fleet-attest` | Sign and verify fleet observation evidence |
| `lms-fleet-gate` | Verify collected archive and release conditions |
| `lms-loadout-matrix` | Construct exact runtime/model/loadout matrices |
| `lms-loadout-compare` | Compare quality and throughput without collapsing them into one score |
| `lms-hermes-bench` | Secret-safe Hermes intelligence suites |
| `lms-loadout-qualify` | Verify cross-component exact-loadout evidence |
| `lms-loadout-qualification-run` | Execute throughput and Hermes suites under one lock |
| `lms-loadout-qualification-attest` | Sign and verify qualification evidence |
| `lms-runtime-canary` | Validate, run, and verify transactional candidate lifecycle |
| `lms-runtime-canary-attest` | Sign and verify canary/rollback evidence |
| `lms-prompt-cache` | Record-only opaque prefix/KV metadata and payload registry |

The package does not install a command named `lms`, preserving the official LM Studio CLI namespace.

## 2. Common execution conventions

### 2.1 Input safety

Secure commands should require:

- absolute paths for executables and critical workspaces;
- exact source commit SHAs where source is part of the result;
- clean source trees or explicit immutable archives;
- nonsymlinked plan/config files where required;
- files not writable by group or world;
- explicit timeouts;
- environment variable names rather than secret values;
- stable run IDs and dedicated run directories.

### 2.2 Command representation

Lifecycle and benchmark subprocesses are represented as argv arrays. Secure entrypoints reject shell interpreters, `sh -c`, `bash -c`, PowerShell command strings, or other mechanisms that reinterpret a string as code.

A command record should retain:

```text
step name
resolved executable path
executable SHA-256
working directory
environment variable names inherited
argv with secret values excluded
timeout
start/end timestamps
exit status
stdout/stderr artifact references and hashes
process termination outcome
```

### 2.3 Process management

Subprocesses run in bounded process groups. Timeout or controller interruption initiates termination of the entire group rather than only the parent process. Cleanup records whether graceful termination, forced kill, or residual process detection occurred.

## 3. Fleet census model

A fleet census identifies every expected inference node and its policy state.

Representative node fields include:

```text
node_id
hostname and SSH alias
user
expected platform
expected role
benchmark state
required/deferred/excluded status
connection budget
workspace paths
observation command profile
```

Valid high-level states include:

- `benchmark_required` — must be executed for a complete run;
- `benchmark_deferred` — explicitly unavailable under reviewed policy, such as powered-off `joyner`;
- `excluded` — not an inference node, such as Raspberry Pi or iPhone devices;
- completed/failed states recorded in a particular run.

A run cannot silently omit a required node.

## 4. Fleet operator workflow

### 4.1 Controller preflight

The operator validates:

- census and environment file permissions;
- unique node identities and aliases;
- safe workspace location and free space;
- controller hostname, clock, and source state;
- known-hosts and SSH configuration availability;
- exact repository commit and clean tree;
- absence of conflicting active/stale run lock;
- expected command binaries and hashes.

### 4.2 Locking

The controller lock records at least:

```text
run_id
controller hostname
boot_id
pid
created_at
source commit
```

A lock is active only when host, boot ID, and PID evidence match a live process. Stale recovery must be explicit and recorded. A PID reused after reboot does not make the old lock valid.

### 4.3 Per-node preflight

Remote preflight verifies:

- strict SSH host-key match;
- expected remote hostname/node identity;
- clock sanity and timestamp capture;
- required directories and free disk;
- source or payload destination safety;
- required runtime/CLI availability;
- absence of conflicting benchmark process;
- bounded connectivity and command latency.

### 4.4 Remote execution

The operator invokes a reviewed remote entrypoint with fixed arguments. Transient network failures may be retried only under an explicit retry policy. Semantic failures, trust failures, wrong-host failures, or unsafe state abort without improvisation.

### 4.5 Collection

Remote artifacts are transferred to a temporary local path. The controller verifies expected size/hash and archive containment before atomic rename to the final run location. Partial SCP output never becomes the authoritative archive.

### 4.6 Complete-fleet gate

Postflight walks every census entry and verifies that each required node has a complete, verifiable result. Deferred and excluded nodes must have explicit reasons. The fleet manifest records overall success only when coverage and all configured gates pass.

## 5. Run directory model

Each workflow owns a dedicated run directory. Common files include:

```text
<normalized-input>.json
<run-state>.json
<manifest>.json
logs/
artifacts or collected archives
optional signatures/
```

All manifest paths are relative to the run root. Verification rejects:

- absolute paths;
- `..` escape;
- duplicate logical paths;
- missing files;
- symlinks where forbidden;
- directories where a file is expected;
- size mismatch;
- SHA mismatch;
- unsafe permissions where policy requires protection;
- unmanifested critical state files.

## 6. Benchmark case model

A throughput case should bind:

```text
case_id
endpoint and route
model/loadout identity
request or prompt fixture identity
context/input size
max output tokens
streaming mode
concurrency
warmup count
measured repetitions
timeouts
acceptance thresholds
```

Raw case results retain request timestamps, status, latency, TTFT, input/output token counts where available, throughput, and bounded error classification. Aggregation computes statistics only over compatible successful samples and records failed sample counts separately.

Quality and throughput remain separate dimensions unless a downstream reviewed policy combines them.

## 7. Model and loadout fingerprinting

### 7.1 Loadout inputs

An exact loadout fingerprint should cover all fields that may materially change behavior, including:

- model artifact SHA-256 and size;
- quantization;
- runtime/engine identity and version;
- executable SHA-256;
- runtime arguments and context settings;
- tokenizer/chat template/adapter identity where applicable;
- endpoint contract;
- hardware/node identity;
- inventory version;
- benchmark cases version;
- LMS repository commit;
- Hermes repository commit;
- relevant environment names or non-secret configuration.

### 7.2 Canonicalization

Fingerprint inputs are normalized into deterministic JSON:

- UTF-8;
- sorted object keys;
- stable number/string representation;
- no timestamps, paths, or volatile values unless intentionally part of identity;
- no secret values.

SHA-256 of canonical bytes becomes the loadout fingerprint. All component evidence stores the exact lowercase fingerprint.

## 8. Hermes intelligence suites

### 8.1 Base suite

The base suite evaluates tool use, instruction following, reasoning, code/repository tasks, and task completion under reviewed fixtures. Each case records machine-readable outcome and bounded logs.

### 8.2 Context-pressure suite

The context suite exercises the same loadout under larger or more distracting context, validating that tool use and task quality do not collapse beyond thresholds.

### 8.3 Secret handling

Endpoint API keys are identified by environment variable name. The launcher inherits the value but does not place it in argv, normalized input, logs, or manifest metadata. Child environment is restricted to reviewed names plus required execution basics.

## 9. One-run exact-loadout qualification

### 9.1 Inputs

The orchestrator receives:

```text
reviewed loadout JSON
inventory CSV/JSON
throughput cases
model artifact path
local endpoint
API key environment name
LMS source path/branch/commit
Hermes source path/branch/commit
protected workspace
```

### 9.2 Preflight

- verify files and permissions;
- verify model artifact hash;
- verify exact clean source commits;
- verify endpoint is loopback/private and expected;
- verify no shell command injection surface;
- acquire qualification lock;
- create unique run directory atomically.

### 9.3 Execution

```text
throughput suite
  -> base Hermes suite
  -> context-pressure Hermes suite
```

The orchestrator may stop on a blocking failure while still sealing partial failure evidence. Each sub-suite writes its own state and manifest under the qualification root.

### 9.4 Cross-check

The qualification verifier independently confirms:

- every suite succeeded when `--require-success` is used;
- all suites contain the same loadout fingerprint;
- artifact and source identities match normalized qualification inputs;
- required case coverage is complete;
- no evidence path escapes or changes;
- summary values recompute from raw results where required.

The qualification state records success separately from attestation status.

## 10. Runtime canary plan

### 10.1 Plan structure

A normalized canary plan binds:

```text
canary_id
loadout_fingerprint
working directory
environment_names[]
lock policy
snapshot command
start_candidate command
candidate_health command
qualification command or qualification input
soak_probe command and cadence
stop_candidate command
rollback command
rollback_health command
per-step timeouts
soak thresholds
```

Every executable path is absolute and hashed during validation. Plan fingerprint is computed over normalized secret-free plan content.

### 10.2 Prohibited content

Validation rejects:

- secret-looking flags or values in argv;
- shell interpreters or command strings;
- relative executable paths;
- missing executables;
- unsafe working directories;
- writable/symlinked plan files;
- invalid loadout fingerprints;
- unbounded or nonpositive timeouts;
- soak gates that cannot be evaluated from the declared probe contract.

## 11. Runtime canary state machine

```text
VALIDATED
  -> LOCKED
  -> SNAPSHOT_COMPLETE
  -> CANDIDATE_STARTED
  -> CANDIDATE_HEALTHY
  -> QUALIFIED
  -> SOAK_PASSED
  -> CANDIDATE_STOPPED
  -> ROLLBACK_COMPLETE
  -> ROLLBACK_HEALTHY
  -> SEALED_SUCCESS
```

Any failure after `CANDIDATE_STARTED` transitions to recovery:

```text
failure
  -> candidate_stop_attempted
  -> rollback_attempted
  -> rollback_health_attempted
  -> SEALED_FAILURE
```

The state file independently records:

```text
success
failed_step
rollback_attempted
rollback_succeeded
candidate_stop_succeeded
step records
loadout_fingerprint
plan_fingerprint
run/canary IDs
```

Rollback success never changes `success=false` for a failed experiment.

## 12. Soak subsystem

### 12.1 Probe output

The probe's final stdout line must be one JSON object with permitted fields:

```json
{
  "ok": true,
  "latency_seconds": 0.42,
  "rss_bytes": 12884901888,
  "temperature_c": 72.5,
  "tps": 24.1,
  "ttft_seconds": 0.18
}
```

Output is size-bounded. Only approved metrics are retained in `soak-samples.jsonl`.

### 12.2 Sampling

The canary records monotonic sample index, timestamp, command result, and parsed metrics. Probe errors produce failed samples rather than being silently discarded.

### 12.3 Gate computation

The verifier recomputes from raw samples:

- sample count;
- success rate;
- maximum consecutive failure streak;
- p95 latency;
- baseline and terminal RSS, absolute/relative growth;
- baseline and terminal TPS, terminal/baseline ratio;
- maximum temperature;
- optional TTFT statistics.

Missing required metric values fail the associated configured gate. Summary values are not trusted without raw-sample recomputation.

## 13. Rollback semantics

The snapshot step captures enough operator-owned state to restore the previous runtime. Rollback wrappers should use fixed system APIs or typed lifecycle tooling, not arbitrary shell input.

A valid successful canary requires:

- candidate stopped;
- rollback command succeeded;
- previous runtime health check succeeded;
- restored identity matches expected previous state where the plan defines identity verification;
- no candidate process remains;
- all evidence sealed.

Rollback failure is a blocking admission defect and must be explicit in attested evidence.

## 14. Manifest construction

### 14.1 Included artifacts

The manifest includes all critical normalized input, state, raw result, sample, command record, and log files intended for verification. Each entry contains:

```text
relative path
size in bytes
SHA-256
artifact role/type
```

### 14.2 Atomic sealing

State files are written to temporary files, flushed, fsynced where required, and atomically replaced. The manifest is created only after execution is terminal. Once signed, the run directory should be treated as immutable.

### 14.3 Verification

Verification independently reads the manifest and filesystem. It does not trust cached state. It revalidates path containment, file type, size, hash, normalized identity, gate computations, and success/rollback conditions.

## 15. OpenSSH attestation

### 15.1 Namespaces

Separate namespaces distinguish evidence classes, for example:

- fleet observation;
- exact-loadout qualification;
- runtime canary.

The exact namespace constants in code are authoritative.

### 15.2 Signed payload

The payload binds critical fields and exact manifest bytes, including:

```text
run_id
workflow/canary/qualification identity
loadout_fingerprint
plan fingerprint where applicable
success
rollback state where applicable
manifest SHA or canonical manifest bytes
```

### 15.3 Verification

Verification invokes OpenSSH signature verification against:

- external allowed-signers file;
- expected identity;
- expected namespace;
- exact signed payload.

It records signer key fingerprint and rejects unknown identity, namespace mismatch, modified manifest, or invalid signature.

## 16. Prompt-cache registry

### 16.1 Metadata

The registry may store:

```text
opaque prefix ID
model artifact/quantization/runtime compatibility
engine cache format
context parameters
privacy scope
payload content hash and size
residency/path reference
created/expires timestamps
candidate hit/miss observations
```

### 16.2 Payload store

Engine-native payloads use an atomic content-addressed local store. Path derivation comes from content hash, not user input. Writes use temporary files and atomic promotion.

### 16.3 Safety boundary

Candidate lookup is observational. The subsystem does not inject restore flags, load cache bytes into a runtime, skip prompt tokens, change route selection, or calculate realized savings without a separate experiment.

## 17. Configuration and secrets

Protected operator configuration commonly lives under:

```text
~/.config/lms-fleet/
/secure/plans/
/secure/keys/
/secure/policy/
/secure/lms-*-runs/
```

Environment files and private keys should be mode `0600`. Allowed-signers policy should be maintained separately from run evidence. Public verification keys may be distributed according to OpenSSH policy, but private signing keys must never enter run directories.

## 18. Error classification and retry

Retries are limited to classified transient failures such as bounded network interruption. The workflow must not retry:

- SSH host-key mismatch;
- wrong hostname or node identity;
- dirty or wrong source commit;
- unsafe file permissions;
- invalid plan/schema;
- command policy violation;
- loadout mismatch;
- failed acceptance gate;
- archive containment violation;
- signature failure.

Retry attempts and backoff are recorded. Exhausted retries produce terminal failure evidence.

## 19. Observability

Run state and manifests provide the primary observability. Human-readable summaries may be generated, but verification consumes machine-readable evidence. Important operator fields include:

- coverage by node;
- exact source and executable hashes;
- command durations and exit states;
- transfer and archive verification;
- throughput/latency/TTFT distributions;
- Hermes case pass/fail coverage;
- soak gate calculations;
- rollback attempt and health;
- manifest and signer fingerprints.

## 20. Test strategy

### Unit tests

- canonicalization and hashing;
- path and permission validation;
- command policy and secret-flag rejection;
- lock ownership and stale recovery;
- process timeout/group cleanup;
- archive containment;
- manifest construction and tamper rejection;
- benchmark aggregation;
- exact-loadout cross-checking;
- soak gate recomputation;
- rollback state handling;
- OpenSSH payload construction and verification adapters;
- prompt-cache content addressing.

### CI integration tests

- synthetic fleet operator runs;
- local SSH or mocked remote execution contracts;
- qualification orchestration over deterministic fixtures;
- canary success and each failure stage;
- cancellation and timeout cleanup;
- stale lock and reboot/boot-ID scenarios;
- disk-pressure and malformed artifact cases;
- signature/allowed-signers verification.

### Physical validation

Separate operator drills are required for real host-key deployment, network partition, disk exhaustion, controller interruption, reboot, power loss, process leakage, thermal/TPS degradation, memory growth, and rollback health on actual nodes.

## 21. Change rules

Changes to fingerprint inputs, command policy, manifest schema, signer payload, soak calculations, rollback success semantics, fleet census completeness, or evidence directory safety require updates to this LLD and the HLD plus backward-compatibility or migration notes. No change may turn evidence generation into autonomous admission.

# Fleet evidence provenance and execution safety

Physical benchmark evidence is useful only when it can be tied to an exact reviewed source tree and an intact collected archive. The Tier-1 rollout therefore fails closed on source drift, dirty checkouts, overlapping runs, unbounded remote execution, and incomplete collection.

## Exact reviewed source commit

Before preparing the private environment file, record the current reviewed LMS branch head:

```bash
git checkout full-auto-reconciliation-20260730
git pull --ff-only origin full-auto-reconciliation-20260730
git rev-parse HEAD
```

Copy the full 40-character SHA into:

```text
LMS_EXPECTED_COMMIT=<full SHA>
```

Every Tier-1 node must be on the configured branch, at that exact commit, with a clean tracked and untracked working tree. `lms-fleet-rollout validate` rejects a missing or abbreviated commit before SSH. The remote run rejects:

- a detached or different branch;
- a different commit, including a branch that advanced after controller review;
- modified tracked files;
- staged changes;
- untracked files in the repository.

The remote run writes `source_control.json` with:

- node and run identity;
- expected and actual branch;
- expected and actual commit;
- clean-tree status;
- a SHA-256 fingerprint of the configured Git origin without recording the URL;
- Python and `lms-agent-bench` versions;
- a canonical `source_fingerprint`;
- `admission.admitted=false`.

The release gate recomputes the source fingerprint and rejects any mismatch.

## Per-node rollout lock

Each node uses an atomic directory lock under:

```text
~/.local/state/lms-fleet/locks/<node-id>.lock
```

Only one rollout may hold a node lock. The lock contains a small `owner.json` with the run ID, node ID, hostname, and process ID. The EXIT trap removes a lock acquired by the current run after evidence packaging.

When a run reports an existing lock, do not delete it immediately. First verify that no rollout or benchmark process is active and preserve any diagnostic archive. Remove a stale lock only after that inspection:

```bash
cat ~/.local/state/lms-fleet/locks/<node-id>.lock/owner.json
ps aux | grep -E 'lms-fleet|fleet_bench|llama-server'
rm -rf ~/.local/state/lms-fleet/locks/<node-id>.lock
```

## Bounded execution and collection

The Tier-1 template defaults to:

```json
{
  "remote_timeout_seconds": 7200,
  "scp_timeout_seconds": 300
}
```

A remote timeout is recorded as exit code `124` with `timed_out=true`. An SCP timeout records `scp_returncode=124` and `scp_timed_out=true`. The installed `lms-fleet-rollout` command returns nonzero when either remote execution or required artifact collection is incomplete.

Increase a timeout only in a private resolved configuration and record the reason with the physical evidence. Do not remove the bounds.

## Bundle and outer archive chain

The remote bundle manifest records:

- `node_id`;
- `run_id`;
- original `remote_exit_code`;
- `source_fingerprint`;
- every artifact path, size, and streaming SHA-256;
- a canonical `bundle_fingerprint` over the manifest core.

After SCP, `rollout_results.json` records:

- collected archive path;
- outer archive byte size;
- outer archive SHA-256;
- remote and SCP timeout state.

`lms-fleet-gate` verifies the outer archive size and SHA-256 before opening it, then verifies tar member safety, manifest membership, every internal hash, the bundle fingerprint, the source provenance, observation evidence, and sweep evidence.

A failed remote run may still produce a diagnostic archive. It remains non-promotable because the manifest preserves the nonzero remote exit code.

## Promotion boundary

Passing provenance and archive gates means the evidence is intact and reviewable. It does not admit, route, install, restart, or expose a runtime. Live authority remains external and must independently verify physical identity, model identity, shared capacity, access paths, sustained stability, evidence freshness, and rollback.

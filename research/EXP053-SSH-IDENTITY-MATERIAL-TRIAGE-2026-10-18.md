# EXP-053 controller credential-material triage (research only) — October 18, 2026

**Status: NO_GO_RESEARCH_ONLY.** This continues the October 18 controller-only
configured-identity overlap audit; it does not replace independently governed
witness custody or authorize any model execution.

## Why the v1 path-overlap result is incomplete

`ssh -G` reports *configured* IdentityFile paths. Multiple SSH aliases may show
the same path without proving that a corresponding local private key exists,
that it was presented for authentication, or that it permits privileged or
rollback-capable operations. Conversely, different path spellings may reference
the same local file or an SSH agent may provide actual identity material.

Version 2 retains privacy-safe configured-path comparison and adds bounded
controller-local observation of regular identity-path file metadata via
`lstat()` (no key bytes or public-key material read). Matching device+inode
numbers are counted as a second kind of overlap, even across different
hard-linked paths. Symlinks, missing files, special files, nonabsolute paths
and unresolved `ssh -G` tokens remain **unknown**, not independent. The
controller's SSH agent *socket presence* is observed; no agent identities are
read and no authentication mechanism is asserted.

The modified source still evaluates existing SSH configuration through `ssh -G`,
which may invoke a preconfigured `Match exec` local command. Only trusted SSH
configuration is appropriate. No new remote sessions, listeners, keys, services,
files, or model processes are created. The script always returns CLI exit code
2 and sets `scientific_execution_authorized=false`.

## Physical controller observations

Observed on `x1-370` against three existing aliases, October 18, 2026:

| Candidate | Configured identity paths | Same configured path as other candidates | Locally statable regular identity paths | Inode-overlap observation | Controller SSH agent socket |
|---|---:|---|---:|---|---|
| lenovo | 1 | Yes | 0 | Unknown | Present |
| destroyer | 1 | Yes | 0 | Unknown | Present |
| xwing | 1 | Yes | 0 | Unknown | Present |

The effective controller SSH configurations also report `IdentitiesOnly=yes`.
An existing controller agent socket does not prove a specific key was used;
missing local IdentityFile observations are **not** evidence that the hosts
have independent custody. No SSH authentication trace was gathered. Shared
configured paths also do not prove the remote account has sudo or root access.

## Reproducible controls

- `research/exp053_ssh_custody_overlap.py` emits privacy-safe schema
  `exp053-controller-ssh-custody-overlap-v2`, no raw path, account name,
  remote address, key fingerprint, private key or agent identity.
- `tests/test_exp053_ssh_custody_overlap.py` tests hard-linked synthetic
  paths, non-following symlink behavior, non-existent paths, mismatched
  controller usernames, malformed data, SSH failures, and NO-GO output.
- This only broadens **negative** triage. It is never an independent approval
  or an authoritative trust graph.

## Next hard custody gates

An independently accountable custodian must attest control over witness
administration, snapshots/backup restoration, signer root, high-water state,
revocation and recovery. Perform human-approved, nonproduction negative access
and rollback tests across scorer/operator/witness trust domains, with immutable
out-of-band evidence. A static SSH configuration or agent-socket indicator
cannot satisfy these gates. Independently signed human source adjudication and
model-byte/scorer code custody must still be completed before model scoring.

**Do not deploy a witness, promote a PR, run EXP-053 model forwards, mutate any
production service, or write NAS/Neo4j based on these observations.**

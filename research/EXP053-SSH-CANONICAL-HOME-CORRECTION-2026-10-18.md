# EXP-053: corrected controller-local identity observation (2026-10-18)

**Decision: NO_GO_RESEARCH_ONLY.** This document corrects the identity-file
observation in `EXP053-SSH-IDENTITY-MATERIAL-TRIAGE-2026-10-18.md`.

## Correction and root cause

The earlier `exp053_ssh_custody_overlap.py` used Python
`os.path.expanduser()` for paths returned by `ssh -G`. On the research host,
Fleet Commander sets the process `HOME` to an isolated directory that differs
from the OS account home returned by `pwd.getpwuid(os.getuid()).pw_dir`.
Consequently, existing configured `~/.ssh/...` identity files were incorrectly
reported as not locally statable. **Those earlier zero-file observations are
superseded**; they did not establish absent local key material.

The updated `_local_file_identity()` resolves **only** `~/` against the
canonical OS account home, using metadata-only `os.lstat()`; absolute paths
continue to be supported. Unknown `~other`, nonabsolute, tokenized, missing,
nonregular and symlinked paths remain UNKNOWN. No key bytes, fingerprints,
private material or SSH agent identities are accessed or published.

## Current observed results (controller-local only)

| Alias | Configured identity path count | Regular identity paths observed | Overlap by path | Overlap by observed local inode |
|---|---:|---:|---|---|
| lenovo | 1 | 1 | Yes | Yes |
| destroyer | 1 | 1 | Yes | Yes |
| xwing | 1 | 1 | Yes | Yes |

SSH agent socket **present**, but actual SSH key presentation was **not**
verified. Every result continues to explicitly return
`scientific_execution_authorized=false` and
`independent_administration_approved=false`; CLI exit status 2.

The three aliases refer to the **same actual controller-local inode**, a
stronger overlap witness than the previous configured path-only result.
However, this observation cannot determine effective root/sudo or restore
privileges on the remote nodes, independently held signing keys, or which
identity an SSH authentication handshake uses. It provides negative custody
triage, not proof of administrative compromise or authorization to create a
scientific witness.

## Reproduction/acceptance

- `python3 research/exp053_ssh_custody_overlap.py lenovo destroyer xwing`
  (expected exit code **2**; source-free JSON with overlap only).
- `python3 -m pytest -q tests/test_exp053_ssh_custody_overlap.py`
  tests both simulated isolated HOME/canonical passwd home, unknown tilde user
  rejection and established source-only negative cases.
- Do not use this probe as a signing, admission or custody approval. Real
  independent custodian review, privileged restore-path segregation, a
  durable externally governed high-water witness and >=40 independent human
  adjudications remain separately required before EXP-053 model forwards.

**Scope:** controller-local read-only SSH configuration and file *metadata*
only; no remote writes, keys, service changes, inference, deployment, Git
merge, NAS or Neo4j writes.

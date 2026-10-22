# EXP-053 — Controller SSH credential overlap audit, 2026-10-18

**Authority decision: NO_GO_RESEARCH_ONLY.** This is a source-free,
controller-local NEGATIVE screening tool, not a witness deployment or a
scientific authorization pathway.

`research/exp053_ssh_custody_overlap.py` performs `ssh -G` using explicitly
named, existing SSH aliases. It never connects to another node, copies files,
creates a key or starts a witness service. It checks only effective controller
SSH usernames and configured *identity-file path* intersections, without
printing any private key, certificate, hostname, IP, account name, remote
commands or file path. Every report sets `independent_administration_approved`
and `scientific_execution_authorized` to `false`, and its CLI returns exit
status `2` even when configuration parsing succeeds.

**Important operational caveat:** SSH config evaluation itself can trigger a
pre-existing `Match exec` local command. Run this tool only with trusted SSH
configuration on the authorized controller. Inspect the local config first if
that possibility is not understood. This was run with the existing authorized
configuration and no remote sessions or services were changed.

## Observed October 18 on x1-370

| Explicit alias | SSH configuration | Matches controller login | Identity-file path count | Shares configured path with another candidate |
|---|---|---|---:|---|
| lenovo | Parsed | Yes | 1 | **Yes** |
| destroyer | Parsed | Yes | 1 | **Yes** |
| xwing | Parsed | Yes | 1 | **Yes** |

This is reproducible, non-secret evidence of controller-side credential-path
**overlap**, not proof of which private key SSH actually used, actual root/sudo
permissions, key uniqueness, key custody or effective rollback authority.
Different identity-file paths would **not** establish independent custody;
symlinks, agents, forwarded privileges and a common administrator could still
collapse the same trust boundary. Shared configured paths also do not by
themselves prove administrator privileges on any target.

## Next explicit custody gates (require independent human/operator)

- Document the control-plane and physical privilege graph (research scorer,
  human owners, root/sudo, hypervisors, backup/snapshot/restore, signing keys,
  certificate authorities and network administration).
- An independent human custodian must approve an administration boundary
  beyond the x1-370 operator, backed by actual privileged-access evidence and
  a key-custody/recovery policy. A changed SSH config alone is not evidence.
- Negatively test that scorer credentials cannot impersonate the witness,
  access its signing key or update/roll back its durable high-water, including
  privileged snapshot/recovery paths, in an explicitly approved disposable
  nonproduction environment.
- Establish independently published append-only checkpoint observations and
  inspect native monotonic sequence / replay / fork resistance across resets.
- Independently adjudicate >=40 article-disjoint items, sign the frozen
  pre-inference fixture digest, and retain separately reviewed model-byte
  custody before **any** EXP-053 forward pass.

The existing external high-water adapter authority pins remain intentionally
empty. Do not install a real witness, expose a new listener, create keys, change
runtime routing, deploy, merge, perform inference, write NAS/Neo4j, or mistake
this audit for research/production approval.

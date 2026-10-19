# EXP-053 — sealed packet and signed-review private JSON reads

**October 19, 2026 — synthetic-only integrity hardening. NO_GO_RESEARCH_ONLY.**

## Reproduced gap

The scientific `verified_admission` function had four private JSON entry
points: `packet.json`, `sealed_exam_key.json`, reviewer A/B receipts and the
optional adjudicator receipt. It previously read them with `Path.read_text()`
plus default `json.loads()`, which follows symlinked path components, places
no size bound on private content, and silently keeps only the last duplicate
JSON object key. A disposable **synthetic** symlink and duplicate
`packet_sha256` key reproduced the problem. The source-pinned packet digest
and signed-review validators remain necessary; this is a path/content-trust
weakness, not evidence of forged real reviewer approval.

## Repair

All private admissions now route through `_read_private_admission_json`.
This bounded reader uses the existing secure Linux no-symlink-ancestor O_PATH
walk to open the final regular file with O_NOFOLLOW|O_NONBLOCK, enforces owner
uid/mode, exactly one link and size 1..8,000,000 bytes, reads via a single
file descriptor, verifies before/after device/inode/size/mtime/ctime, and
reopens the final lexical path to confirm it still names the same inode.
It refuses files in the public source checkout. It uses strict UTF-8 decoding,
rejects duplicate fields at any depth, invalid numeric constants, nonfinite
floating-point exponent overflow, incomplete JSON and non-object root types.
Rejected content or absolute paths are never printed or admitted as evidence.

Synthetic regression tests exercise safe reads, duplicate object keys,
nonfinite JSON, symlinks in the final/ancestor path, hardlinks, world-readable
files, unknown owner, input exhaustion, FIFO, missing secure OS flags,
interruptions, identical-byte inode replacement, ancestor swaps into a
synthetic public repo and refusal with the owner pin blank. No actual licensed
RACE passage, private signer key, human response or model forward was touched.

## Limits / outstanding governance

These filesystem controls are an *additional* denial boundary, not external
observer authority. Hostile same-UID code may still swap path components
after the last identity recheck; a privileged observer, external independent
checkpoints and protected custody roots are separately required. The
private files must be stored on a filesystem capable of ownership/mode
controls; permissive exFAT mounts may correctly HOLD instead of admitting.
The source SHA, manifest/packet SHA, owner-approved roster SHA, sealed exam
answer SHA, Ed25519 A/B key custody, distinct adjudicator and 40-item minimum
are unchanged. The live owner roster SHA is unset. The research preflight
still reports zero independently accepted human reviews.

Reproduce with only invented bytes:
`python3 -m pytest -q tests/test_exp053_private_admission_json.py`

**NO_GO_RESEARCH_ONLY.** No merge, publication of private source, deployment
or scientific inference follows from this patch.

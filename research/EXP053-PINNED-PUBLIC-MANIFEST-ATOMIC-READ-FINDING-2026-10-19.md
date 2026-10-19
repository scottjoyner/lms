# EXP-053 — public-manifest hash/read substitution detected and refused

**2026-10-19. Synthetic-only source-integrity finding. NO_GO_RESEARCH_ONLY.**

The prior admission bootstrap performed `filehash(MANIFEST)` and later
`json.loads(MANIFEST.read_text())` as two independent path accesses. With a
fully synthetic preregistered public manifest and a deliberately patched
`filehash()` that replaced the file after hashing, `verified_admission`
accepted a `packet_sha256` from the replacement manifest. It handed that
packet SHA to the separately pinned roster-loader before any licensed source
or private human evidence was accessed. This demonstrated a time-of-check/
time-of-use defect in the public trust anchor, **not** a demonstrated
successful false human admission. The independently pinned private roster
and other gate checks remain important additional protections.

## Code repair

`_read_pinned_public_manifest` opens the preregistered public source-free
file through a Linux `O_PATH|O_DIRECTORY|O_NOFOLLOW` ancestor walk and an
`O_RDONLY|O_NOFOLLOW|O_NONBLOCK` leaf. It requires a regular, single-linked
nonempty file <=1 MiB; reads through one fd; and compares before/after
`fstat` device/inode/size/mtime_ns/ctime_ns. It verifies the SHA-256 of
precisely the bytes in that buffer against the exact preregistered pinned
manifest digest **before** parsing. It then checks the lexical pathname
and final named inode, parses strict UTF-8 JSON with no duplicate keys at
any depth, NaN, Infinity, overflowed floats, or nonobject roots, and only
then supplies the packet digest to the owner-approved roster. Missing secure
kernel file flags or any mismatch yield a sanitized HOLD.

A changed final inode during a synthetic read is refused even when the
original pinned bytes were read successfully. Synthetic tests also cover
missing OS file flags, symlinks, hardlinks, FIFO, oversized and malformed
manifest bytes, duplicate keys, wrong exact digest, unchanged pinned public
manifest, and the original owner-pin early refusal without file reads.

This is not a rollback-resistant independently administered witness and
cannot protect against arbitrary privileged tampering after the last path
identity check. No licensed RACE passage, private signing key, signed human
receipt or model forward was read by these tests. No scientific authority
is granted and all owner/weight approval roots remain unset.

Reproduce on Linux:
`python3 -m pytest -q tests/test_exp053_public_manifest_atomic.py`

**Decision: NO_GO_RESEARCH_ONLY; no merge, deployment or scientific inference.**

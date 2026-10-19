# EXP-053 — private admission JSON trust boundary

**Frozen before implementation, October 19, 2026. NO_GO_RESEARCH_ONLY.**

Reproduced solely using disposable synthetic JSON: the scientific admission
entrypoint reads its private packet, sealed answer key and signed A/B/optional
third-party review files via unbounded path-following `Path.read_text()` and
`json.loads()` with last-wins duplicate-field interpretation. A symlink
redirected the read and two `packet_sha256` keys collapsed silently. Other
components (signer draft, roster) already fail closed for these inputs.

## Narrow implementation requirements

- Replace exactly the private reads in `verified_admission` with one shared,
  deny-only `_read_private_admission_json()` routine. Do not modify the public
  manifest, source, model data, signing keys or upstream owner roster logic.
- Require Linux `O_PATH|O_DIRECTORY|O_NOFOLLOW|O_NONBLOCK` for safe ancestor
  walk using the existing `_open_no_symlink_ancestors()` helper. Refuse
  symlinked path components. Enforce outside-the-public-repo placement and
  private, effective-user-owned regular files with one link and zero mode
  bits for group/other. The caller must independently keep the private
  directory and all ancestor custody safe from same-UID adversaries.
- Cap a packet and every signed JSON input at <= 8,000,000 bytes (same as
  the existing offline browser packet importer). Refuse zero-length files,
  directories, FIFO, devices and oversized files. Open/read via one descriptor
  with bounded `os.read`; compare `(dev,ino,size,mtime_ns,ctime_ns)` before
  and after, and securely reopen the final path for identity confirmation.
  Deny if any path component changes, is outside the expected private
  provenance at final check, or is redirected to a different inode.
  Always close descriptors and never echo source/private bytes or paths.
- Parse strict UTF-8 JSON with duplicate-object-field denial at all nesting
  depths, no non-finite constants or 1e999 float exponent overflow, no
  malformed content or non-object root. No content from a rejected file may
  become trusted review, packet or answer-key evidence.
- Test only with disposable synthetic JSON, including static symlinks,
  symlinked ancestors, hardlinks, modes/owners, malformed and duplicate JSON,
  write/race failures, file swaps, FIFO, oversize, and a normal private file.
  Test both empty owner pin early-HOLD and a source-free direct helper. Do
  not load RACE text, real signed forms, private keys or model runtimes.
- The owner-approved roster SHA and model weight SHA remain unset. Source
  hash, packet SHA, sealed exam answer SHA, reviewer signatures, 40-item
  scientific threshold and independent custody requirements remain intact.
  No code merge, release deployment or model forward is authorized.

A successful synthetic file read is NOT an independently enrolled human
reviewer or authority to admit scientific results. **NO_GO_RESEARCH_ONLY.**

# EXP-053 owner roster directory fd handoff — error cleanup

**October 19, 2026. Local synthetic negative tests. NO_GO_RESEARCH_ONLY.**

## Reproduced issue

The new secure owner-roster ancestor walk opens each directory descriptor
before closing the previous one. Under an injected `os.close(parent)` error,
the newly opened directory descriptor was stranded, even though the function
failed and no private roster was accepted. A separate injected close failure
after opening the roster leaf could strand the last parent directory fd. A
synthetic reproduction observed one leaked fd in the prior implementation.
This is a resource leak, not a cryptographic or human enrollment bypass.

## Repair and limits

The walk now cleans up `next_dir` if release of the previous parent fails;
normal success still transfers ownership of that fd. If the final parent
release fails after opening the file, it closes the unreturned leaf and
conditionally retries final parent cleanup only when `fstat` shows the same
device/inode captured immediately before the failing close. A blind close
retry would risk closing a reused fd; the identity guard reduces but cannot
eliminate concurrent same-user fd-reuse races and does not provide absolute
assurance after a kernel-close error. All errors still cause refusal.

Synthetic pytest cases inject intermediate close failure, KeyboardInterrupt,
and final-parent release failure. They confirm no captured open descriptors
remain and that no reviewer file is accepted. Normal valid synthetic roster
loading, symlink rejection, owner and SHA pin checks remain unchanged. Tests
did not read any licensed RACE text, private reviewer form or real signing key.

No actual independent human technical review, signed review, approved owner
roster, external witness, 40 co-reviewed articles, or model-forward authority
exists as a result of this repair. NO_GO_RESEARCH_ONLY remains in force.

Run: `python3 -m pytest -q tests/test_exp053_roster_ancestor_race.py`

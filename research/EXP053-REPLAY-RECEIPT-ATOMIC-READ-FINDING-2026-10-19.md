# EXP-053 — verified scoring replay pinned metadata hash/read race

**October 19, 2026. Only source-free synthetic metadata. NO_GO_RESEARCH_ONLY.**

## Evidence

The scoring-replay helper `admitted_weight_inventory` computed
`filehash(WEIGHT_REPORT)` against a separately approved checkpoint inventory
receipt SHA and then independently parsed `WEIGHT_REPORT.read_text()`. An
experiment with two **invented internally consistent** receipts A/B changed
one synthetic checkpoint shard SHA and rederived the synthetic per-model and
aggregate receipts. A scripted file substitution *immediately after hashing
A* caused the old verifier to return a model checkpoint identity from B,
despite B's file bytes not matching the pinned SHA of A. **No neural weight,
safetensor data, model output or licensed RACE source was touched.**

The actual owner weight receipt pin remains `WEIGHT_REPORT_SHA=""` and the
owner's independent human roster pin remains empty. Thus this is a potential
future trust-anchor TOCTOU defect, **not a live forged model approval**.

Additionally, `require_sources` parsed the preregistered source-free tokenizer
report and public cohort manifest by a separate second path read after their
hash check, and read caller-selected private `packet.json` using permissive
path-following unbounded JSON. That duplicated unsafe accesses already fixed
in the admission path by previous PRs.

## Narrow fix

A new `_read_pinned_source_free_receipt` requires the exact canonical pin and
Linux no-symlink-ancestor descriptor walk; regular nonempty single-linked
source-free public metadata <=1 MiB; stable `(dev,ino,size,mtime_ns,ctime_ns)`
and final named inode; one bounded fd buffer SHA-checked before strict UTF-8
JSON parsing; duplicate/nonfinite/overflow and nonobject denial. The weight
inventory validates exactly the resulting pinned object. The tokenizer
preflight also uses the same reader; the public cohort manifest uses the
previous `_read_pinned_public_manifest` source-free reader, while selected
private packet bytes use `_read_private_admission_json` with owner-only mode,
uid and path requirements. Other tokenizer, receipt, checkpoint shard,
execution-attestation, operator-run and experimental safeguards are unchanged.

The regression suite uses fake model identities and never opens actual weight
files. Negative cases include swapped public metadata during fd read,
invalid pins, changed digest, duplicate nested fields, nonfinite constants,
symlinks, hardlinks, FIFO, oversized files, missing safe kernel flags, and
an explicit early-HOLD when the real weight SHA remains unset. An ordinary
invented inventory and real **source-free** pinned tokenizer receipt still
validate.

```
python3 -m pytest -q tests/test_exp053_verified_replay_atomic_receipts.py
```

No scientific review was admitted, no real scorer run or model forward was
performed, no owner/weight pins were set, and no code was merged or deployed.
Independent human technical review and external operator/witness custody
remain mandatory. **NO_GO_RESEARCH_ONLY.**

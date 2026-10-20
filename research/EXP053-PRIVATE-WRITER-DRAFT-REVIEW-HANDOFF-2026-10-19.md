# EXP-053 — independent source review of private artifact output safety

This is a stacked draft technical source review based on exact PR #39 head
`0a473861fc8306728d3a62ab93919aed3513a356` (green hosted CI).
Only source/test/doc files and invented synthetic fixtures are in scope.
The private scientific packet, sealed exam answer, blinded human review
forms and eventual admission receipt require securely located output.

The stacked PR's old implementation created the file before setting mode
0600 and followed symlinked ancestor paths. The full local unified branch
already contained a descriptor-bound 0600/exclusive readback writer, which
this draft exposes for independent review; the new change also opens each
output parent through a kernel-enforced nonfollowing component walk and
checks owner UID before writing any output bytes.

Please review: O_PATH/O_DIRECTORY/O_NOFOLLOW requirements, parent-fd
lifecycle, same-UID path swaps, 0600-from-birth mode, `O_EXCL` leaf open,
partial-write/fsync/inode readback, error cleanup without removing unrelated
files, and signer integration behavior. The tests contain ONLY fake JSON
and ephemeral synthetic signing keys. Do not export any genuine reviewer
response, private key or licensed RACE passage.

```
python3 -m pytest -q tests/test_exp053_private_writer.py \
  tests/test_exp053_private_writer_ancestors.py
python3 -m pytest -q tests/test_exp053_*.py
python3 -m ruff check research/exp053_race_review.py \
  tests/test_exp053_private_writer_ancestors.py
```

This is a **draft technical source review**. It does not establish human
identity independence, out-of-band witness custody, 40 distinct co-reviewed
articles, owner-approved roster pin or model-weight approval. Do not merge,
deploy, rebuild licensed packets, create human signatures, or run inference.
**NO_GO_RESEARCH_ONLY.**

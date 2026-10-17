# EXP-053 read-only checkpoint byte-custody audit — 2026-10-17

**Research preparation, NOT human adjudication or permission to score/generate.**

The exact prospective protocol was frozen as
`EXP053-WEIGHT-ID-PREFLIGHT-SPEC-2026-10-16.md` SHA256
`b89f573acc2567b114e65746b3026e36636aee9add6c6568c0fac6319f620a81`.
The scanner's original strict resolved-path check rejected a pre-existing Hugging
Face hub-level content-addressed blob symlink under `hub/blobs/NN/<64-lowercase-hex>`.
The code was corrected to allow **only** an exact repo snapshot under
`hub/models--*/snapshots/<rev>` to resolve into its repository blob directory
or the specifically shaped shared hub blob path. Path traversal, unknown
targets and malformed shared content-addressed names remain HOLD. A synthetic
positive/negative test covers this arrangement.

The read-only scan then completed on research host `x1-370` with local files
only; it loaded no neural model, no GPU, no network and generated no logits.

| Source-bound model | Hash-verified weight files | Bytes | GiB |
|---|---:|---:|---:|
| K2-Horizon-3.7B | 36 | 10,116,554,080 | 9.422 |
| Qwen3.5-4B-Base | 2 | 9,319,828,056 | 8.680 |
| MiniCPM-V 4.7-1B | 1 | 3,109,054,480 | 2.896 |
| **Total** | **39** | **22,545,436,616** | **21.0** |

The exact source-free archive is
`EXP053-WEIGHT-IDENTITY-2026-10-16.json`, **SHA256**
`e95c771517869f2060b84c7a270bc3e59bd659500c5187d2227d2f94d00440b6`.
Its aggregate canonical checkpoint digest is
`62fb2c89fca1d64d6a154ecdf0712b97e79b8fc3c1eacacda1f3a98efb6d7462`.

**Independent second pass:** after the scanner completed, the system
`sha256sum` executable re-read each of the same 39 files and matched each
file's saved SHA256 **39/39** across all three checkpoint families.
The second implementation corroborates byte integrity but cannot by itself
prove immutable upstream downloads or an inference runner's future weight use.
The named models' tokenizer/config identities matched the earlier pinned
candidate-only tokenization receipt. The model-scoring runner has NOT run.

The archive holds only sanitized model identity metadata, sizes, file
basenames, SHA256 sums, and constant custody references—**no RACE passages,
private answer keys, reviewer details, tokenizer vocabulary, weights or
model predictions**. Archive unit tests verify source-free structure and the
stored archive hash; tests do not falsely claim to rehash gigabytes in CI.

**Non-negotiable separation of controls:** The archive checksum pinned by
the software audit test is archival integrity only. In the actual model-free
reporting entry point `WEIGHT_REPORT_SHA=""` remains intentionally **EMPTY**,
until an independent human operator reviews and approves the full real receipt
in a source-reviewed change. Likewise,
`APPROVED_REVIEWER_ROSTER_SHA256=""` remains EMPTY pending distinct real
human identities and manual signing-key enrollment.

**Outcome:** checkpoint byte hashing is complete and independently reproduced.
Owner-reviewed receipt admission, >=40 approved human cases and EXP-053
model forward measurements remain HOLD. PR remains DRAFT and unmerged; no
production routing, training, weight modifications, NAS/Neo4j operations,
human signoffs or scientific model-performance claims resulted.

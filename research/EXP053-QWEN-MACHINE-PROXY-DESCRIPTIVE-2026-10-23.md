# EXP-053 Qwen3.5-4B-Base — machine-only 64-case descriptive evidence

**Status:** complete **machine-proxy** analysis only, October 23, 2026. Not independently human-adjudicated, not a validated semantic-grounding claim, not evidence of pretraining non-contamination. No human GitHub reviewer or manual sign-off is needed to maintain this source-only report.

## Evidence and scope

The private source-locked runner previously computed six next-token candidate
log-probability comparisons per frozen case (three source arms × two label
rotations), using the exact pinned Qwen3.5-4B-Base checkpoint. The subsequent
October 23 **read-only** audit verified the full immutable private record set:

- **64 / 64** frozen case receipts; **384 / 384** source-arm/rotation conditions
- All receipts from **one** pinned checkpoint, tokenizer and runner source revision
- Receipt claims remain `machine_only=true`, `human_adjudicated=false`, `study_complete=false`
- Recorded runtime: Python 3.12.3, Transformers 5.12.1, Torch 2.13.0+rocm7.1 with the CPU-only execution contract
- Stored final summary's 64 case hashes and descriptive metrics **exactly replayed** from the underlying private files
- No source excerpts, private options, case-level scores, credentials or original passages appear in this document

The public exporter `research/exp053_agent_qwen_public.py` performs this
read-only replay before emitting only an approved schema of aggregate
statistics and provenance. It refuses missing/extra cases, failure receipts,
mixed model identities, invalid source-custody data, summary disagreement
and malformed claims.

### Descriptive results

Let the margin equal log P(pinned exam key) − log P(deterministic other
exam option), averaged across the two preregistered label rotations.

| Contrast | Mean log-probability difference | Median | Positive cases |
|---|---:|---:|---:|
| **D = supported − withheld** | 2.639648 | 2.828125 | 59 / 64 |
| **N = decoy − withheld** | −0.281738 | −0.218750 | 25 / 64 |
| **S = supported − decoy** | 2.921387 | 3.078125 | 60 / 64 |

These are **descriptive machine-proxy statistics**, without confidence
intervals or null-hypothesis testing. The foil is selected mechanically from
other exam options. It is **not** a blinded, independently verified,
semantically plausible false answer. Positive contrasts therefore **do not
establish genuine semantic grounding or causal reliance on supplied evidence**.

### Exact provenance

- Frozen 64-case manifest: `research/EXP053-COHORT-MANIFEST-2026-10-16.json`
- Private final aggregate SHA-256: `32a4a0168ffd38f6bf4790b6e39da4568f7f1d33a830ae1670843b1fdc84f0c8`
- Qwen checkpoint receipt SHA-256: `c9ed98bfbd9711289a957808342df94ae1ffc2ca6c292b9eea585c94206535f1`
- Exact runner SHA-256: `07bea44ac07eeed46c8f460a1153c532d2528d5d5fb7aab8edbb25fd45f67f62`
- Private case archive remains under a restricted local path and is **not** published to Git
- RACE source usage: noncommercial research only; no licensed source passages reproduced here

### Reproduction (read-only)

From the repository on the authorized private evidence machine:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 research/exp053_agent_qwen_summary.py --count 64
PYTHONDONTWRITEBYTECODE=1 python3 research/exp053_agent_qwen_public.py
```

The first command validates the 64-receipt archive without writing output;
the second replays and prints only source-free aggregate JSON. Both refuse
missing or inconsistent pinned private inputs; neither loads a model, reads
licensed article text, requests a reviewer or creates a new output file.

**Next work:** replicate source-only feasibility across pinned K2 and
MiniCPM models, preregister uncertainty and hard-negative checks, and keep
any fully validated, human-adjudicated scientific claim separate. No result
from this machine-only probe should be relabeled as human-reviewed.

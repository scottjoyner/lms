# EXP-053 measurement plan — source-specificity and negative-control pre-registration

**2026-10-16. Separate analysis-only supplement to immutable EXP053 protocol. No model inference or genuine human annotations have been performed.**

## Cohort and exact exclusions
Use the 64 sealed RACE high-test candidates and only **at least 40 genuinely independently adjudicated** approved distinct articles, as determined by EXP053 human gate. The local review packet and per-item gold / foil choice remain private. The public data output is de-identified article-hash aggregates only; no passage text, options, reviewers or keys. These articles are disjoint from earlier source collections but not proven absent from any checkpoint's pretraining. No hidden replacement of failed/rejected items.

## Three source arms, two label-rotation controls
For every approved case and each model, score exactly three passage arms with unchanged question/option *semantics*: `supported` (the same article), `withheld` (fixed uniform no-passage marker), and `decoy` (the next *approved* article by ascending opaque case ID, wrapping around, never the same article). The choice labels rotate twice. Rotation 0 uses the sealed blinded original four-option order. Rotation 1 deterministically shifts displayed label positions by two places, without altering answer-option strings. Both gold and human-reviewed plausible false alternative are tracked through these permutations. All arms in a case/rotation must use **identical option permutations**; the three prompts must differ.

Score the **single next-token label** (`A/B/C/D`) after the exact `Answer:` prefix for the human-supported correct option and the approved foil. Gate refuses cases where either target label tokenizes to !=1 token; do not choose targets by model scores or tune prompts post hoc. Persist exact checkpoint+tokenizer hashes, prompt hash, option permutation hash, case/source identities, all prompt/candidate token lengths, logs of failed/ retried trials and the exact prompt builder code checksum. Base-mode and instruction/chat-mode results are separate cohorts and cannot be pooled. No caching behavior assumption.

## Paired per-article metrics (within-model only)

For each model, case `i`, arm `a`, rotation `r`, set `margin(i,a,r)=log p(correct_label)−log p(approved_foil_label)`, summed over the fixed one-token label. Average margin over the two rotations.

- **Primary evidence sensitivity:** `D_i = mean_r(margin(supported))−mean_r(margin(withheld))`.
- **Negative-control decoy sensitivity:** `N_i = mean_r(margin(decoy))−mean_r(margin(withheld))`.
- **Source-specificity contrast:** `S_i = D_i−N_i = mean_r(margin(supported)−margin(decoy))`. THIS is the stronger supporting evidence requirement, not just a positive answer prior.
- Report means, medians, positive proportions and deterministic 95% **article bootstrap** percentile intervals for D/N/S (2,000 resamples, frozen seed `EXP053-metrics-bootstrap-v1`). Cases are articles; rotations and source arms are repeated measures, never counted as independent samples. For primary D, one-sided paired article sign-flip Monte Carlo p-values (5,000 draws, frozen seed `EXP053-metrics-signflip-v1`) are adjusted by Holm across the three model families. Sign-flip symmetry is an assumption; treat resulting p-values as exploratory in non-symmetric distributions.
- Do not interpret positive D alone as semantic grounding: surface lexical matching, exam-writing artifacts and pretraining overlap can generate D>0. Meaningful evidence support requires S>0 with positive interval plus traceable human review, and independent generation/abstention followup (not supplied by this module). Report negative/null outcomes without cherry-picking.

## Measurement isolation and explicit HOLD

All 3 specified checkpoints: `K2-Horizon-3.7B`, `Qwen3.5-4B-Base`, `MiniCPM-V-4.7-1B`; same exact approved item set for each model. Fail closed on duplicates, missing arm/rotation, any unapproved case, repeated source passage hashes, improperly rotating correct/foil labels, non-finite or positive log probabilities, a missing model, inconsistent model checkpoint/tokenizer hashes, incorrect decoy source, or inconsistent option permutation. No inter-model ranking by raw nats or uncontrolled CPU time. Human-answerable exam multiple choice does not yet measure abstention/free generation; those remain explicitly OPEN for future freeze.

**No model-forward script, model execution, aggregate real results, publication, merge, training, NAS/Neo4j writes, serving or production promotions authorized in this phase.** Synthetic unit fixtures are software tests only, never scientific data.

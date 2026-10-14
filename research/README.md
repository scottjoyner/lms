# K2-Horizon model-lineage layer matrices

Research-only model comparison, conforming to LMS separation of
**exact model identity, observational similarity, and behavioral quality**.
This is not an inference endpoint, a new benchmark run, or a deployment gate.

`k2_layer_overlap.py` reads SHA-pinned prior experiments EXP-016, 018, 019,
021 and produces a full human-readable Markdown layer matrix and behavioral
timeline in the **central knowledge repository**, not stand-alone results JSON.
It validates every 5×5 pool/role CKA slice (all 25 cells), preserves
undefined early token states as undefined, and computes:

- Same-depth average and final-depth CKA for every adjacent K2 stage.
- Row nearest-neighbor mapping, mutual nearest sampled depths.
- Maximum mean bijective sampled-layer matching on *measurable* cells.
- Separate masked-mean and last-token, whole-response and signed-role grids.
- Cross-architecture K2↔Qwen3.5-0.8B CKA matrices.
- Exact K2 checkpoint identifiers, actor/recipient behavior and
  answer-format/abstention evidence, without conflating metrics.

## Repeatable read-only analysis

```bash
R=/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026
python3 research/k2_layer_overlap.py --evidence-dir "$R" \
  --out "$R/EXPERIMENT-022-LAYER-OVERLAP.md" --replace
python3 -m unittest -v tests.test_k2_layer_overlap
sha256sum "$R/EXPERIMENT-022-LAYER-OVERLAP.md"
```

Expected report SHA256:
`4cbdbdf96a597097b2ef8be66d6b82ec502a16ad316abdc10ac19a32c48bbaf0`.

This code uses **previously executed** frozen model runs;
it does not load a model, generate embeddings or reexecute experiments.
The `~/knowledge` directory contains original source scripts,
fixtures, numbered historical protocols, reports, and secondary
machine evidence. The original HF weights must be independently
restaged at the pinned revisions for fresh neural measurements.
No auto-download, NAS operation, production service effect,
fine-tuning, graph mutation or model-promotion authority.

## EXP023 direct weight-drift analysis

`research/k2_weight_drift.py` reads all 252 transformer tensors per
checkpoint transition and writes only the central human-readable
`EXPERIMENT-023-WEIGHT-DRIFT.md`. It computes parameter-weighted
cosine, L2 drift, elementwise BF16 change fraction, layer and module
contributions; identifies each model shard by SHA256. It preserves
weights and never outputs parameters. Synthetic Safetensors regression
checks live in `tests/test_k2_weight_drift.py`; twelve combined LMS
research unit tests passed on x1-370 with pinned frozen-model venv.

Canonical report SHA256:
`56c4b353462ced77cec1637aa675cb2edd4e1c1190b7f7f2bbe47cec142a69a4`.
This confirms weight continuity, **not** causal RL change or any
common Qwen3.5 weight ancestry. Beware default-RoPE↔YaRN confound.

## EXP024 RoPE × model-weight factorial

`research/k2_rope_factorial.py` loads two pinned K2 BF16 checkpoint
weights sequentially and evaluates each with default and YaRN positional
encoding **by replacing only the rotary-embedding module in memory**.
It verifies native EXP018 A/B totals (149/256 and 161/256), checks
representative learned-tensor SHA256 before and after, runs the same
256 fact/reversal QA rows, and generates exact 5×5 CKA grids on
64 reversed fact statements for four controlled contrasts plus the
native comparison. Nothing is registered with fleet model serving.
Human-readable output is saved in central `~/knowledge` as
`EXPERIMENT-024-ROPE-FACTORIAL.md`; SHA256
`ffd0752b7d160bb0d9582b2d3c73b05a47703c74a3acf5acf692ce3d42b967a3`.
Tests: `tests/test_k2_rope_factorial.py`.
**18/18 combined EXP022/023/024 source/evidence tests PASS.**

Interpretation: fixed mid2 learned weights with YaRN scored 160/256
versus native default 149/256, nearly reproducing native RL/YaRN
161/256. A nonzero weight×RoPE interaction precludes unique causal
allocation to historical RL training. Source facts remain synthetic.

## EXP025 full checkpoint parameter closure

`research/k2_full_weight_lineage.py` recomputes all 255 BF16 tensor
comparisons per adjacent K2 checkpoint (1,078,285,824 total parameters),
including the input embedding, norm and LM head omitted by EXP023.
Fails closed on exact EXP018 SHA, index, 255-key inventory, shape/dtype,
expected full parameter count, per-tensor finiteness and EXP023 252
transformer-tensor numerical parity. No raw weight or hidden vectors
exported. Observations only in central `~/knowledge` as
`EXPERIMENT-025-FULL-WEIGHT-DRIFT.md`; checksum
`169507cd918359b189abf208c5708af5a301a8017096b23642864f455f86821a`.
Tests: `tests/test_k2_full_weight_lineage.py`.

## EXP026 fact-neutral RoPE factorial

`research/k2_rope_fact_control.py` evaluates the fixed frozen K2 2×2
weight-stage and default/YaRN RoPE intervention on all 512 EXP019
factual and neutral prompt rows. Enforces pinned original EXP019 and
fixture SHA256, native baseline counts, fixed checkpoint sources,
greedy answer-label metrics and representative tensor identity.
Outputs only central Markdown, with a paired 32-source-group 2000-draw
descriptive bootstrap for five factorial contrasts. Result SHA256:
`4452696eb2b567332112b392baac941a403c718053ccb33377dbe6cca020653c`.
Regression: `tests/test_k2_rope_fact_control.py` (27 LMS scientific
source/evidence tests passing for EXP022–026).

## EXP027 YaRN frequency versus amplitude

`research/k2_yarn_components.py` uses frozen K2 midtraining-2 weights and
separately copies official default/YaRN inverse frequencies and rotary
cosine/sine `attention_scaling` into four in-memory rotary modules.
`research/k2_rope_components_fixture.py` deterministically generates 256
source-disjoint authored synthetic A/B and factual/neutral prompts (SHA256
`ef0f99caa44a96f689c4b80ebe8f842032289238513d99bfbfb44dfa3ca79523`).
The 2×2 factorial shows a 24/64 both-order correctness gain with *scaling
alone*, no gain with frequencies alone, and only 37→39/64 debiased
correct-entity change. Do not conflate the changed letter-readout prior
with semantic competence. Saved Markdown under `~/knowledge`:
`EXPERIMENT-027-YARN-COMPONENTS.md`, SHA256
`d37b5e4babfc3db23ba2a02745e15f987a1552e82404741ef9bfb41f8a797ff2`.
Regression: `tests/test_k2_yarn_components.py`. No JSON results.

## EXP028 analytic phase and context gap controls

`research/k2_rotary_phase.py` imports only the pinned native K2
rotary classes and compares frequency-channel unit-circle phase
geometry across token distances without model inference. Its central
Markdown `EXPERIMENT-028A-ROTARY-PHASE.md` has SHA256
`3612d1e0012e79233fc4f25ec03884480cdc4fbc415bb6e57047dc477d29404b`.

`research/k2_context_gap.py` and `tests/test_k2_context_gap.py`
are a separately preregistered, bounded **EXP028 research-only** pilot
using 2 published EXP027 source groups with both reversed states,
actor/recipient, factual/neutral and both answer choices. Per paired
prompt, filler contents and total token lengths (within 4 tokens)
are matched but filler placement changes fact-question separation.
Tests and source SHA are pinned before inference. **All 256 model
forwards completed and the eight cells passed an exact-result regression
check.** Central measured Markdown `EXPERIMENT-028-CONTEXT-GAP.md`
SHA256 `97adff195b2d50b5a57877434db4313994ef1a700920517f9d98f0ea7a65eeec`. Frequency-only switching did not produce a
consistent improvement across these four small, placement-matched cells.
The two-source-group pilot remains **too small for generalization**.

## EXP029 public human-annotated QA + abstention (preflight)

`research/k2_human_grounded_qa.py` deterministically filters the official
Stanford SQuAD 2.0 dev split (source SHA256
`80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8`),
selects eight unique article topics each with one one-word answerable
and one unanswerable human-written question, and uses direct token
generation rather than A/B logits. The same 768-token neutral filler
appears either before source evidence or between evidence and question,
with per-question matched input length. Gold source selection digest:
`aad6e578544e2a3f100ab4b5c93d7ee6158fbfbe14bd426b8bffd9f9d1cde23b`.
All source checks, annotations/spans, strict/lenient/UNKNOWN scoring,
K2 tokenizer and native Qwen chat tokenizer passed 7 offline tests.
No raw human dataset, prediction strings or label sources are committed;
source dataset must be staged locally with original CC BY-SA 4.0 attribution.
Each model requires an independently complete resource-gated 48-prompt
pass before any outcome is added to the canonical Markdown evidence.
K2 is not assigned a fictitious chat template. Nothing deployed.

## EXP029 measured comparison — SQuAD2 public human-gold dev

Both 48-prompt generation passes completed, K2 mOPD plain (443.5 seconds)
and Qwen3.5-0.8B native chat (268.8 seconds), on the same eight article
sources and three near-length-matched placements. Human-gold selection
SHA256 `aad6e578544e2a3f100ab4b5c93d7ee6158fbfbe14bd426b8bffd9f9d1cde23b`.
K2 measured Markdown SHA256 `0ab5ac2d744362969b97618b32ce9ad536ae32bda820a4bfac79410cc7c17c71`;
Qwen measured SHA256 `a166f1727620bc2cc162f07c388584a97201bf2064e233cb92c87c2d3d60cd85`.
`research/k2_human_qa_compare.py` consumes only those two SHA-pinned
Markdown reports and emits the aggregate-only central
`EXPERIMENT-029-COMPARISON.md` (SHA256
`6d78895839a4efd9c6962c7942a4b75cc69f8a70f0afa6d81a7fd4535ae44a1e`).
No raw source data or model generations are saved or committed.
Native chat versus K2 plain is **not** an architecture-isolated comparison.
Tests: `tests/test_k2_human_grounded_qa.py` and
`tests/test_k2_human_qa_compare.py`, with deterministic source/score
parity and corrupt-input rejection. The 36GiB initial memory floor
was prospectively registered before inference; 22GiB in-run stop retained.

Grounded K2 output-format and abstention remain NO-GO on this pilot.

## EXP030 K2 answer-output contract and budget (research-only)

`research/k2_human_output_contract.py` compares two **separate one-factor**
changes against the exact 32-token EXP029 human-gold K2 report:
`strict32` changes only the leading answer/UNKNOWN instruction;
`budget64` keeps every original input token and increases only the
number of generated tokens from 32 to 64. 16 human-annotated SQuAD2
questions repeated as short and long-gap (32 outputs per arm). Source
selection, original prompts and measured baseline require fixed SHA256.
All 6 pre-inference `tests/test_k2_human_output_contract.py` tests passed.
Strict exact answers, lenient gold spans, first lexical answer word,
exact/first UNKNOWN and EOS/token cap are separately counted. A correct
first word is **not** a strictly valid complete response. Human-gold
source/predictions are never committed or logged. No model promotion.

## EXP030 completed strict32

Frozen mOPD K2 with only the leading instruction stronger generated all
32 human-gold short/gap responses. 0/8 strict answers, 0/8 strict UNKNOWN
for each position; 5/8 answers matched first lexical word in each;
**all 32 outputs reached the 32-token limit** (baseline 23/32).
Measured central Markdown `EXPERIMENT-030-STRICT32.md` SHA256
`4bf704a6b64ef79f0ca5b73cfe2e55b73c8a39990ad5328c8a90367614ba175e`.
This does not establish correct whole-response output or safe abstention.
Independent `budget64` arm remains resource-admission gated.

## EXP030 both interventions completed

`research/k2_human_output_contract.py` completed 32 SQuAD2 direct model
generations per arm on the pinned official K2 mOPD checkpoint. Stronger
single-word/UNKNOWN instruction at 32 tokens did not fix exact outputs,
and reached the cap on 32/32 items. Holding prompts completely identical
and raising max_new_tokens to 64 still produced 0/8 exact human gold and
0/8 exact UNKNOWN at both short and long-gap placements.

`research/k2_human_output_comparison.py` is a **separate SHA-pinned
Markdown-only** reconciliation with the untouched original 32-token
EXP029 result. Reports in central knowledge:
`EXPERIMENT-030-STRICT32.md`, SHA256
`4bf704a6b64ef79f0ca5b73cfe2e55b73c8a39990ad5328c8a90367614ba175e`;
`EXPERIMENT-030-BUDGET64.md`, SHA256
`ac206d810c20668e22c790f687c803be55ef868e02ab846bd576bec12b40b490`;
`EXPERIMENT-030-COMPARISON.md`, SHA256
`a47b18bc55bc8f5de83be03726ee4161973e8231684d2f65ebe1eac7b524afc0`.
Tests `tests/test_k2_human_output_contract.py` and
`tests/test_k2_human_output_comparison.py` verify all counts and fail-closed
on tampered source. No original source weights, training, NAS or services
changed. This study DOES NOT solve UNKNOWN abstention or authorize promotion.

## EXP031 native EOS/alternative terminator witness

`research/k2_termination_audit.py` replays eight exact short EXP029 SQuAD2
human-annotated QA prompts (first four distinct source article groups) with
frozen K2 mOPD weights, greedy 32 tokens and **unchanged native EOS=1**.
It counts five other official tokenizer boundary markers and native EOS
without saving any raw generations or input passages. Tokenizer, model
config and native generation config agree on EOS `<|endoftext|>` ID 1.
A real `<|ifm|im_end|>` (ID 64019) occurred in 1/8 outputs at position
13; only one native EOS occurred and 7/8 hit the output cap without EOS.
This identifies **possible harness stop ambiguity**, not validated alternate
answer-termination semantics or production permission to alter EOS.

Central Markdown `EXPERIMENT-031-TERMINATORS.md` SHA256:
`a0e7db3d681eb81cc3880715aab9f9542948a4008b1e61516ad7e8c1499beb50`.
Regression: `tests/test_k2_termination_audit.py`. Original data and
model service state untouched.

## EXP033 gated MLP neuron experiment (completed)

`research/k2_neuron_sniffer.py` hooks the last input-token pre-down-projection
MLP intermediate activation in K2 midtraining-2 and RL (layers 0/7/14/27)
and Qwen3.5-0.8B official native chat (layers 0/6/12/23). The protocol
is independent EXP033 to preserve another research stream's EXISTING EXP032
stop/lexical source-gate protocol. Exactly 16 human SQuAD2 question inputs
from eight articles are pinned with source and model SHA256; outputs are
aggregate Markdown with no raw prompt, per-example neuron activation or
neuron IDs persisted. K2 top-32 within-coordinate matches and 16-example
K2/Qwen Gram CKAs have paired/sign-flip/shuffle controls. This is NOT
causal neuron identification or teacher weight provenance.

Evidence: `~/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-033-NEURON-SNIFFER.md`,
SHA256 `f9c0b10b6846eab1e4f675fd0303ce770f80e58ca77748cd9e278b561e7765ff`.
Tests: `tests/test_k2_neuron_sniffer.py`. No production/graph/NAS mutation.

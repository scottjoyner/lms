# EXP-053 tokenization preflight: frozen prospective addendum (2026-10-16)

Research-only, BEFORE model-tokenizer candidate scan, independent of human labels.
This does not modify the pinned EXP053 scientific protocol or measurement plan.

## Source and model limits
Use the existing 64 candidate article/question/option cases from the private
packet, validated against the immutable public manifest; do NOT inspect the
private exam answer key, model logits, human reviews or production runtime.
Load only local checkpoint config and tokenizer files for K2-Horizon-3.7B,
Qwen3.5-4B-Base and MiniCPM-V 4.7-1B. No model weights or forwards.

## Frozen raw-text prompt
Read ONLY the passage and answer the multiple-choice question. Choose the
single option best supported by the passage. If the passage is insufficient,
output UNKNOWN. Write exactly one answer letter if supported.
Passage:
{passage}
Question: {question}
A) {optionA}
B) {optionB}
C) {optionC}
D) {optionD}
Answer:

The actual string ends immediately after the colon, with NO newline.
Supported source is own article. Withheld source is exactly
"The source passage is withheld.". The decoy is the NEXT candidate article
by ascending opaque case ID, wrapping around (not self). Rotation 0 retains
the existing blinded option order. Rotation 1 shifts displayed position
by two letters: options C,D,A,B. This is only candidate feasibility:
the approved cohort's final decoy source must be recomputed after reviews.

## Technical gates
For 64 x 3 source arms x 2 rotations = 384 prompts per checkpoint:
- tokenize raw prompt with add_special_tokens=True, require 1..4096 tokens;
- for each A,B,C,D, the continuation is a single leading SPACE + letter,
  and the full prompt plus continuation must tokenize to exactly the
  original prompt token IDs + one distinct candidate token ID;
- pin source, public manifest, protocol, analysis plan, checkpoint config
  and tokenizer digest; retain source-free aggregate counts and lengths only;
- no licensed article text, private IDs, option strings, passage token IDs or
  reviewer details in public outputs;
- refuse mutation of the frozen prompt or a failed tokenizer boundary check.
The forced 4-letter metric cannot measure UNKNOWN abstention, which requires
a separate study. This scan proves tokenization feasibility, NOT evidence
grounding, pretraining absence, valid model probabilities, or answer accuracy.
Zero neural forwards / zero signed human reviews in this phase.

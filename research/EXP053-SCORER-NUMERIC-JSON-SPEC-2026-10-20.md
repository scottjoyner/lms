# EXP-053 — scorer score-JSON numeric fidelity (frozen spec)

**Frozen before code edits, 2026-10-20. NO_GO_RESEARCH_ONLY.**

## Synthetic reproduced data-integrity failure

The source-only scorer evidence reader's `unambiguous_json` blocked explicit
JSON `NaN`/`Infinity` constants and duplicate keys but allowed arbitrary
JSON decimal float lexemes through Python's default binary64 parser.
`-1e-999` became `-0.0` without a rejection, while `1e999` became `inf`.
A completely fictional score archive containing `{"gold_logp":-1e-999}`
passed `read_attested_scores` as signed-pinned bytes yet exposed a zero score
to later scientific computation. The real statistician's `math.isfinite`
check rejects `inf` but cannot detect underflow once a nonzero score is 0.
This does not demonstrate any real RACE/model run; authorization pins EMPTY.

## Required change and test contract

- Add strict `parse_float` to shared `unambiguous_json` parser for *all*
  private scorer evidence JSON (signed attestation, score and retry records,
  operator signed authorization). Refuse nonfinite binary64 conversion of
  numeric JSON tokens (`1e999`, `-1e999`) and refuse conversion of a
  nonzero numeric mantissa to either `0.0` or `-0.0` (`-1e-999`). Preserve
  ordinary finite binary64 values and true-zero forms, including signed
  zero and exponents on zero. Distinguish the mantissa from exponent digits.
- Existing duplicate-key, forbidden NaN/Infinity constants, and safe
  private fd inode/path guards remain mandatory. Errors return sanitized
  `HOLD EXP053` and never echo scored text or private numbers.
- Add synthetic tests for direct JSON parse and pinned small private score
  archive decode: overflow, underflow, normal finite/zero/subnormal values,
  nested numbers, duplicate fields and explicit JSON constants. Confirm
  existing cryptographic-attestation, attempt-log, operator, external CAS,
  all EXP-053 and broad non-K2 tests pass unchanged.
- Zero actual licensed article text, model weights, signed real human
  responses, true inference, signature creation on real custody roots.
  Keep all operator, scorer, reviewer, witness, and weight pins EMPTY.
  Publish only a stacked DRAFT technical PR on exact public #45 head.

This narrow parser validation does not independently certify physical scorer
execution, scientific precision/quantization, reviewer integrity, or witness
custody. All such approval gates remain NO_GO_RESEARCH_ONLY.

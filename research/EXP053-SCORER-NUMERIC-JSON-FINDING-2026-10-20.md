# EXP-053 — scientific score JSON exponent overflow/underflow

**2026-10-20. Synthetic-only. NO_GO_RESEARCH_ONLY.**

The common execution evidence `unambiguous_json` parser rejected explicitly
written `NaN`/`Infinity` and duplicate JSON keys, but allowed numeric tokens
such as `-1e-999` and `1e999` through the default Python binary64 decoder.
In a source-free fixture, `-1e-999` became `-0.0` and a pinned fictional
score archive exposed that zero to analysis despite its nonzero signed JSON
number. `1e999` became `inf` and was only refused later by the final metrics
module; that later finite-value check cannot detect underflow to zero.
Nothing here represents real RACE text, scorer execution, human approval, or
an approved operator run, because all real trust roots remain intentionally
empty.

## Source-only repair

The scorer's shared strict JSON parser now sets `parse_float` to reject
nonfinite binary64 conversion and reject numeric tokens whose mantissa
contains nonzero digits but whose converted float equals either 0.0 or
-0.0. Genuine zero mantissas, even when an exponent has nonzero digits,
continue to round to binary64 zero; normal finite decimals and representable
subnormals remain accepted. Duplicate keys and explicit JSON nonfinite
constants still cause a sanitized HOLD without echoing any private text.

Invented tests cover positive/negative overflow and underflow; exponent
separation from mantissa, genuine positive/negative zero and ordinary finite
negative log-probabilities; nested arrays/dictionaries; duplicate keys and
constants; signed-attestation compatibility and same-descriptor pinned
synthetic private score archive rejection before downstream metrics.

This prevents **silent zero conversion** at parsing; it is not a proof that
real future hardware deterministically computes correctly rounded scientific
log-probabilities, nor proof of independently controlled model identity or
human annotations. Both operator witness and weight approval remain absent.

**NO_GO_RESEARCH_ONLY. No merge, licensed corpus work or model execution.**

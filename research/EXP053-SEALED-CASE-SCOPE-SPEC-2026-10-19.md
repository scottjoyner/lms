# EXP-053 sealed answer-key case coverage — frozen source-only specification

Date: 2026-10-19. **NO_GO_RESEARCH_ONLY**.
No real reviewer, RACE text, signing key, or scientific execution used.

## Narrow reproduced structural gap

`verified_admission()` checked the sealed answer-key digest against a
preregistered manifest and then iterated ONLY the sealed keys. There was no
independent assertion that the sealed key set equals all 64 unique packet
case identifiers. In a **purely synthetic integration experiment that mocked
all of the real trust roots** (immutable manifest, private roster, signature
verification and input file reads), a 64-case invented packet with only
40 sealed keys returned a synthetic `human_approval=True` receipt. That
cannot bypass the real pinned manifest, but reveals a missing structural
invariant if future trusted artifacts or mocking mistakes change upstream.

## Required contract

- After verifying exact pinned packet and answer-key digests, but before
  reading any signed reviewer files, ensure packet is a dict with 64 cases,
  its `schema` is `exp053-packet-v1`, each case is a dictionary with a
  unique nonempty string `case_id`, and the sealed answer key is an object
  mapping **exactly these 64 identifiers** to one of `A,B,C,D` (single
  ASCII character). No missing, extra, duplicate, non-string or empty keys,
  numeric/list/dict values, or multiple-letter answers are allowed.
- Validate the outer sealed object is exactly keys `packet_sha256` and
  `answer_key`, and `packet_sha256` must equal digest(packet); missing keys,
  wrong types and malformed structures must return a sanitized HOLD rather
  than raw KeyError, TypeError, or AttributeError.
- The preregistered public manifest sealed-key SHA must be a canonical
  lowercase 64-character digest. Preserve the existing exact hash binding.
- Review A/B coverage and optional adjudication remain unchanged; distinct
  real blinded human reviewers, at least 40 **accepted** articles among
  the 64, independent roster/weight pins and witness custody still required.
- Test with only invented case IDs and fake answer letters; negative cases:
  40-of-64 partial keys, extra foreign key, wrong and missing outer keys,
  wrong label, duplicate packet case ID, 63-case packet, non-list cases,
  malformed manifest pin. A full invented 64-case map should pass structural
  checks without involving real signatures or reading a licensed article.
- Publish only source code, synthetic tests and documentation in a stacked
  **draft** PR based on exact PR #40 head. Never set owner/weight pins, merge,
  deploy, export real human review bytes, generate fake real signers or run
  scientific model forwards.

Structural source validation does NOT establish human independence, scientific
approval, or cryptographic witness authority. NO_GO_RESEARCH_ONLY.

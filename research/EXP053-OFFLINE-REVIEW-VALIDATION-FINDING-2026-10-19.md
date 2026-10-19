# EXP-053 — offline-review form and admission validation mismatch fixed

**October 19, 2026. Research-only. NO_GO_RESEARCH_ONLY. No human reviews
created, no new signed receipts, no source text exported, no model inference.**

## Confirmed defect / consequence

A local synthetic Node reproduction showed that the offline reviewer interface
`complete()` returned `true` for an empty supported answer (`""`) and the
multi-character string `"AB"`. JavaScript's string substring membership
`"ABCD".includes(x)` is not a one-character choice check. An adversarial
parity test then found the authoritative Python `check_decision()` used
`x in LETTERS` with `LETTERS="ABCD"`. Python accepts `""` and `"AB"` as
substrings too.

**Why this matters:** the browser can overcount supposedly completed review
items; the Python signing-intake validation can accept malformed answer/foil
strings. Even though the primary supported answer is independently compared
to the sealed key at a later admission step, an invalid multi-character foil
could satisfy that later agreement check when both reviewers supplied the
same invalid foil. This could taint an otherwise admitted scientific foil.
The defect is in code logic, not evidence that anyone submitted such a form.
There are still **zero independently admitted human review articles** and
the separately approved reviewer roster pin remains empty.

## Exact remediation

- The offline UI `complete()` uses `/^[ABCD]$/` with an explicit string-type
  guard for both supported and foil option. It also checks that exact source
  quotes and rationales are strings and meet the same lengths as Python; a
  rejected case needs a string reject reason of at least 12 characters.
- Python `exp053_race_review.check_decision()` now requires **exactly one**
  A/B/C/D letter per answer/foil before accepting signed review content.
  Signature custody, blinding, preregistered cohort, and approved owner
  roster values remain unchanged. No automatic approval is introduced.
- Synthetic Node VM parity tests cover 19 valid/invalid accept/reject cases.
  Python negative tests exercise forbidden empty or multi-letter answer/foil
  values, and verify a malicious synthetic draft cannot reach key loading in
  `exp053_sign_review.sign()`. No real reviewer key is generated or opened.

## Downstream source, metrics and archive replay parity

A separate source scan found the same Python substring validation pattern in
frozen-cohort answer intake (`cohort`), model-free metrics (`analyze`), and
private score archive replay (`check_private_archive`). All now require
**exactly one** A/B/C/D character for gold and foil identities. This prevents
malformed approved maps from reaching label rotation or yielding an accidental
valid score-receipt comparison. The corpus/protocol/manifests and signing
authorization pins remain unchanged; the repair applies only validation
logic. Synthetic negative tests cover blank, multi-character and non-string
choices at every layer and still accept valid single-letter decisions.

## Independent human checks still necessary

An actual study owner and real A/B reviewers must establish identity,
independence, verified key control, blinding, source-quotation diligence and
at least 40 source-distinct independently adjudicated RACE articles. Third-
party adjudication remains required for disputes. This regression test is not
one human review; no fake signatures or approvals count toward the threshold.
No change to scoring authority, witness signing root, external high-water
state, production deployment, data ingest or model execution is approved.

Source-only reproduction:

```
python3 -m pytest -q tests/test_exp053_review_ui_parity.py \
  tests/test_exp053_human_review_gate.py tests/test_exp053_offline_ui.py \
  tests/test_exp053_review_ui_dirty.py
```

All acceptance and deployment controls remain NO_GO_RESEARCH_ONLY.

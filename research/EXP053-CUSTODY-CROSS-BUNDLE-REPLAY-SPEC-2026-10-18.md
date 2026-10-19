# EXP-053 cross-bundle replay and equivocation screen — prospective source-only specification

Frozen before implementation, October 18, 2026. This is a read-only, untrusted
source-free **negative** receipt collection audit, NOT an independent witness,
trusted checkpoint, authentication oracle, durable journal, or scientific gate.

Inputs: one bounded EXP-053 custody claims file, and 1–12 individual source-free
receipt bundles. Each bundle must pass the *unchanged frozen* receipt
consistency audit (`exp053_custody_receipt_consistency.assess`), including the
plan binding, date sanity and per-bundle signature verification against its
**self-supplied** Ed25519 key. Files must be bounded regular files opened via
O_NOFOLLOW and O_NONBLOCK; reject symlinks, pipes, malformed JSON, duplicate
fields, oversized input, invalid signatures or mismatched plan/candidate.
Time is a single fixed local-UTC observation per audit invocation, not an
independent trusted timestamp; it cannot establish historic freshness.

The collection is an **exclusive single plan and observer context**. When
nonempty, all receipt bundles MUST claim the same exact canonical observer
public key and checkpoint-reference SHA256. This restriction is just a
consistency constraint; it cannot establish that either key or checkpoint is
real or independently owned. An empty collection member may occur only when
it is the **sole** bundle, representing 0/12 observations. A nonempty member
may contain 1–12 receipts from the fixed 12 prospective challenge IDs.
Reject semantically identical bundles after canonical JSON normalization,
regardless of whitespace or key order. Reject
repeated signed receipt rows (including reused signature or nonce), repeated
nonces across members, any repeated scenario ID across members (whether
identical, contradictory, or retested), mixed empty and nonempty bundles,
changed observer key and changed checkpoint reference. This deliberately
refuses even some benign retesting; another human-approved plan revision is
necessary for independently authorized retests. No automated rollover,
merge, deletion, repair, or reset is permitted.

The report schema is `exp053-custody-cross-bundle-replay-v1` and includes ONLY:
`schema`, `candidate_alias`, `bundle_count`, `unique_scenario_claim_count`,
`missing_scenario_ids`, `claimed_denied_count`, `claimed_bypass_count`,
`claimed_unknown_count`, `decision`, `scientific_execution_authorized`,
`independent_administration_approved`, `witness_durability_approved`,
`all_challenges_independently_attested`, `signer_authority`, `durable_replay_protection`.
All approval flags remain **false** under every input. `decision` MUST be
`NO_GO_RESEARCH_ONLY`; `signer_authority` MUST be
`UNTRUSTED_SELF_SUPPLIED_KEY`; `durable_replay_protection` MUST be `NOT_PROVEN`.
Counts refer to *untrusted consistent claims* and never executed negative
tests or verified external evidence. Do not report raw observer key or public
fingerprint, signature, nonce, timestamp, digest, private evidence, file path,
username, IP address, or source text. The allowed alias and fixed scenario IDs
are the only specific identifiers in the report.

The CLI returns exit status 2 for both clean and invalid inputs; malformed
input prints only safe NO-GO status. No persistent high-water state is kept.
If the scorer omits an older contradictory bundle, hides an observation,
replays the whole file set, rewrites local reports, forges a new self-signed
bundle, or resets the local clock, this tool CANNOT detect it. Independent
key custody, independently published immutable monotonic checkpoints,
independent receipt observer, sealed provenance and negative physical tests
remain outside its trust boundary. Never promote code/PR, deploy services,
create keys, access remote hosts, write NAS/Neo4j, or conduct model inference
from this instrument.

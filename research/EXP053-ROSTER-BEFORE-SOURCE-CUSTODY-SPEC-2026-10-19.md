# EXP-053 — authenticate pinned reviewer roster before licensed source reads

**Frozen before code changes, October 19, 2026. NO_GO_RESEARCH_ONLY.**

A valid-looking 64-character owner-roster pin is merely a string. The existing
`verified_admission()` checks the pin's syntax but then hashes the licensed
RACE source and reads private packet/sealed-key files before `load_roster()`
verifies the exact private roster bytes. A synthetic filehash trap proved that
an absent/invalid reviewer roster does not prevent this earlier access.

## Required source-only change

1. After checking the pin's lowercase hex syntax, verify the immutable,
   already-public source-free manifest exists and matches the preregistered
   `MANIFEST_SHA`; safely read its public JSON and extract the exact 64-byte
   lowercase-hex `packet_sha256` field. This is not a license to read RACE.
2. Load and verify the private roster by the original `load_roster()` with
   (`keys_path`, the pinned manifest packet digest, `APPROVED_REVIEWER_ROSTER_SHA256`,
   the repo root), **before** any access to licensed `SOURCE`, private packet,
   sealed answer key, or blinded reviewer forms. A merely syntactically valid
   owner pin, nonexistent roster, wrong roster SHA, unsafe roster mode, wrong
   packet binding or invalid slot enrollment must HOLD without those reads.
3. After exact private roster verification, retain all original SHA checks
   on source/protocol/public manifest, full packet digest, sealed answer-key
   digest, A/B distinct signing identities, independent third adjudicator and
   >=40 approved source-disjoint cases. The roster must be bound transitively
   to *the pinned public packet SHA* and then to the actual private packet.
   No shortcut to human/source admission; owner roster pin remains EMPTY.
4. Fully source-free synthetic tests instrument the source hash function and
   private-file access to prove absence of reads for a missing, invalid,
   mismatched, or unsafe roster even when the configured owner pin has a valid
   64-hex shape. A separately prepared valid **synthetic** roster permits
   flow to the still-mandatory source hash check, which must HOLD on mismatch.
   No live RACE file, reviewer secret, human approval, model inference,
   production write, Git push/merge, or deployment is used.

The source-only test can establish a local order-of-operations invariant, not
human identity independence, owner consent, external witness custody, or an
admitted scientific experiment. Current decision remains NO_GO_RESEARCH_ONLY.

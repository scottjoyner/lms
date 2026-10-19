# EXP-053 — independent custodian request (source-free / nonproduction)

**Authority: NO_GO_RESEARCH_ONLY. This is an evidence request, NOT a work order to deploy or operate a witness.**

The scorer-controlled research host (`x1-370`) has a controller SSH identity
file used across Lenovo, Destroyer and Xwing. Off-host physical capacity and
SSH reachability do **not** establish independence from that control plane.
The current Lenovo source-free claims fixture is a CONTROLLER OBSERVATION,
not a real witness, signature, custody signoff, or privileged-access proof.
No externally controlled publisher, independent head/key pin or durable CAS
has been established. All model inference and external witness deployment
remain explicitly blocked.

## Prospective role boundary — before any physical test

A separate accountable custodian, not the scorer operator or a fleet agent,
must independently decide whether any disposable off-fleet system can qualify.
The custodian must own the following **outside** the scorer-administered
infrastructure: administrative/root authority, hypervisor/VM control, snapshot
and backup restore, witness signer material, signer rotation/revocation,
monotonic durable high-water state, and append-only checkpoint publication.
An independent observer must retain the latest checkpoint head and verified
signer root; the scorer must not choose or replace either. A distinct
human reviewer approves/records the custodian privilege graph and the
negative-test results. One person pretending to act through multiple software
keys or roles is not adequate separation.

## Evidence custodian must obtain privately — not in this Git repository

1. Privilege and recovery matrix covering scorer operator, fleet automation,
   witness administrator, hypervisor/backup owner, root signer owner and
   independent checkpoint observer. Verify effective authorities; claimed
   ``CANNOT`` is not proof of denial. Retain human accountability/consent.
2. Evidence of separately governed trust roots, verifier-pinned independent
   public key, separate external latest head, revocation and key-recovery
   procedures. Never use a self-supplied signer or local CLI digest as proof.
3. Written authorization for a **disposable, nonproduction** test environment,
   a resource/time budget, rollback/destruction plan, and explicit refusal
   to expose private RACE text or change production data.
4. Independently controlled negative experiments across the twelve scenarios
   in the frozen custody challenge specification. Test scorer and fleet access
   to privileged login, signer, CAS state, snapshot/backup restore, trust
   rotation and checkpoint fork; independently attest both denial and bypass.
5. Native signed append-only, independently read-back high-water checkpoints
   with monotonic sequence and replay/fork/crash/reboot/restore recovery tests.
   Confirm that prior externally retained head cannot be replaced along with
   a scorer-controlled local reference. Fail closed when either side is absent.
6. Private signed adjudication: a human who was **not** the scorer checks
   primary evidence and test methodology. Only public-safe cryptographic
   references and a custody/no-go disposition may be returned to research.

## Independent scientific gates remain separate

Even a genuinely independent witness **cannot** approve model execution
until at least 40 original source-disjoint articles have genuine signed,
independently blinded human reviews, including a distinct adjudicator for
conflicts; exact model bytes, tokenizer, runner source and pre-inference
fixture digest must receive separately reviewed custody/authorization.
Never put original RACE passages, reviews, personal identity/consent, private
keys, signing receipts or raw native privileged test logs into public Git.

## Safe handoff behavior

The accompanying ZIP contains only frozen source-free specifications, this
request and a source-free Lenovo claims fixture. SHA256 file checks only prove
packet *byte consistency*, NOT its publisher's identity, independence,
endorsement or freshness. The on-controller packager requires an explicit
user-local output path, creates no default release, and cannot configure a
witness, send mail or grant model/production privileges.

The human custodian must decline or return a clear HOLD whenever independence,
funding/permission, native durability, external publication, privileged
negative tests, signer root custody, or source-blind review cannot be proven.
There is **no automatic approval path** in this packet.

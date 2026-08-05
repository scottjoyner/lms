# LMS Agent Benchmarking Toolkit Documentation

## Core design

- [`HIGH_LEVEL_DESIGN.md`](HIGH_LEVEL_DESIGN.md) — system context, non-admitting evidence boundary, fleet observation, exact-loadout qualification, runtime canary, rollback, attestation, and architectural decisions.
- [`LOW_LEVEL_DESIGN.md`](LOW_LEVEL_DESIGN.md) — CLI structure, census and locks, subprocess policy, fingerprints, run directories, qualification orchestration, soak calculations, rollback state machine, manifests, signatures, prompt-cache registry, and tests.

## Operator and evidence references

- [`DETERMINISTIC_FLEET_OPERATOR.md`](DETERMINISTIC_FLEET_OPERATOR.md)
- [`FLEET_OPERATIONAL_RELIABILITY.md`](FLEET_OPERATIONAL_RELIABILITY.md)
- [`PHYSICAL_FLEET_ROLLOUT.md`](PHYSICAL_FLEET_ROLLOUT.md)
- [`EXACT_LOADOUT_QUALIFICATION.md`](EXACT_LOADOUT_QUALIFICATION.md)
- [`LOADOUT_QUALIFICATION_OPERATOR.md`](LOADOUT_QUALIFICATION_OPERATOR.md)
- [`RUNTIME_CANARY_AND_ROLLBACK.md`](RUNTIME_CANARY_AND_ROLLBACK.md)
- [`OPERATIONAL_GAP_AUDIT.md`](OPERATIONAL_GAP_AUDIT.md)

The root [`README.md`](../README.md) remains the command and quick-start reference. The HLD and LLD are the canonical design pair and must be updated when evidence identity, command safety, qualification, rollback, manifest, or attestation contracts change.

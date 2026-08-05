# Physical fleet rollout

The supported observation path is `lms-fleet-operator`. Do not hand-compose per-node rollout and release-gate commands.

See `docs/DETERMINISTIC_FLEET_OPERATOR.md` for the exact setup and one-command workflow.

## Current scope

Nine runnable nodes:

```text
destroyer
beelink-ryzen-7-mini-pc
deathstar-xps-8920
scott-lenovo-ideapad-330s-15ikb
scott-optiplex-9030-aio
scotts-macbook-air
scotts-macbook-pro-2
x1-370
xwing
```

One deferred node:

```text
joyner
```

`joyner` remains in the census but is not contacted while powered off. The Raspberry Pi and iPhone are not fleet inference nodes.

## Exact observation command

```bash
lms-fleet-operator observe \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --workspace ~/lms-fleet-runs \
  --update-code
```

This single command validates coverage, preflights all nine Tailscale SSH targets, verifies remote source and paths, renders scripts, collects observation artifacts, and executes the observation release gate.

The run fails closed when any runnable node fails preflight or evidence gating. It preserves diagnostics but never admits or routes a runtime.

## Candidate sweeps

Observation does not execute candidates. Candidate IDs must be taken from collected plans and explicitly reviewed. Do not allow an agent to invent or substitute a candidate ID.

Every sweep remains subject to strict protocol checks, complete repeated trials, raw-output integrity, reliability thresholds, archive verification, and non-admission.

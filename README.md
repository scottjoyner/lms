# LMS Agent Benchmarking Toolkit

LMS profiles and benchmarks local or Tailscale-reachable OpenAI-compatible inference nodes. It produces deterministic benchmark evidence, guarded physical execution, census validation, and non-admitted runtime recommendations.

> The package intentionally does not install a command named `lms`; that name belongs to LM Studio's official CLI.

## Installed commands

```text
lms-agent             Agent-facing doctor/probe/profile/quick/route CLI
lms-bench             Manifest-driven benchmark CLI
lms-fleet              Hardware observation, planning, and selection
lms-fleet-bench        Guarded loopback-only candidate execution
lms-fleet-models       Model inventory and selected-model hashing
lms-fleet-rollout      Low-level census-validated SSH rollout
lms-fleet-operator     Deterministic preflight/render/observe/gate workflow
lms-fleet-gate         Collected-archive release gate
```

## Install

```bash
python3 -m pip install -e '.[test]'
```

## Current fleet scope

The operator-confirmed fleet contains ten nodes. Nine are runnable now:

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

`joyner` remains in the census as `benchmark_deferred` because it is powered off. It is not silently deleted and must return to `benchmark_required` when it comes online.

The Raspberry Pi and iPhone are not fleet inference nodes and are not part of this census or rollout.

Canonical files:

```text
examples/fleet-benchmark-census.v1.json
examples/fleet-rollout.full-fleet.template.json
examples/fleet-rollout.full-fleet.env.example
docs/DETERMINISTIC_FLEET_OPERATOR.md
```

## Private setup

Create the private configuration outside the repository:

```bash
mkdir -p ~/.config/lms-fleet ~/lms-fleet-runs
cp examples/fleet-rollout.full-fleet.template.json \
  ~/.config/lms-fleet/full-fleet.json
cp examples/fleet-benchmark-census.v1.json \
  ~/.config/lms-fleet/fleet-benchmark-census.v1.json
cp examples/fleet-rollout.full-fleet.env.example \
  ~/.config/lms-fleet/full-fleet.env
chmod 600 ~/.config/lms-fleet/full-fleet.env
$EDITOR ~/.config/lms-fleet/full-fleet.env
```

The environment file contains only:

```text
LMS_EXPECTED_COMMIT
LMS_FLEET_SSH_USER
LMS_LINUX_REPO_DIR
LMS_LINUX_PYTHON
LMS_LINUX_MODEL_ROOT
LMS_MACOS_REPO_DIR
LMS_MACOS_PYTHON
LMS_MACOS_MODEL_ROOT
```

Node SSH targets are generated as `<user>@<canonical-node-id>`; operators do not manually compose nine target strings.

## One-command observation run

After filling the private environment file, run exactly:

```bash
lms-fleet-operator observe \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --workspace ~/lms-fleet-runs \
  --update-code
```

The operator performs, in order:

1. Complete census and configuration validation.
2. A fixed non-interactive Tailscale SSH preflight against all nine nodes.
3. Remote repository, Python, model-root, and clean-working-tree checks.
4. Rendered-script generation.
5. Observation-only rollout and artifact collection.
6. A release gate requiring successful evidence from every runnable node.
7. A durable `operator-state.json` plus per-stage logs.

Any unexpected preflight failure stops the rollout before remote observation begins. During collection, failures are retained diagnostically, but the final gate still fails unless every required node produced valid evidence.

Use a preflight-only pass when correcting paths:

```bash
lms-fleet-operator preflight \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --workspace ~/lms-fleet-runs \
  --update-code
```

`--update-code` permits the guarded rollout to fast-forward the reviewed branch. Without it, every remote checkout must already be on the exact configured branch and commit. A dirty remote checkout is always rejected.

## Reliability contract

A selectable candidate requires exact model identity, strict streaming completion, complete sample accounting, unique retained raw outputs, at least three valid complete trials, bounded retries, Wilson confidence, bounded TPS/TTFT dispersion, post-trial health, memory headroom, and no observed crash.

All evidence remains non-admitted. No operator command modifies routers, registries, persistent services, or live admission.

See:

```text
docs/DETERMINISTIC_FLEET_OPERATOR.md
docs/BENCHMARK_RELIABILITY.md
docs/PHYSICAL_FLEET_ROLLOUT.md
```

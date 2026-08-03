# Deterministic fleet operator

This is the supported physical observation path. It exists so an operator or automation system does not construct a sequence of ad-hoc SSH, rollout, and release-gate commands.

## Scope

Runnable nodes:

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

Deferred node:

```text
joyner
```

`joyner` remains in the census but is not contacted while powered off. The Raspberry Pi and iPhone are not fleet inference nodes.

## Safety properties

`lms-fleet-operator`:

- accepts a complete checked-in JSON structure plus one private environment file;
- uses a fixed non-interactive SSH option set;
- derives SSH targets from canonical node IDs;
- never invokes `shell=True` locally;
- validates the complete census before selecting or contacting nodes;
- requires all runnable nodes to pass preflight before rollout begins;
- rejects dirty remote repositories;
- checks repository, branch/commit policy, Python, `requests`, and model roots;
- renders scripts before execution;
- preserves diagnostic output when a remote stage fails;
- requires the observation release gate to pass for every runnable node;
- holds a controller-side lock so two fleet runs cannot overlap;
- records the exact commands and stage outputs under one run directory;
- never selects candidate IDs, admits runtimes, or changes routing.

## Prepare private configuration

From the reviewed LMS checkout:

```bash
mkdir -p ~/.config/lms-fleet ~/lms-fleet-runs
cp examples/fleet-rollout.full-fleet.template.json \
  ~/.config/lms-fleet/full-fleet.json
cp examples/fleet-benchmark-census.v1.json \
  ~/.config/lms-fleet/fleet-benchmark-census.v1.json
cp examples/fleet-rollout.full-fleet.env.example \
  ~/.config/lms-fleet/full-fleet.env
chmod 600 ~/.config/lms-fleet/full-fleet.env
```

Edit `~/.config/lms-fleet/full-fleet.env` once. Do not pass path or host overrides interactively during a run.

The template assumes one common Linux checkout/Python/model root and one common macOS checkout/Python/model root. When a machine differs, edit the private JSON copy once and retain it as reviewed operator configuration.

## Preflight only

```bash
lms-fleet-operator preflight \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --workspace ~/lms-fleet-runs \
  --update-code
```

The report is written to:

```text
~/lms-fleet-runs/preflight.json
```

Every result must have `ok=true`. Fix the private configuration or remote checkout rather than bypassing a failed node.

## Full observation run

```bash
lms-fleet-operator observe \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --workspace ~/lms-fleet-runs \
  --update-code
```

A successful run creates:

```text
~/lms-fleet-runs/<UTC_RUN_ID>/
  operator-state.json
  preflight.json
  render/
  observe/
    rollout_results.json
    release-gate.json
    artifacts/
  logs/
    render.log
    observe.log
    gate.log
```

Success requires:

```text
operator-state.json: success=true
operator-state.json: gate_returncode=0
release-gate.json: passed=true
admission.admitted=false
```

## Exact source behavior

With `--update-code`, preflight still requires a clean repository. The guarded rollout then performs a fast-forward-only update and the provenance gate requires the exact configured 40-character commit.

Without `--update-code`, preflight requires the branch and commit to already match before any rollout starts.

No mode permits a dirty checkout.

## Failure behavior

- A preflight failure prevents observation from starting on every node.
- A render failure stops before SSH rollout.
- A remote rollout failure preserves logs and any collected diagnostic archives.
- A release-gate failure leaves the run non-admitted and unsuitable for profile import.
- The controller lock prevents an overlapping run from starting in the same workspace.

Do not manually rerun fragments from the logs. Correct the cause and start a new operator run so the evidence chain remains coherent.

## Candidate sweeps

Observation does not execute inference candidates. Candidate sweeps remain a separate reviewed phase because candidate IDs must come from collected `benchmark_plan.json` artifacts and must be explicitly approved. Do not allow an agent to invent or substitute candidate IDs.

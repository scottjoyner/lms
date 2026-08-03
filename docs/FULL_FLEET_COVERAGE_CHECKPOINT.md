# Full-fleet benchmark coverage checkpoint

The controller-confirmed fleet contains ten inference nodes.

## Runnable now

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

## Deferred

```text
joyner
```

`joyner` is powered off. It remains in the census as `benchmark_deferred` and must return to `benchmark_required` when online.

The Raspberry Pi and iPhone are not fleet inference nodes and are intentionally absent from the census.

## Coverage contract

A valid current configuration reports:

```text
coverage_complete=true
benchmark_interface_complete=true
current_execution_scope_complete=false
fleet_device_count=10
benchmark_required_count=9
benchmark_deferred_count=1
configured_benchmark_count=9
accounted_device_count=10
benchmark_deferred_node_ids=[joyner]
```

The deferred node is a qualification blocker, not an excuse to delete the machine from the fleet history.

## Operator contract

The supported observation path is one deterministic command:

```bash
lms-fleet-operator observe \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --workspace ~/lms-fleet-runs \
  --update-code
```

All nine runnable nodes must pass fixed SSH preflight before rollout begins. The release gate then requires successful observation evidence from all nine.

Fleet coverage, benchmark reliability, profile import, and live admission remain separate gates. None of these artifacts admits or routes a runtime.

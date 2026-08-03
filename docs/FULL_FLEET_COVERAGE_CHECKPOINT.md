# Full-fleet benchmark coverage checkpoint

This checkpoint supersedes any interpretation that the three-node Tier-1 tranche represents the complete fleet.

## Census

The controller accounts for 11 devices:

- ten `benchmark_required` nodes;
- one explicitly `unsupported` iOS device.

The required benchmark nodes are:

```text
destroyer
raspberrypi
beelink-ryzen-7-mini-pc
deathstar-xps-8920
scott-lenovo-ideapad-330s-15ikb
scott-optiplex-9030-aio
scotts-macbook-air
scotts-macbook-pro-2
x1-370
xwing
```

`iphone-12-pro-max` remains census-accounted and carries a recorded exclusion reason because the current runner requires a remotely executable OpenAI-compatible runtime plus filesystem evidence collection.

## Fail-closed rule

The canonical rollout uses `coverage_mode=full`. Before rendering or SSH, the public rollout command verifies that:

- all ten required nodes are present;
- no configured node is absent from the census;
- unsupported devices are not configured as benchmark nodes;
- the census contains unique nodes and valid policies;
- unsupported devices have explicit reasons.

The three-node Tier-1 template is retained only as `coverage_mode=partial`. Its validation report is intentionally incomplete.

## Physical completion rule

A required node is not removed because it is offline, slow, model-less, or lacks a viable runtime. It must produce either:

1. a passing reliability-qualified sweep and release gate; or
2. an explicit reviewed remediation record.

A remediation record is diagnostic state, not routable capacity.

## Privacy

The public census and templates omit device IDs, account emails, private domains, Tailscale IP addresses, endpoints, SSH targets, and filesystem paths. Those values belong only in a private environment file.

## Admission

Fleet coverage, benchmark reliability, profile import, and live admission are distinct gates. None of these artifacts admits or routes a runtime.

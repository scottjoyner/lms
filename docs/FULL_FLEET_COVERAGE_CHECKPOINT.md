# Full-fleet benchmark coverage checkpoint

This checkpoint supersedes any interpretation that the three-node Tier-1 tranche represents the complete fleet.

## Census

The controller accounts for all 11 devices:

- ten `benchmark_required` remote-runner nodes;
- one `adapter_required` iOS device;
- zero permanently unsupported devices.

The existing remote benchmark targets are:

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

`iphone-12-pro-max` remains in benchmark scope. The current SSH/filesystem runner cannot control an iOS-local inference runtime, so issue #9 tracks the signed mobile benchmark adapter required before qualification can complete.

## Fail-closed rule

The canonical rollout uses `coverage_mode=full`. Before rendering or SSH, the public rollout command verifies that:

- all ten current remote-runner nodes are present;
- no configured node is absent from the census;
- adapter-required or unsupported devices are not configured as SSH rollout nodes;
- the census contains unique nodes and valid policies;
- adapter-required and unsupported policies have explicit reasons.

The three-node Tier-1 template remains only as `coverage_mode=partial`. Its validation report is intentionally incomplete.

## Coverage versus qualification

A valid current configuration reports:

```text
coverage_complete=true
benchmark_interface_complete=false
fleet_device_count=11
benchmark_required_count=10
adapter_required_count=1
configured_benchmark_count=10
accounted_device_count=11
adapter_required_node_ids=[iphone-12-pro-max]
```

Configuration coverage is complete because every device is explicitly accounted for. Fleet benchmark qualification is not complete until the mobile adapter exists and produces physical reliability evidence.

## Physical completion rule

A census node is not removed because it is offline, slow, model-less, lacks a viable runtime, or requires another controller interface.

Each remote-runner node must produce either:

1. a passing reliability-qualified sweep and release gate; or
2. an explicit reviewed remediation record.

The iPhone must produce a passing physical mobile-adapter reliability artifact. A remediation or adapter blocker is diagnostic state, not routable capacity.

## Privacy

The public census and templates omit device IDs, account emails, private domains, Tailscale IP addresses, endpoints, SSH targets, and filesystem paths. Those values belong only in private configuration and evidence handling.

## Admission

Fleet coverage, benchmark reliability, profile import, and live admission are distinct gates. None of these artifacts admits or routes a runtime.

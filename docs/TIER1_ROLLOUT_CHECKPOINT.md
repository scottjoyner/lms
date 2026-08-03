# Tier-1 partial rollout checkpoint

The Tier-1 template contains only:

```text
x1-370
xwing
scotts-macbook-air
```

It remains `coverage_mode=partial` and is not the supported complete observation workflow.

The current fleet contains nine runnable nodes and one deferred node (`joyner`). The Raspberry Pi and iPhone are not fleet inference nodes.

Use `lms-fleet-operator` with the complete private configuration instead of manually expanding the Tier-1 sequence:

```bash
lms-fleet-operator observe \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --workspace ~/lms-fleet-runs \
  --update-code
```

The Tier-1 template is retained only for regression testing and isolated troubleshooting. A successful Tier-1 run must never be represented as complete fleet evidence or admission.

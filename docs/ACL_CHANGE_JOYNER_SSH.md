# Ready-to-apply tailnet ACL change — unblock SSH to joyner

Date prepared: 2026-08-23
Blocked by: `tailnet policy does not permit you to SSH as user "scott"`
(x1-370 → joyner over Tailscale SSH; regular sshd on joyner has no authorized key)

## Why

Current tailnet policy only allows `autogroup:self` SSH (own devices).
Orchestrator (x1-370) needs cross-node SSH to census/benchmark joyner.

## Apply (admin console, kipnerter@gmail.com tailnet)

Access Controls → add inside the top-level object:

```json
"ssh": [
  {
    "action": "accept",
    "src":    ["x1-370"],
    "dst":    ["joyner", "beelink-ryzen-7-mini-pc"],
    "users":  ["scott"]
  }
]
```

Notes:
- Scoped to the orchestrator only; widen `dst` to other fleet nodes as needed
  (deathstar, destroyer, lenovo, optiplex) if Tailscale SSH should replace
  per-node keys fleet-wide.
- `autogroup:nonroot` can be used instead of `"scott"` if any non-root user is
  acceptable.
- After saving, verify from x1-370:

```bash
tailscale ssh scott@joyner 'hostname'
# then re-run the census:
bash ~/git/lms/scripts/refresh-fleet-routing-matrix.sh
```

## Also required on joyner itself (once reachable or at console)

```bash
sudo tailscale set --ssh
```

Reference: docs/tailscale_ssh.md in this repo.

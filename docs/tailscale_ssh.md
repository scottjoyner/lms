# Tailscale SSH setup (fleet-wide)

Standardize fleet access on Tailscale SSH so the orchestrator/reporter deploy
script (`deploy_reporter.sh`) can reach every node uniformly with
`tailscale ssh <host>` — no per-node SSH keys to manage. This is what unblocks
the remaining nodes (beelink, joyner, avbells-macbook-pro, and the offline Macs)
that currently reject the `hermes-agent-key`.

## 1. Enable SSH in the tailnet (admin console)

- Go to your Tailscale admin (e.g. `login.tailscale.com` or your custom domain)
  → **Access → SSH** → turn it **on**.

## 2. Add an `ssh` policy rule

Edit the tailnet **Policy** (Access → Policy) and add an `ssh` section, e.g.
let tailnet members SSH to any node as a non-root user:

```json
"ssh": [
  {
    "action": "accept",
    "src":    ["autogroup:members"],
    "dst":    ["autogroup:self"],
    "users":  ["autogroup:nonroot"]
  }
]
```

Adjust `src`/`dst` if you want tighter scoping (e.g. only the orchestrator host
may initiate).

## 3. Enable SSH on each device's tailscaled

- **Linux:** `sudo tailscale set --ssh` (or ensure the service runs with `--ssh`).
- **macOS:** Tailscale menu → **Allow Tailscale SSH**, or `sudo tailscale set --ssh`.

Each device must be logged into the tailnet with the identity permitted by the
policy (the hermes-agent / tailnet owner account).

## 4. Connect

```bash
tailscale ssh beelink
tailscale ssh joyner
tailscale ssh avbells-macbook-pro
```

Once this works, the reporter can be deployed to every remaining node with the
same command used elsewhere:

```bash
cd /home/scott/git/lms
./deploy_reporter.sh beelink            # Tailscale SSH resolves the host
./deploy_reporter.sh joyner
./deploy_reporter.sh avbells-macbook-pro
# offline Macs (scotts-macbook-pro, scotts-macbook-pro-2) once they are online:
./deploy_reporter.sh scotts-macbook-pro
./deploy_reporter.sh scotts-macbook-pro-2
```

The reporter reports real per-node specs, so the orchestrator's `plan` will stop
showing `[model_fit(CONTAMINATED)]` for those nodes.

## Notes / blockers

- The only thing blocking fleet-wide reporter coverage today is **access**, not
  code: beelink/joyner reject the `hermes-agent-key`, and avbells-macbook-pro has
  no SSH config entry. Tailscale SSH removes all of that.
- For `tailscale ssh` to succeed, each node's `tailscaled` must run with `--ssh`
  **and** the admin policy must allow the connecting identity.
- macOS nodes use the launchd path in `deploy_reporter.sh` automatically
  (detected via `uname`); Linux nodes use the systemd user service path.

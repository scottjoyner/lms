# Systemd user units for lms

Copied from `~/.config/systemd/user/` (and normalized for packaging) so the fleet
graph/session sync + node reporter services are version-controlled. See
docs/LLD_UNIFIED_FLEET.md W-74.

Install (user-level, no sudo):

```sh
mkdir -p ~/.config/systemd/user
cp deploy/systemd/*.service deploy/systemd/*.timer ~/.config/systemd/user/
# Provide NEO4J_PASSWORD + ROUTER_URL via environment drop-ins (NOT hardcoded):
systemctl --user import-environment NEO4J_PASSWORD ROUTER_URL
systemctl --user daemon-reload
systemctl --user enable --now fleet-graph-sync.service
systemctl --user enable --now fleet-session-sync.timer
```

Units reference the installed `lms_agent_bench` package (`python3 -m lms_agent_bench...`)
and the repo-root `fleet_node_reporter.py` via `LMS_REPO`. `NEO4J_PASSWORD` is read
from the environment — never committed (W-71).

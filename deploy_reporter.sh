#!/usr/bin/env bash
# Deploy the fleet node reporter to a remote tailnet node as a *user-level* service
# (no sudo required). The reporter phones the auto-router's pubsub fleet endpoint
# every --interval seconds with this node's LM Studio library + loaded models +
# real specs, so the orchestrator can plan/benchmark/mount across the fleet.
#
# The script auto-detects the remote platform:
#   Linux  -> systemd user service (fleet-node-reporter.service)
#   macOS  -> launchd agent (~/Library/LaunchAgents/com.hermes.fleet-node-reporter.plist)
#
# Usage:
#   ./deploy_reporter.sh <user@host> [router-url]
#
# router-url defaults to x1-370 (this host) over tailscale. Run from the lms dir.
set -euo pipefail

TARGET="${1:?usage: deploy_reporter.sh <user@host> [router-url]}"
# Router URL is parameterized (no hardcoded personal IP). Override via ROUTER_URL.
ROUTER_URL="${2:-${ROUTER_URL:-http://100.64.43.123:8088}}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPORTER="$HERE/fleet_node_reporter.py"

if [[ ! -f "$REPORTER" ]]; then
  echo "fleet_node_reporter.py not found next to this script ($REPORTER)" >&2
  exit 1
fi

echo "Deploying fleet node reporter to $TARGET (router=$ROUTER_URL)"

# Detect remote platform once.
REMOTE_UNAME="$(ssh -o StrictHostKeyChecking=no "$TARGET" 'uname' 2>/dev/null || echo Unknown)"

if [[ "$REMOTE_UNAME" == "Darwin" ]]; then
  echo "Remote is macOS -> installing launchd agent"
  RHOME="$(ssh -o StrictHostKeyChecking=no "$TARGET" 'echo $HOME')"
  ssh -o StrictHostKeyChecking=no "$TARGET" 'bash -s' <<RMT
set -e
pkill -f fleet_node_reporter.py 2>/dev/null || true
mkdir -p ~/Library/LaunchAgents
cat > ~/fleet_node_reporter.py <<'PYEOF'
$(cat "$REPORTER")
PYEOF
cat > ~/Library/LaunchAgents/com.hermes.fleet-node-reporter.plist <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hermes.fleet-node-reporter</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>${RHOME}/fleet_node_reporter.py</string>
        <string>--router-url</string>
        <string>$ROUTER_URL</string>
        <string>--interval</string>
        <string>30</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${RHOME}/fleet-node-reporter.log</string>
    <key>StandardErrorPath</key>
    <string>${RHOME}/fleet-node-reporter.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
</dict>
</plist>
EOF
launchctl unload ~/Library/LaunchAgents/com.hermes.fleet-node-reporter.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.hermes.fleet-node-reporter.plist
sleep 2
echo "status: \$(launchctl list | grep com.hermes.fleet-node-reporter || echo 'not-loaded')"
RMT
else
  echo "Remote is Linux (or unknown) -> installing systemd user service"
  ssh -o StrictHostKeyChecking=no "$TARGET" 'bash -s' <<RMT
set -e
# stop any ad-hoc background loop we may have launched earlier
pkill -f fleet_node_reporter.py 2>/dev/null || true
mkdir -p ~/.config/systemd/user
cat > ~/fleet_node_reporter.py <<'PYEOF'
$(cat "$REPORTER")
PYEOF
cat > ~/.config/systemd/user/fleet-node-reporter.service <<EOF
[Unit]
Description=Fleet node reporter (pubsub model library/health to auto-router)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 %h/fleet_node_reporter.py --router-url $ROUTER_URL --interval 30
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now fleet-node-reporter
# keep the service running even when this user is logged out
loginctl enable-linger "$USER" 2>/dev/null || true
echo "status: \$(systemctl --user is-active fleet-node-reporter)"
RMT
fi

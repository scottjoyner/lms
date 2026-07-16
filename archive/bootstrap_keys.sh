#!/bin/bash
# Run this in a terminal on a fleet node (via RustDesk / tailscale SSH).
# Installs the runner's PUBLIC keys so the runner can SSH in and collect
# hardware. The keys are public halves only (safe to distribute); override
# them via RUNNER_KEYS / RUNNER_KEY2 if you rotate.
set -e
mkdir -p ~/.ssh && chmod 700 ~/.ssh

# Defaults are the runner's published PUBLIC ed25519 keys. Override via env.
RUNNER_KEYS="${RUNNER_KEYS:-ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKPYyFx2ucZylkPOV7xEDRor/5JyLgzzKLqSq4hTeKbH hermes-agent@x1-370}"
RUNNER_KEY2="${RUNNER_KEY2:-ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOkodatvrCSygQIGtV4nH2wPQ0ws+KNexc+Aa0WeDkSy hermes@kipnerter}"

cat >> ~/.ssh/authorized_keys <<EOF
$RUNNER_KEYS
$RUNNER_KEY2
EOF
chmod 600 ~/.ssh/authorized_keys
echo "Keys installed on $(hostname). Runner can now SSH in."

#!/bin/bash
set -x
set -euo pipefail

# -------------------------------------------------------
# Bootstrap a GitHub Actions self-hosted runner on Ubuntu
# -------------------------------------------------------

RUNNER_USER="runner"
RUNNER_HOME="/home/$RUNNER_USER"
RUNNER_DIR="$RUNNER_HOME/actions-runner"

# --- System packages & Docker ---
apt-get update -y
apt-get install -y \
  curl jq git unzip docker.io docker-buildx awscli

systemctl enable --now docker

# Create a non-root user for the runner
useradd -m -s /bin/bash "$RUNNER_USER"
usermod -aG docker "$RUNNER_USER"

# --- Install the GitHub Actions runner ---
mkdir -p "$RUNNER_DIR" && cd "$RUNNER_DIR"
curl -o actions-runner-linux-x64-2.331.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.331.0/actions-runner-linux-x64-2.331.0.tar.gz
echo "5fcc01bd546ba5c3f1291c2803658ebd3cedb3836489eda3be357d41bfcf28a7  actions-runner-linux-x64-2.331.0.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-x64-2.331.0.tar.gz
chown -R "$RUNNER_USER":"$RUNNER_USER" "$RUNNER_DIR"

sudo -u "$RUNNER_USER" bash -c "
  ./config.sh \
    --url 'https://github.com/${github_repo}' \
    --token '${github_pat}' \
    --name '${runner_name}' \
    --labels '${runner_labels}' \
    --unattended \
    --replace
"

# --- Install and start as a systemd service under the runner user ---
cd "$RUNNER_DIR"
./svc.sh install "$RUNNER_USER"
./svc.sh start

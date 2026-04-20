#!/bin/bash
# RunPod bootstrap: install deps, then start the probe service.
# Paste this into "Container Start Command" on a PyTorch 2.x / CUDA 12 template,
# or run it after shelling into the pod.
#
# Expects the repo to be mounted/cloned at /workspace/probe with artifacts/ present.
# If not, edit PROBE_REPO / ARTIFACT_URL below.

set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace/probe}
PROBE_CONFIG=${PROBE_CONFIG:-$REPO_DIR/artifacts/probe_config.json}
PORT=${PORT:-8000}

if [ ! -d "$REPO_DIR" ]; then
    echo "[bootstrap] cloning repo into $REPO_DIR"
    git clone https://github.com/sundi133/probe.git "$REPO_DIR"
fi

cd "$REPO_DIR"

echo "[bootstrap] installing python deps"
pip install --no-cache-dir -r service/requirements.txt

if [ ! -f "$PROBE_CONFIG" ]; then
    echo "[bootstrap] no probe_config.json at $PROBE_CONFIG"
    echo "[bootstrap] run the training pipeline first: ./run_pipeline.sh"
    echo "[bootstrap] or scp artifacts/ into $REPO_DIR/artifacts/"
    exit 1
fi

echo "[bootstrap] probe_config: $PROBE_CONFIG"
nvidia-smi || echo "[bootstrap] warning: nvidia-smi not found, service will fall back to CPU"

export PROBE_CONFIG PORT
exec python service/server.py

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo0}"
else
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
fi

python run_all.py "$@"

#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

export GPIOZERO_PIN_FACTORY="${GPIOZERO_PIN_FACTORY:-lgpio}"
export PYTHONUNBUFFERED=1

if [[ ! -x venv/bin/python || ! -f venv/bin/activate ]]; then
    echo "ERROR: Python virtual environment is missing: $BASE_DIR/venv" >&2
    echo "Run ./setup.sh first. If setup previously stopped early, rerun the updated setup.sh." >&2
    exit 2
fi

# shellcheck disable=SC1091
source venv/bin/activate
exec python agent.py

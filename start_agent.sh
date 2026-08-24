#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

# Pi 5 uses the RP1 GPIO controller; gpiozero's lgpio backend handles it.
export GPIOZERO_PIN_FACTORY="${GPIOZERO_PIN_FACTORY:-lgpio}"
export PYTHONUNBUFFERED=1

source venv/bin/activate
exec python agent.py

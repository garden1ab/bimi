#!/bin/bash
set -euo pipefail

MODEL="${THOR_MODEL:-qwen3.5:27b}"

echo "Configuring Jetson AGX Thor as a LAN Ollama LLM/VLM server."
echo "This exposes port 11434 on the LAN; restrict access with your firewall/router."

if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_KEEP_ALIVE=-1m"
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now ollama
ollama pull "$MODEL"

echo
hostname -I | awk '{print "Thor LAN IP: "$1}'
echo "Model: $MODEL"
echo "Set config.json -> ai.backends.thor.base_url to http://<THOR-IP>:11434 if jetson-thor.local does not resolve."

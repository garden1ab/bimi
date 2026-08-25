# Adaptive wake-word recording fix

This patch replaces the fixed RMS silence detector with an adaptive noise-floor VAD.

Default behavior after a wake-word trigger:
- Calibrate microphone/room noise for 0.35 s.
- Wait up to 6 s for speech to begin.
- Require 2 consecutive speech chunks to reject clicks/noise.
- Stop after 1.0 s of trailing silence.
- Hard-stop at 20 s as a safety limit.
- Keep 0.25 s of pre-roll so the beginning of the utterance is retained.

Settings are under `recording` in `config.json`.

Useful tuning:
- If it still never stops: increase `end_noise_multiplier` from 1.7 to 2.0 or 2.2.
- If it cuts off quiet words: reduce `end_noise_multiplier` toward 1.4-1.5 or increase `end_silence_seconds`.
- If it fails to notice you started talking: lower `noise_multiplier` from 2.4 toward 2.0.


1. Clean out the previous Ollama server attempts

Run these commands on the Jetson AGX Thor:

docker rm -f ollama-thor ollama 2>/dev/null || true
sudo systemctl stop ollama 2>/dev/null || true
sudo pkill -f "ollama serve" 2>/dev/null || true

Now check port 11434:

sudo ss -lntp | grep 11434

At this stage it should return nothing.

Also verify there are no containers still running:

docker ps
2. Verify the Thor itself first

Run:

uname -m

Expected:

aarch64

Then:

cat /etc/nv_tegra_release

And most importantly:

nvidia-smi

You need nvidia-smi to actually display the NVIDIA Thor GPU before continuing. Thor uses NVIDIA's SBSA GPU driver, and NVIDIA specifically recommends nvidia-smi for monitoring the Thor GPU.

Something resembling:

NVIDIA-SMI ...
Driver Version: ...
CUDA Version: ...

GPU  Name
0    NVIDIA Thor

is what matters.

If:

No devices were found

appears, stop there—the GPU/JetPack installation needs to be repaired before Ollama.

3. Optional: put Thor into maximum-performance mode

For testing:

sudo nvpmodel -m 0

Then check:

sudo nvpmodel -q

If it tells you a reboot is required:

sudo reboot
4. Install native Ollama

After reboot:

sudo apt update
sudo apt install -y curl

Then install Ollama using the official installer:

curl -fsSL https://ollama.com/install.sh | sh

Both NVIDIA's Jetson AI Lab and Ollama's own Linux documentation support this installation method. The installer creates a systemd service for ollama serve.

Check it:

ollama --version

Then:

sudo systemctl status ollama --no-pager

You want:

active (running)
5. First test Ollama locally

Before changing any networking:

curl http://127.0.0.1:11434/api/version

Then:

curl http://127.0.0.1:11434/api/tags

If these fail, do not move on to the Pi yet.

Check the logs:

journalctl -u ollama --no-pager -n 100

Ollama documents journalctl -u ollama as the standard Linux server-log location.

6. Expose Ollama to your LAN

This is probably the step your current setup is missing.

By default, Ollama listens only on:

127.0.0.1:11434

which means Thor itself can connect, but your:

Raspberry Pi    ✗
Phone           ✗
Other PCs       ✗

cannot.

Ollama officially uses OLLAMA_HOST=0.0.0.0:11434 to expose the server to the network.

Create a systemd override:

sudo mkdir -p /etc/systemd/system/ollama.service.d

Then:

sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF

Reload systemd:

sudo systemctl daemon-reload
sudo systemctl restart ollama
sudo systemctl enable ollama

Now check:

sudo ss -lntp | grep 11434
This is the critical result

Good:

0.0.0.0:11434

or:

[::]:11434

Bad:

127.0.0.1:11434

If you still see 127.0.0.1, run:

sudo systemctl cat ollama

and verify you see:

Environment="OLLAMA_HOST=0.0.0.0:11434"
7. Open the firewall

Check:

sudo ufw status

If it's active:

sudo ufw allow 11434/tcp

Then:

sudo ufw reload

NVIDIA's own Jetson VLM troubleshooting specifically calls out opening TCP port 11434 when remote Ollama clients cannot connect.

Do not forward port 11434 through your internet router. Ollama's local API does not require authentication, so you want this accessible only inside your trusted LAN.

8. Find Thor's actual LAN address

Use this rather than .local hostnames:

ip route get 1.1.1.1

You'll see something resembling:

1.1.1.1 via 192.168.1.1 dev eth0 src 192.168.1.82

The important part is:

src 192.168.1.82

So your Thor IP would be:

192.168.1.82

You can extract it automatically:

THOR_IP=$(ip route get 1.1.1.1 | awk '{print $7; exit}')
echo "$THOR_IP"
9. Test the LAN IP from Thor itself

Still on Thor:

curl http://$THOR_IP:11434/api/tags

This is an important test.

You now need both of these to work:

curl http://127.0.0.1:11434/api/tags

and:

curl http://$THOR_IP:11434/api/tags

If localhost works but the LAN address doesn't, the server is still not properly bound or the host firewall is interfering.

10. Install Qwen3.6 35B

Now install your model:

ollama pull qwen3.6:35b

Ollama currently lists qwen3.6:35b as a 36B MoE model, approximately 24 GB in the standard Q4_K_M build, with a 256K context window and text/image support.

Verify:

ollama list

You should see:

qwen3.6:35b
11. Test Qwen directly on Thor

Run:

curl http://127.0.0.1:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6:35b",
    "messages": [
      {
        "role": "user",
        "content": "Reply exactly: Thor Qwen server is working"
      }
    ],
    "stream": false,
    "think": false,
    "keep_alive": -1
  }'

You should get a response containing:

Thor Qwen server is working

Remember:

"keep_alive": -1

is a number.

Do not use:

"keep_alive": "-1"

because that caused your previous duration parsing error.

12. Verify it is actually using Thor's GPU

Open another terminal on Thor:

watch -n 1 nvidia-smi

Then run another Qwen request.

GPU utilization and memory usage should rise.

Also:

ollama ps

should show qwen3.6:35b loaded.

This verifies:

Ollama server
      ↓
Qwen3.6 35B
      ↓
Thor GPU

rather than CPU inference.

13. Test from your phone BEFORE the Raspberry Pi

Put your phone on the same Wi-Fi/LAN as Thor.

In its browser enter:

http://192.168.1.82:11434/api/tags

using your actual Thor IP.

You should get JSON showing your model.

For example:

{
  "models": [
    {
      "name": "qwen3.6:35b"
    }
  ]
}

If this works, we know external LAN access is fixed.

If your phone still cannot connect

The likely problem is now the network, not Thor.

Thor local API	Thor-IP API	Phone	Likely cause
Fail	Fail	Fail	Ollama itself
Works	Fail	Fail	Ollama binding/firewall
Works	Works	Fail	Wi-Fi/VLAN/client isolation
Works	Works	Works	Thor server is correct

If the last remaining failure is the phone, check that it is not connected to:

Guest Wi-Fi
IoT VLAN
isolated SSID
cellular data

Many routers prevent wireless clients on guest networks from reaching LAN devices.

14. Test from the Raspberry Pi

On the Pi:

ping -c 3 192.168.1.82

Then:

curl --connect-timeout 5 \
  http://192.168.1.82:11434/api/tags

Again substitute your IP.

You should see:

qwen3.6:35b

Then test actual inference:

curl http://192.168.1.82:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6:35b",
    "messages": [
      {
        "role": "user",
        "content": "Reply exactly: Raspberry Pi connected to Thor"
      }
    ],
    "stream": false,
    "think": false,
    "keep_alive": -1
  }'

Expected:

Raspberry Pi connected to Thor

At that point do not change anything else on the Thor. The server is proven working.

15. Point BMO at the numeric IP

On the Pi, change BMO's config.json from:

"base_url": "http://jetson-thor.local:11434"

to:

"base_url": "http://192.168.1.82:11434"

with your real IP.

Your Thor section should ultimately resemble:

"thor": {
  "type": "ollama",
  "base_url": "http://192.168.1.82:11434",
  "text_model": "qwen3.6:35b",
  "vision_model": "qwen3.6:35b",
  "keep_alive": -1,
  "think": false,
  "connect_timeout_seconds": 3.0,
  "timeout_seconds": 45.0
}

Then start BMO:

cd ~/be-more-agent-main
source venv/bin/activate
./start_agent.sh

Say:

Hey BMO, switch to Thor.

The corrected build should not announce success unless the Pi can both contact Thor and find qwen3.6:35b.

Then ask:

What model are you using?

And watch the Pi terminal.

You specifically want:

[AI ROUTE] requested=thor actual=thor model=qwen3.6:35b

That is now the definitive test.

The complete working architecture
Raspberry Pi 5
│
├── USB microphone
├── Whisper STT
├── wake word
├── GY-521
├── Pi camera
├── Hailo-8
├── Piper TTS
│
└── Ethernet / Wi-Fi
        │
        ▼
192.168.1.82:11434
        │
        ▼
Jetson AGX Thor
        │
      Ollama
        │
        ▼
Qwen3.6 35B
   ├── LLM
   ├── VLM
   └── tools

The single most important checkpoint is this command on Thor:

sudo ss -lntp | grep 11434

It must show 0.0.0.0:11434 or [::]:11434. Once that does, curl http://THOR_IP:11434/api/tags should work from Thor, then your phone, then the Pi. If one stage fails, that tells us exactly which layer is broken.

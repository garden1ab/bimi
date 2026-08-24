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


5. Pull NVIDIA's Thor Ollama image

Run:

docker pull ghcr.io/nvidia-ai-iot/ollama:r38.2.arm64-sbsa-cu130-24.04

Despite the older-looking R38 tag, this is still the specific Ollama image NVIDIA's current Jetson AI Lab documentation lists for Jetson Thor / SM110.

Create persistent storage:

mkdir -p ~/ollama-data
6. Start the Thor Ollama server

Use:

docker run -d \
  --name ollama-thor \
  --restart unless-stopped \
  --runtime nvidia \
  --network host \
  -e OLLAMA_HOST=0.0.0.0:11434 \
  -e OLLAMA_KEEP_ALIVE=-1m \
  -v "$HOME/ollama-data:/data" \
  ghcr.io/nvidia-ai-iot/ollama:r38.2.arm64-sbsa-cu130-24.04

The important pieces are:

--runtime nvidia

which gives Ollama access to Thor's GPU, and:

--network host

which exposes:

http://THOR-IP:11434

to your Raspberry Pi.

There have been Thor cases where Ollama silently fell back to CPU when the NVIDIA runtime wasn't active, so explicitly specifying --runtime nvidia is important.

Check the server:

docker logs ollama-thor
7. Install Qwen 3.5 27B

Your current Pi configuration expects:

qwen3.5:27b

Install it:

docker exec -it ollama-thor ollama pull qwen3.5:27b

Qwen 3.5 27B is about 17 GB in Ollama's Q4_K_M package, and it supports:

text
images
tool calling
thinking/reasoning

so it can serve as both your Thor LLM and VLM.

Then test:

docker exec -it ollama-thor ollama run qwen3.5:27b

Ask:

What are you?

Exit with:

/bye
8. Make sure the GPU is actually being used

Open a second terminal on the Thor:

sudo tegrastats

Then run another Qwen query.

You should see GPU activity/power increase.

Also run:

docker exec ollama-thor ollama ps

You do not want this:

PROCESSOR
100% CPU

You want GPU usage reported.

Jetson uses tegrastats for monitoring CPU/GPU/memory/thermals.

9. Test the API directly on Thor

Run:

curl http://127.0.0.1:11434/api/tags

You should see JSON containing:

qwen3.5:27b

Then test actual inference:

curl http://127.0.0.1:11434/api/chat \
  -d '{
    "model":"qwen3.5:27b",
    "messages":[
      {
        "role":"user",
        "content":"Reply with exactly: Thor AI online"
      }
    ],
    "stream":false,
    "think":false,
    "keep_alive":-1
  }'

You should get a JSON response containing:

Thor AI online

Notice:

"keep_alive": -1

is numeric. Don't use:

"keep_alive": "-1"

because that caused the duration error you encountered on the Pi.

10. Test from the Raspberry Pi

Now go to your Pi and run:

curl http://jetson-thor.local:11434/api/tags

If that works, you're essentially connected.

If not, use the Thor's IP:

curl http://192.168.1.82:11434/api/tags

substituting your actual Thor address.

Then test Qwen remotely:

curl http://jetson-thor.local:11434/api/chat \
  -d '{
    "model":"qwen3.5:27b",
    "messages":[
      {
        "role":"user",
        "content":"Say Thor connection successful."
      }
    ],
    "stream":false,
    "think":false
  }'
11. Set your Pi configuration

On the Raspberry Pi, your config.json should contain approximately:

"ai": {
  "default_backend": "local",

  "backends": {
    "local": {
      "type": "ollama",
      "base_url": "http://127.0.0.1:11434",
      "text_model": "qwen3.5:2b",
      "vision_model": "qwen3.5:2b"
    },

    "thor": {
      "type": "ollama",
      "base_url": "http://jetson-thor.local:11434",
      "text_model": "qwen3.5:27b",
      "vision_model": "qwen3.5:27b"
    }
  }
}

I would leave:

"default_backend": "local"

because that's what you requested.

Then you can verbally say:

Hey BMO, switch to Thor.

and the route becomes:

USB microphone
      ↓
Raspberry Pi
      ↓
Whisper transcription
      ↓
Jetson Thor
      ↓
Qwen 3.5 27B
      ↓
Raspberry Pi
      ↓
Piper
      ↓
speaker

Say:

Hey BMO, switch to local.

and it returns to Pi inference.

12. Camera/VLM through Thor

This is one of the more useful parts of the setup.

When Thor mode is active and you say:

Hey BMO, what do you see?

the intended pipeline is:

Pi Camera
    ↓
capture image
    ↓
Hailo-8
    ↓
object detections
    │
    ├─────────────────┐
    ↓                 ↓
image JPEG       object hints
    │                 │
    └────────┬────────┘
             ↓
         LAN/Wi-Fi
             ↓
      Jetson AGX Thor
             ↓
       Qwen 3.5 27B
        multimodal
             ↓
    natural-language
      scene analysis
             ↓
       Raspberry Pi
             ↓
          speaker

Qwen 3.5 27B accepts image input, so you don't need a separate dedicated VLM initially.

That simplifies the architecture considerably.

13. I would also install the 35B model

Once 27B works correctly, Thor has enough memory to experiment with:

docker exec -it ollama-thor ollama pull qwen3.5:35b

It's about 24 GB in Ollama and is also multimodal/tool-capable.

Then test:

docker exec -it ollama-thor ollama run qwen3.5:35b

For your robot, I'd compare:

Model	Role
qwen3.5:2b	Pi emergency/offline fallback
qwen3.5:27b	Thor default
qwen3.5:35b	Thor higher-quality option
qwen3.5:122b	Not my first choice for this robot

Even though Thor's large unified memory makes very large models possible, latency matters more than squeezing the largest model into RAM for a conversational robot.

I expect 27B or 35B to be a substantially better balance.

14. Final test sequence

Once everything is installed, I would test in this exact order:

# THOR
docker ps

docker exec ollama-thor ollama list

docker exec ollama-thor ollama ps

curl http://127.0.0.1:11434/api/tags

Then on the Pi:

ping jetson-thor.local

curl http://jetson-thor.local:11434/api/tags

source venv/bin/activate
python hardware_check.py

Then launch:

./start_agent.sh

Say:

Hey BMO, what model are you using?

Then:

Hey BMO, switch to Thor.

Then:

Hey BMO, what model are you using?

Finally:

Hey BMO, look at the camera and describe what you see.

If all four work, the complete Pi ↔ Thor LLM/VLM architecture is operational.

One improvement I would make after that is to have the Pi automatically use Thor whenever it is reachable and plugged into your home network, while preserving Pi-local mode as the fallback, rather than requiring manual switching every time.

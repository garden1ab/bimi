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

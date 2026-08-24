#!/usr/bin/env python3
"""Quick target-hardware diagnostics for the modified Be More Agent."""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import requests


def run(label, command):
    print(f"\n[{label}] {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        output = (result.stdout + result.stderr).strip()
        print(output if output else f"exit={result.returncode}")
        return result.returncode == 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return False


def check_mpu():
    print("\n[GY-521 / MPU-6050]")
    try:
        from smbus2 import SMBus
        with SMBus(1) as bus:
            who = bus.read_byte_data(0x68, 0x75)
        print(f"WHO_AM_I=0x{who:02X} (expected 0x68)")
        return who == 0x68
    except Exception as exc:
        print(f"ERROR: {exc}")
        return False


def check_ollama(name, base_url):
    print(f"\n[{name} AI] {base_url}")
    try:
        response = requests.get(base_url.rstrip("/") + "/api/tags", timeout=4)
        response.raise_for_status()
        models = [m.get("name") or m.get("model") for m in response.json().get("models", [])]
        print("Models:", ", ".join(x for x in models if x) or "none")
        return True
    except Exception as exc:
        print(f"Unavailable: {exc}")
        return False


def main():
    print("Be More Agent hardware check")
    print(f"Platform: {platform.platform()} | machine={platform.machine()}")
    config = json.loads(Path("config.json").read_text(encoding="utf-8"))

    print("\n[Configured wiring]")
    print("GY-521: SDA GPIO2, SCL GPIO3, INT GPIO24")
    print("MAX98357A: DIN GPIO21, BCLK GPIO18, LRCLK GPIO19")

    run("I2C bus", ["i2cdetect", "-y", "1"])
    check_mpu()
    run("Hailo", ["hailortcli", "fw-control", "identify"])
    run("Camera", ["rpicam-still", "--list-cameras"])
    run("ALSA playback", ["aplay", "-l"])
    run("ALSA capture", ["arecord", "-l"])

    backends = config.get("ai", {}).get("backends", {})
    for name in ("local", "thor"):
        backend = backends.get(name, {})
        if backend.get("type", "ollama") == "ollama":
            check_ollama(name.upper(), backend.get("base_url", ""))

    print("\nIf the MAX98357A or I2C overlay was just installed, reboot before treating a failed check as a wiring fault.")


if __name__ == "__main__":
    main()

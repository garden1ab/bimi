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


def check_mpu(config):
    print("\n[GY-521 / MPU-6050]")
    try:
        from smbus2 import SMBus
        bus_id = int(config.get("i2c_bus", 1))
        address = int(str(config.get("address", "0x68")), 0)
        with SMBus(bus_id) as bus:
            who = bus.read_byte_data(address, 0x75)
        print(f"WHO_AM_I=0x{who:02X} (expected 0x68/0x69)")
        if who not in (0x68, 0x69):
            return False

        from hardware import MPU6050Monitor
        monitor = MPU6050Monitor(config)
        if not monitor.start():
            print("Monitor start failed:", monitor.snapshot().get("last_error"))
            return False
        try:
            import time
            print("Keep the robot still for rest calibration, then tilt it if you want to watch the values change.")
            for index in range(3):
                time.sleep(0.4)
                state = monitor.snapshot()
                print(
                    f"sample {index+1}: moving={state['moving']} "
                    f"roll_rest={state['roll_from_rest_deg']}° "
                    f"pitch_rest={state['pitch_from_rest_deg']}° "
                    f"tilt_rest={state['total_tilt_from_rest_deg']}° "
                    f"accel={state['acceleration_g']} gyro={state['gyro_dps']}"
                )
        finally:
            monitor.stop()
        return True
    except Exception as exc:
        print(f"ERROR: {exc}")
        return False


def check_ollama(name, base_url, expected_model=None):
    print(f"\n[{name} AI] {base_url}")
    try:
        response = requests.get(base_url.rstrip("/") + "/api/tags", timeout=4)
        response.raise_for_status()
        models = [m.get("name") or m.get("model") for m in response.json().get("models", [])]
        models = [x for x in models if x]
        print("Models:", ", ".join(models) or "none")
        if expected_model:
            if expected_model in models:
                print(f"Configured model OK: {expected_model}")
            else:
                print(f"ERROR: configured model {expected_model} is not installed on this backend")
                return False
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
    check_mpu(config.get("hardware", {}).get("mpu6050", {}))
    run("Hailo", ["hailortcli", "fw-control", "identify"])
    run("Camera", ["rpicam-still", "--list-cameras"])
    run("ALSA playback", ["aplay", "-l"])
    run("ALSA capture", ["arecord", "-l"])

    backends = config.get("ai", {}).get("backends", {})
    for name in ("local", "thor"):
        backend = backends.get(name, {})
        if backend.get("type", "ollama") == "ollama":
            check_ollama(name.upper(), backend.get("base_url", ""), backend.get("text_model"))

    print("\nIf the MAX98357A or I2C overlay was just installed, reboot before treating a failed check as a wiring fault.")


if __name__ == "__main__":
    main()

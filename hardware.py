"""Raspberry Pi hardware integration for the GY-521 / MPU-6050 motion sensor.

The MPU-6050 provides accelerometer + gyro data.  Roll/pitch and total tilt can
be measured relative to a calibrated rest gravity vector.  Absolute yaw is not
observable from an MPU-6050 alone because it has no magnetometer; reporting a
made-up yaw would drift and be misleading.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, Optional, Tuple

try:
    from smbus2 import SMBus
except Exception:  # pragma: no cover - target hardware dependency
    SMBus = None

try:
    from gpiozero import DigitalInputDevice
except Exception:  # pragma: no cover - target hardware dependency
    DigitalInputDevice = None


class MPU6050Monitor:
    MPU6050_ADDR = 0x68

    PWR_MGMT_1 = 0x6B
    CONFIG = 0x1A
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C
    MOT_THR = 0x1F
    MOT_DUR = 0x20
    INT_PIN_CFG = 0x37
    INT_ENABLE = 0x38
    INT_STATUS = 0x3A
    ACCEL_XOUT_H = 0x3B
    WHO_AM_I = 0x75

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.bus_id = int(self.config.get("i2c_bus", 1))
        self.address = int(str(self.config.get("address", "0x68")), 0)
        self.interrupt_gpio = int(self.config.get("interrupt_gpio", 24))
        self.poll_hz = float(self.config.get("poll_hz", 10.0))
        self.accel_delta_threshold_g = float(self.config.get("accel_delta_threshold_g", 0.08))
        self.accel_magnitude_threshold_g = float(self.config.get("accel_magnitude_threshold_g", 0.12))
        self.gyro_threshold_dps = float(self.config.get("gyro_threshold_dps", 18.0))
        self.motion_hold_seconds = float(self.config.get("motion_hold_seconds", 2.5))
        self.rest_calibration_samples = max(5, int(self.config.get("rest_calibration_samples", 30)))
        self.rest_calibration_interval_seconds = max(
            0.005, float(self.config.get("rest_calibration_interval_seconds", 0.02))
        )

        self._bus = None
        self._gpio = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._io_lock = threading.RLock()
        self._available = False
        self._last_error = ""
        self._consecutive_errors = 0
        self._last_vector: Optional[Tuple[float, float, float]] = None
        self._last_motion_time = 0.0
        self._last_interrupt_time = 0.0
        self._motion_count = 0
        self._accel = (0.0, 0.0, 0.0)
        self._gyro = (0.0, 0.0, 0.0)
        self._temperature_c = 0.0
        self._roll_deg = 0.0
        self._pitch_deg = 0.0
        self._rest_vector: Optional[Tuple[float, float, float]] = None
        self._rest_roll_deg = 0.0
        self._rest_pitch_deg = 0.0
        self._rest_calibrated = False
        self._rest_calibrated_at = 0.0

    # ------------------------------------------------------------------
    # Lifecycle / device I/O
    # ------------------------------------------------------------------
    def start(self) -> bool:
        if not self.enabled:
            return False
        if SMBus is None:
            self._last_error = "smbus2 is not installed"
            return False
        try:
            self._stop_event.clear()
            self._bus = SMBus(self.bus_id)
            self._write(self.PWR_MGMT_1, 0x00)
            time.sleep(0.05)
            self._write(self.CONFIG, 0x03)
            self._write(self.GYRO_CONFIG, 0x00)  # +/-250 dps
            # DHPF=5 Hz improves hardware motion interrupt, +/-2g range retained.
            self._write(self.ACCEL_CONFIG, 0x01)

            threshold_mg = int(self.config.get("interrupt_threshold_mg", 80))
            duration_ms = int(self.config.get("interrupt_duration_ms", 20))
            self._write(self.MOT_THR, max(1, min(255, round(threshold_mg / 2))))
            self._write(self.MOT_DUR, max(1, min(255, duration_ms)))
            self._write(self.INT_PIN_CFG, 0x20)  # latch until INT_STATUS is read
            self._write(self.INT_ENABLE, 0x40)   # motion interrupt

            who = self._read_byte(self.WHO_AM_I)
            if who not in (0x68, 0x69):
                raise RuntimeError(f"unexpected MPU-6050 WHO_AM_I=0x{who:02X}")

            self._read_sample()
            self._available = True

            # Establish a real rest reference before the polling thread starts.
            # Keep the robot stationary for roughly 0.6 s during startup.
            self.calibrate_rest(
                samples=self.rest_calibration_samples,
                interval_seconds=self.rest_calibration_interval_seconds,
            )

            if DigitalInputDevice is not None:
                try:
                    self._gpio = DigitalInputDevice(
                        self.interrupt_gpio,
                        pull_up=False,
                        bounce_time=0.01,
                    )
                    self._gpio.when_activated = self._on_motion_interrupt
                except Exception as exc:
                    print(f"[MPU6050] GPIO{self.interrupt_gpio} interrupt unavailable: {exc}", flush=True)
                    self._gpio = None

            self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="mpu6050-monitor")
            self._thread.start()
            print(
                f"[MPU6050] Ready on I2C-{self.bus_id} address 0x{self.address:02X}; "
                f"INT GPIO{self.interrupt_gpio}; rest roll={self._rest_roll_deg:.1f}°, "
                f"pitch={self._rest_pitch_deg:.1f}°",
                flush=True,
            )
            return True
        except Exception as exc:
            self._last_error = str(exc)
            print(f"[MPU6050] Initialization failed: {exc}", flush=True)
            self.stop()
            return False

    def stop(self) -> None:
        self._stop_event.set()
        self._available = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._gpio is not None:
            try:
                self._gpio.close()
            except Exception:
                pass
            self._gpio = None
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None

    def _write(self, register: int, value: int) -> None:
        with self._io_lock:
            self._bus.write_byte_data(self.address, register, value)

    def _read_byte(self, register: int) -> int:
        with self._io_lock:
            return self._bus.read_byte_data(self.address, register)

    @staticmethod
    def _signed16(high: int, low: int) -> int:
        value = (high << 8) | low
        return value - 65536 if value & 0x8000 else value

    def _read_values(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
        with self._io_lock:
            block = self._bus.read_i2c_block_data(self.address, self.ACCEL_XOUT_H, 14)
        ax = self._signed16(block[0], block[1]) / 16384.0
        ay = self._signed16(block[2], block[3]) / 16384.0
        az = self._signed16(block[4], block[5]) / 16384.0
        temp_raw = self._signed16(block[6], block[7])
        gx = self._signed16(block[8], block[9]) / 131.0
        gy = self._signed16(block[10], block[11]) / 131.0
        gz = self._signed16(block[12], block[13]) / 131.0
        return (ax, ay, az), (gx, gy, gz), temp_raw / 340.0 + 36.53

    @staticmethod
    def _angles_from_accel(accel: Tuple[float, float, float]) -> Tuple[float, float]:
        ax, ay, az = accel
        # Standard gravity-referenced tilt.  These angles describe the sensor's
        # mounted axes; the rest-relative values are what the assistant exposes.
        roll = math.degrees(math.atan2(ay, az))
        pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
        return roll, pitch

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return (value + 180.0) % 360.0 - 180.0

    @staticmethod
    def _normalize_vector(v: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        mag = math.sqrt(sum(x * x for x in v))
        if mag < 1e-6:
            return None
        return tuple(x / mag for x in v)  # type: ignore[return-value]

    def _read_sample(self) -> None:
        accel, gyro, temperature_c = self._read_values()
        ax, ay, az = accel
        gx, gy, gz = gyro
        now = time.monotonic()

        accel_mag = math.sqrt(ax * ax + ay * ay + az * az)
        gyro_mag = math.sqrt(gx * gx + gy * gy + gz * gz)
        delta = 0.0
        if self._last_vector is not None:
            delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(accel, self._last_vector)))
        self._last_vector = accel

        motion = (
            abs(accel_mag - 1.0) >= self.accel_magnitude_threshold_g
            or delta >= self.accel_delta_threshold_g
            or gyro_mag >= self.gyro_threshold_dps
        )
        roll, pitch = self._angles_from_accel(accel)

        with self._lock:
            self._accel = accel
            self._gyro = gyro
            self._temperature_c = temperature_c
            self._roll_deg = roll
            self._pitch_deg = pitch
        if motion:
            self._mark_motion(now)

    # ------------------------------------------------------------------
    # Motion interrupt / polling
    # ------------------------------------------------------------------
    def _on_motion_interrupt(self, _device=None) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_interrupt_time = now
        self._mark_motion(now)
        try:
            self._read_byte(self.INT_STATUS)
        except Exception:
            pass

    def _mark_motion(self, when: Optional[float] = None) -> None:
        now = when or time.monotonic()
        with self._lock:
            if now - self._last_motion_time > 0.20:
                self._motion_count += 1
            self._last_motion_time = now

    def _poll_loop(self) -> None:
        delay = 1.0 / max(1.0, self.poll_hz)
        while not self._stop_event.is_set():
            try:
                self._read_sample()
                # When GPIO24 is active, gpiozero already services the latched
                # motion interrupt and reads INT_STATUS in the callback. Polling
                # INT_STATUS again every sample doubles I2C traffic for no gain.
                if self._gpio is None:
                    status = self._read_byte(self.INT_STATUS)
                    if status & 0x40:
                        self._mark_motion()
                self._last_error = ""
                self._consecutive_errors = 0
                self._available = True
            except Exception as exc:
                self._last_error = str(exc)
                self._consecutive_errors += 1
                # A missing/wedged bus should not become a tight 10-20 Hz error
                # loop. Back off after repeated failures while keeping the last
                # cached state available to the rest of the agent.
                if self._consecutive_errors >= 3:
                    self._available = False
                    self._stop_event.wait(min(1.0, 0.10 * self._consecutive_errors))
            self._stop_event.wait(delay)

    # ------------------------------------------------------------------
    # Rest calibration and public state
    # ------------------------------------------------------------------
    def calibrate_rest(self, samples: Optional[int] = None, interval_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Set the current stationary pose as the zero/rest orientation.

        This uses the gravity vector, so roll/pitch/total tilt are meaningful at
        rest.  Yaw cannot be calibrated from gravity and remains unavailable.
        """
        if not self._available and self._bus is None:
            return {"ok": False, "error": self._last_error or "MPU-6050 is unavailable"}

        count = max(5, int(samples or self.rest_calibration_samples))
        delay = max(0.005, float(interval_seconds or self.rest_calibration_interval_seconds))
        vectors = []
        gyros = []
        try:
            for _ in range(count):
                accel, gyro, _temp = self._read_values()
                vectors.append(accel)
                gyros.append(gyro)
                time.sleep(delay)
        except Exception as exc:
            self._last_error = str(exc)
            return {"ok": False, "error": str(exc)}

        avg = tuple(sum(v[i] for v in vectors) / len(vectors) for i in range(3))
        normalized = self._normalize_vector(avg)
        if normalized is None:
            return {"ok": False, "error": "invalid gravity vector during rest calibration"}
        roll, pitch = self._angles_from_accel(avg)
        avg_gyro_mag = sum(math.sqrt(sum(x * x for x in g)) for g in gyros) / len(gyros)

        with self._lock:
            self._rest_vector = normalized
            self._rest_roll_deg = roll
            self._rest_pitch_deg = pitch
            self._rest_calibrated = True
            self._rest_calibrated_at = time.monotonic()

        print(
            f"[MPU6050] Rest calibrated: roll={roll:.2f}°, pitch={pitch:.2f}°, "
            f"avg gyro={avg_gyro_mag:.2f}°/s",
            flush=True,
        )
        return {
            "ok": True,
            "rest_roll_deg": round(roll, 2),
            "rest_pitch_deg": round(pitch, 2),
            "average_gyro_dps": round(avg_gyro_mag, 2),
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            age = None if self._last_motion_time <= 0 else max(0.0, now - self._last_motion_time)
            moving = age is not None and age <= self.motion_hold_seconds
            accel = self._accel
            current_roll = self._roll_deg
            current_pitch = self._pitch_deg
            rest_vector = self._rest_vector
            rest_calibrated = self._rest_calibrated
            rest_roll = self._rest_roll_deg
            rest_pitch = self._rest_pitch_deg

            rel_roll = self._wrap_angle(current_roll - rest_roll) if rest_calibrated else None
            rel_pitch = self._wrap_angle(current_pitch - rest_pitch) if rest_calibrated else None
            total_tilt = None
            if rest_calibrated and rest_vector is not None:
                current_unit = self._normalize_vector(accel)
                if current_unit is not None:
                    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(current_unit, rest_vector))))
                    total_tilt = math.degrees(math.acos(dot))

            return {
                "available": self._available,
                "moving": moving,
                "seconds_since_motion": age,
                "motion_count": self._motion_count,
                "acceleration_g": tuple(round(x, 4) for x in accel),
                "gyro_dps": tuple(round(x, 2) for x in self._gyro),
                "temperature_c": round(self._temperature_c, 1),
                "absolute_roll_deg": round(current_roll, 2),
                "absolute_pitch_deg": round(current_pitch, 2),
                "rest_calibrated": rest_calibrated,
                "rest_roll_deg": round(rest_roll, 2) if rest_calibrated else None,
                "rest_pitch_deg": round(rest_pitch, 2) if rest_calibrated else None,
                "roll_from_rest_deg": round(rel_roll, 2) if rel_roll is not None else None,
                "pitch_from_rest_deg": round(rel_pitch, 2) if rel_pitch is not None else None,
                "total_tilt_from_rest_deg": round(total_tilt, 2) if total_tilt is not None else None,
                "yaw_from_rest_deg": None,
                "yaw_note": "Unavailable from MPU-6050 alone; a magnetometer is required for stable absolute yaw.",
                "interrupt_gpio": self.interrupt_gpio,
                "last_error": self._last_error,
            }

    def orientation_context_for_llm(self) -> str:
        state = self.snapshot()
        if not state["available"]:
            return f"MPU-6050 unavailable. {state.get('last_error', '')}".strip()
        if not state["rest_calibrated"]:
            return "MPU-6050 is available but the rest orientation is not calibrated."
        movement = "moving" if state["moving"] else "stationary"
        return (
            f"Fresh MPU-6050 orientation: robot is {movement}. Relative to the calibrated rest pose: "
            f"roll={state['roll_from_rest_deg']:.2f} degrees, "
            f"pitch={state['pitch_from_rest_deg']:.2f} degrees, "
            f"total tilt={state['total_tilt_from_rest_deg']:.2f} degrees. "
            "Yaw is not reliably observable with this MPU-6050 because it has no magnetometer. "
            f"Raw acceleration XYZ={state['acceleration_g']} g and gyro XYZ={state['gyro_dps']} deg/s."
        )

    def context_for_llm(self) -> str:
        state = self.snapshot()
        if not state["available"]:
            return f"Motion sensor unavailable. {state.get('last_error', '')}".strip()
        if state["moving"]:
            age = state["seconds_since_motion"] or 0.0
            movement = f"The robot is being moved or was moved {age:.1f} seconds ago."
        else:
            age = state["seconds_since_motion"]
            movement = "The robot is currently stationary."
            if age is not None:
                movement += f" The last detected movement was {age:.1f} seconds ago."

        orientation = ""
        if state["rest_calibrated"]:
            orientation = (
                f" Relative to its calibrated rest pose: roll={state['roll_from_rest_deg']:.2f}°, "
                f"pitch={state['pitch_from_rest_deg']:.2f}°, "
                f"total tilt={state['total_tilt_from_rest_deg']:.2f}°."
            )
        else:
            orientation = " Rest orientation is not calibrated."

        return (
            f"{movement}{orientation} Accelerometer XYZ={state['acceleration_g']} g; "
            f"gyro XYZ={state['gyro_dps']} deg/s. Yaw is unavailable without a magnetometer."
        )

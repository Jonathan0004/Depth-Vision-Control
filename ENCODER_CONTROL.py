"""Closed-loop steering controller for the Depth-Vision platform.

This module wraps the brushed motor H-bridge outputs and an AS5048A SPI
magnetic encoder into a reusable controller that can be driven by
``detectAvoid_V2.py``.  The controller provides calibration, persistence and
status overlays so the steering command coming from the vision pipeline maps
linearly to the physical steering angle.

Hardware hook-up (Jetson Nano ↔ AS5048A):
    * 3V3  -> AS5048A VCC (pin 1 on the Jetson Nano 40-pin header)
    * GND  -> AS5048A GND (pin 6)
    * SPI0_SCLK -> AS5048A CLK (pin 23)
    * SPI0_MOSI -> AS5048A DI  (pin 19)
    * SPI0_MISO -> AS5048A DO  (pin 21)
    * Choose any free GPIO for chip-select, e.g. SPI0_CS0 on pin 24. Tie it to
      the AS5048A CS pin.  The code below assumes ``spi_device=0`` which uses
      CS0.

Additional notes:
    * Enable SPI on the Nano using ``sudo /opt/nvidia/jetson-io/jetson-io.py``
      (or by editing the device-tree overlay).
    * Keep all logic at 3.3 V.  The AS5048A is **not** 5 V tolerant on the SPI
      lines.
    * The motor enable pins are driven using Jetson.GPIO in BOARD numbering and
      the PWM output is routed via ``/sys/class/pwm`` as in the previous motor
      controller.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import Jetson.GPIO as GPIO

try:
    import spidev  # type: ignore
except ImportError as exc:  # pragma: no cover - hardware dependency
    raise RuntimeError(
        "spidev is required for the AS5048A encoder. Install it with 'sudo apt "
        "install python3-spidev' and ensure SPI is enabled."
    ) from exc


@dataclass
class Calibration:
    """Persistent mapping between screen space and encoder angles."""

    angle_left: float
    angle_right: float
    span: float
    screen_min: float
    screen_max: float
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "angle_left": self.angle_left,
            "angle_right": self.angle_right,
            "span": self.span,
            "screen_min": self.screen_min,
            "screen_max": self.screen_max,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Calibration":
        return cls(
            angle_left=float(payload["angle_left"]),
            angle_right=float(payload["angle_right"]),
            span=float(payload["span"]),
            screen_min=float(payload["screen_min"]),
            screen_max=float(payload["screen_max"]),
            timestamp=str(payload.get("timestamp", "")),
        )


class EncoderSteeringController:
    """Closed-loop steering helper around an AS5048A absolute encoder."""

    def __init__(
        self,
        *,
        motor_pin_left: int,
        motor_pin_right: int,
        pwm_chip_path: str,
        pwm_channel: str,
        pwm_frequency_hz: int,
        calibration_file: str,
        spi_bus: int = 0,
        spi_device: int = 0,
        tolerance_deg: float = 1.0,
        kp: float = 0.9,
        min_pwm_pct: float = 15.0,
        max_pwm_pct: float = 75.0,
    ) -> None:
        self.motor_pin_left = motor_pin_left
        self.motor_pin_right = motor_pin_right
        self.pwm_chip_path = pwm_chip_path
        self.pwm_channel = pwm_channel
        self.pwm_frequency_hz = pwm_frequency_hz
        self.calibration_file = calibration_file
        self.spi_bus = spi_bus
        self.spi_device = spi_device
        self.tolerance_deg = tolerance_deg
        self.kp = kp
        self.min_pwm_pct = min_pwm_pct
        self.max_pwm_pct = max_pwm_pct

        self.screen_min = 0.0
        self.screen_max = 1.0
        self.calibration: Optional[Calibration] = None
        self.calibration_stage: Optional[str] = None
        self._calibration_samples = {}
        self._status_messages: List[str] = []
        self._target_unwrapped: Optional[float] = None
        self._last_error: Optional[float] = None

        self.spi = spidev.SpiDev()
        self.spi.open(self.spi_bus, self.spi_device)
        self.spi.max_speed_hz = 1_000_000
        self.spi.mode = 0b01

        self.pwm_channel_path = os.path.join(self.pwm_chip_path, self.pwm_channel)
        self._ensure_pwm_ready()
        self._setup_gpio()

        self.calibration = self._load_calibration()
        if self.calibration is None:
            self._status_messages.append("Encoder: press 'c' to calibrate")
        else:
            self.screen_min = self.calibration.screen_min
            self.screen_max = self.calibration.screen_max
            self._status_messages.append(
                f"Calibration loaded ({self.calibration.timestamp})"
            )

    # ------------------------------------------------------------------
    # Public API used from detectAvoid_V2.py
    # ------------------------------------------------------------------
    def update_screen_range(self, zone_left: float, zone_right: float) -> None:
        self.screen_min = float(zone_left)
        self.screen_max = float(zone_right)

    def drive_to_screen_position(self, screen_x: float) -> None:
        if self.calibration_stage:
            # During calibration the user moves the wheels manually.
            self.stop_motor()
            return
        if self.calibration is None:
            self.stop_motor()
            return
        if math.isclose(self.screen_min, self.screen_max):
            return

        target_unwrapped = self._screen_to_unwrapped(screen_x)
        current = self._read_unwrapped_angle()
        error = target_unwrapped - current
        self._last_error = error
        self._target_unwrapped = target_unwrapped

        if abs(error) <= self.tolerance_deg:
            self.stop_motor()
            return

        direction = 1 if error > 0 else -1
        duty_pct = min(
            self.max_pwm_pct,
            max(self.min_pwm_pct, abs(error) * self.kp),
        )
        self._apply_motor(direction, duty_pct)

    def handle_key(self, key: int) -> None:
        if key in (-1, 255):
            return
        if key == ord("c"):
            self._begin_calibration()
            return
        if self.calibration_stage is None:
            return
        if key == ord(" "):
            # Record current stage sample
            angle = self._read_raw_angle()
            if self.calibration_stage == "left":
                self._calibration_samples["left"] = angle
                self.calibration_stage = "right"
                msg = "Move steering to RIGHT limit, press SPACE"
                self._status_messages = [msg]
                print("[Calibration] Recorded left limit at %.2f°" % angle)
            elif self.calibration_stage == "right":
                self._calibration_samples["right"] = angle
                print("[Calibration] Recorded right limit at %.2f°" % angle)
                self._finish_calibration()
        elif key == 27:
            # ESC cancels calibration but allow main loop to exit as well.
            self._status_messages = ["Calibration cancelled"]
            self.calibration_stage = None
            self._calibration_samples = {}

    def overlay_lines(self) -> List[str]:
        lines = list(self._status_messages)
        if self.calibration_stage == "left":
            lines.insert(0, "Calibration: move LEFT, press SPACE")
        elif self.calibration_stage == "right":
            lines.insert(0, "Calibration: move RIGHT, press SPACE")
        elif self.calibration is None:
            lines.insert(0, "Encoder: press 'c' to calibrate")
        else:
            if self._last_error is not None and self._target_unwrapped is not None:
                lines.insert(
                    0,
                    "Encoder err: %.1f° target %.1f°" % (
                        self._last_error,
                        self._target_unwrapped,
                    ),
                )
        return lines

    def shutdown(self) -> None:
        self.stop_motor()
        try:
            self._pwm_write("duty_cycle", 0)
            self._pwm_write("enable", 0)
        except OSError:
            pass
        self.spi.close()
        GPIO.cleanup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def stop_motor(self) -> None:
        GPIO.output(self.motor_pin_left, GPIO.LOW)
        GPIO.output(self.motor_pin_right, GPIO.LOW)
        self._pwm_write("duty_cycle", 0)

    def _apply_motor(self, direction: int, duty_pct: float) -> None:
        if direction >= 0:
            GPIO.output(self.motor_pin_left, GPIO.LOW)
            GPIO.output(self.motor_pin_right, GPIO.HIGH)
        else:
            GPIO.output(self.motor_pin_left, GPIO.HIGH)
            GPIO.output(self.motor_pin_right, GPIO.LOW)
        duty_ns = self._duty_from_percent(duty_pct)
        self._pwm_write("duty_cycle", duty_ns)

    def _begin_calibration(self) -> None:
        self.stop_motor()
        self.calibration_stage = "left"
        self._calibration_samples = {}
        self._status_messages = [
            "Calibration started", "Move steering to LEFT limit, press SPACE"
        ]
        print("[Calibration] Starting new calibration. Move to LEFT limit and press SPACE.")

    def _finish_calibration(self) -> None:
        left = self._calibration_samples.get("left")
        right = self._calibration_samples.get("right")
        if left is None or right is None:
            self._status_messages = ["Calibration failed: missing samples"]
            self.calibration_stage = None
            return

        span = self._unwrap_difference(left, right)
        timestamp = datetime.utcnow().isoformat()
        self.calibration = Calibration(
            angle_left=left,
            angle_right=right,
            span=span,
            screen_min=self.screen_min,
            screen_max=self.screen_max,
            timestamp=timestamp,
        )
        self._save_calibration(self.calibration)
        self._status_messages = [f"Calibration saved {timestamp}"]
        self.calibration_stage = None
        self._calibration_samples = {}
        print("[Calibration] Completed. Span %.2f°" % span)

    def _load_calibration(self) -> Optional[Calibration]:
        if not os.path.isfile(self.calibration_file):
            return None
        try:
            with open(self.calibration_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Failed to read calibration file {self.calibration_file}: {exc}")
            return None
        try:
            return Calibration.from_dict(data)
        except (KeyError, TypeError, ValueError):
            print(f"Calibration file {self.calibration_file} is malformed. Ignoring.")
            return None

    def _save_calibration(self, calibration: Calibration) -> None:
        directory = os.path.dirname(self.calibration_file)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        payload = calibration.to_dict()
        with open(self.calibration_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _screen_to_unwrapped(self, screen_x: float) -> float:
        ratio = (screen_x - self.screen_min) / (self.screen_max - self.screen_min)
        ratio = max(0.0, min(1.0, ratio))
        base = self.calibration.angle_left
        span = self.calibration.span
        return base + span * ratio

    def _read_unwrapped_angle(self) -> float:
        angle = self._read_raw_angle()
        base = self.calibration.angle_left
        diff = self._unwrap_difference(base, angle)
        return base + diff

    def _unwrap_difference(self, start: float, end: float) -> float:
        diff = (end - start + 540.0) % 360.0 - 180.0
        return diff

    def _read_raw_angle(self) -> float:
        command = self._build_command(0x3FFF)
        self.spi.xfer2([(command >> 8) & 0xFF, command & 0xFF])
        time.sleep(0.00001)
        result = self.spi.xfer2([0x00, 0x00])
        value = ((result[0] << 8) | result[1]) & 0x3FFF
        return (value * 360.0) / 16383.0

    def _build_command(self, address: int) -> int:
        command = 0x4000 | (address & 0x3FFF)
        if self._parity(command):
            command |= 0x8000
        return command

    @staticmethod
    def _parity(value: int) -> int:
        return bin(value).count("1") % 2

    def _ensure_pwm_ready(self) -> None:
        if not os.path.isdir(self.pwm_channel_path):
            if not os.path.isdir(self.pwm_chip_path):
                raise FileNotFoundError(
                    f"PWM chip {self.pwm_chip_path} not found. Check Jetson configuration."
                )
            self._pwm_write(os.path.join(self.pwm_chip_path, "export"), 0)
            for _ in range(200):
                if os.path.isdir(self.pwm_channel_path):
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError("PWM channel did not appear after export")
        try:
            self._pwm_write("enable", 0)
        except OSError:
            pass
        period = int(round(1e9 / float(self.pwm_frequency_hz)))
        self.period_ns = period
        self._pwm_write("period", period)
        self._pwm_write("duty_cycle", 0)
        self._pwm_write("enable", 1)

    def _setup_gpio(self) -> None:
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(self.motor_pin_left, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.motor_pin_right, GPIO.OUT, initial=GPIO.LOW)

    def _duty_from_percent(self, percent: float) -> int:
        pct = max(0.0, min(100.0, percent))
        return int(self.period_ns * (pct / 100.0))

    def _pwm_write(self, leaf: str, value: int) -> None:
        path = leaf if os.path.isabs(leaf) else os.path.join(self.pwm_channel_path, leaf)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(str(value))


__all__ = ["EncoderSteeringController"]

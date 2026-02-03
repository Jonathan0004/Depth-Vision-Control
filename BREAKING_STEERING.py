import json
import queue
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

import subprocess
import sys
import os



# Run INITIALIZE_PWM.py first
script_path = os.path.join(os.path.dirname(__file__), "INITIALIZE_PWM.py")
subprocess.run([sys.executable, script_path])



try:  # pragma: no cover - hardware dependency
    import Jetson.GPIO as GPIO
except ImportError:  # pragma: no cover - hardware dependency
    GPIO = None

try:
    from smbus2 import SMBus
except ImportError:  # pragma: no cover - hardware dependency
    try:
        from smbus import SMBus  # type: ignore
    except ImportError:  # pragma: no cover - hardware dependency
        SMBus = None


# ---------------------------------------------------------------------------
# Configuration — tweak these values to adjust behaviour without diving into
# the rest of the code. Where possible, related values are grouped together and
# documented so their impact is clear.
# ---------------------------------------------------------------------------

# Grid resolution for sampling depth points across each frame (higher = denser)
rows, cols = 25, 50

# Obstacle detection thresholds
depth_diff_threshold = 8      # minimum mean-depth difference to flag a point
std_multiplier = 0.3         # scales standard deviation term for adaptive thresholding

# Cutoff bands (green) — obstacles outside these vertical limits are ignored
top_cutoff_pixels = 10        # pixels from the top edge of each camera frame
bottom_cutoff_pixels = 54     # pixels from the bottom edge of each camera frame
cutoff_line_color = (0, 255, 0)
cutoff_line_thickness = 1

# Obstacle marker (red dots)
obstacle_dot_radius_px = 5
obstacle_dot_color = (0, 0, 255)

# Blue steering cue (circle) rendered on the combined frame
blue_circle_radius_px = 12
blue_circle_color = (255, 0, 0)
blue_circle_thickness = 3

# Blue vertical cutoff lines indicate the "pull zone" used for gating logic
pull_zone_line_color = (255, 0, 0)
pull_zone_line_thickness = 1
pull_influence_radius_px = 120   # half-width of the pull zone around centre
pull_zone_center_offset_px = 0   # shift pull zone horizontally (+ right, - left)

# Smoothing factor for the steering cue (1 → frozen, 0 → instant response)
# The smoothing factor also governs how quickly the motor PWM ramps to match the
# steering target (higher = slower response, lower = snappier).
blue_x_smoothness = 0.7

# Motor tuning parameters ---------------------------------------------------
motor_max_duty_pct = 100.0          # absolute cap on PWM duty cycle
motor_full_speed_error_px = 120     # error (px) required to request max duty

motor_dead_zone_px = 5              # +/- range in which motor output is disabled

jog_default_duty_pct = 50.0         # default jog duty percentage when jog mode toggles on
jog_duty_step_pct = 5.0             # amount to adjust jog duty via arrow keys

# ---------------------------------------------------------------------------


# Extra steering push (pixels) applied in the commanded direction (+/- from centre)
divert_boost = 50


# Motor hardware configuration (Jetson Nano)
motor_pwm_chip = "/sys/class/pwm/pwmchip3"
motor_pwm_channel = 0
motor_pwm_frequency_hz = 5000
motor_pwm_pin = 32                  # informational only (physical pin number)
motor_direction_pin = 29            # HIGH = steer right, LOW = steer left
motor_power_pin = 23                # HIGH when PWM is driving, LOW when idle/dead-zone/calibration

# Braking motor hardware configuration (Jetson Nano)
braking_motor_pwm_chip = "/sys/class/pwm/pwmchip2"
braking_motor_pwm_channel = 0
braking_motor_pwm_frequency_hz = 5000
braking_motor_pwm_pin = 33               # informational only (physical pin number)
braking_motor_direction_pin = 37         # HIGH = jog right, LOW = jog left
braking_motor_power_pin = 22             # HIGH when PWM is driving, LOW when idle/dead-zone/calibration

braking_motor_max_duty_pct = 100.0
braking_motor_full_speed_error_norm = 0.35
braking_motor_dead_zone_norm = 0.01
braking_jog_default_duty_pct = 50.0
braking_jog_duty_step_pct = 5.0

breakingThresh = None

# HUD text overlays on the combined frame
hud_text_position = (10, 30)
hud_text_color = (255, 255, 255)
hud_text_scale = 0.5
hud_text_thickness = 1
hud_text_outline_color = (0, 0, 0)
hud_text_outline_thickness = 2
status_text_scale = 0.58
status_text_padding = 6
help_title_scale = 0.8
help_text_scale = 0.48
help_text_line_gap = 18
help_text_column_gap = 40

# Encoder + steering control configuration
encoder_i2c_bus = 7
encoder_i2c_address = 0x36
calibration_file = Path("steering_calibration.json")
simulated_step_norm = 0.01        # arrow-key increment when in simulated encoder mode

# Encoder + braking control configuration
braking_encoder_i2c_bus = 0
braking_encoder_i2c_address = 0x36
braking_calibration_file = Path("braking_calibration.json")


# ---------------------------------------------------------------------------
# Runtime state (initialised once, then updated as frames stream in)
# ---------------------------------------------------------------------------
blue_x = None              # persistent, smoothed x-position of the blue circle
steer_control_x = None     # integer pixel location displayed as HUD text


# Cache for sampling grids so expensive mesh computations run once per shape
_sample_grid_cache = {}


# ---------------------------------------------------------------------------
# PWM helpers (mirrors PWM_CONTROL.py logic with graceful fallbacks)
# ---------------------------------------------------------------------------

def _ns_period(freq_hz: float) -> int:
    return int(round(1e9 / float(freq_hz)))


def _ns_duty_from_pct(period_ns: int, pct: float) -> int:
    pct = max(0.0, min(100.0, pct))
    return int(round(period_ns * (pct / 100.0)))


def _pwm_channel_path(chip: str, channel: int) -> Path:
    return Path(chip) / f"pwm{channel}"


def _pwm_write(path: Path, value) -> None:
    with path.open("w") as fh:
        fh.write(str(value))


def _ensure_pwm_channel(chip: str, channel: int) -> Path:
    ch_path = _pwm_channel_path(chip, channel)
    if ch_path.is_dir():
        return ch_path
    chip_path = Path(chip)
    if not chip_path.is_dir():
        raise FileNotFoundError(f"{chip} does not exist. Check PWM availability.")
    _pwm_write(chip_path / "export", channel)
    for _ in range(200):
        if ch_path.is_dir():
            break
        time.sleep(0.01)
    else:  # pragma: no cover - hardware timing issue
        raise TimeoutError(f"pwm{channel} did not appear after export")
    return ch_path


# ---------------------------------------------------------------------------
# Helper utilities and pipeline initialisation
# ---------------------------------------------------------------------------

# Utility: clamp values between two bounds (used to keep overlays on-screen)
def clamp(val, minn, maxn):
    return max(min(val, maxn), minn)


def draw_text_with_outline(frame, text, org, font, scale, color, thickness, outline_color, outline_thickness):
    cv2.putText(frame, text, org, font, scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_status_banner(frame, text, org, font, scale, color, padding):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, 1)
    x, y = org
    top_left = (x - padding, y - text_h - padding)
    bottom_right = (x + text_w + padding, y + baseline + padding)
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)
    draw_text_with_outline(
        frame,
        text,
        org,
        font,
        scale,
        color,
        1,
        hud_text_outline_color,
        hud_text_outline_thickness,
    )


def draw_bottom_lines(
    frame,
    lines,
    origin,
    font,
    scale,
    color,
    thickness,
    outline_color,
    outline_thickness,
    line_gap,
):
    x, y = origin
    for offset, line in enumerate(reversed(lines)):
        draw_text_with_outline(
            frame,
            line,
            (x, y - offset * line_gap),
            font,
            scale,
            color,
            thickness,
            outline_color,
            outline_thickness,
        )


def draw_help_overlay(
    frame,
    title,
    lines,
    origin,
    font,
    title_scale,
    scale,
    color,
    thickness,
    outline_color,
    outline_thickness,
    line_gap,
    column_gap,
):
    x, y = origin
    draw_text_with_outline(
        frame,
        title,
        (x, y),
        font,
        title_scale,
        color,
        thickness,
        outline_color,
        outline_thickness,
    )
    (title_w, title_h), _ = cv2.getTextSize(title, font, title_scale, 1)
    start_y = y + title_h + line_gap
    split_index = (len(lines) + 1) // 2
    left_lines = lines[:split_index]
    right_lines = lines[split_index:]
    left_width = 0
    for line in left_lines:
        (line_w, _), _ = cv2.getTextSize(line, font, scale, 1)
        left_width = max(left_width, line_w)
    right_x = x + left_width + column_gap
    for index, line in enumerate(left_lines):
        draw_text_with_outline(
            frame,
            line,
            (x, start_y + index * line_gap),
            font,
            scale,
            color,
            thickness,
            outline_color,
            outline_thickness,
        )
    for index, line in enumerate(right_lines):
        draw_text_with_outline(
            frame,
            line,
            (right_x, start_y + index * line_gap),
            font,
            scale,
            color,
            thickness,
            outline_color,
            outline_thickness,
        )


def set_status_message(text, duration_s=None):
    if duration_s is None:
        return text, 0.0
    return text, time.time() + duration_s


# ---------------------------------------------------------------------------
# Encoder helpers and calibration persistence
# ---------------------------------------------------------------------------

encoder_bus = None
encoder_available = False
calibration_data = {"encoder_min_raw": None, "encoder_max_raw": None}
calibration_loaded = False


def initialise_encoder():
    global encoder_bus, encoder_available
    if SMBus is None:
        encoder_available = False
        return
    try:
        encoder_bus = SMBus(encoder_i2c_bus)
        encoder_available = True
    except (FileNotFoundError, OSError):
        encoder_bus = None
        encoder_available = False


def shutdown_encoder():
    global encoder_bus
    if encoder_bus is not None:
        try:
            encoder_bus.close()
        finally:
            encoder_bus = None


def load_calibration():
    global calibration_data, calibration_loaded
    if calibration_loaded:
        return
    if calibration_file.exists():
        try:
            data = json.loads(calibration_file.read_text())
            if {
                "encoder_min_raw",
                "encoder_max_raw",
            } <= data.keys():
                calibration_data = {
                    "encoder_min_raw": int(data["encoder_min_raw"]),
                    "encoder_max_raw": int(data["encoder_max_raw"]),
                }
        except (json.JSONDecodeError, ValueError):
            pass
    calibration_loaded = True


def save_calibration(min_raw, max_raw):
    global calibration_data, calibration_loaded
    calibration_data = {
        "encoder_min_raw": int(min_raw),
        "encoder_max_raw": int(max_raw),
    }
    calibration_file.write_text(json.dumps(calibration_data, indent=2))
    calibration_loaded = True


def encoder_span():
    if calibration_data["encoder_min_raw"] is None or calibration_data["encoder_max_raw"] is None:
        return None
    span = calibration_data["encoder_max_raw"] - calibration_data["encoder_min_raw"]
    return span if span > 0 else None


def read_encoder_raw():
    if not encoder_available or encoder_bus is None:
        return None
    try:
        high = encoder_bus.read_byte_data(encoder_i2c_address, 0x0C)
        low = encoder_bus.read_byte_data(encoder_i2c_address, 0x0D)
    except OSError:
        return None
    raw = ((high & 0x0F) << 8) | low
    return raw


def encoder_raw_to_norm(raw):
    span = encoder_span()
    if span is None:
        return None
    norm = (raw - calibration_data["encoder_min_raw"]) / span
    return float(clamp(norm, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Braking encoder helpers and calibration persistence
# ---------------------------------------------------------------------------

braking_encoder_bus = None
braking_encoder_available = False
braking_calibration_data = {"encoder_min_raw": None, "encoder_max_raw": None}
braking_calibration_loaded = False


def initialise_braking_encoder():
    global braking_encoder_bus, braking_encoder_available
    if SMBus is None:
        braking_encoder_available = False
        return
    try:
        braking_encoder_bus = SMBus(braking_encoder_i2c_bus)
        braking_encoder_available = True
    except (FileNotFoundError, OSError):
        braking_encoder_bus = None
        braking_encoder_available = False


def shutdown_braking_encoder():
    global braking_encoder_bus
    if braking_encoder_bus is not None:
        try:
            braking_encoder_bus.close()
        finally:
            braking_encoder_bus = None


def load_braking_calibration():
    global braking_calibration_data, braking_calibration_loaded
    if braking_calibration_loaded:
        return
    if braking_calibration_file.exists():
        try:
            data = json.loads(braking_calibration_file.read_text())
            if {
                "encoder_min_raw",
                "encoder_max_raw",
            } <= data.keys():
                braking_calibration_data = {
                    "encoder_min_raw": int(data["encoder_min_raw"]),
                    "encoder_max_raw": int(data["encoder_max_raw"]),
                }
        except (json.JSONDecodeError, ValueError):
            pass
    braking_calibration_loaded = True


def save_braking_calibration(min_raw, max_raw):
    global braking_calibration_data, braking_calibration_loaded
    braking_calibration_data = {
        "encoder_min_raw": int(min_raw),
        "encoder_max_raw": int(max_raw),
    }
    braking_calibration_file.write_text(json.dumps(braking_calibration_data, indent=2))
    braking_calibration_loaded = True


def braking_encoder_span():
    if (
        braking_calibration_data["encoder_min_raw"] is None
        or braking_calibration_data["encoder_max_raw"] is None
    ):
        return None
    span = braking_calibration_data["encoder_max_raw"] - braking_calibration_data["encoder_min_raw"]
    return span if span > 0 else None


def read_braking_encoder_raw():
    if not braking_encoder_available or braking_encoder_bus is None:
        return None
    try:
        high = braking_encoder_bus.read_byte_data(braking_encoder_i2c_address, 0x0C)
        low = braking_encoder_bus.read_byte_data(braking_encoder_i2c_address, 0x0D)
    except OSError:
        return None
    raw = ((high & 0x0F) << 8) | low
    return raw


def braking_encoder_raw_to_norm(raw):
    span = braking_encoder_span()
    if span is None:
        return None
    norm = (raw - braking_calibration_data["encoder_min_raw"]) / span
    return float(clamp(norm, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Runtime steering state (encoder + simulation)
# ---------------------------------------------------------------------------

sim_encoder_enabled = False
sim_encoder_norm = 0.5

latest_encoder_raw = None
latest_encoder_norm = None

calibration_active = False
calibration_stage = None
calibration_status_text = ""
calibration_status_until = 0.0
calibration_samples = {}
calibration_jog_direction = 0
ui_notice_text = ""
ui_notice_until = 0.0
help_overlay_enabled = False


# ---------------------------------------------------------------------------
# Motor control runtime state
# ---------------------------------------------------------------------------

motor_pwm_period_ns = _ns_period(motor_pwm_frequency_hz)
motor_pwm_channel_path = None
motor_control_available = False
motor_gpio_initialised = False
motor_last_duty_ns = None
motor_pwm_enabled = False
jog_mode_enabled = False
jog_direction = 0  # -1 left, 0 idle, +1 right
jog_duty_pct = jog_default_duty_pct

braking_motor_pwm_period_ns = _ns_period(braking_motor_pwm_frequency_hz)
braking_motor_pwm_channel_path = None
braking_motor_control_available = False
braking_motor_gpio_initialised = False
braking_motor_last_duty_ns = None
braking_motor_pwm_enabled = False
braking_jog_mode_enabled = False
braking_jog_direction = 0  # -1 left, 0 idle, +1 right
braking_jog_duty_pct = braking_jog_default_duty_pct

braking_latest_encoder_raw = None
braking_latest_encoder_norm = None
braking_calibration_active = False
braking_calibration_stage = None
braking_calibration_status_text = ""
braking_calibration_status_until = 0.0
braking_calibration_samples = {}
braking_calibration_jog_direction = 0
braking_target_norm = None


def initialise_motor_control():
    """Prepare GPIO direction/power pins and PWM channel for motor drive."""
    global motor_pwm_channel_path, motor_control_available, motor_gpio_initialised, motor_last_duty_ns, motor_pwm_enabled

    motor_control_available = False
    if GPIO is None:
        return

    try:
        if not motor_gpio_initialised:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)
            GPIO.setup(motor_direction_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(motor_power_pin, GPIO.OUT, initial=GPIO.LOW)  # keep LOW until PWM actually drives
            motor_gpio_initialised = True
    except RuntimeError:
        return

    try:
        motor_pwm_channel_path = _ensure_pwm_channel(motor_pwm_chip, motor_pwm_channel)
        _pwm_write(motor_pwm_channel_path / "enable", 0)
        _pwm_write(motor_pwm_channel_path / "period", motor_pwm_period_ns)
        _pwm_write(motor_pwm_channel_path / "duty_cycle", 0)
        _pwm_write(motor_pwm_channel_path / "enable", 1)
        motor_last_duty_ns = 0
        motor_pwm_enabled = True
        motor_control_available = True
    except (OSError, FileNotFoundError, TimeoutError):  # pragma: no cover - hardware dependency
        motor_pwm_channel_path = None
        motor_control_available = False



def _set_motor_direction(error: float) -> None:
    """Update the GPIO direction pin based on the steering error."""
    if GPIO is None or not motor_gpio_initialised:
        return
    GPIO.output(motor_direction_pin, GPIO.HIGH if error > 0 else GPIO.LOW)


def _set_motor_pwm_pct(pct: float) -> None:
    """Write the requested PWM duty cycle percentage to sysfs."""
    global motor_last_duty_ns
    if (
        not motor_control_available
        or motor_pwm_channel_path is None
        or not motor_pwm_enabled
    ):
        return
    duty_ns = _ns_duty_from_pct(motor_pwm_period_ns, pct)
    if motor_last_duty_ns == duty_ns:
        return
    try:
        _pwm_write(motor_pwm_channel_path / "duty_cycle", duty_ns)
        motor_last_duty_ns = duty_ns
    except OSError:  # pragma: no cover - hardware dependency
        pass


def disable_motor_pwm() -> None:
    """Temporarily disable PWM output while keeping the channel exported."""
    global motor_pwm_enabled, motor_last_duty_ns
    if not motor_control_available or motor_pwm_channel_path is None:
        return
    _set_motor_pwm_pct(0.0)
    try:
        _pwm_write(motor_pwm_channel_path / "enable", 0)
    except OSError:  # pragma: no cover - hardware dependency
        return
    motor_last_duty_ns = 0
    motor_pwm_enabled = False
    # Ensure motor power indicator is LOW while disabled (e.g., during calibration)
    if GPIO is not None and motor_gpio_initialised:
        GPIO.output(motor_power_pin, GPIO.LOW)


def enable_motor_pwm() -> None:
    """Re-enable PWM output after a temporary disable."""
    global motor_pwm_enabled, motor_last_duty_ns
    if not motor_control_available or motor_pwm_channel_path is None:
        return
    try:
        _pwm_write(motor_pwm_channel_path / "duty_cycle", 0)
        _pwm_write(motor_pwm_channel_path / "enable", 1)
    except OSError:  # pragma: no cover - hardware dependency
        return
    motor_last_duty_ns = 0
    motor_pwm_enabled = True
    # Keep motor power indicator LOW until a non-zero duty is commanded
    if GPIO is not None and motor_gpio_initialised:
        GPIO.output(motor_power_pin, GPIO.LOW)


def update_motor_control(steer_target_px, encoder_px):
    """Drive the motor to align the encoder with the steering target and assert motor power pin when driving."""
    # If we can't compute a target or we shouldn't drive, force PWM off and power pin LOW
    if steer_target_px is None or encoder_px is None or calibration_active:
        _set_motor_direction(0)
        _set_motor_pwm_pct(0.0)
        if GPIO is not None and motor_gpio_initialised:
            GPIO.output(motor_power_pin, GPIO.LOW)
        return

    error = float(steer_target_px) - float(encoder_px)

    # Dead zone → no drive, power pin LOW
    if abs(error) <= motor_dead_zone_px:
        _set_motor_direction(0)
        _set_motor_pwm_pct(0.0)
        if GPIO is not None and motor_gpio_initialised:
            GPIO.output(motor_power_pin, GPIO.LOW)
        return

    _set_motor_direction(error)
    denom = float(motor_full_speed_error_px) if motor_full_speed_error_px > 0 else 1.0
    scaled_pct = (abs(error) / denom) * motor_max_duty_pct
    duty_pct = max(0.0, min(motor_max_duty_pct, scaled_pct))

    _set_motor_pwm_pct(duty_pct)

    # Assert/deassert motor power indicator based on actual drive conditions
    if GPIO is not None and motor_gpio_initialised:
        if motor_pwm_enabled and not calibration_active and duty_pct > 0.0:
            GPIO.output(motor_power_pin, GPIO.HIGH)  # driving
        else:
            GPIO.output(motor_power_pin, GPIO.LOW)   # idle/off


def apply_jog_drive(direction: int) -> None:
    """Directly drive the steering motor for manual jogging."""
    if GPIO is not None and motor_gpio_initialised:
        GPIO.output(motor_power_pin, GPIO.LOW)

    if direction not in (-1, 1):
        _set_motor_direction(0)
        _set_motor_pwm_pct(0.0)
        return

    _set_motor_direction(float(direction))
    duty = clamp(jog_duty_pct, 0.0, motor_max_duty_pct)
    _set_motor_pwm_pct(duty)
    if GPIO is not None and motor_gpio_initialised:
        if duty > 0.0:
            GPIO.output(motor_power_pin, GPIO.HIGH)
        else:
            GPIO.output(motor_power_pin, GPIO.LOW)


def shutdown_motor_control():
    """Return the motor hardware to a safe idle state."""
    global motor_control_available, motor_pwm_channel_path, motor_last_duty_ns, motor_gpio_initialised, motor_pwm_enabled

    _set_motor_direction(0)
    _set_motor_pwm_pct(0.0)

    # Ensure motor power indicator is LOW before cleanup
    if GPIO is not None and motor_gpio_initialised:
        try:
            GPIO.output(motor_power_pin, GPIO.LOW)
        except RuntimeError:  # pragma: no cover - hardware dependency
            pass

    if motor_control_available and motor_pwm_channel_path is not None:
        try:
            _pwm_write(motor_pwm_channel_path / "enable", 0)
        except OSError:  # pragma: no cover - hardware dependency
            pass

    motor_control_available = False
    motor_pwm_channel_path = None
    motor_last_duty_ns = None
    motor_pwm_enabled = False

    if GPIO is not None and motor_gpio_initialised:
        try:
            GPIO.cleanup([motor_direction_pin, motor_power_pin])
        except RuntimeError:  # pragma: no cover - hardware dependency
            pass
        motor_gpio_initialised = False


def initialise_braking_motor_control():
    """Prepare GPIO direction/power pins and PWM channel for braking motor drive."""
    global braking_motor_pwm_channel_path
    global braking_motor_control_available
    global braking_motor_gpio_initialised
    global braking_motor_last_duty_ns
    global braking_motor_pwm_enabled

    braking_motor_control_available = False
    if GPIO is None:
        return

    try:
        if not braking_motor_gpio_initialised:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)
            GPIO.setup(braking_motor_direction_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(braking_motor_power_pin, GPIO.OUT, initial=GPIO.LOW)
            braking_motor_gpio_initialised = True
    except RuntimeError:
        return

    try:
        braking_motor_pwm_channel_path = _ensure_pwm_channel(
            braking_motor_pwm_chip, braking_motor_pwm_channel
        )
        _pwm_write(braking_motor_pwm_channel_path / "enable", 0)
        _pwm_write(braking_motor_pwm_channel_path / "period", braking_motor_pwm_period_ns)
        _pwm_write(braking_motor_pwm_channel_path / "duty_cycle", 0)
        _pwm_write(braking_motor_pwm_channel_path / "enable", 1)
        braking_motor_last_duty_ns = 0
        braking_motor_pwm_enabled = True
        braking_motor_control_available = True
    except (OSError, FileNotFoundError, TimeoutError):  # pragma: no cover - hardware dependency
        braking_motor_pwm_channel_path = None
        braking_motor_control_available = False


def _set_braking_motor_direction(error: float) -> None:
    if GPIO is None or not braking_motor_gpio_initialised:
        return
    GPIO.output(braking_motor_direction_pin, GPIO.HIGH if error > 0 else GPIO.LOW)


def _set_braking_motor_pwm_pct(pct: float) -> None:
    global braking_motor_last_duty_ns
    if (
        not braking_motor_control_available
        or braking_motor_pwm_channel_path is None
        or not braking_motor_pwm_enabled
    ):
        return
    duty_ns = _ns_duty_from_pct(braking_motor_pwm_period_ns, pct)
    if braking_motor_last_duty_ns == duty_ns:
        return
    try:
        _pwm_write(braking_motor_pwm_channel_path / "duty_cycle", duty_ns)
        braking_motor_last_duty_ns = duty_ns
    except OSError:  # pragma: no cover - hardware dependency
        pass


def disable_braking_motor_pwm() -> None:
    global braking_motor_pwm_enabled, braking_motor_last_duty_ns
    if not braking_motor_control_available or braking_motor_pwm_channel_path is None:
        return
    _set_braking_motor_pwm_pct(0.0)
    try:
        _pwm_write(braking_motor_pwm_channel_path / "enable", 0)
    except OSError:  # pragma: no cover - hardware dependency
        return
    braking_motor_last_duty_ns = 0
    braking_motor_pwm_enabled = False
    if GPIO is not None and braking_motor_gpio_initialised:
        GPIO.output(braking_motor_power_pin, GPIO.LOW)


def enable_braking_motor_pwm() -> None:
    global braking_motor_pwm_enabled, braking_motor_last_duty_ns
    if not braking_motor_control_available or braking_motor_pwm_channel_path is None:
        return
    try:
        _pwm_write(braking_motor_pwm_channel_path / "duty_cycle", 0)
        _pwm_write(braking_motor_pwm_channel_path / "enable", 1)
    except OSError:  # pragma: no cover - hardware dependency
        return
    braking_motor_last_duty_ns = 0
    braking_motor_pwm_enabled = True
    if GPIO is not None and braking_motor_gpio_initialised:
        GPIO.output(braking_motor_power_pin, GPIO.LOW)


def update_braking_motor_control(target_norm, encoder_norm):
    if target_norm is None or encoder_norm is None or braking_calibration_active:
        _set_braking_motor_direction(0)
        _set_braking_motor_pwm_pct(0.0)
        if GPIO is not None and braking_motor_gpio_initialised:
            GPIO.output(braking_motor_power_pin, GPIO.LOW)
        return

    error = float(target_norm) - float(encoder_norm)

    if abs(error) <= braking_motor_dead_zone_norm:
        _set_braking_motor_direction(0)
        _set_braking_motor_pwm_pct(0.0)
        if GPIO is not None and braking_motor_gpio_initialised:
            GPIO.output(braking_motor_power_pin, GPIO.LOW)
        return

    _set_braking_motor_direction(error)
    denom = (
        float(braking_motor_full_speed_error_norm)
        if braking_motor_full_speed_error_norm > 0
        else 1.0
    )
    scaled_pct = (abs(error) / denom) * braking_motor_max_duty_pct
    duty_pct = max(0.0, min(braking_motor_max_duty_pct, scaled_pct))

    _set_braking_motor_pwm_pct(duty_pct)
    if GPIO is not None and braking_motor_gpio_initialised:
        if braking_motor_pwm_enabled and not braking_calibration_active and duty_pct > 0.0:
            GPIO.output(braking_motor_power_pin, GPIO.HIGH)
        else:
            GPIO.output(braking_motor_power_pin, GPIO.LOW)


def apply_braking_jog_drive(direction: int) -> None:
    if GPIO is not None and braking_motor_gpio_initialised:
        GPIO.output(braking_motor_power_pin, GPIO.LOW)

    if direction not in (-1, 1):
        _set_braking_motor_direction(0)
        _set_braking_motor_pwm_pct(0.0)
        return

    _set_braking_motor_direction(float(direction))
    duty = clamp(braking_jog_duty_pct, 0.0, braking_motor_max_duty_pct)
    _set_braking_motor_pwm_pct(duty)
    if GPIO is not None and braking_motor_gpio_initialised:
        if duty > 0.0:
            GPIO.output(braking_motor_power_pin, GPIO.HIGH)
        else:
            GPIO.output(braking_motor_power_pin, GPIO.LOW)


def shutdown_braking_motor_control():
    global braking_motor_control_available
    global braking_motor_pwm_channel_path
    global braking_motor_last_duty_ns
    global braking_motor_gpio_initialised
    global braking_motor_pwm_enabled

    _set_braking_motor_direction(0)
    _set_braking_motor_pwm_pct(0.0)

    if GPIO is not None and braking_motor_gpio_initialised:
        try:
            GPIO.output(braking_motor_power_pin, GPIO.LOW)
        except RuntimeError:  # pragma: no cover - hardware dependency
            pass

    if braking_motor_control_available and braking_motor_pwm_channel_path is not None:
        try:
            _pwm_write(braking_motor_pwm_channel_path / "enable", 0)
        except OSError:  # pragma: no cover - hardware dependency
            pass

    braking_motor_control_available = False
    braking_motor_pwm_channel_path = None
    braking_motor_last_duty_ns = None
    braking_motor_pwm_enabled = False

    if GPIO is not None and braking_motor_gpio_initialised:
        try:
            GPIO.cleanup([braking_motor_direction_pin, braking_motor_power_pin])
        except RuntimeError:  # pragma: no cover - hardware dependency
            pass
        braking_motor_gpio_initialised = False


def get_braking_encoder_norm():
    global braking_latest_encoder_raw, braking_latest_encoder_norm
    raw = read_braking_encoder_raw()
    braking_latest_encoder_raw = raw
    if raw is None:
        braking_latest_encoder_norm = None
        return None
    norm = braking_encoder_raw_to_norm(raw)
    braking_latest_encoder_norm = norm
    return norm


def start_braking_calibration():
    global braking_calibration_active
    global braking_calibration_stage
    global braking_calibration_samples
    global braking_calibration_status_text
    global braking_calibration_status_until
    global braking_calibration_jog_direction
    if not braking_encoder_available or braking_encoder_bus is None:
        braking_calibration_status_text, braking_calibration_status_until = set_status_message(
            "Brake cal unavailable: encoder not detected",
            3.0,
        )
        braking_calibration_active = False
        braking_calibration_stage = None
        return
    disable_braking_motor_pwm()
    braking_calibration_jog_direction = 0
    apply_braking_jog_drive(0)
    braking_calibration_active = True
    braking_calibration_stage = "min"
    braking_calibration_samples = {}
    braking_calibration_status_text, braking_calibration_status_until = set_status_message(
        "Brake cal: jog LEFT (V/N), press SPACE to set min",
        None,
    )


def capture_braking_calibration_point():
    global braking_calibration_active
    global braking_calibration_stage
    global braking_calibration_status_text
    global braking_calibration_status_until
    global braking_calibration_samples
    global braking_calibration_jog_direction
    if not braking_encoder_available or braking_encoder_bus is None:
        braking_calibration_status_text, braking_calibration_status_until = set_status_message(
            "Brake cal failed: encoder not detected",
            3.0,
        )
        braking_calibration_active = False
        braking_calibration_stage = None
        braking_calibration_jog_direction = 0
        apply_braking_jog_drive(0)
        enable_braking_motor_pwm()
        return
    raw = read_braking_encoder_raw()
    if raw is None:
        braking_calibration_status_text, braking_calibration_status_until = set_status_message(
            "Brake cal failed: encoder read error",
            3.0,
        )
        braking_calibration_active = False
        braking_calibration_stage = None
        braking_calibration_jog_direction = 0
        apply_braking_jog_drive(0)
        enable_braking_motor_pwm()
        return
    if braking_calibration_stage == "min":
        braking_calibration_samples["min"] = raw
        braking_calibration_stage = "max"
        braking_calibration_status_text, braking_calibration_status_until = set_status_message(
            "Brake cal: jog RIGHT (V/N), press SPACE to set max",
            None,
        )
    elif braking_calibration_stage == "max":
        braking_calibration_samples["max"] = raw
        min_raw = min(braking_calibration_samples["min"], braking_calibration_samples["max"])
        max_raw = max(braking_calibration_samples["min"], braking_calibration_samples["max"])
        if min_raw == max_raw:
            braking_calibration_status_text, braking_calibration_status_until = set_status_message(
                "Brake cal failed: encoder range is zero",
                3.0,
            )
        else:
            save_braking_calibration(min_raw, max_raw)
            braking_calibration_status_text, braking_calibration_status_until = set_status_message(
                "Brake calibration saved",
                3.0,
            )
        braking_calibration_active = False
        braking_calibration_stage = None
        braking_calibration_jog_direction = 0
        apply_braking_jog_drive(0)
        enable_braking_motor_pwm()


def get_encoder_norm():
    global latest_encoder_raw, latest_encoder_norm
    if sim_encoder_enabled:
        latest_encoder_raw = None
        latest_encoder_norm = sim_encoder_norm
        return sim_encoder_norm
    raw = read_encoder_raw()
    latest_encoder_raw = raw
    if raw is None:
        latest_encoder_norm = None
        return None
    norm = encoder_raw_to_norm(raw)
    latest_encoder_norm = norm
    return norm


def start_calibration():
    global calibration_active, calibration_stage, calibration_samples, calibration_status_text, calibration_status_until
    global calibration_jog_direction
    if not encoder_available or encoder_bus is None:
        calibration_status_text, calibration_status_until = set_status_message(
            "Steer cal unavailable: encoder not detected",
            3.0,
        )
        calibration_active = False
        calibration_stage = None
        return
    disable_motor_pwm()
    calibration_jog_direction = 0
    apply_jog_drive(0)
    calibration_active = True
    calibration_stage = "min"
    calibration_samples = {}
    calibration_status_text, calibration_status_until = set_status_message(
        "Steer cal: jog LEFT (J/L), press SPACE to set min",
        None,
    )


def capture_calibration_point():
    global calibration_active, calibration_stage, calibration_status_text, calibration_status_until
    global calibration_samples, calibration_jog_direction
    if not encoder_available or encoder_bus is None:
        calibration_status_text, calibration_status_until = set_status_message(
            "Steer cal failed: encoder not detected",
            3.0,
        )
        calibration_active = False
        calibration_stage = None
        calibration_jog_direction = 0
        apply_jog_drive(0)
        enable_motor_pwm()
        return
    raw = read_encoder_raw()
    if raw is None:
        calibration_status_text, calibration_status_until = set_status_message(
            "Steer cal failed: encoder read error",
            3.0,
        )
        calibration_active = False
        calibration_stage = None
        calibration_jog_direction = 0
        apply_jog_drive(0)
        enable_motor_pwm()
        return
    if calibration_stage == "min":
        calibration_samples["min"] = raw
        calibration_stage = "max"
        calibration_status_text, calibration_status_until = set_status_message(
            "Steer cal: jog RIGHT (J/L), press SPACE to set max",
            None,
        )
    elif calibration_stage == "max":
        calibration_samples["max"] = raw
        min_raw = min(calibration_samples["min"], calibration_samples["max"])
        max_raw = max(calibration_samples["min"], calibration_samples["max"])
        if min_raw == max_raw:
            calibration_status_text, calibration_status_until = set_status_message(
                "Steer cal failed: encoder range is zero",
                3.0,
            )
        else:
            save_calibration(min_raw, max_raw)
            calibration_status_text, calibration_status_until = set_status_message(
                "Steering calibration saved",
                3.0,
            )
        calibration_active = False
        calibration_stage = None
        calibration_jog_direction = 0
        apply_jog_drive(0)
        enable_motor_pwm()


# Build the camera-specific GStreamer pipeline string for a Jetson device
def make_gst(sensor_id, fps=30):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width=300,height=150,framerate={fps}/1,format=NV12 ! "
        "nvvidconv flip-method=2 ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "appsink sync=false max-buffers=1 drop=true enable-last-sample=0"
    )


# Open both cameras and trim their internal buffers for minimal latency
cap0 = cv2.VideoCapture(make_gst(0, fps=30), cv2.CAP_GSTREAMER)
cap1 = cv2.VideoCapture(make_gst(1, fps=30), cv2.CAP_GSTREAMER)
if not cap0.isOpened() or not cap1.isOpened():
    raise RuntimeError("Failed to open one or both cameras.")

cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Warm up cameras so Argus allocates buffers before CUDA model grabs memory
for _ in range(30):
    cap0.read()
    cap1.read()

# Now load depth model (smaller + FP16 to reduce memory pressure)
depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    device=0,
    torch_dtype=torch.float16,
)


# Thread-safe queues for streaming frames and depth inference results
frame_q0, frame_q1, result_q = queue.Queue(1), queue.Queue(1), queue.Queue(1)
running = True


# ---------------------------------------------------------------------------
# Background threads: one per camera for capture, one shared for inference
# ---------------------------------------------------------------------------

def grab(cam, q):
    """Continuously read frames from `cam` and keep the freshest one in `q`."""
    while running:
        ret, frame = cam.read()
        if ret:
            if q.full(): q.get_nowait()
            q.put(frame)

Thread(target=grab, args=(cap0, frame_q0), daemon=True).start()
Thread(target=grab, args=(cap1, frame_q1), daemon=True).start()


def _ensure_sample_grid(height, width, x_offset):
    """Build and cache integer pixel coordinates for sparse depth sampling."""
    key = (height, width, x_offset)
    grid = _sample_grid_cache.get(key)
    if grid is None:
        col_coords = ((np.arange(cols, dtype=np.float32) + 0.5) * width / cols).astype(np.int32)
        row_coords = ((np.arange(rows, dtype=np.float32) + 0.5) * height / rows).astype(np.int32)
        px_local = np.repeat(col_coords, rows)
        py = np.tile(row_coords, cols)
        grid = {
            "px_local": px_local,
            "py": py,
            "px_global": (px_local + int(x_offset)).astype(np.int32),
        }
        _sample_grid_cache[key] = grid
    return grid


def infer():
    """Fetch paired frames, run depth inference, and push the results downstream."""
    while running:
        try:
            f0 = frame_q0.get(timeout=0.01)
            f1 = frame_q1.get(timeout=0.01)
        except queue.Empty:
            continue

        img0 = Image.fromarray(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB))
        img1 = Image.fromarray(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB))

        with torch.inference_mode():
            o0 = depth_pipe(img0)
            o1 = depth_pipe(img1)

        d0 = np.array(o0["depth"])
        d1 = np.array(o1["depth"])

        if result_q.full():
            result_q.get_nowait()
        result_q.put((f0, d0, f1, d1))


Thread(target=infer, daemon=True).start()

load_calibration()
initialise_encoder()
initialise_motor_control()
load_braking_calibration()
initialise_braking_encoder()
initialise_braking_motor_control()

# ---------------------------------------------------------------------------
# Main processing loop — consume depth results, annotate, and display
# ---------------------------------------------------------------------------
while True:
    try:
        frame0, depth0, frame1, depth1 = result_q.get(timeout=0.01)
    except queue.Empty:
        continue

    # Build a sparse point cloud by sampling each depth map on a coarse grid
    h0, w0 = depth0.shape
    h1, w1 = depth1.shape
    grid0 = _ensure_sample_grid(h0, w0, 0)
    grid1 = _ensure_sample_grid(h1, w1, w0)

    px0_local, py0, px0_global = grid0["px_local"], grid0["py"], grid0["px_global"]
    px1_local, py1, px1_global = grid1["px_local"], grid1["py"], grid1["px_global"]

    z0 = depth0[py0, px0_local]
    z1 = depth1[py1, px1_local]

    count0 = px0_global.size
    count1 = px1_global.size
    cloud = np.empty(count0 + count1, dtype=[('x','f4'),('y','f4'),('z','f4'),('cam','i4')])

    cloud['x'][:count0] = px0_global.astype(np.float32)
    cloud['x'][count0:] = px1_global.astype(np.float32)
    cloud['y'][:count0] = py0.astype(np.float32)
    cloud['y'][count0:] = py1.astype(np.float32)
    cloud['z'][:count0] = z0.astype(np.float32)
    cloud['z'][count0:] = z1.astype(np.float32)
    cloud['cam'][:count0] = 0
    cloud['cam'][count0:] = 1

    # Determine obstacle candidates by comparing each sample to the scene average
    zs = cloud['z']
    mean_z, std_z = zs.mean(), zs.std()
    thresh = max(depth_diff_threshold, std_multiplier * std_z)
    mask = (mean_z - zs) > thresh

    # Convert depth arrays into coloured heatmaps for easier interpretation
    norm0 = cv2.normalize(depth0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cmap0 = cv2.applyColorMap(cv2.resize(norm0, (frame0.shape[1], frame0.shape[0])), cv2.COLORMAP_MAGMA)
    norm1 = cv2.normalize(depth1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cmap1 = cv2.applyColorMap(cv2.resize(norm1, (frame1.shape[1], frame1.shape[0])), cv2.COLORMAP_MAGMA)

    # Draw cutoff lines on each camera to mark the active sensing band
    for cmap, h, w in [(cmap0, frame0.shape[0], frame0.shape[1]), (cmap1, frame1.shape[0], frame1.shape[1])]:
        top_y = top_cutoff_pixels
        bottom_y = h - bottom_cutoff_pixels
        cv2.line(cmap, (0, top_y), (w, top_y), cutoff_line_color, cutoff_line_thickness)
        cv2.line(cmap, (0, bottom_y), (w, bottom_y), cutoff_line_color, cutoff_line_thickness)

    # Show combined view
    combined = np.hstack((cmap0, cmap1))

    
    
    # === 🟦 HORIZONTAL GAP ROUTE PLANNER ────────────────────────────────
    h_combined, w_combined = frame0.shape[0], frame0.shape[1] + frame1.shape[1]

    # vertical midpoint between the two green cutoff lines (for blue circle)
    line_y = (top_cutoff_pixels + (h_combined - bottom_cutoff_pixels)) // 2

    # combined-frame center X, optionally offset for asymmetric steering bias
    center_x = (w_combined // 2) + pull_zone_center_offset_px

    # pull zone boundaries (clamped to image)
    zone_left = clamp(center_x - pull_influence_radius_px, 0, w_combined)
    zone_right = clamp(center_x + pull_influence_radius_px, 0, w_combined)

    # Collect red-dot X positions that fall between the cutoff green lines (both cams)
    red_xs_all = []
    red_xs_in_zone = []

    if mask.any():
        obstacles = cloud[mask]
        if obstacles.size:
            cam_indices = obstacles['cam']
            py_vals = obstacles['y'].astype(np.int32)
            px_vals = obstacles['x'].astype(np.int32)
            heights = np.where(cam_indices == 0, frame0.shape[0], frame1.shape[0])
            valid = (py_vals >= top_cutoff_pixels) & (py_vals <= (heights - bottom_cutoff_pixels))
            if np.any(valid):
                px_filtered = px_vals[valid]
                red_xs_all = np.unique(px_filtered).tolist()
                in_zone = (px_filtered >= zone_left) & (px_filtered <= zone_right)
                red_xs_in_zone = np.unique(px_filtered[in_zone]).tolist()

    # Deduplicate & sort to simplify gap computation
    red_xs_all = sorted(set(red_xs_all))
    red_xs_in_zone = sorted(set(red_xs_in_zone))

    # Global blockers across the full width
    blockers_all = [0] + red_xs_all + [w_combined]

    def widest_gap_center(blockers, preferred_x):
        """
        Choose the center of the widest gap considering ALL blockers.
        Tie-breaker: pick the gap whose center is closest to preferred_x.
        """
        if len(blockers) < 2:
            return preferred_x
        best_width = -1
        best_dist = 1e18
        best_cx = preferred_x
        for left, right in zip(blockers[:-1], blockers[1:]):
            width = right - left
            cx = (left + right) // 2
            dist = abs(cx - preferred_x)
            if (width > best_width) or (width == best_width and dist < best_dist):
                best_width = width
                best_dist = dist
                best_cx = cx
        return int(best_cx)

    # Gating:
    # - No in-zone obstacles -> ignore outside, keep center
    # - Any in-zone obstacles -> plan using ALL obstacles
    if len(red_xs_in_zone) == 0:
        gap_cx = center_x
    else:
        gap_cx = widest_gap_center(blockers_all, preferred_x=center_x)

    # Smooth horizontal motion (keep as FLOAT; don't floor!)
    if blue_x is None:
        blue_x = float(gap_cx)
    else:
        blue_x = blue_x * blue_x_smoothness + gap_cx * (1 - blue_x_smoothness)

    # Apply divert boost in the direction of travel (left/right) before rendering
    boosted_blue_x = blue_x
    if divert_boost and center_x is not None:
        if blue_x < center_x:
            boosted_blue_x = max(0.0, blue_x - float(divert_boost))
        elif blue_x > center_x:
            boosted_blue_x = min(float(w_combined - 1), blue_x + float(divert_boost))

    # Round ONLY for rendering / display to avoid bias
    draw_x = int(round(boosted_blue_x))

    # Update "Steer Control" variable every loop
    steer_control_x = draw_x

    encoder_norm = get_encoder_norm()
    encoder_px = None
    if encoder_norm is not None:
        scale = max(1, w_combined - 1)
        encoder_px = int(round(encoder_norm * scale))
        encoder_px = int(clamp(encoder_px, 0, w_combined - 1))
        cv2.line(
            combined,
            (encoder_px, 0),
            (encoder_px, h_combined),
            (0, 255, 255),
            1,
        )

    if jog_mode_enabled:
        if jog_direction != 0:
            apply_jog_drive(jog_direction)
        else:
            apply_jog_drive(0)
    elif calibration_active and calibration_jog_direction != 0:
        if not motor_pwm_enabled:
            enable_motor_pwm()
        apply_jog_drive(calibration_jog_direction)
    else:
        update_motor_control(boosted_blue_x if blue_x is not None else None, encoder_px)

    braking_encoder_norm = get_braking_encoder_norm()
    if braking_jog_mode_enabled:
        if braking_jog_direction != 0:
            apply_braking_jog_drive(braking_jog_direction)
        else:
            apply_braking_jog_drive(0)
    elif braking_calibration_active and braking_calibration_jog_direction != 0:
        if not braking_motor_pwm_enabled:
            enable_braking_motor_pwm()
        apply_braking_jog_drive(braking_calibration_jog_direction)
    else:
        update_braking_motor_control(braking_target_norm, braking_encoder_norm)

    # Draw the guidance circle
    blue_pos = (draw_x, line_y)
    cv2.circle(combined, blue_pos, blue_circle_radius_px, blue_circle_color, blue_circle_thickness)

    if not help_overlay_enabled:
        # Display "Steer Control" on the left frame
        draw_text_with_outline(
            combined,
            f"Steer Control: {steer_control_x}",
            hud_text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            hud_text_scale,
            hud_text_color,
            hud_text_thickness,
            hud_text_outline_color,
            hud_text_outline_thickness,
        )

        encoder_text = "Steer Encoder: --"
        if encoder_px is not None:
            encoder_text = f"Steer Encoder: {encoder_px} px"
        elif sim_encoder_enabled:
            encoder_text = "Steer Encoder: simulated"
        draw_text_with_outline(
            combined,
            encoder_text,
            (hud_text_position[0], hud_text_position[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            hud_text_scale,
            hud_text_color,
            hud_text_thickness,
            hud_text_outline_color,
            hud_text_outline_thickness,
        )

        brake_encoder_text = "Break Encoder: --"
        if braking_encoder_norm is not None:
            brake_encoder_text = f"Break Encoder: {braking_encoder_norm:.3f}"
        draw_text_with_outline(
            combined,
            brake_encoder_text,
            (hud_text_position[0], hud_text_position[1] + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            hud_text_scale,
            hud_text_color,
            hud_text_thickness,
            hud_text_outline_color,
            hud_text_outline_thickness,
        )

    # Visualize the pull zone boundaries (blue vertical lines)
    cv2.line(
        combined,
        (int(zone_left), 0),
        (int(zone_left), h_combined),
        pull_zone_line_color,
        pull_zone_line_thickness,
    )
    cv2.line(
        combined,
        (int(zone_right), 0),
        (int(zone_right), h_combined),
        pull_zone_line_color,
        pull_zone_line_thickness,
    )
    # ────────────────────────────────────────────────────────────────────

    if sim_encoder_enabled:
        slider_left = 20
        slider_right = w_combined - 20
        slider_y = h_combined - 30
        cv2.line(
            combined,
            (slider_left, slider_y),
            (slider_right, slider_y),
            (200, 200, 200),
            2,
        )
        slider_x = int(round(slider_left + sim_encoder_norm * (slider_right - slider_left)))
        cv2.circle(combined, (slider_x, slider_y), 8, (0, 200, 255), -1)

    bottom_lines = []
    control_active = (
        sim_encoder_enabled
        or jog_mode_enabled
        or braking_jog_mode_enabled
        or calibration_active
        or braking_calibration_active
    )
    now = time.time()
    if calibration_status_text and calibration_status_until and now > calibration_status_until:
        calibration_status_text = ""
        calibration_status_until = 0.0
    if braking_calibration_status_text and braking_calibration_status_until and now > braking_calibration_status_until:
        braking_calibration_status_text = ""
        braking_calibration_status_until = 0.0

    if help_overlay_enabled:
        help_title = "Help (press H to close)"
        help_lines = [
            "ESC: Exit",
            "H: Toggle this help overlay",
            "S: Toggle simulated steer encoder",
            "\u2190/\u2192: Adjust simulated steer encoder",
            "K: Toggle steer jog mode",
            "J/L: Steer jog left/right",
            "^/v: Adjust jog speed",
            "B: Toggle brake jog mode",
            "V/N: Brake jog left/right",
            "C: Start steer calibration",
            "X: Start brake calibration",
            "SPACE: Capture calibration point",
        ]
        draw_help_overlay(
            combined,
            help_title,
            help_lines,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            help_title_scale,
            help_text_scale,
            hud_text_color,
            hud_text_thickness,
            hud_text_outline_color,
            hud_text_outline_thickness,
            help_text_line_gap,
            help_text_column_gap,
        )
    else:
        if calibration_status_text:
            bottom_lines.append(calibration_status_text)
        if braking_calibration_status_text:
            bottom_lines.append(braking_calibration_status_text)
        if jog_mode_enabled:
            if jog_direction < 0:
                jog_state = "left"
            elif jog_direction > 0:
                jog_state = "right"
            else:
                jog_state = "idle"
            bottom_lines.append(
                f"Steer jog: {jog_state} @ {jog_duty_pct:.0f}% (J/L to drive, ^/v speed, K to exit)"
            )
        if braking_jog_mode_enabled:
            if braking_jog_direction < 0:
                braking_jog_state = "left"
            elif braking_jog_direction > 0:
                braking_jog_state = "right"
            else:
                braking_jog_state = "idle"
            bottom_lines.append(
                f"Brake jog: {braking_jog_state} @ {braking_jog_duty_pct:.0f}% (V/N to drive, ^/v speed, B to exit)"
            )
        if sim_encoder_enabled:
            bottom_lines.append("Sim encoder: \u2190/\u2192 adjust, S to exit")
        if not control_active:
            bottom_lines.append('Show help: "h"')
        if bottom_lines:
            draw_bottom_lines(
                combined,
                bottom_lines,
                (20, h_combined - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                hud_text_scale,
                hud_text_color,
                hud_text_thickness,
                hud_text_outline_color,
                hud_text_outline_thickness,
                20,
            )

    # Present the final annotated view
    cv2.imshow("Depth: Camera 0 | Camera 1", combined)
    key = cv2.waitKey(1) & 0xFF

    # Handle keys — currently only ESC to exit
    if key == 27:
        break

    if key == ord('h'):
        help_overlay_enabled = not help_overlay_enabled

    if key == ord('s'):
        sim_encoder_enabled = not sim_encoder_enabled
        if sim_encoder_enabled:
            if latest_encoder_norm is not None:
                sim_encoder_norm = latest_encoder_norm
            else:
                sim_encoder_norm = 0.5
            ui_notice_text = "Sim encoder enabled"
            ui_notice_until = time.time() + 3.0
        else:
            ui_notice_text = "Sim encoder disabled"
            ui_notice_until = time.time() + 3.0

    if key == ord('k'):
        jog_mode_enabled = not jog_mode_enabled
        jog_direction = 0
        apply_jog_drive(0)
        if jog_mode_enabled:
            ui_notice_text = "Steer jog enabled"
            ui_notice_until = time.time() + 3.0
        else:
            ui_notice_text = "Steer jog disabled"
            ui_notice_until = time.time() + 3.0

    if key == ord('b'):
        braking_jog_mode_enabled = not braking_jog_mode_enabled
        braking_jog_direction = 0
        apply_braking_jog_drive(0)
        if braking_jog_mode_enabled:
            ui_notice_text = "Brake jog enabled"
            ui_notice_until = time.time() + 3.0
        else:
            ui_notice_text = "Brake jog disabled"
            ui_notice_until = time.time() + 3.0

    if calibration_active and key in (ord('j'), ord('l')):
        requested_direction = -1 if key == ord('j') else 1
        if calibration_jog_direction == requested_direction:
            calibration_jog_direction = 0
            apply_jog_drive(0)
        else:
            calibration_jog_direction = requested_direction
            if not motor_pwm_enabled:
                enable_motor_pwm()
            apply_jog_drive(calibration_jog_direction)

    if braking_calibration_active and key in (ord('v'), ord('n')):
        requested_direction = -1 if key == ord('v') else 1
        if braking_calibration_jog_direction == requested_direction:
            braking_calibration_jog_direction = 0
            apply_braking_jog_drive(0)
        else:
            braking_calibration_jog_direction = requested_direction
            if not braking_motor_pwm_enabled:
                enable_braking_motor_pwm()
            apply_braking_jog_drive(braking_calibration_jog_direction)

    elif jog_mode_enabled and key in (ord('j'), ord('l')):
        requested_direction = -1 if key == ord('j') else 1
        if jog_direction == requested_direction:
            jog_direction = 0
        else:
            jog_direction = requested_direction
        apply_jog_drive(jog_direction)

    elif braking_jog_mode_enabled and key in (ord('v'), ord('n')):
        requested_direction = -1 if key == ord('v') else 1
        if braking_jog_direction == requested_direction:
            braking_jog_direction = 0
        else:
            braking_jog_direction = requested_direction
        apply_braking_jog_drive(braking_jog_direction)

    if sim_encoder_enabled and key in (81, 83):
        delta = -simulated_step_norm if key == 81 else simulated_step_norm
        sim_encoder_norm = clamp(sim_encoder_norm + delta, 0.0, 1.0)

    if jog_mode_enabled and key in (82, 84):
        delta = jog_duty_step_pct if key == 82 else -jog_duty_step_pct
        jog_duty_pct = clamp(jog_duty_pct + delta, 0.0, motor_max_duty_pct)
    if braking_jog_mode_enabled and key in (82, 84):
        delta = braking_jog_duty_step_pct if key == 82 else -braking_jog_duty_step_pct
        braking_jog_duty_pct = clamp(delta + braking_jog_duty_pct, 0.0, braking_motor_max_duty_pct)

    if key == ord('c'):
        start_calibration()
    if key == ord('x'):
        start_braking_calibration()

    if braking_calibration_active and key == 32:  # space bar
        capture_braking_calibration_point()
    elif calibration_active and key == 32:  # space bar
        capture_calibration_point()

# === Cleanup ===
running = False
cap0.release()
cap1.release()
cv2.destroyAllWindows()
shutdown_encoder()
shutdown_motor_control()
shutdown_braking_encoder()
shutdown_braking_motor_control()





# ---------------------------------------------------------------------------
# AS5600 → Jetson Nano wiring quick reference
# ---------------------------------------------------------------------------
# 1. Power: connect AS5600 VCC to the Jetson Nano's 3.3 V pin (pin 1 or 17) and
#    AS5600 GND to any ground pin (for example pin 6).
# 2. Signal pins (left to right when text is facing you):
#      • OUT can optionally feed an analog reader; it is unused in this script.
#      • DIR selects count direction; tie to GND for default behaviour.
#      • SCL → Jetson Nano I²C SCL (physical pin 5 on I2C bus 1).
#      • SDA → Jetson Nano I²C SDA (physical pin 3 on I2C bus 1).
#      • GPO is an optional programmable output; leave unconnected unless used.
# 3. Keep wiring short and add a 0.1 µF decoupling capacitor close to the sensor
#    supply pins for stable readings.
# 4. Enable the I²C interface on the Jetson Nano via
#    `sudo /opt/nvidia/jetson-io/jetson-io.py` (interface configuration) and
#    reboot afterwards. Once enabled, `/dev/i2c-1` should exist and the script
#    will be able to talk to the encoder.
# 5. Mount the AS5600 so the magnet sits centred above the die at the specified
#    air gap (≈1–2 mm). Ensure the magnet rotates with the steering shaft.

# ---------------------------------------------------------------------------
# Braking AS5600 wiring quick reference
# ---------------------------------------------------------------------------
# 1. Power: connect AS5600 VCC to 3.3 V and GND to any ground pin.
# 2. Signal pins (left to right when text is facing you):
#      • OUT can optionally feed an analog reader; it is unused in this script.
#      • DIR selects count direction; tie to GND for default behaviour.
#      • SCL → Jetson Nano pin 28 (I2C on braking_encoder_i2c_bus).
#      • SDA → Jetson Nano pin 27 (I2C on braking_encoder_i2c_bus).
#      • GPO is an optional programmable output; leave unconnected unless used.
# 3. Keep wiring short and add a 0.1 µF decoupling capacitor close to the sensor.

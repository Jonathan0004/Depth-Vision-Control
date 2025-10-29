"""Stereo depth estimation with obstacle detection and steering guidance.

The script streams from two cameras, infers depth using the
``depth-anything`` model, highlights obstacles inside configurable cutoff
bands, and visualises a blue steering cue that points toward the widest gap.
All tunable parameters live together for quick iteration.
"""

import json
import os
import queue
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

try:
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

# Rendering toggle — set to False to run everything headless without UI
enable_visualization = True

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
blue_x_smoothness = 0.7

# HUD text overlays on the combined frame
hud_text_position = (10, 30)
hud_text_color = (255, 255, 255)
hud_text_scale = 0.54
hud_text_thickness = 1

# Motor dynamics — duty cycle ramps linearly with steering error
motor_max_duty_pct = 100.0          # absolute cap on PWM duty cycle
motor_full_speed_error_px = 200    # error (px) required to request max duty

# PWM configuration — single output used to control motor speed (pin 32)
motor_pwm_output = {"chip": "/sys/class/pwm/pwmchip3", "channel": 0}
motor_direction_gpio_pins = {"forward": 26, "reverse": 24}
pwm_frequency_hz = 8000

# GPIO output that signals when the motor is actively driving (Jetson physical pin 29)
motor_activity_gpio_pin = 29

# Encoder + steering control configuration
encoder_i2c_bus = 7
encoder_i2c_address = 0x36
encoder_deadband_px = 5          # stop when within this many screen pixels of target
calibration_file = Path("steering_calibration.json")
simulated_step_norm = 0.01        # arrow-key increment when in simulated encoder mode


# ---------------------------------------------------------------------------
# Runtime state (initialised once, then updated as frames stream in)
# ---------------------------------------------------------------------------
blue_x = None              # persistent, smoothed x-position of the blue circle
steer_control_x = None     # integer pixel location displayed as HUD text


# Cache for sampling grids so expensive mesh computations run once per shape
_sample_grid_cache = {}


# ---------------------------------------------------------------------------
# Helper utilities and pipeline initialisation
# ---------------------------------------------------------------------------

# Utility: clamp values between two bounds (used to keep overlays on-screen)
def clamp(val, minn, maxn):
    return max(min(val, maxn), minn)


# ---------------------------------------------------------------------------
# Motor + PWM helpers
# ---------------------------------------------------------------------------

import errno

def _pwm_wr(path, val, retries=40, delay_s=0.05):
    """
    Robust write for /sys/class/pwm attributes.
    Retries briefly to tolerate udev applying perms after export,
    and the common 'Invalid argument'/'busy' races if the driver
    isn’t ready yet or enable hasn't settled to 0.
    """
    s = str(val)
    last_err = None
    for _ in range(retries):
        try:
            with open(path, "w") as f:
                f.write(s)
            return
        except PermissionError as e:
            last_err = e
        except OSError as e:
            # EINVAL (22) => timing/order issue (e.g., trying to set period too early)
            # EBUSY  (16) => attribute busy while driver flips state
            # EACCES/EPERM => perms not applied yet
            if e.errno not in (errno.EINVAL, errno.EBUSY, errno.EACCES, errno.EPERM):
                raise
            last_err = e
        time.sleep(delay_s)
    # Surface the last error if we never succeeded
    raise last_err if last_err is not None else RuntimeError(f"Failed to write {path}")



def _pwm_ns_period(freq_hz):
    return int(round(1e9 / float(freq_hz)))


def _pwm_ns_duty(period_ns, pct):
    pct = max(0.0, min(100.0, float(pct)))
    return int(period_ns * (pct / 100.0))


def _pwm_channel_path(chip_path, channel_idx):
    return f"{chip_path}/pwm{channel_idx}"


def _ensure_pwm_channel(chip_path, channel_idx):
    ch_path = _pwm_channel_path(chip_path, channel_idx)
    if os.path.isdir(ch_path):
        return ch_path

    if not os.path.isdir(chip_path):
        raise FileNotFoundError(f"{chip_path} does not exist. Check PWM configuration.")

    _pwm_wr(f"{chip_path}/export", str(channel_idx))
    for _ in range(200):
        if os.path.isdir(ch_path):
            break
        time.sleep(0.01)
    else:
        raise TimeoutError(f"PWM channel pwm{channel_idx} did not appear after export")

    return ch_path


motor_pwm_channel_path = None


def initialise_motor_outputs():
    global motor_pwm_channel_path
    motor_pwm_channel_path = None

    period_ns = _pwm_ns_period(pwm_frequency_hz)

    chip_path = motor_pwm_output["chip"]
    channel_idx = motor_pwm_output["channel"]
    ch_path = _ensure_pwm_channel(chip_path, channel_idx)

    # Always disable first (ignore if already 0), then force duty=0
    try:
        _pwm_wr(f"{ch_path}/enable", 0)
    except OSError:
        pass
    try:
        _pwm_wr(f"{ch_path}/duty_cycle", 0)
    except OSError:
        pass

    # Small settle; some drivers need a moment after disable before period change
    time.sleep(0.02)

    # Now set the target period (duty is guaranteed <= period)
    _pwm_wr(f"{ch_path}/period", period_ns)

    # Redundant but harmless: ensure duty is 0 before enabling
    _pwm_wr(f"{ch_path}/duty_cycle", 0)

    # Enable output
    _pwm_wr(f"{ch_path}/enable", 1)

    motor_pwm_channel_path = ch_path

    return period_ns


def _apply_motor_output(direction, duty_pct, period_ns):
    duty_pct = max(0.0, min(100.0, duty_pct))

    duty_ns = _pwm_ns_duty(period_ns, duty_pct if direction != 0 else 0.0)

    if motor_pwm_channel_path is None:
        raise RuntimeError("Motor PWM channel has not been initialised.")

    set_motor_direction_gpio(direction)
    _pwm_wr(f"{motor_pwm_channel_path}/duty_cycle", duty_ns)


def shutdown_motor_outputs():
    global motor_pwm_channel_path
    if motor_pwm_channel_path is None:
        set_motor_direction_gpio(0)
        return
    try:
        _pwm_wr(f"{motor_pwm_channel_path}/duty_cycle", 0)
        _pwm_wr(f"{motor_pwm_channel_path}/enable", "0")
    except OSError:
        pass
    finally:
        set_motor_direction_gpio(0)
        motor_pwm_channel_path = None


motor_activity_gpio_initialized = False
motor_direction_gpio_initialized = False


def initialise_motor_activity_gpio():
    global motor_activity_gpio_initialized
    motor_activity_gpio_initialized = False
    if GPIO is None:
        return
    try:
        mode = GPIO.getmode()
        if mode != GPIO.BOARD:
            GPIO.setmode(GPIO.BOARD)
        GPIO.setup(motor_activity_gpio_pin, GPIO.OUT, initial=GPIO.LOW)
        motor_activity_gpio_initialized = True
    except (RuntimeError, ValueError):
        motor_activity_gpio_initialized = False


def set_motor_activity_gpio(active):
    if not motor_activity_gpio_initialized:
        return
    try:
        GPIO.output(motor_activity_gpio_pin, GPIO.HIGH if active else GPIO.LOW)
    except (RuntimeError, ValueError):
        pass


def shutdown_motor_activity_gpio():
    global motor_activity_gpio_initialized
    if not motor_activity_gpio_initialized:
        return
    try:
        GPIO.output(motor_activity_gpio_pin, GPIO.LOW)
        GPIO.cleanup(motor_activity_gpio_pin)
    except (RuntimeError, ValueError):
        pass
    finally:
        motor_activity_gpio_initialized = False


def initialise_motor_direction_gpio():
    global motor_direction_gpio_initialized
    motor_direction_gpio_initialized = False
    if GPIO is None:
        return
    try:
        mode = GPIO.getmode()
        if mode != GPIO.BOARD:
            GPIO.setmode(GPIO.BOARD)
        for pin in motor_direction_gpio_pins.values():
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        motor_direction_gpio_initialized = True
    except (RuntimeError, ValueError):
        motor_direction_gpio_initialized = False


def set_motor_direction_gpio(direction):
    if not motor_direction_gpio_initialized:
        return
    try:
        forward_pin = motor_direction_gpio_pins["forward"]
        reverse_pin = motor_direction_gpio_pins["reverse"]
        # Always clear both pins first so they are never simultaneously HIGH
        GPIO.output(forward_pin, GPIO.LOW)
        GPIO.output(reverse_pin, GPIO.LOW)
        if direction > 0:
            GPIO.output(forward_pin, GPIO.HIGH)
        elif direction < 0:
            GPIO.output(reverse_pin, GPIO.HIGH)
    except (RuntimeError, ValueError, KeyError):
        pass


def shutdown_motor_direction_gpio():
    global motor_direction_gpio_initialized
    if not motor_direction_gpio_initialized:
        return
    try:
        for pin in motor_direction_gpio_pins.values():
            GPIO.output(pin, GPIO.LOW)
            GPIO.cleanup(pin)
    except (RuntimeError, ValueError):
        pass
    finally:
        motor_direction_gpio_initialized = False


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
# Runtime steering state (encoder + simulation + motor ramp)
# ---------------------------------------------------------------------------

current_motor_direction = 0
current_motor_duty_pct = 0.0

sim_encoder_enabled = False
sim_encoder_norm = 0.5

latest_encoder_raw = None
latest_encoder_norm = None

calibration_active = False
calibration_stage = None
calibration_status_text = ""
calibration_samples = {}


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


def update_motor_control(target_px, actual_px, _dt, period_ns, motor_enabled):
    global current_motor_direction, current_motor_duty_pct

    if calibration_active or not motor_enabled:
        current_motor_direction = 0
        current_motor_duty_pct = 0.0
        set_motor_activity_gpio(False)
        _apply_motor_output(0, 0.0, period_ns)
        return

    desired_direction = 0
    desired_duty_pct = 0.0

    if actual_px is not None and target_px is not None:
        error_px = target_px - actual_px
        abs_error = abs(error_px)
        if abs_error > encoder_deadband_px:
            desired_direction = 1 if error_px > 0 else -1
            span = max(1.0, motor_full_speed_error_px - encoder_deadband_px)
            ratio = (abs_error - encoder_deadband_px) / span
            ratio = clamp(ratio, 0.0, 1.0)
            desired_duty_pct = clamp(motor_max_duty_pct * ratio, 0.0, motor_max_duty_pct)

    if desired_direction == 0:
        current_motor_direction = 0
        current_motor_duty_pct = 0.0
    else:
        current_motor_direction = desired_direction
        current_motor_duty_pct = desired_duty_pct

    set_motor_activity_gpio(current_motor_direction != 0 and current_motor_duty_pct > 0.0)

    _apply_motor_output(current_motor_direction, current_motor_duty_pct, period_ns)


def start_calibration():
    global calibration_active, calibration_stage, calibration_samples, calibration_status_text
    if not encoder_available or encoder_bus is None:
        calibration_status_text = "Cannot start calibration: encoder interface unavailable"
        calibration_active = False
        calibration_stage = None
        return
    calibration_active = True
    calibration_stage = "min"
    calibration_samples = {}
    calibration_status_text = "Calibration: move steering to LEFT limit, press SPACE"


def capture_calibration_point():
    global calibration_active, calibration_stage, calibration_status_text, calibration_samples
    if not encoder_available or encoder_bus is None:
        calibration_status_text = "Calibration failed: encoder interface unavailable"
        calibration_active = False
        calibration_stage = None
        return
    raw = read_encoder_raw()
    if raw is None:
        calibration_status_text = "Calibration failed: unable to read encoder"
        calibration_active = False
        calibration_stage = None
        return
    if calibration_stage == "min":
        calibration_samples["min"] = raw
        calibration_stage = "max"
        calibration_status_text = "Calibration: move steering to RIGHT limit, press SPACE"
    elif calibration_stage == "max":
        calibration_samples["max"] = raw
        min_raw = min(calibration_samples["min"], calibration_samples["max"])
        max_raw = max(calibration_samples["min"], calibration_samples["max"])
        if min_raw == max_raw:
            calibration_status_text = "Calibration failed: encoder range is zero"
        else:
            save_calibration(min_raw, max_raw)
            calibration_status_text = "Calibration saved"
        calibration_active = False
        calibration_stage = None


# Initialise the depth estimation model once (heavy call, so keep global)
depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    device=0
)

# Build the camera-specific GStreamer pipeline string for a Jetson device
def make_gst(sensor_id):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM),width=300,height=150,framerate=60/1 ! "
        "nvvidconv flip-method=2 ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink sync=false max-buffers=1 drop=true"
    )

# Open both cameras and trim their internal buffers for minimal latency
cap0 = cv2.VideoCapture(make_gst(0), cv2.CAP_GSTREAMER)
cap1 = cv2.VideoCapture(make_gst(1), cv2.CAP_GSTREAMER)
if not cap0.isOpened() or not cap1.isOpened():
    raise RuntimeError("Failed to open one or both cameras.")
cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in (f0, f1)]
        with torch.inference_mode():
            with torch.amp.autocast(device_type="cuda"):
                outs = depth_pipe(imgs)
        d0, d1 = [np.array(o['depth']) for o in outs]
        if result_q.full(): result_q.get_nowait()
        result_q.put((f0, d0, f1, d1))

Thread(target=infer, daemon=True).start()

# Initialise motor outputs and PWM once
motor_period_ns = initialise_motor_outputs()
initialise_motor_direction_gpio()
initialise_motor_activity_gpio()
load_calibration()
initialise_encoder()
last_loop_time = time.perf_counter()

# ---------------------------------------------------------------------------
# Main processing loop — consume depth results, annotate, and display
# ---------------------------------------------------------------------------
while True:
    try:
        frame0, depth0, frame1, depth1 = result_q.get(timeout=0.01)
    except queue.Empty:
        continue

    now = time.perf_counter()
    dt = max(1e-4, now - last_loop_time)
    last_loop_time = now

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

    combined = None
    if enable_visualization:
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
    obstacle_in_pull_zone = len(red_xs_in_zone) > 0

    if not obstacle_in_pull_zone:
        gap_cx = center_x
    else:
        gap_cx = widest_gap_center(blockers_all, preferred_x=center_x)

    # Smooth horizontal motion (keep as FLOAT; don't floor!)
    if blue_x is None:
        blue_x = float(gap_cx)
    else:
        blue_x = blue_x * blue_x_smoothness + gap_cx * (1 - blue_x_smoothness)

    # Round ONLY for rendering / display to avoid bias
    draw_x = int(round(blue_x))

    # Update "Steer Control" variable every loop
    steer_control_x = draw_x

    encoder_norm = get_encoder_norm()
    encoder_px = None
    if encoder_norm is not None:
        scale = max(1, w_combined - 1)
        encoder_px = int(round(encoder_norm * scale))
        encoder_px = int(clamp(encoder_px, 0, w_combined - 1))
        if enable_visualization and combined is not None:
            cv2.line(
                combined,
                (encoder_px, 0),
                (encoder_px, h_combined),
                (0, 255, 255),
                1,
            )

    update_motor_control(draw_x, encoder_px, dt, motor_period_ns, obstacle_in_pull_zone)





    if enable_visualization and combined is not None:
        # Draw the guidance circle
        blue_pos = (draw_x, line_y)
        cv2.circle(combined, blue_pos, blue_circle_radius_px, blue_circle_color, blue_circle_thickness)

        # Display "Steer Control" on the left frame
        cv2.putText(
            combined,
            f"Steer Control: {steer_control_x}",
            hud_text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            hud_text_scale,
            hud_text_color,
            hud_text_thickness,
            cv2.LINE_AA
        )

        encoder_text = "Encoder: --"
        if encoder_px is not None:
            encoder_text = f"Encoder: {encoder_px} px"
        elif sim_encoder_enabled:
            encoder_text = "Encoder: simulated"
        cv2.putText(
            combined,
            encoder_text,
            (hud_text_position[0], hud_text_position[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            hud_text_scale,
            hud_text_color,
            hud_text_thickness,
            cv2.LINE_AA
        )

        motor_text = f"Motor duty: {current_motor_duty_pct:.1f}%"
        cv2.putText(
            combined,
            motor_text,
            (hud_text_position[0], hud_text_position[1] + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            hud_text_scale,
            hud_text_color,
            hud_text_thickness,
            cv2.LINE_AA
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
            cv2.putText(
                combined,
                "Sim encoder: arrow keys to jog, 's' to exit",
                (20, h_combined - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        message_to_show = calibration_status_text
        if not message_to_show and encoder_span() is None:
            message_to_show = "Press 'c' to calibrate steering range"
        if message_to_show:
            cv2.putText(
                combined,
                message_to_show,
                (20, h_combined - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Present the final annotated view
        cv2.imshow("Depth: Camera 0 | Camera 1", combined)
        key = cv2.waitKey(1) & 0xFF
    else:
        key = -1

    # Handle keys — currently only ESC to exit
    if key == 27:
        break

    if key == ord('s'):
        sim_encoder_enabled = not sim_encoder_enabled
        if sim_encoder_enabled:
            if latest_encoder_norm is not None:
                sim_encoder_norm = latest_encoder_norm
            else:
                sim_encoder_norm = 0.5
            calibration_status_text = "Simulated encoder enabled"
        else:
            calibration_status_text = "Simulated encoder disabled"

    if sim_encoder_enabled and key in (81, 83):
        delta = -simulated_step_norm if key == 81 else simulated_step_norm
        sim_encoder_norm = clamp(sim_encoder_norm + delta, 0.0, 1.0)

    if key == ord('c'):
        start_calibration()

    if calibration_active and key == 32:  # space bar
        capture_calibration_point()

# === Cleanup ===
running = False
cap0.release()
cap1.release()
cv2.destroyAllWindows()
shutdown_motor_outputs()
shutdown_encoder()
shutdown_motor_activity_gpio()
shutdown_motor_direction_gpio()





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

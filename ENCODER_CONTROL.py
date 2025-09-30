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
import Jetson.GPIO as GPIO
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

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
blue_x_smoothness = 0.5

# HUD text overlays on the combined frame
hud_text_position = (10, 30)
hud_text_color = (255, 255, 255)
hud_text_scale = 0.54
hud_text_thickness = 1

# Motor control configuration
motor_pin_left = 29        # physical pin for left turn enable
motor_pin_right = 31       # physical pin for right turn enable
motor_brake_pct = 40.0     # duty cycle applied for active braking (0 disables)
motor_predictive_slowdown_px = 130.0  # distance (in px) over which speed ramps down

# PWM configuration (pin 32 routed via pwmchip sysfs)
pwm_chip_path = "/sys/class/pwm/pwmchip3"
pwm_channel = "pwm0"
pwm_frequency_hz = 8000

# Encoder + steering control configuration
encoder_i2c_bus = 7
encoder_i2c_address = 0x36
encoder_deadband_px = 5          # stop when within this many screen pixels of target
motor_speed_pct = 25.0            # Max motor speed
motor_min_duty_pct = 18.0         # Minimum duty cycle required to overcome stiction
motor_accel_pct_per_s = 80.0     # Larger number = slower acceleration
reverse_switch_threshold_pct = 20.0  # switch direction min duty
calibration_file = Path("steering_calibration.json")
simulated_step_norm = 0.04        # arrow-key increment when in simulated encoder mode


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

def _pwm_wr(path, val):
    with open(path, "w") as f:
        f.write(str(val))


def _pwm_ns_period(freq_hz):
    return int(round(1e9 / float(freq_hz)))


def _pwm_ns_duty(period_ns, pct):
    pct = max(0.0, min(100.0, float(pct)))
    return int(period_ns * (pct / 100.0))


pwm_channel_path = f"{pwm_chip_path}/{pwm_channel}"


def initialise_motor_outputs():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(motor_pin_left, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(motor_pin_right, GPIO.OUT, initial=GPIO.LOW)

    if not os.path.isdir(pwm_channel_path):
        if not os.path.isdir(pwm_chip_path):
            raise FileNotFoundError(f"{pwm_chip_path} does not exist. Check PWM configuration.")
        _pwm_wr(f"{pwm_chip_path}/export", "0")
        for _ in range(200):
            if os.path.isdir(pwm_channel_path):
                break
            time.sleep(0.01)
        else:
            raise TimeoutError("PWM channel did not appear after export")

    try:
        _pwm_wr(f"{pwm_channel_path}/enable", "0")
    except OSError:
        pass

    period_ns = _pwm_ns_period(pwm_frequency_hz)
    _pwm_wr(f"{pwm_channel_path}/period", period_ns)
    _pwm_wr(f"{pwm_channel_path}/duty_cycle", 0)
    _pwm_wr(f"{pwm_channel_path}/enable", "1")
    return period_ns


def _apply_motor_output(direction, duty_pct, period_ns, brake_duty_pct=0.0):
    duty_pct = max(0.0, min(100.0, duty_pct))
    brake_duty_pct = max(0.0, min(100.0, brake_duty_pct))

    if brake_duty_pct > 0.0:
        GPIO.output(motor_pin_right, GPIO.HIGH)
        GPIO.output(motor_pin_left, GPIO.HIGH)
        duty_ns = _pwm_ns_duty(period_ns, brake_duty_pct)
    else:
        if direction > 0:
            GPIO.output(motor_pin_right, GPIO.HIGH)
            GPIO.output(motor_pin_left, GPIO.LOW)
        elif direction < 0:
            GPIO.output(motor_pin_right, GPIO.LOW)
            GPIO.output(motor_pin_left, GPIO.HIGH)
        else:
            GPIO.output(motor_pin_right, GPIO.LOW)
            GPIO.output(motor_pin_left, GPIO.LOW)
        duty_ns = _pwm_ns_duty(period_ns, duty_pct if direction != 0 else 0.0)
    _pwm_wr(f"{pwm_channel_path}/duty_cycle", duty_ns)


def shutdown_motor_outputs():
    try:
        _pwm_wr(f"{pwm_channel_path}/duty_cycle", 0)
        _pwm_wr(f"{pwm_channel_path}/enable", "0")
    except OSError:
        pass
    finally:
        GPIO.cleanup()


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


def update_motor_control(target_px, actual_px, dt, period_ns):
    global current_motor_direction, current_motor_duty_pct

    if calibration_active:
        current_motor_direction = 0
        current_motor_duty_pct = 0.0
        _apply_motor_output(0, 0.0, period_ns)
        return

    remaining_px = None
    if actual_px is None or target_px is None:
        required_direction = 0
    else:
        error_px = target_px - actual_px
        remaining_px = abs(error_px)
        if abs(error_px) <= encoder_deadband_px:
            required_direction = 0
        else:
            required_direction = 1 if error_px > 0 else -1

    def predictive_target(base_pct):
        if (
            remaining_px is None
            or base_pct <= 0.0
            or motor_predictive_slowdown_px <= 0.0
        ):
            return base_pct
        ratio = remaining_px / motor_predictive_slowdown_px
        ratio = clamp(ratio, 0.0, 1.0)
        return base_pct * ratio

    base_target_pct = 0.0
    new_direction = current_motor_direction
    kick_pct = min(reverse_switch_threshold_pct, motor_speed_pct)

    if required_direction == 0:
        if current_motor_duty_pct <= kick_pct:
            new_direction = 0
    else:
        if current_motor_direction == 0:
            new_direction = required_direction
            base_target_pct = motor_speed_pct
        elif current_motor_direction == required_direction:
            base_target_pct = motor_speed_pct
        else:
            if current_motor_duty_pct <= kick_pct:
                new_direction = required_direction
                base_target_pct = motor_speed_pct

    target_speed_pct = predictive_target(base_target_pct)

    if base_target_pct > 0.0:
        target_speed_pct = max(target_speed_pct, min(motor_min_duty_pct, motor_speed_pct))

    if base_target_pct > 0.0 and target_speed_pct >= kick_pct and current_motor_duty_pct < kick_pct:
        current_motor_duty_pct = kick_pct

    max_delta = motor_accel_pct_per_s * dt
    if target_speed_pct > current_motor_duty_pct:
        current_motor_duty_pct = min(target_speed_pct, current_motor_duty_pct + max_delta)
    else:
        current_motor_duty_pct = max(target_speed_pct, current_motor_duty_pct - max_delta)

    braking = False
    if motor_brake_pct > 0.0:
        stopping = required_direction == 0 and current_motor_direction != 0 and current_motor_duty_pct > 0.0
        reversing = (
            required_direction != 0
            and current_motor_direction != 0
            and current_motor_direction != required_direction
            and current_motor_duty_pct > 0.0
        )
        braking = stopping or reversing

    if braking:
        new_direction = 0

    current_motor_direction = new_direction
    if braking:
        _apply_motor_output(0, 0.0, period_ns, brake_duty_pct=motor_brake_pct)
    else:
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
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
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
        cv2.line(
            combined,
            (encoder_px, 0),
            (encoder_px, h_combined),
            (0, 255, 255),
            1,
        )

    update_motor_control(draw_x, encoder_px, dt, motor_period_ns)





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

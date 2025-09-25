"""Stereo depth estimation with PWM steering control.

This script combines the perception pipeline from ``detectAvoid_V2.py`` with the
sysfs PWM setup used in ``pwmControl.py``. Steering decisions follow the logic
from ``PWM_VISION.py`` while retaining an optional GUI toggle so the processing
can run headless when desired.
"""

import os
import time
import queue
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


# ---------------------------------------------------------------------------
# Configuration — tweak these values to adjust behaviour without diving into
# the rest of the code. Where possible, related values are grouped together and
# documented so their impact is clear.
# ---------------------------------------------------------------------------

# Toggle visualization (heatmaps, overlays, window). False = headless speed run.
SHOW_GUI = True

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

# --- PWM configuration ----------------------------------------------------
PWMCHIP = "/sys/class/pwm/pwmchip3"  # adjust to the one present on your board
PWM_CHANNEL = "pwm0"
PWM_PERIOD_NS = 20_000_000            # 50 Hz (nanoseconds)
PWM_DUTY_NEUTRAL_NS = 1_500_000       # 1.5 ms neutral
PWM_DIR = f"{PWMCHIP}/{PWM_CHANNEL}"
PWM_ENABLE = f"{PWM_DIR}/enable"
PWM_PERIOD = f"{PWM_DIR}/period"
PWM_DUTY = f"{PWM_DIR}/duty_cycle"

PWM_MIN_NS = 1_000_000
PWM_MAX_NS = 2_000_000
PWM_SPAN_PX = 300                     # pixels from centre to reach min/max


# ---------------------------------------------------------------------------
# Runtime state (initialised once, then updated as frames stream in)
# ---------------------------------------------------------------------------
blue_x = None              # persistent, smoothed x-position of the blue circle
steer_control_x = None     # integer pixel location displayed as HUD text

# PWM state
pwm_initialized = False
pwm_error_reported = False
last_duty_ns = None


# ---------------------------------------------------------------------------
# Helper utilities and pipeline initialisation
# ---------------------------------------------------------------------------

def clamp(val, minn, maxn):
    """Clamp values between two bounds (used to keep overlays on-screen)."""
    return max(min(val, maxn), minn)


def pwm_write(path: str, value):
    with open(path, "w") as handle:
        handle.write(str(value))


def init_pwm():
    """Initialise the sysfs PWM channel if available."""
    global pwm_initialized, pwm_error_reported, last_duty_ns

    if pwm_initialized or pwm_error_reported:
        return

    if not os.path.isdir(PWM_DIR):
        if not os.path.isdir(PWMCHIP):
            print(f"[PWM] {PWMCHIP} does not exist. Check which pwmchipN is present.")
            pwm_error_reported = True
            return
        try:
            pwm_write(f"{PWMCHIP}/export", "0")
        except OSError as exc:
            print(f"[PWM] Failed to export channel: {exc}")
            pwm_error_reported = True
            return
        for _ in range(100):
            if os.path.isdir(PWM_DIR):
                break
            time.sleep(0.01)
        else:
            print("[PWM] pwm0 did not appear after export.")
            pwm_error_reported = True
            return

    try:
        pwm_write(PWM_ENABLE, "0")
    except OSError:
        pass

    try:
        pwm_write(PWM_PERIOD, PWM_PERIOD_NS)
        pwm_write(PWM_DUTY, PWM_DUTY_NEUTRAL_NS)
        pwm_write(PWM_ENABLE, "1")
    except OSError as exc:
        print(f"[PWM] Failed to configure channel: {exc}")
        pwm_error_reported = True
        return

    pwm_initialized = True
    last_duty_ns = PWM_DUTY_NEUTRAL_NS
    print("PWM initialized: 50 Hz @ 1.5 ms neutral.")


def set_pwm_duty(duty_ns: int):
    """Write a new duty cycle with a small deadband to avoid redundant writes."""
    global last_duty_ns
    if not pwm_initialized:
        return
    if (last_duty_ns is None) or (abs(duty_ns - last_duty_ns) >= 500):
        try:
            pwm_write(PWM_DUTY, duty_ns)
            last_duty_ns = duty_ns
        except OSError as exc:
            print(f"[PWM] Duty write failed: {exc}")


def disable_pwm():
    if not pwm_initialized:
        return
    try:
        pwm_write(PWM_ENABLE, "0")
        print("PWM disabled.")
    except OSError as exc:
        print(f"[PWM] Cleanup error: {exc}")


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
            if q.full():
                q.get_nowait()
            q.put(frame)


Thread(target=grab, args=(cap0, frame_q0), daemon=True).start()
Thread(target=grab, args=(cap1, frame_q1), daemon=True).start()


def infer():
    """Fetch paired frames, run depth inference, and push the results downstream."""
    while running:
        try:
            f0 = frame_q0.get(timeout=0.01)
            f1 = frame_q1.get(timeout=0.01)
        except queue.Empty:
            continue
        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in (f0, f1)]
        with torch.amp.autocast(device_type="cuda"), torch.no_grad():
            outs = depth_pipe(imgs)
        d0, d1 = [np.array(o["depth"]) for o in outs]
        if result_q.full():
            result_q.get_nowait()
        result_q.put((f0, d0, f1, d1))


Thread(target=infer, daemon=True).start()


# ---------------------------------------------------------------------------
# Main processing loop — consume depth results, plan, PWM, and (optional) GUI
# ---------------------------------------------------------------------------
while True:
    try:
        frame0, depth0, frame1, depth1 = result_q.get(timeout=0.01)
    except queue.Empty:
        continue

    # Build a sparse point cloud by sampling each depth map on a coarse grid
    h0, w0 = depth0.shape
    h1, w1 = depth1.shape
    points = []
    for cam_idx, (depth, x_offset, w, h) in enumerate([
        (depth0, 0, w0, h0), (depth1, w0, w1, h1)
    ]):
        for c in range(cols):
            for r in range(rows):
                px = int((c + 0.5) * w / cols) + x_offset
                py = int((r + 0.5) * h / rows)
                z = float(depth[int(py), int(px - x_offset)])
                points.append((px, py, z, cam_idx))
    cloud = np.array(points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("cam", "i4")])

    # Determine obstacle candidates by comparing each sample to the scene average
    zs = cloud["z"]
    mean_z, std_z = zs.mean(), zs.std()
    thresh = max(depth_diff_threshold, std_multiplier * std_z)
    mask = (mean_z - zs) > thresh

    # Combined-frame metrics for steering logic
    w_combined = w0 + w1
    h_combined = frame0.shape[0]

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

    for is_obst, pt in zip(mask, cloud):
        if not is_obst:
            continue
        px, py, cam = int(pt["x"]), int(pt["y"]), pt["cam"]
        frame_h = frame0.shape[0] if cam == 0 else frame1.shape[0]
        if py < top_cutoff_pixels or py > (frame_h - bottom_cutoff_pixels):
            continue
        red_xs_all.append(px)
        if zone_left <= px <= zone_right:
            red_xs_in_zone.append(px)

    red_xs_all = sorted(set(red_xs_all))
    red_xs_in_zone = sorted(set(red_xs_in_zone))

    blockers_all = [0] + red_xs_all + [w_combined]

    def widest_gap_center(blockers, preferred_x):
        """Choose the center of the widest gap. Tie-break: closest to preferred_x."""
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

    if len(red_xs_in_zone) == 0:
        gap_cx = center_x
    else:
        gap_cx = widest_gap_center(blockers_all, preferred_x=center_x)

    if blue_x is None:
        blue_x = float(gap_cx)
    else:
        blue_x = blue_x * blue_x_smoothness + gap_cx * (1 - blue_x_smoothness)

    draw_x = int(round(blue_x))
    steer_control_x = draw_x

    if not pwm_initialized and not pwm_error_reported:
        init_pwm()

    if pwm_initialized:
        delta_px = steer_control_x - center_x
        duty = int(np.clip(
            PWM_DUTY_NEUTRAL_NS + (delta_px / PWM_SPAN_PX) * (PWM_MAX_NS - PWM_DUTY_NEUTRAL_NS),
            PWM_MIN_NS,
            PWM_MAX_NS,
        ))
        set_pwm_duty(duty)

    if SHOW_GUI:
        norm0 = cv2.normalize(depth0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cmap0 = cv2.applyColorMap(cv2.resize(norm0, (frame0.shape[1], frame0.shape[0])), cv2.COLORMAP_MAGMA)
        norm1 = cv2.normalize(depth1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cmap1 = cv2.applyColorMap(cv2.resize(norm1, (frame1.shape[1], frame1.shape[0])), cv2.COLORMAP_MAGMA)

        for cmap, h, w in [
            (cmap0, frame0.shape[0], frame0.shape[1]),
            (cmap1, frame1.shape[0], frame1.shape[1]),
        ]:
            top_y = top_cutoff_pixels
            bottom_y = h - bottom_cutoff_pixels
            cv2.line(cmap, (0, top_y), (w, top_y), cutoff_line_color, cutoff_line_thickness)
            cv2.line(cmap, (0, bottom_y), (w, bottom_y), cutoff_line_color, cutoff_line_thickness)

        combined = np.hstack((cmap0, cmap1))

        blue_pos = (draw_x, line_y)
        cv2.circle(combined, blue_pos, blue_circle_radius_px, blue_circle_color, blue_circle_thickness)

        cv2.putText(
            combined,
            f"Steer Control: {steer_control_x}",
            hud_text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            hud_text_scale,
            hud_text_color,
            hud_text_thickness,
            cv2.LINE_AA,
        )

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

        cv2.imshow("Depth: Camera 0 | Camera 1", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

# === Cleanup ===
running = False
cap0.release()
cap1.release()
if SHOW_GUI:
    cv2.destroyAllWindows()

disable_pwm()

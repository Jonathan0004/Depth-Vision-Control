"""Stereo depth estimation with obstacle detection and steering guidance.

The script streams from two cameras, infers depth using the
``depth-anything`` model, highlights obstacles inside configurable cutoff
bands, and visualises a blue steering cue that points toward the widest gap.
All tunable parameters live together for quick iteration.

Now includes PWM steering output (Jetson sysfs pwmchip3/pwm0) updated from the
computed steering x-position.
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

try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
except Exception as exc:  # pragma: no cover - Jetson specific hardware dependency
    GPIO = None
    GPIO_AVAILABLE = False
    print(f"Jetson GPIO unavailable ({exc}). Running in no-GPIO mode.")


# ---------------------------------------------------------------------------
# Configuration — tweak these values to adjust behaviour without diving into
# the rest of the code. Where possible, related values are grouped together and
# documented so their impact is clear.
# ---------------------------------------------------------------------------

# Grid resolution for sampling depth points across each frame (higher = denser)
rows, cols = 25, 50

# Obstacle detection thresholds
depth_diff_threshold = 8      # minimum mean-depth difference to flag a point
std_multiplier = 0.3          # scales standard deviation term for adaptive thresholding

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
blue_x_smoothness = 0.7

# HUD text overlays on the combined frame
hud_text_position = (10, 30)
hud_text_color = (255, 255, 255)
hud_text_scale = 0.54
hud_text_thickness = 1

# ------------------------ PWM Output Configuration --------------------------
# Jetson sysfs PWM path & parameters (adjust PWMCHIP/pwm index for your board)
PWMCHIP = "/sys/class/pwm/pwmchip3"
PWM_CHANNEL_INDEX = 0           # -> pwm0
PWM_PERIOD_NS = 200_000         # 5 kHz period (200 µs)

# TB6612FNG direction control pins (Jetson Nano BOARD numbering):
#  - MOTOR_IN1_PIN goes HIGH for forward rotation (connect to TB6612 AIN1)
#  - MOTOR_IN2_PIN goes HIGH for reverse rotation (connect to TB6612 AIN2)
MOTOR_IN1_PIN = 33
MOTOR_IN2_PIN = 35
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Runtime state (initialised once, then updated as frames stream in)
# ---------------------------------------------------------------------------
blue_x = None              # persistent, smoothed x-position of the blue circle
steer_control_x = None     # integer pixel location displayed as HUD text

# PWM runtime flags/state
pwm_initialized = False
pwm_error_reported = False
gpio_initialized = False
gpio_error_reported = False
last_duty_ns = None
last_direction_state = None
pwm_driver = None


# ---------------------------------------------------------------------------
# Helper utilities and pipeline initialisation
# ---------------------------------------------------------------------------

# Utility: clamp values between two bounds (used to keep overlays on-screen)
def clamp(val, minn, maxn):
    return max(min(val, maxn), minn)


# Minimal sysfs PWM driver keeping the duty file handle open for fast updates
class PWMDriver:
    def __init__(self, chip_path: str, channel_index: int):
        self.chip_path = chip_path
        self.channel_index = int(channel_index)
        self.pwm_path = os.path.join(self.chip_path, f"pwm{self.channel_index}")
        self._duty_file = None

    @staticmethod
    def _wr(path, value):
        with open(path, "w") as f:
            f.write(str(value))

    def init(self, period_ns: int, duty_ns: int):
        # Sanity check chip exists
        if not os.path.isdir(self.chip_path):
            raise FileNotFoundError(f"{self.chip_path} does not exist. Check which pwmchipN is present.")

        # Export channel if needed
        if not os.path.isdir(self.pwm_path):
            PWMDriver._wr(os.path.join(self.chip_path, "export"), str(self.channel_index))
            # Wait for sysfs to create pwmN directory
            for _ in range(200):
                if os.path.isdir(self.pwm_path):
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError(f"pwm{self.channel_index} did not appear after export")

        # Disable before configuration (some drivers require this)
        try:
            PWMDriver._wr(os.path.join(self.pwm_path, "enable"), "0")
        except OSError:
            pass  # okay during initialisation races

        # Configure while disabled: period -> duty -> enable
        PWMDriver._wr(os.path.join(self.pwm_path, "period"), period_ns)
        PWMDriver._wr(os.path.join(self.pwm_path, "duty_cycle"), duty_ns)
        PWMDriver._wr(os.path.join(self.pwm_path, "enable"), "1")

        # Keep the duty file open for low-latency updates
        self._duty_file = open(os.path.join(self.pwm_path, "duty_cycle"), "w")

    def set_duty(self, duty_ns: int):
        if self._duty_file is None:
            raise RuntimeError("PWM not initialised. Call init() first.")
        # Write with rewind to avoid leftover digits when writing fewer chars
        self._duty_file.seek(0)
        self._duty_file.write(str(int(duty_ns)))
        self._duty_file.flush()

    def close(self):
        try:
            if self._duty_file:
                self._duty_file.close()
        finally:
            try:
                PWMDriver._wr(os.path.join(self.pwm_path, "enable"), "0")
            except Exception:
                pass


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

def infer():
    """Fetch paired frames, run depth inference, and push the results downstream."""
    while running:
        try:
            f0 = frame_q0.get(timeout=0.01)
            f1 = frame_q1.get(timeout=0.01)
        except queue.Empty:
            continue
        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in (f0, f1)]
        with torch.amp.autocast(device_type='cuda'), torch.no_grad():
            outs = depth_pipe(imgs)
        d0, d1 = [np.array(o['depth']) for o in outs]
        if result_q.full(): result_q.get_nowait()
        result_q.put((f0, d0, f1, d1))

Thread(target=infer, daemon=True).start()

# ---------------------------------------------------------------------------
# Main processing loop — consume depth results, annotate, and display
# ---------------------------------------------------------------------------
try:
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
                    z  = float(depth[int(py), int(px - x_offset)])
                    points.append((px, py, z, cam_idx))
        cloud = np.array(points, dtype=[('x','f4'),('y','f4'),('z','f4'),('cam','i4')])

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

        # Draw obstacle points only within cutoffs
        for is_obst, pt in zip(mask, cloud):
            if not is_obst:
                continue
            px, py, cam = int(pt['x']), int(pt['y']), pt['cam']
            # check cutoffs
            if py < top_cutoff_pixels or py > ( (frame0.shape[0] if cam==0 else frame1.shape[0]) - bottom_cutoff_pixels ):
                continue
            # target = cmap0 if cam == 0 else cmap1
            # offset_x = 0 if cam == 0 else w0
            # Optional: draw red dots
            # cv2.circle(target, (px - offset_x, py), obstacle_dot_radius_px, obstacle_dot_color, -1)

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

        for is_obst, pt in zip(mask, cloud):
            if not is_obst:
                continue
            px, py, cam = int(pt['x']), int(pt['y']), pt['cam']
            frame_h = frame0.shape[0] if cam == 0 else frame1.shape[0]
            if py < top_cutoff_pixels or py > (frame_h - bottom_cutoff_pixels):
                continue
            red_xs_all.append(px)
            if zone_left <= px <= zone_right:
                red_xs_in_zone.append(px)

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

        # ========================== PWM SECTION: START ==========================
        # Initialize PWM ONCE, immediately after 'steer_control_x = draw_x'.
        # Keeps duty file handle open for FAST future updates tied to 'steer_control_x'.
        if not pwm_initialized and not pwm_error_reported:
            try:
                pwm_driver = PWMDriver(PWMCHIP, PWM_CHANNEL_INDEX)
                pwm_driver.init(PWM_PERIOD_NS, 0)
                pwm_initialized = True
                print("PWM initialized: 5 kHz speed control on pwmchip3/pwm0.")

                if GPIO_AVAILABLE and not gpio_initialized:
                    GPIO.setmode(GPIO.BOARD)
                    GPIO.setup(MOTOR_IN1_PIN, GPIO.OUT, initial=GPIO.LOW)
                    GPIO.setup(MOTOR_IN2_PIN, GPIO.OUT, initial=GPIO.LOW)
                    gpio_initialized = True
                    last_direction_state = "stop"
                    print(
                        "GPIO direction pins ready: pin 33 -> AIN1 (forward HIGH), "
                        "pin 35 -> AIN2 (reverse HIGH)."
                    )
            except Exception as e:
                pwm_error_reported = True
                print(f"[PWM] Initialization error: {e}")

        if pwm_initialized:
            span_px = 300  # pixels from center to reach full speed (tune for your geometry)

            delta_px = steer_control_x - center_x
            normalized = float(np.clip(delta_px / span_px, -1.0, 1.0))

            max_duty = PWM_PERIOD_NS - 1

            if abs(normalized) < 0.02:
                duty = 0
                desired_direction = "stop"
            elif normalized > 0:
                duty = int(min(abs(normalized), 0.999) * max_duty)
                desired_direction = "forward"
            else:
                duty = int(min(abs(normalized), 0.999) * max_duty)
                desired_direction = "reverse"

            if gpio_initialized and not gpio_error_reported:
                try:
                    if desired_direction == "forward" and last_direction_state != "forward":
                        GPIO.output(MOTOR_IN1_PIN, GPIO.HIGH)
                        GPIO.output(MOTOR_IN2_PIN, GPIO.LOW)
                        last_direction_state = "forward"
                    elif desired_direction == "reverse" and last_direction_state != "reverse":
                        GPIO.output(MOTOR_IN1_PIN, GPIO.LOW)
                        GPIO.output(MOTOR_IN2_PIN, GPIO.HIGH)
                        last_direction_state = "reverse"
                    elif desired_direction == "stop" and last_direction_state != "stop":
                        GPIO.output(MOTOR_IN1_PIN, GPIO.LOW)
                        GPIO.output(MOTOR_IN2_PIN, GPIO.LOW)
                        last_direction_state = "stop"
                except Exception as e:
                    gpio_error_reported = True
                    print(f"[GPIO] Direction control error: {e}")

            if (last_duty_ns is None) or (abs(duty - last_duty_ns) >= 500):
                pwm_driver.set_duty(duty)
                last_duty_ns = duty
        # =========================== PWM SECTION: END ===========================

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

        # Present the final annotated view
        cv2.imshow("Depth: Camera 0 | Camera 1", combined)
        key = cv2.waitKey(1) & 0xFF

        # Handle keys — currently only ESC to exit
        if key == 27:
            break

finally:
    # === Cleanup ===
    running = False
    try:
        cap0.release()
    except Exception:
        pass
    try:
        cap1.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    # Turn off PWM if it was enabled
    try:
        if pwm_driver is not None:
            pwm_driver.close()
    except Exception as e:
        print(f"[PWM] Cleanup error: {e}")
    if gpio_initialized and GPIO_AVAILABLE:
        try:
            GPIO.output(MOTOR_IN1_PIN, GPIO.LOW)
            GPIO.output(MOTOR_IN2_PIN, GPIO.LOW)
        except Exception:
            pass
        finally:
            try:
                GPIO.cleanup([MOTOR_IN1_PIN, MOTOR_IN2_PIN])
            except Exception:
                pass

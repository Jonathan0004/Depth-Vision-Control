"""
Stereo depth estimation with obstacle detection and steering guidance.

UPDATED FOR TB6612FNG:
- One DC motor on channel A (A01/A02)
- 5 kHz PWM on PWMA (Jetson sysfs: pwmchip3/pwm0), duty-cycle only for speed
- Direction via AIN1/AIN2; STBY held HIGH
- Center steering => 0% duty (stop). Farther left/right => higher duty.
- Right/Left chooses direction (configurable with RIGHT_IS_FORWARD).

Notes:
- We DO NOT explicitly use "coast" or "brake" states—just duty = 0 to stop.
- When direction changes, we drop duty to 0 first, then flip IN1/IN2.
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

import Jetson.GPIO as GPIO  # <-- for AIN1/AIN2/STBY control


# ---------------------------------------------------------------------------
# Configuration — visual processing (unchanged)
# ---------------------------------------------------------------------------

rows, cols = 25, 50
depth_diff_threshold = 8
std_multiplier = 0.3

top_cutoff_pixels = 10
bottom_cutoff_pixels = 54
cutoff_line_color = (0, 255, 0)
cutoff_line_thickness = 1

obstacle_dot_radius_px = 5
obstacle_dot_color = (0, 0, 255)

blue_circle_radius_px = 12
blue_circle_color = (255, 0, 0)
blue_circle_thickness = 3

pull_zone_line_color = (255, 0, 0)
pull_zone_line_thickness = 1
pull_influence_radius_px = 120
pull_zone_center_offset_px = 0

blue_x_smoothness = 0.7
hud_text_position = (10, 30)
hud_text_color = (255, 255, 255)
hud_text_scale = 0.54
hud_text_thickness = 1


# ---------------------------------------------------------------------------
# TB6612FNG + Jetson PWM/GPIO configuration
# ---------------------------------------------------------------------------

# --- PWM (PWMA) on Jetson sysfs ---
PWMCHIP = "/sys/class/pwm/pwmchip3"  # same path you were using
PWM_CHANNEL_INDEX = 0                # -> pwm0
PWM_FREQ_HZ = 5_000                  # 5 kHz fixed
PWM_PERIOD_NS = int(1e9 // PWM_FREQ_HZ)  # 200_000 ns at 5 kHz

# Start at 0% duty so the motor is stopped until steering tells it to move
PWM_START_DUTY_NS = 0

# --- Direction pins (edit these to match your wiring) ---
# Using BOARD numbering (physical header pins):
GPIO.setmode(GPIO.BOARD)
STBY_PIN = 29   # STBY  -> HIGH = enabled
AIN1_PIN = 33   # AIN1  -> HIGH (with AIN2 LOW) = FORWARD
AIN2_PIN = 31   # AIN2  -> HIGH (with AIN1 LOW) = REVERSE

# Print friendly mapping at startup
PIN_LABELS = (
    f"PIN MAP (BOARD numbering): STBY={STBY_PIN}, AIN1={AIN1_PIN}, AIN2={AIN2_PIN}, "
    f"PWMA=pwmchip3/pwm{PWM_CHANNEL_INDEX} (5 kHz)"
)

# Which steering side means "forward"?
# True: Right of center => FORWARD, Left => REVERSE
# False: Right => REVERSE, Left => FORWARD
RIGHT_IS_FORWARD = True

# How close to center counts as "stop" (to avoid jitter)
CENTER_DEADBAND_PX = 5

# Pixel distance from center that maps to 100% duty (tune as you like)
SPAN_PX = pull_influence_radius_px  # use your pull zone half-width


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------
blue_x = None
steer_control_x = None

pwm_initialized = False
pwm_error_reported = False
last_duty_ns = 0
pwm_driver = None

current_dir = None  # "FORWARD" / "REVERSE" / None
last_sign = 0       # -1, 0, +1


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clamp(val, minn, maxn):
    return max(min(val, maxn), minn)


class PWMDriver:
    """
    Minimal sysfs PWM driver for speed. Keeps duty file open for low-latency updates.
    """
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
        if not os.path.isdir(self.chip_path):
            raise FileNotFoundError(f"{self.chip_path} does not exist. Check which pwmchipN is present.")

        if not os.path.isdir(self.pwm_path):
            PWMDriver._wr(os.path.join(self.chip_path, "export"), str(self.channel_index))
            for _ in range(200):
                if os.path.isdir(self.pwm_path):
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError(f"pwm{self.channel_index} did not appear after export")

        try:
            PWMDriver._wr(os.path.join(self.pwm_path, "enable"), "0")
        except OSError:
            pass

        PWMDriver._wr(os.path.join(self.pwm_path, "period"), period_ns)
        # Clamp initial duty to [0, period]
        duty_ns = max(0, min(int(duty_ns), int(period_ns)))
        PWMDriver._wr(os.path.join(self.pwm_path, "duty_cycle"), duty_ns)
        PWMDriver._wr(os.path.join(self.pwm_path, "enable"), "1")
        self._duty_file = open(os.path.join(self.pwm_path, "duty_cycle"), "w")

    def set_duty(self, duty_ns: int):
        if self._duty_file is None:
            raise RuntimeError("PWM not initialised. Call init() first.")
        duty_ns = max(0, min(int(duty_ns), int(PWM_PERIOD_NS)))
        self._duty_file.seek(0)
        self._duty_file.write(str(duty_ns))
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


def set_direction(new_dir: str):
    """
    Set AIN1/AIN2 for desired direction.
    FORWARD  -> STBY=HIGH, AIN1=HIGH, AIN2=LOW
    REVERSE  -> STBY=HIGH, AIN1=LOW,  AIN2=HIGH
    """
    global current_dir
    if new_dir == current_dir:
        return
    if new_dir not in ("FORWARD", "REVERSE"):
        return

    # STBY must be HIGH to operate
    GPIO.output(STBY_PIN, GPIO.HIGH)

    if new_dir == "FORWARD":
        GPIO.output(AIN1_PIN, GPIO.HIGH)
        GPIO.output(AIN2_PIN, GPIO.LOW)
        print("[DIR] FORWARD  — STBY=HIGH, AIN1=HIGH, AIN2=LOW")
    else:
        GPIO.output(AIN1_PIN, GPIO.LOW)
        GPIO.output(AIN2_PIN, GPIO.HIGH)
        print("[DIR] REVERSE  — STBY=HIGH, AIN1=LOW,  AIN2=HIGH")

    current_dir = new_dir


# ---------------------------------------------------------------------------
# Depth model + cameras (unchanged)
# ---------------------------------------------------------------------------

depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    device=0
)

def make_gst(sensor_id):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM),width=300,height=150,framerate=60/1 ! "
        "nvvidconv flip-method=2 ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink sync=false max-buffers=1 drop=true"
    )

cap0 = cv2.VideoCapture(make_gst(0), cv2.CAP_GSTREAMER)
cap1 = cv2.VideoCapture(make_gst(1), cv2.CAP_GSTREAMER)
if not cap0.isOpened() or not cap1.isOpened():
    raise RuntimeError("Failed to open one or both cameras.")
cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_q0, frame_q1, result_q = queue.Queue(1), queue.Queue(1), queue.Queue(1)
running = True

def grab(cam, q):
    while running:
        ret, frame = cam.read()
        if ret:
            if q.full(): q.get_nowait()
            q.put(frame)

Thread(target=grab, args=(cap0, frame_q0), daemon=True).start()
Thread(target=grab, args=(cap1, frame_q1), daemon=True).start()

def infer():
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
# GPIO init (direction + STBY)
# ---------------------------------------------------------------------------
GPIO.setup([STBY_PIN, AIN1_PIN, AIN2_PIN], GPIO.OUT, initial=GPIO.LOW)
GPIO.output(STBY_PIN, GPIO.HIGH)  # enable TB6612FNG
print(PIN_LABELS)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
try:
    while True:
        try:
            frame0, depth0, frame1, depth1 = result_q.get(timeout=0.01)
        except queue.Empty:
            continue

        # Build sparse point cloud
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

        # Obstacle candidates
        zs = cloud['z']
        mean_z, std_z = zs.mean(), zs.std()
        thresh = max(depth_diff_threshold, std_multiplier * std_z)
        mask = (mean_z - zs) > thresh

        # Visualisation
        norm0 = cv2.normalize(depth0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cmap0 = cv2.applyColorMap(cv2.resize(norm0, (frame0.shape[1], frame0.shape[0])), cv2.COLORMAP_MAGMA)
        norm1 = cv2.normalize(depth1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cmap1 = cv2.applyColorMap(cv2.resize(norm1, (frame1.shape[1], frame1.shape[0])), cv2.COLORMAP_MAGMA)

        for cmap, h, w in [(cmap0, frame0.shape[0], frame0.shape[1]), (cmap1, frame1.shape[0], frame1.shape[1])]:
            top_y = top_cutoff_pixels
            bottom_y = h - bottom_cutoff_pixels
            cv2.line(cmap, (0, top_y), (w, top_y), cutoff_line_color, cutoff_line_thickness)
            cv2.line(cmap, (0, bottom_y), (w, bottom_y), cutoff_line_color, cutoff_line_thickness)

        for is_obst, pt in zip(mask, cloud):
            if not is_obst:
                continue
            px, py, cam = int(pt['x']), int(pt['y']), pt['cam']
            if py < top_cutoff_pixels or py > ((frame0.shape[0] if cam==0 else frame1.shape[0]) - bottom_cutoff_pixels):
                continue
            # optional obstacle dots omitted for clarity

        combined = np.hstack((cmap0, cmap1))

        # Route planner
        h_combined, w_combined = frame0.shape[0], frame0.shape[1] + frame1.shape[1]
        line_y = (top_cutoff_pixels + (h_combined - bottom_cutoff_pixels)) // 2
        center_x = (w_combined // 2) + pull_zone_center_offset_px
        zone_left = clamp(center_x - pull_influence_radius_px, 0, w_combined)
        zone_right = clamp(center_x + pull_influence_radius_px, 0, w_combined)

        red_xs_all, red_xs_in_zone = [], []
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
        red_xs_all = sorted(set(red_xs_all))
        red_xs_in_zone = sorted(set(red_xs_in_zone))

        blockers_all = [0] + red_xs_all + [w_combined]

        def widest_gap_center(blockers, preferred_x):
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

        gap_cx = center_x if len(red_xs_in_zone) == 0 else widest_gap_center(blockers_all, preferred_x=center_x)

        # Smooth & draw
        if blue_x is None:
            blue_x = float(gap_cx)
        else:
            blue_x = blue_x * blue_x_smoothness + gap_cx * (1 - blue_x_smoothness)
        draw_x = int(round(blue_x))
        steer_control_x = draw_x

        # ===================== TB6612FNG MOTOR CONTROL (5 kHz) =====================
        # Initialize PWM at 5 kHz, 0% duty once
        if not pwm_initialized and not pwm_error_reported:
            try:
                pwm_driver = PWMDriver(PWMCHIP, PWM_CHANNEL_INDEX)
                pwm_driver.init(PWM_PERIOD_NS, PWM_START_DUTY_NS)  # 5 kHz @ 0% duty
                pwm_initialized = True
                print(f"[PWM] Initialized at {PWM_FREQ_HZ} Hz on pwmchip3/pwm{PWM_CHANNEL_INDEX} (duty=0)")
            except Exception as e:
                pwm_error_reported = True
                print(f"[PWM] Initialization error: {e}")

        if pwm_initialized:
            # Steering offset from center
            delta_px = steer_control_x - center_x

            # Decide sign: -1 left, +1 right, 0 near center
            if abs(delta_px) <= CENTER_DEADBAND_PX:
                sign = 0
            else:
                sign = 1 if delta_px > 0 else -1

            # Map distance to duty fraction [0,1]
            mag = min(abs(delta_px) / float(max(1, SPAN_PX)), 1.0)
            duty_ns = int(mag * PWM_PERIOD_NS) if sign != 0 else 0

            # Pick direction based on side (RIGHT_IS_FORWARD controls mapping)
            if sign == 0:
                # Stop: duty = 0, keep current IN pins as-is
                if last_duty_ns != 0:
                    pwm_driver.set_duty(0)
                    last_duty_ns = 0
            else:
                desired_dir = None
                if (sign > 0 and RIGHT_IS_FORWARD) or (sign < 0 and not RIGHT_IS_FORWARD):
                    desired_dir = "FORWARD"
                else:
                    desired_dir = "REVERSE"

                # If direction changed, first drop duty to 0, then flip pins
                if current_dir != desired_dir:
                    if last_duty_ns != 0:
                        pwm_driver.set_duty(0)
                        last_duty_ns = 0
                        # next loop we'll actually switch pins
                    else:
                        set_direction(desired_dir)

                # After pins are correct, apply duty
                if current_dir == desired_dir:
                    if duty_ns != last_duty_ns:
                        pwm_driver.set_duty(duty_ns)
                        last_duty_ns = duty_ns
        # ==========================================================================

        # Draw blue cue & HUD
        blue_pos = (draw_x, line_y)
        cv2.circle(combined, blue_pos, blue_circle_radius_px, blue_circle_color, blue_circle_thickness)
        cv2.putText(
            combined,
            f"Steer Control X: {steer_control_x}  | Duty: {last_duty_ns}/{PWM_PERIOD_NS}  | Dir: {current_dir}",
            hud_text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            hud_text_scale,
            hud_text_color,
            hud_text_thickness,
            cv2.LINE_AA
        )
        cv2.line(combined, (int(zone_left), 0), (int(zone_left), frame0.shape[0]), pull_zone_line_color, pull_zone_line_thickness)
        cv2.line(combined, (int(zone_right), 0), (int(zone_right), frame0.shape[0]), pull_zone_line_color, pull_zone_line_thickness)

        cv2.imshow("Depth: Camera 0 | Camera 1", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

finally:
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

    # Motor off, standby low
    try:
        if pwm_driver is not None:
            pwm_driver.set_duty(0)
            pwm_driver.close()
    except Exception as e:
        print(f"[PWM] Cleanup error: {e}")

    try:
        GPIO.output(AIN1_PIN, GPIO.LOW)
        GPIO.output(AIN2_PIN, GPIO.LOW)
        GPIO.output(STBY_PIN, GPIO.LOW)  # standby
        GPIO.cleanup()
    except Exception:
        pass

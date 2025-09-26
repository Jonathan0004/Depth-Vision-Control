"""Stereo depth estimation with obstacle detection and steering guidance.

TB6612FNG (one motor, channel A) with adjustable PWM frequency (duty-only
control). Direction from steering sign, speed from magnitude, center => 0%
duty (stop).

Robustness additions:
- PWM auto-probe: find a working pwmchip*/pwm0 that accepts the requested
  carrier frequency
- Safe PWM init order: duty_cycle -> period -> enable (avoids EINVAL)
"""

import os
import glob
import time
import queue
import math
from dataclasses import dataclass
from threading import Thread
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

# =============================================================================
# Tunable parameters (grouped by subsystem)
# =============================================================================
# Tip: Adjust these values to suit your platform. Inline notes explain how each
# setting impacts behaviour so you can iterate quickly without digging through
# the control logic below.

# --- Motor control behaviour -------------------------------------------------
# Base PWM carrier frequency in Hertz. Increase for quieter operation, decrease
# if the driver struggles to keep up. The period in nanoseconds is derived from
# this value automatically; use whole numbers to avoid floating point rounding
# surprises.
PWM_FREQUENCY_HZ = 5_000
if PWM_FREQUENCY_HZ <= 0:
    raise ValueError("PWM_FREQUENCY_HZ must be a positive, non-zero value.")

# Derived PWM period in nanoseconds. Only override directly if you require a
# non-integer frequency. Otherwise, tweak ``PWM_FREQUENCY_HZ`` above.
PWM_PERIOD_NS = int(round(1_000_000_000 / PWM_FREQUENCY_HZ))

# Convenience reference of the actual integer frequency created by
# ``PWM_PERIOD_NS``. Useful for logging or verifying rounding behaviour.
PWM_ACTUAL_FREQUENCY_HZ = int(round(1_000_000_000 / PWM_PERIOD_NS))

# Number of pixels away from the center that map to 100% duty cycle. Lowering
# this value makes the vehicle respond more aggressively to small offsets;
# increasing it makes the steering more relaxed.
SPAN_PX = 150

# Deadband around the image center (in pixels). Within this window the motor is
# held stopped to prevent oscillations when the target is nearly centered.
DEADZONE_PX = 2

# Number of discrete steps used to ramp the PWM duty from 0 to 100%. Higher
# values make duty changes more gradual, mirroring the soft-start behaviour in
# ``pwmControl.py``.
PWM_DUTY_RAMP_STEPS = 20

# Preferred sysfs pwmchip path. If unavailable, an auto-probe searches others.
PREFERRED_PWM_CHIP = "/sys/class/pwm/pwmchip3"

# --- GPIO selection ----------------------------------------------------------
# Whether to bypass Jetson.GPIO and use the sysfs GPIO shim. Toggle this if you
# prefer explicit sysfs control instead of the Jetson.GPIO library.
USE_SYSFS_GPIO = False

# BOARD pin choices when Jetson.GPIO is available.
STBY_BOARD_PIN = 37  # STBY HIGH enables the driver
AIN1_BOARD_PIN = 31  # Forward: AIN1=HIGH, AIN2=LOW
AIN2_BOARD_PIN = 29  # Reverse: AIN1=LOW,  AIN2=HIGH

# Sysfs GPIO line numbers (only used if USE_SYSFS_GPIO becomes True).
STBY_GPIO_NUM = 0    # TB6612 STBY
AIN1_GPIO_NUM = 0    # TB6612 AIN1 (forward)
AIN2_GPIO_NUM = 0    # TB6612 AIN2 (reverse)

GPIO = None
if not USE_SYSFS_GPIO:
    import Jetson.GPIO as GPIO
    GPIO.setmode(GPIO.BOARD)          # BOARD numbering when Jetson.GPIO is active

# --- Depth & perception heuristics ------------------------------------------
# Grid sampling density for the point cloud derived from the depth maps. Higher
# values yield smoother obstacle estimates but cost more compute per frame.
rows, cols = 25, 50

# Threshold logic for obstacle extraction. ``depth_diff_threshold`` is the base
# difference (in depth units) and ``std_multiplier`` scales the per-frame depth
# variation. Tweak to balance sensitivity vs. noise.
depth_diff_threshold = 8
std_multiplier = 0.3

# Pixels to ignore at the top/bottom of the frame, trimming sky/hood regions.
# Increase if environmental clutter is triggering false obstacles.
top_cutoff_pixels = 10
bottom_cutoff_pixels = 54

# Pull-zone configuration encourages steering towards open space. Shift the
# ``pull_zone_center_offset_px`` if the cameras are not perfectly aligned with
# the vehicle center.
pull_influence_radius_px = 120
pull_zone_center_offset_px = 0

# Low-pass smoothing for the blue steering indicator. Set to 0 for immediate
# changes or closer to 1.0 for a slower, more stable response.
blue_x_smoothness = 0.7

# --- Visual output toggle ----------------------------------------------------
# Set to ``False`` to skip all rendering steps (color maps, overlays, display
# windows). Motor control and perception stay active while avoiding the extra
# GPU/CPU work associated with drawing.
ENABLE_VISUAL_OUTPUT = True

# --- On-screen display styling ----------------------------------------------
cutoff_line_color = (0, 255, 0)
cutoff_line_thickness = 1

blue_circle_radius_px = 12
blue_circle_color = (255, 0, 0)
blue_circle_thickness = 3

pull_zone_line_color = (255, 0, 0)
pull_zone_line_thickness = 1

hud_text_position = (10, 30)
hud_text_color = (255, 255, 255)
hud_text_scale = 0.54
hud_text_thickness = 1

# =============================================================================
# Hardware abstraction helpers
# =============================================================================

# --- Sysfs GPIO shim ---------------------------------------------------------
class SysfsGPIO:
    def __init__(self, num):
        self.num = int(num)
        self.base = f"/sys/class/gpio/gpio{self.num}"
        if not os.path.isdir(self.base):
            try:
                with open("/sys/class/gpio/export", "w") as f:
                    f.write(str(self.num))
            except Exception as e:
                raise RuntimeError(f"Export GPIO{self.num} failed: {e}")
            for _ in range(100):
                if os.path.isdir(self.base):
                    break
                time.sleep(0.01)
        try:
            with open(os.path.join(self.base, "direction"), "w") as f:
                f.write("out")
        except Exception as e:
            raise RuntimeError(f"GPIO{self.num} direction set failed: {e}")
        self.vpath = os.path.join(self.base, "value")

    def write(self, val: int):
        try:
            with open(self.vpath, "w") as f:
                f.write("1" if val else "0")
        except Exception as e:
            raise RuntimeError(f"GPIO{self.num} write failed: {e}")

    def cleanup(self):
        try:
            with open("/sys/class/gpio/unexport", "w") as f:
                f.write(str(self.num))
        except Exception:
            pass

# --- Runtime state -----------------------------------------------------------
blue_x = None
steer_control_x = None
pwm_initialized = False
pwm_error_reported = False
last_duty_ns = None
pwm_driver = None
current_dir = 0  # +1 forward, -1 reverse, 0 unknown
gpio_stby = gpio_ain1 = gpio_ain2 = None

# --- Utility helpers ---------------------------------------------------------
def clamp(val, minn, maxn):
    return max(min(val, maxn), minn)


@dataclass
class SceneAnalysis:
    combined: Optional[np.ndarray]
    center_x: int
    gap_center_x: int
    zone_left: int
    zone_right: int
    line_y: int


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
        # Export channel if needed
        if not os.path.isdir(self.pwm_path):
            PWMDriver._wr(os.path.join(self.chip_path, "export"), str(self.channel_index))
            for _ in range(200):
                if os.path.isdir(self.pwm_path):
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError(f"pwm{self.channel_index} did not appear after export")

        # Disable; set duty=0 BEFORE period to avoid EINVAL, then enable
        try:
            PWMDriver._wr(os.path.join(self.pwm_path, "enable"), "0")
        except OSError:
            pass
        # Defensive: duty=0 first, then period, then duty to requested value
        try:
            PWMDriver._wr(os.path.join(self.pwm_path, "duty_cycle"), 0)
        except OSError:
            pass
        PWMDriver._wr(os.path.join(self.pwm_path, "period"), period_ns)
        PWMDriver._wr(os.path.join(self.pwm_path, "duty_cycle"), duty_ns)
        PWMDriver._wr(os.path.join(self.pwm_path, "enable"), "1")
        self._duty_file = open(os.path.join(self.pwm_path, "duty_cycle"), "w")

    def set_duty(self, duty_ns: int):
        if self._duty_file is None:
            raise RuntimeError("PWM not initialised.")
        if duty_ns < 0: duty_ns = 0
        if duty_ns > PWM_PERIOD_NS: duty_ns = PWM_PERIOD_NS
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

def pwm_autoprobe(period_ns: int, initial_duty_ns: int = 0):
    """
    Find a pwmchip whose pwm0 accepts the requested period.
    Returns (driver, chip_path) on success; raises on failure.
    """
    candidates = []
    if PREFERRED_PWM_CHIP and os.path.isdir(PREFERRED_PWM_CHIP):
        candidates.append(PREFERRED_PWM_CHIP)
    for chip in sorted(glob.glob("/sys/class/pwm/pwmchip*")):
        if chip not in candidates:
            candidates.append(chip)
    last_err = None
    for chip in candidates:
        try:
            drv = PWMDriver(chip, 0)          # try channel 0 first
            drv.init(period_ns, initial_duty_ns)
            achieved_hz = int(round(1_000_000_000 / period_ns))
            print(f"[PWM] Using {chip}/pwm0 @ {achieved_hz} Hz")
            return drv, chip
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No pwmchip accepted period_ns={period_ns}. Last error: {last_err}")

def tb6612_pins_init():
    global gpio_stby, gpio_ain1, gpio_ain2
    if not USE_SYSFS_GPIO:
        GPIO.setup(STBY_BOARD_PIN, GPIO.OUT, initial=GPIO.HIGH)  # STBY HIGH
        GPIO.setup(AIN1_BOARD_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(AIN2_BOARD_PIN, GPIO.OUT, initial=GPIO.LOW)
    else:
        if not all([STBY_GPIO_NUM, AIN1_GPIO_NUM, AIN2_GPIO_NUM]):
            raise RuntimeError(
                "Sysfs GPIO fallback active: please set STBY_GPIO_NUM, AIN1_GPIO_NUM, AIN2_GPIO_NUM."
            )
        gpio_stby = SysfsGPIO(STBY_GPIO_NUM)
        gpio_ain1 = SysfsGPIO(AIN1_GPIO_NUM)
        gpio_ain2 = SysfsGPIO(AIN2_GPIO_NUM)
        gpio_stby.write(1)  # STBY HIGH

def tb6612_set_direction(direction: int):
    # Forward: AIN1=HIGH, AIN2=LOW. Reverse: AIN1=LOW, AIN2=HIGH.
    if not USE_SYSFS_GPIO:
        if direction > 0:
            GPIO.output(AIN1_BOARD_PIN, GPIO.HIGH)
            GPIO.output(AIN2_BOARD_PIN, GPIO.LOW)
        elif direction < 0:
            GPIO.output(AIN1_BOARD_PIN, GPIO.LOW)
            GPIO.output(AIN2_BOARD_PIN, GPIO.HIGH)
    else:
        if direction > 0:
            gpio_ain1.write(1); gpio_ain2.write(0)
        elif direction < 0:
            gpio_ain1.write(0); gpio_ain2.write(1)
    # direction == 0 -> leave pins; stop is done with duty=0


# =============================================================================
# High-level perception + control helpers
# =============================================================================

def analyze_scene(frame0, depth0, frame1, depth1) -> SceneAnalysis:
    h0, w0 = depth0.shape
    h1, w1 = depth1.shape
    points = []
    for depth, x_offset, w, h, cam_idx in [
        (depth0, 0, w0, h0, 0),
        (depth1, w0, w1, h1, 1),
    ]:
        for c in range(cols):
            for r in range(rows):
                px = int((c + 0.5) * w / cols) + x_offset
                py = int((r + 0.5) * h / rows)
                z = float(depth[int(py), int(px - x_offset)])
                points.append((px, py, z, cam_idx))
    cloud = np.array(points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("cam", "i4")])

    zs = cloud["z"]
    mean_z, std_z = zs.mean(), zs.std()
    thresh = max(depth_diff_threshold, std_multiplier * std_z)
    mask = (mean_z - zs) > thresh

    combined = None
    h_combined = frame0.shape[0]
    w_combined = frame0.shape[1] + frame1.shape[1]
    line_y = (top_cutoff_pixels + (h_combined - bottom_cutoff_pixels)) // 2
    center_x = (w_combined // 2) + pull_zone_center_offset_px

    if ENABLE_VISUAL_OUTPUT:
        norm0 = cv2.normalize(depth0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cmap0 = cv2.applyColorMap(
            cv2.resize(norm0, (frame0.shape[1], frame0.shape[0])), cv2.COLORMAP_MAGMA
        )
        norm1 = cv2.normalize(depth1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cmap1 = cv2.applyColorMap(
            cv2.resize(norm1, (frame1.shape[1], frame1.shape[0])), cv2.COLORMAP_MAGMA
        )

        for cmap, h, w in [
            (cmap0, frame0.shape[0], frame0.shape[1]),
            (cmap1, frame1.shape[0], frame1.shape[1]),
        ]:
            top_y = top_cutoff_pixels
            bottom_y = h - bottom_cutoff_pixels
            cv2.line(cmap, (0, top_y), (w, top_y), cutoff_line_color, cutoff_line_thickness)
            cv2.line(cmap, (0, bottom_y), (w, bottom_y), cutoff_line_color, cutoff_line_thickness)

        combined = np.hstack((cmap0, cmap1))

    zone_left = clamp(center_x - pull_influence_radius_px, 0, w_combined)
    zone_right = clamp(center_x + pull_influence_radius_px, 0, w_combined)

    red_xs_all, red_xs_in_zone = [], []
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
                best_width, best_dist, best_cx = width, dist, cx
        return int(best_cx)

    gap_cx = (
        center_x
        if len(red_xs_in_zone) == 0
        else widest_gap_center(blockers_all, preferred_x=center_x)
    )

    return SceneAnalysis(
        combined=combined,
        center_x=int(center_x),
        gap_center_x=int(gap_cx),
        zone_left=int(zone_left),
        zone_right=int(zone_right),
        line_y=int(line_y),
    )


def update_guidance(target_x: int) -> int:
    global blue_x, steer_control_x
    if blue_x is None:
        blue_x = float(target_x)
    else:
        blue_x = blue_x * blue_x_smoothness + target_x * (1 - blue_x_smoothness)
    steer_control_x = int(round(blue_x))
    return steer_control_x


def apply_motor_control(center_x: int, steer_x: int) -> None:
    global last_duty_ns, current_dir, pwm_error_reported
    if not pwm_initialized or pwm_error_reported:
        return

    delta_px = steer_x - center_x
    desired_dir = 0
    if abs(delta_px) > DEADZONE_PX:
        desired_dir = 1 if delta_px > 0 else -1

    if desired_dir != 0 and desired_dir != current_dir:
        pwm_driver.set_duty(0)
        tb6612_set_direction(desired_dir)
        current_dir = desired_dir

    mag = 0.0 if desired_dir == 0 else min(abs(delta_px) / float(SPAN_PX), 1.0)
    target_duty_ns = int(round(mag * PWM_PERIOD_NS))

    if PWM_DUTY_RAMP_STEPS <= 0 or last_duty_ns is None:
        smoothed_duty = target_duty_ns
    else:
        step_ns = max(1, int(math.ceil(PWM_PERIOD_NS / float(PWM_DUTY_RAMP_STEPS))))
        if target_duty_ns > last_duty_ns:
            smoothed_duty = min(target_duty_ns, last_duty_ns + step_ns)
        else:
            smoothed_duty = max(target_duty_ns, last_duty_ns - step_ns)

    if (last_duty_ns is None) or (smoothed_duty != last_duty_ns):
        pwm_driver.set_duty(smoothed_duty)
        last_duty_ns = smoothed_duty


def overlay_visuals(analysis: SceneAnalysis, draw_x: int) -> Optional[np.ndarray]:
    if not ENABLE_VISUAL_OUTPUT or analysis.combined is None:
        return None

    combined = analysis.combined
    cv2.circle(
        combined,
        (draw_x, analysis.line_y),
        blue_circle_radius_px,
        blue_circle_color,
        blue_circle_thickness,
    )
    cv2.putText(
        combined,
        f"Steer Control: {draw_x}",
        hud_text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        hud_text_scale,
        hud_text_color,
        hud_text_thickness,
        cv2.LINE_AA,
    )
    cv2.line(
        combined,
        (int(analysis.zone_left), 0),
        (int(analysis.zone_left), combined.shape[0]),
        pull_zone_line_color,
        pull_zone_line_thickness,
    )
    cv2.line(
        combined,
        (int(analysis.zone_right), 0),
        (int(analysis.zone_right), combined.shape[0]),
        pull_zone_line_color,
        pull_zone_line_thickness,
    )
    return combined


# =============================================================================
# Perception pipeline (depth model + stereo capture)
# =============================================================================
depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    device=0
)

def make_gst(sensor_id):
    """Build a GStreamer string for the Jetson camera pipeline.

    Adjust the width/height/framerate caps below to match your sensors. Keep
    ``max-buffers=1`` so latency stays low for the control loop.
    """
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

# =============================================================================
# Main control loop
# =============================================================================


def main():
    global pwm_driver, pwm_initialized, running
    try:
        # Init TB6612 pins + PWM (start stopped)
        tb6612_pins_init()
        pwm_driver, which_chip = pwm_autoprobe(PWM_PERIOD_NS, 0)  # starts disabled->enabled with duty=0
        pwm_initialized = True
        print(
            "PWM initialized: target "
            f"{PWM_FREQUENCY_HZ:,} Hz (achieved {PWM_ACTUAL_FREQUENCY_HZ:,} Hz) on {which_chip}/pwm0 "
            "(duty=0%). TB6612 STBY=HIGH."
        )
        print("Direction mapping: Forward=AIN1 HIGH, AIN2 LOW; Reverse=AIN1 LOW, AIN2 HIGH.")

        while True:
            try:
                frame0, depth0, frame1, depth1 = result_q.get(timeout=0.01)
            except queue.Empty:
                continue

            analysis = analyze_scene(frame0, depth0, frame1, depth1)
            steer_x = update_guidance(analysis.gap_center_x)
            apply_motor_control(analysis.center_x, steer_x)

            if ENABLE_VISUAL_OUTPUT:
                display_img = overlay_visuals(analysis, steer_x)
                if display_img is not None:
                    cv2.imshow("Depth: Camera 0 | Camera 1", display_img)
                    if (cv2.waitKey(1) & 0xFF) == 27:
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
        if ENABLE_VISUAL_OUTPUT:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        # PWM cleanup only if it really initialised
        try:
            if pwm_initialized and pwm_driver is not None:
                pwm_driver.set_duty(0)
                pwm_driver.close()
        except Exception as e:
            print(f"[PWM] Cleanup warning: {e}")
        # GPIO cleanup
        try:
            if not USE_SYSFS_GPIO:
                GPIO.output(STBY_BOARD_PIN, GPIO.LOW)
                GPIO.output(AIN1_BOARD_PIN, GPIO.LOW)
                GPIO.output(AIN2_BOARD_PIN, GPIO.LOW)
                GPIO.cleanup()
            else:
                if isinstance(gpio_stby, SysfsGPIO):
                    gpio_stby.cleanup()
                if isinstance(gpio_ain1, SysfsGPIO):
                    gpio_ain1.cleanup()
                if isinstance(gpio_ain2, SysfsGPIO):
                    gpio_ain2.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()

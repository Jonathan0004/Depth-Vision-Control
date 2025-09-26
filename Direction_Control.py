"""Minimal stereo depth steering loop with fixed 50% PWM output."""

import os
import time
import queue
from dataclasses import dataclass
from threading import Thread
from typing import Optional

import cv2
import numpy as np
import torch
import Jetson.GPIO as GPIO
from PIL import Image
from transformers import pipeline

# Core behaviour (kept small so there are only a few obvious knobs)
PWM_FREQUENCY_HZ = 5_000
PWM_PERIOD_NS = int(round(1_000_000_000 / PWM_FREQUENCY_HZ))
PWM_CHIP_PATH = "/sys/class/pwm/pwmchip3"
DEADZONE_PX = 2

GPIO.setmode(GPIO.BOARD)
STBY_BOARD_PIN, AIN1_BOARD_PIN, AIN2_BOARD_PIN = 37, 31, 29

# Vision heuristics (kept inline for clarity)
ROWS, COLS = 25, 50
DEPTH_DIFF_THRESHOLD = 8
STD_MULTIPLIER = 0.3
TOP_CUTOFF = 10
BOTTOM_CUTOFF = 54
PULL_RADIUS = 120
ENABLE_VISUAL_OUTPUT = True

CUT_LINE_COLOR = (0, 255, 0)
PULL_LINE_COLOR = (255, 0, 0)
BLUE_MARKER_COLOR = (255, 0, 0)
HUD_TEXT_COLOR = (255, 255, 255)

pwm_initialized = False
pwm_driver = None
current_dir = 0

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

def tb6612_pins_init():
    GPIO.setup(STBY_BOARD_PIN, GPIO.OUT, initial=GPIO.HIGH)  # STBY HIGH
    GPIO.setup(AIN1_BOARD_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(AIN2_BOARD_PIN, GPIO.OUT, initial=GPIO.LOW)

def tb6612_set_direction(direction: int):
    # Forward: AIN1=HIGH, AIN2=LOW. Reverse: AIN1=LOW, AIN2=HIGH.
    if direction > 0:
        GPIO.output(AIN1_BOARD_PIN, GPIO.HIGH)
        GPIO.output(AIN2_BOARD_PIN, GPIO.LOW)
    elif direction < 0:
        GPIO.output(AIN1_BOARD_PIN, GPIO.LOW)
        GPIO.output(AIN2_BOARD_PIN, GPIO.HIGH)
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
        for c in range(COLS):
            for r in range(ROWS):
                px = int((c + 0.5) * w / COLS) + x_offset
                py = int((r + 0.5) * h / ROWS)
                z = float(depth[int(py), int(px - x_offset)])
                points.append((px, py, z, cam_idx))
    cloud = np.array(points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("cam", "i4")])

    zs = cloud["z"]
    mean_z, std_z = zs.mean(), zs.std()
    thresh = max(DEPTH_DIFF_THRESHOLD, STD_MULTIPLIER * std_z)
    mask = (mean_z - zs) > thresh

    combined = None
    h_combined = frame0.shape[0]
    w_combined = frame0.shape[1] + frame1.shape[1]
    line_y = (TOP_CUTOFF + (h_combined - BOTTOM_CUTOFF)) // 2
    center_x = w_combined // 2

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
            top_y = TOP_CUTOFF
            bottom_y = h - BOTTOM_CUTOFF
            cv2.line(cmap, (0, top_y), (w, top_y), CUT_LINE_COLOR, 1)
            cv2.line(cmap, (0, bottom_y), (w, bottom_y), CUT_LINE_COLOR, 1)

        combined = np.hstack((cmap0, cmap1))

    zone_left = clamp(center_x - PULL_RADIUS, 0, w_combined)
    zone_right = clamp(center_x + PULL_RADIUS, 0, w_combined)

    red_xs_all, red_xs_in_zone = [], []
    for is_obst, pt in zip(mask, cloud):
        if not is_obst:
            continue
        px, py, cam = int(pt["x"]), int(pt["y"]), pt["cam"]
        frame_h = frame0.shape[0] if cam == 0 else frame1.shape[0]
        if py < TOP_CUTOFF or py > (frame_h - BOTTOM_CUTOFF):
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
    return int(round(target_x))


def apply_motor_control(center_x: int, steer_x: int) -> None:
    global current_dir
    if not pwm_initialized or pwm_driver is None:
        return

    delta_px = steer_x - center_x
    if abs(delta_px) <= DEADZONE_PX:
        pwm_driver.set_duty(0)
        if current_dir != 0:
            print("Direction: Stop | PWM: 0%")
        current_dir = 0
        return

    desired_dir = 1 if delta_px > 0 else -1
    if desired_dir != current_dir:
        tb6612_set_direction(desired_dir)
        direction_label = "Forward" if desired_dir > 0 else "Reverse"
        print(f"Direction: {direction_label} | PWM: 50%")
        current_dir = desired_dir

    pwm_driver.set_duty(PWM_PERIOD_NS // 2)


def overlay_visuals(analysis: SceneAnalysis, draw_x: int) -> Optional[np.ndarray]:
    if not ENABLE_VISUAL_OUTPUT or analysis.combined is None:
        return None

    combined = analysis.combined
    cv2.circle(
        combined,
        (draw_x, analysis.line_y),
        12,
        BLUE_MARKER_COLOR,
        3,
    )
    cv2.putText(
        combined,
        f"Steer Control: {draw_x}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        HUD_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        combined,
        (int(analysis.zone_left), 0),
        (int(analysis.zone_left), combined.shape[0]),
        PULL_LINE_COLOR,
        1,
    )
    cv2.line(
        combined,
        (int(analysis.zone_right), 0),
        (int(analysis.zone_right), combined.shape[0]),
        PULL_LINE_COLOR,
        1,
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
        pwm_driver = PWMDriver(PWM_CHIP_PATH, 0)
        pwm_driver.init(PWM_PERIOD_NS, 0)
        pwm_initialized = True
        print(
            "PWM initialized: target "
            f"{PWM_FREQUENCY_HZ:,} Hz on {PWM_CHIP_PATH}/pwm0 (duty=0%). TB6612 STBY=HIGH."
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
            GPIO.output(STBY_BOARD_PIN, GPIO.LOW)
            GPIO.output(AIN1_BOARD_PIN, GPIO.LOW)
            GPIO.output(AIN2_BOARD_PIN, GPIO.LOW)
            GPIO.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()

"""Minimal depth-based steering: output direction and fixed 50% PWM."""

import os
import time
import queue
from dataclasses import dataclass
from threading import Thread

import cv2
import numpy as np
import torch
import Jetson.GPIO as GPIO
from PIL import Image
from transformers import pipeline

# Core configuration kept intentionally small.
PWM_CHIP_PATH = "/sys/class/pwm/pwmchip3"
PWM_CHANNEL = 0
PWM_FREQUENCY_HZ = 5_000
PWM_PERIOD_NS = int(1_000_000_000 / PWM_FREQUENCY_HZ)
HALF_DUTY_NS = PWM_PERIOD_NS // 2

STBY_BOARD_PIN = 37
AIN1_BOARD_PIN = 31
AIN2_BOARD_PIN = 29

GRID_ROWS = 20
GRID_COLS = 40
TOP_CUTOFF = 10
BOTTOM_CUTOFF = 54
DEADZONE_PX = 4

GPIO.setmode(GPIO.BOARD)


@dataclass
class SceneSummary:
    center_x: int
    gap_center_x: int


class PWMDriver:
    def __init__(self, chip_path: str, channel_index: int):
        self.chip_path = chip_path
        self.channel_index = int(channel_index)
        self.pwm_path = os.path.join(self.chip_path, f"pwm{self.channel_index}")
        self._duty_file = None

    @staticmethod
    def _write(path: str, value) -> None:
        with open(path, "w") as f:
            f.write(str(value))

    def init(self, period_ns: int) -> None:
        if not os.path.isdir(self.pwm_path):
            PWMDriver._write(os.path.join(self.chip_path, "export"), self.channel_index)
            for _ in range(200):
                if os.path.isdir(self.pwm_path):
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError(f"pwm{self.channel_index} did not appear after export")

        try:
            PWMDriver._write(os.path.join(self.pwm_path, "enable"), "0")
        except OSError:
            pass
        try:
            PWMDriver._write(os.path.join(self.pwm_path, "duty_cycle"), 0)
        except OSError:
            pass

        PWMDriver._write(os.path.join(self.pwm_path, "period"), period_ns)
        PWMDriver._write(os.path.join(self.pwm_path, "duty_cycle"), 0)
        PWMDriver._write(os.path.join(self.pwm_path, "enable"), "1")
        self._duty_file = open(os.path.join(self.pwm_path, "duty_cycle"), "w")

    def set_duty(self, duty_ns: int) -> None:
        if self._duty_file is None:
            raise RuntimeError("PWM not initialised.")
        duty_ns = max(0, min(duty_ns, PWM_PERIOD_NS))
        self._duty_file.seek(0)
        self._duty_file.write(str(int(duty_ns)))
        self._duty_file.flush()

    def close(self) -> None:
        try:
            if self._duty_file:
                self._duty_file.close()
        finally:
            try:
                PWMDriver._write(os.path.join(self.pwm_path, "enable"), "0")
            except Exception:
                pass


def tb6612_pins_init() -> None:
    GPIO.setup(STBY_BOARD_PIN, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(AIN1_BOARD_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(AIN2_BOARD_PIN, GPIO.OUT, initial=GPIO.LOW)


def tb6612_set_direction(direction: int) -> None:
    if direction > 0:
        GPIO.output(AIN1_BOARD_PIN, GPIO.HIGH)
        GPIO.output(AIN2_BOARD_PIN, GPIO.LOW)
    elif direction < 0:
        GPIO.output(AIN1_BOARD_PIN, GPIO.LOW)
        GPIO.output(AIN2_BOARD_PIN, GPIO.HIGH)


def analyze_scene(frame0, depth0, frame1, depth1) -> SceneSummary:
    h0, w0 = depth0.shape
    h1, w1 = depth1.shape
    combined_width = frame0.shape[1] + frame1.shape[1]
    center_x = combined_width // 2

    samples = []
    for depth, x_offset, width, height, cam_idx in [
        (depth0, 0, w0, h0, 0),
        (depth1, w0, w1, h1, 1),
    ]:
        for c in range(GRID_COLS):
            for r in range(GRID_ROWS):
                px = int((c + 0.5) * width / GRID_COLS) + x_offset
                py = int((r + 0.5) * height / GRID_ROWS)
                samples.append((px, py, float(depth[int(py), int(px - x_offset)]), cam_idx))

    cloud = np.array(samples, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("cam", "i4")])
    zs = cloud["z"]
    mean_z, std_z = zs.mean(), zs.std()
    threshold = mean_z - max(8, 0.3 * std_z)

    blockers = [0]
    for pt in cloud:
        if pt["z"] >= threshold:
            continue
        py = int(pt["y"])
        frame_h = frame0.shape[0] if pt["cam"] == 0 else frame1.shape[0]
        if py < TOP_CUTOFF or py > (frame_h - BOTTOM_CUTOFF):
            continue
        blockers.append(int(pt["x"]))

    blockers.append(combined_width)
    blockers = sorted(set(blockers))

    best_width = -1
    best_center = center_x
    for left, right in zip(blockers[:-1], blockers[1:]):
        width = right - left
        gap_center = (left + right) // 2
        distance = abs(gap_center - center_x)
        if width > best_width or (width == best_width and distance < abs(best_center - center_x)):
            best_width = width
            best_center = gap_center

    return SceneSummary(center_x=center_x, gap_center_x=best_center)


pwm_driver = None
pwm_ready = False
current_direction = 0


def apply_motor_control(center_x: int, steer_x: int) -> None:
    global current_direction
    if not pwm_ready:
        return

    offset = steer_x - center_x
    if abs(offset) <= DEADZONE_PX:
        if current_direction != 0:
            pwm_driver.set_duty(0)
            current_direction = 0
            print("Direction: stop (deadzone)")
        return

    desired_dir = 1 if offset > 0 else -1
    if desired_dir != current_direction:
        pwm_driver.set_duty(0)
        tb6612_set_direction(desired_dir)
        current_direction = desired_dir
        print("Direction:", "forward" if desired_dir > 0 else "reverse")

    pwm_driver.set_duty(HALF_DUTY_NS)


depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    device=0,
)


def make_gst(sensor_id: int) -> str:
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
            if q.full():
                q.get_nowait()
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
        with torch.amp.autocast(device_type="cuda"), torch.no_grad():
            outs = depth_pipe(imgs)
        d0, d1 = [np.array(o["depth"]) for o in outs]
        if result_q.full():
            result_q.get_nowait()
        result_q.put((f0, d0, f1, d1))


Thread(target=infer, daemon=True).start()


def main() -> None:
    global pwm_driver, pwm_ready, running
    try:
        tb6612_pins_init()
        pwm_driver = PWMDriver(PWM_CHIP_PATH, PWM_CHANNEL)
        pwm_driver.init(PWM_PERIOD_NS)
        pwm_ready = True
        print(f"PWM ready at {PWM_FREQUENCY_HZ} Hz, half duty {HALF_DUTY_NS} ns")

        while True:
            try:
                frame0, depth0, frame1, depth1 = result_q.get(timeout=0.01)
            except queue.Empty:
                continue

            summary = analyze_scene(frame0, depth0, frame1, depth1)
            apply_motor_control(summary.center_x, summary.gap_center_x)
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
            if pwm_ready and pwm_driver is not None:
                pwm_driver.set_duty(0)
                pwm_driver.close()
        except Exception as exc:
            print(f"[PWM] cleanup warning: {exc}")
        try:
            GPIO.output(STBY_BOARD_PIN, GPIO.LOW)
            GPIO.output(AIN1_BOARD_PIN, GPIO.LOW)
            GPIO.output(AIN2_BOARD_PIN, GPIO.LOW)
            GPIO.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()

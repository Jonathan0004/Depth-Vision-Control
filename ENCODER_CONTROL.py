"""Stereo depth estimation with obstacle detection and steering guidance.

The script streams from two cameras, infers depth using the
``depth-anything`` model, highlights obstacles inside configurable cutoff
bands, and visualises a blue steering cue that points toward the widest gap.
All tunable parameters live together for quick iteration.
"""

import json
import os
import queue
import threading
import time
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from typing import Tuple

try:  # Jetson Nano GPIO for motor control
    import Jetson.GPIO as GPIO
    HAS_GPIO = True
except (ImportError, RuntimeError):  # RuntimeError is raised if not run as root on Jetson
    HAS_GPIO = False

    class _DummyPWM:
        def __init__(self, *_, **__):
            self._duty = 0.0

        def start(self, duty):
            self._duty = duty

        def ChangeDutyCycle(self, duty):
            self._duty = duty

        def stop(self):
            pass

    class _DummyGPIO:
        BOARD = BCM = OUT = HIGH = LOW = None

        @staticmethod
        def setmode(*_):
            pass

        @staticmethod
        def setup(*_, **__):
            pass

        @staticmethod
        def output(*_, **__):
            pass

        @staticmethod
        def cleanup():
            pass

        @staticmethod
        def PWM(*_, **__):
            return _DummyPWM()

    GPIO = _DummyGPIO()

try:  # SPI access for AS5048A
    import spidev
    HAS_SPI = True
except ImportError:
    HAS_SPI = False

    class _DummySpiDev:
        def __init__(self):
            self.max_speed_hz = 0
            self.mode = 0

        def open(self, *_, **__):
            pass

        def xfer2(self, *_):
            return [0x00, 0x00]

        def close(self):
            pass

    class spidev:  # type: ignore
        SpiDev = _DummySpiDev


# ---------------------------------------------------------------------------
# Configuration — tweak these values to adjust behaviour without diving into
# the rest of the code. Where possible, related values are grouped together and
# documented so their impact is clear.
# ---------------------------------------------------------------------------

# Motor / encoder configuration ------------------------------------------------
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "as5048a_calibration.json")
ENCODER_COUNTS = 16384  # AS5048A has a 14-bit output (0-16383)
ENCODER_TOLERANCE_COUNTS = 40  # acceptable error band before the motor stops

# GPIO pin assignments (BOARD numbering is used by default)
GPIO_PIN_MODE = GPIO.BOARD
MOTOR_PWM_PIN = 32       # PWM0 on Jetson Nano (adjust to match wiring)
MOTOR_IN1_PIN = 31       # Direction pin A
MOTOR_IN2_PIN = 33       # Direction pin B

# Motion profile configuration
MOTOR_PWM_FREQUENCY = 1000          # Hz
MOTOR_TARGET_SPEED = 0.6            # duty cycle [0.0 - 1.0] when the motor is moving
MOTOR_ACCELERATION_STEP = 0.05      # duty delta applied each control tick
MOTOR_UPDATE_INTERVAL = 0.02        # seconds between control loop iterations

# Calibration prompts
CALIBRATION_SAMPLES = 12            # encoder samples averaged per limit capture

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
blue_x_smoothness = 0.7

# HUD text overlays on the combined frame
hud_text_position = (10, 30)
hud_text_color = (255, 255, 255)
hud_text_scale = 0.54
hud_text_thickness = 1


# ---------------------------------------------------------------------------
# Runtime state (initialised once, then updated as frames stream in)
# ---------------------------------------------------------------------------
blue_x = None              # persistent, smoothed x-position of the blue circle
steer_control_x = None     # integer pixel location displayed as HUD text


# ---------------------------------------------------------------------------
# Motor control helpers
# ---------------------------------------------------------------------------


class CalibrationManager:
    def __init__(self, path: str):
        self.path = path
        self.data = None
        self.current_screen_range = (0, 1)
        self._load()

    @property
    def ready(self) -> bool:
        return self.data is not None

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fp:
                self.data = json.load(fp)
            if self.data is not None:
                self.current_screen_range = (
                    int(self.data.get("steer_min", 0)),
                    int(self.data.get("steer_max", 1)),
                )
        except (OSError, json.JSONDecodeError):
            self.data = None

    def save(self, encoder_min: int, encoder_max: int, screen_range: Tuple[int, int]) -> None:
        payload = {
            "encoder_min": int(encoder_min) % ENCODER_COUNTS,
            "encoder_max": int(encoder_max) % ENCODER_COUNTS,
            "steer_min": int(screen_range[0]),
            "steer_max": int(screen_range[1]),
            "timestamp": time.time(),
        }
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        self.data = payload
        self.current_screen_range = (
            int(screen_range[0]),
            int(screen_range[1]),
        )

    def update_screen_range(self, screen_min: int, screen_max: int) -> None:
        self.current_screen_range = (screen_min, screen_max)

    def map_steer_to_encoder(self, steer_value: float) -> int:
        if not self.ready:
            raise RuntimeError("Encoder calibration not available.")
        steer_min, steer_max = self.current_screen_range
        if steer_max == steer_min:
            raise ZeroDivisionError("Screen range width is zero.")
        norm = (steer_value - steer_min) / (steer_max - steer_min)
        norm = max(0.0, min(1.0, norm))

        encoder_min = self.data["encoder_min"]
        encoder_max = self.data["encoder_max"]
        span = (encoder_max - encoder_min) % ENCODER_COUNTS
        target = (encoder_min + span * norm) % ENCODER_COUNTS
        return int(round(target))


class AS5048AEncoder:
    ANGLE_REGISTER = 0x3FFF

    def __init__(self, bus: int = 0, device: int = 0):
        self.available = HAS_SPI
        self.bus = bus
        self.device = device
        if self.available:
            self.spi = spidev.SpiDev()
            try:
                self.spi.open(bus, device)
                self.spi.max_speed_hz = 1_000_000
                self.spi.mode = 0b01
            except OSError:
                self.available = False
        else:
            self.spi = spidev.SpiDev()

    @staticmethod
    def _even_parity(word: int) -> int:
        parity = 0
        for i in range(15):
            parity ^= (word >> i) & 0x1
        return parity & 0x1

    def _read_register(self, address: int) -> int:
        command = 0x4000 | (address & 0x3FFF)
        command |= self._even_parity(command) << 15
        tx = [(command >> 8) & 0xFF, command & 0xFF]
        self.spi.xfer2(tx)
        # second transfer retrieves data
        data = self.spi.xfer2([0x00, 0x00])
        value = ((data[0] << 8) | data[1]) & 0x3FFF
        return value

    def read_angle(self, samples: int = 1) -> int:
        if not self.available:
            return 0
        values = []
        for _ in range(max(1, samples)):
            try:
                values.append(self._read_register(self.ANGLE_REGISTER))
            except OSError:
                continue
        if not values:
            return 0
        return int(round(sum(values) / len(values)))

    def close(self) -> None:
        if self.available:
            try:
                self.spi.close()
            except OSError:
                pass


class MotorController:
    def __init__(
        self,
        pwm_pin: int,
        in1_pin: int,
        in2_pin: int,
        pwm_frequency: int,
        target_speed: float,
        acceleration_step: float,
    ) -> None:
        self.hardware = HAS_GPIO
        self.pwm_pin = pwm_pin
        self.in1_pin = in1_pin
        self.in2_pin = in2_pin
        self.pwm_frequency = pwm_frequency
        self.target_speed = max(0.0, min(1.0, target_speed))
        self.acceleration_step = max(0.001, acceleration_step)
        self.current_speed = 0.0
        self.current_direction = 0
        self._pwm = None
        self._setup_gpio()

    def _setup_gpio(self) -> None:
        try:
            GPIO.setmode(GPIO_PIN_MODE)
            GPIO.setup(self.in1_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.in2_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.pwm_pin, GPIO.OUT, initial=GPIO.LOW)
            self._pwm = GPIO.PWM(self.pwm_pin, self.pwm_frequency)
            self._pwm.start(0.0)
        except Exception:
            self.hardware = False
            self._pwm = GPIO.PWM(self.pwm_pin, self.pwm_frequency)
            self._pwm.start(0.0)

    def _apply_direction(self, direction: int) -> None:
        if direction > 0:
            GPIO.output(self.in1_pin, GPIO.HIGH)
            GPIO.output(self.in2_pin, GPIO.LOW)
        elif direction < 0:
            GPIO.output(self.in1_pin, GPIO.LOW)
            GPIO.output(self.in2_pin, GPIO.HIGH)
        else:
            GPIO.output(self.in1_pin, GPIO.LOW)
            GPIO.output(self.in2_pin, GPIO.LOW)

    def _apply_pwm(self) -> None:
        if self._pwm is None:
            return
        self._pwm.ChangeDutyCycle(max(0.0, min(100.0, self.current_speed * 100.0)))

    def update(self, direction: int) -> None:
        direction = max(-1, min(1, direction))
        target_speed = self.target_speed if direction != 0 else 0.0
        if self.current_speed < target_speed:
            self.current_speed = min(target_speed, self.current_speed + self.acceleration_step)
        elif self.current_speed > target_speed:
            self.current_speed = max(target_speed, self.current_speed - self.acceleration_step)

        if self.current_speed == 0.0:
            direction = 0
        if direction != self.current_direction:
            self.current_direction = direction
            self._apply_direction(direction)
        self._apply_pwm()

    def stop_immediately(self) -> None:
        self.current_speed = 0.0
        self.current_direction = 0
        self._apply_direction(0)
        self._apply_pwm()

    def cleanup(self) -> None:
        try:
            if self._pwm is not None:
                self._pwm.stop()
        finally:
            try:
                GPIO.cleanup()
            except Exception:
                pass


def encoder_delta(target: int, current: int) -> int:
    half = ENCODER_COUNTS // 2
    diff = (target - current + half) % ENCODER_COUNTS - half
    return int(diff)


calibration_manager = CalibrationManager(CALIBRATION_FILE)
encoder = AS5048AEncoder()
motor_controller = MotorController(
    pwm_pin=MOTOR_PWM_PIN,
    in1_pin=MOTOR_IN1_PIN,
    in2_pin=MOTOR_IN2_PIN,
    pwm_frequency=MOTOR_PWM_FREQUENCY,
    target_speed=MOTOR_TARGET_SPEED,
    acceleration_step=MOTOR_ACCELERATION_STEP,
)

motor_stop_event = threading.Event()
calibration_active_event = threading.Event()

if not calibration_manager.ready:
    print("No encoder calibration found. Press 'c' to perform calibration.")
if not encoder.available:
    print("Warning: AS5048A encoder not detected. Encoder readings will be zero.")
if not motor_controller.hardware:
    print("Warning: Jetson.GPIO unavailable. Motor control is running in simulation mode.")


def motor_control_loop() -> None:
    while not motor_stop_event.is_set():
        if calibration_active_event.is_set():
            motor_controller.update(0)
            time.sleep(MOTOR_UPDATE_INTERVAL)
            continue

        if not calibration_manager.ready or steer_control_x is None or not encoder.available:
            motor_controller.update(0)
            time.sleep(MOTOR_UPDATE_INTERVAL)
            continue

        try:
            target = calibration_manager.map_steer_to_encoder(float(steer_control_x))
            current = encoder.read_angle(samples=3)
            diff = encoder_delta(target, current)
            if abs(diff) <= ENCODER_TOLERANCE_COUNTS:
                motor_controller.update(0)
            else:
                motor_controller.update(1 if diff > 0 else -1)
        except Exception as exc:
            print(f"Motor control error: {exc}")
            motor_controller.update(0)

        time.sleep(MOTOR_UPDATE_INTERVAL)

    motor_controller.stop_immediately()


def run_encoder_calibration(screen_range: Tuple[int, int]) -> None:
    if calibration_active_event.is_set():
        print("Calibration already in progress.")
        return
    if not encoder.available:
        print("AS5048A encoder not detected. Connect the sensor before calibrating.")
        return

    calibration_active_event.set()
    try:
        motor_controller.stop_immediately()
        print("\n=== Steering Calibration ===")
        print("1. Manually move the steering to the LEFT mechanical stop.")
        input("   Press Enter to record the left limit...")
        left = encoder.read_angle(samples=CALIBRATION_SAMPLES)

        print("2. Manually move the steering to the RIGHT mechanical stop.")
        input("   Press Enter to record the right limit...")
        right = encoder.read_angle(samples=CALIBRATION_SAMPLES)

        calibration_manager.save(left, right, screen_range)
        calibration_manager.update_screen_range(*screen_range)
        print(
            f"Calibration saved. Left={left} counts, Right={right} counts."
        )
    finally:
        calibration_active_event.clear()


# ---------------------------------------------------------------------------
# Helper utilities and pipeline initialisation
# ---------------------------------------------------------------------------

# Utility: clamp values between two bounds (used to keep overlays on-screen)
def clamp(val, minn, maxn):
    return max(min(val, maxn), minn)


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

motor_thread = Thread(target=motor_control_loop, daemon=True)
motor_thread.start()

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
        # draw red dot
        target = cmap0 if cam == 0 else cmap1
        offset_x = 0 if cam == 0 else w0

        #Red Dots:
        # cv2.circle(target, (px - offset_x, py), obstacle_dot_radius_px, obstacle_dot_color, -1)

    # Show combined view
    combined = np.hstack((cmap0, cmap1))

    
    
    # === 🟦 HORIZONTAL GAP ROUTE PLANNER ────────────────────────────────
    h_combined, w_combined = frame0.shape[0], frame0.shape[1] + frame1.shape[1]
    calibration_manager.update_screen_range(0, w_combined)

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
    if key == ord('c'):
        run_encoder_calibration((0, w_combined))
    elif key == 27:
        break

# === Cleanup ===
running = False
motor_stop_event.set()
cap0.release()
cap1.release()
cv2.destroyAllWindows()
motor_thread.join(timeout=1.0)
motor_controller.cleanup()
encoder.close()










# =============================================================================
# AS5048A ⇄ Jetson Nano Wiring Notes
# =============================================================================
# • Power the encoder from the Jetson Nano's 3.3 V rail (pin 1 or pin 17) and
#   connect grounds together (pin 6 is a convenient ground).
# • Use the SPI0 header (jetson-io "SPI1" in software):
#       AS5048A SCK  → Jetson pin 23 (SCLK)
#       AS5048A MISO → Jetson pin 21 (MISO)
#       AS5048A MOSI → Jetson pin 19 (MOSI)
#       AS5048A CSn  → Jetson pin 24 (SPI chip-select 0) or another free GPIO
#         (update the spidev bus/device numbers if a different CS is used).
# • Tie the AS5048A's VDD and VDDIO pins to the same 3.3 V supply.
# • Enable the SPI interface on the Jetson Nano with `sudo /opt/nvidia/jetson-io`
#   if it is not already active, then reboot.
# • The motor driver connections in this script assume:
#       PWM (enable) → pin 32 (PWM0)
#       Direction A  → pin 31
#       Direction B  → pin 33
#   Adjust the `MOTOR_*_PIN` constants near the top of the file if you wire to
#   different GPIOs or use an alternate H-bridge.
# =============================================================================

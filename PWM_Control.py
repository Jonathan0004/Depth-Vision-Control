# Outputs PWM on Pin 32 or Pin 33
# Ground is Pin 30

import os
import select
import sys
import termios
import time
import tty

import Jetson.GPIO as GPIO

# ===== User-tunable settings =====
PWM_OUTPUTS = (
    {"chip": "/sys/class/pwm/pwmchip3", "channel": 0, "pin": 32},
    {"chip": "/sys/class/pwm/pwmchip2", "channel": 0, "pin": 33},
)
PIN_HOLD_HIGH = 31      # physical BOARD pin kept high for entire runtime

FREQ_HZ = 5000          # PWM frequency (e.g., 5000 for 5 kHz)
TARGET_DUTY_PCT = 40.0   # final duty cycle in percent (e.g., 10.0 for 10%)
SOFT_START_S = 2.0       # time to ramp from 0% up to TARGET_DUTY_PCT (seconds)
RAMP_STEPS = 2          # number of steps in the soft-start ramp


# ===== Derived timing in sysfs units (nanoseconds) =====
def ns_period(freq_hz: float) -> int:
    return int(round(1e9 / float(freq_hz)))


def ns_duty_from_pct(period_ns: int, pct: float) -> int:
    pct = max(0.0, min(100.0, pct))
    return int(period_ns * (pct / 100.0))


def channel_path(chip: str, channel: int) -> str:
    return f"{chip}/pwm{channel}"


def wr(path, val):
    with open(path, "w") as f:
        f.write(str(val))


def ensure_channel(chip: str, channel: int) -> str:
    """Export the PWM channel if required and return its sysfs path."""
    ch_path = channel_path(chip, channel)
    if os.path.isdir(ch_path):
        return ch_path

    if not os.path.isdir(chip):
        raise FileNotFoundError(f"{chip} does not exist. Check which pwmchipN is present.")

    wr(f"{chip}/export", str(channel))
    for _ in range(200):
        if os.path.isdir(ch_path):
            break
        time.sleep(0.01)
    else:
        raise TimeoutError(f"pwm{channel} did not appear after export")

    return ch_path


def disable_channel(chip: str, channel: int):
    ch_path = channel_path(chip, channel)
    try:
        wr(f"{ch_path}/enable", "0")
    except OSError:
        pass


def configure_channel(cfg: dict, period_ns: int, duty_ns: int, do_soft_start: bool):
    chip = cfg["chip"]
    channel = cfg["channel"]
    ch_path = ensure_channel(chip, channel)

    # Always disable before (re)configuring
    try:
        wr(f"{ch_path}/enable", "0")
    except OSError:
        pass

    wr(f"{ch_path}/period", period_ns)
    if do_soft_start and SOFT_START_S > 0 and RAMP_STEPS > 0 and duty_ns > 0:
        wr(f"{ch_path}/duty_cycle", 0)
        step_sleep = SOFT_START_S / RAMP_STEPS
        for i in range(1, RAMP_STEPS + 1):
            dc = int(duty_ns * (i / RAMP_STEPS))
            wr(f"{ch_path}/duty_cycle", dc)
            time.sleep(step_sleep)
    else:
        wr(f"{ch_path}/duty_cycle", duty_ns)

    wr(f"{ch_path}/enable", "1")


def describe_channel(cfg: dict) -> str:
    chip = os.path.basename(cfg["chip"])
    return f"{chip}/pwm{cfg['channel']}"


PERIOD_NS = ns_period(FREQ_HZ)
TARGET_DUTY_NS = ns_duty_from_pct(PERIOD_NS, TARGET_DUTY_PCT)

pin_configured = False

# Configure GPIO pin 31 to stay high while the program is running
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(PIN_HOLD_HIGH, GPIO.OUT, initial=GPIO.HIGH)
pin_configured = True

current_idx = 0
configure_channel(PWM_OUTPUTS[current_idx], PERIOD_NS, TARGET_DUTY_NS, True)
print(
    f"{FREQ_HZ:,} Hz @ {TARGET_DUTY_PCT:.1f}% set on pin {PWM_OUTPUTS[current_idx]['pin']}"
    " (press 'n' to toggle, Ctrl+C to exit)."
)


def switch_channel():
    global current_idx
    next_idx = (current_idx + 1) % len(PWM_OUTPUTS)
    disable_channel(
        PWM_OUTPUTS[current_idx]["chip"], PWM_OUTPUTS[current_idx]["channel"]
    )
    configure_channel(PWM_OUTPUTS[next_idx], PERIOD_NS, TARGET_DUTY_NS, False)
    current_idx = next_idx
    print(
        f"Switched to {describe_channel(PWM_OUTPUTS[current_idx])} "
        f"on pin {PWM_OUTPUTS[current_idx]['pin']}"
    )


orig_termios = None
stdin_fd = sys.stdin.fileno() if sys.stdin.isatty() else None
if stdin_fd is not None:
    orig_termios = termios.tcgetattr(stdin_fd)
    tty.setcbreak(stdin_fd)

try:
    while True:
        if stdin_fd is not None:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.25)
            if rlist:
                ch = sys.stdin.read(1)
                if ch.lower() == "n":
                    switch_channel()
        else:
            time.sleep(0.25)
except KeyboardInterrupt:
    pass
finally:
    disable_channel(
        PWM_OUTPUTS[current_idx]["chip"], PWM_OUTPUTS[current_idx]["channel"]
    )
    if orig_termios is not None:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, orig_termios)
    if pin_configured:
        GPIO.output(PIN_HOLD_HIGH, GPIO.LOW)
    GPIO.cleanup()

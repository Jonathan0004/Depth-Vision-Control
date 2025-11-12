"""Initialize both PWM outputs by briefly enabling them."""

import os
import time

# ===== User-tunable settings =====
PWM_OUTPUTS = (
    {"chip": "/sys/class/pwm/pwmchip3", "channel": 0, "pin": 32},
    {"chip": "/sys/class/pwm/pwmchip2", "channel": 0, "pin": 33},
)

FREQ_HZ = 5000          # PWM frequency (e.g., 5000 for 5 kHz)
TARGET_DUTY_PCT = 40.0  # duty cycle percentage
INITIALIZE_DURATION_S = 0.5


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


def configure_channel(cfg: dict, period_ns: int, duty_ns: int):
    chip = cfg["chip"]
    channel = cfg["channel"]
    ch_path = ensure_channel(chip, channel)

    # Always disable before (re)configuring
    try:
        wr(f"{ch_path}/enable", "0")
    except OSError:
        pass

    wr(f"{ch_path}/period", period_ns)
    wr(f"{ch_path}/duty_cycle", duty_ns)
    wr(f"{ch_path}/enable", "1")


PERIOD_NS = ns_period(FREQ_HZ)
TARGET_DUTY_NS = ns_duty_from_pct(PERIOD_NS, TARGET_DUTY_PCT)

active_channels = []
try:
    for cfg in PWM_OUTPUTS:
        configure_channel(cfg, PERIOD_NS, TARGET_DUTY_NS)
        active_channels.append((cfg["chip"], cfg["channel"]))

    time.sleep(INITIALIZE_DURATION_S)
finally:
    for chip, channel in active_channels:
        disable_channel(chip, channel)

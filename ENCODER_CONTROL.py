# Outputs PWM on Pin 32 (pwmchip/pwm0)
# Ground is Pin 30

import os, time

import Jetson.GPIO as GPIO

# ===== User-tunable settings =====
PWMCHIP = "/sys/class/pwm/pwmchip3"  # adjust if your board exposes a different pwmchipN
CHANNEL = "pwm0"
PIN_HOLD_HIGH = 31      # physical BOARD pin kept high for entire runtime

FREQ_HZ = 8000          # PWM frequency (e.g., 5000 for 5 kHz)
TARGET_DUTY_PCT = 13.0   # final duty cycle in percent (e.g., 10.0 for 10%)
SOFT_START_S = 2.0       # time to ramp from 0% up to TARGET_DUTY_PCT (seconds)
RAMP_STEPS = 1          # number of steps in the soft-start ramp

# ===== Derived timing in sysfs units (nanoseconds) =====
def ns_period(freq_hz: float) -> int:
    return int(round(1e9 / float(freq_hz)))

def ns_duty_from_pct(period_ns: int, pct: float) -> int:
    pct = max(0.0, min(100.0, pct))
    return int(period_ns * (pct / 100.0))

CH_PATH = f"{PWMCHIP}/{CHANNEL}"

def wr(path, val):
    with open(path, "w") as f:
        f.write(str(val))

# Export channel if needed
if not os.path.isdir(CH_PATH):
    if not os.path.isdir(PWMCHIP):
        raise FileNotFoundError(f"{PWMCHIP} does not exist. Check which pwmchipN is present.")
    wr(f"{PWMCHIP}/export", "0")
    # Wait for pwm0 directory to appear
    for _ in range(200):
        if os.path.isdir(CH_PATH):
            break
        time.sleep(0.01)
    else:
        raise TimeoutError("pwm0 did not appear after export")

# Always disable before (re)configuring
try:
    wr(f"{CH_PATH}/enable", "0")
except OSError:
    pass

# Configure base frequency
PERIOD_NS = ns_period(FREQ_HZ)
wr(f"{CH_PATH}/period", PERIOD_NS)

# Start at 0% duty for soft-start
wr(f"{CH_PATH}/duty_cycle", 0)

# Enable output
wr(f"{CH_PATH}/enable", "1")

pin_configured = False

# Configure GPIO pin 31 to stay high while the program is running
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(PIN_HOLD_HIGH, GPIO.OUT, initial=GPIO.HIGH)
pin_configured = True

# Soft-start ramp to TARGET_DUTY_PCT
final_dc_ns = ns_duty_from_pct(PERIOD_NS, TARGET_DUTY_PCT)
if SOFT_START_S > 0 and RAMP_STEPS > 0 and final_dc_ns > 0:
    step_sleep = SOFT_START_S / RAMP_STEPS
    for i in range(1, RAMP_STEPS + 1):
        dc = int(final_dc_ns * (i / RAMP_STEPS))
        wr(f"{CH_PATH}/duty_cycle", dc)
        time.sleep(step_sleep)
else:
    wr(f"{CH_PATH}/duty_cycle", final_dc_ns)

print(f"{FREQ_HZ:,} Hz @ {TARGET_DUTY_PCT:.1f}% set. Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    try:
        wr(f"{CH_PATH}/enable", "0")
    except OSError:
        pass
    if pin_configured:
        GPIO.output(PIN_HOLD_HIGH, GPIO.LOW)
    GPIO.cleanup()

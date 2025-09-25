# This script creates a 50hz output with a 50% duty cycle
# Outputs PWM on Pin 32
# Ground is Pin 30

import os, time

PWMCHIP="/sys/class/pwm/pwmchip3"  # set to the one that shows on your pin
CH="pwm0"
PERIOD=20_000_000   # 50 Hz (nanoseconds)
DUTY=10_000_000     # 50% (nanoseconds)
PWM=f"{PWMCHIP}/{CH}"

def wr(p, v):
    with open(p, "w") as f:
        f.write(str(v))

# 1) Export if needed
if not os.path.isdir(PWM):
    # sanity check that the chip exists
    if not os.path.isdir(PWMCHIP):
        raise FileNotFoundError(f"{PWMCHIP} does not exist. Check which pwmchipN is present.")
    wr(f"{PWMCHIP}/export", "0")
    # wait for sysfs to create pwm0 directory
    for _ in range(100):
        if os.path.isdir(PWM):
            break
        time.sleep(0.01)
    else:
        raise TimeoutError("pwm0 did not appear after export")

# 2) Make sure it's disabled first (ignore if file not yet ready)
try:
    wr(f"{PWM}/enable", "0")
except OSError:
    # Some drivers EINVAL here before attributes exist; proceed to configure
    pass

# 3) Configure while disabled: period -> duty -> enable
wr(f"{PWM}/period", PERIOD)          # must exist before enabling
wr(f"{PWM}/duty_cycle", DUTY)        # must be <= period
wr(f"{PWM}/enable", "1")

print("50 Hz @ 50% running. Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    try:
        wr(f"{PWM}/enable", "0")
    except OSError:
        pass

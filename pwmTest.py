# esc_hw_pwm_50pct.py
# Uses Linux hardware PWM via /sys/class/pwm so timing is unaffected by CPU load.

import os, time, glob

PERIOD_NS   = 20_000_000   # 20 ms -> 50 Hz
NEUTRAL_NS  = 1_500_000    # 1.50 ms
HALF_FWD_NS = 1_750_000    # 1.75 ms

# Change these if your pwmchip/channel differ after jetson-io setup:
PWMCHIP = None
PWMCHAN = 0

# Try to auto-pick a pwmchip if not set
if PWMCHIP is None:
    chips = sorted(glob.glob("/sys/class/pwm/pwmchip*"))
    if not chips:
        raise SystemExit("No /sys/class/pwm/pwmchip* found. Enable PWM with jetson-io and reboot.")
    PWMCHIP = chips[0]  # pick the first; change if needed

pwm_path = f"{PWMCHIP}/pwm{PWMCHAN}"

def write(path, value):
    with open(path, "w") as f:
        f.write(str(value))

def ensure_exported():
    if not os.path.exists(pwm_path):
        write(f"{PWMCHIP}/export", PWMCHAN)

def main():
    ensure_exported()
    # Order matters: disable -> set period -> set duty -> enable
    write(f"{pwm_path}/enable", 0)
    write(f"{pwm_path}/period", PERIOD_NS)
    write(f"{pwm_path}/duty_cycle", NEUTRAL_NS)
    write(f"{pwm_path}/enable", 1)

    # Arm at neutral
    time.sleep(2.0)

    # Hold ~50% forward “forever”
    write(f"{pwm_path}/duty_cycle", HALF_FWD_NS)
    print("Outputting ~50% forward at 50 Hz via hardware PWM. Ctrl+C to stop.")
    while True:
        time.sleep(10)  # no busy loop needed; hardware keeps running

try:
    main()
except KeyboardInterrupt:
    pass
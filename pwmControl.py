#!/usr/bin/env python3
import os, sys, time

# Target: pin 32 = PWM0 -> typically /sys/class/pwm/pwmchip2/pwm0 on Nano
PWM_DIR   = "/sys/class/pwm/pwmchip2/pwm0"
PERIOD_NS = 20_000_000     # 20 ms
DUTY_NS   = 1_500_000      # 1.5 ms

def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)

def need_root():
    if os.geteuid() != 0:
        fail("Run with sudo: sudo python3 PWM_CONTROL.py")

def need_paths():
    req = ["enable", "period", "duty_cycle"]
    missing = [p for p in req if not os.path.exists(f"{PWM_DIR}/{p}")]
    if missing:
        fail(f"Missing PWM sysfs nodes: {missing}. Ensure PWM0 is exported and Jetson-IO enables pin 32.")

def rd(path):
    with open(path) as f:
        return f.read().strip()

def wr(path, val):
    with open(path, "w") as f:
        f.write(str(val))

def print_state(tag="state"):
    try:
        en = rd(f"{PWM_DIR}/enable")
        per = rd(f"{PWM_DIR}/period")
        duty = rd(f"{PWM_DIR}/duty_cycle")
        print(f"[{tag}] enable={en} period={per} duty={duty}")
    except Exception as e:
        print(f"[{tag}] read error: {e}")

def program_pwm():
    # 1) Validate inputs
    if DUTY_NS > PERIOD_NS:
        fail("DUTY_NS must be <= PERIOD_NS")

    # 2) Required nodes present
    need_paths()

    # 3) Program in safe order: period -> duty -> enable
    print_state("before")
    try:
        # Some drivers reject enable writes first; program values then enable
        wr(f"{PWM_DIR}/period", PERIOD_NS)
        wr(f"{PWM_DIR}/duty_cycle", DUTY_NS)
        wr(f"{PWM_DIR}/enable", 1)
    except OSError as e:
        # Show where it failed and current sysfs state
        print_state("on_error")
        fail(f"Write failed at {e.filename}: {e}")

    print_state("running")
    print("PWM running on pin 32. Press Ctrl+C to stop.")

def main():
    need_root()
    if not os.path.isdir(PWM_DIR):
        fail(f"{PWM_DIR} not found. Confirm the correct pwmchip index and that PWM0 is exported.")
    try:
        program_pwm()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        try:
            wr(f"{PWM_DIR}/enable", 0)
        except Exception:
            pass
        print("\nPWM disabled. Bye.")

if __name__ == "__main__":
    main()

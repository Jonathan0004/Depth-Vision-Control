# Outputs PWM on Pin 32 (pwmchip/pwm0)
# Ground is Pin 30

import os, time

# ===== Optional GPIO configuration =====
# Pin 31 on the 40-pin header typically maps to a GPIO line that can be driven
# as a digital output.  The default below targets the common Jetson Xavier NX
# mapping (physical pin 31 -> gpio298).  You can override this by setting the
# PIN31_GPIO environment variable to the correct Linux GPIO number for your
# board, or set it to an empty string to disable the behaviour entirely.
_pin31_override = os.environ.get("PIN31_GPIO")
if _pin31_override is None:
    GPIO_LINE = 298
elif _pin31_override == "":
    GPIO_LINE = None
else:
    GPIO_LINE = int(_pin31_override)
GPIO_ROOT = "/sys/class/gpio"

# ===== User-tunable settings =====
PWMCHIP = "/sys/class/pwm/pwmchip3"  # adjust if your board exposes a different pwmchipN
CHANNEL = "pwm0"

FREQ_HZ = 5000          # PWM frequency (e.g., 5000 for 5 kHz)
TARGET_DUTY_PCT = 100.0   # final duty cycle in percent (e.g., 10.0 for 10%)
SOFT_START_S = 2.0       # time to ramp from 0% up to TARGET_DUTY_PCT (seconds)
RAMP_STEPS = 2          # number of steps in the soft-start ramp

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

def wait_for_path(path: str, attempts: int = 200, delay_s: float = 0.01) -> bool:
    for _ in range(attempts):
        if os.path.isdir(path) or os.path.isfile(path):
            return True
        time.sleep(delay_s)
    return False

def setup_gpio_high():
    if GPIO_LINE is None:
        return lambda: None

    if not os.path.isdir(GPIO_ROOT):
        raise FileNotFoundError(f"{GPIO_ROOT} does not exist on this system")

    gpio_path = f"{GPIO_ROOT}/gpio{GPIO_LINE}"

    if not os.path.isdir(gpio_path):
        wr(f"{GPIO_ROOT}/export", GPIO_LINE)
        if not wait_for_path(gpio_path):
            raise TimeoutError(f"gpio{GPIO_LINE} did not appear after export")

    wr(f"{gpio_path}/direction", "out")
    wr(f"{gpio_path}/value", "1")

    def cleanup():
        try:
            wr(f"{gpio_path}/value", "0")
        except OSError:
            pass

    return cleanup

def main():
    gpio_cleanup = lambda: None

    try:
        gpio_cleanup = setup_gpio_high()

        # Export channel if needed
        if not os.path.isdir(CH_PATH):
            if not os.path.isdir(PWMCHIP):
                raise FileNotFoundError(f"{PWMCHIP} does not exist. Check which pwmchipN is present.")
            wr(f"{PWMCHIP}/export", "0")
            # Wait for pwm0 directory to appear
            if not wait_for_path(CH_PATH):
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

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            wr(f"{CH_PATH}/enable", "0")
        except OSError:
            pass

        gpio_cleanup()


if __name__ == "__main__":
    main()

# This script creates a 50hz output with a 50% duty cycle
# Outputs PWM on Pin 32
# Ground is Pin 30

import os, time

PWMCHIP="/sys/class/pwm/pwmchip3"  # set to the one that shows on your pin
CH="pwm0"
PERIOD=20_000_000   # 50 Hz
DUTY=10_000_000     # 50%

def wr(p, v):
    with open(p, "w") as f:
        f.write(str(v))

if not os.path.isdir(f"{PWMCHIP}/{CH}"):
    wr(f"{PWMCHIP}/export", "0")

try: wr(f"{PWMCHIP}/{CH}/enable", "0")
except FileNotFoundError: pass

wr(f"{PWMCHIP}/{CH}/period", PERIOD)
wr(f"{PWMCHIP}/{CH}/duty_cycle", DUTY)
wr(f"{PWMCHIP}/{CH}/enable", "1")

print("50 Hz @ 50% running. Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    wr(f"{PWMCHIP}/{CH}/enable", "0")

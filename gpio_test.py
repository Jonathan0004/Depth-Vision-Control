#example of how gpio will be controlled:
import Jetson.GPIO as GPIO
import time

# Set the GPIO pin number mode
GPIO.setmode(GPIO.BOARD)

# Define the pin
PIN = 23

# Set up the pin as an output
GPIO.setup(PIN, GPIO.OUT)

try:
    while True:
        GPIO.output(PIN, GPIO.HIGH)  # Turn ON
        time.sleep(1)                # Wait 1 second
        GPIO.output(PIN, GPIO.LOW)   # Turn OFF
        time.sleep(1)                # Wait 1 second

except KeyboardInterrupt:
    # Clean up on CTRL+C
    GPIO.cleanup()

finally:
    GPIO.cleanup()

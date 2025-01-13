import lgpio
import time
import math


# Pin configuration
PWM_PINL = 13# Left PWM pin
PWM_PINR = 12#Right PWM pin
MotorLPin1 = 5
MotorLPin2 = 6
LedPin = 0
MotorL1 = 0
MotorL2 = 1#High = forward

MotorRPin1 = 16
MotorRPin2 = 26
MotorR1 = 0
MotorR2 = 1#High = forward

PWM_FREQUENCY = 1000  # Frequency in Hz (e.g., 1000 Hz)

# Open a handle to the GPIO chip
h = lgpio.gpiochip_open(0)  # '0' usually corresponds to the main GPIO chip

# Init Left side
lgpio.gpio_claim_output(h, PWM_PINL)
lgpio.gpio_claim_output(h, MotorLPin1)
lgpio.gpio_write(h, MotorLPin1, MotorL1)
lgpio.gpio_claim_output(h, MotorLPin2)
lgpio.gpio_write(h, MotorLPin2, MotorL2)
#init Right side
lgpio.gpio_claim_output(h, PWM_PINR)
lgpio.gpio_claim_output(h, MotorRPin1)
lgpio.gpio_write(h, MotorRPin1, MotorR1)
lgpio.gpio_claim_output(h, MotorRPin2)
lgpio.gpio_write(h, MotorRPin2, MotorR2)
#init Led
lgpio.gpio_claim_output(h, LedPin)
# Start PWM on the pin
#lgpio.tx_pwm(h, PWM_PINR, PWM_FREQUENCY, 100)
#lgpio.tx_pwm(h, PWM_PINL, PWM_FREQUENCY, 100)  # 50% duty cycle


def radius_drive(radius, wanted_speed_outside, reversed):
    global current_speed_l, current_speed_r, current_speed_rear

    def math_sign(value):
        return (value > 0) - (value < 0)

    wheel_base_width = 20  # Example value, replace with the actual value
    wheel_base_length = 15  # Example value, replace with the actual value

    if reversed:
        outer_radius = radius + (wheel_base_width / 2)
        # current_speed_r = wanted_speed_outside
        current_speed_r = wanted_speed_outside
        degrees_per_second = wanted_speed_outside / (outer_radius * 2 * math.pi) * 360
        inner_radius = radius - (wheel_base_width / 2)
        current_speed_l = degrees_per_second * ((inner_radius * 2 * math.pi) / 360)
    else:
        outer_radius = radius + (wheel_base_width / 2)
        # current_speed_l = wanted_speed_outside
        current_speed_l = wanted_speed_outside
        degrees_per_second = wanted_speed_outside / (outer_radius * 2 * math.pi) * 360
        inner_radius = radius - (wheel_base_width / 2)
        current_speed_r = degrees_per_second * ((inner_radius * 2 * math.pi) / 360)
        
    #Skicka datan till motorerna
    print(current_speed_r, current_speed_l)
    if current_speed_l < 0:
        MotorL1 = 1
        MotorL2 = 0
    else:
        MotorL1 = 0
        MotorL2 = 1
    lgpio.gpio_write(h, MotorLPin1, MotorL1)
    lgpio.gpio_write(h, MotorLPin2, MotorL2)
    if current_speed_r < 0:
        MotorR1 = 1
        MotorR2 = 0
    else:
        MotorR1 = 0
        MotorR2 = 1
    lgpio.gpio_write(h, MotorRPin1, MotorR1)
    lgpio.gpio_write(h, MotorRPin2, MotorR2)
    lgpio.tx_pwm(h, PWM_PINR, PWM_FREQUENCY, abs(current_speed_r))
    lgpio.tx_pwm(h, PWM_PINL, PWM_FREQUENCY, abs(current_speed_l))
    lgpio.gpio_write(h, LedPin, 0)

radius_drive(100, 0, 1)
#lgpio.tx_pwm(h, PWM_PINR, PWM_FREQUENCY, 0)
#lgpio.tx_pwm(h, PWM_PINL, PWM_FREQUENCY, 0)
time.sleep(30)

lgpio.gpiochip_close(h)
print("GPIO closed.")


import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
import math

# Initialize I2C
i2c = busio.I2C(board.SCL, board.SDA)
bno = BNO08X_I2C(i2c)

def check_calibration():
    calib_status = bno.calibration_status
    print(f"Accelerometer Calibration: {calib_status}")

def quaternion_to_euler(x, y, z, w):
    # Calculate Roll
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    roll = math.degrees(roll)  # Convert to degrees

    # Calculate Pitch
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(90.0, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
        pitch = math.degrees(pitch)  # Convert to degrees

    # Calculate Yaw
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw = math.degrees(yaw)  # Convert to degrees

    return yaw, pitch, roll

# Main function
def main():
    bno.enable_feature(0x05)
    while True:
        quaternion = bno.quaternion
        x, y, z, w = quaternion
        yaw, pitch, roll = quaternion_to_euler(x, y, z, w)
        

        print(f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}")
        check_calibration()

# Run the main function
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program stopped.")

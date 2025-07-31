import serial
import time

def test_uart(port='/dev/ttyAMA0', baudrate=9600, timeout=1, test_message="Hello UART"):
    try:
        # Open serial port
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        time.sleep(2)  # Wait for UART to stabilize

        print(f"Sending: {test_message}")
        ser.write(test_message.encode())  # Send message

        time.sleep(0.5)  # Wait a bit for data to loop back

        received = ser.read(len(test_message))  # Read the same number of bytes
        print(f"Received: {received.decode()}")

        if received.decode() == test_message:
            print("? RX/TX loopback test passed.")
        else:
            print("? RX/TX loopback test failed.")

        ser.close()
    except Exception as e:
        print(f"?? Error: {e}")

if __name__ == "__main__":
    test_uart()

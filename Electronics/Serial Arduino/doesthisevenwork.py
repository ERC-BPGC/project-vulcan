

import serial  # pip install pyserial
import time

try:
    arduino = serial.Serial(port='COM37', baudrate=115200, timeout=5)  # replace COM4 by appropriate arduino port
except:
    print("arduino not connected issue")

time.sleep(1)
arduino.write(b'5')


time.sleep(1)
arduino.write(b'2')
time.sleep(1)
arduino.write(b'2')
time.sleep(1)
arduino.write(b'2')
time.sleep(1)
arduino.write(b'7')
time.sleep(1)
arduino.write(b'7')
time.sleep(1)
arduino.write(b'7')


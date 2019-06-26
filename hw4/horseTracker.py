import time
import serial
import requests

# configure the serial connections
ser = serial.Serial(port='/dev/ttyAMA0', baudrate=115200,timeout=None)
ser.close()
ser.open()
ser.isOpen()

f = open('gpsinfo.csv','w+')
f.write('Lat, Long\r')
f.flush()

# class GPS_coord:
#     def __init__(self):
#         self.lon = 0
#         self.lat = 0
line = ",,,,,,"
def check_gps():
    while line[0]=","
        input = 'AT+CGPSINFO'
    # send the character to the device
        ser.write(input + '\r\n')
    # let's wait one second before reading output
        time.sleep(3)
        while ser.inWaiting() > 0:
            for head in ser.read():
                if (head =='O'):
                    if(ser.read()==':'):
                        line = ser.readline()

    lat_clock = float(line[2:11])
    lat = int(line[0:2]) + lat_clock/60.0
    if line[12] == "S":
        lat = -lat

    lon_clock = float(line[17:26])
    lon = int(line[14:17]) + lon_clock/60.0
    if line[27] == 'W':
        lon = -lon
#    print lat, lon
    f.write(str(lat)+", "+str(lon)+'\r')
    f.flush()
    return lat, lon

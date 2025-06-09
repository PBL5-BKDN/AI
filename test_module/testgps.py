import serial
import pynmea2
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SERIAL_PORT = '/dev/ttyTHS1'  # Cổng UART Jetson Nano
BAUD_RATE = 115200

def read_gps():
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            logging.info(f"Đang kết nối GPS tại {SERIAL_PORT} tốc độ {BAUD_RATE}")
            while True:
                line = ser.readline().decode('ascii', errors='replace').strip()
                if line.startswith('$GPGGA'):
                    try:
                        msg = pynmea2.parse(line)
                        if msg.latitude and msg.longitude:
                            lat = msg.latitude
                            lon = msg.longitude
                            num_sats = msg.num_sats
                            logging.info(f"Kinh độ: {lon}, Vĩ độ: {lat}, Số vệ tinh: {num_sats}")
                    except pynmea2.ParseError as e:
                        logging.error(f"Lỗi phân tích NMEA: {e}")
                elif line.startswith('$GPRMC'):
                    try:
                        msg = pynmea2.parse(line)
                        if msg.latitude and msg.longitude:
                            lat = msg.latitude
                            lon = msg.longitude
                            logging.info(f"(RMC) Kinh độ: {lon}, Vĩ độ: {lat}")
                    except pynmea2.ParseError as e:
                        logging.error(f"Lỗi phân tích NMEA (RMC): {e}")
    except serial.SerialException as e:
        logging.error(f"Lỗi kết nối serial: {e}")
    except Exception as e:
        logging.error(f"Lỗi không xác định: {e}")

if __name__ == "__main__":
    read_gps()


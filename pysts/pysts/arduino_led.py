import json
import serial
import threading
import platform
import glob

from pysts.led_utils import ColorRGBA


class ArduinoLED:
    def __init__(self, config_dir, print_logs=False):
        self.print_logs = print_logs

        with open(f"{config_dir}/sts0_led.json") as fd:
            self._config = json.load(fd)
        with open(f"{config_dir}/tactile.json") as fd:
            config = json.load(fd)
            self._strip_length = config['strip_len']

        if platform.system() == 'Darwin':  # on MacOS, this changes every time we reconnect, so populate automatically
            self._port_name = glob.glob("/dev/tty.usbmodem*")[0]
        else:
            self._port_name = self._config['port_name']
        self._n_leds = self._config['n_leds']

        self.connect()
        self._set_thread_lock = threading.Lock()

    def connect(self):
        self._arduino = serial.Serial(self._port_name, baudrate=9600)
        self.log_print(f'connected to {self._port_name}')
        return True

    def set_led_values(self, r, g, b, blocking=True):
        return self.set_LEDs([ColorRGBA(r, g, b, 255)] * self._strip_length, blocking)

    def _set_LEDs(self, strip):
        with self._set_thread_lock:
            led_array_colors = [(c.r, c.g, c.b) for c in strip]

            self.log_print(f'led array colours {led_array_colors}')
            if len(led_array_colors) != self._n_leds:
                self.log_print(f'Sending wrong number of colours {len(led_array_colors)} != {self._n_leds}')

            led_bytes = bytes()
            q = str(self._n_leds).zfill(3)
            led_bytes += bytes(q,'ascii')

            for c in led_array_colors:
                for e in c:
                    q = str(int(e)).zfill(3)
                    led_bytes += bytes(q,'ascii')

            led_bytes += bytes('\n','ascii')
            self._arduino.write(led_bytes)
            while self._arduino.in_waiting > 0:
                q = self._arduino.read()

            return True

    def set_LEDs(self, strip, blocking=True):
        if blocking:
            return self._set_LEDs(strip)
        else:
            thread = threading.Thread(target=self._set_LEDs, args=(strip,))
            thread.daemon = True
            thread.start()
            return True

    def log_print(self, string):
        if self.print_logs:
            print(f"Arduino LED: {string}")
        else:
            pass
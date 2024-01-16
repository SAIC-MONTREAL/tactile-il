import sys
import signal
import platform

from pysts.camera import Camera, FakeCamera
from pysts.arduino_led import ArduinoLED
from pysts.led_static import get_pattern_strip
from pysts.camera_param import CameraParam
from pysts.mode import Mode


# TODO to improve camera parameter setting, get the list of parameters from v4l2-ctl automatically and use those


class PySTS:
    """ A class for running an STS without ROS. """
    def __init__(self, config_dir, camera_resolution="", pattern='white', cv2_for_params=True):
        print("Initializing PySTS.")

        self.cam = Camera(config_dir, camera_resolution)
        self.cam_param = CameraParam(config_dir,
            cv2_for_params=platform.system() == 'Darwin' or cv2_for_params, cv2_cap_obj=self.cam._camera)
        self.arduino_led = ArduinoLED(config_dir)

        strip = get_pattern_strip(config_dir, pattern)
        self.arduino_led.set_LEDs(strip, blocking=True)

        self.mode = Mode(config_dir, self.cam_param, self.arduino_led)

    def get_image(self, block_until_latest=True):
        return self.cam.get_image(block_until_latest)

    def set_led_values(self, r, g, b, blocking=True):
        self.arduino_led.set_led_values(r, g, b, blocking)

    def set_camera_parameters(self, params: dict, blocking=True, verify=True):
        self.cam_param.set_camera_parameters_dict(params, blocking, verify)

    def get_camera_parameters(self):
        return self.cam_param.get_camera_parameters()

    def set_mode(self, mode, blocking=True, verify=True):
        return self.mode.set_sts_state(mode, blocking, verify)


class SimPySTS:
    """ A class for running a simulated (i.e. from pre-recorded video) STS without ROS. """
    def __init__(self, source_vid, camera_resolution="", loop=True, rate=30):
        print("Initializing simulated PySTS.")
        self.cam = FakeCamera(source_vid, camera_resolution=camera_resolution, loop=loop, rate=rate)

    def get_image(self, block_until_latest=True):
        return self.cam.get_image()

    def set_mode(self, mode, blocking=True):
        print(f"Sim sensor mode set req: {mode}, not changing anything.")
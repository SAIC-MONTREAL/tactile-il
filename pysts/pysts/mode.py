import os
import warnings
import numbers

from sts.scripts.helper import read_json

from pysts.camera_param import CameraParam
from pysts.arduino_led import ArduinoLED


class Mode:
    def __init__(self, config_dir, camera_param: CameraParam, arduino_led: ArduinoLED):
        self._config_dir = config_dir
        config = read_json(os.path.join(self._config_dir, "sts_modes.json"))

        self._config = {}
        self._default_mode = config['default']

        for mode in config['modes']:
            # TODO this is deprecated, going to get rid of mode_dirs eventually
            # tactile directory is just the main config directory, all other modes in own directories...
            # matches sts-cam-ros2 for now
            if mode == 'tactile':
                mode_dir = ''
            else:
                mode_dir = mode

            # generate halfway mode
            if mode == 'halfway':
                self._config[mode] = {}
                v_settings = read_json(os.path.join(self._config_dir, 'visual.json'))
                t_settings = read_json(os.path.join(self._config_dir, 'tactile.json'))
                self._config[mode] = t_settings
                for k in t_settings:
                    if k in v_settings and k in t_settings and isinstance(t_settings[k], numbers.Number):
                        self._config[mode][k] = int(.5 * (v_settings[k] + t_settings[k]))
                break

            # check to see if mode_dir exists, otherwise fall back on old setup
            mode_json_file = os.path.join(self._config_dir, mode_dir, 'tactile.json')
            if os.path.exists(mode_json_file):
                self._config[mode] = read_json(os.path.join(self._config_dir, mode_dir, 'tactile.json'))
            else:
                print(f"Using {mode}.json file for settings for {mode} mode")
                self._config[mode] = read_json(os.path.join(self._config_dir, mode + '.json'))

        self.camera_param = camera_param
        self.arduino_led = arduino_led

    def set_sts_state(self, mode, blocking=True, verify=True):
        vals = self._config[mode]
        led_suc = self.arduino_led.set_led_values(vals["LED_red"], vals["LED_green"], vals["LED_blue"], blocking)

        cam_params = {}
        for k in vals:
            if k not in ("LED_red", "LED_green", "LED_blue"):
                cam_params[k] = vals[k]
        param_suc = self.camera_param.set_camera_parameters_dict(cam_params, blocking, verify=True)

        if verify:
            self.log_print(f"set to {mode}: successful? {led_suc and param_suc}")
            return led_suc and param_suc
        else:
            self.log_print(f"set to {mode}")
            return True

    def log_print(self, string):
        print(f"Mode: {string}")
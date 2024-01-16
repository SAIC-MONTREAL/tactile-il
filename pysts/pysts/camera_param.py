import json
import subprocess
import threading
import cv2


CV2_PROP_MAP = {
    'auto_exposure': cv2.CAP_PROP_AUTO_EXPOSURE,  # accommodate both types of cameras
    'exposure_auto': cv2.CAP_PROP_AUTO_EXPOSURE,
    'exposure_absolute': cv2.CAP_PROP_EXPOSURE,
    'exposure_time_absolute': cv2.CAP_PROP_EXPOSURE,
    'brightness': cv2.CAP_PROP_BRIGHTNESS,
    'contrast': cv2.CAP_PROP_CONTRAST,
    'saturation': cv2.CAP_PROP_SATURATION,
    'white_balance_automatic': cv2.CAP_PROP_AUTO_WB,
    'white_balance_temperature_auto': cv2.CAP_PROP_AUTO_WB,
}


class CameraParam:
    def __init__(self, config_dir, print_logs=False, cv2_for_params=False, cv2_cap_obj=None):
        """ If cv2_for_params is True, cam_obj must be set to the cv2 camera cap object. """

        self.print_logs = print_logs
        with open(f"{config_dir}/camera_parameters.json") as fd:
            self._keys = json.load(fd)
        with open(f"{config_dir}/tactile.json") as fd:
            self.tactile_config = json.load(fd)

        self._keys_str = ""
        for k in self._keys.keys():
            self._keys_str += f"{k},"
        self._keys_str = self._keys_str[:-1]  # remove final comma

        self._cv2_for_params = cv2_for_params
        self._cv2_cap_obj = cv2_cap_obj

        self.load_camera_parameters()
        self._set_thread_lock = threading.Lock()

    def load_camera_parameters(self):
        """Get current value of all parameters into key dictionary"""
        _args = ""
        invert = {}
        for k in self._keys:
            _args =  _args + self._keys[k]['cmd'] + ","
            invert[self._keys[k]['cmd']] = k
        _args = _args[:-1]
        self.log_print(f'command arguments {_args}')

        if self._cv2_for_params:
            for k in self._keys.keys():
                val = int(self._cv2_cap_obj.get(CV2_PROP_MAP[k]))
                self._keys[k]['val'] = val

        else:
            out = subprocess.Popen(
                ['v4l2-ctl', '-d', str(self.tactile_config['device']), '-C', self._keys_str],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, _ = out.communicate()
            stdout = stdout.decode("utf-8").splitlines()

            for z in stdout:
                self.log_print(f'Loaded param: {z}')
                q = z.split(':')
                iq = invert[q[0]]

                # handle params like exposure_auto, which can be set as "1 (Manual Mode)", and breaks int
                val = q[1].split(' ')[1]

                self._keys[iq]['val'] = int(val)

    def get_camera_parameters(self):
        self.load_camera_parameters()
        return self._keys

    def set_camera_parameters_dict(self, params: dict, blocking=True, verify=True):
        self.log_print(f"setting {params}")
        return self.set_camera_parameters_json(json.dumps(params, indent=4), blocking, verify)

    def _set_camera_parameters_json(self, json_file, verify=True):
        with self._set_thread_lock:
            self.log_print(f'{json_file}')
            cmds = json.loads(json_file)

            if self._cv2_for_params:
                cmd_rets = []
                for c in cmds:
                    if c in CV2_PROP_MAP:
                        self.log_print(f'setting {c} to {cmds[c]}')
                        suc = self._cv2_cap_obj.set(CV2_PROP_MAP[c], cmds[c])
                        cmd_rets.append(suc)
                    else:
                        self.log_print(f'unknown key {c}')
                if verify:
                    return all(cmd_rets)
            else:
                args = ""
                for c in cmds:
                    try:
                        v = cmds[c]
                        key = self._keys[c]['cmd']
                        args = args + f"{key}={int(v)},"
                    except KeyError:
                        self.log_print(f'unknown key {c}')
                args = args[:-1]
                self.log_print(f'sending {args}')

                out = subprocess.Popen(
                    ['v4l2-ctl', '-d', str(self.tactile_config['device']), '-c', args],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                _, _ = out.communicate()
            if verify:
                return self.get_camera_parameters()

    def set_camera_parameters_json(self, json_file, blocking=True, verify=True):
        if blocking:
            return self._set_camera_parameters_json(json_file, verify)
        else:
            thread = threading.Thread(target=self._set_camera_parameters_json, args=(json_file, verify))
            thread.daemon = True
            thread.start()
            if verify:
                return True

    def log_print(self, string):
        if self.print_logs:
            print(f"Camera Param: {string}")
        else:
            pass
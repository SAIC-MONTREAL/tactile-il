import sys
import os
import json
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sts.scripts.helper import read_json, write_json
from sts_interfaces.srv import SetCameraParameters, SetLEDs, SetSTSState
from sts_interfaces.msg import LEDArray, LEDStrip
from std_msgs.msg import ColorRGBA

class SetLEDsNode(Node):
    """Deal with LEDs"""
    def  __init__(self):
        super().__init__('set_sts_mode_led')

    def start_serving(self, config, mode):
        self._config = config
        self._set_leds = self.create_client(SetLEDs, self._config[mode]["led_srv"])
        while not self._set_leds.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'set_LEDs service not available, waiting...')

    def _ColorRGBA(r,g,b,a):
        """This constructor does not seem to exist in ros2 (yet)"""
        c = ColorRGBA()
        c.r = float(r)
        c.g = float(g)
        c.b = float(b)
        c.a = float(a)
        return c

    def _fill_strip(vals, pin=0):
        """Create a LEDArray of 1 strip of size len(vals) with the given values (all others black same length)"""
        n = len(vals)
        led_array = LEDArray()

        for led_strip_id in range(8):
            led_strip = LEDStrip()
            led_strip.colors = [SetLEDsNode._ColorRGBA(0,0,0,255)] * n
            if led_strip_id == pin:
                for i in range(len(vals)):
                    led_strip.colors[i] = vals[i]
            led_array.strips.append(led_strip)
        return led_array

    def _set_sts_state(self, request, response):
        self.get_logger().info(f'set state {request.request}')
        try:
            vals = self._config[request.request]
        except KeyError:
            self.get_logger().error(f'Mode {request.request} not in {self._config.keys()}')
            response.success = False
            return response
        self.get_logger().info(f'sts_mode_node: vals {vals}')

        self.set_led_values(vals["LED_red"], vals["LED_green"], vals["LED_blue"])
        self.get_logger().info(f'sts_mode_node: set leds')

    def set_strip_value(self, strip_length, r, g, b):
        """Set the strip to be white of a given colour"""
        strip = SetLEDsNode._fill_strip([SetLEDsNode._ColorRGBA(r, g, b, 255)] * strip_length)
        request = SetLEDs.Request()
        request.leds = strip
        self._future = self._set_leds.call_async(request)
        self.get_logger().info(f"Setting LED service call started strip len {strip_length} {r} {g} {b}")

    def get_strip_state(self):
        """Query the strip state"""
        return self._future

class SetCameraNode(Node):
    def  __init__(self):
        super().__init__('set_sts_mode_led')

    def start_serving(self, config, mode):
        self._config = config
        self._set_camera_parameters = self.create_client(SetCameraParameters, self._config[mode]["camera_srv"])
        while not self._set_camera_parameters.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_camera_parameter service not available, waiting...')

    def set_camera_parameters(self, v):
        """Set some (or all) of the camera parameters"""
        self.get_logger().info(f"set_camera_parameters called with {v}")
        req = SetCameraParameters.Request()
        req.request = json.dumps(v, indent=4)
        self._future = self._set_camera_parameters.call_async(req)
        self.get_logger().info(f"Camera parameters updated {v}")

    def get_camera_state(self):
        """Query the camera state"""
        return self._future


class SetSTSModeNode(Node):

    def __init__(self, led_server, camera_server):
        super().__init__('set_sts_mode_node')
        self.get_logger().info('set_sts_mode_node created')
        self.declare_parameter('config_dir', value = "/home/rbslab/panda_ros2_ws/src/sts-cam-ros2/configs/sts_circular")

        self._config_dir = self.get_parameter('config_dir').get_parameter_value().string_value

        tmp = os.path.join(self._config_dir, "sts_modes.json")
        try:
            config = read_json(tmp)
        except Exception as e:
            self.get_logger().error(f'Error reading/parsing config file {tmp}')
            sys.exit(0)

        self._config = {}
        self._default_mode = config['default']
        self.get_logger().info(f"Starting in mode {self._default_mode}")
        for mode in config['modes']:
            self.get_logger().info(f"Establishing mode {mode}")
            try:
                f = os.path.join(self._config_dir, mode + ".json")
                c = read_json(f)
            except Exception as e:
                self.get_logger().error(f'Error reading/parsing config file {tmp}')
                sys.exit(0)
            self._config[mode] = c

        self._led_server = led_server
        self._led_server.start_serving(self._config, config['default'])

        self._camera_server = camera_server
        self._camera_server.start_serving(self._config, config['default'])

        self.create_service(SetSTSState, "set_sts_state", self._set_sts_state, callback_group=ReentrantCallbackGroup())
        self.get_logger().info(f"Serving requests....")

    def spin(self):
        while rclpy.ok():
            rclpy.spin_once(self)

    def _set_camera_parameters(self, v):
        """Set some (or all) of the camera parameters"""
        self.get_logger().info(f"set_camera_parameters called with {v}")
        req = SetCameraParameters.Request()
        req.request = json.dumps(v, indent=4)
        res = self._set_camera_parameters.call(req)
        self.get_logger().info(f"Camera parameters updated {res}")

    def _set_sts_state(self, request, response):
        self.get_logger().info(f'set state {request.request}')
        try:
            vals = self._config[request.request]
        except KeyError:
            self.get_logger().error(f'Mode {request.request} not in {self._config.keys()}')
            response.success = False
            return response
        self.get_logger().info(f'sts_mode_node: vals {vals}')

        self._led_server.set_strip_value(vals["strip_len"], vals["LED_red"], vals["LED_green"], vals["LED_blue"])
        self.get_logger().info(f'sts_mode_node: set leds')
        while True:
            rclpy.spin_once(self._led_server)
            x = self._led_server.get_strip_state()
            self.get_logger().info(f"ending execution {x}")
            if x.done():
                break
            time.sleep(0.1)
 
        params = {}
        for k in vals:
            if k not in ("LED_red", "LED_green", "LED_blue"):
                params[k] = vals[k]
        self._camera_server.set_camera_parameters(params)
        while True:
            rclpy.spin_once(self._camera_server)
            x = self._camera_server.get_camera_state()
            self.get_logger().info(f"ending execution {x}")
            if x.done():
                break
            time.sleep(0.1)
        self.get_logger().info(f'sts_mode_node: set {params}')
        response.success = True
        return response

def main(args=None):
    rclpy.init(args=args)
    led_server = SetLEDsNode()
    camera_server = SetCameraNode()
    node = SetSTSModeNode(led_server, camera_server)
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()


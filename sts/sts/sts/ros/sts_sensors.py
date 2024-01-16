import json
import rclpy
from threading import Lock
from cv_bridge import CvBridge
from std_msgs.msg import ColorRGBA
from sts_interfaces.srv import SetCameraParameters, SetLEDs
from sts_interfaces.msg import LEDArray, LEDStrip, STSCompressedImage
    
class STS:
    """Encapsulate interactions with the STS hardware. This is somewhat different from the ros 1 version
       as we use service calls to manipulate various parameters (rather than setting parameters dynamically)"""

    def __init__(self, rclpy_node, config, sim=False):
        self._rclpy_node = rclpy_node
        self._image_lock = Lock()
        self._config = config

        self._bridge = CvBridge()
        self._img = None

        if not sim:
            self._subscriber = rclpy_node.create_subscription(STSCompressedImage, config["topic"], self._camera_callback, 1)
            self._set_camera_parameters = rclpy_node.create_client(SetCameraParameters, config["camera_srv"])
            while not self._set_camera_parameters.wait_for_service(timeout_sec=1.0):
                self._rclpy_node.get_logger().info('set_camera_parameter service not available, waiting...')
            self._parameter_request = SetCameraParameters.Request()

            self._set_leds = rclpy_node.create_client(SetLEDs, config["led_srv"])
            while not self._set_leds.wait_for_service(timeout_sec=1.0):
                self._rclpy_node.get_logger().info(f'set_LEDs service not available, waiting...')

            self._rclpy_node.get_logger().info(f'Connections made to camera and LEDs')

            self.load_config()
        else:
            self._subscriber = rclpy_node.create_subscription(STSCompressedImage, config["topic"], self._camera_callback, 1)

    def get_camera_parameters(self):
        """Get all of the camera parameters by changing its mode"""
        return self.set_camera_parameters({"auto_exposure" : 1 })  # auto_exposure true

    def set_camera_parameters(self, v):
        """Set some (or all) of the camera parameters"""
#        print(f"set camera parameters called with {v}")
        req = SetCameraParameters.Request()
        req.request = json.dumps(v, indent=4)
        future = self._set_camera_parameters.call_async(req)
        while rclpy.ok():
            rclpy.spin_once(self._rclpy_node)
            if future.done():
                try:
                    response = future.result()
                except Exception as e:
                    self._rclpy_node.get_logger().error(f"Error getting camera parameters {e}")
                    return None
                self._rclpy_node.get_logger().info(f"Got response {response.response}")
                return json.loads(response.response)

    def _camera_callback(self, msg):
        """Handle the camera callback"""
        img = self._bridge.compressed_imgmsg_to_cv2(msg.image)
        with self._image_lock:
            self._img = img

    def get_last_image(self):
        """ Get most recent image from the sensor """
        while self._img is None:
            rclpy.spin_once(self._rclpy_node, timeout_sec=0.1)
        with self._image_lock:
            return self._img

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
            led_strip.colors = [STS._ColorRGBA(0,0,0,255)] * n
            if led_strip_id == pin:
                for i in range(len(vals)):
                    led_strip.colors[i] = vals[i]
            led_array.strips.append(led_strip)
        return led_array

    def set_led_values(self, r, g, b):
        self.set_strip_value(self._config['strip_len'], r, g, b)

    def set_strip_value(self, strip_length, r, g, b):
        """Set the strip to be white of a given colour"""
        strip = STS._fill_strip([STS._ColorRGBA(r, g, b, 255)] * strip_length)
        request = SetLEDs.Request()
        request.leds = strip
        future = self._set_leds.call_async(request)
        while not future.done():
            rclpy.spin_once(self._rclpy_node)

    def load_config(self):
        self.set_led_values(self._config["LED_red"], self._config["LED_green"], self._config["LED_blue"])
        params = {}
        for k in self.get_camera_parameters():
            if k in self._config:
                params[k] = self._config[k]
        self.set_camera_parameters(params)




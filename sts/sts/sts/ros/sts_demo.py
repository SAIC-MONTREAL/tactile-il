#
# This does the entire data path for slip (the old 'process') code
# Hopefully this is stratighforward to read (ha)
# 
#
import sys
import os
import math
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sts_interfaces.msg import STSCompressedImage
from sts_interfaces.msg import STSFloat32MultiArray
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import MultiArrayLayout
from sts.scripts.helper import read_json, plot, split_channels
from sts.scripts.flow import STSOpticalFlow
from sts.scripts.slip import STSSlip
from sts.scripts.marker_detection.marker_detection import MarkerDetectionCreator
from sts.scripts.depth_from_markers import DepthDetector
from sts_interfaces.srv import SetSTSState

class EndToEndSlipDetectionNode(Node):
            
    def __init__(self, config_dir = "/home/ros2/ros2_ws/configs/sts_circular", display = True, image = "/sts/image", slip_image = "slip_image", slip_flow = "slip_flow"):
        super().__init__(f'end_to_end_slip_detectio_node')
        self.get_logger().info(f'slip_detection_node created')
        self.declare_parameter('config_dir', value = config_dir)
        self.declare_parameter('display', value = display)
        self.declare_parameter('image', value = image)
        self.declare_parameter('slip_image', value = slip_image)
        self.declare_parameter('slip_flow', value = slip_flow)
        config_dir = self.get_parameter('config_dir').get_parameter_value().string_value
        self._bridge = CvBridge()
        slip_image_topic  = self.get_parameter('slip_image').get_parameter_value().string_value
        slip_flow_topic  = self.get_parameter('slip_flow').get_parameter_value().string_value
        image_topic = self.get_parameter('image').get_parameter_value().string_value
        self._display = self.get_parameter("display").get_parameter_value().bool_value
        self._video_subscriber = self.create_subscription(STSCompressedImage, image_topic, self._image_listener_callback, 1)

        self.get_logger().info(f'image topic : {image_topic}')
#       set up for the optical flow stage
        self._config_object_flow = self._get_config(config_dir, "object_flow.json")
        self._config_marker_flow = self._get_config(config_dir, "marker_flow.json")
        self._image_flow = STSOpticalFlow(self._config_object_flow)
        self._marker_flow = STSOpticalFlow(self._config_marker_flow)
        self.mode = 'tactile'
        #self._slip_mask = cv2.imread(os.path.join(config_dir,'mask.png'))
        #self._slip_mask = self._slip_mask == 255

#       set up for the support multiple marker detection algorithms

        self.marker_detector = MarkerDetectionCreator(config_dir).create_object()
        self.depth_detector = DepthDetector(config_dir)

#      set up for the slip detection phase
        self._config_slip = self._get_config(config_dir, 'slip_detection.json')
        self._slip = STSSlip(self._config_slip)

        self.ref_img = cv2.imread(os.path.join(config_dir, 'reference.png'))
        self.markers_ref = cv2.cvtColor(cv2.imread(os.path.join(config_dir, 'markers_ref.png')), cv2.COLOR_BGR2GRAY)
        self.mask = cv2.cvtColor(cv2.imread(os.path.join(config_dir, 'mask.png')), cv2.COLOR_BGR2GRAY)
        self.mask_depth = cv2.cvtColor(cv2.imread(os.path.join(config_dir, 'mask.png')), cv2.COLOR_BGR2GRAY)
        self.mask = (self.mask / 255).astype('uint8')
        self.mask_depth = (self.mask_depth / 255).astype('uint8')

        self._slip_mask = np.stack([self.mask]*3, axis=2)

        self.sts_client = self.create_client(SetSTSState, '/sts/set_sts_state')
        self._set_mode('tactile')
        self.type = 'tactile'
        while not self.sts_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for service to set STS mode')

    def _get_config(self, config_dir, file):
        tmp = os.path.join(config_dir, file)
        try:
            config = read_json(tmp)
            return config
        except Exception as e:
            self.get_logger().error(f'Error reading/parsing config file {tmp}')
            sys.exit(0)

    def _set_mode(self, mode='tactile'):
       self.mode = mode
       req= SetSTSState.Request()
       req.request = mode
       future = self.sts_client.call_async(req)
       return True

    def _get_base_properties(self, msg):
        plot_dict = {}
        plot_dict['img'] = self._bridge.compressed_imgmsg_to_cv2(msg.image)
        return plot_dict

    def _get_slip_properties(self, msg):
        """Process an image end to end for slip"""
        plot_dict = {}
        img = self._bridge.compressed_imgmsg_to_cv2(msg.image)

        dict, centroids = self.marker_detector.detect(img)
        markers = dict["img"]
        marker_flow_img, MVx, MVy = self._marker_flow.detect(markers, centroids=centroids)
        flow_img, Vx, Vy = self._image_flow.detect(img)
        bgr_list = split_channels(img, 'BGR')

        #plot_dict['background'] = np.clip(cv2.subtract(img, self.ref_img) * self._slip_mask * 1, 0, 255)
        #plot_dict['img'] = img * self._slip_mask
        #plot_dict['zeros'] = np.zeros_like(img)
        plot_dict['image_flow'] = flow_img * self._slip_mask
        plot_dict['marker_flow'] = marker_flow_img * self._slip_mask
        
        is_slip, Dx, Dy = self._slip.detect(Vx, Vy, MVx, MVy, self.get_logger())
        if is_slip:
            self.get_logger().info(f'Slip Occurred')
        else:
            self.get_logger().info(f'No')

        slip_img = self._slip.get_slip_flow(img, Dx, Dy, type='quiver')
        slip_img = slip_img * self._slip_mask
        plot_dict['slip'] = slip_img
        return plot_dict 

    def _get_depth_properties(self, msg):
        """Process an image end to end for slip"""
        plot_dict = {}
        img = self._bridge.compressed_imgmsg_to_cv2(msg.image)

        dict, centroids = self.marker_detector.detect(img)
        markers = dict["img"]
        plot_dict['img'] = img
        plot_dict['markers'] = markers
        depth = self.depth_detector.get_marker_depth(markers, self.markers_ref, self.mask_depth)
        return plot_dict
 
    def _image_listener_callback(self, msg):
        #get properties
        if self.type in ['tactile', 'visual']:
            plot_dict = self._get_base_properties(msg)
            self._display = True
        if self.type=='depth':
            plot_dict = self._get_depth_properties(msg)
            self._display = True
        if self.type=='slip':
            plot_dict = self._get_slip_properties(msg)
            self._display = True

        if self._display:
            key = plot(plot_dict, plot_dict.keys())


        #check if change
        if key==ord('t'):
            self._set_mode('tactile')
            self.type = 'tactile'
            plot_dict = self._get_base_properties(msg)
        if key==ord('v'):
            self._set_mode('visual')
            self.type = 'visual'
            plot_dict = self._get_base_properties(msg)
        if key==ord('d'):
            self._set_mode('tactile')
            plot_dict = self._get_depth_properties(msg)
            self.type = 'depth'
        if key==ord('s'):
            self._set_mode('tactile')
            self.type = 'slip'
            plot_dict = self._get_slip_properties(msg)
            
def main(args=None):
    rclpy.init(args=args)
    try:
        node = EndToEndSlipDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()



import sys 
import os
import cv2
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.util import invert
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from PIL import Image as im

from sts_interfaces.msg import STSCompressedImage
from sts_interfaces.msg import STSFloat32MultiArray
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import MultiArrayLayout
from sts.scripts.depth_from_markers import DepthDetector
from sts.scripts.helper import read_json
from sts.scripts.marker_detection.marker_detection import MarkerDetectionCreator

class DepthEstimatorNode(Node):
    """Estimate depth from marker motion"""

    def __init__(self, config_dir="/root/sts-cam-ros2/configs/demo", display=True, marker_image_topic='/sts/marker_image', depth_image_topic='/sts/depth_image', depth_topic='/sts/depth'):
        super().__init__('depth_estimator_node')
        self.get_logger().info(f'depth_estimator node created')
        self.declare_parameter('config_dir', value=config_dir)
        self.declare_parameter('display', value=display)
        self.declare_parameter('depth_image_topic', value=depth_image_topic)
        self.declare_parameter('depth_topic', value=depth_topic)
        self.declare_parameter('marker_image_topic', value=marker_image_topic)

        config_dir = self.get_parameter('config_dir').get_parameter_value().string_value
        self._display = self.get_parameter("display").get_parameter_value().bool_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        depth_image_topic  = self.get_parameter('depth_image_topic').get_parameter_value().string_value
        marker_image_topic  = self.get_parameter('marker_image_topic').get_parameter_value().string_value
        self.detector = DepthDetector(config_dir)

        self._bridge = CvBridge()

        self._subscriber = self.create_subscription(STSCompressedImage, marker_image_topic, self._listener_callback, 1)
        self._depth_publisher = self.create_publisher(STSFloat32MultiArray, depth_topic, 1)
        self._depth_image_publisher = self.create_publisher(STSCompressedImage, depth_image_topic , 1)


    def _listener_callback(self, msg):
        img = self._bridge.compressed_imgmsg_to_cv2(msg.image)
        depth = self.detector.get_marker_depth(img)

        frame = img
        if self._display:
            frame = np.concatenate((cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), depth), axis=1)
            cv2.imshow('depth_output', frame)
            cv2.waitKey(10)

        depth_image_messages = self._bridge.cv2_to_compressed_imgmsg(depth)
        depth_image_messages.header.stamp = self.get_clock().now().to_msg()
        sts_compressed_image = STSCompressedImage()
        sts_compressed_image.image = depth_image_messages
        sts_compressed_image.capture_seq_num.data = msg.capture_seq_num.data
        sts_compressed_image.capture_time = msg.capture_time
        self._depth_image_publisher.publish(sts_compressed_image)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = DepthEstimatorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()

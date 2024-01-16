import sys
import os
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sts_interfaces.msg import STSCompressedImage
from sts_interfaces.msg import STSFloat32MultiArray
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import MultiArrayLayout

from sts.scripts.helper import read_json, plot

from sts.scripts.sts_transform import STSTransform
from sts.scripts.marker_detection.marker_detection import MarkerDetectionCreator
from sts.ros.sts_sensors import STS
from sts.scripts.sts_transform import STSTransform

class MarkerDetectionNode(Node):
    """Run one (of possibly many) different marker detection algorithms"""


    def __init__(self, config_dir = "/root/sts-cam-ros2/configs/demo", rostopic = "/sts/image", display=True, marker_image_topic="/sts/marker_image", marker_topic="/sts/markers", marker_dots_topic='/sts/marker_dots'):
        super().__init__('marker_node')
        self.get_logger().info(f'marker_node created')
        self.declare_parameter('config_dir', value = config_dir)
        self.declare_parameter('camera_topic', value = rostopic)
        self.declare_parameter('display', value = display)
        self.declare_parameter('marker_topic', value = marker_topic)
        self.declare_parameter('marker_image_topic', value = marker_image_topic)
        self.declare_parameter('marker_dots_topic', value = marker_dots_topic)

        config_dir = self.get_parameter('config_dir').get_parameter_value().string_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        marker_image_topic = self.get_parameter('marker_image_topic').get_parameter_value().string_value
        marker_dots_topic = self.get_parameter('marker_dots_topic').get_parameter_value().string_value
        self._marker_detector = MarkerDetectionCreator(config_dir).create_object()
        self._sts_transform = STSTransform(config_dir)
        self.get_logger().info(f'config_dir {config_dir}')
        self.get_logger().info(f'camera_topic {camera_topic}')
        self.get_logger().info(f'marker_image_topic {marker_image_topic}')
        self.get_logger().info(f'marker_topic {marker_topic}')
        self.get_logger().info(f'marker_dots_topic {marker_dots_topic}')

        self.mask = (cv2.imread(os.path.join(config_dir, 'mask.png')) / 255).astype('uint8')

        self._bridge = CvBridge()
        self._subscriber = self.create_subscription(STSCompressedImage, camera_topic, self._listener_callback, 1)

        self._display = self.get_parameter("display").get_parameter_value().bool_value

        self._marker_publisher = self.create_publisher(STSFloat32MultiArray, marker_topic, 1)

        self._mask_publisher = self.create_publisher(STSCompressedImage, marker_image_topic, 1)

        self._dots_publisher = self.create_publisher(STSCompressedImage, marker_dots_topic, 1)
        self.sts_transform = STSTransform(config_dir)


    def _listener_callback(self, msg):
        img = self._bridge.compressed_imgmsg_to_cv2(msg.image)

        self.transformed_img = self._sts_transform.transform(img)
        img_dict, vals = self._marker_detector.detect(self.transformed_img)
        vals, disp_image = self._marker_detector.filter_markers_kalman(self.transformed_img,  vals)
        centroid_img_only = self._marker_detector.img_from_centroids(0*img_dict['mask'], vals, color=[255,255,255])
        centroid_img_overlay = self._marker_detector.img_from_centroids(img_dict['mask'], vals, color=[0,0,255])
        img_dict['img'] = img
        img_dict['filtered_on_img'] = centroid_img_overlay
        img_dict['filtered'] = centroid_img_only
        img_dict['displacement'] = disp_image
        if self._display:
            plot(img_dict,  ['img', 'mask', 'centroids', 'displacement', 'filtered', 'filtered_on_img'])

        image_message = self._bridge.cv2_to_compressed_imgmsg(cv2.cvtColor(img_dict['filtered'], cv2.COLOR_BGR2GRAY))
        image_message.header.stamp = self.get_clock().now().to_msg()
        sts_compressed_image = STSCompressedImage()
        sts_compressed_image.image = image_message
        sts_compressed_image.capture_seq_num.data = msg.capture_seq_num.data
        sts_compressed_image.capture_time = msg.capture_time
        self._mask_publisher.publish(sts_compressed_image)

        image_message = self._bridge.cv2_to_compressed_imgmsg(centroid_img_only)
        image_message.header.stamp = self.get_clock().now().to_msg()
        sts_compressed_image = STSCompressedImage()
        sts_compressed_image.image = image_message
        sts_compressed_image.capture_seq_num.data = msg.capture_seq_num.data
        sts_compressed_image.capture_time = msg.capture_time
        self._dots_publisher.publish(sts_compressed_image)

        # populate a n x 2 float array
        array = []
        for pt in vals:
            array.append(pt[0])
            array.append(pt[1])
        markers = STSFloat32MultiArray()
        markers.capture_seq_num.data = msg.capture_seq_num.data
        markers.capture_time = msg.capture_time

        vals = Float32MultiArray()
        dim0 = MultiArrayDimension()
        dim0.label = "points"
        dim0.size = len(array)
        dim0.stride = dim0.size * 2
        dim1 = MultiArrayDimension()
        dim1.label = "xy"
        dim1.size = 2
        dim1.stride = 2
        layout = MultiArrayLayout()
        layout.dim = [dim0, dim1]
        layout.data_offset = 0
        vals.layout = layout
        vals.data = array
        markers.vals = vals
        self._marker_publisher.publish(markers)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = MarkerDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()

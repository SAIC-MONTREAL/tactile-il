import sys
import os
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
from sts.scripts.helper import read_json
from sts.scripts.flow import STSOpticalFlow


class FlowDetectionNode(Node):
    """Do flow detection"""

    def __init__(self, config_dir = "config", camera_topic = "image", display=True, outvideo_topic = 'flow_image', outflow_topic = 'flow'):
        super().__init__(f'flow_detection_node')
        self.get_logger().info(f'flow_detection_node created')
        self.declare_parameter('config_dir', value = config_dir)
        self.declare_parameter('camera_topic', value = camera_topic)
        self.declare_parameter('outvideo_topic', value = outvideo_topic)
        self.declare_parameter('outflow_topic', value = outflow_topic)

        self.declare_parameter('display', value = display)

        config_dir = self.get_parameter('config_dir').get_parameter_value().string_value
        tmp = os.path.join(config_dir, "object_flow.json")
        try:
            config = read_json(tmp)
        except Exception as e:
            self.get_logger().error(f'Error reading/parsing config file {tmp}')
            sys.exit(0)

        self._bridge = CvBridge()
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        outvideo_topic = self.get_parameter('outvideo_topic').get_parameter_value().string_value
        outflow_topic = self.get_parameter('outflow_topic').get_parameter_value().string_value
        self._video_subscriber = self.create_subscription(STSCompressedImage, camera_topic, self._listener_callback, 1)
        self._video_flow_publisher = self.create_publisher(STSCompressedImage, outvideo_topic, 1)
        self._flow_publisher = self.create_publisher(STSFloat32MultiArray, outflow_topic, 1)

        self._display = self.get_parameter("display").get_parameter_value().bool_value
        self._flow = STSOpticalFlow(config)


    def _listener_callback(self, msg):
        img = self._bridge.compressed_imgmsg_to_cv2(msg.image)

        flow_img, Vx, Vy = self._flow.detect(img)
        if self._display:
            cv2.imshow('Image', img)
            cv2.imshow('flow', flow_img)
            cv2.waitKey(10)

        image_message = self._bridge.cv2_to_compressed_imgmsg(flow_img)
        image_message.header.stamp = self.get_clock().now().to_msg()
        sts_compressed_image = STSCompressedImage()
        sts_compressed_image.image = image_message
        sts_compressed_image.capture_seq_num.data = msg.capture_seq_num.data
        sts_compressed_image.capture_time = msg.capture_time

        self._video_flow_publisher.publish(sts_compressed_image)


        # populate a 2 x n float array
        Vx = Vx.flatten()
        Vy = Vy.flatten()
        array = []
        for i in range(len(Vx)):
            array.append(Vx[i].item())
            array.append(Vy[i].item())
        flow = STSFloat32MultiArray()
        vals = Float32MultiArray()
        dim0 = MultiArrayDimension()
        dim0.label = "points"
        dim0.size = len(Vx)
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
        flow.vals = vals
        flow.capture_seq_num.data = msg.capture_seq_num.data
        flow.capture_time = msg.capture_time
        self._flow_publisher.publish(flow)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = FlowDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()



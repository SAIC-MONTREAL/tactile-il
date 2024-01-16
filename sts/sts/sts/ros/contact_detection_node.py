import cv2
import numpy as np

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sts.scripts.sts_transform import STSTransform
from sts.scripts.contact_detection.contact_detection import ContactDetectionCreator
from sts_interfaces.msg import STSCompressedImage
from sts.scripts.sts_transform import STSTransform

class ContactDetectionNode(Node):
    def __init__(self, config_dir="/root/sts-cam-ros2/configs/sts_rectangular", image_topic='/sts/image', contact_image_topic='/sts/contact_image', display=True):
        super().__init__('contact_detection_node')
        self._display = display
        self.get_logger().info(f'contact detection node created')

        self.declare_parameter('config_dir', value=config_dir)
        self.declare_parameter('image_topic', value=image_topic)
        self.declare_parameter('contact_image_topic', value=contact_image_topic)
        self.declare_parameter('display', value = display)

        self._config_dir = self.get_parameter('config_dir').get_parameter_value().string_value
        image_topic  = self.get_parameter('image_topic').get_parameter_value().string_value
        contact_image_topic  = self.get_parameter('contact_image_topic').get_parameter_value().string_value

        self.get_logger().info(f'config_dir {self._config_dir}')
        self.get_logger().info(f'image_topic {image_topic}')
        self.get_logger().info(f'contact_image_topic {contact_image_topic}')

        self._subscriber = self.create_subscription(STSCompressedImage, image_topic, self._listener_callback, 1)
        self._contact_publisher = self.create_publisher(STSCompressedImage, contact_image_topic, 1)
        self._bridge = CvBridge()

        self._display = self.get_parameter("display").get_parameter_value().bool_value
        self.sts_transform = STSTransform(self._config_dir)
        self.detector = ContactDetectionCreator(self._config_dir).create_object()

    def plot(self, img_list, name="contact"):
        frame = None
        for _img in img_list:
            if frame is None:
                frame = _img
            else:
                frame = np.concatenate((frame, _img), axis=1)

        cv2.imshow(name, frame)
        cv2.waitKey(10)

    def _listener_callback(self, msg):
        img = self._bridge.compressed_imgmsg_to_cv2(msg.image)

        transformed_image = self.sts_transform.transform(img)
        img_list = self.detector.get_channels(transformed_image)
        contact = self.detector.detect(transformed_image)

        if self._display:
            self.plot(img_list + [contact])

        image_message = self._bridge.cv2_to_compressed_imgmsg(contact)
        image_message.header.stamp = self.get_clock().now().to_msg()
        sts_compressed_image = STSCompressedImage()
        sts_compressed_image.image = image_message
        sts_compressed_image.capture_seq_num.data = msg.capture_seq_num.data
        sts_compressed_image.capture_time = msg.capture_time
        self._contact_publisher.publish(sts_compressed_image)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = ContactDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
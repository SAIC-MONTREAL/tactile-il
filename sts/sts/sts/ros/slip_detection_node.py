#
# This computes slip between two flow fields. It does this at the image level
# it subscribes to the marker image (used to mask results) and the tactile_flow and 
# marker_flow 2d flow fields. The output here is the subtracted field set to zero anywhere
# where the field corresponds to a marker. We also output a set of vectors that correspond to this
# flow
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
from sts.scripts.helper import read_json


class SlipDetectionNode(Node):
    """Do slip detection"""

    def __init__(self, config_dir = "config", display = True, marker_image = "marker_image", tactile_topic = 'tactile_flow', marker_topic = 'marker_flow'):
        super().__init__(f'flow_detection_node')
        self.get_logger().info(f'slip_detection_node created')
        self.declare_parameter('config_dir', value = config_dir)
        self.declare_parameter('marker_image', value = marker_image)
        self.declare_parameter('tactile_topic', value = tactile_topic)
        self.declare_parameter('marker_topic', value = marker_topic)

        self.declare_parameter('display', value = display)

        config_dir = self.get_parameter('config_dir').get_parameter_value().string_value
        tmp = os.path.join(config_dir, "slip_detection.json")
        try:
            config = read_json(tmp)
        except Exception as e:
            self.get_logger().error(f'Error reading/parsing config file {tmp}')
            sys.exit(0)

        self._images = {}
        self._markers = {}
        self._tactiles = {}
        self._bridge = CvBridge()
        image_topic = self.get_parameter('marker_image').get_parameter_value().string_value
        tactile_topic = self.get_parameter('tactile_topic').get_parameter_value().string_value
        marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        self._video_subscriber = self.create_subscription(STSCompressedImage, image_topic, self._image_listener_callback, 1)
        self._image_flow_subscriber = self.create_subscription(STSFloat32MultiArray, tactile_topic, self._tactile_flow_listener_callback, 1)
        self._marker_flow_subscriber = self.create_subscription(STSFloat32MultiArray, marker_topic, self._marker_flow_listener_callback, 1)

        self._display = self.get_parameter("display").get_parameter_value().bool_value
        self._maxv = 0

            
#
#   this is still not optimal
    def _check_match(self, slop=2):
        if self._images and self._tactiles and self._markers:
            oldest_image = min(self._images)
            oldest_tactile = min(self._tactiles)
            oldest_marker = min(self._markers)
            newest = max(oldest_image, oldest_tactile, oldest_marker)
#            self.get_logger().info(f"Oldest image {oldest_image} tactile {oldest_tactile} marker {oldest_marker} newest is {newest}")
            while self._images and (oldest_image < newest-slop):
                self._images.pop(oldest_image, None)
                if self._images:
                    oldest_image = min(self._images)
            while self._tactiles and (oldest_tactile < newest-slop):
                self._tactiles.pop(oldest_tactile, None)
                if self._tactiles:
                    oldest_tactile = min(self._tactiles)
            while self._markers and (oldest_marker < newest-slop):
                self._markers.pop(oldest_marker, None)
                if self._markers:
                    oldest_marker = min(self._markers)
            if self._images and self._tactiles and self._markers and abs(oldest_image-oldest_tactile)<=slop and abs(oldest_image-oldest_marker)<=slop:
                self.get_logger().info(f"Match {oldest_image} {oldest_tactile} {oldest_marker}")
                self._process(self._images[oldest_image], self._tactiles[oldest_tactile], self._markers[oldest_marker])
                self._images.pop(oldest_image, None) 
                self._tactiles.pop(oldest_tactile, None) 
                self._markers.pop(oldest_marker, None) 

    def _process(self, marker_image, tactile_flow, marker_flow):
        self.get_logger().info(f"Match {marker_image.image.header.stamp} {tactile_flow.capture_time} {marker_flow.capture_time}")
        tactile_dim = tactile_flow.vals.layout.dim
        marker_dim = marker_flow.vals.layout.dim
        if (len(tactile_dim) != 2) or (len(marker_dim) != 2):
            self.get_logger().error(f"Marker and tactile flow field dimensions incorrect {len(tactile_dim)} {len(marker_dim)}")
            sys.exit(1)
        if (tactile_dim[0].size != marker_dim[0].size) or (tactile_dim[1].size != marker_dim[1].size):
            self.get_logger().error(f"Marker and tactile flow field size mismatch")
            sys.exit(1)
        
        img = self._bridge.compressed_imgmsg_to_cv2(marker_image.image)
        if (tactile_dim[0].size * tactile_dim[1].size) != (img.shape[0] * img.shape[1] * 2):
            self.get_logger().error(f"Image and flow field sizes mismatch")
            sys.exit(1)


    #    result = np.array(tactile_flow.vals.data) - np.array(marker_flow.vals.data)
        result = np.array(tactile_flow.vals.data)
        result = result.reshape([img.shape[0], img.shape[1], 2])
        result2 = np.array(marker_flow.vals.data)
        result2 = result2.reshape([img.shape[0], img.shape[1], 2])

        """quiver plot"""
        spacing = 5
        margin = 0
        w = img.shape[1]
        h = img.shape[0]
        nx = int((w - 2 * margin) / spacing)
        ny = int((h - 2 * margin) / spacing)

        xs = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
        ys = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)

        out = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
                if img[y][x] > 0:
                  dx = result[y,x,0]-result2[y,x,0]
                  dy = result[y,x,1]-result2[y,x,1]
                  v = math.sqrt(dx * dx + dy * dy)
                  if v > self._maxv:
                      self._maxv =  v
                  iv = int(255*v/self._maxv)
                  out[y,x,:]=[iv,iv,iv]
       
        for x in xs:
            for y in ys:
                if img[y][x] > 0:
                    disp_x = 10 * result[y, x, 0]
                    disp_y = 10 * result[y, x, 1]
                    out = cv2.arrowedLine( out, (x, y), (int(x + disp_x), int(y + disp_y)), (0,0,255), 1)
                    disp_x = 10 * result2[y, x, 0]
                    disp_y = 10 * result2[y, x, 1]
                    out = cv2.arrowedLine( out, (x, y), (int(x + disp_x), int(y + disp_y)), (255,255,255), 1)

        cv2.imshow('flow', out)
        cv2.waitKey(10)


        
        


    def _tactile_flow_listener_callback(self, msg):
        self._tactiles[msg.capture_seq_num.data] = msg
        self._check_match()

    def _marker_flow_listener_callback(self, msg):
        self._markers[msg.capture_seq_num.data] = msg
        self._check_match()


    def _image_listener_callback(self, msg):
        self._images[msg.capture_seq_num.data] = msg
        self._check_match()

#        flow_img, Vx, Vy = self._flow.detect(img)
#        if self._display:
#            cv2.imshow('Image', img)
#            cv2.imshow('flow', flow_img)
#            cv2.waitKey(10)

#        image_message = self._bridge.cv2_to_compressed_imgmsg(flow_img)
#        image_message.header.stamp = self.get_clock().now().to_msg()
#        sts_compressed_image = STSCompressedImage()
#        sts_compressed_image.image = image_message
#        sts_compressed_image.capture_seq_num.data = msg.capture_seq_num.data
#        sts_compressed_image.capture_time = msg.capture_time
#
#        self._video_flow_publisher.publish(sts_compressed_image)


#        # populate a 2 x n float array
#        Vx = Vx.flatten()
#        Vy = Vy.flatten()
#        array = []
#        for i in range(len(Vx)):
#            array.append(Vx[i].item())
#            array.append(Vy[i].item())
#        flow = Float32MultiArray()
#        dim0 = MultiArrayDimension()
#        dim0.label = "points"
#        dim0.size = len(Vx)
#        dim0.stride = dim0.size * 2
#        dim1 = MultiArrayDimension()
#        dim1.label = "xy"
#        dim1.size = 2
#        dim1.stride = 2
#        layout = MultiArrayLayout()
#        layout.dim = [dim0, dim1]
#        layout.data_offset = 0
#        flow.layout = layout
#        flow.data = array
#        self._flow_publisher.publish(flow)
#

def main(args=None):
    rclpy.init(args=args)
    try:
        node = SlipDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()



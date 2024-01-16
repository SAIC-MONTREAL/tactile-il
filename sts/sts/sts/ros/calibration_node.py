import sys
import os
import json
import cv2
import numpy as np
import copy

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sts.scripts.helper import read_json, write_json
from sts.scripts.sts_transform import STSTransform
from sts.ros.sts_sensors import STS
from sts_interfaces.msg import STSCompressedImage
from sts.scripts.select_mask import CircularMask, RectangularMask
from sts.scripts.marker_detection.marker_detection import MarkerDetectionCreator
from sts.scripts.marker_detection.marker_detection_adaptive import MarkerDetectionAdaptive
from sts.scripts.contact_detection.contact_detection import ContactDetectionCreator

class CalibrateSTSNode(Node):
    MAX_DIM = 512
    def __init__(self, config_dir='/root/sts-cam-ros2/configs/demo'):
        super().__init__('calibration_node')
        self.get_logger().info(f'main_sts_node created')
        self.declare_parameter('config_dir', value=config_dir)
        self.declare_parameter('mode', value="tactile")
        self.declare_parameter('sim', value=True)
        self.declare_parameter('markers', value=False)
        self.declare_parameter('contact', value=True)

        self._config_dir = self.get_parameter('config_dir').get_parameter_value().string_value
        self._mode = self.get_parameter('mode').get_parameter_value().string_value
        self._sim = self.get_parameter('sim').get_parameter_value().bool_value
        self._markers = self.get_parameter('markers').get_parameter_value().bool_value
        self._contact = self.get_parameter('contact').get_parameter_value().bool_value


        self.get_logger().info(f'contact {self._contact}')
        self.get_logger().info(f'config_dir {self._config_dir}')

        tmp = os.path.join(self._config_dir, self._mode + ".json")
        try:
            self.get_logger().info(f'{tmp}')
            self._config = read_json(tmp)
        except Exception as e:
            self.get_logger().error(f'Error reading/parsing config file {tmp}')
            sys.exit(0)

        #initialize sts features
        self._sts = STS(self, self._config, sim=self._sim)
        self.sts_transform = STSTransform(self._config_dir)
        if self._mode == 'visual':
            self.marker_detector = MarkerDetectionCreator(self._config_dir).create_object(config_file_pre='visual_')
        else:
            self.marker_detector = MarkerDetectionCreator(self._config_dir).create_object()
        self.contact_detector = ContactDetectionCreator(self._config_dir).create_object()

        if not self._sim:
            self._camera_parameters = self._sts.get_camera_parameters()

        self.get_logger().info(f'{self._config}')
        self._led_names = {'LED_red', 'LED_green', 'LED_blue'}
        self._leds = {key: self._config[key] for key in self._led_names}


    def plot(self, img_list, name="markers"):
        frame = None
        for _img in img_list:
            if frame is None:
                frame = _img
            else:
                frame = np.concatenate((frame, _img), axis=1)

        cv2.imshow(name, frame)

    def detect_markers(self, img, scale, display=True):
        img_dict, vals = self.marker_detector.detect(self.transformed_img)
        vals, disp_image = self.marker_detector.filter_markers_kalman(self.transformed_img, vals)
        centroid_img = self.marker_detector.img_from_centroids(img_dict['mask'], vals)
        markers = centroid_img
        self.markers = img_dict['mask']
        img_list = self.marker_detector.get_channels(img) + [markers]
        self.plot(img_list, "markers")

    def detect_contact(self, img, scale, display=True):
        contact = self.contact_detector.detect(self.transformed_img)
        img_list = self.contact_detector.get_channels(self.transformed_img)
        self.plot(img_list + [contact], name="contact")

    def update_image(self, display=True):
        self._image = self._sts.get_last_image()
        self.transformed_img = self.sts_transform.transform(self._image)

        # otherwise erosion + other parameters won't actually match references correctly
        if type(self.marker_detector == MarkerDetectionAdaptive):
            im = self._image
            scale = None  # unused?
        else:
            scale = CalibrateSTSNode.MAX_DIM / max(self._image.shape[0], self._image.shape[1])
            im = cv2.resize(self._image, (int(scale * self._image.shape[1]), int(scale * self._image.shape[0])))

        if display:
            cv2.imshow("image", im)
        if self._markers:
            self.detect_markers(im, scale=scale, display=display)
        if self._contact:
            self.detect_contact(im, scale=scale, display=display)

    def _trackbar_callback(self, ignore):
        if not self._sim:
            params={}
            for k in self._camera_parameters:
                x = cv2.getTrackbarPos(k, "image")
                if x >= 0:
                    self._camera_parameters[k]["val"] = x
                    params[k] = x

            self._sts.set_camera_parameters(params)
            for k in self._leds:
                x = cv2.getTrackbarPos(k, "image")
                if x >= 0:
                    self._leds[k] = x
            self._sts.set_led_values(self._leds["LED_red"], self._leds["LED_green"], self._leds["LED_blue"])
        if self._markers:
            for k in self.marker_detector.calibration_params:
                x = cv2.getTrackbarPos(k, "markers")
                if x >= 0:
                    self.marker_detector.config[k] = x
        if self._contact:
            for k in self.contact_detector.calibration_params:
                x = cv2.getTrackbarPos(k, "contact")
                if x >= 0:
                    self.contact_detector.config[k] = x

    def create_image_and_sliders(self):
        self.update_image()
        if not self._sim:
            for k in self._camera_parameters.keys():
                p = self._camera_parameters[k]
                cv2.createTrackbar(k, "image", p["min"], p["max"], self._trackbar_callback)
                cv2.setTrackbarPos(k, "image", p["val"])
        for k in self._leds:
            cv2.createTrackbar(k, "image", 0, 255, self._trackbar_callback)
            cv2.setTrackbarPos(k, "image", self._leds[k])

        if self._markers:
            for k in self.marker_detector.calibration_params:
                p = self.marker_detector.config[k]

                if k=='invert':
                    cv2.createTrackbar(k, "markers", 0, 1, self._trackbar_callback)
                elif k=='channel':
                    cv2.createTrackbar(k, "markers", 0, 2, self._trackbar_callback)
                else:
                    cv2.createTrackbar(k, "markers", 0, 255, self._trackbar_callback)
                cv2.setTrackbarPos(k, "markers", int(p))
        if self._contact:
            for k in self.contact_detector.calibration_params:
                p = self.contact_detector.config[k]
                cv2.createTrackbar(k, "contact", 0, 255, self._trackbar_callback)
                cv2.setTrackbarPos(k, "contact", int(p))

    def merge_values(self):
        all = self._config.copy()
        if not self._sim:
            for k in self._leds:
                all[k] = self._leds[k]
            for k in self._camera_parameters:
                all[k] = self._camera_parameters[k]["val"]
        return all

    def save_config(self):
        all = self.merge_values()
        tmp = os.path.join(self._config_dir, self._mode + ".json")
        try:
            write_json(tmp, all)
        except Exception as e:
            self.get_logger().info(f'Error writing config file {tmp}')
        if self._contact:
            tmp = os.path.join(self._config_dir, self.contact_detector.config_file)
            try:
                write_json(tmp, self.contact_detector.config)
            except Exception as e:
                self.get_logger().info(f'Error writing config file {tmp}')


        if self._markers:
            mode_pre = ''
            tmp = os.path.join(self._config_dir, mode_pre + self.marker_detector.config['config_file'])
            try:
                write_json(tmp, self.marker_detector.config)
            except Exception as e:
                self.get_logger().info(f'Error writing config file {tmp}')

    def load_config(self):
        for k in self._leds:
            if k in self._config:
                cv2.setTrackbarPos(k, "image", self._leds[k])
        if not self._sim:
            params = {}
            for k in self._camera_parameters:
                if k in self._config:
                    cv2.setTrackbarPos(k, "image", self._config[k])


def main(args=None):
    rclpy.init(args=args)
    try:
        node = CalibrateSTSNode()
        node.create_image_and_sliders()
        node.load_config()
        cv2.waitKey(1)
        while rclpy.ok():
            rclpy.spin_once(node)
            node.update_image()
            key = chr(cv2.waitKey(1) & 0xff)
            if (key == 'q') or (key == 27):
                break
            elif key == 's':
                node.get_logger().info('saving config')
                node.save_config()
                print("Saving")
            elif key == 'r':
                node.get_logger().info('saving reference images')
                pre_img = copy.deepcopy(node._image)
                img_pre = ''
                if node._mode == 'visual':
                    img_pre = 'visual_'


                if node.sts_transform.config['mask']=='rectangular':
                    mask_selector = RectangularMask()
                if node.sts_transform.config['mask']=='circular':
                    mask_selector = CircularMask()

                mask_selector.select_mask(node._image, node._config_dir, img_pre=img_pre)
                cv2.destroyAllWindows()

                node.sts_transform = STSTransform(node._config_dir) #reload the new warp matrix
                node.transformed_image = node.sts_transform.transform(pre_img)

                cv2.imwrite(os.path.join(node._config_dir, img_pre + 'reference.png'), node.transformed_image)

                if node._markers:
                    dict, vals = node.marker_detector.detect(node.transformed_image)
                    cv2.imwrite(os.path.join(node._config_dir, img_pre + 'markers_reference.png'), dict['mask'])



    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()


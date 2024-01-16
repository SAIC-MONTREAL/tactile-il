import sys
import os
import json
import cv2
import numpy as np
import copy
import argparse

from pysts.sts import PySTS, SimPySTS

from sts.scripts.helper import read_json, write_json
from sts.scripts.sts_transform import STSTransform
from sts.scripts.select_mask import CircularMask, RectangularMask
from sts.scripts.marker_detection.marker_detection import MarkerDetectionCreator
from sts.scripts.marker_detection.marker_detection_adaptive import MarkerDetectionAdaptive
from sts.scripts.contact_detection.contact_detection import ContactDetectionCreator

class CalibrateSTS:
    MAX_DIM = 512
    def __init__(self, args):
        self.log_print(f'initialized')

        self._config_dir = args.config_dir
        self._mode = args.mode
        self._sim = args.source_vid != ''
        self._markers = not args.no_markers
        self._contact = args.contact
        self._kalman_markers = not args.no_kalman_markers

        self.log_print(f'contact {self._contact}')
        self.log_print(f'config_dir {self._config_dir}')
        self.log_print(f'mode {self._mode}')
        self._config = read_json(os.path.join(self._config_dir, self._mode + ".json"))

        #initialize sts features
        if self._sim:
            self._sts = SimPySTS(args.source_vid)
        else:
            self._sts = PySTS(self._config_dir)
        self.sts_transform = STSTransform(self._config_dir)
        self.marker_detector = MarkerDetectionCreator(self._config_dir).create_object()
        self.contact_detector = ContactDetectionCreator(self._config_dir).create_object()

        if not self._sim:
            self._camera_parameters = self._sts.get_camera_parameters()

        self.log_print(f'{self._config}')
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

        if len(vals) > 0 and self._kalman_markers:
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
        self._image = self._sts.get_image()
        self.transformed_img = self.sts_transform.transform(self._image)

        # otherwise erosion + other parameters won't actually match references correctly
        if type(self.marker_detector == MarkerDetectionAdaptive):
            im = self._image
            scale = None  # unused?
        else:
            scale = CalibrateSTS.MAX_DIM / max(self._image.shape[0], self._image.shape[1])
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

                # fix trackbar bug
                cv2.createTrackbar(k, "image", max(p["min"], 0), p["max"], self._trackbar_callback)
                if p["min"] < 0:
                    cv2.setTrackbarMin(k, "image", p["min"])

                cv2.setTrackbarPos(k, "image", p["val"])
        for k in self._leds:
            cv2.createTrackbar(k, "image", self._leds[k], 255, self._trackbar_callback)
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
            self.log_print(f'Error writing config file {tmp}')
        if self._contact:
            # doesn't use mode dir since parameters are shared!
            tmp = os.path.join(self._config_dir, self.contact_detector.config_file)
            try:
                write_json(tmp, self.contact_detector.config)
            except Exception as e:
                self.log_print(f'Error writing config file {tmp}')

        if self._markers:
            tmp = os.path.join(self._config_dir, self.marker_detector.config['config_file'])
            try:
                write_json(tmp, self.marker_detector.config)
            except Exception as e:
                self.log_print(f'Error writing config file {tmp}')

    def load_config(self):
        for k in self._leds:
            if k in self._config:
                cv2.setTrackbarPos(k, "image", self._leds[k])
        if not self._sim:
            params = {}
            for k in self._camera_parameters:
                if k in self._config:
                    cv2.setTrackbarPos(k, "image", self._config[k])

    def log_print(self, string):
        print(f"Calibration: {string}")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('config_dir', type=str, help="directory with sts config (same as used for sts-cam-ros2)")
    parser.add_argument('--source_vid', type=str, default='', help="Add if you're using a simulated/video sensor.")
    parser.add_argument('--mode', type=str, default="tactile", help="Mode to use for calibration.")
    parser.add_argument('--no_markers', action='store_true', help="Don't show marker calibration.")
    parser.add_argument('--contact', action='store_true', help="Whether to calibrate contact detection.")
    parser.add_argument('--no_kalman_markers', action='store_true', help="turn off kalman filtering on markers.")

    args = parser.parse_args()

    node = CalibrateSTS(args)
    node.create_image_and_sliders()
    node.load_config()
    cv2.waitKey(1)
    while True:
        node.update_image()
        key = chr(cv2.waitKey(1) & 0xff)
        if (key == 'q') or (key == 27):
            break
        elif key == 's':
            node.log_print('Saving config...')
            node.save_config()
            node.log_print("Config saved.")
        elif key == 'r':
            node.log_print('Saving reference images...')
            pre_img = copy.deepcopy(node._image)

            if node.sts_transform.config['mask']=='rectangular':
                mask_selector = RectangularMask()
            if node.sts_transform.config['mask']=='circular':
                mask_selector = CircularMask()

            mask_selector.select_mask(node._image, os.path.join(node._config_dir))
            cv2.destroyAllWindows()

            node.sts_transform = STSTransform(os.path.join(node._config_dir)) #reload the new warp matrix
            node.transformed_image = node.sts_transform.transform(pre_img)

            cv2.imwrite(os.path.join(node._config_dir, 'reference.png'), node.transformed_image)

            if node._markers:
                dict, vals = node.marker_detector.detect(node.transformed_image)
                cv2.imwrite(os.path.join(node._config_dir, 'markers_reference.png'), dict['mask'])

            node.log_print('Reference images saved.')

if __name__ == '__main__':
    main()


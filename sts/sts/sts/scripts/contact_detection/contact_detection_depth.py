import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from sts.scripts.contact_detection.contact_detection import ContactDetection
from sts.scripts.depth_from_markers import DepthDetector
from sts.scripts.marker_detection.marker_detection import MarkerDetectionCreator
from sts.scripts.helper import filter_image, read_json
from sts.scripts.sts_transform import STSTransform

class ContactDetectionDepth(ContactDetection):
    def __init__(self, config_dir):
        super().__init__()
        self.config_file = os.path.join(config_dir, "contact_detection_depth.json")
        config = read_json(self.config_file)

        self.sts_transform = STSTransform(config_dir)
        self.ref_img = cv2.imread(os.path.join(config_dir, 'reference.png'))
        self.config = config
        self.calibration_params = ["thresh"]
        self.mask = cv2.cvtColor(cv2.imread(os.path.join(config_dir, 'mask.png')), cv2.COLOR_BGR2GRAY)
        self.mask = (self.mask / 255).astype('uint8')
        self.depth_detector = DepthDetector(config_dir)
        self.marker_detector = MarkerDetectionCreator(config_dir).create_object()

    def get_channels(self, img):
        img_dict, vals = self.marker_detector.detect(img)
        depth = self.depth_detector.get_marker_depth(img_dict['mask'],  self.mask, display=False)
        return [cv2.cvtColor(img_dict['mask'], cv2.COLOR_GRAY2BGR), depth]

    def detect(self, img):
        """ marker_indices
        Segment contact area from image
        """

        img_dict, vals = self.marker_detector.detect(img)
        vals, disp_image = self.marker_detector.filter_markers_kalman(img,  vals)
        centroid_img_only = self.marker_detector.img_from_centroids(0*img_dict['mask'], vals, color=[255,255,255])
        depth = self.depth_detector.get_marker_depth(cv2.cvtColor(centroid_img_only, cv2.COLOR_BGR2GRAY), mask=self.mask, display=False)

        marker_mask = self.mask * img_dict['mask']
        marker_indices = np.where(marker_mask > 1)
        points = np.vstack(marker_indices).transpose()
        values = depth[marker_indices][:,2]


        x=np.linspace(0, marker_mask.shape[0], num=480)
        y=np.linspace(0, marker_mask.shape[1], num=640)
        grid_x, grid_y = np.meshgrid(x, y)
        interpolated_values = np.nan_to_num(griddata(points, values, (grid_x, grid_y), method='linear')).transpose().astype('uint8')

        #1. threshold dots
        indices = np.where(interpolated_values > self.config['thresh'])
        mask = np.zeros_like(interpolated_values)
        mask[indices] = 255
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)




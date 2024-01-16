import cv2, os
import numpy as np
from sts.scripts.marker_detection.marker_detection import MarkerDetection
from sts.scripts.helper import filter_image, read_json

class MarkerDetectionAdaptive(MarkerDetection):
    def __init__(self, config_dir):
        self.config = read_json(os.path.join(config_dir, "marker_segmentation_adaptive.json"))
        super().__init__(config_dir)
        self.calibration_params = ["block_size", "invert", "erosion", "channel"]

    def get_channels(self, img):
        img_list = filter_image(img, filter_type=self.config['type'])
        return img_list

    def detect(self, img):
        """ 
        Segment dots from image and extract centroid locations

        Use cv2 adaptive threshold instead.
        """

        if self.config['type']=='HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.config['type']=='BGR':
            pass
        elif self.config['type']=='LAB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        rgb_channels = cv2.split(img) 

        # block_size must be >= 3 and odd
        block_size = max(3, self.config["block_size"])
        if block_size % 2 == 0:
            block_size += 1
        img_mask = cv2.adaptiveThreshold(
            rgb_channels[self.config["channel"]], 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, block_size, 2)

        if self.config["invert"]:
            img_mask = ~img_mask

        erosion = self.config["erosion"]
        if erosion > 0:
            kernel_erosion = np.ones((erosion, erosion), np.uint8)
            dots_img = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel_erosion)
        else:
            dots_img = img_mask
        dots_img = cv2.GaussianBlur(dots_img, (3, 3), cv2.BORDER_DEFAULT)

        # 5. Extract centroid from image mask
        centroids = self.get_centroids(dots_img)
        centroid_img = self.img_from_centroids(dots_img, centroids)

        img_dict = {}
        img_dict["centroids"] = centroid_img
        img_dict["mask"] = dots_img

        return img_dict, np.array(centroids)



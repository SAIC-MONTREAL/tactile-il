import cv2, os
import numpy as np
from sts.scripts.marker_detection.marker_detection import MarkerDetection
from sts.scripts.helper import filter_image, read_json

class MarkerDetectionHSV(MarkerDetection):
    def __init__(self, config_dir):
        self.config = read_json(os.path.join(config_dir, "marker_segmentation_hsv.json"))
        super().__init__(config_dir)
        self.calibration_params = ["A_low", "A_high", "B_low", "B_high", "C_low", "C_high", "invert", "erosion"]

    def get_channels(self, img):
        img_list = filter_image(img, filter_type=self.config['type'])
        return img_list

    def detect(self, img):
        """ 
        Segment dots from image and extract centroid locations

          H_low, S_low, V_low and H_high, S_high and V_high define HSV range for markers
          erosion defines the width of the erosion stage (set to 0 for none)
          crop_radius B
        """

        if self.config['type']=='HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.config['type']=='BGR':
            pass
        elif self.config['type']=='LAB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        #from sts.helper import read_json, write_json, filter_image
        #bgr_list = filter_image(hsv_img, filter_type="BGR")
        #img_green = bgr_list[1]

        img_mask = cv2.inRange(img,
            (self.config["A_low"], self.config["B_low"], self.config["C_low"]),
            (self.config["A_high"], self.config["B_high"], self.config["C_high"]),
        )

        if self.config["invert"]:
            img_mask = ~img_mask

        # H_mask = cv2.cvtColor(img_mask.astype('uint8'), cv2.COLOR_GRAY2BGR)
        # 4. Morphological opening (to get rid of small dots)

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




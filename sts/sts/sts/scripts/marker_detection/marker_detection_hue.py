import cv2, os
import numpy as np
from sts.scripts.marker_detection.marker_detection import MarkerDetection
from sts.scripts.helper import read_json

class MarkerDetectionHUE(MarkerDetection):
    def __init__(self, config_dir):
        self.config = read_json(os.path.join(config_dir, "marker_segmentation_hue.json"))
        self.calibration_params = ["kernel", "crop_radius", "min_area", "erosion"]
        super().__init__(config_dir)

    def detect(self, img):

        kernel = (self.config["kernel"])
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel))

        (Blue, Green, Red) = cv2.split(img) 

        ret,BinaryFrameImage = cv2.threshold(Green,127,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU);

        # Clean up segmentation using morphological operations
        BinaryFrameImage = cv2.morphologyEx(BinaryFrameImage,cv2.MORPH_CLOSE,kernel5)
        BinaryFrameImage = cv2.morphologyEx(BinaryFrameImage,cv2.MORPH_OPEN,kernel5)
        BinaryFrameImage  = ~BinaryFrameImage  

        # 5. Extract centroid from image mask
        centroids = self.get_centroids(BinaryFrameImage)
        centroid_img = self.img_from_centroids(BinaryFrameImage, centroids)

        img_dict = {}
        img_dict["centroids"] = centroid_img
        img_dict["mask"] = BinaryFrameImage


        return img_dict, centroids

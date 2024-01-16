import os
import numpy as np
from sts.scripts.helper import read_json
from sts.scripts.marker_detection.marker_detection import MarkerDetection
from sts.scripts.unet_bridge import UNetInference

class MarkerDetectionUnet(MarkerDetection):
    def __init__(self, config_dir):
        self.config = read_json(os.path.join(config_dir, "marker_segmentation_unet.json"))
        super().__init__(config_dir)
        print("Marker detection unet constructor called")

        self.network = UNetInference(
            n_channels=config["n_channels"],
            n_classes=config["n_classes"],
            model=config["marker_model"],
            scale_factor=config["scale"],
        )


    def detect(self, img):
        """ Segment dots from image and extract centroid locations """
        pred = self.network.predict(img)
        pred = np.argmax(pred, axis=0)
        pred = pred.astype("uint8") * 255

        centroids = self.get_centroids(pred)
        centroid_img = self.img_from_centroids(pred, centroids)

        img_dict = {}
        img_dict["centroids"] = centroid_img
        img_dict["mask"] = pred


        return img_dict, centroids


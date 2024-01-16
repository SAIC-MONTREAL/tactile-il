import cv2
import os
import numpy as np
from sts.scripts.contact_detection.contact_detection import ContactDetection
from sts.scripts.helper import filter_image, read_json


class ContactDetectionBackSub(ContactDetection):
    def __init__(self, config_dir):
        super().__init__()
        self.config_file = os.path.join(config_dir, "contact_detection_back_sub.json")
        config = read_json(self.config_file)
        self.ref_img = cv2.imread(os.path.join(config_dir, 'reference.png'))
        self.config = config
        self.calibration_params = ["A_low", "A_high", "B_low", "B_high", "C_low", "C_high", "invert", "erosion"]

    def _get_gradients(self, img):

        img_height, img_width, _ = img.shape
        blur_k = self.config["blur_k"]
        grad_k = self.config['gradient_k']

        img_blur = cv2.GaussianBlur(img, (blur_k, blur_k), cv2.BORDER_DEFAULT)
        ref_blur = cv2.GaussianBlur(self.ref_img, (blur_k, blur_k), cv2.BORDER_DEFAULT)
        img_sub = np.clip(cv2.subtract(img_blur, ref_blur), 0, 255)

        img_sub = cv2.GaussianBlur(img_sub, (blur_k, blur_k), cv2.BORDER_DEFAULT)
        img_sub_gray = cv2.cvtColor(img_sub, cv2.COLOR_BGR2GRAY) 

        bgr = filter_image(img_sub, "BGR")
        #laplacian = cv2.sobel(img,cv2.CV_64F)
        sobelx = cv2.Sobel(img_sub, cv2.CV_64F, 1, 0, ksize=grad_k)[:, :, 0]
        sobely = cv2.Sobel(img_sub,cv2.CV_64F, 0, 1, ksize=grad_k)[:, :, 0]
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.convertScaleAbs(sobely)
        sobel = np.zeros((img_height, img_width, 2))

        sobel[:,:,0] = sobelx
        sobel[:,:,1] = sobely
        mag = np.linalg.norm(cv2.convertScaleAbs(sobel), axis=2).astype('uint8')
        mag_color =  cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
        return mag_color


    def get_channels(self, img):
        img_sub = np.clip(cv2.subtract(img, self.ref_img), 0, 255)
        mag_color = self._get_gradients(img)
        return [img, img_sub, mag_color]

    def detect(self, img):
        """ 
        Segment contact area from image
        """
        img_sub = np.clip(cv2.subtract(img, self.ref_img), 0, 255)
        img_sub_gray = cv2.cvtColor(img_sub, cv2.COLOR_BGR2GRAY)
        mag_color = self._get_gradients(img)
        mag = cv2.cvtColor(mag_color, cv2.COLOR_BGR2GRAY)
        ret,mask = cv2.threshold(mag,self.config["A_low"],255,cv2.THRESH_BINARY)
        erosion = self.config["erosion"]
        if erosion > 0:
            kernel_erosion = np.ones((erosion, erosion), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_erosion)
        else:
            mask = mask
 
#        bgr_list = filter_image(img_sub, filter_type="BGR")
#        blue_img = bgr_list[0]
#        mask = cv2.inRange(img_sub,
#            (self.config["A_low"], self.config["B_low"], self.config["C_low"]),
#            (self.config["A_high"], self.config["B_high"], self.config["C_high"]),
#        )
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)




import os
import cv2
import numpy as np
from sts.scripts.helper import read_json, mask_image, warp_image

class STSTransform(object):
    def __init__(self, config_dir, mode_dir=""):
        self._config_dir = os.path.join(config_dir, mode_dir)
        self._initialize_mask()
        self.ref_img = cv2.imread(os.path.join(self._config_dir, 'reference.png'))

    def _initialize_mask(self,):
        tmp = os.path.join(self._config_dir, 'tactile.json')
        tactile_config = read_json(tmp)
        mask = tactile_config['mask']
        if mask=='rectangular':
            tmp = os.path.join(self._config_dir, 'warp_matrix.json')
            self.config = read_json(tmp)
            self.config['mask'] = 'rectangular'
        elif mask=='circular':
            self.config = {'mask':'circular'}

    def transform(self, img):
        if self.config['mask']=='rectangular':
            return warp_image(img, self.config['M'])
        elif self.config['mask']=='circular':
            mask = cv2.cvtColor(cv2.imread(os.path.join(self._config_dir, 'mask.png')), cv2.COLOR_BGR2GRAY)
            mask = np.stack([mask], axis=2)
            mask = (mask / 255).astype('uint8')
            return mask_image(img, mask)
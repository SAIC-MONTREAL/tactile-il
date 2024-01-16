import cv2
import os
import time
from pysts.sts import PySTS


sts = PySTS(config_dir=os.environ['STS_CONFIG'], camera_resolution="640x480")

for i in range(50):
    img = sts.get_image()
    cv2.imshow('test', img)
    cv2.waitKey(1)
    # time.sleep(.1)
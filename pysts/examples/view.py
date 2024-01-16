import cv2
import os
import time
import argparse
from pysts.sts import PySTS


parser = argparse.ArgumentParser()
parser.add_argument('config_dir', type=str, help="sts config dir")
parser.add_argument('--mode', type=str, default='tactile', help="mode: visual or tactile")
parser.add_argument('--resolution', default='', type=str,
    help="resolution to use, defaults to source")

args = parser.parse_args()

sts = PySTS(config_dir=args.config_dir, camera_resolution=args.resolution)
sts.set_mode(args.mode)

while True:
    img = sts.get_image()
    cv2.imshow('test', img)
    cv2.waitKey(1)
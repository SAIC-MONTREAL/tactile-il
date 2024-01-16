import cv2
import os
import time
import argparse

from pysts.sts import SimPySTS


parser = argparse.ArgumentParser()
parser.add_argument('source', type=str, help="video file with sts data")
parser.add_argument('--resolution', default='', type=str,
    help="resolution to use, defaults to source")

args = parser.parse_args()

sts = SimPySTS(source_vid=args.source, camera_resolution=args.resolution)

while True:
    img = sts.get_image()
    cv2.imshow('test', img)
    cv2.waitKey(1)
import cv2
import os
import time
import argparse
from pysts.sts import PySTS

parser = argparse.ArgumentParser()
parser.add_argument('config_dir', type=str, default=os.environ['STS_CONFIG'], help="sts config dir")
parser.add_argument('--resolution', type=str, default='640x480', help="resolution to use, defaults to source")
parser.add_argument('--no_cv2_for_params', action='store_true', help="if set, use old v4l2-ctl way of setting params")
parser.add_argument('--block_on_switch', action='store_true', help="if set, block and verify on mode switch")

args = parser.parse_args()

sts = PySTS(config_dir=args.config_dir, camera_resolution=args.resolution, cv2_for_params=not args.no_cv2_for_params)

for i in range(300):
    start_img = time.time()
    img = sts.get_image()
    # print(f"IMG TIME: {time.time() - start_img}")
    cv2.imshow('test', img)
    cv2.waitKey(1)

    start_switch = time.time()
    if i % 50 == 0:
        if i % 100 == 0:
            if args.block_on_switch:
                suc = sts.set_mode('tactile', blocking=True, verify=True)
            else:
                sts.set_mode('tactile', blocking=False, verify=False)
        else:
            if args.block_on_switch:
                suc = sts.set_mode('visual', blocking=True, verify=True)
            else:
                sts.set_mode('visual', blocking=False, verify=False)
        print(f"MODE SWITCH TIME: {time.time() - start_switch}")

    # time.sleep(.1)
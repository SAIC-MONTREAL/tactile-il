import cv2
import os
import time
import argparse
import shutil
from datetime import datetime

from pysts.sts import PySTS
from pysts.processing import STSProcessor


parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', type=str, default=os.path.join(os.environ['STS_PARENT_DIR'], "sts-cam-ros2/configs/demo"),
    help="directory with sts config (same as used for sts-cam-ros2)")
parser.add_argument('--resolution', default="640x480", help="camera resolution")
parser.add_argument('--mode', type=str, default="tactile", help="Mode to record")
parser.add_argument('--length', type=float, default=15.0, help="Num seconds to record")
parser.add_argument('--rate', type=int, default="30", help="Rate of video")
parser.add_argument('--mode_switching', action='store_true', help="Use sts processor to include mode switching in vid.")

args = parser.parse_args()

if args.mode_switching:
    sts_p = STSProcessor(
    config_dir=args.config_dir,
    allow_both_modes=True,
    resolution=args.resolution,
    filter_markers='average',
    mode_switch_opts={
        'initial_mode': 'visual',
        'mode_switch_type': 'displacement',  # none (string), displacement, depth, contact, internal_ft
        'mode_switch_req_ts': 4,
        'tac_thresh': 0.5,
        'vis_thresh': 0.25,
        'tactile_mode_object_flow_channel': 'b',
        'mode_switch_in_thread': True,
    }
)
else:
    sts = PySTS(config_dir=args.config_dir, camera_resolution=args.resolution)
    sts.set_mode(args.mode)
imgs = []

input("Sensor initialized. Press enter to start recording, ctrl-c to stop...")

start = time.time()
while time.time() - start < args.length:
    if args.mode_switching:
        img = sts_p.get_processed_sts(modes={'raw_image'})['raw_image']
    else:
        img = sts.get_image()
    imgs.append(img)
    cv2.imshow('STS Record', img)
    cv2.waitKey(1)

cv2.destroyAllWindows()

# convert images into video
dir_name = './tmp-imgs'
vid_dir_name = './recordings'
os.makedirs(dir_name)
os.makedirs(vid_dir_name, exist_ok=True)
for i, img in enumerate(imgs):
    cv2.imwrite(os.path.join(dir_name, str(i).zfill(5) + ".png"), img)

date_str = datetime.now().strftime("%m-%d-%y-%H_%M_%S")
os.system(f"ffmpeg -r {args.rate} -i {dir_name}/%05d.png -pix_fmt yuv420p {vid_dir_name}/sts-{date_str}.mp4")
shutil.rmtree(dir_name)
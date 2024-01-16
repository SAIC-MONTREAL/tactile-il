import pickle
import os
import argparse

import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from place_from_pick_learning.datasets import MultimodalManipBCDataset
from contact_il.data.dict_dataset import DictDataset
import place_from_pick_learning.utils.debugging as debug
from pysts.utils import img_list_to_vid


# ---------- options ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('demo_dir', type=str, help="absolute path to dataset"),
parser.add_argument('ep', type=int, help="single/first ep to make video from"),
parser.add_argument('--last_ep', type=int,
    help="if defined, the last ep in a range of eps to make videos from"),
parser.add_argument('--data_key', type=str, default='sts_raw_image',
    help="data key to make video from"),

args = parser.parse_args()
main_dir = args.demo_dir
n_frame_stacks = 0
seq_length = 1
ep_range = None

# ----------------------------------------------------------------------------------------------
debug.nice_print()

vid_subdir_dict = dict(
    ep=os.path.join('data_videos', args.data_key),
    raw_ep=os.path.join('raw_data_videos', args.data_key),
)

# vid_subdir = os.path.join('data_videos', args.data_key)
# os.makedirs(os.path.join(args.demo_dir, vid_subdir), exist_ok=True)

ds = DictDataset(None, main_dir=main_dir, dataset_name="")
env_config_file = ds._env_parameters_file

ep_range = range(args.ep, args.last_ep) if args.last_ep is not None else range(args.ep, args.ep + 1)
for i in ep_range:
    if os.path.exists(os.path.join(main_dir, 'raw_demo_data')):
        ep, raw_ep = ds.load_ep(i, include_raw_demo=True)
        ep_dict = dict(ep=ep, raw_ep=raw_ep)
    else:
        ep_dict = dict(ep=ds.load_ep(i))

    for ep_type, ep in ep_dict.items():
        # make a dict with each key from obs + act
        data = dict(act=[])
        for k in ep[0][0]:
            if k == 'wrist_rgbd':
                data['wrist_rgb'] = []
                data['wrist_d'] = []
            data[k] = []

        # ep is a list of sas tuples, we instead want traj-len arrays of each data type
        for sas in ep:
            for k, v in sas[0].items():
                if k == 'wrist_rgbd':
                    data['wrist_rgb'].append(v[:, :, :3].astype('uint8'))
                    data['wrist_d'].append(v[:, :, 3])
                else:
                    data[k].append(v)
            data['act'].append(sas[1])

        vid_subdir = vid_subdir_dict[ep_type]
        os.makedirs(os.path.join(args.demo_dir, vid_subdir), exist_ok=True)
        img_list_to_vid(data[args.data_key], rate=10, top_dir=args.demo_dir,
            vid_dir_name=vid_subdir, vid_name=f"ep{i}", save_img_list=True)

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


# ---------- options ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('demo_dir', type=str, help="absolute path to dataset"),

args = parser.parse_args()
main_dir = args.demo_dir
# main_dir = os.path.join(os.environ['CIL_DATA_DIR'], "demonstrations")
# dataset_name = "poly-top-glass-orb-reset-rel-no-random"
# main_dir = os.path.join(os.environ['CIL_DATA_DIR'], "tests/model_2023-02-08_id=no_sts_19-19-15")
# dataset_name = "10-demo-door-no-sts-no-force-fix"
n_frame_stacks = 0
seq_length = 1
# ep_range = [0, 3]
ep_range = None

# ----------------------------------------------------------------------------------------------
debug.nice_print()

# ds = DictDataset(None, main_dir=main_dir, dataset_name=dataset_name)
ds = DictDataset(None, main_dir=main_dir, dataset_name="")
env_config_file = ds._env_parameters_file

# train_ds = MultimodalManipBCDataset(
#     data_dirs=[os.path.join(ds._dir, "data")],
#     seq_length=seq_length,
#     n_frame_stacks=n_frame_stacks,
#     env_config_file=env_config_file,
#     device='cpu')

# training_data = train_ds.build_cached_dataset()

all_replay_data = []
all_raw_data = []
# for i in range(len(ds)):
if ep_range is None:
    ep_range = [0, ds._params['actual_n_episodes']]
for i in range(*ep_range):
    # ep = ds.load_ep(i)
    replay_ep, raw_ep = ds.load_ep(i, include_raw_demo=True)

    for ep, all_data in [[replay_ep, all_replay_data], [raw_ep, all_raw_data]]:
        # make a dict with each key from obs + act
        data = dict(act=[])
        for k in ep[0][0]:
            data[k] = []

        # ep is a list of sas tuples, we instead want traj-len arrays of each data type
        for sas in ep:
            for k, v in sas[0].items():
                data[k].append(v)
            if len(sas) > 1:
                data['act'].append(sas[1])

        for k in data:
            data[k] = np.array(data[k])

        all_data.append(data)

import ipdb; ipdb.set_trace()

trans_fig = plt.figure()
ax = trans_fig.add_subplot(projection='3d')

for data in all_data:
    poses = data['pose']
    ax.plot(poses[:, 0],poses[:, 1], poses[:, 2])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("Positions")
ax.legend()

import ipdb; ipdb.set_trace()

plt.show()


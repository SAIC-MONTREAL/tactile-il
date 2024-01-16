import argparse
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

from contact_il.data.dict_dataset import DictDataset
from transform_utils.pose_transforms import PoseTransformer, matrix2pose


parser = argparse.ArgumentParser()
parser.add_argument('top_dir', type=str, help="absolute path to top level of dataset")
parser.add_argument('sub_dir_list', type=str, help="sub_dir dataset names as comma sep list")
parser.add_argument('--axes', type=str, default='z', help="string of axes to compare, given as sts axis.")
parser.add_argument('--plot_dir', type=str, default="/mnt/nvme_data_drive/t.ablett/datasets/contact-il/plots/sts_force")
args = parser.parse_args()


AXES_MAP = {
    'x': {'sts': 0, 'robot': 2},
    'y': {'sts': 1, 'robot': 0},
    'z': {'sts': 2, 'robot': 1},
    'a': {'sts': 5, 'robot': 4}  # a for yaw instead of y
}

# make fig for all data
fig, axes = plt.subplots(ncols=len(args.axes), figsize=(6, 4), tight_layout=True)
fig_sub_dir = "-".join(args.sub_dir_list.split(","))

# load data
sub_dir_list = args.sub_dir_list.split(",")
for sub_dir in sub_dir_list:
    data_dir = os.path.join(args.top_dir, sub_dir)
    pose, target, sts_force, in_contact = [], [], [], []
    ds = DictDataset(None, main_dir=data_dir, dataset_name="")
    sas_tups = ds.load_ep(0)

    # first pose -- will be used to rotate all poses into frame of first pose
    init_pose_mat_inv = np.linalg.inv(PoseTransformer(sas_tups[0][0]['pose']).get_matrix())

    for sas in sas_tups:
        state, act, _ = sas
        pose.append(init_pose_mat_inv @ PoseTransformer(state['pose']).get_matrix())
        target.append(init_pose_mat_inv @ PoseTransformer(state['target_pose']).get_matrix())
        sts_force.append(state['sts_avg_force'])
        in_contact.append(state['sts_in_contact'])

    pose = np.array(pose)
    target = np.array(target)
    sts_force = np.array(sts_force)
    in_contact = np.array(in_contact)
    pose_xyz = pose[:, :3, 3]
    target_xyz = target[:, :3, 3]
    pose_r_mat = pose[:, :3, :3]
    target_r_mat = target[:, :3, :3]

    pose_to_targ_vec = np.zeros([len(target_xyz), 6])
    pose_to_targ_vec[:, :3] = target_xyz - pose_xyz

    # to get rot diffs, need to convert to rvec first
    for i, (po, ta) in enumerate(zip(pose, target)):
        p_pt = PoseTransformer(matrix2pose(po))
        t_pt = PoseTransformer(matrix2pose(ta))
        diff_mat = np.linalg.inv(p_pt.get_matrix()) @ t_pt.get_matrix()
        pose_to_targ_vec[i, 3:] = PoseTransformer(matrix2pose(diff_mat)).get_rvec()

    for i, axis in enumerate(args.axes):
        axes[i].plot(pose_to_targ_vec[:, AXES_MAP[axis]['robot']], sts_force[:, AXES_MAP[axis]['sts']],
                     label=ds._params['experiment_name'])

        axes[i].legend(loc="best")
        if axis in 'xyz':
            axes[i].set_xlabel('pose - target (m)')
        else:
            axes[i].set_xlabel('pose - target (rad)')
        if i == 0:
            axes[0].set_ylabel('STS Force')

        axes[i].set_title(f"{axis}-axis")

os.makedirs(os.path.join(args.plot_dir, fig_sub_dir), exist_ok=True)
fig.savefig(os.path.join(args.plot_dir, fig_sub_dir, "force_fig.png"))
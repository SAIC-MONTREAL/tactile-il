import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle
import argparse
import os
import math
import numpy as np

from transform_utils.pose_transforms import (
    PoseTransformer,
    transform_local_body,
    transform_pose,
)
from control.utils.utils import geodesic_error

def open_pkl(save_path):
    with open(save_path, "rb") as handle:
        stored_grasps = pickle.load(handle)
    return stored_grasps

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dirs', nargs='+', required=True, help='List of directory names of datasets')
    args = parser.parse_args()
    return args

def get_success_failures(result_dirs):
    # Extract images for success / fail
    for result_dir in result_dirs: 
        episodes = [os.path.join(result_dir, v) for v in os.listdir(result_dir) if v.endswith('.pkl')]
    successes = []
    failures = []
    for e in episodes:
        tag = e.split("/")[-1]
        if tag.startswith("X"):
            failures.append(open_pkl(e))
        else:
            successes.append(open_pkl(e))
    return successes, failures

def step_pose(pose_t, a_t):
    translation, rotation = a_t[:3], a_t[3:-1]
    tr = PoseTransformer(pose=(*translation, 1, 0, 0, 0))
    pose_t1 = transform_pose(pose_t, tr)
    rot = PoseTransformer(
        pose=(0, 0, 0, *rotation), 
        rotation_representation="rvec"
    )
    pose_t1 = transform_local_body(pose_t1, rot)
    return pose_t1

def get_pose_error_norm(traj):
    ee_0 = PoseTransformer(
        pose=traj[0][0]["ee"], 
        rotation_representation="rvec"
    )
    ee_T = PoseTransformer(
        pose=traj[-1][0]["ee"], 
        rotation_representation="rvec"
    )

    ee_t = ee_0
    for t in traj:
        _, a_t, _ = t
        ee_t1 = step_pose(ee_t, a_t)
        ee_t = ee_t1
    ee_T_target = ee_t
    return np.sum(geodesic_error(ee_T, ee_T_target))


if __name__=="__main__":
    args = parse_args()

    # Read data
    successes, failures = get_success_failures(args.result_dirs)
    
    n_seq_tac = successes[0]["grasp_tac_img"].shape[0]

    # Plot grids
    for tup in [(failures, "fail"), (successes, "success")]:
        vs, name = tup

        fig = plt.figure(figsize=(32., 32.))
        grid = ImageGrid(
            fig, 
            111,  # similar to subplot(111)
            nrows_ncols=(len(vs), n_seq_tac),
            axes_pad=0.1,  # pad between axes in inch.
        )

        imgs = []
        for v in vs:
            for img in v["grasp_tac_img"]:
                imgs.append(img)
        for ax, v in zip(grid, imgs):
            ax.imshow(v)
            ax.set_xticks(np.arange(0, 640, 80))
            # pose_error_norm = get_pose_error_norm(v["data"])
            # ax.text(320, 250, f"{round(pose_error_norm, 3)}", fontsize=20, color='red')

        plt.savefig(f'tactile_images_{name}.png', bbox_inches="tight", dpi=200)
        plt.savefig(f'tactile_images_{name}.pdf', bbox_inches="tight")
import argparse
import copy
import os
import pickle
import time
from datetime import datetime
import glob
import pprint

import cv2
import numpy as np
import matplotlib.pyplot as plt

from contact_il.data.dict_dataset import DictDataset
from contact_il.imitation.device_utils import CollectDevice
from sts.scripts.helper import read_json, write_json
from pysts.utils import eul_rot_json_to_mat, eul_rot_to_mat
from place_from_pick_learning.utils.debugging import nice_print
from transform_utils.pose_transforms import PoseTransformer, matrix2pose


parser = argparse.ArgumentParser()
parser.add_argument('sts_config_dir', type=str, help="String for sts config dir")
parser.add_argument('calib_data_dir', type=str, help="Path to calibration data dir after main dir.")
parser.add_argument('--main_dir', type=str,
                    default="/mnt/nvme_data_drive/t.ablett/datasets/contact-il/experiments")
parser.add_argument('--plot_dir', type=str,
                    default="/mnt/nvme_data_drive/t.ablett/datasets/contact-il/plots/sts_force_calibration")
parser.add_argument('--fancy_plots', action='store_true')
parser.add_argument('--one_png_per_dot', action='store_true')

args = parser.parse_args()


##### Options ######
font_size = 14
flat_plot = True

if flat_plot:
    fig_size = [5, 2]
else:
    fig_size = [3, 5]

####################

if args.fancy_plots:
    pream = "\n".join([
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{amsfonts}",
        r"\usepackage{bbm}",
        r"\usepackage{mathtools}",
        ])
    plt.rcParams.update({"font.family": "serif", 'font.serif': ["Computer Modern Roman"], "text.usetex": True,
                         "pgf.rcfonts": False, "pgf.preamble": pream, "text.latex.preamble": pream})

nice_print(4)
STS_AXES_MAP = {'shear_x': 0, 'shear_y': 1, 'torque': 5, 'normal': 2}
# STS_AXES_MAP = {'shear_x': 0, 'shear_y': 1, 'normal': 2}
# STS_AXES_MAP = {'normal': 2}

################### SETUP #######################
sts_config_name = args.sts_config_dir.split('/')[-1]
fig_sub_dir = f"{args.sts_config_dir.split('/')[-1]}-{args.calib_data_dir.split('/')[-1]}"
full_plot_dir = os.path.join(args.plot_dir, fig_sub_dir)
os.makedirs(full_plot_dir, exist_ok=True)

params_file = os.path.join(args.sts_config_dir, "ft_adapted_replay.json")
params = read_json(params_file)
old_params = copy.deepcopy(params)
sts_to_ee_rot_mat = eul_rot_to_mat(params['sts_to_ee_eul_rxyz'])
large_sts_to_ee_rot_mat = np.eye(6)
large_sts_to_ee_rot_mat[:3, :3] = sts_to_ee_rot_mat
large_sts_to_ee_rot_mat[3:, 3:] = sts_to_ee_rot_mat

# we're going to grab all z trajs from all modes before calibrating normal
normal_pose_to_targ_vecs = []
normal_forces = []

################ LOOP THROUGH TRAJS #############
for calib_type_i, calib_type in enumerate(STS_AXES_MAP.keys()):
    if calib_type == 'normal': assert calib_type_i == len(STS_AXES_MAP) - 1, "normal must be last!!"
    traj_dirs = glob.glob(os.path.join(args.main_dir, args.calib_data_dir, calib_type) + "*")
    calib_type_pose_to_targ_vecs = []
    calib_type_forces = []
    for traj_dir in traj_dirs:
        print(f"Processing dataset at {traj_dir}")

        ds = DictDataset(None, main_dir=traj_dir, dataset_name="")
        sas_tups = ds.load_ep(0)
        pose, target, sts_force, acts = [], [], [], []

        # first pose -- will be used to rotate all poses into frame of first pose
        init_pose_mat_inv = np.linalg.inv(PoseTransformer(sas_tups[0][0]['pose']).get_matrix())

        for sas in sas_tups:
            state, act, _ = sas
            acts.append(act)
            pose.append(init_pose_mat_inv @ PoseTransformer(state['pose']).get_matrix())
            target.append(init_pose_mat_inv @ PoseTransformer(state['target_pose']).get_matrix())
            sts_force.append(state['sts_avg_force'])

        pose = np.array(pose)
        target = np.array(target)
        sts_force = np.array(sts_force)
        acts = np.array(acts)
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

        # rotate into sts frame
        pose_to_targ_vec_in_sts = (large_sts_to_ee_rot_mat @ pose_to_targ_vec.T).T

        # add any that were normal only movement to the normal lists
        acts_in_sts = (large_sts_to_ee_rot_mat @ acts.T).T
        z_only_acts = acts_in_sts[:, 2] > 1e-4  # careful, will only work with tool-relative actions!!

        normal_forces.append(sts_force[z_only_acts, 2])
        normal_pose_to_targ_vecs.append(pose_to_targ_vec_in_sts[z_only_acts, 2])

        if calib_type != "normal":
            axis = STS_AXES_MAP[calib_type]
            non_z_acts = np.invert(z_only_acts)

            # reverse any that were done in negative direction
            if acts_in_sts[non_z_acts, axis].mean() < 0:
                calib_type_forces.append(-sts_force[non_z_acts, axis])
                calib_type_pose_to_targ_vecs.append(-pose_to_targ_vec_in_sts[non_z_acts, axis])
            else:
                calib_type_forces.append(sts_force[non_z_acts, axis])
                calib_type_pose_to_targ_vecs.append(pose_to_targ_vec_in_sts[non_z_acts, axis])

    if calib_type == "normal":  # last in order, otherwise this wouldn't work
        calib_type_forces = normal_forces
        calib_type_pose_to_targ_vecs = normal_pose_to_targ_vecs

    calib_type_forces.append(np.atleast_1d(0))
    calib_type_pose_to_targ_vecs.append(np.atleast_1d(0))
    calib_type_forces = np.concatenate(calib_type_forces)
    calib_type_pose_to_targ_vecs = np.concatenate(calib_type_pose_to_targ_vecs)

    # now do linear fit and save params -- x axis dist change, y axis force
    # we're using a hack to force it to have y-intercept of 0
    weights = np.ones_like(calib_type_forces)
    weights[-1] = 1000

    m, b = np.polyfit(calib_type_pose_to_targ_vecs, calib_type_forces, 1, w=weights)

    params[calib_type]['m_0'] = m
    params[calib_type]['b_0'] = 0.0

    if calib_type == 'torque':
        params[calib_type]['max_dist'] = min(params[calib_type]['max_calib_dist'],
                                             180 / np.pi * calib_type_pose_to_targ_vecs.max())
    else:
        params[calib_type]['max_dist'] = min(params[calib_type]['max_calib_dist'], calib_type_pose_to_targ_vecs.max())

    if calib_type == "normal":
        # two separate linear fits in this case
        x_trans_point = params[calib_type]['x_trans_point']
        l1_indices = calib_type_pose_to_targ_vecs < x_trans_point
        l2_indices = np.invert(l1_indices)

        weights = np.ones_like(calib_type_pose_to_targ_vecs[l1_indices])
        weights[-1] = 1000
        m_0, b_0 = np.polyfit(calib_type_pose_to_targ_vecs[l1_indices], calib_type_forces[l1_indices], 1, w=weights)
        params[calib_type]['m_0'] = m_0
        params[calib_type]['b_0'] = 0.0

        # force lines to intercept at x trans point
        y_trans_point = m_0 * x_trans_point + params[calib_type]['b_0']

        calib_type_forces = np.concatenate([calib_type_forces, np.array([y_trans_point])])
        calib_type_pose_to_targ_vecs = np.concatenate([calib_type_pose_to_targ_vecs, np.array([x_trans_point])])
        l2_indices = np.concatenate([l2_indices, np.array([True])])
        weights = np.ones_like(calib_type_pose_to_targ_vecs[l2_indices])
        weights[-1] = 1000

        m_1, b_1 = np.polyfit(calib_type_pose_to_targ_vecs[l2_indices], calib_type_forces[l2_indices], 1, w=weights)
        params[calib_type]['m_1'] = m_1
        params[calib_type]['b_1'] = b_1

    else:
        l1_indices = np.ones_like(calib_type_forces)

    if not args.one_png_per_dot:
        # generate plot
        fig, ax = plt.subplots(figsize=fig_size, tight_layout=True)

        ax.scatter(calib_type_pose_to_targ_vecs, calib_type_forces, c='k', s=2)

        # generate line data
        if calib_type == "normal":
            # m0x + b0 = m1x + b1 --> x = (b1 - b0) / (m0 - m1)
            x_transition_val = (params[calib_type]['b_1'] - params[calib_type]['b_0']) / \
                (params[calib_type]['m_0'] - params[calib_type]['m_1'])

            if x_transition_val < 0:
                print("transition between lines not >0, something's wrong...")
                import ipdb; ipdb.set_trace()

            x_vals = np.linspace(0, x_transition_val, 100)
            y_vals = params[calib_type]['m_0'] * x_vals
            ax.plot(x_vals, y_vals, linewidth=3)

            x_vals = np.linspace(x_transition_val, params[calib_type]['max_dist'], 100)
            y_vals = params[calib_type]['m_1'] * x_vals + params[calib_type]['b_1']
            ax.plot(x_vals, y_vals, linewidth=3)

        else:
            if calib_type == 'torque':
                x_vals = np.linspace(0, params[calib_type]['max_dist'] * np.pi / 180, 100)
            else:
                x_vals = np.linspace(0, params[calib_type]['max_dist'], 100)
            y_vals = params[calib_type]['m_0'] * x_vals
            ax.plot(x_vals, y_vals, linewidth=3)

        if args.fancy_plots:
            # map = {'normal': 'z', 'shear_x': 'x', 'shear_y': 'y', 'torque': 'z'}

            if calib_type == 'normal':
                ax.set_ylabel(r"$\Tilde{\boldsymbol{\mathcal{F}}}^{(z\text{-ax})}$", fontsize=font_size)
                ax.set_xlabel(r"$\boldsymbol{\mathbf{e}}^{(z\text{-ax})}$ (m)", fontsize=font_size)
            elif calib_type == 'shear_x':
                ax.set_ylabel(r"$\Tilde{\boldsymbol{\mathcal{F}}}^{(x\text{-ax})}$", fontsize=font_size)
                ax.set_xlabel(r"$\boldsymbol{\mathbf{e}}^{(x\text{-ax})}$ (m)", fontsize=font_size)
            elif calib_type == 'shear_y':
                ax.set_ylabel(r"$\Tilde{\boldsymbol{\mathcal{F}}}^{(y\text{-ax})}$", fontsize=font_size)
                ax.set_xlabel(r"$\boldsymbol{\mathbf{e}}^{(y\text{-ax})}$ (m)", fontsize=font_size)
            elif calib_type == 'torque':
                ax.set_ylabel(r"$\Tilde{\boldsymbol{\mathcal{F}}}^{(z\text{-rot-ax})}$", fontsize=font_size)
                ax.set_xlabel(r"$\boldsymbol{\mathbf{e}}^{(z\text{-rot-ax})}$ (rad)", fontsize=font_size)

            fig.savefig(os.path.join(full_plot_dir, f"{calib_type}.png"), bbox_inches='tight', dpi=300)
        else:
            if calib_type == 'torque':
                ax.set_xlabel("pose - target (rad)")
            else:
                ax.set_xlabel("pose - target (m)")

            ax.set_ylabel("STS Force")
            ax.set_title(f"{calib_type}")

            fig.savefig(os.path.join(full_plot_dir, f"{calib_type}.png"))

    else:
        all_ctpttv = []
        all_ctf = []
        for dot_i, (ctpttv, ctf) in enumerate(zip(calib_type_pose_to_targ_vecs, calib_type_forces)):
            all_ctpttv.append(ctpttv)
            all_ctf.append(ctf)

            fig, ax = plt.subplots(figsize=fig_size, tight_layout=True)

            ax.scatter(all_ctpttv, all_ctf, c='k', s=2)

            # generate line data
            if calib_type == "normal":
                # m0x + b0 = m1x + b1 --> x = (b1 - b0) / (m0 - m1)
                x_transition_val = (params[calib_type]['b_1'] - params[calib_type]['b_0']) / \
                    (params[calib_type]['m_0'] - params[calib_type]['m_1'])

                if x_transition_val < 0:
                    print("transition between lines not >0, something's wrong...")
                    import ipdb; ipdb.set_trace()

                x_vals = np.linspace(0, x_transition_val, 100)
                y_vals = params[calib_type]['m_0'] * x_vals
                ax.plot(x_vals, y_vals, linewidth=3, color='C0')

                x_vals = np.linspace(x_transition_val, params[calib_type]['max_dist'], 100)
                y_vals = params[calib_type]['m_1'] * x_vals + params[calib_type]['b_1']
                ax.plot(x_vals, y_vals, linewidth=3, color='C1')

            else:
                if calib_type == 'torque':
                    x_vals = np.linspace(0, params[calib_type]['max_dist'] * np.pi / 180, 100)
                else:
                    x_vals = np.linspace(0, params[calib_type]['max_dist'], 100)
                y_vals = params[calib_type]['m_0'] * x_vals
                ax.plot(x_vals, y_vals, linewidth=3, color='C0')

            if args.fancy_plots:
                # map = {'normal': 'z', 'shear_x': 'x', 'shear_y': 'y', 'torque': 'z'}

                if calib_type == 'normal':
                    ax.set_ylabel(r"$\Tilde{\boldsymbol{\mathcal{F}}}^{(z\text{-ax})}$", fontsize=font_size)
                    ax.set_xlabel(r"$\boldsymbol{\mathbf{e}}^{(z\text{-ax})}$ (m)", fontsize=font_size)
                    ax.set_xlim(-0.003, 0.035)
                    ax.set_ylim(-0.025, 8.25)
                elif calib_type == 'shear_x':
                    ax.set_ylabel(r"$\Tilde{\boldsymbol{\mathcal{F}}}^{(x\text{-ax})}$", fontsize=font_size)
                    ax.set_xlabel(r"$\boldsymbol{\mathbf{e}}^{(x\text{-ax})}$ (m)", fontsize=font_size)
                elif calib_type == 'shear_y':
                    ax.set_ylabel(r"$\Tilde{\boldsymbol{\mathcal{F}}}^{(y\text{-ax})}$", fontsize=font_size)
                    ax.set_xlabel(r"$\boldsymbol{\mathbf{e}}^{(y\text{-ax})}$ (m)", fontsize=font_size)
                elif calib_type == 'torque':
                    ax.set_ylabel(r"$\Tilde{\boldsymbol{\mathcal{F}}}^{(z\text{-rot-ax})}$", fontsize=font_size)
                    ax.set_xlabel(r"$\boldsymbol{\mathbf{e}}^{(z\text{-rot-ax})}$ (rad)", fontsize=font_size)

                fig.savefig(os.path.join(full_plot_dir, f"{calib_type}-{dot_i}.png"), bbox_inches='tight', dpi=300)
            else:
                if calib_type == 'torque':
                    ax.set_xlabel("pose - target (rad)")
                else:
                    ax.set_xlabel("pose - target (m)")

                ax.set_ylabel("STS Force")
                ax.set_title(f"{calib_type}")

                fig.savefig(os.path.join(full_plot_dir, f"{calib_type}.png"))

            print(f"Saved fig {calib_type}, {dot_i}/{len(calib_type_forces) - 1}")

            plt.close(fig)

print(f"New config:")
# pprint.pprint(params)
for k in params.keys():
    if type(params[k]) == dict:
        for sk in params[k].keys():
            new_param = params[k][sk]
            old_param = old_params[k][sk]

            if new_param != old_param:
                print(f"param {k}-{sk}:  {old_param} --> {new_param}")
    else:
        new_param = params[k]
        old_param = old_params[k]

        if new_param != old_param:
            print(f"param {k}:  {old_param} --> {new_param}")

resp = input(f"Save new config? y for yes, all other for no: ")
if resp == 'y':
    write_json(params_file, params)
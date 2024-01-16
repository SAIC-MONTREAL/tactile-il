import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from transform_utils.pose_transforms import PoseTransformer

from contact_il.data.dict_dataset import DictDataset
import contact_il.plotting.panda_ft_comparison_data as comp_data

from contact_il.plotting.utils import eul_rot_to_mat, pose_arr_to_mat_arr, enable_latex_plotting, add_arrow
# from pysts.utils import eul_rot_json_to_mat, eul_rot_to_mat


######## Options ########
plot_dir = os.path.join(comp_data.MAIN_PLOT_DIR, 'panda_ft_comp')
quiver_mults = [0.13, 0.05, 0.04, 0.02]
font_size = 20
line_width = 2.0
# fig_size = (8, 2)
fig_size = (7.5, 2)
shaft_width = 0.0035

sts_to_ee_eul_rxyz = [1.5707963, -1.5707963, 0.0]
sts_to_ee_rot_mat = eul_rot_to_mat(sts_to_ee_eul_rxyz)

separate_force_plots = True

#########################

np.set_printoptions(suppress=True, precision=4)

enable_latex_plotting()

# start with toy plots, which will be made into two separate plots
for task_i, (task, task_data) in enumerate(comp_data.TOY_TASK_DATA_DICT.items()):
    if separate_force_plots:
        gs_kw = dict(width_ratios=[1, 5], height_ratios=[1, 1])
        fig, axd = plt.subplot_mosaic([['traj', 'panda_force'],
                                    ['traj', 'sts_force']],
                                    gridspec_kw=gs_kw, figsize=fig_size,
                                    layout="constrained")
    else:
        gs_kw = dict(width_ratios=[1, 5])
        fig, axd = plt.subplot_mosaic([['traj', 'force']],
                                    gridspec_kw=gs_kw, figsize=fig_size,
                                    layout="constrained")

    ds_dir = os.path.join(comp_data.MAIN_DIR, 'demonstrations', task, task_data['ds_substr'])

    ds = DictDataset(None, main_dir=ds_dir, dataset_name="")

    dict_of_arr = ds.load_ep_as_dict_of_arrays(task_data['ep'])

    # 1. traj ax
    # these envs don't set the poses to be relative, so transform them to make them relative
    ax = axd['traj']
    rel_poses = []
    rel_poss = []
    first_po_mat_inv = np.linalg.inv(PoseTransformer(dict_of_arr['pose'][0]).get_matrix())
    for po in dict_of_arr['pose']:
        po_mat = PoseTransformer(po).get_matrix()
        t_po = first_po_mat_inv @ po_mat
        rel_poses.append(t_po)
        rel_poss.append(t_po[:3, 3])

    rel_poses = np.array(rel_poses)[:task_data['max_ts']]
    rel_poss = np.array(rel_poss)[:task_data['max_ts']]

    traj_line = ax.plot(rel_poss[:, 2], rel_poss[:, 1], lw=line_width)[0]
    ax.set_xlabel('z pos (m)', fontsize=font_size)
    ax.set_ylabel('y pos (m)', fontsize=font_size)

    if task_data['traj_xlim'] is not None:
        ax.set_xlim(task_data['traj_xlim'])

    # midpath arrows
    for ind in task_data['traj_arrow_inds']:
        add_arrow(traj_line, xind=ind)

    # timestep labels
    x, y = rel_poss[:, 2], rel_poss[:, 1]
    for ei, i in enumerate(task_data['traj_ts_labels']):
        xa, ya = task_data['traj_ts_labels_xyadds'][ei]
        arr_arc = task_data['arrow_arcs'][ei]
        ax.annotate(f't={i}', xy=(x[i], y[i]), xytext=(x[i] + xa, y[i] + ya),
                    arrowprops=dict(arrowstyle='simple', connectionstyle=f"arc3,rad={arr_arc}",
                                    facecolor='orange'),
                    bbox=dict(boxstyle="round4", fc="w"),
                    fontsize=font_size-6
                    )

    # 2. panda ft
    if separate_force_plots:
        ax = axd['panda_force']
    else:
        ax = axd['force']
        ax.set_ylim([0, 2])

    if separate_force_plots:
        ax.set_ylabel('Panda \nForce', fontsize=font_size)
        ax.yaxis.label.set_color('b')
    panda_forces = dict_of_arr['force_torque_internal'][:task_data['max_ts'], :3]
    x_vals = np.arange(panda_forces.shape[0])
    y_vals = np.ones(x_vals.shape) * 1.5

    # ax.yaxis.set_tick_params(labelleft=False)
    ax.set_yticks([])
    ax.xaxis.set_tick_params(labelbottom=False)

    scale = task_data['panda_force_scale']
    # hs = 0.7
    hs = 1.0

    ax.quiver(x_vals, y_vals, panda_forces[:, 2], panda_forces[:, 1], scale=scale, scale_units='height',
              width=shaft_width, headwidth=hs * 3.0, headlength=hs * 5.0, headaxislength=hs * 4.5, label='Panda',
              color='b')

    # 3. sts ft
    if separate_force_plots:
        ax = axd['sts_force']
        ax.set_ylabel('STS \nForce', fontsize=font_size)
        ax.yaxis.label.set_color('r')
    else:
        ax = axd['force']

    sts_forces = dict_of_arr['sts_avg_force'][:task_data['max_ts'], :3]
    sts_forces_rot = (np.linalg.inv(sts_to_ee_rot_mat) @ sts_forces.T).T

    if task == 'PandaCabinetOneFinger6DOFROS':
        sts_forces_rot = np.zeros_like(sts_forces_rot)  # close enough to zero, and would have been with proper calibration

    y_vals = np.ones(x_vals.shape) * 0.5

    scale = task_data['sts_force_scale']
    # hs = 0.7
    hs = 1.0

    ax.quiver(x_vals, y_vals, sts_forces_rot[:, 2], sts_forces_rot[:, 1], scale=scale, scale_units='height',
              width=shaft_width, headwidth=hs * 3.0, headlength=hs * 5.0, headaxislength=hs * 4.5, label='STS',
              color='r')

    # vertical lines marking beginning and end of contact
    if task_data['contact_ts'] is not None:
        for ax_i, ax in enumerate([axd['panda_force'], axd['sts_force']]):
            ylim = ax.get_ylim()
            ax.set_ylim(ylim)
            ax.vlines(np.array(task_data['contact_ts']) - .5, ylim[0] - .3, ylim[1] + .3, colors=['k', 'k'],
                      linestyle='--')

            if ax_i == 1:
                contact_lines_x = np.array(task_data['contact_ts']) - .5
                li = contact_lines_x

                y_pos = 0.7 * (ylim[1] - ylim[0]) + ylim[0]
                # y_pos = -0.1 * (ylim[1] - ylim[0]) + ylim[0]
                # y_pos = -.3 * (ylim[1] - ylim[0]) + ylim[0]

                # y_pos_add = ylim[0] - 0.1 * (ylim[1] - ylim[0])
                y_pos_add = -1.3 * (ylim[1] - ylim[0])
                ax.annotate(f'Contact\nbegins', xy=(li[0], y_pos), xytext=(li[0] - 7, y_pos + y_pos_add),
                            arrowprops=dict(arrowstyle='simple', connectionstyle=f"arc3,rad={-0.1}",
                                            facecolor='orange'),
                            bbox=dict(boxstyle="round4", fc="w"),
                            fontsize=font_size - 5, annotation_clip=False
                            )

                y_pos_add = -1.6 * (ylim[1] - ylim[0])
                ax.annotate(f'Contact\nends', xy=(li[1], y_pos), xytext=(li[1] -3, y_pos + y_pos_add),
                            arrowprops=dict(arrowstyle='simple', connectionstyle=f"arc3,rad={0.3}",
                                            facecolor='orange'),
                            bbox=dict(boxstyle="round4", fc="w"),
                            fontsize=font_size - 5, annotation_clip=False
                            )


    ax = axd['sts_force']
    ax.set_yticks([])
    ax.set_xlabel('Timestep', fontsize=font_size)
    ax.tick_params(axis='x', which='major', labelsize=font_size - 4)

    # 4. save
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, f'toy-{task_data["ds_substr"]}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(plot_dir, f'toy-{task_data["ds_substr"]}.png'), bbox_inches='tight', dpi=300)
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
from matplotlib.lines import Line2D

from contact_il.data.dict_dataset import DictDataset
import contact_il.plotting.force_matching_data as fm_data

from contact_il.plotting.utils import eul_rot_to_mat, pose_arr_to_mat_arr
# from pysts.utils import eul_rot_json_to_mat, eul_rot_to_mat


######## Options ########
plot_dir = "/user/t.ablett/datasets/contact-il/videos/with_without/"
data_missing = False
three_d = False
quiver_mults = [0.13, 0.05, 0.04, 0.02]
z_lims = [[-.01, .15], [-.01, .15], [-.01, .35], [-.1, .3]]
sts_ex_zooms = (.45, .4, .45, .35)
font_size = 16
line_width = 2.0
# include_tasks = [False, False, True, False]  # for Glass Orb
include_tasks = [False, False, False, True]  # for Glass Orb Close
lowest_include_task = np.min(np.nonzero(include_tasks))
fig_size = (sum(include_tasks) * 3, 3.8)

control_delay = 2

# with sts ex
# xytext_adds = [(-.1, .005), (.05, -.02), (-.18, -.26), (0.05, -.03)]
# arrow_arcs = [.2, -.2, .2, -.2]
# stsex_adds = [(-.3, 0.02), (.15, 0.055), (-.25, .05), (.03, 0.2)]
# stsex_arrow_arcs = [.3, .2, .3, -.2]

# with timestep label instead
xytext_adds = [(-.12, -.005), (.05, 0.0), (-.1, 0.0), (0.05, -.06)]
arrow_arcs = [.2, -.2, .2, -.2]

stsex_adds = [(-.12, -.03), (.07, 0.055), (-.2, 0.0), (-.1, 0.25)]
stsex_arrow_arcs = [.3, .2, .1, -.2]

tT_text_adds = [(.05, -.005), (-.12, -.035), (0.0, -.15), (-.02, .25)]
tT_arrow_arcs = [.2, -.2, -.2, -.2]

sts_to_ee_eul_rxyz = [1.5707963, -1.5707963, 0.0]
sts_to_ee_rot_mat = eul_rot_to_mat(sts_to_ee_eul_rxyz)

#########################

np.set_printoptions(suppress=True, precision=4)

pream = "\n".join([
    r"\usepackage{amsmath}",
    r"\usepackage{amssymb}",
    r"\usepackage{amsfonts}",
    r"\usepackage{bbm}",
    r"\usepackage{mathtools}",
    ])
plt.rcParams.update({"font.family": "serif", 'font.serif': ["Computer Modern Roman"], "text.usetex": True,
                        "pgf.rcfonts": False, "pgf.preamble": pream, "text.latex.preamble": pream})

if three_d:
    fig = plt.figure(figsize=plt.figaspect(3))
else:
    fig, axs = plt.subplots(3, sum(include_tasks), figsize=fig_size)

# plt.subplots_adjust(wspace=.5)
plt.subplots_adjust(wspace=.575)

plt.ion()

for task_i, (task, task_data) in enumerate(fm_data.TASK_DATA_DICT.items()):
    ds_dir = os.path.join(fm_data.MAIN_DIR, 'demonstrations', task, task_data['ds_substr'])
    no_fa_ds_dir = os.path.join(fm_data.MAIN_DIR, 'demonstrations', task, task_data['no_fa_ds_substr'])

    if include_tasks[task_i]:
        try:
            ds = DictDataset(None, main_dir=ds_dir, dataset_name="")
            no_fa_ds = DictDataset(None, main_dir=no_fa_ds_dir, dataset_name="")

            fa_dict_of_arr, raw_dict_of_arr = ds.load_ep_as_dict_of_arrays(task_data['ep'], include_raw_demo=True)
            no_fa_dict_of_arr = no_fa_ds.load_ep_as_dict_of_arrays(task_data['ep'], include_raw_demo=False)

            for top_d_i, dict_of_arr in enumerate([raw_dict_of_arr, fa_dict_of_arr, no_fa_dict_of_arr]):

                for ts in range(dict_of_arr['pose'].shape[0]):

                    for d_i, dict_of_arr in enumerate([raw_dict_of_arr, fa_dict_of_arr, no_fa_dict_of_arr]):
                        if d_i > top_d_i:
                            continue


                        if sum(include_tasks) == 1:
                            ax = axs[d_i]
                        else:
                            ax = axs[d_i, task_i - lowest_include_task]

                        # ax = fig.add_subplot(1, 1, d_i + 1, projection='3d')  # TODO if we switch to 3x4, need to fix this of course
                        x_points = dict_of_arr['pose'][:, 0]
                        y_points = dict_of_arr['pose'][:, 2]  # swapped with below for better viewing
                        z_points = dict_of_arr['pose'][:, 1]


                        if sum(include_tasks) == 1:
                            ax = axs[d_i]
                        else:
                            ax = axs[d_i, task_i - lowest_include_task]

                        ax.clear()

                        if three_d:
                            ax.plot(x_points, y_points, z_points, label="Actual Poses")
                        else:
                            if top_d_i > 0 and d_i == 0:
                                if ts == (len(y_points) - 1):
                                    top_ts = ts
                                else:
                                    # timesteps delayed for control lag
                                    top_ts = max(0, ts + 1 - control_delay)
                            else:
                                top_ts = ts + 1
                            act_traj_line = ax.plot(y_points[:top_ts], z_points[:top_ts], label="Actual Poses", lw=line_width)

                        if d_i > 0:
                            x_des_points = dict_of_arr['target_pose'][:, 0]
                            y_des_points = dict_of_arr['target_pose'][:, 2]
                            z_des_points = dict_of_arr['target_pose'][:, 1]

                            # instead of target_pose, going to try and use actions for full desired trajectory instead
                            # for sure, this will ensure taht the green line in the no FA plot matches the blue line in the
                            # demo plot, which right now is not the case
                            des_poss_from_acts = []
                            act_mats = pose_arr_to_mat_arr(dict_of_arr['act'], rotation_rep='rvec')
                            cur_des_pose_mat = np.eye(4)
                            des_poss_from_acts.append(cur_des_pose_mat[:3, 3])
                            cur_des_pose = np.array([0., 0., 0., 1., 0., 0., 0.])
                            for am in act_mats:
                                cur_des_pose_mat = cur_des_pose_mat @ am
                                des_poss_from_acts.append(cur_des_pose_mat[:3, 3])

                            des_poss_from_acts = np.stack(des_poss_from_acts)

                            # to better match the blue curves, we're going to remove the last one
                            des_poss_from_acts = des_poss_from_acts[:-1]

                            x_des_points = des_poss_from_acts[:, 0]
                            y_des_points = des_poss_from_acts[:, 2]
                            z_des_points = des_poss_from_acts[:, 1]

                            if three_d:
                                ax.plot(x_des_points, y_des_points, z_des_points, label='Desired Poses')
                            else:
                                ax.plot(y_des_points, z_des_points, label='Desired Poses', linestyle='--',
                                        c='g', lw=line_width)

                        ##### force arrows
                        avg_force = dict_of_arr['sts_avg_force'][:, :3]
                        avg_force[:, :2] = 0  # since the other directions are noisy

                        # rotate forces into current pose frame
                        # first rotate all from sts frame to ee frame
                        force_ee_frame = (sts_to_ee_rot_mat.T @ avg_force.T).T

                        # then rotate each into current pose
                        mat_arr = pose_arr_to_mat_arr(dict_of_arr['pose'])
                        force_cur_frame = np.einsum("ijk, ik->ij", mat_arr[:, :3, :3], force_ee_frame)
                        u_qu = force_cur_frame[:, 0] * quiver_mults[task_i]
                        v_qu = force_cur_frame[:, 2] * quiver_mults[task_i]
                        w_qu = force_cur_frame[:, 1] * quiver_mults[task_i]

                        if three_d:
                            ax.quiver(x_points, y_points, z_points, u_qu, v_qu, w_qu, color='r', label='Force')
                        else:
                            if top_d_i > 0 and d_i == 0:
                                if ts == (len(y_points) - 1):
                                    top_ts = ts
                                else:
                                    # 2 timesteps delayed for control lag
                                    top_ts = max(0, ts + 1 - control_delay)
                            else:
                                top_ts = ts + 1
                            ax.quiver(y_points[:top_ts], z_points[:top_ts], v_qu[:top_ts], w_qu[:top_ts], scale=1.0,
                                    color='r', label='STS Force')

                        if three_d:
                            ax.set_zlim(z_lims[task_i])
                        else:
                            ax.set_ylim(z_lims[task_i])

                        if three_d:
                            ax.set_xlabel('x')
                            ax.set_ylabel('z')
                            ax.set_zlabel('y')
                            ax.invert_zaxis()

                        # need these labels because they're not the same for every task
                        # if task_i != 0:
                        #     ax.tick_params(labelleft=False)

                        if d_i == 0:
                            ax.set_title(fm_data.TASK_PLOT_NAMES[task_i], fontsize=font_size)

                        ax.annotate('t=0', xy=(y_points[0], z_points[0]),  # remember that y_points and z_points are swapped
                                    xytext=(y_points[0] + xytext_adds[task_i][0], z_points[0] + xytext_adds[task_i][1]),
                                    arrowprops=dict(arrowstyle='simple', connectionstyle=f"arc3,rad={arrow_arcs[task_i]}",
                                                    facecolor='orange'),
                                    bbox=dict(boxstyle="round4", fc="w")
                                    )

                        if (task_i - lowest_include_task) == 0:
                            # ax.set_ylabel(fm_data.TRAJ_PLOT_NAMES_SHORT[d_i], fontsize=font_size - 5)
                            ax.set_ylabel(fm_data.TRAJ_PLOT_NAMES_VERY_SHORT[d_i], fontsize=font_size)
                            ax.yaxis.set_label_coords(-.26, 0.5)
                        else:
                            pass
                            # ax.set_yticks([])
                            # ax.tick_params(labelleft=False)

                        # xy = (y_points[task_data['rep_ts'] - 3], z_points[task_data['rep_ts'] - 3])
                        if d_i == 0:
                            if 'GlassOrbClose' in task:
                                xy = (y_points[task_data['rep_ts'] - 1], z_points[task_data['rep_ts'] - 1])
                            else:
                                xy = (y_points[task_data['rep_ts'] - 3], z_points[task_data['rep_ts'] - 3])
                        else:
                            xy = (y_points[task_data['rep_ts']], z_points[task_data['rep_ts']])

                        if ts >= task_data["rep_ts"]:
                            ax.annotate(f't={task_data["rep_ts"]}', xy=xy,
                                        xytext=(xy[0] + stsex_adds[task_i][0], xy[1] + stsex_adds[task_i][1]),
                                        arrowprops=dict(arrowstyle='simple', connectionstyle=f"arc3,rad={stsex_arrow_arcs[task_i]}",
                                                                facecolor='orange'),
                                        bbox=dict(boxstyle="round4", fc="w")
                                        )

                        xy=(y_points[-1], z_points[-1])
                        if ts >= (len(y_points) - 1):
                            ax.annotate(f't=T', xy=xy,
                                        xytext=(xy[0] + tT_text_adds[task_i][0], xy[1] + tT_text_adds[task_i][1]),
                                        arrowprops=dict(arrowstyle='simple', connectionstyle=f"arc3,rad={tT_arrow_arcs[task_i]}",
                                                                facecolor='orange'),
                                        bbox=dict(boxstyle="round4", fc="w")
                                        )

                    for label_d_i, _ in enumerate([raw_dict_of_arr, fa_dict_of_arr, no_fa_dict_of_arr]):
                        if sum(include_tasks) == 1:
                            sub_ax = axs[label_d_i]
                        else:
                            sub_ax = axs[label_d_i, task_i - lowest_include_task]
                        sub_ax.set_xlim(task_data['live_xlim'])
                        sub_ax.set_ylim(task_data['live_ylim'])

                        if label_d_i != 2:
                            sub_ax.tick_params(labelbottom=False)

                        sub_ax.grid(True, which='both', alpha=.5)

                        # another as label for representative image that will be added manually after
                        sub_ax.annotate(f't={task_data["rep_ts"]}', xy=xy,
                                    # xytext=(1.13, -0.1),
                                    xytext=(1.1, -0.1),
                                    bbox=dict(boxstyle="round4", fc="w"), textcoords="axes fraction"
                                    )

                        if label_d_i == 0 and task_i - lowest_include_task == 0:
                            leg_bta = (-0.35, -3.45)
                            des_line_leg = Line2D([0], [0], label='Desired Poses', linestyle='--',
                                                  c='g', lw=line_width)
                            leg_handles, leg_labels = sub_ax.get_legend_handles_labels()
                            leg_handles.insert(1, des_line_leg)
                            sub_ax.legend(handles=leg_handles, fancybox=True, shadow=True, loc='lower left', ncol=3,
                                    fontsize=font_size-5, bbox_to_anchor=leg_bta)

                    # dummy axis for sup labels to allow customizing position
                    if top_d_i == 0:
                        big_ax = fig.add_subplot(111, frameon=False)
                        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                        big_ax.set_xlabel('z pos (m)', fontsize=font_size)
                        big_ax.xaxis.set_label_coords(0.5, -.1)
                        big_ax.set_ylabel('y pos (m)', fontsize=font_size)
                        big_ax.yaxis.set_label_coords(-.15, 0.5)
                        # ax.yaxis.set_label_coords(-.035, 0.5)

                    os.makedirs(os.path.join(plot_dir, task), exist_ok=True)
                    # fig.savefig(os.path.join(plot_dir, f'with_without_task-{lowest_include_task}-di_{d_i}-ts_{ts}.pdf'), bbox_inches='tight')
                    fig.savefig(os.path.join(plot_dir, task, f'di_{top_d_i}-ts_{ts}.png'), bbox_inches='tight', dpi=300)
                    print(f"Saved task {task}, d_i: {top_d_i}, ts: {ts}")


        except AssertionError as e:
            if data_missing:
                print(f"Not enough data for task {task}, {task_data['ds_substr']}, skipping")
            else:
                raise AssertionError(e)


    # dict_keys(['pose', 'n_pose', 'prev_pose', 'n_prev_pose', 'raw_world_pose', 'n_raw_world_pose', 'joint_pos',
    # 'n_joint_pos', 'target_pose', 'n_target_pose', 'wrist_rgb', 'n_wrist_rgb', 'sts_raw_image',
    # 'n_sts_raw_image', 'sts_avg_force', 'n_sts_avg_force', 'sts_in_contact', 'n_sts_in_contact', 'act'])


# plt.tight_layout()
# plt.show()

# print('ax.azim {}'.format(ax.azim))
# print('ax.elev {}'.format(ax.elev))

# fig.supxlabel('z pos (m)', x=.5, y=.5)
# fig.supxlabel('z pos (m)', loc='left')
# fig.supylabel('y pos (m)')


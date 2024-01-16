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
import contact_il.plotting.mode_switch_contact_data as msc_data

from contact_il.plotting.utils import eul_rot_to_mat, pose_arr_to_mat_arr, enable_latex_plotting, add_arrow
# from pysts.utils import eul_rot_json_to_mat, eul_rot_to_mat


######## Options ########
plot_dir = os.path.join(msc_data.MAIN_PLOT_DIR, 'mode_switch_contact')
font_size = 16
fig_size = (8, 3)

#########################

np.set_printoptions(suppress=True, precision=4)
enable_latex_plotting()

fig, axs = plt.subplots(2, 4, figsize=fig_size)

all_demo_stats_dict = dict()

for task_i, (task, task_data) in enumerate(msc_data.DEMO_DATA_DICT.items()):
    all_demo_stats_dict[task] = dict()
    ax = axs[0, task_i]

    ds_dir = os.path.join(msc_data.MAIN_DIR, 'demonstrations', task, task_data['ds_substr'])
    ds = DictDataset(None, main_dir=ds_dir, dataset_name="")

    raw_contact_ts = np.array(task_data['raw_contact_ts'])
    contact_ts = np.array(task_data['contact_ts'])

    # get timestep of switch from actions in each ep
    switch_ts = []
    for ep in range(len(ds)):
        dict_of_arr, raw_dict_of_arr = ds.load_ep_as_dict_of_arrays(ep, include_raw_demo=True)
        switch_acts = dict_of_arr['act'][:, -1]
        ep_switch_ts = np.where(switch_acts > 0)[0].min()
        switch_ts.append(ep_switch_ts)

    switch_ts = np.array(switch_ts)

    diff = switch_ts - contact_ts

    all_demo_stats_dict[task]['diff_mean'] = diff.mean()
    all_demo_stats_dict[task]['diff_std'] = diff.std()

    # plot histogram
    ax.hist(diff, bins=range(-5, 6))
    # ax.set_xlim([-5, 5])
    if task_i == 0:
        ax.set_ylabel("Demos", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size - 4)
    ax.xaxis.set_tick_params(labelbottom=False)

    # ax.annotate(f'Avg: {diff.mean():.2f}', xy=[0.02, 0.88], xycoords='axes fraction', textcoords='axes fraction')
    # ax.annotate(f'Std: {diff.std():.2f}', xy=[0.048, 0.73], xycoords='axes fraction', textcoords='axes fraction')
    ax.annotate(r'$\overline{x}$: '+f'{diff.mean():.2f}', xy=[0.02, 0.85], xycoords='axes fraction', textcoords='axes fraction',
                fontsize=font_size - 2)
    ax.annotate(r'$s$: '+f'{diff.std():.2f}', xy=[0.032, 0.67], xycoords='axes fraction', textcoords='axes fraction',
                fontsize=font_size - 2)

    ax.set_title(msc_data.TASK_PLOT_NAMES[task_i], fontsize=font_size - 2)


for task, data in all_demo_stats_dict.items():
    print(f"DEMO: Task {task}: switch_ts - contact_ts mean: {data['diff_mean']}, std: {data['diff_std']}")


all_test_stats_dict = dict()

for task_i, (task, task_data) in enumerate(msc_data.TEST_DATA_DICT.items()):
    all_test_stats_dict[task] = dict()
    ax = axs[1, task_i]

    switch_ts = []
    for s in ['1', '2', '3']:
        ds_dir = os.path.join(msc_data.MAIN_DIR, 'tests', task, task_data['ds_substr'], s, task_data['postseed_str'])
        ds = DictDataset(None, main_dir=ds_dir, dataset_name="")

        # get timestep of switch from actions in each ep
        for ep in range(len(ds)):
            dict_of_arr = ds.load_ep_as_dict_of_arrays(ep, include_raw_demo=False)
            switch_acts = dict_of_arr['act'][:, -1]
            ep_switch_ts = np.where(switch_acts > 0)[0].min()
            switch_ts.append(ep_switch_ts)

    contact_ts = np.concatenate(task_data['contact_ts'])
    switch_ts = np.array(switch_ts)

    diff = switch_ts - contact_ts

    all_test_stats_dict[task]['diff_mean'] = diff.mean()
    all_test_stats_dict[task]['diff_std'] = diff.std()

    # plot histogram
    ax.hist(diff, bins=range(-5, 6), color='C1')
    # ax.set_xlim([-5, 5])
    if task_i == 0:
        ax.set_ylabel("Policies", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size - 4)

    # ax.annotate(f'Avg: {diff.mean():.2f}', xy=[0.02, 0.88], xycoords='axes fraction', textcoords='axes fraction')
    # ax.annotate(f'Std: {diff.std():.2f}', xy=[0.048, 0.73], xycoords='axes fraction', textcoords='axes fraction')
    ax.annotate(r'$\overline{x}$: '+f'{diff.mean():.2f}', xy=[0.02, 0.85], xycoords='axes fraction', textcoords='axes fraction',
                fontsize=font_size - 2)
    ax.annotate(r'$s$: '+f'{diff.std():.2f}', xy=[0.032, 0.67], xycoords='axes fraction', textcoords='axes fraction',
                fontsize=font_size - 2)

# dummy axis for sup labels to allow customizing position
ax = fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
ax.set_xlabel(r'Switch timestep $-$ Contact timestep', fontsize=font_size)
ax.xaxis.set_label_coords(0.5, -.13)
# ax.set_ylabel('y pos (m)', fontsize=font_size)
# ax.yaxis.set_label_coords(-.06, 0.5)
# ax.yaxis.set_label_coords(-.035, 0.5)


os.makedirs(plot_dir, exist_ok=True)
fig.savefig(os.path.join(plot_dir, 'mode_contact_diff_histograms.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(plot_dir, 'mode_contact_diff_histograms.png'), bbox_inches='tight', dpi=300)
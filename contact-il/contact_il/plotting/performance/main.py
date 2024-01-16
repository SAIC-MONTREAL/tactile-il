import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

import contact_il.plotting.test_data as test_data
from contact_il.plotting.utils import eul_rot_to_mat, pose_arr_to_mat_arr, enable_latex_plotting, add_arrow


######## Options ########
plot_dir = os.path.join(test_data.MAIN_PLOT_DIR, 'performance')
data_missing = False
bar_width = 0.1
bar_label_padding = 3
cap_size = 0
cmap = matplotlib.colormaps['tab20']
num_stds = 1
legend_under = True
bold_msfm = False
bold_best = True
bold_lw = 4

if legend_under:
    fig_size = (10, 4)
    font_size = 24
else:
    fig_size = (10, 2)
    font_size = 16

#########################

# plt.rcParams.update({"font.family": "serif", "font.serif": "Times", "text.usetex": True, "pgf.rcfonts": False})
# plt.rcParams.update({"font.family": "Times New Roman", "text.usetex": False, "pgf.rcfonts": False})

enable_latex_plotting()

os.makedirs(plot_dir, exist_ok=True)
df, _, _, main_perf_data = test_data.get_main_perf_data('main', data_missing)

fig, ax = plt.subplots(constrained_layout=True, figsize=fig_size)

x = np.arange(len(test_data.TASKS))
for va_i, (va, va_data) in enumerate(main_perf_data.items()):
    offset = bar_width * va_i
    color = cmap(test_data.COLOR_MODE_MAP[test_data.MAIN_VARIANT_IDX_COMBOS[va_i]])

    pattern = ""
    if 'FM' in va and 'MS' in va:
        pattern = "xxx"

        if bold_msfm or bold_best and 'WSR' in va:
            va = r'\textbf{%s}' % va

    elif 'FM' in va:
        pattern = "///"
    elif 'MS' in va:
        pattern = "\\\\\\"

    # manually remove VO from VO, WR
    if va == 'VO, WR':
        va = 'WR'

    # rects = ax.bar(x + offset, va_data['suc_means'], bar_width, color=color, align='edge', label=va,
    #                yerr=va_data['suc_stds'], edgecolor='k', hatch=pattern, ecolor='darkslategrey', capsize=cap_size)
    rects = ax.bar(x + offset, va_data['suc_means'], bar_width, color=color, align='edge', label=va,
                   yerr=num_stds * va_data['suc_stds'], edgecolor='k', hatch=pattern, ecolor='darkslategrey')
    # ax.bar_label(rects, padding=bar_label_padding)

    # print out avg success rates for use in paper
    print(f"Avg suc rate -- variant {va}: mean -- {va_data['suc_all_mean']}, std -- {va_data['suc_all_std']}")

    if bold_best and 'MS-FM, WSR' in va:
        for r in rects:
            r.set_linewidth(bold_lw)

ax.set_ylabel('Success Rate', fontsize=font_size)
ax.set_xticks(x + bar_width * 0.5 * len(main_perf_data), test_data.TASK_PLOT_NAMES, fontsize=font_size)
# ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], fontsize=font_size)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], ['0.0', '0.25', '0.5', '0.75', '1.0'], fontsize=font_size-3)


# manually add hatches
forslash_patch = mpatches.Patch(facecolor='w', hatch='///', edgecolor='k', label='Force Match (FM)')
backslash_patch = mpatches.Patch(facecolor='w', hatch='\\\\\\', edgecolor='k', label='Mode Switch (MS)')
# both_patch = mpatches.Patch(facecolor='w', hatch='xxx', edgecolor='k', label='MS \& FM')
both_patch = mpatches.Patch(facecolor='w', hatch='xxx', edgecolor='k', label='MS-FM')
leg_handles, leg_labels = ax.get_legend_handles_labels()
leg_handles.insert(0, backslash_patch)
leg_handles.insert(1, forslash_patch)
leg_handles.insert(2, both_patch)

if legend_under:
    # ax.legend(handles=leg_handles, fancybox=True, shadow=True, loc='lower center',
    #           ncol=5, fontsize=font_size-3.2, bbox_to_anchor=(0.46, -0.75))
    ax.legend(handles=leg_handles, fancybox=True, shadow=True, loc='lower center',
              ncol=3, fontsize=font_size-3, bbox_to_anchor=(0.46, -0.87))
else:
    ax.legend(handles=leg_handles, fancybox=True, shadow=True, loc='center right', ncol=1, fontsize=font_size-7,
            bbox_to_anchor=(1.2, 0.43))
    # ax.legend(fancybox=True, shadow=True, loc='center right', ncol=1, fontsize=font_size-6, bbox_to_anchor=(1.2, 0.5))

ax.grid(alpha=0.5, axis='y')
ax.set_axisbelow(True)

if data_missing:
    ax.set_ylim(-0.2, 1.1)
else:
    ax.set_ylim(-0.1, 1.1)

if legend_under:
    fig.savefig(os.path.join(plot_dir, 'main_leg_und.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(plot_dir, 'main_leg_und.png'), bbox_inches='tight', dpi=300)
else:
    fig.savefig(os.path.join(plot_dir, 'main.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(plot_dir, 'main.png'), bbox_inches='tight', dpi=300)

# df.pivot('task', 'variant', 'suc_mean').plot(kind='bar')

# plt.savefig(os.path.join(plot_dir, 'main.pdf'), bbox_inches='tight')
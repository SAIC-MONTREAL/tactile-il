import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

import contact_il.plotting.test_data as test_data


######## Options ########
plot_dir = os.path.join(test_data.MAIN_PLOT_DIR, 'performance')
data_missing = False
bar_width = 0.1
bar_label_padding = 3
cap_size = 0
cmap = matplotlib.colormaps['tab20']
num_stds = 1
std_alpha = .5
line_width = 4
legend_under = True
bold_best = True
bold_lw = line_width * 2

if legend_under:
    # fig_size = (10, 4)
    fig_size = (16, 4)
    font_size = 32
else:
    fig_size = (10, 2)
    font_size = 16

#########################

# plt.rcParams.update({"font.family": "serif", "font.serif": "Times", "text.usetex": True, "pgf.rcfonts": False})
# plt.rcParams.update({"font.family": "Times New Roman", "text.usetex": False, "pgf.rcfonts": False})

pream = "\n".join([
    r"\usepackage{amsmath}",
    r"\usepackage{amssymb}",
    r"\usepackage{amsfonts}",
    r"\usepackage{bbm}",
    r"\usepackage{mathtools}",
    ])
plt.rcParams.update({"font.family": "serif", 'font.serif': ["Computer Modern Roman"], "text.usetex": True,
                        "pgf.rcfonts": False, "pgf.preamble": pream, "text.latex.preamble": pream})

os.makedirs(plot_dir, exist_ok=True)
df, _, _, _ = test_data.get_main_perf_data('data_variant', data_missing)

# fig, axs = plt.subplots(nrows=1, ncols=4, constrained_layout=True, figsize=fig_size)
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=fig_size)

for ta_i, ta in enumerate(test_data.TASK_PLOT_NAMES):
    ax = axs[ta_i]
    for va_i, va in enumerate(test_data.DATA_VARIANT_NAMES_SHORT):
        color = cmap(test_data.COLOR_MODE_MAP[test_data.DATA_VARIANT_IDX_COMBOS[va_i]])

        mean = df.loc[(df['task'] == ta) & (df['variant'] == va)].suc_mean.to_numpy()
        std = df.loc[(df['task'] == ta) & (df['variant'] == va)].suc_std.to_numpy()
        amount = df.loc[(df['task'] == ta) & (df['variant'] == va)].amount.to_numpy()

        if bold_best and va == 'MS-FM, WSR':
            va = r'\textbf{%s}' % va

        # manually remove VO from VO, WR
        if va == 'VO, WR':
            va = 'WR'

        if legend_under:
            if bold_best and 'MS-FM, WSR' in va:
                ax.plot(amount, mean, color=color, label=va, linewidth=bold_lw)
            else:
                ax.plot(amount, mean, color=color, label=va, linewidth=line_width)
        else:
            ax.plot(amount, mean, color=color, label=va)
        ax.fill_between(amount, mean - num_stds * std, mean + num_stds * std, facecolor=color, alpha=std_alpha)

    ax.set_title(ta, fontsize=font_size)
    # ax.set_xticks([5, 10, 15, 20], fontsize=font_size)
    ax.set_xticks([5, 10, 15, 20], ['5', '10', '15', '20'], fontsize=font_size-3)
    # ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], fontsize=font_size)
    # ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], ['0.0', '0.25', '0.5', '0.75', '1.0'], fontsize=font_size-3)
    if ta_i != 0 and legend_under:
        ax.yaxis.set_tick_params(labelleft=False)
    ax.grid(alpha=0.5)
    ax.set_axisbelow(True)

    if data_missing:
        ax.set_ylim(-0.2, 1.1)
    else:
        ax.set_ylim(-0.1, 1.1)

if legend_under:
    ax.legend(fancybox=True, shadow=True, loc='lower center',
              ncol=2, fontsize=font_size, bbox_to_anchor=(-1.4, -0.87))
else:
    ax.legend(fancybox=True, shadow=True, loc='center right', ncol=1, fontsize=font_size-5,
            bbox_to_anchor=(1.97, 0.43))
    # ax.legend(fancybox=True, shadow=True, loc='center right', ncol=1, fontsize=font_size-6, bbox_to_anchor=(1.2, 0.5))

if legend_under:
    # dummy axis for sup labels to allow customizing position
    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel('Number of Demonstrations', fontsize=font_size)
    ax.xaxis.set_label_coords(0.5, -.18)
    ax.set_ylabel('Success Rate', fontsize=font_size)
    # ax.yaxis.set_label_coords(-.06, 0.5)
    ax.yaxis.set_label_coords(-.065, 0.5)
else:
    fig.supylabel('Success Rate', fontsize=font_size)
    fig.supxlabel('Number of Demonstrations', fontsize=font_size)


if legend_under:
    fig.savefig(os.path.join(plot_dir, 'data_variant_leg_und.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(plot_dir, 'data_variant_leg_und.png'), bbox_inches='tight', dpi=300)
else:
    fig.savefig(os.path.join(plot_dir, 'data_variant.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(plot_dir, 'data_variant.png'), bbox_inches='tight', dpi=300)

# df.pivot('task', 'variant', 'suc_mean').plot(kind='bar')

# plt.savefig(os.path.join(plot_dir, 'main.pdf'), bbox_inches='tight')
import numpy as np
from transform_utils.pose_transforms import PoseTransformer
import matplotlib.pyplot as plt


def eul_rot_to_mat(eul_rxyz):
    pt = PoseTransformer(pose=[0, 0, 0, *eul_rxyz], rotation_representation='euler', axes='rxyz')
    ee_to_sts_rot_mat = pt.get_matrix()[:3, :3]
    return ee_to_sts_rot_mat


def pose_arr_to_mat_arr(pose_arr, rotation_rep='quat', axes='rxyz'):
    mat_arr = []
    for pose in pose_arr:
        pt = PoseTransformer(pose=pose, rotation_representation=rotation_rep, axes=axes)
        mat = pt.get_matrix()
        mat_arr.append(mat)
    return np.array(mat_arr)


def enable_latex_plotting():
    pream = "\n".join([
    r"\usepackage{amsmath}",
    r"\usepackage{amssymb}",
    r"\usepackage{amsfonts}",
    r"\usepackage{bbm}",
    r"\usepackage{mathtools}",
    ])
    plt.rcParams.update({"font.family": "serif", 'font.serif': ["Computer Modern Roman"], "text.usetex": True,
                            "pgf.rcfonts": False, "pgf.preamble": pream, "text.latex.preamble": pream})
    # plt.rcParams.update({"font.family": "serif", "text.usetex": True,
    #                         "pgf.rcfonts": False, "pgf.preamble": pream, "text.latex.preamble": pream})
    # plt.rcParams.update({"font.family": "serif", 'font.serif': ['DejaVu Serif'],
    #                       "text.usetex": True,
    #                         "pgf.rcfonts": False, "pgf.preamble": pream, "text.latex.preamble": pream})


def add_arrow(line, position=None, xind=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    xind:       x index of data, used instead of position if set
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
        # find closest index
        start_ind = np.argmin(np.absolute(xdata - position))
        if direction == 'right':
            end_ind = start_ind + 1
        else:
            end_ind = start_ind - 1
    if xind is not None:
        start_ind = xind
        end_ind = start_ind + 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )
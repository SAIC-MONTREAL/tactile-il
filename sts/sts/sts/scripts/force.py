import copy
import os
import argparse
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Arc
from matplotlib import gridspec
from datetime import datetime
from scipy.interpolate import griddata
from matplotlib import cm

from sts.scripts.helper import FakeCamera
from sts.scripts.helper import read_json, plot
from sts.scripts.sts_transform import STSTransform
from sts.scripts.marker_detection.marker_detection import MarkerDetectionCreator
from sts.scripts.depth_from_markers import DepthDetector


class ForceDetector:
    """ This class assumes that a separate class object has already detected marker displacement and depth
    detection. """
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.cfg = read_json(os.path.join(self.config_dir,'force.json'))
        self.prev_displ = None
        self.prev_cents = None
        self.count = 0

    def get_force(self, markers_fixed_ref, markers_current, z_displacement_all, get_average=False, show_image=False):
        if len(markers_current) == 0:
            print("sts force: Warning..no markers found, setting force to zero")
            displ = np.zeros_like(markers_fixed_ref)
            cents = markers_fixed_ref.astype(np.float32)
        else:
            displ = markers_current - markers_fixed_ref
            cents = markers_current.astype(np.float32)
        self.prev_displ = displ
        self.prev_cents = cents

        # only get depth at the current marker locations
        valid_depths = cv2.remap(z_displacement_all, cents[:, 0], cents[:, 1], interpolation=cv2.INTER_LINEAR)

        # rescale and concatenate
        z_rescaled = self.cfg['z_scale'] * valid_depths
        xyz_displ = np.concatenate([displ, z_rescaled], axis=-1)

        # removed because this is no longer true for normal force version.
        # xyz_displ[:, 1] = -xyz_displ[:, 1]  # makes xyz be a true right hand axis!

        depth_surface = False
        if depth_surface:

            if not hasattr(self, 'depth_surf_fig'):
                plt.ion()
                self.depth_surf_fig, self.depth_surf_ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=[10, 10])
                self.i = 0
            self.i += 1

            self.depth_surf_ax.clear()

            num_interp = 200
            # cmap = cm.coolwarm
            cmap = cm.inferno
            # cmap = cm.plasma

            # first need to interpolate all valid data
            grid_x, grid_y = np.mgrid[0:z_displacement_all.shape[1]:(num_interp * 1j),
                                      0:z_displacement_all.shape[0]:(num_interp * 1j)]
            interpolated = griddata(cents, z_rescaled, (grid_x, grid_y), method='cubic', fill_value=0)

            surf = self.depth_surf_ax.plot_surface(
                grid_x, grid_y, interpolated.squeeze(), cmap=cmap, linewidth=0, antialiased=False,
                vmax=31, vmin=0)  # manually set based on result of true max below

            print(f"MAX: {interpolated.max()}")

            # Customize the z axis.
            self.depth_surf_ax.set_zlim(0.0, 100.0)

            self.depth_surf_ax.invert_yaxis()
            self.depth_surf_ax.grid(False)
            # self.depth_surf_ax.view_init(21.0, 32.0)

            # for plotting calibration/force reading examples in paper
            # stops = [11, 23, 35]
            # # stops = [37, 41, 45]

            # if self.i in stops:
            #     self.depth_surf_fig.savefig(f'/user/t.ablett/tmp/contact-il-plots/normal-depth-surface-{self.i}.png',
            #                                 bbox_inches='tight')
            #     import ipdb; ipdb.set_trace()

            # import os
            # plot_dir = '/user/t.ablett/tmp/contact-il-plots/depth_surf_sx3'
            # os.makedirs(plot_dir, exist_ok=True)
            # self.depth_surf_fig.savefig(os.path.join(plot_dir, f'{self.i}.png'),
            #                                 bbox_inches='tight')


        if get_average:
            xyz_displ_avg = xyz_displ.mean(axis=0)

            # optionally adjust z force based on selected parameters if shear is high
            # if self.cfg['z_shear_fix']['enable']:
            #     cfg = self.cfg['z_shear_fix']
            #     xy_norm = np.linalg.norm(xyz_displ_avg[:2])
            #     xyz_displ_avg[2] = xyz_displ_avg[2] - \
            #         cfg['subtract_mult'] * max((min(cfg['xy_norm_max'], xy_norm) - cfg['xy_norm_min']), 0)

            # try to find the center of touch based on depth only since xy displ is unreliable for this
            z_weighted_cents = np.expand_dims(xyz_displ[:, 2], -1) * cents
            z_sum = xyz_displ[:, 2].sum()
            z_cent = cents.mean(axis=0)

            if z_sum > 1e-8:
                z_cent = z_weighted_cents.sum(axis=0) / z_sum

            torque_cent = z_cent

            # torque about z is based purely on x + y
            dists_from_tcent = cents - torque_cent
            zy_torques = xyz_displ[:, 1] * dists_from_tcent[:, 0]
            # zy_torques = -zy_torques  # think we have to do this because xyz doesn't make a proper right hand frame
            zx_torques = xyz_displ[:, 0] * dists_from_tcent[:, 1]
            z_torque = zx_torques.mean() + zy_torques.mean()
            # print(f"TORQUE CENT: {torque_cent}")
            # print(f"Z TORQUE: {z_torque}")
            # print(f"TORQUE TIME: {time.time() - start_torque}")

            torque_avg = np.array([0, 0, z_torque * self.cfg['z_torque_scale']])  #TODO fill in!
            xyz_displ_avg = np.concatenate([xyz_displ_avg, torque_avg])

            # non_zeros = []
            # for i in range(3):
            #     valid = np.abs(xyz_displ[:, i]) > self.cfg['avg_mag_thresh']
            #     if np.any(valid):
            #         non_zeros.append(xyz_displ[:, i][valid])
            #     else:
            #         non_zeros.append(np.array([0]))

            # xyz_displ_avg = np.array([non_zeros[0].mean(), non_zeros[1].mean(), non_zeros[2].mean()])

        if show_image:
            if not hasattr(self, "threed_fig") or get_average and not hasattr(self, "avg_ax"):
                plt.ion()
                if get_average:
                    # self.threed_fig = plt.figure(figsize=plt.figaspect(0.333))
                    self.threed_fig = plt.figure(figsize=(14, 5))
                    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1])
                    # self.plot_ax = self.threed_fig.add_subplot(1, 3, 1, projection='3d')
                    self.plot_ax = self.threed_fig.add_subplot(gs[0], projection='3d')
                    # self.avg_ax = self.threed_fig.add_subplot(1, 3, 2, projection='3d')
                    self.avg_ax = self.threed_fig.add_subplot(gs[1], projection='3d')
                    # self.torque_ax = self.threed_fig.add_subplot(1, 3, 3)
                    self.torque_ax = self.threed_fig.add_subplot(gs[2])
                else:
                    self.threed_fig = plt.figure()
                    self.plot_ax = self.threed_fig.add_subplot(1, 1, 1, projection='3d')

            # reset plot
            self.plot_ax.clear()
            self.plot_ax.set_zlim([0, 20 * self.cfg['z_scale']])
            self.plot_ax.set_xlabel('x')
            self.plot_ax.set_ylabel('y')
            self.plot_ax.set_zlabel('z')
            self.plot_ax.invert_yaxis()
            # self.plot_ax.invert_xaxis()

            X, Y, Z = cents[:, 0], cents[:, 1], np.zeros_like(cents[:, 0])
            uvw = copy.deepcopy(xyz_displ)
            U, V, W = uvw[:, 0], uvw[:, 1], uvw[:, 2]
            self.plot_ax.quiver(X, Y, Z, U, V, W)

            if get_average:
                self.avg_ax.clear()
                self.avg_ax.set_xlim([-10, 10])
                self.avg_ax.set_ylim([-10, 10])
                self.avg_ax.set_zlim([0, 10])
                self.avg_ax.set_xlabel('x')
                self.avg_ax.set_ylabel('y')
                self.avg_ax.set_zlabel('z')
                self.avg_ax.invert_yaxis()
                # self.avg_ax.invert_xaxis()

                uvw = copy.deepcopy(xyz_displ_avg)
                U, V, W = uvw[0], uvw[1], uvw[2]
                self.avg_ax.quiver(0, 0, 0, U, V, W, color='r')

                # torque
                self.torque_ax.clear()
                self.torque_ax.set_title("Z Torque")
                self.torque_ax.set_xlim([0, 1])
                self.torque_ax.set_ylim([-15, 15])
                self.torque_ax.bar(.5, torque_avg[2])

                # this is a pain, so bar instead
                # radius = torque_avg[2]
                # if torque_avg[2] > 0:
                #     theta1 = -85
                #     theta2 = 265
                # else:
                #     theta1 = 265
                #     theta2 = -350
                # arc = Arc((0, 0), radius, radius, theta1=theta1, theta2=theta2 lw=torque_avg[2] / 5)
                # self.torque_ax.add_patch(arc)

            plt.draw()
            plt.pause(.0001)

            # paper/video plots
            # plot_dir = "/home/t.ablett/tmp/contact-il-plots/force_plots"
            # os.makedirs(plot_dir, exist_ok=True)
            # self.threed_fig.savefig(os.path.join(plot_dir, f"force_plot_{self.count}.png"), bbox_inches='tight')

        self.count += 1

        if get_average:
            return xyz_displ, xyz_displ_avg
        else:
            return xyz_displ


class ForceDetectorEndToEnd:
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.sts_transform = STSTransform(config_dir)
        self.marker_detector = MarkerDetectionCreator(config_dir).create_object()
        self.depth_detector = DepthDetector(config_dir)
        self.force_detector = ForceDetector(config_dir)

    def get_force(self, sts_raw_img, get_average=False, show_force_image=False):
        t_img = self.sts_transform.transform(sts_raw_img)
        marker_dict, marker_vals = self.marker_detector.detect(t_img)
        marker_vals, _ = self.marker_detector.filter_markers_average(t_img, marker_vals)
        _, depth = self.depth_detector.get_marker_depth(marker_dict['mask'])
        force_out = self.force_detector.get_force(self.marker_detector.markers_initial, marker_vals, depth,
            get_average=get_average, show_image=show_force_image)

        return force_out

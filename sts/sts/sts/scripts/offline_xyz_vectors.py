import copy
import os
import argparse
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

from sts.scripts.helper import FakeCamera
from sts.scripts.helper import read_json, plot
from sts.scripts.sts_transform import STSTransform
from sts.scripts.marker_detection.marker_detection import MarkerDetectionCreator
from sts.scripts.depth_from_markers import DepthDetector

"""
To run this script, you don't need ROS (yay!).

After changing into sts-cam-ros2/src/sts, run `pip install -e .` (probably in a conda or virtual env).

Then, to run it on a video, run something like:

```
python -m sts.scripts.offline_xyz_vectors ~/datasets/sts-cam-ros2-data/version-2/sample-video/bolt_v2.mov \
--threed_plot --save_vid --no_display`
```

To run it on a live camera, run something like:
```
python -m sts.scripts.offline_xyz_vectors /dev/video2 --config_dir $STS_CONFIG --threed_avg_plot --no_display
```
"""


class KalmanFilterMarkers:
    def __init__(self, config_dir, display=True):

        self._marker_detector = MarkerDetectionCreator(config_dir).create_object()
        self._sts_transform = STSTransform(config_dir)
        self._display = display

    def kf_on_markers(self, raw_img, transformed_img, img_dict, centroids):
        centroids, disp_image = self._marker_detector.filter_markers_kalman(transformed_img,  centroids)
        centroid_img_only = self._marker_detector.img_from_centroids(0*img_dict['mask'], centroids, color=[255,255,255])
        centroid_img_overlay = self._marker_detector.img_from_centroids(img_dict['mask'], centroids, color=[0,0,255])
        img_dict['img'] = raw_img
        img_dict['filtered_on_img'] = centroid_img_overlay
        img_dict['filtered'] = centroid_img_only
        img_dict['displacement'] = disp_image
        if self._display:
            plot(img_dict,  ['img', 'mask', 'centroids', 'displacement', 'filtered', 'filtered_on_img'])

        return img_dict, centroids


def main(args):
    cam = FakeCamera(args.source, not args.no_loop)
    sts_transform = STSTransform(args.config_dir)
    kf_markers = KalmanFilterMarkers(args.config_dir, not args.no_display)
    depth_detector = DepthDetector(args.config_dir)

    if args.threed_plot or args.threed_avg_plot:
        start_elev = 34
        start_azim = -70
        period = 50
        mag_change = 15
        threed_fig= plt.figure(figsize=plt.figaspect(0.5))
        img_ax = threed_fig.add_subplot(1, 2, 1)
        ax = threed_fig.add_subplot(1, 2, 2, projection='3d')

    if args.xy_xz_plot:
        fig, (xy_ax, xz_ax) = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8))

    if args.save_vid:
        cam.loop = False
        date_str = datetime.now().strftime("%m-%d-%y-%H_%M_%S")
        vid_dir = f"./force_vids/{date_str}"
        os.makedirs(vid_dir, exist_ok=True)

    while True:
        if not cam.loop:
            print(f"Frame {cam.count}/{cam.num_frames}")

        success, frame = cam.get_frame()
        if not success:
            break

        # detect markers, get xy displacment, get z displacement, form 3d vector
        transformed_img = sts_transform.transform(frame)
        img_dict, centroids = kf_markers._marker_detector.detect(transformed_img)
        img_dict, centroids = kf_markers.kf_on_markers(frame, transformed_img, img_dict, centroids)
        xy_centroid_displ = kf_markers._marker_detector.get_centroid_xy_displacement()
        z_displ_img, z_displ = depth_detector.get_marker_depth(
            cv2.cvtColor(img_dict['filtered'], cv2.COLOR_BGR2GRAY), return_displacement=True)

        # get only the depths at the centroid locations
        # cents = kf_markers._marker_detector.markers_initial.astype(np.float32)
        cents = kf_markers._marker_detector.markers_ref.astype(np.float32)
        valid_depths = cv2.remap(z_displ, cents[:, 0], cents[:, 1], interpolation=cv2.INTER_LINEAR)
        xyz_displ = np.concatenate([xy_centroid_displ, valid_depths], axis=-1)

        # make a "pretty" plot?
        cents_initial = kf_markers._marker_detector.markers_initial.astype(np.float32)

        ############## 3d Plot ###############
        if args.threed_plot:
            X, Y, Z = cents_initial[:, 0], cents_initial[:, 1], np.zeros_like(cents_initial[:, 0])
            uvw = copy.deepcopy(xyz_displ)
            U, V, W = uvw[:, 0], uvw[:, 1], uvw[:, 2]
            ax.clear()
            img_ax.clear()

            # these limits are hardcoded for the sts-cam-ros2-data/version-2/sample-video/ videos
            lims = [[150, 550], [50, 450], [0, 20]]
            ranges = [l[1] - l[0] for l in lims]

            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
            ax.set_zlim(*lims[2])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            # xyz values are reasonably on the same scale, but quiver plot is not b/c ranges in each axis don't match yet
            # therefore normalize u + v values based on x and y limits
            manual_xy_scale = .2
            x_scale = (ranges[0]) / (ranges[2]) * manual_xy_scale
            y_scale = (ranges[1]) / (ranges[2]) * manual_xy_scale
            U *= x_scale
            V *= y_scale

            overall_scale = .5
            U *= overall_scale
            V *= overall_scale
            W *= overall_scale

            ax.invert_yaxis()  # to match standard computer vision/other images

            ax.view_init(elev=start_elev + mag_change * np.sin(cam.count * 2 * np.pi / period),
                         azim=start_azim + mag_change * np.cos(cam.count * 2 * np.pi / period))

            ax.quiver(X, Y, Z, U, V, W)

            rgb_order = np.flip(img_dict['displacement'], 2)
            img_ax.imshow(rgb_order)

            if args.save_vid:
                threed_fig.savefig(os.path.join(vid_dir, f"{str(cam.count).zfill(4)}.png"))

        ############# XY, XZ Plot #############
        if args.xy_xz_plot:
            X, Y, Z = cents_initial[:, 0], cents_initial[:, 1], cents_initial[:, 1]
            uvw = copy.deepcopy(xyz_displ)
            U, V, W = uvw[:, 0], uvw[:, 1], uvw[:, 2]
            xy_ax.clear()
            xz_ax.clear()
            xy_ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=.3)
            xz_ax.quiver(X, Z, U, -2*W, angles='xy', scale_units='xy', scale=.3)

            xy_ax.invert_yaxis()
            xz_ax.invert_yaxis()

            xy_ax.set_title("XY forces")
            xz_ax.set_title("XZ forces")

        ############## 3d avg Plot ###############
        if args.threed_avg_plot:
            ax.clear()
            img_ax.clear()
            X, Y, Z = [0, 0, 0]
            U, V, W = xyz_displ.mean(axis=0)
            W *= 5  # to put in similar range..XXX this will be sensor dependent
            lims = [[-10, 10], [-10, 10], [0, 20]]
            ranges = [l[1] - l[0] for l in lims]

            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
            ax.set_zlim(*lims[2])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            ax.invert_yaxis()
            ax.view_init(elev=start_elev + mag_change * np.sin(cam.count * 2 * np.pi / period),
                         azim=start_azim + mag_change * np.cos(cam.count * 2 * np.pi / period))

            ax.quiver(X, Y, Z, U, V, W)

            rgb_order = np.flip(img_dict['displacement'], 2)
            img_ax.imshow(rgb_order)

            if args.save_vid:
                threed_fig.savefig(os.path.join(vid_dir, f"{str(cam.count).zfill(4)}.png"))

        # if (args.threed_plot or args.xy_xz_plot or args.threed_avg_plot) and not args.save_vid:
        if (args.threed_plot or args.xy_xz_plot or args.threed_avg_plot):
            plt.draw()
            plt.pause(.0001)

        print(f"x max: {np.abs(xyz_displ[:, 0]).max()}, y max: {np.abs(xyz_displ[:, 1]).max()}, "\
              f"z max: {np.abs(xyz_displ[:, 2]).max()}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help="Video file with sts data.")
    parser.add_argument('--config_dir', default=os.path.join(os.environ['STS_PARENT_DIR'], "sts-cam-ros2/configs/demo"))
    parser.add_argument('--no_loop', action='store_true', help="Don't loop video.")
    parser.add_argument('--no_display', action='store_true', help="Don't display.")
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--threed_plot', action='store_true')
    parser.add_argument('--threed_avg_plot', action='store_true')
    parser.add_argument('--xy_xz_plot', action='store_true')
    parser.add_argument('--save_vid', action='store_true')
    args = parser.parse_args()

    if args.profile:
        import cProfile
        from pstats import Stats

        pr = cProfile.Profile()
        pr.enable()

    plt.ion()

    main(args)

    if args.profile:
        pr.disable()
        stats = Stats(pr)
        stats.sort_stats('tottime').print_stats(30)
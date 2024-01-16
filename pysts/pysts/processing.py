import sys
import os
import timeit
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from sts.scripts.flow import STSOpticalFlow
from sts.scripts.marker_detection.marker_detection import MarkerDetectionCreator
from sts.scripts.depth_from_markers import DepthDetector
from sts.scripts.helper import read_json
from sts.scripts.sts_transform import STSTransform
from sts.scripts.force import ForceDetector

from pysts.sts import PySTS, SimPySTS


def normalize_depth(img):
    min_depth = np.min(img)
    max_depth = np.max(img)
    depth_data = (img - min_depth) / (max_depth - min_depth)
    return depth_data


def binarize_gray_read(img_file):
    return (
        cv2.cvtColor(
        cv2.imread(img_file), cv2.COLOR_BGR2GRAY
        ) / 255
    ).astype('uint8')


def read_json_with_error(file):
    try:
        return read_json(file)
    except Exception as e:
        print(f'Error reading/parsing config file {file}')
        sys.exit(0)




class STSProcessor:
    """ A tool for acquiring and processing STS data. """
    FLOW_MODE_CHECKS = {'flow', 'flow_image', 'slip', 'slip_image'}
    MARKER_FLOW_MODE_CHECKS = {'marker_flow', 'marker_flow_image', 'slip', 'slip_image'}
    DEPTH_MODE_CHECKS = {'depth', 'depth_image', 'contact', 'contact_image', 'force', 'avg_force'}
    CONTACT_MODE_CHECKS = {'contact', 'contact_image'}
    IN_CONTACT_MODE_CHECKS = {'in_contact'}
    FORCE_MODE_CHECKS = {'force'}
    AVG_FORCE_MODE_CHECKS = {'force', 'avg_force'}
    BGR_MAP = {'b': 0, 'g': 1, 'r': 2}
    HSV_MAP = {'h': 0, 's': 1, 'v': 2}
    DTYPE_NUM_CHANNELS = {'flow': 2, 'marker_flow': 2, 'depth': 1, 'contact': 1, 'marker': 1}
    DTYPE_SIZE = {'avg_force': 6, 'in_contact': 1}
    MODE_SWITCH_THRESH_DEFAULTS = {
        'displacement': dict(tac=0.5, vis=0.25),
        'depth': dict(tac=1.0, vis=0.5),
        'contact' :dict(tac=1200, vis=200),
        'internal_ft': dict(tac=10, vis=5),
    }
    FILTER_DISPLACEMENT_CHECKS = {'displacement', 'disp_no_switch'}
    MODE_SWITCH_DEFAULTS = {
        'initial_mode': "visual",
        'mode_switch_type': "displacement",  # none (string), displacement, depth, contact, internal_ft
        'mode_switch_req_ts': 4,
        'tac_thresh': None,  # use default if None
        'vis_thresh': None,  # use default if None
        'tactile_mode_object_flow_channel': 'b',
        'mode_switch_in_thread': True,
        'no_switch_override': False,  # if true, do all measurements for switch but don't actually switch
    }
    def __init__(
        self,
        config_dir: str,  # same config dir as all other sts code
        allow_both_modes: bool = False,
        resolution="",
        sensor_sim_vid: str = None,  # if set, use this file as a simulated sensor
        time_debug: bool = False,  # debug timing of processing
        filter_markers: str = "average",
        mode_switch_opts: dict = MODE_SWITCH_DEFAULTS,
    ):
        print("Starting STS Processor.")
        if sensor_sim_vid:
            self.sts = SimPySTS(source_vid=sensor_sim_vid, camera_resolution=resolution)
            self.sim = True
        else:
            self.sts = PySTS(config_dir=config_dir, camera_resolution=resolution)
            self.sim = False

        self._allow_both_modes = allow_both_modes
        self._mode = None
        self._filter_markers = filter_markers
        self._mode_switch_type = "none"
        self.time_debug = time_debug
        self._tactile_mode_object_flow_channel = mode_switch_opts.get('tactile_mode_object_flow_channel',
            STSProcessor.MODE_SWITCH_DEFAULTS['tactile_mode_object_flow_channel'])
        self._mode_switch_in_thread = False

        self._initial_mode = mode_switch_opts['initial_mode']

        # mode switching options not from config dir
        if self._allow_both_modes:

            # add in defaults if they weren't set -- test this
            if mode_switch_opts != STSProcessor.MODE_SWITCH_DEFAULTS:
                for k in STSProcessor.MODE_SWITCH_DEFAULTS:
                    if k not in mode_switch_opts:
                        mode_switch_opts[k] = STSProcessor.MODE_SWITCH_DEFAULTS[k]

            self._initial_mode = mode_switch_opts['initial_mode']

            if self._initial_mode == 'tactile':
                self._alternate_mode = 'visual'
            elif self._initial_mode == 'visual':
                self._alternate_mode = 'tactile'
            else:
                self._alternate_mode = self._initial_mode

            self._mode_switch_type = mode_switch_opts['mode_switch_type']
            self.mode_switch_req_ts = mode_switch_opts['mode_switch_req_ts']

            self.tac_thresh = mode_switch_opts['tac_thresh'] if mode_switch_opts['tac_thresh'] is not None else \
                STSProcessor.MODE_SWITCH_THRESH_DEFAULTS[self._mode_switch_type]['tac']
            self.vis_thresh = mode_switch_opts['vis_thresh'] if mode_switch_opts['vis_thresh'] is not None else \
                STSProcessor.MODE_SWITCH_THRESH_DEFAULTS[self._mode_switch_type]['vis']

            self._tactile_mode_object_flow_channel = mode_switch_opts['tactile_mode_object_flow_channel']
            self._mode_switch_in_thread = mode_switch_opts['mode_switch_in_thread']
            self._no_switch_override = mode_switch_opts['no_switch_override']
            self._in_contact = False
            self._in_contact_count = 0
            self._not_in_contact_count = 0

            self._num_contact_history = np.zeros(10)

            if self._mode_switch_type == 'internal_ft':
                raise NotImplementedError("Not exposed by polymetis! Needs to be added there first.")
                self._ft_for_filter = None
                self._ft_filter_size = 20

            if self._filter_markers != "":
                # we don't want to use a separate set of mode options when we're doing marker filtering/tracking
                self._mode_dirs = {
                    'tactile': '',  # just use the default configs, that aren't in a subfolder
                    'visual': '',
                    'halfway': '',
                }
            else:
                raise NotImplementedError("Consider deprecating the mode folders!")
                self._mode_dirs = {
                    'tactile': '',  # just use the default configs, that aren't in a subfolder
                    'visual': 'visual',
                    'halfway': '',
                }

        self._mode_dirs = {
            'tactile': '',  # just use the default configs, that aren't in a subfolder
            'visual': '',
            'halfway': '',
        }

        self.load_config(config_dir)

    def load_config(self, config_dir):
        """ Load things that are from the sts config dir, as opposed to the arguments to this class, here. """
        self._config_dir = config_dir

        # sts libary objects
        self._marker_detector = MarkerDetectionCreator(config_dir).create_object()
        self._flow = STSOpticalFlow(config=read_json_with_error(os.path.join(config_dir, "object_flow.json")))
        self._marker_flow = STSOpticalFlow(config=read_json_with_error(os.path.join(config_dir, "marker_flow.json")))
        self._depth = DepthDetector(config_dir=config_dir)
        self._force = ForceDetector(config_dir=config_dir)
        self._sts_transform = STSTransform(config_dir, mode_dir='')  # will reload if starting in visual

        # sts masks/refs
        self._depth_markers_ref = 1 - binarize_gray_read(os.path.join(config_dir, 'markers_reference.png'))

        self._depth_mask = binarize_gray_read(os.path.join(config_dir, 'mask.png'))

        # sts contact TODO if/when the STS version of this is cleaned up, add it here
        self._contact_config = read_json_with_error(os.path.join(config_dir, "contact_detection_depth.json"))

        self._depth_markers_refs = {
            'tactile': self._depth_markers_ref,
            'visual': 1 - binarize_gray_read(os.path.join(config_dir, self._mode_dirs['visual'], 'markers_reference.png')),
            'halfway': self._depth_markers_ref,
        }

        if self._allow_both_modes:
            self._depth_masks = {
                'tactile': self._depth_mask,
                'visual': binarize_gray_read(os.path.join(config_dir, self._mode_dirs['visual'], 'mask.png')),
                'halfway': self._depth_mask
            }

            mode_dir = self._mode_dirs[self._initial_mode]

            self._sts_transform = STSTransform(config_dir, mode_dir=mode_dir)
            self._marker_detector = MarkerDetectionCreator(config_dir, mode_dir=mode_dir).create_object()

            self._depth_markers_ref = self._depth_markers_refs[self._initial_mode]
            self._depth.markers_ref = self._depth_markers_ref
            self._depth_mask = self._depth_masks[self._initial_mode]



        # initialize mode
        self.set_mode(self._initial_mode)

        # allow sensor to get into mode
        time.sleep(.2)

    def reset(self):
        if self._allow_both_modes:
            self._in_contact = False
            self._in_contact_count = 0
            self._not_in_contact_count = 0

            self.set_mode(self._initial_mode)
            time.sleep(.2)

        # if use source vid for sensor, reset it
        if isinstance(self.sts, SimPySTS):
            self.sts.cam.reset_video()

    def get_processed_sts(self,
                          modes: set={
                            'raw_image',
                            'marker',
                            'marker_image',
                            'marker_dots',
                            'flow',
                            'flow_image',
                            'marker_flow',
                            'marker_flow_image',
                            'marker_displacement',
                            'depth',
                            'depth_image',
                            'contact',
                            'contact_image',
                            'force',
                            'avg_force'
                          },
                          displays: tuple = ()):
        """
        Allows the caller to get as many modes as they want, automatically sorting out mode dependencies.
        modes can be a list/tuple of strings, all options are given as defaults.
        displays can be a list/tuple of the same strings, and any that are entered are displayed by cv2 (if possible).
        """
        ret_dict = dict()

        ################ get latest image ####################
        raw_img = self.sts.get_image()
        if 'raw_image' in modes: ret_dict['raw_image'] = raw_img

        ################ STS processing ####################
        start_pro_tic = timeit.default_timer()

        ####### Markers #######
        if not modes == {'raw_image'} or self._mode_switch_type != 'none':
            marker_start = timeit.default_timer()
            transformed_image = self._sts_transform.transform(raw_img)
            marker_dict, marker_vals = self._marker_detector.detect(transformed_image)

            if (self._filter_markers != "" or self._mode_switch_type == 'displacement') and len(marker_vals) > 0:
                kf_start = timeit.default_timer()
                if self._filter_markers == 'kalman':
                    marker_vals, disp_image = self._marker_detector.filter_markers_kalman(transformed_image,  marker_vals)
                elif self._filter_markers == 'average':
                    marker_vals, disp_image = self._marker_detector.filter_markers_average(transformed_image,  marker_vals)
                else:
                    raise NotImplementedError(f"Not implemented for filter_markers argument {self._filter_markers}")

                # create the mask used for depth from the output centroids from the filter
                marker_dict['mask'] = self._marker_detector.gray_img_from_centroids(
                    np.zeros_like(marker_dict['mask']), marker_vals,
                    radius=self._marker_detector.marker_displacement_config['centroid_radius'])

                # TODO this monstrosity of fixes for shear + force, if we ever need them, doesn't really work, but
                # there might be something we need here later?
                # if True:

                #     # TODO delete if this doesn't work
                #     # going to try to normalize dot positions if there's an excess of xy displacement
                #     marker_displ = marker_vals - self._marker_detector.markers_initial
                #     marker_displ_mean = marker_displ.mean(axis=0)
                #     marker_displ_norm = np.linalg.norm(marker_displ, axis=-1)

                    # print(f"marker displ mean: {marker_displ_mean}")

                    # test subtracting mean from all
                    # marker_displ_mean_sub = marker_displ - marker_displ_mean
                    # depth_markers = self._marker_detector.markers_initial + marker_displ_mean_sub

                    # from watching the video, during shear, the dots near contact all move _a lot_, but not much
                    # relative to one another...while the dots not near contact also all start to move
                    # 38 is the first frame in x-mid-push0 that the shear really starts to have a large effect
                    # if self.sts.cam.count > 35:
                    #     print(f"marker displ mean: {marker_displ_mean}")
                    #     import ipdb; ipdb.set_trace()

                    # 2 things:
                    # 1. the current method for detecting x/y motion is flawed...we detect x or y motion
                    #    just based on where the sensor is touched (e.g. if touched near the top, we'll get lots of
                    #    positive/negative movement in y, even though there is none)
                    # 2. given 1., we might have to rethink how we're doing any type of shear detection for then
                    #    fixing depth, because we might want to also consider the center of the place the sensor is
                    #    being touched.
                    # these things can be at least partially fixed by ensuring the sensor is touched

                    # if self.sts.cam.count > 35:
                    #     import ipdb; ipdb.set_trace()

                    # if marker_displ_norm > max_norm:

                    # to_sub = (marker_displ_norm - max_norm) * marker_displ_mean / marker_displ_norm

                    # this actually makes no sense!
                    # to_sub = np.maximum(marker_displ - max_norm, 0) * marker_displ / np.expand_dims(np.maximum(marker_displ_norm, 1e-8), 1)
                    # depth_markers = marker_vals - to_sub

                    # goal: to make displacement of all dots be maximized at a particular amount..hmm probably not ideal
                    # actually this might be just fine. let's do it.
                    # this works reasonably well...but let's try a few more
                    # max_norm = 3.0
                    # marker_displ_dir = marker_displ / np.expand_dims(np.maximum(marker_displ_norm, 1e-8), -1)
                    # fixed_marker_displ_norm = np.minimum(marker_displ_norm, max_norm)
                    # depth_markers = self._marker_detector.markers_initial + marker_displ_dir * np.expand_dims(fixed_marker_displ_norm, -1)

                    # marker_dict['mask'] = self._marker_detector.gray_img_from_centroids(
                    #     np.zeros_like(marker_dict['mask']), depth_markers,
                    #     radius=self._marker_detector.marker_displacement_config['centroid_radius'])
                    # print(f"old mean displ {marker_displ.mean(axis=0)}, new mean displ {(depth_markers - self._marker_detector.markers_initial).mean(axis=0)}")


                if 'marker_displacement' in  modes: ret_dict['marker_displacement'] = disp_image
                if self.time_debug: print(f"KF time: {timeit.default_timer() - kf_start}")

            if 'marker' in modes: ret_dict['marker'] = marker_vals
            if 'marker_image'in modes: ret_dict['marker_image'] = marker_dict['mask']
            if 'marker_dots'in modes: ret_dict['marker_dots'] = marker_dict['centroids']
            if self.time_debug: print(f"marker time: {timeit.default_timer() - marker_start}")

        ####### Flow ######
        # TODO flow is computed on the transformed images...should it be on the raw images instead?
        if modes & STSProcessor.FLOW_MODE_CHECKS:
            start_flow_tic = timeit.default_timer()

            if self._tactile_mode_object_flow_channel is not None:
                ch = self._tactile_mode_object_flow_channel
                if ch in 'bgr':
                    bgr_channels = cv2.split(transformed_image)  # in bgr order
                    self._add_flow_to_dict(bgr_channels[STSProcessor.BGR_MAP[ch]],
                        ret_dict, modes, pre='', debug_start_time=start_flow_tic)
                elif ch in 'hsv':
                    hsv_img = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2HSV)
                    hsv_channels = cv2.split(hsv_img)  # in bgr order
                    self._add_flow_to_dict(hsv_channels[STSProcessor.HSV_MAP[ch]],
                        ret_dict, modes, pre='', debug_start_time=start_flow_tic)
                else:
                    raise NotImplementedError(f"Not implemented for tactile_mode_object_flow_channel {ch}")
            else:
                self._add_flow_to_dict(transformed_image, ret_dict, modes, pre='', debug_start_time=start_flow_tic)

        if modes & STSProcessor.MARKER_FLOW_MODE_CHECKS:
            start_tacflow_tic = timeit.default_timer()
            self._add_flow_to_dict(marker_dict['mask'], ret_dict, modes, pre='marker_', debug_start_time=start_tacflow_tic)

        ####### Depth #######
        if modes & STSProcessor.DEPTH_MODE_CHECKS or self._mode_switch_type == 'depth' or self._mode_switch_type == 'contact':
            if self._filter_markers != "":
                self._depth.update_markers_ref(self._marker_detector.gray_markers_img_ref)

            start_depth_tic = timeit.default_timer()
            if 'depth_surface' in displays:
                depth_image, depth = self._depth.get_marker_depth(marker_dict['mask'], mask=1, return_displacement=True,
                                                                  display=True)
            else:
                depth_image, depth = self._depth.get_marker_depth(marker_dict['mask'], mask=1, return_displacement=True)
            if 'depth_image' in modes: ret_dict['depth_image'] = depth_image
            if 'depth' in modes: ret_dict['depth'] = depth
            if self.time_debug: print(f"depth time: {timeit.default_timer() - start_depth_tic}")

        ####### Force #######
        if modes & STSProcessor.FORCE_MODE_CHECKS or modes & STSProcessor.AVG_FORCE_MODE_CHECKS:
            assert self._filter_markers != "", "Must use something to filter/estimate marker displacements for force"
            force, avg_force = self._force.get_force(self._marker_detector.markers_initial, marker_vals, depth,
                get_average='avg_force' in modes, show_image = 'force' in displays or 'avg_force' in displays)
            if 'force' in modes: ret_dict['force'] = force
            if 'avg_force' in modes: ret_dict['avg_force'] = avg_force
            # np.set_printoptions(suppress=True, precision=4)
            # print(f"FORCE: {avg_force}")

        ####### Contact (manually implemented here until STS version is cleaned up) #######
        if modes & STSProcessor.CONTACT_MODE_CHECKS or self._mode_switch_type == 'depth' or self._mode_switch_type == 'contact':
            start_contact_tic = timeit.default_timer()
            marker_indices = np.where(marker_dict['mask'] > 1)
            values = depth_image[marker_indices][:,2]

            if modes & STSProcessor.CONTACT_MODE_CHECKS or self._mode_switch_type == 'contact':
                marker_mask = self._depth_mask * marker_dict['mask']
                points = np.vstack(marker_indices).transpose()

                x=np.linspace(0, marker_mask.shape[0], num=marker_mask.shape[0])
                y=np.linspace(0, marker_mask.shape[1], num=marker_mask.shape[1])
                grid_x, grid_y = np.meshgrid(x, y)

                contact_mask = np.zeros_like(marker_dict['mask'])
                if len(points) > 0 and len(points) < 6000:  # XXX: <3000 is just to fix error where every single point is white
                    interpolated_values = np.nan_to_num(griddata(points, values, (grid_x, grid_y), method='linear')).transpose().astype('uint8')
                    indices = np.where(interpolated_values > self._contact_config['thresh'])  # 5 works well
                    contact_mask[indices] = 255
                contact_image = cv2.cvtColor(contact_mask, cv2.COLOR_GRAY2BGR)
                if 'contact_image' in modes: ret_dict['contact_image'] = contact_image

                if self.time_debug: print(f"contact time: {timeit.default_timer() - start_contact_tic}")

        ####### Mode switching #######
        if self._mode_switch_type != "none":
            if self._mode_switch_type == 'depth':
                max_nonoutlier = values.mean() + 3 * values.std()
                values_nonoutlier = values[values <= max_nonoutlier]
                mode_switch_val = values_nonoutlier.mean()
            elif self._mode_switch_type == 'displacement':
                mode_switch_val = np.linalg.norm(self._marker_detector.displ_to_add, axis=1).mean()
            elif self._mode_switch_type == 'contact':
                mode_switch_val = contact_mask.sum()
            elif self._mode_switch_type == 'internal_ft':
                raise NotImplementedError("Internal FT not yet exposed by polymetis.")

            # print(f"MODE SWITCH VAL: {mode_switch_val}")

            if not self._in_contact:
                self._in_contact_count = self._in_contact_count + 1 if mode_switch_val > self.tac_thresh else 0
            else:
                self._not_in_contact_count = self._not_in_contact_count + 1 if mode_switch_val < self.vis_thresh else 0

            # self._num_contact_history = np.roll(self._num_contact_history, shift=1)
            # self._num_contact_history[0] = mean_depth
            # print(f"num contact pixels: {mean_depth}, avg: {self._num_contact_history.mean()}")

            if self._in_contact_count >= self.mode_switch_req_ts and not self._in_contact:
                if not self._no_switch_override:
                    self.set_mode('tactile')
                self._in_contact = True
                self._not_in_contact_count = 0
            elif self._not_in_contact_count >= self.mode_switch_req_ts and self._in_contact:
                if not self._no_switch_override:
                    self.set_mode('visual')
                self._in_contact = False
                self._in_contact_count = 0

        ################ In Contact ###################
        if modes & STSProcessor.IN_CONTACT_MODE_CHECKS:
            assert self._mode_switch_type != "none", "In contact requires mode_switch_type to not be none. "\
                "If you want to use it without actually switching modes, also set no_switch_override to True. "\
                "Both of these options are part of mode_switch_opts dict."
            ret_dict['in_contact'] = int(self._in_contact)

        if self.time_debug: print(f"total time: {timeit.default_timer() - start_pro_tic}")

        ################ Display ####################
        for disp_mode in displays:
            # assert disp_mode in ret_dict, f"Requested display option {disp_mode} not in manually entered modes {modes}"
            # assert "image" in disp_mode, f"Cannot display non-image mode {disp_mode}"
            if disp_mode == 'depth_image':
                frame = np.concatenate((cv2.cvtColor(marker_dict['mask'], cv2.COLOR_GRAY2BGR), depth_image), axis=1)
                frame = np.concatenate((frame, cv2.cvtColor((1-self._depth.markers_ref) * 255, cv2.COLOR_GRAY2BGR)), axis=1)
                cv2.imshow('depth_image', frame)
            elif disp_mode in ['flow', 'marker_flow']:
                frame = np.concatenate([ret_dict[disp_mode],
                    np.zeros([ret_dict[disp_mode].shape[0], ret_dict[disp_mode].shape[1], 1])], axis=2)
                cv2.imshow(disp_mode, frame)
            elif 'force' in disp_mode:  # taken care of in the sts code directly since it's mpl
                pass
            elif disp_mode == 'depth_surface':
                pass  # taken care of in sts code
            else:
                cv2.imshow(disp_mode, ret_dict[disp_mode])
        if len(displays) > 0:
            cv2.waitKey(1)

        # for plotting calibration/force reading examples in paper
        # stops = [11, 23, 35]
        # # stops = [37, 41, 45]

        # if self.sts.cam.count in stops:
        #     cv2.imwrite(f'/user/t.ablett/tmp/contact-il-plots/normal-again-0-{self.sts.cam.count}.png',
        #                 ret_dict['marker_displacement'])
        #     import ipdb; ipdb.set_trace()

        # if self.sts.cam.loop_count > 0:
        #     import sys
        #     sys.exit(0)
        # else:
        #     plot_dir = '/user/t.ablett/tmp/contact-il-plots/disp_plots_shearx3/'
        #     os.makedirs(plot_dir, exist_ok=True)
        #     cv2.imwrite(os.path.join(plot_dir, f'disp_{self.sts.cam.count}.png'), ret_dict['marker_displacement'])

        return ret_dict

    def _add_flow_to_dict(self, img, ret_dict, modes=('flow', 'flow_image'), pre='', debug_start_time=None):
        try:
            flow_img, Vx_flow, Vy_flow = getattr(self, f"_{pre}flow").detect(img)  # if this is too slow, can switch to compute_flow
        except:
            raise NotImplementedError(f"flow not implemented for pre {pre}")

        if pre + 'flow' in modes: ret_dict[pre + 'flow'] = np.transpose(np.vstack([Vx_flow, Vy_flow]), [1, 2, 0])
        if pre + 'flow_image' in modes: ret_dict[pre + 'flow_image'] = flow_img
        if self.time_debug and debug_start_time is not None:
            print(f"{pre}flow time: {timeit.default_timer() - debug_start_time}")

    def set_mode(self, mode: str):
        if self._mode == mode:
            return

        start_mode_time = timeit.default_timer()

        if self._filter_markers == "":
            # do internal class changes
            mode_dir = self._mode_dirs[mode]
            self._marker_detector = MarkerDetectionCreator(self._config_dir, mode_dir=mode_dir).create_object()
            self._depth.markers_ref = self._depth_markers_refs[mode]
            self._sts_transform = STSTransform(self._config_dir, mode_dir=mode_dir)

        # for rectangular sensor, depth mask is taken care of at the point of getting dots
        # self._depth.mask = self._depth_masks[mode]
        self._mode = mode

        # set LEDs + camera params
        self.sts.set_mode(mode, blocking=not self._mode_switch_in_thread)
        if self.time_debug:
            print(f"mode switch time: {timeit.default_timer() - start_mode_time}")
import os
import cv2
import json
import numpy as np
from copy import deepcopy
from scipy.interpolate import griddata
from scipy.spatial import KDTree

from sts.scripts.helper import read_json

class MarkerDetectionCreator:
    """The Factory Class"""
    def __init__(self, config_dir, mode_dir=""):
        self.config_dir = os.path.join(config_dir, mode_dir)

    def create_object(self, ):
        tmp = os.path.join(self.config_dir, "tactile.json")
        tactile_config = read_json(tmp)

        """A static method to get a concrete product"""
        if tactile_config['marker_detection'] == "hsv":
            from sts.scripts.marker_detection.marker_detection_hsv import MarkerDetectionHSV
            return MarkerDetectionHSV(self.config_dir)
        elif tactile_config['marker_detection'] == "adaptive":
            from sts.scripts.marker_detection.marker_detection_adaptive import MarkerDetectionAdaptive
            return MarkerDetectionAdaptive(self.config_dir)
        elif tactile_config['marker_detection']  == "unet":
            from sts.scripts.marker_detection.marker_detection_unet import MarkerDetectionUnet
            return MarkerDetectionUnet(self.config_dir)
        elif tactile_config['marker_detection'] == "hue":
            from sts.scripts.marker_detection.marker_detection_hue import MarkerDetectionHUE
            return MarkerDetectionHUE(self.config_dir)

class MarkerDetection(object):
    """An abstract class for marker detection for the STS"""

    def __init__(self, config_dir):
        """Config is a dictionary of configuration parameters"""
        self.config_dir = config_dir
        self.img_ref = cv2.imread(os.path.join(self.config_dir,'reference.png'))
        self.marker_displacement_config = read_json(os.path.join(self.config_dir,'marker_displacement.json'))
        self.mask = cv2.imread(os.path.join(self.config_dir, 'mask.png'))
        self.mask = (self.mask / 255).astype('uint8')
        self.rectangular_sensor = json.load(open(os.path.join(self.config_dir, 'tactile.json'), 'r'))['mask'] == 'rectangular'
        if self.rectangular_sensor:
            self.marker_dict, self.markers_ref = self.detect(self.img_ref)
        else:
            self.marker_dict, self.markers_ref = self.detect(self.img_ref * self.mask)

        if self.marker_displacement_config['ignore_edge_dots']:
            self.markers_ref = self._remove_edge_markers(self.markers_ref)

        self.prev_markers_ref = deepcopy(self.markers_ref)

        self.markers_initial= deepcopy(self.markers_ref)
        self.markers_img_ref = self.img_from_centroids(np.zeros_like(cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)), self.markers_ref, color=[255,255,255],
            radius=self.marker_displacement_config['centroid_radius'])
        self.gray_markers_img_ref = self.gray_img_from_centroids(
            np.zeros_like(self.img_ref), self.markers_ref, color=255, radius=self.marker_displacement_config['centroid_radius'])
        self.displ_to_add = np.zeros(self.markers_ref.shape)
        self.prev_displ_to_add = np.zeros(self.markers_ref.shape)
        self._initialize_filter()
        self.reset_counter = 0
        self.reset_img_distance = 0
        self._initialize_average_filter()

        self.first = True

    def _remove_edge_markers(self, dots):
        h, w, _ = self.img_ref.shape
        lim = self.marker_displacement_config['edge_pix_limits']
        dots = dots[dots[:, 0] >= lim]
        dots = dots[dots[:, 1] >= lim]
        dots = dots[dots[:, 0] <= (w - 1) - lim]
        dots = dots[dots[:, 1] <= (h - 1) - lim]
        return dots

    def _initialize_filter(self,):
        from filterpy.kalman import KalmanFilter
        from filterpy.common import Q_discrete_white_noise, inv_diagonal
        self.kalman_filter = KalmanFilter(dim_x=len(self.markers_ref.flatten()), dim_z=len(self.markers_ref.flatten()))
        self.kalman_filter.x = self.markers_ref.flatten()
        self.kalman_filter.F = np.eye(len(self.markers_ref.flatten()))
        self.kalman_filter.H = np.eye(len(self.markers_ref.flatten()))
        #IC/prior covariance
        self.kalman_filter.P *= self.marker_displacement_config['std_IC']**2
        #Process/dynamics noise
        self.kalman_filter.Q *= self.marker_displacement_config['std_dynamics']**2
        #Significantly speed up calculations with diagonal inv since uncertainties are diagonal
        self.kalman_filter.inv = inv_diagonal

    def _initialize_average_filter(self):
        self.undetected_count = np.zeros(self.markers_ref.shape[0])
        self.simul_undetected_count = np.zeros(self.markers_ref.shape[0])

        tree = KDTree(self.markers_initial)
        dists, self.nbr_inds = tree.query(self.markers_initial, k=self.marker_displacement_config['num_nbrs'] + 1)
        # remove self as neighbour
        dists = dists[:, 1:]
        self.nbr_inds = self.nbr_inds[:, 1:]

    def load_config(self, config_dir, config_file):
        """Load a config file, could be useful for switching between visual and tactile mode"""
        config = read_json(os.path.join(config_dir, config_file))
        config['config_file'] = config_file
        self.config = config

    def detect(self, img):
        """The actual detection process defined by sub-classes"""
        raise NotImplementedError()

    def img_from_centroids(self, _img, centroids, color=[0,0,255], radius=5):
        """generate marker mask image from list of marker centroid locations"""
        img = deepcopy(_img)
        img_color = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_GRAY2BGR)
        for i, centroid in enumerate(centroids):
            cv2.circle(img_color, (int(centroid[0]), int(centroid[1])), radius=radius, color=color, thickness=-1)
        return img_color

    def gray_img_from_centroids(self, _img, centroids, color=255, radius=5):
        """generate marker mask image from list of marker centroid locations"""
        img = deepcopy(_img)
        for i, centroid in enumerate(centroids):
            cv2.circle(img, (int(centroid[0]), int(centroid[1])), radius=radius, color=255, thickness=-1)
        return img

    def get_centroids(self, dots_img):
        """Find centroids from a grey scale 1 channel dots image"""
        min_area = self.config["min_area"]
        max_area = self.config["max_area"]
        binary = (dots_img / 252).astype("uint8")
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        centroid_list = []
        for i, centroid in enumerate(centroids):
            if ((i > 0) and (stats[i][-1] > min_area) and (stats[i][-1] < max_area)):
                centroid_list.append(centroid)
        return centroid_list

    def get_calibration_params(self, img):
        return NotImplementedError

    def reset_references(self, img):
        """
        Reset marker displacement references when no contact is detected (make sure errors are eliminated when state is known)
        """
        self.reset_counter +=1
        # print ('RESET MARKER DEFAULT', self.reset_counter)
        if self.marker_displacement_config['reset_ref_img']:
            self.img_ref = img  # keep img_ref using the original ref
        if self.rectangular_sensor:
            self.marker_dict, self.markers_ref = self.detect(self.img_ref)
        else:
            self.marker_dict, self.markers_ref = self.detect(self.img_ref * self.mask)
        if self.marker_displacement_config['ignore_edge_dots']:
            self.markers_ref = self._remove_edge_markers(self.markers_ref)
        self.prev_markers_ref = deepcopy(self.markers_ref)
        self.markers_initial = deepcopy(self.markers_ref)
        self.gray_markers_img_ref = self.gray_img_from_centroids(
            np.zeros_like(self.img_ref), self.markers_ref, color=255, radius=self.marker_displacement_config['centroid_radius'])
        self.displ_to_add= np.zeros(self.markers_ref.shape)
        self.prev_displ_to_add= np.zeros(self.markers_ref.shape)
        self._initialize_filter()
        self._initialize_average_filter()

    def is_contact(self, markers, img, max_shift_len=30):
        """
        Determine if object is in contact or not based on marker displacement (relative to initial image)
        """
        markers_reference, markers_paired, displ_to_add, marker_indices, valid_indices= self.find_marker_associations(markers, self.markers_initial, self.displ_to_add*0, max_shift_len=max_shift_len)
        self.reset_img_distance = np.linalg.norm(markers_paired  - markers_reference)
        # print(f"reset img dist: {self.reset_img_distance}")
        return self.reset_img_distance < self.marker_displacement_config['reset_img_similarity']

    def _filter_common(self, img, markers, display=True):
        #2. find closest neighboor in previous img (self.markers_ref) to detected markers (markers)
        markers_reference, markers_paired, displ_to_add, marker_indices, valid_indices = self.find_marker_associations(markers, self.markers_ref, self.displ_to_add, self.marker_displacement_config['max_shift_len'])

        #3. Define marker state (detected + undetected)
        undetected_mask = np.ones(self.markers_ref.shape[0], dtype=bool)
        undetected_mask[marker_indices] = False

        # print(f"num undetected: {undetected_mask.sum()}")

        #4. Interpolate position of deteted dots to estimate undetected dot locations
        if self.marker_displacement_config.get('interpolate_undetected', False):
            displacement = markers_paired - markers_reference
            interpolated_displacement = np.nan_to_num(griddata(markers_reference, displacement, self.markers_ref, method='linear'))

        #5. Define measurement model
        z = np.zeros_like(self.markers_ref)
        z[marker_indices] = markers_paired

        if self.marker_displacement_config.get('interpolate_undetected', False):
            z[undetected_mask] = self.markers_ref[undetected_mask] + interpolated_displacement[undetected_mask]
        else:
            # instead of using interpolated distace, just set undetected markers back to initial
            z[undetected_mask] = self.markers_initial[undetected_mask]

        # test keeping it the same as before
        # z[undetected_mask] = self.markers_ref[undetected_mask] + self.displ_to_add[undetected_mask]

        return z, marker_indices, undetected_mask

    def filter_markers_average(self, img, markers, display=True):
        """
        Filter marker positions based on a) throwing away consistently undetected estimates, and b) taking markers
        that have an outlier displacement compared to their neighbours, and replace their estimate with the
        average from their neighbours.
        """

        if self.first and self.marker_displacement_config.get("reset_on_first", False):
            self.first = False
            self.img_ref = img  # keep img_ref using the original ref
            self.reset_references(img)

        if not self.rectangular_sensor:
            img = img * self.mask

        # If no contact, reset reference
        if  self.is_contact(markers, img, max_shift_len=self.marker_displacement_config['reset_max_shift_len']):
            self.reset_references(img)
        else:
            z, _, undetected_mask = self._filter_common(img, markers, display)

            self.displ_to_add += z - self.markers_ref
            self.markers_ref = z

            # FIX 1
            # if a marker goes undetected for some fixed number of steps, throw it away
            # self.simul_undetected_count[undetected_mask] += 1
            # self.simul_undetected_count[np.invert(undetected_mask)] = 0
            # disappeared_markers = self.simul_undetected_count >= self.marker_displacement_config['max_undetected']
            # self.markers_ref[disappeared_markers] = self.markers_initial[disappeared_markers]
            # self.displ_to_add[disappeared_markers] = 0

            # FIX 2
            # if a marker's displacement is an outlier compared with its neighbours, replace it with
            # (possibly weighted) average of neighbours displacements
            nbr_displs = self.displ_to_add[self.nbr_inds]
            nbr_displs_mean = nbr_displs.mean(axis=1)  # TODO consider weighted mean instead
            displ_diff = np.linalg.norm(self.displ_to_add - nbr_displs_mean, axis=1)
            displ_diff_max = displ_diff.mean() + self.marker_displacement_config['num_stds_outlier'] * displ_diff.std()
            displ_outliers = displ_diff > displ_diff_max
            # print(f"num outliers: {displ_outliers.sum()}")
            # print(f"outliers displ mean {self.displ_to_add[displ_outliers].mean(axis=0)}, overall mean: {self.displ_to_add.mean(axis=0)}")
            self.markers_ref[displ_outliers] = self.markers_initial[displ_outliers] + nbr_displs_mean[displ_outliers]
            self.displ_to_add[displ_outliers] = nbr_displs_mean[displ_outliers]

            # FIX 3
            # LPF on marker displacements
            tau = self.marker_displacement_config.get('lpf_tau', 1.0)  # default to no filter
            self.markers_ref = tau * self.markers_ref + (1 - tau) * self.prev_markers_ref
            self.prev_markers_ref = deepcopy(self.markers_ref)

        return self.markers_ref, self.show_marker_displacement(img)

    def filter_markers_kalman(self, img, markers, display=True):
        """
        Kalman filter implementation of marker tracking.
        Parameters
        ---
        img (cv2.image)
            Raw STS output
        markers (np.array)
            Marker pixel positions
        """
        full_img = img
        if not self.rectangular_sensor:
            img = img * self.mask

        # If no contact, reset reference
        if  self.is_contact(markers, img, max_shift_len=self.marker_displacement_config['reset_max_shift_len']):
            self.reset_references(img)
        else:
            z, marker_indices, undetected_mask = self._filter_common(img, markers, display)

            cov_matrix =  np.zeros_like(self.markers_ref)
            cov_matrix[marker_indices] = self.marker_displacement_config['std_detected_dots']**2
            cov_matrix[undetected_mask] = self.marker_displacement_config['std_undetected_dots']**2

            z = z.flatten()
            std_vec = cov_matrix.flatten()
            self.kalman_filter.R = np.diag(std_vec)

            #Kalman Update
            self.kalman_filter.predict()
            self.kalman_filter.update(z)

            #reinitialize references
            self.displ_to_add += self.kalman_filter.x.reshape((-1,2)) - self.markers_ref
            self.markers_ref = self.kalman_filter.x.reshape((-1,2))

        fig = self.show_marker_displacement(img)
        return self.markers_ref, fig

    def get_centroid_xy_displacement(self):
        return self.displ_to_add

    def show_marker_displacement(self, img):
        #plot
        displacement_img = deepcopy(img)

        # customizations for paper plots
        paper_plot = False
        size_mod = 1.0
        if paper_plot:
            size_mod = 4.0
            displacement_img = cv2.resize(displacement_img, [int(size_mod * img.shape[1]), int(size_mod * img.shape[0])])
            # "color": [255,255,0],
            # "thickness": 2,
            self.marker_displacement_config['color'] = [0, 0, 255]
            self.marker_displacement_config['thickness'] = 2
            # self.marker_displacement_config['thickness'] = 4

        for i in range(self.displ_to_add.shape[0]):
            color = self.marker_displacement_config['color']

            start = self.markers_initial[i] * size_mod
            end = self.markers_ref[i] * size_mod
            # displacement_img = cv2.line(displacement_img,
            displacement_img = cv2.arrowedLine(displacement_img,
                                            pt1=tuple(start.astype('int')),
                                            pt2=tuple(end.astype('int')),
                                            color=color,
                                            thickness=self.marker_displacement_config['thickness'],
                                            tipLength=0.7)

        return displacement_img

    def find_marker_associations(self, markers, markers_ref, displ_to_add, max_shift_len=10):
        """
        Associate current marker locations to reference marker locations.
        Parameters
        ---
        markers (np.array)
            Marker pixel positions
        markers_ref (np.array)
            Marker pixel positions of reference
        displ_to_add (np.array)
            Marker pixel displacements (keeps memory of past displacements and adds to current.) This prevents markers with large displacements to be associated with neighboor markers.
        """
        if any(elem.shape[0]==0 for elem in [markers, markers_ref, displ_to_add]):
            return None, None, None, None, None

        #1. compute distance between markers dots and reference dots
        markers_tiled = np.array([np.tile(i, (markers_ref.shape[0], 1)) for i in markers])
        markers_diff = markers_tiled - markers_ref
        distances = np.linalg.norm(markers_diff, axis=2)
        #2. compute distance between markers dots and reference dots
        marker_ref_indices = np.argmin(distances, axis=1)
        min_distances = np.linalg.norm(markers - markers_ref[marker_ref_indices], axis=1)
        #3. discard displacement higher than some threshold (outliers)
        valid_indices = np.where(min_distances < max_shift_len)
        marker_ref_indices = marker_ref_indices[valid_indices[0]]
        disp_to_add = markers[valid_indices[0]] - markers_ref[marker_ref_indices] + displ_to_add[marker_ref_indices]
        return markers_ref[marker_ref_indices], markers[valid_indices[0]], disp_to_add, marker_ref_indices, valid_indices

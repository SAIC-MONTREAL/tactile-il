# June, 2022. Python code to implement depth from marker deformation, translated from my Matlab implementation

import os
import argparse
from ast import arguments
from random import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.util import invert
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from PIL import Image as im
from sts.scripts.sts_transform import STSTransform
from sts.scripts.surface_animation import SurfaceAnimation, interpolate_pcl
from sts.scripts.marker_detection.marker_detection import MarkerDetectionCreator

class DepthDetector:
    def __init__(self, config_dir):
        self.smoothed_img = None
        self.sts_transform = STSTransform(config_dir)
        self.mask = cv2.cvtColor(self.sts_transform.transform(cv2.imread(os.path.join(config_dir, 'mask.png'))), cv2.COLOR_BGR2GRAY)
        self.mask = (self.mask / 255).astype('uint8')
        self.sigma= 1.0                   #slight spatial smoothing of depth estimates at markers, for visualiation (this was originally 20)
        self.smoothing_factor = 0.1       #[0, 1], for pixel-wise smoothing in time of the sequence of reconstructed frames.
        self.focal_length = 6.0          #These are the two camera parameters. self.focal_length was arbitrarily chosen.
        self.membrane_dist = 1.0          #Rough estimate from François.
        self.minimum_disp = 2.0
        self.scaling = 15
        self.float_epsilon = np.finfo(float).eps
        self.Z = np.ones(self.mask.shape) * self.membrane_dist
        self.f = np.ones(self.mask.shape) * self.focal_length
        self.interpolate = False #interpolate between marker locations
        # self.surface_animation = SurfaceAnimation()

        #initialize markers ref
        self.marker_detector = MarkerDetectionCreator(config_dir).create_object()
        self.markers_ref = cv2.cvtColor(self.marker_detector.markers_img_ref, cv2.COLOR_BGR2GRAY)
        self.markers_ref = self.binarize_img(self.markers_ref)

    def update_markers_ref(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.markers_ref = self.binarize_img(img)

    def binarize_img(self, img, thresh=0):
        return np.where(img>thresh,0,1).astype(np.uint8)

    def get_skeleton(self, img):
        base_skeleton = skeletonize(img)
        base_skeleton = base_skeleton.astype(np.uint8)
        return (1 - base_skeleton)

    def get_skeleton_distance(self, skeleton):
        base_distance = cv2.distanceTransform(skeleton.astype('uint8'),cv2.DIST_L2,5)
        base_distance = cv2.GaussianBlur(base_distance,(0,0),2.0,cv2.BORDER_DEFAULT)
        return base_distance * self.mask

    def get_distance(self, img):
        skeleton = self.get_skeleton(img)
        return self.get_skeleton_distance(skeleton)

    def get_surface(self, depth_img, mask):
        x_ind, y_ind = np.where(1-mask)
        x = np.arange(0, depth_img.shape[1], 1)
        y = np.flip(np.arange(0, depth_img.shape[0], 1))
        X, Y = np.meshgrid(x,y)
        x = X[x_ind, y_ind]
        y = Y[x_ind, y_ind]
        z = depth_img[x_ind, y_ind]
        mask_array = self.mask[x_ind, y_ind]
        zero_indices = np.where(mask_array==0)

        x = np.delete(x, zero_indices)
        y = np.delete(y, zero_indices)
        z =np.delete(z, zero_indices)

        if len(x)==0:
            x = y = z = [0]
        return np.vstack((x,y,z)).transpose()

    def get_marker_depth(self, img_dots, mask=None, display=False, return_displacement=False):
        img_dots = self.binarize_img(img_dots, thresh=125)

        base_distance = self.get_distance(self.markers_ref)
        frame_distance = self.get_distance(img_dots)

        L = np.multiply(np.divide((self.f + self.Z),self.f),base_distance)

        Displacement = self.Z - (np.divide(np.multiply(self.f,L),(frame_distance + self.float_epsilon)) - self.f)

        if mask is None:
            Displacement = Displacement * self.mask
        else:
            Displacement = Displacement * mask

        # Now replace -ve values in displacement array with 0 since we cannot have -ve displacements
        Displacement = np.where(Displacement < self.minimum_disp, 0, Displacement)

        # Print minimum and maximum of Displacement

        # height map
        Displacement = np.multiply(Displacement,(1 - img_dots))
        Displacement = cv2.GaussianBlur(Displacement,(0,0),self.sigma,cv2.BORDER_DEFAULT)

        DisplacementNorm = cv2.normalize(Displacement, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        DisplacementNorm = cv2.cvtColor(DisplacementNorm,cv2.COLOR_GRAY2BGR)
        x_ind, y_ind = np.where(1-img_dots)
        surface = self.get_surface(Displacement* self.scaling, img_dots)

        xrange = (int(np.min(surface[:,0])), int(np.max(surface[:,0])))
        yrange = (int(np.min(surface[:,1])), int(np.max(surface[:,1])))
        if self.interpolate:
            interpolated_surface = interpolate_pcl(surface, num=100, xrange=xrange, yrange=yrange, method='linear')
        if display:
            just_created = False
            if not hasattr(self, 'surface_animation'):
                self.surface_animation = SurfaceAnimation()
                just_created = True
            self.surface_animation.update_points(surface)
            # if just_created:
            #     self.surface_animation.vis.run()

        # rgb_image = 255.0 * np.array(self.surface_animation.vis.capture_screen_float_buffer())
        # img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR).astype(np.uint8)
        # cv2.imshow('TEST', img)

        if return_displacement:
            return DisplacementNorm, Displacement
        else:
            return DisplacementNorm

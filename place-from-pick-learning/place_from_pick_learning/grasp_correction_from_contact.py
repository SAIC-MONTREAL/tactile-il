import numpy as np 
import cv2
import os
import matplotlib as mpl
#mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GM
import itertools
from scipy import linalg
from sklearn import linear_model
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.ensemble import IsolationForest as iForest
from sts.scripts.contact_detection.contact_detection import ContactDetectionCreator
import glob 
import ipdb
import pandas as pd 



class GraspClassification:

    def __init__(self, 
                 ideal_coords=(0,0), 
                 ero_kernel=3, 
                 dil_kernel=3, 
                 error_margin=200):

        self.ideal_spot = ideal_coords
        self.error_margin_translate = error_margin
        self.erosion_kernel = np.ones((ero_kernel, ero_kernel), 
                                np.uint8)
        self.dilation_kernel = np.ones((dil_kernel, dil_kernel),
                                 np.uint8)
        self._outlier_detector = LOF(n_neighbors=100)
        self.pts_thresh = 500
        self.covar_ratio = 0.4


    def _detect_plate_edge(self, means, covariances, frame, ):
        '''
        Given the means and the covariance matrix of the points, find if there is an edge 
        of a plate detected.
        Logic:
            1. A gaussian mixture with considerable number of points,
            2. A low major/minor axis ratio. 
            3. Length of the major axis has be long enough. Atleast half of the width
            of the image.
        
        w,v = linalg.eign(covariances)
        ratio = w[0]/w[1]


        has_edge = False
        if ratio < 0.2 and w[1] > 
            has_edge = True
        
        '''

        return 0

    def classify_grasp(self, contact_frame, display=False, img_dict={}, modalities=[]):
        '''
        Given an image of the contact signature of a grip from an 
        STS sensor, returns the correction in terms of rotation and translation
        that needs to be performed to get a pre determined "ideal" grip.
        :input:
            :param contact_frame: Numpy array [height x width x 3] A BW image 
                showing the patch of contact.
            :param display: Boolean. If true will plot the fitted ellipse and 
                the needed change in gripper position.
            :param img_dict: dictionary of images
                            {'image description': np array (img)}
        '''
        frame_height, frame_width, _ = contact_frame.shape
        aspect_rat = frame_height/frame_width
        
        if display:
            img_seq = len(modalities)
            total_subplots = img_seq
            i = 0
            for key in img_dict.keys():
                if key in modalities:
                    splot = plt.subplot(total_subplots, 1, i+1)     
                    splot.imshow(img_dict[key])
                    splot.set_axis_off()
                    splot.set_aspect(aspect_rat)
                    plt.title(key)
                    i +=1

            #splot = plt.subplot(total_subplots, 1, img_seq+1, projection='3d')
            #plt.hist(img_list[-1].flatten(), bins=50, log=True) 
            #splot.set_aspect(aspect_rat)

            if 'contact' in modalities:
                splot = plt.subplot(total_subplots, 1, total_subplots-1)
                splot.imshow(contact_frame)
                splot.set_aspect(aspect_rat)
                splot.set_axis_off()
                plt.title("Contact detection")

        suggested_translate_pixels = np.asarray([0, 0])
        #image processing
        frame_erode = cv2.erode(contact_frame, self.erosion_kernel)
        frame_dial = cv2.dilate(frame_erode, self.dilation_kernel)

        #gaussian fitting to get clusters.
        r, c = np.where(frame_dial[:, :, 0] == 255)
        r = 480 - r
        points = np.concatenate([np.expand_dims(c, axis=1),
                                        np.expand_dims(r, axis=1)]
                                    , axis=1)
 
        has_lip = False

        if 'ellipse' in modalities:
            if display:
                splot = plt.subplot(total_subplots, 1, total_subplots)
                splot.set_aspect(aspect_rat)
                plt.xlim(0, frame_width)
                plt.ylim(0, frame_height)
                plt.xticks(())
                plt.yticks(())
                plt.title('Grasp classifier (Ellipse fitting)')

        #random threshold point to remove noise
        if len(points) > self.pts_thresh:

            #outlier filtering
            outlier_labels = self._outlier_detector.fit_predict(points)
            points_inlier =  points[outlier_labels==1]
            points_outlier = points[outlier_labels==-1]

            #points_inlier = points
            gm = GM(1).fit(points_inlier)

            #check if there is a lib
            w,v = linalg.eigh(gm.covariances_[0])
            ratio = np.sqrt(w[0]/w[1])
            if ratio < self.covar_ratio:
                has_lip = True

            # if display:

            #     if 'contact' in modalities:
            #         splot = plt.subplot(total_subplots, 1, total_subplots-2)
            #         splot.text(0,0,
            #             "Ratio :{:.2f}\nHeight :{:.2f}\nHas lip :{}".format(ratio, 2*np.sqrt(w[1]), has_lip),
            #             fontsize=10,
            #             )
            #         splot.set_aspect(aspect_rat)

            #TODO convert pixel space to cms??
            if display:
                if 'ellipse' in modalities:
                    splot.scatter(points_inlier[:, 0], points_inlier[:, 1], 0.8, color='navy', alpha=0.3)
                    splot.scatter(points_outlier[:, 0], points_outlier[:, 1], 0.8, color='gold', alpha=0.6)
                    self._draw_ellipse(points_inlier, 
                                    gm.means_, 
                                    gm.covariances_, 
                                    splot,
                                    )


    
        if display:
            plt.draw()
            plt.pause(0.001)

        diff = None

        if has_lip:
            diff = np.linalg.norm(gm.means_[0][0] - frame_width)

            print(diff, frame_width)
            print(gm.means_)
        return diff


    def _draw_ellipse(self, X, means, covariances, artist):

        '''
        Method to plot the fitted ellipse on the contact point and the 
        suggested change for an ideal grip.
        :input:
            :param X: Numpy array (n x 2) containing the points used to fit 
                the Gaussian mixture. (where n is the number of points)
            :param means: Means of the fitted gaussian mixture. Numpy array.
            :param covariances: Covariances of the fitted GM. Numpy array.
            :param title: Title of the figure. String. 
        '''
        color_iter = itertools.cycle(["navy", 
                            "c", "cornflowerblue", "gold", "darkorange"])
        for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
            ell.set_clip_box(artist.bbox)
            ell.set_alpha(0.5)
            artist.add_artist(ell)




class GraspClassificationSVM:

    def __init__(self):

        pass 

    def classify_regrasp(self):

        pass 

if __name__=='__main__':

    traj_data_path = '../dataset/srj/video_tactile_regrasp/'
    traj_fnames = glob.glob('{}/*.pkl'.format(traj_data_path))

    save_seq =True
    '''
    Data format:
    {
        'data' : [
                   (dict ['ee', 'q', 'gripper', 'rgb', 'depth', 
                      'sts_transformed_image'], 
                   numpy_arr (7,),
                   dict ['ee', 'q', 'gripper',
                     'rgb' (128x128x3), 'depth' (128x128), 'sts_transformed_image' (480x640x3)]), 
        'grasp_tac_img': list [480, 640, 3] x n where n is the length of the sequence 
                    of tactile images captured 
     }

    '''

    config_dir = '../../sts-cam-ros2/configs/sts_rectangular'
    #select_modalities = ['raw_image', 'difference', 'gradient', 'contact', 'ellipse']
    select_modalities = ['raw_image', 'ellipse' ]
    save_dir = os.path.join(traj_data_path,'contact_sequence')
    if save_seq:
        os.makedirs(save_dir, exist_ok=True)
    contact_back_sub = ContactDetectionCreator(config_dir).create_object()

    grasp_classifier = GraspClassification((0,0))
    traj_indx = 0
    plt.figure(figsize=(12,10), dpi=100)

    for traj_f in traj_fnames:
        traj_data = pd.read_pickle(traj_f)
        trans_img_seq = traj_data['grasp_tac_img']
        contact_back_sub.ref_img = trans_img_seq[0]
        seq_len = len(trans_img_seq)
        save_dir_traj = os.path.join(save_dir,'{}'.format(traj_f.split('/')[-1]))

        if save_seq:
            os.makedirs(save_dir_traj, exist_ok=True)
        
        for seq_id in range(seq_len):

            trans_img = trans_img_seq[seq_id]
            contact_patch = contact_back_sub.detect(trans_img)
            img_list = contact_back_sub.get_channels(trans_img)
            img_dict = {
                         'raw_image':img_list[0],
                         'difference':img_list[1],
                         'gradient': img_list[2]
                         }
            grasp_classifier.classify_grasp(contact_patch, 
                                            display=True, 
                                            img_dict=img_dict,
                                            modalities=select_modalities
                                        )
            if save_seq:
                plt.savefig("{}/image_seq_{:04d}.png".format(save_dir_traj, seq_id), dpi=128, bbox_inches='tight')
            plt.clf()
        
        traj_indx +=1




import pickle
import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt 
from place_from_pick_learning.datasets import MultimodalManipBCDataset
import cv2
from mpl_toolkits.mplot3d import Axes3D
import argparse


def parse_cmd_args():
    args = argparse.ArgumentParser()
    args.add_argument('--path-to-trajectory', required=True, help="Path to the pkl trajectory.")
    args.add_argument('--save-folder', default=None, help="Path to save the visualizations")
    return args.parse_args()

def open_pkl(file_path):
    with open(os.path.join(file_path), 'rb') as file_data:
        data = pickle.load(file_data)
    return data

def visualize_dataset(traj_path, save_folder=None):
    '''
    Method that enumerates all the trajectories of the dataset and visualizes them.
    '''
    traj = open_pkl(traj_path)
    save_folder_traj = os.path.join(os.path.dirname(traj_path), f'{traj_path.split("/")[-1]}_viz')
    visualize_trajectory(traj, 
                        0,
                        len(traj), 
                        save_folder=save_folder_traj)

def visualize_trajectory(trajectory, start, end, save_folder=None):
    '''
    Method to visualize a given trajectory.
    '''
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)

    obs_sequence = []
    h,w,_  = trajectory[start][0]['rgb'].shape
    h_buffer_size = 20
    vertical_buffer = (np.ones((h,h_buffer_size,3))*255).astype('uint8')
    for i in range(start, end):
        obs_sequence.append(trajectory[i][0])
    #visualize the rgb sequence
    rgb_seq_array = []
    for obs in obs_sequence:
        rgb_seq_array.append(obs['rgb'])
        rgb_seq_array.append(vertical_buffer)
        
    rgb_seq = cv2.cvtColor(np.concatenate(rgb_seq_array, axis=1), cv2.COLOR_BGR2RGB)
    #cv2.imshow('rgb_seq', rgb_seq)
    
    if save_folder:
        fname = os.path.join(save_folder, 'rgb_seq.png')    
        cv2.imwrite(fname, rgb_seq)

    #visualize the depth sequence 
    depth_seq_array = []
    vertical_buffer = (np.ones((h,h_buffer_size))).astype('uint8')
    raw_depth_seq = []
    for obs in obs_sequence:
        depth_seq_array.append((obs['depth']).astype('uint8'))
        depth_seq_array.append(vertical_buffer)
        raw_depth_seq.append(obs['depth'])
        
    depth_seq = np.concatenate(depth_seq_array, axis=1)
    #cv2.imshow('depth_seq', depth_seq)    
    if save_folder:
        fname = os.path.join(save_folder, 'depth_seq.png')    
        cv2.imwrite(fname, depth_seq)
    #cv2.waitKey(0)
    
    #visualize the 3d ee trajectory
    ee_sequence = []
    for obs in obs_sequence:
        ee_sequence.append(np.expand_dims(np.array(obs['ee']), axis=0))
    
    ee_seq_arr = np.concatenate(ee_sequence, axis=0)
    fig, ax = plt.subplots(2)
    ax[0] = plt.axes(projection='3d')
    ax[0].plot3D(ee_seq_arr[:, 0], 
              ee_seq_arr[:, 1],
              ee_seq_arr[:, 2])
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_zlabel('z')
    
    #visualize gripper states
    gripper_state_seq = []
    for obs in obs_sequence:
        gripper_state_seq.append(np.expand_dims(np.array(obs['gripper']), axis=0))
        
        
    gripper_state_arr = np.concatenate(gripper_state_seq, axis=0)
    gripper_open_pts = ee_seq_arr[gripper_state_arr[:, 0] > 0]
    gripper_close_pts = ee_seq_arr[gripper_state_arr[:, 0] < 0]

    #ax.scatter3D()
    ax[0].scatter3D(gripper_close_pts[:, 0], gripper_close_pts[:,1], gripper_close_pts[:, 2], color='red')
    ax[0].scatter3D(gripper_open_pts[:, 0], gripper_open_pts[:,1], gripper_open_pts[:, 2], color='green')
    if save_folder:
        fname = os.path.join(save_folder, "ee_trajectory.png")
        fig.savefig(fname)
    plt.close()
    plt.plot(np.arange(gripper_state_arr[:, 0].shape[0]), gripper_state_arr[:, 0])
    plt.xlabel('timestep')
    plt.ylabel('gripper cosine')
    if save_folder:
        fname = os.path.join(save_folder, 'gripper_cosine.png')
        plt.savefig(fname)

    return depth_seq_array, raw_depth_seq

def print_traj_stats(traj_path):
    traj = open_pkl(traj_path)

    for t in traj:
        o_t, a_t, o_t1 = t
        if type(a_t) == tuple:
            a_t = np.array(a_t)

if __name__=='__main__':
    args = parse_cmd_args()
    # visualize_dataset(args.path_to_trajectory, args.save_folder)
    print_traj_stats(args.path_to_trajectory)

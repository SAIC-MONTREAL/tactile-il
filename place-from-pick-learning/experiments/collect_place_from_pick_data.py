import os
import pickle
import shutil
import argparse 
import numpy as np
import json
from collections import OrderedDict
import uuid

from omegaconf import OmegaConf
from hydra.utils import instantiate

from transformations import quaternion_multiply
from place_from_pick_learning.utils.utils import add_pose_noise
from place_from_pick_learning.utils.learning_utils import set_seed
from transform_utils.pose_transforms import PoseTransformer


class CollectPlaceFromPick:
    """
    Collect place data from grasps/picks
    """
    def __init__(self, gym):
        self.gym = gym

    def collect_data(
        self, 
        dt=0.20,
        add_traj_noise=False,
    ):

        """
        Collect demonstrations for placing from picking

        Args:
            dt (float): Time interval between "steps"
            add_traj_noise (bool): Add noise to the demonstration
                trajectory
            add_retrieval_noise (bool): Add noise to the 
                retrieval pose
        Returns:
            demonstration_data (list): A list of tuples of
                (o_t, a_t, o_t1)
        """

        # Grasp and retrieve sequence, return high-frequency pose data
        (_, grasp_tac_img, high_freq_retrieval_poses) = self.gym.reset()
    
        # Generate downsampled traj for place at dt rate
        pose_list = self.generate_downsampled_traj(
            high_freq_retrieval_poses,
            dt
        )

        # Reverse poses
        pose_list.reverse()
        
        # Add noise
        if add_traj_noise:
            # Add noise to only the first 75% of the trajectory
            idx = int(.75 * len(pose_list))
            pose_list = [add_pose_noise(p) if ii < idx else p for ii, p in enumerate(pose_list)]

        # Convert poses into pose deltas for network to learn action deltas
        pose_delta_list = self.pose_to_posedelta(pose_list)
        
        # Move to initial pose
        o_t = self.gym.go_to_pose(pose=pose_list[0])

        # Place by reversing and keep gripper close
        demonstration_data = {"data":[], "grasp_tac_img":grasp_tac_img}
        for pose_delta in pose_delta_list:
            a_t = np.array((*pose_delta.get_array(rotation_representation="rvec"),1))
            o_t1, _, _, _, _= self.gym.step(a=a_t)
            demonstration_data["data"].append((o_t, a_t, o_t1))
            o_t = o_t1

        # Open gripper to pinch mode after reversing
        a_t = np.array([0,0,0,0,0,0,-1])
        for _ in range(5):
            o_t1, _, _, _, _= self.gym.step(a=a_t)
            demonstration_data["data"].append((o_t, a_t, o_t1))
            o_t = o_t1

        return demonstration_data

    def pose_to_posedelta(self, pose_list):
        """
        Converts a list of absolute poses into a list of pose deltas

        Args:
            pose_list (list): A list of absolute poses (PoseTransformer)

        Returns:
            pose_delta_list (list): A list of relative 
                or delta poses (PoseTransformer)
        """
        pose_delta_list = []
        pose_t = pose_list[0]
        for pose_t1 in pose_list[1:]:
            
            # position difference
            pos_delta = pose_t1.get_pos() - pose_t.get_pos()

            # Rotation difference
            rot_delta = quaternion_multiply(
                pose_t.get_quat_inverse(order="wxyz", normalize=True),
                pose_t1.get_quat(order="wxyz", normalize=True)
            )
            pose_delta = PoseTransformer([
                    pos_delta[0],
                    pos_delta[1],
                    pos_delta[2],
                    rot_delta[0],
                    rot_delta[1],
                    rot_delta[2],
                    rot_delta[3],
            ])
            
            pose_delta_list.append(pose_delta)
            pose_t = pose_t1

        return pose_delta_list

    def generate_downsampled_traj(self, traj_list, dt, include_last=True):
        """
        Downsample a high-frequency pose trajectory `traj_list` 
            into a lower-frequency one with the nearest pose
            at intervals of `dt`

        Args:
            traj_list (list): A list of Tuples of (time (float), pose (PoseTransformer)), 
                e.g., [(t0, pose_0), ..., (tn, pose_n)]
            dt (float): Interval at which to sample nearest poses
            include_last (bool): Include last endpoint pose

        Returns:
            pose_list (list): A list of of poses (PoseTransformer)
        """
        def find_nearest_pose(time):
            idx = min(range(len(traj_list)), key=lambda i: abs(traj_list[i][0]-time))
            time, pose = traj_list[idx]
            return pose
        
        final_time = traj_list[-1][0]
        initial_time = traj_list[0][0]
        total_time = final_time - initial_time

        n_poses, r = divmod(total_time, dt)
        n_poses = int(n_poses)
        t_list = np.linspace(0, int(n_poses * dt), n_poses + 1)
        pose_list = []

        for t in t_list:
            pose_t = find_nearest_pose(initial_time + t)
            pose_list.append(pose_t)

        # Add last pose that's less than `dt`` away if `total_time`
        # is not cleanly divided by `dt`` chunks
        if r != 0 and include_last:
            pose_list.append(traj_list[-1][1])
        
        return pose_list

def save_pkl(data, save_path):
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=str, default="demonstration_data",
                help='Name of file where demonstration data will saved')
    parser.add_argument('--save_dir', type=str, default="/home/rbslab/place-from-pick-learning/datasets",
                help='Directory to save data')
    parser.add_argument('--random_seed', type=int, default=144,
                help='Random seed')
    parser.add_argument('--n_episodes', type=int, default=16,
                help='Number of demonstration episodes to collect')
    parser.add_argument('--env_config_path', type=str, default=None,
                help='Path to where we have the env config')
    parser.add_argument('--add_traj_noise', action='store_true')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()

    # Set random seed
    set_seed(args.random_seed)

    # Create env with config
    env_cfg = OmegaConf.load(args.env_config_path)
    env_cfg = env_cfg.env_config
    gym = instantiate(env_cfg)

    # Save result dir
    save_dir = os.path.join(
        args.save_dir, 
        f"{args.dataset_id}_{args.n_episodes}-episodes_{uuid.uuid4().hex}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Save dataset parameters
    with open(os.path.join(save_dir, "dataset_parameters.txt"), "w") as f:
        json.dump(
            OrderedDict(sorted(args.__dict__.items(), key=lambda t: t[0])), 
            f, 
            indent=2
        )

    # Save STS config dir used for this specific dataset
    if env_cfg.sts_config_dir is not None:
        shutil.copytree(
            src=env_cfg.sts_config_dir, 
            dst=os.path.join(save_dir, "sts_config")
        )

    # Save env config
    with open(os.path.join(save_dir, "env-config.yaml"), "w") as f:
        OmegaConf.save(env_cfg, f)

    # Start experiment
    c = CollectPlaceFromPick(gym=gym)

    for ii in range(args.n_episodes):

        print(f"Collecting episode {ii + 1}/{args.n_episodes}")

        # Collect place demonstration
        demonstration_data = c.collect_data(
            dt=env_cfg.dt,
            add_traj_noise=args.add_traj_noise,
        )

        # Save episode data
        save_pkl(
            demonstration_data, 
            os.path.join(save_dir, f"{ii}_{uuid.uuid4().hex}.pkl")
        )
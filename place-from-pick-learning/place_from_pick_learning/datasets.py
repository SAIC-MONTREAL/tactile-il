import pickle
import numpy as np
import os

from torch.utils.data import Dataset

from place_from_pick_learning.utils.learning_utils import build_state_action_pair

class MultimodalManipBCDataset(Dataset):
    '''
    Cached multimodal manipulation dataset for behaviour cloning (BC).
    '''
    def __init__(self,
                 data_dirs,
                 seq_length=1,
                 rotation_representation="rvec",  # keep for backwards compatibility
                 obs_rotation_representation="rvec",
                 stored_obs_rotation_representation="rvec",
                 act_rotation_representation="rvec",
                 stored_act_rotation_representation="rvec",
                 load_tensor=False,
                 device="cuda",
                 n_frame_stacks=1,
                 env_config_file=None,
                 n_max_episodes=None,
                 ):
        self.seq_length = seq_length
        self.data_dirs = data_dirs
        if rotation_representation is not None:
            self.obs_rotation_representation = rotation_representation
            self.stored_obs_rotation_representation = rotation_representation
            self.act_rotation_representation = rotation_representation
            self.stored_act_rotation_representation = rotation_representation
        else:
            self.obs_rotation_representation = obs_rotation_representation
            self.stored_obs_rotation_representation = stored_obs_rotation_representation
            self.act_rotation_representation = act_rotation_representation
            self.stored_act_rotation_representation = stored_act_rotation_representation
        self.n_frame_stacks = n_frame_stacks
        self.n_max_episodes = n_max_episodes
        self.load_tensor = load_tensor
        self.device = device

        # automatically set parameters based on a loaded env config file
        self.env_config = None
        self.grip_in_action = True
        self.sts_switch_in_action = False
        if env_config_file is not None:
            import json
            with open(env_config_file, "r") as f:
                self.env_config = json.load(f)
            self.grip_in_action = self.env_config['grip_in_action']
            self.sts_switch_in_action = self.env_config['sts_switch_in_action']

        self.dataset = self.build_cached_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        '''
        Returns the training tuple from the index idx
        :param idx: int
        '''
        return self.dataset[idx]

    def build_cached_dataset(self):
        '''
        A method that reads information of different trajectories and builds a training dataset.
        The training dataset is made up of contiguious sequence of observations as state and action taken at the
        end of the sequence as the corresponding action.

        The format of the data:
            List of demonstration trajectory.
            Each demonstration is a list of tuples (obs_t, act_t, obc_{t+1}).
                Each observation is a dictionary containing the following keys:
                    {"ee": numpy array <>,
                     "q" : numpy array <>,
                     "gripper": ,
                     "rgb": ,
                     "depth": ,
                     "tactile": }
                Action: object of PoseTransformer class (PoseStamped msg)
        Output:


        '''
        training_data = []

        # Collect all episodes
        total_episodes = []
        for d in self.data_dirs:
            # ensure that order is always the same
            files = sorted(os.listdir(d))
            episodes = [os.path.join(d, f) for f in files if f.endswith(".pkl")]
            total_episodes.extend(episodes)

        # Limit amount of episodes
        if self.n_max_episodes:
            total_episodes = total_episodes[:self.n_max_episodes]
        print(f"Training on {len(total_episodes)} episodes")

        for e in total_episodes:
            # Load episode
            with open(os.path.join(d, f"{e}"), 'rb') as file_data:
                traj = pickle.load(file_data)
            # traj = traj["data"]

            # Process each chunk possible
            for i in range(len(traj) - self.seq_length):
                obs_sequence, action = build_state_action_pair(
                    traj[i:i+self.seq_length],
                    self.obs_rotation_representation,
                    self.stored_obs_rotation_representation,
                    self.act_rotation_representation,
                    self.stored_act_rotation_representation,
                    self.n_frame_stacks,
                    load_tensor=self.load_tensor,
                    device=self.device,
                    grip_in_action=self.grip_in_action,
                    sts_switch_in_action=self.sts_switch_in_action
                )
                training_data.append((obs_sequence, action))
        return training_data
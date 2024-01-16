import pickle
import os
import shutil
import argparse 
import numpy as np
import json
from collections import OrderedDict, deque
import uuid
import time 

from omegaconf import OmegaConf
from hydra.utils import instantiate

from place_from_pick_learning.utils.learning_utils import (
    set_seed, 
    load_model, 
    process_obs
)

class ControlPlaceFromPick:
    """
    Run control experiment / inference with model
    """
    def __init__(
        self,
        gym,
        model,
        history_length,
        device
    ):
        self.gym = gym
        self.model = model
        self.history_length = history_length
        self.device = device

    def run_control_experiment(
        self, 
        n_steps_horizon=16
    ):

        """
        Run an episode of inference or control

        Args:
            n_steps_horizon (int): Amount of time steps 
                to run control
            dt (float): Time interval between "steps"
        Returns:
            experiment_data (list): A list of tuples of
                (o_t, a_t, o_t1)
        """

        # Grasp and retrieve plate
        (o_t, _, _) = self.gym.reset()

        # Populate history
        history = deque([], maxlen=self.history_length)
        a_t = np.array([0,0,0,0,0,0,1])
        for _ in range(self.history_length):
            o_t1, _, _, _, _= self.gym.step(a_t)
            history.append((o_t, a_t, o_t1))
            o_t = o_t1
        
        # Closed-loop place for some horizon
        experiment_data = []
        for _ in range(n_steps_horizon):
            
            # Processing and network call
            o_t_processed = process_obs(
                history,
                rotation_representation=self.model.rotation_representation,
                n_frame_stacks=self.model.n_frame_stacks,
                load_tensor=True,
                device=self.device
            )
            o_t_processed = {
                k:v.unsqueeze(0) for k,v in o_t_processed.items()
            }
            a_t, _ = self.model(o_t_processed)

            a_t = a_t[0, -1]
            a_t = a_t.detach().cpu().numpy()

            # XXX: Test overwrite
            sample_x = np.random.uniform(-0.005,0.005)
            sample_y = np.random.uniform(-0.005,0.005)
            sample_z = np.random.uniform(-0.005,0.005)
            a_t = np.array([sample_x,sample_y,sample_z,0,0,0,1])
            o_t1, _, _, _, _= self.gym.step(
                a=a_t, 
                rotation_representation="rvec"
            )

            # Step
            o_t1, _, _, _, _= self.gym.step(
                a=a_t, 
                rotation_representation=self.model.rotation_representation
            )

            # Update state
            experiment_data.append((o_t, a_t, o_t1))
            history.append((o_t, a_t, o_t1))
            o_t = o_t1

        # Label success or not
        results = {}
        user_input = input('Log episode as success (y/n): ')
        if user_input.lower() == 'y':
            results["success"] = True
            print('Episode logged as success!')
        elif user_input.lower() == 'n':
            results["success"] = False
            print('Episode logged as failure!')
        else:
            results["success"] = None
            print('Episode logged as N/A!')
        results["data"] = experiment_data
        
        print("OPENING GRIPPER, CATCH!")
        time.sleep(2)
        self.gym.step(a=np.array([0,0,0,0,0,0,-1]))
        time.sleep(1)

        return results

def open_pkl(save_path):
    with open(save_path, "rb") as handle:
        stored_grasps = pickle.load(handle)
    return stored_grasps

def save_pkl(data, save_path):
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="/home/rbslab/place-from-pick-learning/results/2022.09.30/test_visualbc_sequential_rvec_16-55-54",
                help='Name of file containing weights and hyperparameters for trained model')
    parser.add_argument('--device', type=str, default="cuda",
                help='Device used to run experiments')
    parser.add_argument('--n_episodes', type=int, default=1,
                help='Number of episodes to test on')
    parser.add_argument('--n_steps_horizon', type=int, default=30,
                help='Amount of time steps to run control')
    parser.add_argument('--random_seed', type=int, default=144,
                help='Random seed')
    parser.add_argument('--save_dir', type=str, default="/home/rbslab/place-from-pick-learning/results/control_runs/",
                help='Directory to save data')
    parser.add_argument('--env_config_path', type=str, default=None,
                help='Path to where we have the env config')
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

    # Load model
    model_config_path = os.path.join(args.model_dir, "model-config.yaml")
    model_weights_path = os.path.join(args.model_dir, "model-weights.pth")
    cfg = OmegaConf.load(model_config_path)
    model = load_model(
        cfg.model.model_config, 
        path=model_weights_path,
        device=args.device, 
        mode="eval"
    )

    # Save result dir
    os.makedirs(args.save_dir, exist_ok=True)

    # Save experiment parameters
    with open(os.path.join(args.save_dir, "experiment_parameters.txt"), "w") as f:
        json.dump(
            OrderedDict(sorted(args.__dict__.items(), key=lambda t: t[0])), 
            f, 
            indent=2
        )

    # Save STS config dir used for this specific dataset
    if env_cfg.sts_config_dir is not None:
        shutil.copytree(
            src=env_cfg.sts_config_dir, 
            dst=os.path.join(args.save_dir, "sts_config")
        )

    # Save env config
    with open(os.path.join(args.save_dir, "env-config.yaml"), "w") as f:
        OmegaConf.save(env_cfg, f)
        
    # Start experiment
    c = ControlPlaceFromPick(
        gym=gym,
        model=model,
        history_length=cfg.seq_length,
        device=args.device,
    )

    for jj in range(args.n_episodes):

        print(f"Running control experiment {jj + 1}/{args.n_episodes}")

        # Run inference / control experiment
        results = c.run_control_experiment(n_steps_horizon=args.n_steps_horizon)
        results["random_seed"] = args.random_seed
        results["episode"] = jj
        results["model_dir"] = args.model_dir
        success = results["success"]

        # Save episode data
        save_pkl(
            results, 
            os.path.join(args.save_dir, f"episode-{jj}_success-{success}_{uuid.uuid4().hex}.pkl")
        )

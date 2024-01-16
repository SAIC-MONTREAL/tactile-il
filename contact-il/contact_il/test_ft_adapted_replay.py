import argparse
import copy
import os
import pickle
import time
from datetime import datetime

import cv2
import numpy as np
# import PySimpleGUI as sg

from contact_panda_envs.envs import *
from panda_polymetis.control.panda_client import PandaClient
from contact_il.data.dict_dataset import DictDataset
from contact_il.imitation.device_utils import CollectDevice
from contact_il.imitation.ft_adapted_replay import FTAdapter
from contact_il.imitation.utils import env_movement_change_threshold

parser = argparse.ArgumentParser()
parser.add_argument('environment', type=str, help="Environment")
parser.add_argument('replay_dataset_name', type=str, help="Dataset folder name (under save_dir) containing raw human "
    "trajectories to replay.")
parser.add_argument('sts_config_dir', type=str, help="String for sts config dir")
parser.add_argument('experiment_name', type=str,  help='Experiment name (folder it will be stored in)')
parser.add_argument('--save_dir', type=str, default=os.path.join(os.environ['CIL_DATA_DIR'], 'experiments'),
                    help='Top level directory for saving experiment dataset.')
parser.add_argument('--load_dir', type=str, default=os.path.join(os.environ['CIL_DATA_DIR'], 'demonstrations'),
                    help='Top level directory for loading replay dataset.')
parser.add_argument('--device', type=str, default='keyboard')
parser.add_argument('--init_move_thresh', type=float, default=.005, help='Minimum geodesic change before data is recorded.')
parser.add_argument('--sim', action='store_true', help="Use a simulated robot")
parser.add_argument('--sts_source_vid', type=str, default="", help="Use a simulated sts sensor from video.")
parser.add_argument('--render', action='store_true', default=False, help='render using env.render.')

args = parser.parse_args()

############################ SETUP #############################
# get env & device
env_args = {}
env_args['sts_config_dir'] = args.sts_config_dir
if args.sts_source_vid != "": env_args['sts_source_vid'] = args.sts_source_vid
if args.sim: env_args['sim_override'] = args.sim
env = globals()[args.environment](**env_args)
dev = CollectDevice(device_type=args.device)

# ft adapter
ft_replay_adapter = FTAdapter('closed_loop_pid', env)

# interactive tools
# layout = [
#     [sg.Text("Kp", size=(10, 1)), sg.Input(key="Kp", default_text=str(ft_replay_adapter.adapter.pid.Kp))],
# ]
# window = sg.Window("Force Adaptation Settings", layout)
# event, values = window.read(timeout=100)
# This doesn't work properly because this gui framework doesn't run in its own thread or have true callbacks
# TODO maybe we don't need to update parameters here...for now we'll just relaunch the script
print(f"CURRENT SETTINGS: ")
print(f"Kp: {ft_replay_adapter.adapter.pid.Kp}")

# dataset setup
ds = DictDataset(
    pa_args=args,
    dataset_name=args.experiment_name,
    main_dir=args.save_dir,
    compress=False,
    env=env,
    load_saved_sts_config=False,
    csv_save_keys=('pose', 'sts_avg_force', 'sts_in_contact', 'action')
)
raw_ds = DictDataset(
    pa_args=None,
    dataset_name=args.replay_dataset_name,
    main_dir=args.load_dir,
    env=env,  # worth verifying that envs are the same
    load_saved_sts_config=False  # only want the trajectories
)
ds.attach_raw_dataset_for_replay(raw_ds)
raw_demonstration_data, reset_joint_position, recorded_actions = ds.get_rdfr_ep_params(ep_i=0)

################################ REPLAY LOOP #####################################
while True:
    ################ RESET ################
    print("Freedrive is engaged, will disengage on space or reset. Hold backspace to delete last replay.")
    env.arm_client.activate_freedrive()
    reset = False
    while not reset:
        reset, delete, toggle_freedrive = dev.return_on_any_press([dev.get_reset, dev.get_delete, dev.get_start_stop])
        if delete:
            ds.remove_last_ep()
        if toggle_freedrive:
            env.arm_client.toggle_freedrive()
            print(f"Freedrive toggled. Freedrive activated? {env.arm_client._freedrive_is_active}")

    if args.sts_source_vid != "":
        env.sts_clients['sts'].sts.cam.reset_video(args.sts_source_vid)

    print("Resetting environment.")
    o_t = env.reset(reset_joint_position=reset_joint_position)
    dev.reset_states()
    ft_replay_adapter.reset(demo_traj=raw_demonstration_data)

    print(f"Collecting replay {ds._params['actual_n_episodes'] + 1}")

    done = False
    demonstration_data = []
    ts = 0

    ########### USER OPTIONS ############
    # print("Modify full traj parameters now. TODO list these params..")
    use_force_adapt = input("Use force adapt? y for yes, all other is no.") == 'y'

    ############## REPLAY ################
    while not done:
        step_start = time.time()
        if ts % 10 == 0:
            print(f"Running replay {ds._params['actual_n_episodes'] + 1}, timestep {ts}")

        dev.update()

        a_t = copy.deepcopy(recorded_actions[ts])

        if use_force_adapt:
            a_t += ft_replay_adapter.get_action_modifier(ts, o_t)

        o_t1, r, term, trunc, info = env.step(a_t)

        demonstration_data.append((o_t, a_t, o_t1))
        o_t = o_t1

        # end on timeout, success/failure, or user pressing space
        done = done or term or trunc or dev.start_stop or ts == len(recorded_actions) - 1  # once done is True, it stays True
        ts += 1
        if args.render: env.render()

    ds.save_ep(demonstration_data, raw_demo_data=raw_demonstration_data)

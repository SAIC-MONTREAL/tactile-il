import argparse
import copy
import os
import pickle
import time
from datetime import datetime

import cv2
import numpy as np

from contact_panda_envs.envs import *
from panda_polymetis.control.panda_client import PandaClient
from contact_il.data.dict_dataset import DictDataset
from contact_il.imitation.device_utils import CollectDevice
from contact_il.imitation.ft_adapted_replay import FTAdapter
from contact_il.imitation.utils import env_movement_change_threshold
from panda_polymetis.utils.blocking_keyboard import BlockingKeyboard, KEY_MOTION_ACT_MAP
from sts.scripts.helper import read_json
from pysts.utils import eul_rot_json_to_mat, eul_rot_to_mat

parser = argparse.ArgumentParser()
parser.add_argument('environment', type=str, help="Environment")
parser.add_argument('sts_config_dir', type=str, help="String for sts config dir")
parser.add_argument('experiment_name', type=str,  help='Experiment name (folder it will be stored in)')
parser.add_argument('--save_dir', type=str, default=os.path.join(os.environ['CIL_DATA_DIR'], 'experiments'),
                    help='Top level directory for saving experiment dataset.')
parser.add_argument('--sim', action='store_true', help="Use a simulated robot")
parser.add_argument('--sts_source_vid', type=str, default="", help="Use a simulated sts sensor from video.")
parser.add_argument('--render', action='store_true', default=False, help='render using env.render.')
parser.add_argument('--t_amt', type=float, default=.001, help='Translation amount (m)')
parser.add_argument('--r_amt', type=float, default=1.0, help='Rotation amount (deg)')
parser.add_argument('--sleep_after_act_time', type=float, default=0.2,
                    help='Time to sleep after selected action before getting obs.')
parser.add_argument('--force_release_time', type=float, default=2.0, help='Time to release force when done.')


args = parser.parse_args()

############################ SETUP #############################
# get env & device
env_args = {}
env_args['config_override_dict'] = {
    'init_gripper_random_lim': [0, 0, 0, 0, 0, 0],
}
env_args['sts_config_dir'] = args.sts_config_dir
if args.sts_source_vid != "": env_args['sts_source_vid'] = args.sts_source_vid
if args.sim: env_args['sim_override'] = args.sim
env = globals()[args.environment](**env_args)
bk = BlockingKeyboard()

# dataset setup
ds = DictDataset(
    pa_args=args,
    dataset_name=args.experiment_name,
    main_dir=args.save_dir,
    compress=False,
    env=env,
    load_saved_sts_config=False,
    csv_save_keys=('pose', 'target_pose', 'sts_avg_force', 'sts_in_contact', 'action'),
    mp4_save_keys=('sts_raw_image', 'wrist_rgb')
)

################################ MAIN EXPERIMENT #####################################
try:
    ################ RESET ################
    print("Freedrive is engaged, space to toggle, r to reset.")
    env.arm_client.activate_freedrive()
    reset = False
    while not reset:
        key = bk.get_key()
        if key == 'r':
            env.arm_client.deactivate_freedrive()
            reset = True
        elif key == ' ':
            env.arm_client.toggle_freedrive()
            print(f"Freedrive toggled. Freedrive activated? {env.arm_client._freedrive_is_active}")
        else:
            print(key)

    if args.sts_source_vid != "":
        env.sts_clients['sts'].sts.cam.reset_video(args.sts_source_vid)

    print("Resetting environment (but robot will NOT move!).")
    o_t = env.reset(reset_joint_position=env.arm_client.robot.get_joint_positions())

    print(f"Collecting test trajectory {ds._params['actual_n_episodes'] + 1}")

    done = False
    demonstration_data = []
    total_motion_act = np.zeros(6)
    ts = 0

    # env.arm_client.activate_freedrive()
    # input("Freedrive robot to desired initial pose, then press enter to start controlling/recording observations.")
    # env.arm_client.deactivate_freedrive()
    # env.arm_client.start_controller()
    # o_t, _, _, _, _ = env.step(np.zeros(env.action_space.shape))

    ############## KEYBOARD MOVEMENT LOOP ################
    while not done:
        step_start = time.time()

        a_t = None
        while a_t is None:
            print("Press a key to take an action.")
            key = bk.get_key()
            if key == 'd':
                break
            elif key in KEY_MOTION_ACT_MAP:
                a_t = np.array(KEY_MOTION_ACT_MAP[key], dtype=np.float32)
                if not env.grip_in_action:
                    a_t = a_t[:-1]
            else:
                print(f"key {key} not in map, choose one of: {KEY_MOTION_ACT_MAP.keys()}")

        if key == 'd':
            break

        a_t[:3] = a_t[:3] * args.t_amt
        a_t[3:] = a_t[3:] * args.r_amt * np.pi / 180

        o_t1, r, term, trunc, info = env.step(a_t)
        time.sleep(args.sleep_after_act_time)
        o_t1, r, term, trunc, info = env.step(np.zeros_like(a_t))

        total_motion_act += a_t
        print(f"Total act: {total_motion_act}, force: {o_t1['sts_avg_force']}")

        demonstration_data.append((o_t, a_t, o_t1))
        o_t = o_t1

        ts += 1
        if args.render: env.render()

    ds.save_ep(demonstration_data)

    input(f"Press enter to release force over {args.force_release_time}s")
    num_release_steps = int(env.control_freq * args.force_release_time)
    release_act = -total_motion_act / num_release_steps
    for ts in range(num_release_steps):
        env.step(release_act)

    env.arm_client.activate_freedrive()
    input("Freedrive activated, force should be dropped. Move arm close to reset, press enter to end script.")
    env.arm_client.deactivate_freedrive()

except Exception as e:
    print(e)
finally:
    bk.restore_terminal_settings()
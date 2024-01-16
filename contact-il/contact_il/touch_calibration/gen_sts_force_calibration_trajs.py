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
from sts.scripts.helper import read_json, write_json
from pysts.utils import eul_rot_json_to_mat, eul_rot_to_mat
from place_from_pick_learning.utils.debugging import nice_print

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
parser.add_argument('--max_total_dist', type=float, default=0.1, help='Max total distance to move target, in case we slip.')



args = parser.parse_args()

# Calibration motion will include:
# 5 presses in z direction (with very small shifts in x and y)
# 2 presses in z then x, 2 presses in z then negative x
    # z dist here will be calculated based on max z + max shear
# 2 presses in z then y, 2 presses in z then negative y
# 2 presses in z then rotz, 2 presses in z then negative rotz

########################## CONSTANTS ###########################
STS_MOTION_DICT = {
    'normal': np.array([0, 0, args.t_amt, 0, 0, 0]),
    'shear_x': np.array([args.t_amt, 0, 0, 0, 0, 0]),
    'shear_y': np.array([0, args.t_amt, 0, 0, 0, 0]),
    'torque': np.array([0, 0, 0, 0, 0, args.r_amt * np.pi / 180]),
}

STS_AXIS_DICT = {'normal': 2, 'shear_x': 0, 'shear_y': 1, 'torque': 5}

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
nice_print(4)

# get ft adapted replay params for calibration
params = read_json(os.path.join(args.sts_config_dir, "ft_adapted_replay.json"))
sts_to_ee_rot_mat = eul_rot_to_mat(params['sts_to_ee_eul_rxyz'])
large_sts_to_ee_rot_mat = np.eye(6)
large_sts_to_ee_rot_mat[:3, :3] = sts_to_ee_rot_mat
large_sts_to_ee_rot_mat[3:, 3:] = sts_to_ee_rot_mat

# dataset setup
common_ds_args = dict(
    pa_args=args,
    main_dir=os.path.join(args.save_dir, args.experiment_name),
    compress=False,
    env=env,
    load_saved_sts_config=False,
    csv_save_keys=('pose', 'target_pose', 'sts_avg_force', 'sts_in_contact', 'action'),
    mp4_save_keys=('sts_raw_image', 'wrist_rgb')
)

################################ MAIN EXPERIMENT #####################################
try:
    ################ RESET ################
    print("Freedrive is engaged. Move robot into initial calibration pose touching object. Press r to reset.")
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
    reset_joint_pos = env.arm_client.robot.get_joint_positions()
    o_t = env.reset(reset_joint_position=env.arm_client.robot.get_joint_positions(), no_enter_for_reset=True)
    reset_pose = o_t['pose']

    print(f"Press space to start calibration. Ensure that environment remains fixed.")
    bk.wait_for_key(" ")

    ############## CALIBRATION LOOP ################
    for calib_type in ('normal', 'shear_x', 'shear_y', 'torque'):
        num_trajs = params[calib_type]['num_calib_trajs']
        max_calib_dist = params[calib_type]['max_calib_dist']

        if calib_type == "torque":
            max_calib_dist *= np.pi / 180

        for i in range(num_trajs):
            total_motion_act = np.zeros(6)
            demonstration_data = []
            ts = 0

            ds = DictDataset(**common_ds_args, dataset_name=f"{calib_type}-{i}")
            if ds._loaded:
                print(f"Calib traj for {calib_type}, iteration {i+1}/{num_trajs} exists, skipping.")
                continue
            print("Resetting environment (robot should only move minorly).")
            o_t = env.reset(reset_joint_position=reset_joint_pos, no_enter_for_reset=True)

            print(f"pose before mod: {o_t['pose']}")

            # divide params by two since that's the total max allowable change
            reset_mod = (np.random.rand(6) - .5) * 2 * np.array(params['calib_reset_random_change_sts_frame']) / 2
            reset_mod[3:] *= np.pi / 180
            reset_mod_rot = large_sts_to_ee_rot_mat @ reset_mod
            o_t, r, term, trunc, info = env.step(reset_mod_rot)
            time.sleep(args.sleep_after_act_time)
            o_t, r, term, trunc, info = env.step(np.zeros_like(reset_mod_rot))

            print(f"pose after mod: {o_t['pose']}")

            motion = STS_MOTION_DICT[calib_type]
            if calib_type != "normal" and i / num_trajs >= 0.5:
                motion = -motion
                print(f"Starting calibration of negative {calib_type}, iteration {i+1}/{num_trajs}.")
            else:
                print(f"Starting calibration of {calib_type}, iteration {i+1}/{num_trajs}.")

            if calib_type != "normal":
                # initial_z_dist = np.sqrt(params['normal']['max_calib_dist']**2 - max_calib_dist**2)
                initial_z_dist = params['normal']['max_calib_dist']
            else:
                initial_z_dist = -np.inf

            while np.abs((large_sts_to_ee_rot_mat @ total_motion_act)[STS_AXIS_DICT[calib_type]]) < max_calib_dist:
                if np.abs((sts_to_ee_rot_mat @ total_motion_act[:3])[2]) < initial_z_dist:
                    a_t = large_sts_to_ee_rot_mat.T @ STS_MOTION_DICT['normal']
                else:
                    a_t = large_sts_to_ee_rot_mat.T @ motion

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

            print(f"Motion for {calib_type}-{i} complete, releasing force.")
            num_release_steps = int(env.control_freq * args.force_release_time)
            release_act = -total_motion_act / num_release_steps
            for ts in range(num_release_steps):
                env.step(release_act)

except Exception as e:
    print(e)
finally:
    bk.restore_terminal_settings()
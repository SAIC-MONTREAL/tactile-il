import sys
import copy
import os
import time
import argparse
from datetime import datetime
import numpy as np
from omegaconf import OmegaConf
import json
import pathlib
from functools import partial
import torch
import hydra
from grpc import RpcError

from place_from_pick_learning.utils.learning_utils import (
    set_seed,
    load_model,
    process_obs
)
import place_from_pick_learning.utils.debugging as debug

from contact_il.imitation.device_utils import CollectDevice
from contact_il.data.dict_dataset import DictDataset

import contact_il
ROS_INSTALLED = contact_il.ROS_INSTALLED


# test results will be saved as ${CIL_DATA_DIR}/tests/MODEL_ID/${experiment_id}
# where MODEL_ID is the subfolder the model is saved in (with / replaced with _) after {CIL_DATA_DIR}/models

parser = argparse.ArgumentParser()
parser.add_argument('--save_top_dir', type=str, default=os.path.join(os.environ['CIL_DATA_DIR'], 'tests'),
                    help='Top level directory for saving results.')
parser.add_argument('--experiment_id', type=str, default=datetime.today().strftime('%Y-%m-%d_%H-%M-%S'),
            help='Name of experiment, subfolder prefix to save results in')
parser.add_argument('--model_top_dir', type=str, default=os.path.join(os.environ['CIL_DATA_DIR'], 'models'),
                    help='Top level directory for loading models.')
parser.add_argument('--model_subdir', type=str, default="2022-12-16/_10-37-13",
            help='Name of subfolder containing weights and hyperparameters for trained model')
parser.add_argument('--device', type=str, default="cuda", help='Device used to run experiments')
parser.add_argument('--n_episodes', type=int, default=1, help='Number of episodes to test on')
parser.add_argument('--random_seed', type=int, default=102, help='Random seed')
parser.add_argument('--render', action='store_true', default=False, help='render using env.render.')
parser.add_argument('--collect_device', type=str, default='keyboard')
parser.add_argument('--sim', action='store_true', help="Use a simulated robot")
parser.add_argument('--sts_source_vid', type=str, default="", help="Use a simulated sts sensor from video.")
parser.add_argument('--auto_reset', action='store_true', help="call reset with auto_reset set to true.")
parser.add_argument('--reset_model_subdir', type=str, default="")
parser.add_argument('--sts_config_dir_override', type=str, default="")
args = parser.parse_args()


debug.nice_print()
# --------- hydra loading + version handling ----------------------------------------------
model_config_path = os.path.join(args.model_top_dir, args.model_subdir, "model-config.yaml")

cfg = OmegaConf.load(model_config_path)
if hydra.__version__ == "1.0.6":
    dataset_config = cfg.dataset_config
    model_config = cfg.model_config
else:
    dataset_config = cfg.dataset.dataset_config
    model_config = cfg.model.model_config

if args.reset_model_subdir != "":
    reset_model_config_path = os.path.join(args.model_top_dir, args.reset_model_subdir, "model-config.yaml")

    reset_cfg = OmegaConf.load(reset_model_config_path)
    if hydra.__version__ == "1.0.6":
        reset_dataset_config = reset_cfg.dataset_config
        reset_model_config = reset_cfg.model_config
    else:
        reset_dataset_config = reset_cfg.dataset.dataset_config
        reset_model_config = reset_cfg.model.model_config

# --------- env setup, automatically load as much as possible --------------------------------------------

test_main_dir = os.path.join(args.save_top_dir, args.model_subdir, str(args.n_episodes) + '_test_eps')
if args.reset_model_subdir != "":
    reset_test_main_dir = os.path.join(args.save_top_dir, args.reset_model_subdir, str(args.n_episodes) + '_test_eps')

# check if we already have all test eps, then don't bother loading model and just end script
fake_test_ds = DictDataset(
    pa_args=args, dataset_name=args.experiment_id, main_dir=test_main_dir, env=None, load_saved_sts_config=False)
if fake_test_ds._params['actual_n_episodes'] >= args.n_episodes:
    exit = True
    print(f"Test set at {fake_test_ds._dir} already complete.")
    if args.reset_model_subdir != "":
        exit = False
        reset_fake_test_ds = DictDataset(
            pa_args=args, dataset_name=args.experiment_id, main_dir=reset_test_main_dir, env=None, load_saved_sts_config=False)
        if reset_fake_test_ds._params['actual_n_episodes'] >= args.n_episodes:
            exit = True
            print(f"Test set at {reset_fake_test_ds._dir} already complete.")
    if exit:
        print("exiting")
        sys.exit(0)

# this import is a bit slow, so only do it once we confirm we need to do this test
from contact_panda_envs.envs import *

env_cfg_file = dataset_config.env_config_file
with open(env_cfg_file, "r") as f:
    env_params = json.load(f)
env_name = env_params["env_name"]
env_args = {}

# load sts config from saved dataset, not env param file
ds_dir = os.path.dirname(env_cfg_file)
ds_main_dir = os.path.dirname(ds_dir)
ds_name = pathlib.PurePath(os.path.dirname(env_cfg_file)).name

if args.sts_config_dir_override != "":
    env_args['sts_config_dir'] = args.sts_config_dir_override
elif "sts_namespaces" in env_params and len(env_params['sts_namespaces']) > 0:
    if len(env_params['sts_namespaces']) > 1:
        env_args['sts_config_dirs'] = [os.path.join(ds_dir, "sts_configs", ns) for ns in env_params['sts_namespaces']]
    else:
        env_args['sts_config_dir'] = os.path.join(ds_dir, "sts_configs", env_params['sts_namespaces'][0])

if args.sim:
    env_args['sim_override'] = args.sim
if args.sts_source_vid != "":
    env_args['sts_source_vid'] = args.sts_source_vid

# initialize env
main_env = globals()[env_name](**env_args)
main_env.seed(args.random_seed)
set_seed(args.random_seed)

if args.reset_model_subdir != "":
    reset_env_cfg_file = reset_dataset_config.env_config_file
    with open(reset_env_cfg_file, "r") as f:
        reset_env_params = json.load(f)
    reset_env_name = reset_env_params["env_name"]
    reset_env_args = {}

    # load sts config from saved dataset, not env param file
    reset_ds_dir = os.path.dirname(reset_env_cfg_file)
    reset_ds_main_dir = os.path.dirname(reset_ds_dir)
    reset_ds_name = pathlib.PurePath(os.path.dirname(reset_env_cfg_file)).name

    if args.sts_config_dir_override != "":
        reset_env_args['sts_config_dir'] = args.sts_config_dir_override
    if "sts_namespaces" in reset_env_params and len(reset_env_params['sts_namespaces']) > 0:
        if len(reset_env_params['sts_namespaces']) > 1:
            reset_env_args['sts_config_dirs'] = [os.path.join(reset_ds_dir, "sts_configs", ns) for ns in reset_env_params['sts_namespaces']]
        else:
            reset_env_args['sts_config_dir'] = os.path.join(reset_ds_dir, "sts_configs", reset_env_params['sts_namespaces'][0])

    if args.sim:
        reset_env_args['sim_override'] = args.sim
    if args.sts_source_vid != "":
        reset_env_args['sts_source_vid'] = args.sts_source_vid

    # initialize env
    reset_env_args['client_override_dict'] = {
        'arm_client': main_env.arm_client,
        'gripper_client': main_env.gripper_client,
        'camera_client': main_env.camera_client,
        'sts_clients': {'sts': main_env.sts_clients['sts']}
    }
    reset_env = globals()[reset_env_name](**reset_env_args)
    reset_env.seed(args.random_seed)


polymetis_control = hasattr(main_env, 'polymetis_control') and main_env.polymetis_control
freedrive_available = not main_env.sim or polymetis_control

# use DictDataset to confirm that dataset/model/new env match
demo_ds = DictDataset(pa_args=None, dataset_name=ds_name, main_dir=ds_main_dir, env=main_env)

# new dataset for testing data, also saves parameters and evaluation/performance data
# model_id = args.model_subdir.replace('/', '_')
# test_main_dir = os.path.join(args.save_top_dir, "model_" + model_id)

# same dir structure as trained models
main_test_ds = DictDataset(  # don't need to load saved sts config, since we already loaded it from the model folder config
    pa_args=args, dataset_name=args.experiment_id, main_dir=test_main_dir, env=main_env, load_saved_sts_config=False,
    csv_save_keys=('pose', 'sts_avg_force', 'sts_in_contact', 'action'),
    mp4_save_keys=('sts_raw_image', 'wrist_rgb'),
)

if args.reset_model_subdir != "":
    reset_test_ds = DictDataset(  # don't need to load saved sts config, since we already loaded it from the model folder config
        pa_args=args, dataset_name=args.experiment_id, main_dir=reset_test_main_dir, env=reset_env, load_saved_sts_config=False,
        csv_save_keys=('pose', 'sts_avg_force', 'sts_in_contact', 'action'),
        mp4_save_keys=('sts_raw_image', 'wrist_rgb'),
    )

# device for starts, interventions, resets, and recording success/fail
dev = CollectDevice(device_type=args.collect_device)

# auto reset setup
auto_reset_kwargs = {}
if args.auto_reset:
    auto_reset_kwargs['auto_reset'] = True
    auto_reset_kwargs['collect_device'] = dev
first = True
reset_env_running = False
rpc_error = False

# ----------- load model --------------------------------------------------------------------------------
model_weights_path = os.path.join(args.model_top_dir, args.model_subdir, "model-weights.pth")
main_model = load_model(model_config, path=model_weights_path, device=args.device, mode="eval")

p_obs = partial(process_obs,
    rotation_representation=dataset_config.obs_rotation_representation,
    stored_rotation_representation=dataset_config.stored_obs_rotation_representation,
    n_frame_stacks=dataset_config.n_frame_stacks,
    load_tensor=True,
    device=args.device,
)

if args.reset_model_subdir != "":
    reset_model_weights_path = os.path.join(args.model_top_dir, args.reset_model_subdir, "model-weights.pth")
    reset_model = load_model(reset_model_config, path=reset_model_weights_path, device=args.device, mode="eval")

    reset_p_obs = partial(process_obs,
        rotation_representation=reset_dataset_config.obs_rotation_representation,
        stored_rotation_representation=reset_dataset_config.stored_obs_rotation_representation,
        n_frame_stacks=reset_dataset_config.n_frame_stacks,
        load_tensor=True,
        device=args.device,
    )

if cfg.seq_length > 1:
    raise NotImplementedError("Need to implement history for this! Look at Oliver's code.")

# ----------- start gym-style loop for testing ----------------------------------------------------------
while main_test_ds._params['actual_n_episodes'] < args.n_episodes or \
    (args.reset_model_subdir != "" and reset_test_ds._params['actual_n_episodes'] < args.n_episodes):

    if main_test_ds._params['actual_n_episodes'] >= args.n_episodes:
        reset_env_running = True
    elif reset_test_ds._params['actual_n_episodes'] >= args.n_episodes:
        reset_env_running = False

    env = main_env if not reset_env_running else reset_env
    test_ds = main_test_ds if not reset_env_running else reset_test_ds
    model = main_model if not reset_env_running else reset_model

    if not freedrive_available:
        print("Stop interactive marker if it's running, then press space.")
        dev.return_on_press(dev.get_start_stop)
    else:
        print("Deactivating freedrive.")
        env.arm_client.deactivate_freedrive()

    print("Resetting environment.")
    # if rpc_error:
    #     import ipdb; ipdb.set_trace()  # attempt to debug why we always lose it in this case

    if first:
        extra_reset_kwargs = {}
    else:
        extra_reset_kwargs = auto_reset_kwargs

    o_t = env.reset(**extra_reset_kwargs)
    dev.reset_states()
    if args.render:
        env.render()

    if reset_env_running:
        print("Env reset. Press space to start execution IN RESET ENV.")
    else:
        print("Env reset. Press space to start execution.")
    dev.return_on_press(dev.get_start_stop)
    print("Starting execution. Press space to stop execution.")

    while not env.arm_client.robot.is_running_policy():
        print("Controller not started or stopped, attempting to restart..")
        env.arm_client.start_controller()

    done = False
    ep_sas_pairs = []
    ts = 0

    while not done:
        if ts % 10 == 0:
            print(f"Running episode {test_ds._params['actual_n_episodes'] + 1}, timestep {ts}")

        dev.update()

        # forward_start = time.time()
        o_t_processed = p_obs([(o_t, None, None)])
        # Remove filler batch
        o_t_processed = {k:v.unsqueeze(0) for k,v in o_t_processed.items()}

        forward_start = time.time()
        a_t, _ = model(o_t_processed)
        # REMINDER: forward time is average of 15ms, 22ms worst case.
        # print(f"FORWARD TIME: {time.time() - forward_start}")

        # Remove filler batch, use only latest action
        a_t = a_t[0, -1]
        a_t = a_t.detach().cpu().numpy()

        # print(f"ACTION: {a_t}")

        # print(f"Processing + forward time: {time.time() - forward_start}")

        try:
            o_t1, r, term, trunc, info = env.step(a_t)

            ep_sas_pairs.append((o_t, a_t, o_t1))
            o_t = o_t1

        except RpcError as e:
            print(f"RpcError: {e}, ending episode early.")
            term = True
            # rpc_error = True

        # end on timeout, success/failure, or user pressing space
        done = term or trunc or dev.start_stop or done  # once done is True, it stays True

        ts += 1

        if args.render:
            env.render()

        if ROS_INSTALLED and env.arm_client.in_reflex():
            print("Robot in reflex! Check output from nuc panda_control.launch.")
            print("Press e to recover from error and discard episode.")
            done = True
            keep_ep = False
            dev.return_on_press(dev.get_error_recovery)
            if not env.arm_client.recover_from_errors():
                raise ValueError("Arm not able to recover from errors automatically...")

    if term:
        end_str = 'terminated'
    elif trunc:
        end_str = 'timed out'
    else:
        end_str = 'stopped by user'


    print(f"Episode {end_str}. Press s for success, or f for failure.")
    suc, _ = dev.return_on_any_press([dev.get_success, dev.get_failure])

    perf_dict = {'success': suc}
    test_ds.save_ep(ep_sas_pairs, perf_dict=perf_dict)
    if args.reset_model_subdir != "":
        reset_env_running = not reset_env_running

    if reset_env_running:
        print(f"Press r to reset TO RESET ENV, or hold backspace to delete last ep.")
    else:
        print(f"Press r to reset env, or hold backspace to delete last ep.")
    print(f"Freedrive toggled to allow moving robot to better reset pose. Press space to toggle.")
    env.arm_client.activate_freedrive()
    reset = False
    while not reset:
        reset, delete, toggle_freedrive = dev.return_on_any_press([dev.get_reset, dev.get_delete, dev.get_start_stop])
        if delete:
            test_ds.remove_last_ep()
            if args.reset_model_subdir != "":
                reset_env_running = not reset_env_running
        if toggle_freedrive and freedrive_available:
            env.arm_client.toggle_freedrive()
            print(f"Freedrive toggled. Freedrive activated? {env.arm_client._freedrive_is_active}")

    first = False

print("Dataset complete, completing final reset")
env = main_env if not reset_env_running else reset_env
o_t = env.reset(**extra_reset_kwargs)
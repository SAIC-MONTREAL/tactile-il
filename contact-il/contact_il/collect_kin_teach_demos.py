import copy
import os
import time
import argparse
from datetime import datetime
import numpy as np
import json

from contact_panda_envs.envs import *
import contact_il
ROS_INSTALLED = contact_il.ROS_INSTALLED
# if ROS_INSTALLED:
#     from place_from_pick_learning.envs.place_from_pick_env import PlaceFromPickEnv
from panda_polymetis.utils.poses import geodesic_error
from transform_utils.pose_transforms import PoseTransformer
from sts.scripts.helper import read_json
from place_from_pick_learning.utils.debugging import nice_print
from pysts.utils import eul_rot_json_to_mat

from contact_il.imitation.device_utils import CollectDevice
from contact_il.imitation.utils import DoneMask, env_movement_change_threshold
from contact_il.data.dict_dataset import DictDataset
from contact_il.imitation.ft_adapted_replay import FTAdapter


parser = argparse.ArgumentParser()
parser.add_argument('environment', type=str)
parser.add_argument('dataset_name', type=str,  help='Name of demonstration folder')
parser.add_argument('--save_dir', type=str, default=os.path.join(os.environ["CIL_DATA_DIR"], 'demonstrations'),
                    help='Top level directory for dataset.')
parser.add_argument('--dataset_id', type=str, default="demonstration_data", help='DEPRECATED:Sub-name of file where demonstration data will saved')
parser.add_argument('--n_episodes', type=int, default=16, help='Number of demonstration episodes to collect')
parser.add_argument('--device', type=str, default='keyboard')
parser.add_argument('--act_type', type=str, default='dpose_rvec')
parser.add_argument('--dt', type=float, default=0.30, help='Time step of data collected, if not predefined in env.')
parser.add_argument('--res', type=int, default=128, required=False, help='Image resolution, if not predefined in env.')
parser.add_argument('--render', action='store_true', default=False, help='render using env.render.')
parser.add_argument('--compress', action='store_true', default=False, help='Compress dataset trajs.')
parser.add_argument('--action_multiplier', type=float, default=1.0)
parser.add_argument('--gripper_action_multiplier', type=float, default=1.0)
parser.add_argument('--sts_action_multiplier', type=float, default=0.05)
parser.add_argument('--init_move_thresh', type=float, default=.005, help='Minimum geodesic change before data is recorded.')
parser.add_argument('--random_seed', type=int, default=0, help='Random seed, used for env resets.')
parser.add_argument('--sts_config_dir', type=str, default=os.environ['STS_CONFIG'], help="String for first sts config dir")
parser.add_argument('--sts_config_dir2', type=str, required=False, help="String for second sts config dir")
parser.add_argument('--sim', action='store_true', help="Use a simulated robot")
parser.add_argument('--sts_source_vid', type=str, default="", help="Use a simulated sts sensor from video.")
parser.add_argument('--sts_replay_source_vid', type=str, default="", help="Use a separate video for replays.")
parser.add_argument('--no_replay_demo', action='store_true', help="Turn off replaying the demo to recreate true dynamics.")
parser.add_argument('--ft_adapted_replay', type=str, default="", help="Use force-torque adapted replay. Options: "
                    "open_loop_delft, closed_loop_pid")
parser.add_argument('--replay_dataset_name', type=str, default="", help="Dataset folder name (under save_dir) containing raw human "
    "trajectories to replay.")
parser.add_argument('--auto_reset', action='store_true', help="call reset with auto_reset set to true.")
parser.add_argument('--reset_environment', type=str, default="")
parser.add_argument('--reset_replay_dataset_name', type=str, default="")

args = parser.parse_args()

nice_print(5)

# env setup
sts_client_cfg_dirs = dict()
if args.environment == "PlaceFromPickEnv":
    from place_from_pick_learning.envs.place_from_pick_env import PlaceFromPickEnv
    main_env = PlaceFromPickEnv(
        img_res=(args.res,args.res),
        obs_act_fixed_time=args.dt - 0.10,
        dt=args.dt
    )
    sim = False
    grip_in_act = True
    if hasattr(main_env, 'tactile_client'):
        sts_client_cfg_dirs['sts'] = main_env.tactile_client._config_dir
    load_saved_sts_config = False
    query_error_type = False
    csv_save_keys = ()
    mp4_save_keys = ()
else:
    rate = None
    env_args = {}
    if args.sts_config_dir: env_args['sts_config_dir'] = args.sts_config_dir
    if args.sts_source_vid != "": env_args['sts_source_vid'] = args.sts_source_vid
    if args.sim: env_args['sim_override'] = args.sim
    main_env = globals()[args.environment](**env_args)
    # main_env.seed(args.random_seed)  # do NOT want this if we're stopping and restarting
    sim = main_env.sim
    grip_in_act = main_env.grip_in_action
    if main_env._has_sts:
        for ns in main_env.sts_clients:
            sts_client_cfg_dirs[ns] = main_env.sts_clients[ns]._config_dir
    load_saved_sts_config = True
    # query_error_type = True
    query_error_type = False
    csv_save_keys = ['pose', 'sts_avg_force', 'sts_in_contact', 'action']
    mp4_save_keys = ('sts_raw_image', 'wrist_rgb')

    if "ROS" in args.environment:
        csv_save_keys.extend(['raw_world_pose', 'force_torque_internal'])

    if args.reset_environment:
        env_args['client_override_dict'] = {
            'arm_client': main_env.arm_client,
            'gripper_client': main_env.gripper_client,
            'camera_client': main_env.camera_client,
            'sts_clients': {'sts': main_env.sts_clients['sts']}
        }
        reset_env = globals()[args.reset_environment](**env_args)
        # reset_env.seed(args.random_seed)  # do NOT want this if we're stopping and restarting

polymetis_control = hasattr(main_env, 'polymetis_control') and main_env.polymetis_control
freedrive_available = not sim or polymetis_control
doing_replay = False
doing_raw_dataset_replay = False
first = True
reset_env_running = False

# if args.ft_adapted_replay != "": ft_replay_adapter = FTAdapter(args.ft_adapted_replay, env.sts_clients['sts'])
# it's okay to only use the main env for this since we don't use the reset characteristics (which is the only diff thing)
if args.ft_adapted_replay != "": ft_replay_adapter = FTAdapter(args.ft_adapted_replay, main_env)

# dataset setup
main_ds = DictDataset(
    pa_args=args,
    dataset_name=args.dataset_name,
    main_dir=os.path.join(args.save_dir, args.environment),
    compress=args.compress,
    env=main_env,
    load_saved_sts_config=load_saved_sts_config,
    csv_save_keys=csv_save_keys,
    mp4_save_keys=mp4_save_keys
)

raw_ds_sts_switch_in_action = False
if args.replay_dataset_name != "":
    raw_ds = DictDataset(
        pa_args=None,
        dataset_name=args.replay_dataset_name,
        main_dir=os.path.join(args.save_dir),
        env=main_env,  # worth verifying that envs are the same
        load_saved_sts_config=False,  # only want the trajectories
        load_env_ignore_keys={'env_name', 'sts_initial_mode', 'sts_no_switch_override', 'state_data',
                              'sts_switch_in_action'}  # allow tactile only env
    )
    main_ds.attach_raw_dataset_for_replay(raw_ds)

    # need whether raw_ds had sts switch in action for later
    with open(os.path.join(raw_ds._dir, 'env_parameters.json')) as f:
        raw_ds_env_cfg = json.load(f)
        raw_ds_sts_switch_in_action = raw_ds_env_cfg['sts_switch_in_action']

# reset dataset setup
if args.reset_environment != "":
    reset_ds = DictDataset(
        pa_args=args,
        dataset_name=args.dataset_name,
        main_dir=os.path.join(args.save_dir, args.reset_environment),
        compress=args.compress,
        env=reset_env,
        load_saved_sts_config=load_saved_sts_config,
        csv_save_keys=csv_save_keys,
        mp4_save_keys=mp4_save_keys
    )

    if args.reset_replay_dataset_name != "":
        reset_raw_ds = DictDataset(
            pa_args=None,
            dataset_name=args.reset_replay_dataset_name,
            main_dir=os.path.join(args.save_dir),
            env=reset_env,  # worth verifying that envs are the same
            load_saved_sts_config=False,  # only want the trajectories
            load_env_ignore_keys={'env_name', 'sts_initial_mode', 'sts_no_switch_override', 'state_data',
                                  'sts_switch_in_action'}  # allow tactile only env
        )
        reset_ds.attach_raw_dataset_for_replay(reset_raw_ds)

# device setup
dev = CollectDevice(device_type=args.device)

# auto reset setup
auto_reset_kwargs = {}
if args.auto_reset:
    auto_reset_kwargs['auto_reset'] = True
    auto_reset_kwargs['collect_device'] = dev

# start gym-style loop
while main_ds._params['actual_n_episodes'] < args.n_episodes or \
      (args.reset_environment != "" and reset_ds._params['actual_n_episodes'] < args.n_episodes):

    if args.reset_environment != "":
        if main_ds._params['actual_n_episodes'] >= args.n_episodes:
            reset_env_running = True
        elif reset_ds._params['actual_n_episodes'] >= args.n_episodes:
            reset_env_running = False

    env = main_env if not reset_env_running else reset_env
    ds = main_ds if not reset_env_running else reset_ds

    if not freedrive_available:
        print("Stop interactive marker if it's running, then press space.")
        dev.return_on_press(dev.get_start_stop)
    else:
        if env.arm_client._freedrive_is_active:
            print("Deactivating freedrive before reset.")
            env.arm_client.deactivate_freedrive()

    # optionally grab trajectory from previously recorded human data (sets doing_raw_dataset_replay)
    if args.replay_dataset_name:
        if ds._params['rdfr_current_ep'] < len(ds._rdfr):
            print(f"Running replay from raw dataset for replay, "
                  f"ep {ds._params['rdfr_current_ep']}, max {len(ds._rdfr) - 1}")
            doing_replay = True
            doing_raw_dataset_replay = True
            raw_demonstration_data, reset_joint_position, recorded_actions = ds.get_rdfr_ep_params()
        else:
            if doing_raw_dataset_replay:  # so we only print this message once
                print(f"No more episodes in raw dataset for replay! Switching to regular human demo mode.")
            doing_raw_dataset_replay = False

    # optionally switch simulated video if there's a different one for replays
    if args.sts_replay_source_vid != "":
        if doing_replay:
            env.sts_clients['sts'].sts.cam.reset_video(args.sts_replay_source_vid)
        else:
            env.sts_clients['sts'].sts.cam.reset_video(args.sts_source_vid)

    # reset with optional auto reset
    print("Resetting environment.")
    if first:
        extra_reset_kwargs = {}
    else:
        extra_reset_kwargs = auto_reset_kwargs
    if doing_replay:
        o_t = env.reset(reset_joint_position=reset_joint_position, **extra_reset_kwargs)
    else:
        o_t = env.reset(**extra_reset_kwargs)
        if env.sts_switch_in_action and not doing_replay:
            env.sts_clients['sts'].set_mode('tactile')
            dummy_act = np.zeros(env.action_space.shape)
            dummy_act[env.act_ind_dict['sts']] = 1.0 * args.sts_action_multiplier
            for _ in range(3):
                o_t, _, _, _, _ = env.step(dummy_act)  # get first data with sensor in tactile
    dev.reset_states()
    if args.render: env.render()

    # allow user to start replay or delete recorded actions
    if doing_replay:
        if args.ft_adapted_replay != "": ft_replay_adapter.reset(demo_traj=raw_demonstration_data)
        if doing_raw_dataset_replay:
            print("Press space to start replay of dataset demo.")
            dev.return_on_press(dev.get_start_stop)
        else:
            print("Press space to start replay of collected demo, or hold backspace to delete collected trajectory.")
            if env.sts_switch_in_action:
                print("Press T during replay to switch to tactile mode.")
            start, delete = dev.return_on_any_press([dev.get_start_stop, dev.get_delete])
            if delete:
                print("Deleted recorded actions.")
                doing_replay = False
                recorded_actions = []
                raw_demonstration_data = []

        if polymetis_control:
            while not env.arm_client.robot.is_running_policy():
                print("Controller not started or stopped, attempting to restart..")
                env.arm_client.start_controller()

    # allow user to start demo or delete previous ep
    if not doing_replay:
        # initialize recorded actions
        recorded_actions = []
        raw_demonstration_data = []

        # record joint position so that we reset to the exact same position on the replay

        if polymetis_control:
            reset_joint_position = env.arm_client.robot.get_joint_positions()
        else:
            reset_joint_position = env.arm_client.joint_position
        # print("Press space to start freedrive + collection simultaneously, or hold backspace to delete last ep.")
        print("Press space to start freedrive + collection simultaneously")
        if not freedrive_available:
            print("Start interactive marker before pressing space!")

        start = False
        while not start:
            start, delete = dev.return_on_any_press([dev.get_start_stop, dev.get_delete])
            # if delete:
            #     ds.remove_last_ep(query_error_type=query_error_type)

    print(f"Collecting episode {ds._params['actual_n_episodes'] + 1}/{args.n_episodes}")
    if not doing_replay:
        if freedrive_available:
            env.arm_client.activate_freedrive()
        env_movement_change_threshold(env, init_move_thresh=args.init_move_thresh, polymetis_control=polymetis_control)

    done = False
    demonstration_data = []

    ts = 0
    keep_ep = True
    while not done:
        step_start = time.time()
        if ts % 10 == 0:
            print(f"Recording episode {ds._params['actual_n_episodes'] + 1}, timestep {ts}")

        dev.update()

        # print(f"IN CONTACT?: {o_t['sts_in_contact']}")

        if grip_in_act:
            a_g_t = dev.gripper * args.gripper_action_multiplier

        if doing_replay:
            # replay actions
            a_t = copy.deepcopy(recorded_actions[ts])

            if env.sts_switch_in_action or (not env.sts_switch_in_action and raw_ds_sts_switch_in_action):
                a_t = a_t[:-1]

            if args.ft_adapted_replay != "":
                a_t = ft_replay_adapter.get_action_modifier(ts, o_t)

            if env.sts_switch_in_action:
                if doing_raw_dataset_replay:
                    a_t = np.concatenate([a_t, [recorded_actions[ts][-1]]])
                else:
                    a_t = np.concatenate([a_t, [dev.sts_switch * args.sts_action_multiplier]])

            o_t1, r, term, trunc, info = env.step(a_t)

        else:
            # collect actions
            dummy_act = np.zeros(env.action_space.shape)
            if grip_in_act:
                dummy_act[env.act_ind_dict['grip']] = a_g_t

            if env.sts_switch_in_action:
                sts_act = 1.0 * args.sts_action_multiplier  # always tactile mode during demo
                dummy_act[env.act_ind_dict['sts']] = sts_act

            o_t1, r, term, trunc, info = env.step(dummy_act, ignore_motion_act=True)

            # get action from delta pos or vel from info
            a_t = info['recorded_motion'][args.act_type] * args.action_multiplier

            if grip_in_act:
                a_t = np.concatenate([a_t, [a_g_t]])

            if env.sts_switch_in_action:
                a_t = np.concatenate([a_t, [sts_act]])

            recorded_actions.append(a_t)

            if not args.no_replay_demo:
                raw_demonstration_data.append((o_t, a_t, o_t1))

        # ------------------- data recording --------------------------------------------------------
        demonstration_data.append((o_t, a_t, o_t1))
        o_t = o_t1

        # end on timeout, success/failure, or user pressing space
        done = term or trunc or dev.start_stop or done  # once done is True, it stays True
        done = done or (doing_replay and ts == len(recorded_actions) - 1)  # also end if doing replay and no more actions

        ts += 1

        if args.render: env.render()

        if ROS_INSTALLED and env.arm_client.in_reflex():
            print("Robot in reflex! Check output from nuc panda_control.launch.")
            print("Press e to recover from error and discard episode.")
            done = True
            keep_ep = False
            dev.return_on_press(dev.get_error_recovery)

        # print(f"STEP TIME: {time.time() - step_start}")

    if args.no_replay_demo:
        if keep_ep:
            ds.save_ep(demonstration_data)
            if args.reset_environment != "":
                reset_env_running = not reset_env_running
    else:
        if doing_replay:
            if keep_ep:
                ds.save_ep(demonstration_data, raw_demo_data=raw_demonstration_data)
                if args.reset_environment != "":
                    reset_env_running = not reset_env_running
            doing_replay = False
        elif not doing_replay:
            doing_replay = True

    if not doing_replay:  # set to False after a replay just finished
        if reset_env_running:
            print(f"Press r to reset TO RESET ENV, or hold backspace to delete last ep.")
        else:
            print(f"Press r to reset env, or hold backspace to delete last ep.")
        if freedrive_available:
            env.arm_client.activate_freedrive()
    else:
        if reset_env_running:
            print(f"Press r to reset TO RESET ENV, or hold backspace to delete recorded action set for replay.")
        else:
            print(f"Press r to reset env, or hold backspace to delete recorded action set for replay.")
    print("Freedrive is still engaged, will disengage on space or reset.")

    reset = False
    while not reset:
        reset, delete, toggle_freedrive = dev.return_on_any_press([dev.get_reset, dev.get_delete, dev.get_start_stop])
        if doing_replay:
            if delete:
                print("Deleted recorded actions.")
                recorded_actions = []
                raw_demonstration_data = []
                doing_replay = False
        else:
            if delete:
                ds.remove_last_ep(query_error_type=query_error_type)
                if args.reset_environment != "":
                    reset_env_running = not reset_env_running
        if toggle_freedrive and freedrive_available:
            env.arm_client.toggle_freedrive()
            print(f"Freedrive toggled. Freedrive activated? {env.arm_client._freedrive_is_active}")

    first = False


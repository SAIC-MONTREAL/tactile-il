import os
import argparse
from datetime import datetime

import numpy as np
import pickle

import contact_panda_envs
from contact_panda_envs.envs import *
from panda_polymetis.control.panda_client import PandaClient
from contact_il.imitation.device_utils import CollectDevice
from contact_il.imitation.utils import env_movement_change_threshold


parser = argparse.ArgumentParser()
parser.add_argument('environment', type=str)
# parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#                     help='Top level directory for dataset. Defaults to this folder.')
parser.add_argument('--save_dir', type=str, default='',
                    help='Top level directory for dataset. Defaults to contact_panda_envs/envs/cabinet/configs/reset_trajs.')
parser.add_argument('--device', type=str, default='keyboard')
parser.add_argument('--act_type', type=str, default='dpose_rvec')
parser.add_argument('--action_multiplier', type=float, default=1.0)
parser.add_argument('--gripper_action_multiplier', type=float, default=1.0)
parser.add_argument('--init_move_thresh', type=float, default=.005, help='Minimum geodesic change before data is recorded.')
parser.add_argument('--sim', action='store_true', help="Use a simulated robot")

args = parser.parse_args()

if args.save_dir == "":
    args.save_dir = os.path.join(os.path.dirname(contact_panda_envs.__file__), 'envs/cabinet/configs/reset_trajs')

# get env & device
# env = globals()[args.environment](sim_override=args.sim, no_sensors_env=args.sim, sts_config_dir=None)
env = globals()[args.environment](sim_override=args.sim, no_sensors_env=True, sts_config_dir=None)
dev = CollectDevice(device_type=args.device)

# move robot to initial goto position
input("Press enter to enable freedrive and move robot + env to first desired goto joint pos")
env.arm_client.activate_freedrive()
input("Freedrive activated. Once arm+env objects are in final state, press enter to set goto joint pos.")
goto_joint_pos = env.arm_client.robot.get_joint_positions()
env.arm_client.deactivate_freedrive()

# reset env state without moving arm
print("Resetting env to goto joint pos to reset various parameters")
env.reset(reset_joint_position=goto_joint_pos, no_enter_for_reset=True)

recorded_actions = []
ts = 0
input("Env reset complete. Press enter to activate freedrive and start collecting reset traj.")
env.arm_client.activate_freedrive()
env_movement_change_threshold(env, init_move_thresh=args.init_move_thresh)

# start collecting reset traj
while not dev.start_stop:
    if ts % 10 == 0:
        print(f"Recording reset traj, timestep {ts}")
    dev.update()

    dummy_act = np.zeros_like(env.action_space)
    if env.grip_in_action:
        dummy_act[-1] = dev.gripper * args.gripper_action_multiplier
    o_t1, r, term, trunc, info = env.step(dummy_act, ignore_motion_act=True)

    # get action from delta pos or vel from info
    a_t = info['recorded_motion'][args.act_type] * args.action_multiplier
    if env.grip_in_action:
        a_t = np.concatenate([a_t, dummy_act[-1]])
    recorded_actions.append(a_t)

    ts += 1

env.arm_client.deactivate_freedrive()
# TODO consider adding one more goto pose here, but i don't think it'll be necessary

input("Recording stopped, freedrive deactivated. Press enter to call regular env reset to verify auto reset will work.")
env.reset()
input("Reset complete. If reset was successful, press enter to activate freedrive.")
env.arm_client.activate_freedrive()
input("Move env + arm close to end pose again, then press enter to verify reset traj")
env.arm_client.deactivate_freedrive()

# start playback
# env.arm_client.move_to_joint_positions(goto_joint_pos)
env.reset(reset_joint_position=goto_joint_pos)
print("move to goto pos complete, starting playback of kin teach traj.")
for act in recorded_actions:
    env.step(act)
print("recorded kin teach traj complete, calling env reset")
env.reset()
input("Env reset complete. If reset was successful again, press enter to save reset traj, otherwise ctrl-c.")

# save it
full_save_dir = os.path.join(args.save_dir, args.environment)
os.makedirs(full_save_dir, exist_ok=True)
save_file = os.path.join(full_save_dir, datetime.now().strftime("%m-%d-%y-%H_%M_%S") + '.pkl')
data = {
    'goto_joint_pos': np.array(goto_joint_pos),
    'recorded_actions': np.array(recorded_actions)
}
with open(save_file, 'wb') as f:
    pickle.dump(data, f)
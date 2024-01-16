import os
import time
import copy

import torch
import matplotlib.pyplot as plt
import numpy as np

from polymetis import RobotInterface
from torchcontrol.transform import Rotation as R
import torchcontrol as toco

from lilac.robot.env import Rate
from lilac.robot.env import FrankaEnv


# options
rate = 10
collect_length = 10  # seconds
replay_using_vel = False
sim = False
# dp_replay_multiplier = 1.0  # currently using absolute poses for speed of implementation, so this is meaningless
# v_replay_multiplier = 1.0


# set up env
ip_address = os.environ['NUC_IP'] if not sim else "localhost"
n_steps = int(collect_length * rate)
robot = RobotInterface(ip_address=ip_address, enforce_version=False)
env = FrankaEnv(franka_ip=ip_address, home_joint_pos=robot.get_joint_positions(),
                control_hz=rate, include_gripper=False)
rate_obj = Rate(rate)

# tracking variables
ts = 0
# obss = []
# ee_vels = []
# ee_d_poss = []
ee_poss = []
ee_rots = []
r_ee_poss = []
r_ee_rots = []

# set "freedrive"
input("Press enter to start freedriving...")
env.reset()
env.set_kinesthetic(True)
print("Start moving now!!")

for i in range(n_steps):
    obs = env.step(None)  # this includes a sleep
    # ee_poss.append(obs['ee_pose'])

    ee_pos, ee_rot = env.robot.get_ee_pose()
    ee_poss.append(ee_pos.numpy())
    ee_rots.append(ee_rot.numpy())

    if i % 10 == 0:
        print(f"Recording: timestep {i} of {n_steps}")

input("Recording complete. Return arm close to initial pose and press enter for reset...")
env.set_kinesthetic(False)  # does a reset
env.reset()

input("Press enter to start replay of recorded trajectory...")


for i in range(n_steps):
    # for now, we'll just use the poses and the underlying controller directly..technically, should produce identical
    # results as deltas, but can't use the multiplier.
    env.robot.update_desired_ee_pose(position=torch.from_numpy(ee_poss[i]), orientation=torch.from_numpy(ee_rots[i]))
    rate_obj.sleep()

    ee_pos, ee_rot = env.robot.get_ee_pose()
    r_ee_poss.append(ee_pos.numpy())
    r_ee_rots.append(ee_rot.numpy())

    if i % 10 == 0:
        print(f"Replaying: timestep {i} of {n_steps}")

env.set_kinesthetic(True)
input("Move robot to new near reset pose, then press enter (no reset occurs)...")
# env.reset()

env.close()

ee_poss = np.array(ee_poss)
ee_rots = np.array(ee_rots)
r_ee_poss = np.array(r_ee_poss)
r_ee_rots = np.array(r_ee_rots)

trans_fig = plt.figure()
ax = trans_fig.add_subplot(projection='3d')

for (poss, label) in zip([ee_poss, r_ee_poss], ["Recorded", "Replayed"]):
    ax.plot(poss[:, 0], poss[:, 1], poss[:, 2], label=label)
ax.set_title("Translations")
ax.legend()

# rot_fig = plt.figure()
# for (poss, label) in zip([ee_poss, replay_ee_poss], ["Recorded", "Replayed"]):
#     for i in range(3, 6):
#         plt.plot(poss[:, i], label=f"{label}_{i}")
# plt.title("Rotations")
# plt.legend()



import ipdb; ipdb.set_trace()
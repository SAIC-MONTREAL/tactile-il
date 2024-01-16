""" Quick test to see if external force torque estimate from libfranka is useful during a kin teach episode. """

import torch
import time
import copy

from polymetis import RobotInterface
from torchcontrol.transform import Rotation as R
import torchcontrol as toco
import os
import numpy as np
from lilac.robot.env import Rate
from lilac.robot.env import FrankaEnv
import frankx


# options
rate = 10
collect_length = 10  # seconds

# set up env
n_steps = int(collect_length * rate)
fx_robot = frankx.Robot(fci_ip=os.environ['ROBOT_IP'])
robot = RobotInterface(ip_address=os.environ['NUC_IP'], enforce_version=False)
env = FrankaEnv(franka_ip=os.environ['NUC_IP'], home_joint_pos=robot.get_joint_positions(), control_hz=rate)

# tracking variables
ee_poss = []
O_F_exts = []
K_F_exts = []

# set "freedrive"
input("Press enter to start freedriving...")
env.set_kinesthetic(True)
print("Start moving now!!")

for i in range(n_steps):
    state = fx_robot.read_once()
    O_F_exts.append(state.O_F_ext_hat_K)
    K_F_exts.append(state.K_F_ext_hat_K)
    print(f"Estimated F_ext (base frame): {state.O_F_ext_hat_K}")
    print(f"Estimated F_ext (base frame): {state.K_F_ext_hat_K}")

    obs = env.step(None)
    ee_poss.append(obs['ee_pose'])

    if i % 10 == 0:
        print(f"Recording: timestep {i} of {n_steps}")


input("Recording complete. Return arm close to initial pose and press enter for reset...")
env.set_kinesthetic(False)
env.close()

O_F_exts = np.array(O_F_exts)
K_F_exts = np.array(K_F_exts)
ee_poss = np.array(ee_poss)

np.set_printoptions(suppress=True, precision=4)
import ipdb; ipdb.set_trace()
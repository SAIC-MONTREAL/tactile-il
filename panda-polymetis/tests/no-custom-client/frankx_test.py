import torch
import time
import copy

from polymetis import RobotInterface
from torchcontrol.transform import Rotation as R
import torchcontrol as toco
import os
import numpy as np
import frankx

# options
pos_diff = torch.Tensor([0.0, 0.0025, 0.0])
num_pts = 40

# set up env
fx_robot = frankx.Robot(fci_ip=os.environ['ROBOT_IP'])
robot = RobotInterface(ip_address=os.environ['NUC_IP'])

# test getting state without running impedance controller
state = fx_robot.read_once()
print(f"Estimated F_ext (base frame): {state.O_F_ext_hat_K}")
print(f"Estimated F_ext (base frame): {state.K_F_ext_hat_K}")

import ipdb; ipdb.set_trace()

ee_pos, ee_quat = robot.get_ee_pose()
robot.start_cartesian_impedance()

for i in range(num_pts):
    state = fx_robot.read_once()
    print(f"Estimated F_ext (base frame): {state.O_F_ext_hat_K}")
    print(f"Estimated F_ext (base frame): {state.K_F_ext_hat_K}")

    ee_pos += pos_diff
    robot.update_desired_ee_pose(position=ee_pos)
    time.sleep(0.1)

robot.terminate_current_policy()
print("test complete!")

import ipdb; ipdb.set_trace()
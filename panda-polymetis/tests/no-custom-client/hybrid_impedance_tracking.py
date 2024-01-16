import torch
import time
import copy

from polymetis import RobotInterface
from torchcontrol.transform import Rotation as R
import torchcontrol as toco
import os
import numpy as np


if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(ip_address=os.environ['NUC_IP'], enforce_version=False)
    torch_policy = toco.policies.HybridJointImpedanceControl(
        joint_pos_current=robot.get_joint_positions(),
        Kq=robot.Kq_default,
        Kqd=robot.Kqd_default,
        Kx=robot.Kx_default,
        Kxd=robot.Kxd_default,
        robot_model=robot.robot_model,
        ignore_gravity=robot.use_grav_comp,
    )
    robot.send_torch_policy(torch_policy=torch_policy, blocking=False)

    torch.set_printoptions(sci_mode=False, precision=4)

    # Cartesian impedance control
    print("Performing Cartesian impedance control...")
    ee_pos, ee_quat = robot.get_ee_pose()
    first_ee_pos, first_ee_quat = robot.get_ee_pose()
    print(f"Start pos: {ee_pos}")
    # num_pts = 30
    num_pts = 10
    pos_diff = torch.Tensor([0.0, 0.0, -.0025])
    # pos_diff = torch.Tensor([0.01, 0.0, 0.0])
    # pos_diff = torch.Tensor([0.0, 0.0, 0.0])
    # rot_diff_R = R.from_rotvec(torch.Tensor([0.0, -1 * np.pi / 180, 0.]))
    rot_diff_R = R.from_rotvec(torch.Tensor([0.0, 0., 0.]))

    last_pos = copy.deepcopy(first_ee_pos)
    last_quat = copy.deepcopy(first_ee_quat)

    robot.start_cartesian_impedance()

    for i in range(num_pts):
        ee_pos += pos_diff
        ee_quat = (R.from_quat(ee_quat) * rot_diff_R).as_quat()
        robot.update_desired_ee_pose(position=ee_pos, orientation=ee_quat)
        time.sleep(0.1)
        cur_ee_pos, cur_ee_quat = robot.get_ee_pose()
        print(f"pos diff, expected: {pos_diff}, actual: {cur_ee_pos - last_pos}")
        print(f"rot diff, expected: {rot_diff_R.as_rotvec()}, actual: "\
              f"{(R.from_quat(last_quat).inv() * R.from_quat(cur_ee_quat)).as_rotvec()}")
        last_pos = copy.deepcopy(cur_ee_pos)
        last_quat = copy.deepcopy(cur_ee_quat)

    time.sleep(1.0)  # give time to settle before calculating
    cur_ee_pos, cur_ee_quat = robot.get_ee_pose()
    print(f"Final pos diff, expected: {ee_pos - first_ee_pos}, actual: {cur_ee_pos - first_ee_pos}")
    print(f"Final rot diff, expected: {(R.from_quat(first_ee_quat).inv() * R.from_quat(ee_quat)).as_rotvec()}, "\
          f"actual: {(R.from_quat(first_ee_quat).inv() * R.from_quat(cur_ee_quat)).as_rotvec()}")

    robot.terminate_current_policy()

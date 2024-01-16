import time
from enum import Enum

import numpy as np

from panda_polymetis.utils.poses import geodesic_error


class DoneMask(Enum):
    TIMEOUT = 0.0
    FAILURE = 1.0
    SUCCESS = 2.0


def env_movement_change_threshold(env, init_move_thresh, polymetis_control=True):
    print(f"Waiting for initial movement above threshold before starting collection.")
    movement_change = np.zeros(4)
    init_pose = env.arm_client.EE_pose
    wait_count = 0
    while np.linalg.norm(movement_change) < init_move_thresh:
        if wait_count == 10:
            print(f"Movement change norm: {np.linalg.norm(movement_change)}, Threshold: {init_move_thresh}")
            wait_count = 0
        if polymetis_control:
            env.arm_client.get_and_update_ee_pose()
        cur_pose = env.arm_client.EE_pose
        movement_change = geodesic_error(init_pose, cur_pose)
        time.sleep(.01)
        wait_count += 1
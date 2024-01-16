import numpy as np
from numpy.linalg import norm
import transforms3d as tf3d
import transforms3d.quaternions as quat
import time


MAX_JERK = 100
MAX_ACC = 1
MAX_VEL = .1
DES_POSES = np.array([
    [3, 4, 5, 1, 0, 0, 0],
])
T_GOAL_TOL = .001
R_GOAL_TOL = .01
# SIM_TIMESTEP = .001
SIM_TIMESTEP = .01


cur_pose = np.array([0., 0., 0., 1., 0., 0., 0.])
set_jerk = np.zeros(6)
set_acc = np.zeros(6)
set_vel = np.zeros(6)
vel_capped = False
acc_capped = False
jer_capped = False
iterations = 0


# TODO pretty sure we're going to have to add a stopping trajectory
# TODO verify that the set jerks, accelerations, and velocities actually do what we expect

# TODO i think what we do instead is:
# 1. given max jerk + max accel + max vel, calculate stopping distance
# 2.
# 1. given a max jerk, immediately start changing set values, using max jerk, until we reach max vel


for des_pose in DES_POSES:
    pos_diff = des_pose[:3] - cur_pose[:3]
    pos_err = norm(pos_diff)
    rot_diff_ax, ang_err = quat.quat2axangle(quat.qmult(quat.qinverse(des_pose[3:]), cur_pose[3:]))

    while pos_err > T_GOAL_TOL or ang_err > R_GOAL_TOL:
        # TODO do with positions first, verify it works, then add in orientation

        # translational vel
        des_vel_t = pos_diff / SIM_TIMESTEP
        des_vel_t_mag = norm(des_vel_t)

        if des_vel_t_mag > MAX_VEL:
            vel_capped = True
            t_dir = des_vel_t / des_vel_t_mag
            capped_vel_t = t_dir * MAX_VEL
        else:
            vel_capped = False
            capped_vel_t = des_vel_t

        # translational acc
        des_acc_t = (capped_vel_t - set_vel[:3]) / SIM_TIMESTEP
        des_acc_t_mag = norm(des_acc_t[:3])

        if des_acc_t_mag > MAX_ACC:
            acc_capped = True
            a_dir = des_acc_t / des_acc_t_mag
            capped_acc_t = a_dir * MAX_ACC
        else:
            acc_capped = False
            capped_acc_t = des_acc_t

        # translational jerk
        des_jer_t = (capped_acc_t - set_acc[:3]) / SIM_TIMESTEP
        des_jer_t_mag = norm(des_jer_t[:3])

        if des_jer_t_mag > MAX_JERK:
            jer_capped = True
            j_dir = des_jer_t / des_jer_t_mag
            capped_jer_t = a_dir * MAX_JERK
        else:
            jer_capped = False
            capped_jer_t = des_jer_t

        # set new internal variables
        set_jerk[:3] = capped_jer_t

        # set pose change based on which caps were activated
        if jer_capped:  # compute new acc and pos diff based on jerk

            # TODO not sure if i need to check acc magnitude again here
            set_acc[:3] = set_acc[:3] + set_jerk[:3] * SIM_TIMESTEP
            set_vel[:3] = set_vel[:3] + set_acc[:3] * SIM_TIMESTEP

        elif acc_capped:
            # acc is capped but jerk is not
            set_acc[:3] = capped_acc_t
            set_vel[:3] = set_vel[:3] + set_acc[:3] * SIM_TIMESTEP

        elif vel_capped:
            # vel is capped but acc and jerk are not
            set_acc[:3] = capped_acc_t
            set_vel[:3] = capped_vel_t

        # TODO in control code, this is where you use 0_T_EE_d instead of cur_pose as the initial value
        new_des_pos = cur_pose[:3] + set_vel[:3] * SIM_TIMESTEP

        # TODO in control code, this line is replaced with actually setting the new pose that is fed to the controller
        # this code just assumes we always reach the new desired pose exactly
        cur_pose[:3] = new_des_pos  # dummy line just to make above clear

        pos_diff = des_pose[:3] - cur_pose[:3]
        pos_err = norm(des_pose[:3] - cur_pose[:3])
        rot_diff_ax, ang_err = quat.quat2axangle(quat.qmult(quat.qinverse(des_pose[3:]), cur_pose[3:]))

        print(f"ERROR: {pos_err}")
        print(f"set_jerk: {set_jerk}")
        print(f"set_acc: {set_acc}")
        print(f"set_vel: {set_vel}")
        print(f"des_pos: {new_des_pos}")


        iterations += 1

        if iterations > 10000:
            import ipdb; ipdb.set_trace()
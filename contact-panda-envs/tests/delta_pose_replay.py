import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import quaternion
import transforms3d as tf3d

from contact_panda_envs.envs import *
# from contact_panda_envs.envs.cabinet.cabinet_contact import PandaCabinetOneFingerNoSTS6DOF
from transform_utils.pose_transforms import PoseTransformer

# conflict between cv2 and matplotlib resolved by this
# import matplotlib
# matplotlib.use("")


# options
rate = 10
collect_length = 10  # seconds
replay_using_vel = False
dp_replay_multiplier = 1.0
v_replay_multiplier = 1.0

# env = PandaCabinetOneFingerNoSTS6DOF(config_override_dict=dict(control_freq=rate, max_real_time=collect_length), sim_override=True)
# env = PandaTestOneFingerNoSTS6DOFRealNoRandomInit(config_override_dict=dict(control_freq=rate, max_real_time=collect_length))
env = PandaTopGlassOrbOneFingerNoSTS6DOFRealNoRandomInit(config_override_dict=dict(control_freq=rate, max_real_time=collect_length))

sim = env.sim

obs = env.reset()
done = False
prev_pose = env.arm_client.EE_pose_arr
ts = 0

obss = []
ee_vels = []
ee_d_poss = []
ee_poss = []

if sim and not env.polymetis_control:
    input("Start up the interactive marker controller then press enter to continue...")

# 1. switch robot to freedrive (only for real robot...for sim we don't use this)
if not sim or env.polymetis_control:
    input("Press enter to activate freedrive and start collection...")
    env.arm_client.activate_freedrive()

obss.append(obs)  # because, as we mention later, this obs should be paired with the next pose minus the initial pose

# 2. collect some amount of movement from the robot as it is pushed around (both velocities and positions)
#    remember, to match sim vs. real, where equilibrium pose will change in sim but not in real, you
#    CAN'T use the equilibrium pose for this
#    also do it at 10Hz, 5Hz, 20Hz, 3Hz, just to see differences in behaviour
done = False
while not done:
    obs, r, term, trunc, info = env.step(None, ignore_motion_act=True)
    done = term or trunc
    # dpos = info['recorded_motion']['pose'][:3] - prev_pose[:3]

    # commented out because it's in the client now, but i'm only partially sure that the code there is correct
    # this line gives us drot relative to the current EE pose...but we want it in the base frame..give options for both in data collection
    # drot_quat_tool = tf3d.quaternions.qmult(tf3d.quaternions.qinverse(prev_pose[3:]), info['recorded_motion']['pose'][3:])
    # drot_quat_inter = tf3d.quaternions.qmult(tf3d.quaternions.qinverse(prev_pose[3:]), drot_quat_tool)
    # drot_quat_base = tf3d.quaternions.qmult(drot_quat_inter, prev_pose[3:])
    # dpose = PoseTransformer(pose=np.concatenate([dpos, drot_quat_base]))

    # remember, the current observation should actually be paired with the _next_ pose minus the current pose
    # ee_d_poss.append(dpose.get_array_rvec())
    ee_poss.append(env.arm_client.EE_pose.get_array_euler())

    ee_d_poss.append(info['recorded_motion']['dpose_rvec'])

    obss.append(obs)
    prev_pose = info['recorded_motion']['pose']
    # ee_vels.append(info['recorded_motion']['vel'])
    if ts % 10 == 0:
        print(f"Recording: timestep {ts} of {env._max_episode_steps}")
    ts += 1

# now, as expected, you have one more obs than number of dpos, because the final obs is just the final "next obs"

# 3. deactivate freedrive reset the robot back to its initial pose
if not sim or env.polymetis_control:
    input("Press enter to deactivate freedrive.")
    env.arm_client.deactivate_freedrive()
if sim and not env.polymetis_control:
    print("Remember to shut down the interactive marker controller before finishing reset!")
obs = env.reset()
done = False
ts = 0
replay_ee_poss = []

# 4. replay the actions through the environment with step commands
while not done:
    if replay_using_vel:
        obs, r, term, trunc, info = env.step(v_replay_multiplier * ee_vels[ts] / rate)
    else:
        obs, r, term, trunc, info = env.step(dp_replay_multiplier * ee_d_poss[ts])
    done = term or trunc
    replay_ee_poss.append(env.arm_client.EE_pose.get_array_euler())
    if ts % 10 == 0:
        print(f"Replaying: timestep {ts} of {env._max_episode_steps}")
    ts += 1

# 5. plot original and rerun trajectories to see differences
ee_poss = np.array(ee_poss)
replay_ee_poss = np.array(replay_ee_poss)

# specifically wrap the x axis since it starts at pi and jumps between pi and -pi
ee_poss[ee_poss[:, 3] < 0, 3] = ee_poss[ee_poss[:, 3] < 0, 3] + 2 * np.pi
replay_ee_poss[replay_ee_poss[:, 3] < 0, 3] = replay_ee_poss[replay_ee_poss[:, 3] < 0, 3] + 2 * np.pi

trans_fig = plt.figure()
ax = trans_fig.add_subplot(projection='3d')

for (poss, label) in zip([ee_poss, replay_ee_poss], ["Recorded", "Replayed"]):
    ax.plot(poss[:, 0], poss[:, 1], poss[:, 2], label=label)
ax.set_title("Translations")
ax.legend()

rot_fig = plt.figure()
for (poss, label) in zip([ee_poss, replay_ee_poss], ["Recorded", "Replayed"]):
    for i in range(3, 6):
        plt.plot(poss[:, i], label=f"{label}_{i}")
plt.title("Rotations")
plt.legend()


# ax = rot_fig.add_subplot(projection='3d')

# for (poss, label) in zip([ee_poss, replay_ee_poss], ["Recorded", "Replayed"]):
#     ax.plot(poss[:, 3], poss[:, 4], poss[:, 5], label=label)
# ax.set_title("Rotations")
# ax.legend()

import ipdb; ipdb.set_trace()

plt.show()


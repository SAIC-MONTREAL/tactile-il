import copy
import time
import numpy as np
from panda_polymetis.control.panda_client import PandaClient
from panda_polymetis.utils.rate import Rate
from transform_utils.pose_transforms import PoseTransformer, matrix2pose

control_hz = 10
vel_lim = .1  # m/s
rvel_lim = 1.0  # rad/s
pc = PandaClient(ee_config_json="../conf/franka-desk/franka-gripper-and-blue-realsense.json", sim=True,
                 delta_pos_limit=vel_lim / control_hz, delta_rot_limit=rvel_lim / control_hz )
rate = Rate(control_hz)


# for testing state
# for i in range(20):
#     state = pc.get_and_update_state()
#     print(state["EE_pose_arr"])
#     rate.sleep()


# for testing controller
# pc.toggle_freedrive()
# print("freedrive activated!")
# time.sleep(5)
# pc.toggle_freedrive()


# for testing state time -- in sim, appears to be 1-2ms on my machine
# for i in range(20):
#     start_read = time.time()
#     state = pc.get_and_update_state()
#     print(f"state read time: {time.time() - start_read}")
#     rate.sleep()


# test move command
# state = pc.get_and_update_state()
# print(f"pose before move: {state['EE_pose']}")
# new_pose = copy.deepcopy(state["EE_pose"])

# # trans
# # new_pose.pose.position.z -= .2

# # rot
# rot_diff = PoseTransformer([0, 0, 0, 0.0, 0.0, -.5], rotation_representation='rvec')
# new_pose = matrix2pose(new_pose.get_matrix() @ rot_diff.get_matrix())

# pc.move_EE_to(new_pose)

# state = pc.get_and_update_state()
# print(f'pose after move: {state["EE_pose"]}')


# test shift EE command + motion recordings
state = pc.get_and_update_state()
print(f"pos before move: {state['EE_pose'].get_pos()}")
print(f"rot before move: {state['EE_pose'].get_euler()}")

translation = [0., 0., 0.02]
rotation = [0.0, 0.0, 0.0]
recorded_dposes = []
rot_base_frame = True

pc.start_controller()
pc.reset_pose_delta_prev_pose(pc.get_and_update_ee_pose()[0])
rate.sleep()  # initial sleep, since first will be extremely fast...TODO should be done during reset
for i in range(10):
    pc.shift_EE_by(translation, rotation, rotation_rep='rvec', rot_base_frame=True)
    # pc.shift_EE_by(translation, rotation, rotation_rep='rvec', rot_base_frame=False)
    rate.sleep()
    recorded_dposes.append(pc.get_motion_recordings(rot_base_frame=True)['dpose_rvec'])

time.sleep(1)
state = pc.get_and_update_state()
print(f"pos after move: {state['EE_pose'].get_pos()}")
print(f"rot after move: {state['EE_pose'].get_euler()}")

np.set_printoptions(suppress=True, precision=4)
recorded_dposes = np.array(recorded_dposes)
print(recorded_dposes)

pc.stop_controller()
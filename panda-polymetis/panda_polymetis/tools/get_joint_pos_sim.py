import os
# from polymetis import RobotInterface
import numpy as np
import panda_polymetis
from panda_polymetis.control.panda_client import PandaClient


pp_dir = os.path.dirname(panda_polymetis.__file__)
json_file = os.path.join(pp_dir, 'conf/franka-desk/franka-gripper-and-blue-realsense.json')
pc = PandaClient(server_ip='localhost', sim=True, ee_config_json=json_file)
state = pc.get_and_update_state()
num_dec = 7

base_tool_tf = state['EE_pose'].get_array_euler(axes='sxyz')
bttf_beginning = ', '.join([f"{e:.{num_dec}}" for e in base_tool_tf[:-1]])
print(f"reset_base_tool_tf: [{bttf_beginning}, {base_tool_tf[-1]:.{num_dec}}]")

joint_pos = np.array(state['joint_position'])
jp_beginning = ', '.join([f"{e:.{num_dec}}" for e in joint_pos[:-1]])
print(f"reset_joint_pos: [{jp_beginning}, {joint_pos[-1]:.{num_dec}}]")
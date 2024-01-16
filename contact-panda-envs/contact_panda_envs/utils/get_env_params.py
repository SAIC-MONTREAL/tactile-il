import time
import numpy as np

import rospy
import transformations as tf_trans
from control.panda_client import PandaClient


def get_current_pose_joints():
    """ Get the current 6-dof pose and joint pos to use as reset params in a new env. """

    pc = PandaClient()
    time.sleep(1)

    base_frame_pose = pc.EE_pose.get_array_euler(axes='sxyz')
    joint_pos = pc._q

    print(
        f"reset_base_tool_tf: {base_frame_pose}\n"\
        f"reset_joint_pos: {joint_pos}"
    )


if __name__ == "__main__":
    rospy.init_node("get_env_params")
    get_current_pose_joints()
import numpy as np

from transform_utils.pose_transforms import (
    transform_local_body,
    PoseTransformer
)

def add_pose_noise(pose, var_t=.005, var_angle=0.5, var_axis=.005):
    """
    Add some random translational N(mu_t, var_t) and rotational N(mu_r, var_r) 
        Gaussian noise to a pose.
    
    Args:
        pose (PoseTransformer):
        var_t (float): Variance [m] of translation noise
        var_angle (float): Variance [deg] of rotational noise
        var_axis (float): Variance [m] of axis end point noise

    Returns:
        noisy_pose (PoseTransformer): 
    """
    noisy_pose = PoseTransformer()

    # Add noise to translation
    pose_axis = pose.get_array_axisa()
    pose_axis[:3] = pose_axis[:3] + np.random.normal(0, var_t, size=3)

    # Add noise to rotation
    pose_axis[3] = pose_axis[3] + np.random.normal(0, np.radians(var_angle), size=1)
    pose_axis[4:] = pose_axis[4:] + np.random.normal(0, var_axis, size=3)

    # Set new pose values
    noisy_pose.set_array_axisa(pose_axis)        

    return noisy_pose

def generate_pregrasp(pose, distance=0.10, frame="opencv"):
    """
    Given a grasp pose generate a pre-grasp pose.

    Args:
        pose (PoseStamped or PoseTransformer): The grasp pose.
        distance (float): The distance to add to the grasp 
            pose in order to get the pregrasp pose.
        frame (string): The convention of the grasp 
            pose's frame
    
    Returns:
        pregrasp_pose (PoseStamped or PoseTransformer): The pregrasp pose.
    """
    
    #XXX: For now we assume that the grasp frame 
    # is defined in OpenCV convention
    if frame != "opencv":
        raise NotImplementedError
    
    # Shift pose by some distance in the negative z direction
    delta_pose = PoseTransformer(
        [0,0,-distance,1,0,0,0]
    )
    pregrasp_pose = transform_local_body(
        pose, delta_pose
    )

    return pregrasp_pose

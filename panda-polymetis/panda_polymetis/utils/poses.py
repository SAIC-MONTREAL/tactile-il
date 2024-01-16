import numpy as np

import transformations

from transform_utils.pose_transforms import (
    pose2matrix,
    PoseTransformer,
    matrix2pose,
)

def geodesic_error(pose1, pose2):
    """
    Computes the error in global coordinates between two transforms.

    Args:
        pose1: current transform
        pose2: goal transform

    Returns:
        A 4-vector of [dx, dy, dz, solid angle]
    """
    t1 = pose2matrix(pose1)
    t2 = pose2matrix(pose2)

    trel = np.dot(np.linalg.inv(t1), t2)
    trans = np.dot(t1[0:3, 0:3], trel[0:3, 3])
    omega, direction, point = transformations.rotation_from_matrix(trel)
    angle = np.linalg.norm(omega)
    return np.hstack((trans, angle))

def interpolate_pose(pose_initial, pose_final, N=100, V=0.05):
    """Converts an initial and final into a smooth trajectory of speed V"""

    assert N >= 2, 'Interpolation has to be of length at least N >= 2'

    dist = np.linalg.norm(pose_final.get_pos() - pose_initial.get_pos())
    if dist == 0:
        dt = 1/120.0
    else:
        dt = (dist / V) / N
    poses = []
    pose_initial_list = pose_initial.get_array_quat()
    pose_final_list = pose_final.get_array_quat()

    trans_initial = pose_initial_list[0:3]
    quat_initial = pose_initial_list[3:7]
    trans_final = pose_final_list[0:3]
    quat_final = pose_final_list[3:7]

    trans_interp_total = [np.linspace(trans_initial[0], trans_final[0], num=N),
                          np.linspace(trans_initial[1], trans_final[1], num=N),
                          np.linspace(trans_initial[2], trans_final[2], num=N)]
    for ii in range(N):
        quat_interp = transformations.quaternion_slerp(
            quat_initial,
            quat_final,
            (ii / (N-1))
        )
        pose_interp = PoseTransformer([
            trans_interp_total[0][ii],
            trans_interp_total[1][ii],
            trans_interp_total[2][ii],
            quat_interp[0],
            quat_interp[1],
            quat_interp[2],
            quat_interp[3]
        ])
        poses.append(pose_interp)
    return poses, dt

if __name__ == "__main__":

    pose1 = PoseTransformer([0,0,0,1,0,0,0])
    pose2 = PoseTransformer([0,0,10,1,0,0,0])

    out = interpolate_pose(pose1, pose2, 2)

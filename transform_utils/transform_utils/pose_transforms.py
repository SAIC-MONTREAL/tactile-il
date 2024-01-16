import numpy as np
import quaternion

from copy import deepcopy

from geometry_msgs.msg import PoseStamped, TransformStamped

#TODO: Move from using tf transformations to quaternion
from transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_from_matrix,
    quaternion_about_axis,
    translation_from_matrix,
    quaternion_matrix,
    quaternion_inverse,
    unit_vector
)
class PoseTransformer(PoseStamped):
    def __init__(self, pose=None, rotation_representation="quat", axes='rxyz'):
        """
        Super class of PoseStamped, adding common pose type transformations and operations.
        Can be initialized either with an 7d array type pose [x,y,z,w,x,y,z] , a PoseStamped, or empty as PoseStamped types
        Parameters
        ---
        pose=(nd.array, PoseStamped, None)
            Initialize body pose using preferred type
        """
        super().__init__()
        if pose is None:
            pose = [0, 0, 0, 1, 0, 0, 0]

        if isinstance(pose, PoseStamped):
            self.pose = pose.pose
        elif isinstance(pose, list) or isinstance(pose, np.ndarray) or isinstance(pose, tuple):
            self.set_array(
                pose,
                rotation_representation,
                axes=axes  # only used for euler
            )

    def get_array(self, rotation_representation="quat", axes='rxyz'):
        if rotation_representation == "euler":
            return self.get_array_euler(axes=axes)
        elif rotation_representation == "quat":
            return self.get_array_quat()
        elif rotation_representation == "axisa":
            return self.get_array_axisa()
        elif rotation_representation == "rvec":
            return self.get_array_rvec()

    def get_array_euler(self, axes='rxyz'):
        """
        Get 6d array of pose with a euler
            representation of rotation
        """
        return np.concatenate([self.get_pos(), self.get_euler(axes=axes)])

    def get_array_quat(self, order="wxyz", normalize=True):
        """
        Get 7d array of pose with a quaternion
            representation of rotation
        """
        return np.concatenate([self.get_pos(), self.get_quat(order=order, normalize=normalize)])

    def get_array_axisa(self):
        """
        Get 7d array of pose with a axis angle
            representation of rotation
        """
        return np.concatenate([self.get_pos(), self.get_axisa()])

    def get_array_rvec(self):
        """
        Get 6d array of pose with a rotation vector
            representation of rotation
        """
        return np.concatenate([self.get_pos(), self.get_rvec()])

    def get_euler(self, axes='rxyz'):
        """
        Get 3d array of euler orientation ex, ey, ez
        """
        return np.array(euler_from_quaternion(self.get_quat(), axes))

    def get_quat(self, order='wxyz', normalize=True):
        """
        Get quaternion [w, x, y, z] as numpy array
        """
        if normalize:
            return unit_vector(get_quat(self, order))
        else:
            return get_quat(self, order)

    def get_quat_inverse(self, order='wxyz', normalize=True):
        """
        Get quaternion inverse [w, x, y, z] as numpy array
        """
        if normalize:
            return unit_vector(get_quat_inverse(self, order))
        else:
            return get_quat_inverse(self, order)

    def get_axisa(self):
        """
        Get axis angle [theta, x, y, z] as numpy array
        """
        axis, angle = to_axis_angle(
            np.quaternion(*self.get_quat())
        )
        angle = wrap_to_pi(angle)
        return np.array([angle, *axis])

    def get_rvec(self):
        """
        Get rotation vector [theta * x, theta * y, theta * z] as numpy array
        """
        axis, angle = to_axis_angle(
            np.quaternion(*self.get_quat())
        )
        angle = wrap_to_pi(angle)
        return angle * axis

    def get_matrix(self):
        """
        Get 4x4 transformation matrix of pose
        """
        return pose2matrix(self)

    def get_pos(self):
        """
        Get [x, y, z] coordinates as numpy array
        """
        return get_pos(self)

    def get_rot(self):
        """
        Get rotation matrix (DCM) as 3x3 matrix
        """
        matrix = self.get_matrix()
        return matrix[0:3,0:3]

    def set_array(self, pose, rotation_representation="quat", axes='rxyz'):
        pose = np.array(pose).astype('float64')
        if rotation_representation == "euler":
            self.set_array_euler(pose, axes=axes)
        elif rotation_representation == "quat":
            self.set_array_quat(pose)
        elif rotation_representation == "axisa":
            self.set_array_axisa(pose)
        elif rotation_representation == "rvec":
            self.set_array_rvec(pose)

    def set_array_euler(self, pose, axes='rxyz'):
        self.set_pos(pose[:3])
        self.set_euler(pose[3:], axes=axes)

    def set_array_quat(self, pose):
        self.set_pos(pose[:3])
        self.set_quat(pose[3:])

    def set_array_axisa(self, pose):
        self.set_pos(pose[:3])
        self.set_axisa(pose[3:])

    def set_array_rvec(self, pose):
        self.set_pos(pose[:3])
        self.set_rvec(pose[3:])

    def set_pos(self, pos):
        """
        Set [x,y,z] position of pose object
        Parameters
        ---
        pos (nd.array)
            [x,y,z] position of pose object
        """
        pos = np.array(pos).astype('float64')
        self.pose = set_pos(self, pos).pose

    def set_euler(self, euler, axes='rxyz'):
        """
        Set [ex, ey, ez] euler orientation of pose object
        Parameters
        ---
        euler (nd.array)
            [ex, ey, ez] orientation of object
        """
        euler = np.array(euler).astype('float64')
        self.set_quat(quaternion_from_euler(euler[0], euler[1], euler[2], axes=axes)) #sequential euler sequence in body frame

    def set_quat(self, quat, order='wxyz'):
        """
        Set [w,x,y,z] or [x,y,z,w] quaternion of pose object
        Parameters
        ---
        quat (nd.array)
            [w,x,y,z] quat of pose object
        """
        quat = np.array(quat).astype('float64')
        self.pose = set_quat(self, quat, order).pose

    def set_axisa(self, axisa):
        """
        Set [theta,x,y,z] axis angle of pose object
        Parameters
        ---
        axisa (nd.array)
            [theta,x,y,z] axis angle of pose object
        """
        axisa = np.array(axisa).astype('float64')
        self.set_quat(quaternion_about_axis(axisa[0], axisa[1:]))

    def set_rvec(self, rvec):
        """
        Set [theta * x, theta * y, theta * z] rotation vector of pose object
        Parameters
        ---
        rvec (nd.array)
            [theta * x, theta * y, theta * z] rotation vector of pose object
        """
        rvec = np.array(rvec).astype('float64')
        theta = np.linalg.norm(rvec)
        if theta >= 1e-8:  # avoid numerical errors
            rvec = rvec / theta
        self.set_quat(quaternion_about_axis(theta, rvec))

    def as_transform(self, child_id=""):
        """
        Get a ROS transform object with the same data.
        """
        tf = TransformStamped()
        tf.child_frame_id = child_id
        tf.header = self.header
        tf.transform.translation.x = self.pose.position.x
        tf.transform.translation.y = self.pose.position.y
        tf.transform.translation.z = self.pose.position.z
        tf.transform.rotation.w = self.pose.orientation.w
        tf.transform.rotation.x = self.pose.orientation.x
        tf.transform.rotation.y = self.pose.orientation.y
        tf.transform.rotation.z = self.pose.orientation.z

        return tf

def convert_reference_frame(pose_source, pose_frame_target, pose_frame_source, frame_id=''):
    """
    Converts pose source from source frame to target frame
    Parameters
    ---
    pose_source (PoseStamped)
        pose of object relative to pose_frame_source (object pose)
    pose_frame_target (PoseStamped)
        pose of target reference frame (target frame)
    pose_frame_source (PoseStamped)
        pose of source reference frame (base frame)
    frame_id='' (str)
        Optionally give a name to the pose to describe it's parent frame
    """
    T_pose_source = pose2matrix(pose_source)
    pose_transform_target2source = get_transform(pose_frame_source, pose_frame_target)
    T_pose_transform_target2source = pose2matrix(pose_transform_target2source)
    T_pose_target = np.matmul(T_pose_transform_target2source, T_pose_source)
    pose_target = matrix2pose(T_pose_target)
    return PoseTransformer(pose_target)

def transform_local_body(pose_source_world, pose_transform_target_body):
    """
    Apply transformation to a pose in the object's local reference frame
    Parameters
    ---
    pose_source_world (PoseStamped)
        Body reference frame wih respect to the world frame
    pose_transform_target_body (PoseStamped)
        Transformation pose wih respect to the body frame. Note this is often a 4x4 matrix, which should be converted to PoseStamped using PoseTransformer or matrix2posematrix2pose
    """
    #convert source to target frame
    pose_source_body = convert_reference_frame(pose_source_world,
                                                 pose_source_world,
                                                 PoseStamped(),
                                                 frame_id="body_frame")
    #perform transformation in body frame
    pose_source_rotated_body = transform_pose(pose_source_body,
                                              pose_transform_target_body)
    # rotate back
    pose_source_rotated_world = convert_reference_frame(pose_source_rotated_body,
                                                         PoseStamped(),
                                                         pose_source_world,
                                                         frame_id="EE")
    return PoseTransformer(pose_source_rotated_world)

def transform_pose(pose_source, pose_transform):
    """
    Apply transformation to a pose in the world reference frame
    Parameters
    ---
    pose_source(PoseStamped)
        Body reference frame with respect to the world frame
    pose_transform(PoseStamped)
        Transformation pose with respect to the world frame. Note this is often a 4x4 matrix, which should be converted to PoseStamped using PoseTransformer or matrix2posematrix2pose
    """
    T_pose_source = pose2matrix(pose_source)
    T_transform_source = pose2matrix(pose_transform)
    T_pose_final_source = np.matmul(T_transform_source, T_pose_source)
    pose_final_source = matrix2pose(T_pose_final_source)
    return PoseTransformer(pose_final_source)

def get_pos(pose):
    return np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])

def get_quat(pose, order='wxyz'):
        if order == 'wxyz':
            return np.array([pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z])
        elif order == 'xyzw':
            return np.array([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w])
        else:
            raise NotImplementedError(f"{order} is not a valid convention for quaternions")

def get_quat_inverse(pose, order='wxyz'):
        q_inv = quaternion_inverse(get_quat(pose, "wxyz"))
        if order == 'wxyz':
            return q_inv
        elif order == 'xyzw':
            return np.array([q_inv[1], q_inv[2], q_inv[3], q_inv[0]])
        else:
            raise NotImplementedError(f"{order} is not a valid convention for quaternions")

def pose2array(msg, order='wxyz'):
    if order == 'wxyz':
        return [float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
                float(msg.pose.orientation.w),
                float(msg.pose.orientation.x),
                float(msg.pose.orientation.y),
                float(msg.pose.orientation.z),
                ]
    elif order == 'xyzw':
        return [float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
                float(msg.pose.orientation.x),
                float(msg.pose.orientation.y),
                float(msg.pose.orientation.z),
                float(msg.pose.orientation.w),
                ]
    else:
        raise NotImplementedError("Only implemented for wxyz or xyzw.")

def to_axis_angle(q):
    rot = quaternion.as_rotation_vector(q)
    angle = np.linalg.norm(rot, axis=-1)
    mask = angle >= 1e-15
    axis = np.zeros_like(rot)
    if angle >= 1e-8:
        axis = rot / angle
    return axis, angle

def pose2matrix(pose):
    """
    Convert PoseStamped to 4x4 transformation matrix
    Parameters
    ---
    pose (PoseStamped)
        Object pose
    """
    trans = get_pos(pose)
    quat = get_quat(pose)
    T = quaternion_matrix(quat)
    T[0:3,3] = trans
    return T

def matrix2pose(matrix, frame_id=''):
    """
    Convert 4x4 transformation matrix to PoseStamped
    Parameters
    ---
    matrix (nd.array)
        4x4 transformation matrix
    frame_id='' (str)
        Optionally give name to pose reference frame
    """
    trans = translation_from_matrix(matrix)
    quat = quaternion_from_matrix(matrix)
    pose = list(trans) + list(quat)
    return array2pose(pose, frame_id=frame_id)

def array2pose(pose, frame_id=''):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = pose[0]
    msg.pose.position.y = pose[1]
    msg.pose.position.z = pose[2]
    msg.pose.orientation.w = pose[3]
    msg.pose.orientation.x = pose[4]
    msg.pose.orientation.y = pose[5]
    msg.pose.orientation.z = pose[6]
    return msg

def point2pose(point, frame_id=''):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = point[0]
    msg.pose.position.y = point[1]
    msg.pose.position.z = point[2]
    msg.pose.orientation.w = 1
    msg.pose.orientation.x = 0
    msg.pose.orientation.y = 0
    msg.pose.orientation.z = 0
    return msg

def rot2quat(orient_mat_3x3):
    orient_mat_4x4 = [[orient_mat_3x3[0][0],orient_mat_3x3[0][1],orient_mat_3x3[0][2],0],
                       [orient_mat_3x3[1][0],orient_mat_3x3[1][1],orient_mat_3x3[1][2],0],
                       [orient_mat_3x3[2][0],orient_mat_3x3[2][1],orient_mat_3x3[2][2],0],
                       [0,0,0,1]]

    orient_mat_4x4 = np.array(orient_mat_4x4)
    quat = quaternion_from_matrix(orient_mat_4x4)
    return quat

def get_transform(pose_frame_target, pose_frame_source):
    """
    Find transform that transforms pose source to pose target
    :param pose_frame_target:
    :param pose_frame_source:
    :return:
    """
    #both poses must be expressed in same reference frame
    T_target_world = pose2matrix(pose_frame_target)
    T_source_world = pose2matrix(pose_frame_source)
    T_relative_world = np.matmul(T_target_world, np.linalg.inv(T_source_world))
    pose_relative_world = matrix2pose(T_relative_world)
    return pose_relative_world

def set_pos(pose, position):
    pose_new = deepcopy(pose)
    pose_new.pose.position.x = position[0]
    pose_new.pose.position.y = position[1]
    pose_new.pose.position.z = position[2]
    return pose_new

def set_quat(pose, quaternion, order='wxyz'):
    pose_new = deepcopy(pose)
    if order == 'wxyz':
        pose_new.pose.orientation.w = quaternion[0]
        pose_new.pose.orientation.x = quaternion[1]
        pose_new.pose.orientation.y = quaternion[2]
        pose_new.pose.orientation.z = quaternion[3]
    elif order == 'xyzw':
        pose_new.pose.orientation.w = quaternion[3]
        pose_new.pose.orientation.x = quaternion[0]
        pose_new.pose.orientation.y = quaternion[1]
        pose_new.pose.orientation.z = quaternion[2]
    else:
        raise NotImplementedError(f"{order} is not a valid convention for quaternions")
    return pose_new

def vec_from_pose(pose):
    quat = pose.pose.orientation
    R = quaternion_matrix([quat.w, quat.x, quat.y, quat.z])
    x_vec = R[0:3, 0]
    y_vec = R[0:3, 1]
    x_vec = R[0:3, 2]
    return x_vec, y_vec, x_vec

def transform_global_pos(pose, delta_pos=[0,0,0]):
    """
    Transforms a pose by small cartesian adjustements in the global frame
    """
    pose.pose.position.x += delta_pos[0]
    pose.pose.position.y += delta_pos[1]
    pose.pose.position.z += delta_pos[2]
    return pose

def transform_local_pos(pose, delta_pos=[0,0,0], frame_id='map'):
    """
    Transforms a pose by small cartesian adjustements in the local frame
    """
    pose_transform = init_pose_stamped()
    pose_transform.pose.position.x = delta_pos[0]
    pose_transform.pose.position.y = delta_pos[1]
    pose_transform.pose.position.z = delta_pos[2]
    return convert_reference_frame(pose_source=pose_transform,
                            pose_frame_target=init_pose_stamped(),
                            pose_frame_source=pose,
                            frame_id = frame_id)

def wrap_to_pi(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def vec_to_skew_symmetric(vec):
    return np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])

def plot_frames(pose_list=None, color_list=None):
    import open3d as o3d
    if color_list is not None:
        assert (len(color_list)==len(pose_list)), 'color list must have same lenght as pose_list'
    mesh_list = []
    for counter, pose in enumerate(pose_list):
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh = mesh.transform(pose.get_matrix())
        if color_list is not None:
            if color_list[counter] is not None:
                mesh.paint_uniform_color(color_list[counter])
        mesh_list.append(mesh)
    o3d.visualization.draw_geometries(mesh_list)

def test_frame_conversion():
    pose_object_world = PoseTransformer([0,0,3,1,0,0,0])
    pose_source_world = PoseTransformer([0,0,0,1,0,0,0])
    pose_target_world = PoseTransformer([0,0,2,1,0,0,0])

    #convert pose_object_world to target frame -> pose_object_target
    pose_object_target = convert_reference_frame(pose_source=pose_object_world,
                                         pose_frame_target=pose_target_world,
                                         pose_frame_source=pose_source_world)

    print (pose_object_target.get_array())
    plot_frames(pose_list=[pose_object_world, pose_target_world, pose_source_world],
        color_list=[[0,1,0], [1,0,0], [0,0,1]])

def test_body_transform():
    pose_object_world = PoseTransformer([0,0,0,1,0,0,0])
    pose_transform_body = PoseTransformer([0,0,0,1,0,0,0])
    pose_transform_body.set_euler([10 * np.pi/180, 0, 0])
    new_pose_body_world = transform_local_body(pose_object_world, pose_transform_body)

    print (new_pose_body_world.get_array())
    plot_frames(pose_list=[pose_object_world, new_pose_body_world],
        color_list=[None, None])

def test_rotations():
    p = PoseTransformer([0,0,0,1,2,2,3])
    print("Not normalized quaternion: ", p.get_quat(normalize=False), "norm: ", np.linalg.norm(p.get_quat(normalize=False)))
    print("Normalized quaternion: ", p.get_quat(normalize=True), "norm: ", np.linalg.norm(p.get_quat(normalize=True)))

    # From https://www.andre-gaschler.com/rotationconverter/,
    # the equivalent axis angle should be: { [ 0.4850713, 0.4850713, 0.7276069 ], 2.6657104 }
    # the equivalent rotation vector should be: [ 1.2930595, 1.2930595, 1.9395892 ]
    print("Axis angle representation: ", p.get_axisa())
    print("Rotation vector representation: ", p.get_rvec())

    print("=" * 10)
    print(f"Before axisa: {p.get_axisa()}")
    p.set_axisa(p.get_axisa())
    print(f"After axisa (should match): {p.get_axisa()}")

    print("=" * 10)
    print(f"Before quat: {p.get_quat()}")
    p.set_quat(p.get_quat())
    print(f"After quat (should match): {p.get_quat()}")

    print("=" * 10)
    print(f"Before rvec: {p.get_rvec()}")
    p.set_rvec(p.get_rvec())
    print(f"After rvec (should match): {p.get_rvec()}")

    print("=" * 10)
    print(f"Before euler: {p.get_euler()}")
    p.set_euler(p.get_euler())
    print(f"After euler (should match): {p.get_euler()}")

    print("=" * 10)
    print(f"Before axisa pose: {p.get_array_axisa()}")
    p.set_array_axisa(p.get_array_axisa())
    print(f"After axisa pose (should match): {p.get_array_axisa()}")

    print("=" * 10)
    print(f"Before quat pose: {p.get_array_quat()}")
    p.set_array_quat(p.get_array_quat())
    print(f"After quat pose (should match): {p.get_array_quat()}")

    print("=" * 10)
    print(f"Before rvec pose: {p.get_array_rvec()}")
    p.set_array_rvec(p.get_array_rvec())
    print(f"After rvec pose (should match): {p.get_array_rvec()}")

    print("=" * 10)
    print(f"Before euler pose: {p.get_array_euler()}")
    p.set_array_euler(p.get_array_euler())
    print(f"After euler pose (should match): {p.get_array_euler()}")

if __name__=="__main__":
    # test_frame_conversion()
    # test_body_transform()
    test_rotations()

import warnings
import numpy as np
import json
import torch
import time
import copy

from polymetis import RobotInterface
import torchcontrol as toco
import torchcontrol.transform as toco_tf

from transform_utils.pose_transforms import (
    PoseTransformer,
    pose2array,
    matrix2pose,
    transform_local_body,
    transform_pose,
    rot2quat,
    vec_to_skew_symmetric,
)
from geometry_msgs.msg import PoseStamped

# Differences from version in panda-bm_ros_packages (may not be exhaustive!! Just noting as I go)
# 1. Stiffness can be set in both at cartesian AND joint level
# 2. Defaults from polymetis seem to be quite good in reality (but pretty bad in sim), so just leaving as is
#    until someone needs them to be different from default
# 3. EE as set in desk is not by default exposed in polymetis, so it needs to be manually entered
#    - Since desk saves a json file, we can load the json file and use it directly!
# 4. Collision FT thresholds cannot be set on the fly. Defaults worked for me, but can be modified easily
#    during the launch of the server via hydra/config file. Unclear if this can be done on the client side,
#    so leaving for now.
#    - what i've added now is a conf directory where these values can be overridden at the server level when the server
#      is launched. i think this should be good enough for most use cases.
# 5. Global pos limits are implemented by polymetis, which is a better way to handle it since it operates at
#    1kHz. Modify via hydra/config file.
# 6. Removed delta_pos_limits being set individually per DOF, and instead use a single one that's the norm of all.
# 7. Removed EE velocities since they're unused.
# 8. Since there is no ROS/subscriber, to get any state, get_state is called, and the caller chooses what to do with
#    the returned values.
#    - for backwards compatibility, this also updates relevant class members such as EE_pose.
# 9. The estimated force torque at the stiffness frame is not included since it is not yet exposed in polymetis.
# 10. Error resetting is not exposed by polymetis (i think), so not included in here either.
# 11. Can't find a way to communicate from the server to the client that it is in sim vs. not, so user
#     must manually pass this at the time of creating this client.
# 12. Currently no built-in way to dynamically modify stiffness+damping as there is with dynamic reconfigure.
#     If someone needs this, it would pretty easy to implement with a gui, but leaving this alone for now.
# 13. All polymetis quats are xyzw, but this file will generally use wxyz in all other cases. Specifically, when
#     reading from polymetis, we get xyzw, but all held variables and returns are wxyz!

# configs for launching can be seen here https://github.com/facebookresearch/fairo/tree/main/polymetis/polymetis/conf


class PandaClient:
    """
    Client for interacting with Panda, using FAIR's polymetis as the main driver.
    """

    def __init__(self,
        server_ip="localhost",  # nuc IP, or localhost if running on same computer, which should only be done in sim
        delta_pos_limit=.1,
        delta_rot_limit=.5,
        home_joints=None,
        only_positive_ee_quat=False,
        ee_config_json=None,
        sim=False,
    ):
        print("Starting polymetis panda client.")
        self._server_ip = server_ip
        self._delta_pos_limit = delta_pos_limit
        self._delta_rot_limit = delta_rot_limit
        self._only_positive_ee_quat = only_positive_ee_quat

        # torch rotation tools (which is installed by polymetis, i think) are somewhat broken, so we're going to
        # avoid using them as much as possible!!
        self._ee_tf_mat = np.eye(4)

        # load custom ee setup
        if ee_config_json is None:
            warnings.warn("No ee_config_json file set. Save the desk ee config json if you want to run actions "
                          "in the frame defined in desk.", RuntimeWarning)
        else:
            with open(ee_config_json, "r") as f:
                ee_config = json.load(f)
            trans = np.array(ee_config['transformation'])

            self._ee_tf_mat[:, 0] = trans[:4]
            self._ee_tf_mat[:, 1] = trans[4:8]
            self._ee_tf_mat[:, 2] = trans[8:12]
            self._ee_tf_mat[:, 3] = trans[12:16]

        self._ee_tf_mat_inv = np.linalg.inv(self._ee_tf_mat)

        # initialize polymetis client
        # self.robot = RobotInterface(ip_address=self._server_ip, enforce_version=False)

        # 2 for time to go forces move commands (not shift commands) to go a bit slower
        self.robot = RobotInterface(ip_address=self._server_ip, time_to_go_default=2.0)
        self.home_joints = self.robot.get_joint_positions() if home_joints is None else \
            torch.tensor(home_joints, dtype=torch.float)
        self.robot.set_home_pose(self.home_joints)

        # state tracking
        self.joint_position = None
        self.EE_pose = None
        self.EE_pose_arr = None
        self._pose_delta_prev_pose = None
        self._pose_delta_prev_pose_arr = None
        self.store_ee_poses = False
        self.stored_ee_poses = []
        self.sim = sim
        self._freedrive_is_active = False
        self._poly_error_code = 0
        self._controller_running = False
        self._freedrive_is_active = False
        self._kq = copy.deepcopy(self.robot.Kq_default)
        self._kqd = copy.deepcopy(self.robot.Kqd_default)
        self._kx = copy.deepcopy(self.robot.Kx_default)
        self._kxd = copy.deepcopy(self.robot.Kxd_default)
        self.target_pose = None

        # reset stored/target poses
        self.reset()

    def move_to_joint_positions(self, positions: torch.Tensor, time_to_go: float = None):
        if self._controller_running:
            self.stop_controller()
            self._controller_running = False
        # self.robot.move_to_joint_positions(positions=positions, time_to_go=time_to_go)

        error = 10
        tries = 0
        while error > 0.1:
            if tries > 0:
                print(f"move_to_joint_positions unsuccessful, trying again...")
            self.robot.move_to_joint_positions(positions=positions, time_to_go=time_to_go)
            cur_joint_pos = self.robot.get_joint_positions()
            error = np.linalg.norm(cur_joint_pos - positions)
            tries += 1
            if tries >= 5:
                print(f"Controller error? Cur joint pos: {cur_joint_pos}, but should be {positions}")
                import ipdb; ipdb.set_trace()

    def move_EE_to(self, pose: PoseStamped, time_to_go=None):
        """ Blocking command to move robot to a new pose. Blocks until it succeeds. """
        if self._controller_running:
            self.stop_controller()
            self._controller_running = False  # polymetis automatically overrides our own controller with move_to_ee_pose

        # transform desired pose into flange frame used by polymetis
        posetf = PoseTransformer(pose=pose)
        pose_arr = pose2array(matrix2pose(posetf.get_matrix() @ self._ee_tf_mat_inv), order='xyzw')

        self.robot.move_to_ee_pose(position=pose_arr[:3], orientation=pose_arr[3:], time_to_go=time_to_go)

    def shift_EE_by(self, translation, rotation=None, base_frame=True, rot_base_frame=False, rotation_rep='rvec',
                    target_pose=True):
        """
        Shift desired EE pose. Uses controller instead of

        Args:
            translation (np.ndarray): 3d array representing cartesian position delta
            rotation (np.ndarray, optional): array representing rotation command, can be 3d rxyz eul, or 3d rot vec
            base_frame (bool, optional): send the translation delta commands relatively to the base_frame
            rot_base_frame (bool, optional): same as above, but for rotations
            rotation_rep (np.ndarray, optional): options: euler_axes, rvec, quat, axisa
            target_pose (bool, optional): Update an underlying target pose (carrot), rather than using the current pose.
        """
        if not self._controller_running:
            start = time.time()
            self.start_controller()
            warnings.warn(f"Call start_controller before calling shift_EE_by..caused delay of {time.time() - start}s")

        shift_start = time.time()

        translation, rotation = self.apply_delta_safety_bounds(translation, rotation)

        # set current pose, target (carrot) or true current
        if target_pose:
            cur_posetf = self.target_pose
        else:
            cur_posetf, _ = self.get_and_update_ee_pose()

        # set shift pose
        if rotation is None:
            shift_posetf = PoseTransformer(pose=[*translation, 1., 0., 0., 0.])
        else:
            pose = [*translation, *rotation]
            if rotation_rep.startswith('e'):
                axes = rotation_rep.split('_')[-1]
                shift_posetf = PoseTransformer(pose=pose, rotation_representation='euler', axes=axes)
            else:
                shift_posetf = PoseTransformer(pose=pose, rotation_representation=rotation_rep)

        # transform cur pose based on shift and choice of base frames
        cur_posetf_mat = cur_posetf.get_matrix()
        shift_posetf_mat = shift_posetf.get_matrix()

        if base_frame and not rot_base_frame:
            des_pose_mat = np.eye(4)
            des_pose_mat[:3, :3] = cur_posetf_mat[:3, :3] @ shift_posetf_mat[:3, :3]
            des_pose_mat[:3, 3] = cur_posetf.get_pos() + shift_posetf.get_pos()
        elif not base_frame and not rot_base_frame:
            des_pose_mat = cur_posetf_mat @ shift_posetf_mat
        elif base_frame and rot_base_frame:
            des_pose_mat = np.eye(4)
            des_pose_mat[:3, 3] = cur_posetf.get_pos() + shift_posetf.get_pos()
            des_pose_mat[:3, :3] = shift_posetf_mat[:3, :3] @ cur_posetf_mat[:3, :3]
        else:
            raise NotImplementedError("Not implemented for base_frame False and rot_base_frame True")

        # Continue updating target pose and not actual pose
        if target_pose:
            self.target_pose = PoseTransformer(matrix2pose(des_pose_mat))

        # transform desired pose into flange frame used by polymetis
        poly_des_pose = torch.tensor(pose2array(matrix2pose(des_pose_mat @ self._ee_tf_mat_inv), order='xyzw'))

        self.robot.update_desired_ee_pose(position=poly_des_pose[:3], orientation=poly_des_pose[3:])

        # print(f"shift time: {time.time() - shift_start}")

    def get_motion_recordings(self, base_frame=True, rot_base_frame=False, update_ee_pose=True):

        # can optionally use existing ee pose if the calling loop has already updated it
        if update_ee_pose:
            cur_posetf, cur_posearr = self.get_and_update_ee_pose()
        else:
            cur_posetf, cur_posearr = self.EE_pose, self.EE_pose_arr
        ret_dict = dict()
        ret_dict['pose'] = cur_posearr
        ret_dict['dpose_rvec'] = self.get_pose_delta(
            cur_posetf, base_frame=base_frame, rot_base_frame=rot_base_frame).get_array_rvec()

        return ret_dict

    def get_pose_delta(self, cur_posetf: PoseTransformer, base_frame=False, rot_base_frame=False):
        if self._pose_delta_prev_pose is None:
            self._pose_delta_prev_pose = cur_posetf
        prev_posetf = self._pose_delta_prev_pose

        # just rearrange the equations from shift_EE_by -- des from there is cur here, cur there is prev here
        if base_frame and not rot_base_frame:
            shift_pose_mat = np.eye(4)
            shift_pose_mat[:3, :3] = prev_posetf.get_matrix()[:3, :3].T @ cur_posetf.get_matrix()[:3, :3]
            shift_pose_mat[:3, 3] = cur_posetf.get_pos() - prev_posetf.get_pos()
        elif not base_frame and not rot_base_frame:
            shift_pose_mat = np.linalg.inv(prev_posetf.get_matrix()) @ cur_posetf.get_matrix()
        elif base_frame and rot_base_frame:
            shift_pose_mat = np.eye(4)
            shift_pose_mat[:3, 3] = cur_posetf.get_pos() - prev_posetf.get_pos()
            shift_pose_mat[:3, :3] = cur_posetf.get_matrix()[:3, :3] @ prev_posetf.get_matrix()[:3, :3].T
        else:
            raise NotImplementedError("Not implemented for base_frame False and rot_base_frame True")

        pose_diff_pt = PoseTransformer(pose=matrix2pose(shift_pose_mat))

        self._pose_delta_prev_pose = cur_posetf

        return pose_diff_pt

    def apply_delta_safety_bounds(self, trans=None, rot=None):
        """ Assuming that delta_pose is 6-tuple, rot is 3-tuple of euler angles or rvec. """
        if trans is not None:
            norm_trans = np.linalg.norm(trans)
            if norm_trans > self._delta_pos_limit:
                print(f"Warning: commanded delta pos norm: {norm_trans}, being capped at: {self._delta_pos_limit}")
                trans = trans / norm_trans * self._delta_pos_limit

        if rot is not None:
            norm_rot = np.linalg.norm(rot)
            if norm_rot > self._delta_rot_limit:
                print(f"Warning: commanded delta rot norm: {norm_rot}, being capped at: {self._delta_rot_limit}")
                rot = rot / norm_rot * self._delta_rot_limit

        return trans, rot

    def reset_target_pose(self, pose=None):
        self.target_pose = pose

    def reset_pose_delta_prev_pose(self, pose=None):
        self._pose_delta_prev_pose = pose

    def reset(self, target_pose=None, init_pose=None):
        self.reset_target_pose(target_pose)
        self.reset_pose_delta_prev_pose(init_pose)

    def get_and_update_ee_pose(self):
        ee_pos_raw, ee_quat_raw = self.robot.get_ee_pose()  # REMEMBER: polymetis quats are xyzw!
        if self._only_positive_ee_quat and ee_quat_raw[3] < 0:
            ee_quat_raw = -ee_quat_raw

        # transform pose to custom ee frame
        ee_pose_raw_posetf = self.pos_quat_to_posetf(ee_pos_raw, ee_quat_raw)
        ee_pose_T = ee_pose_raw_posetf.get_matrix() @ self._ee_tf_mat
        self.EE_pose = PoseTransformer(matrix2pose(ee_pose_T, frame_id="panda_link0"))
        self.EE_pose_arr = self.EE_pose.get_array_quat()

        return self.EE_pose, self.EE_pose_arr

    def get_and_update_state(self):
        state_time = time.time()
        self.EE_pose, self.EE_pose_arr = self.get_and_update_ee_pose()
        robot_state = self.robot.get_robot_state()  # lots more in here if you need it! torque info, etc.
        self.joint_position = self.robot.get_joint_positions()

        if self.store_ee_poses:
            self.stored_ee_poses.append((state_time, self.EE_pose))

        self._poly_error_code = robot_state.error_code

        # better to use this to ensure stale data isn't used, but member variables also updated for
        # backwards compatibility
        state = {
            "EE_pose": self.EE_pose,
            "EE_pose_arr": self.EE_pose_arr,
            "poly_error_code": self._poly_error_code,
            "joint_position": self.joint_position,
            "robot_state": robot_state,
        }

        return state

    def go_home(self):
        self._controller_running = False
        self.robot.go_home()

    def activate_freedrive(self, translation=True, rotation=True):
        # TODO lilac uses values of 1 for Kx for some reason?
        if not self._freedrive_is_active:
            kx_params = copy.deepcopy(self._kx)
            if translation:
                kx_params[:3] = torch.ones(3) * 1e-6
            if rotation:
                kx_params[3:] = torch.ones(3) * 1e-6

            self.start_controller(kq=[1e-6] * 7, kqd=[1e-6] * 7, kx=kx_params, kxd=kx_params)
            self._freedrive_is_active = True

    def deactivate_freedrive(self):
        if self._freedrive_is_active:
            self.stop_controller()
            self._freedrive_is_active = False

    def toggle_freedrive(self):
        """ Switch freedrive mode, regardless of current mode. """
        self.deactivate_freedrive() if self._freedrive_is_active else self.activate_freedrive()

    def start_controller(self, kq=None, kqd=None, kx=None, kxd=None):
        """ Can manually set stiffness + damping here, but might be better to set in the hydra yaml files
            when launching the server instead."""
        self._kq = self.robot.Kq_default if kq is None else kq
        self._kqd = self.robot.Kqd_default if kqd is None else kqd
        self._kx = self.robot.Kx_default if kx is None else kx
        self._kxd = self.robot.Kxd_default if kxd is None else kxd
        torch_policy = toco.policies.HybridJointImpedanceControl(
            joint_pos_current=self.robot.get_joint_positions(),
            Kq=self._kq, Kqd=self._kqd, Kx=self._kx, Kxd=self._kxd,
            robot_model=self.robot.robot_model,
            ignore_gravity=self.robot.use_grav_comp,
        )
        self.robot.send_torch_policy(torch_policy=torch_policy, blocking=False)
        while not self.robot.is_running_policy():
            print("Controller not started or stopped, attempting to restart..")
            self.robot.send_torch_policy(torch_policy=torch_policy, blocking=False)
        self._controller_running = True
        self.target_pose, _ = self.get_and_update_ee_pose()

    def pos_quat_to_posetf(self, pos, xyzw_quat):
        pose = [*pos, xyzw_quat[3], *xyzw_quat[:3]]
        posetf = PoseTransformer(pose=pose)
        return posetf

    def stop_controller(self):
        if self._controller_running:
            if self.robot.is_running_policy():  # will avoid error that breaks run
                self.robot.terminate_current_policy()
            self._controller_running = False

    def start_ee_storage(self):
        self.stored_ee_poses = []
        self.store_ee_poses = True

    def stop_ee_storage(self):
        self.store_ee_poses = False
        return self.stored_ee_poses

    def recover_from_errors(self):
        raise NotImplementedError("This doesn't appear to be exposed by polymetis.")

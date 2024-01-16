import time
import os
from threading import Thread, Lock
from queue import Queue
import pickle
from functools import partial
import copy

import numpy as np
from numpy.linalg import norm
import gym
from gym import spaces
from gym.utils import seeding
import yaml
import cv2

import contact_panda_envs
ROS_INSTALLED = contact_panda_envs.ROS_INSTALLED
if ROS_INSTALLED:
    import rospy
    from visualization_msgs.msg import Marker
    import tf2_ros

    from control.panda_client import PandaClient
    from control.srj_gripper_client import SRJGripperClient
    from control.panda_gripper_client import PandaGripperClient
    from perception.sts_client_direct import STSClientDirect
    from perception.realsense_client import RealSenseClient

from transform_utils.pose_transforms import PoseTransformer, matrix2pose

# non-ros stuff
import panda_polymetis
from panda_polymetis.utils.poses import geodesic_error
from panda_polymetis.utils.rate import Rate
# from panda_polymetis.control.panda_client import PandaClient as PolyPandaClient
# from panda_polymetis.control.panda_gripper_client import PandaGripperClient as PolyPandaGripperClient
# from panda_polymetis.control.panda_gripper_client import FakePandaGripperClient as FakePolyPandaGripperClient
from pysts.processing import STSProcessor
from pysts.sts import SimPySTS, PySTS
from contact_il.imitation.device_utils import CollectDevice
from realsense_wrapper import RealsenseAPI

XY_DEFAULTS = dict(valid_act_t_dof=[1, 1, 0], valid_act_r_dof=[0, 0, 0])
XZ_DEFAULTS = dict(valid_act_t_dof=[1, 0, 1], valid_act_r_dof=[0, 0, 0])
XY_RZ_DEFAULTS = dict(valid_act_t_dof=[1, 1, 0], valid_act_r_dof=[0, 0, 1])
XYZ_DEFAULTS = dict(valid_act_t_dof=[1, 1, 1], valid_act_r_dof=[0, 0, 0])
SIXDOF_DEFAULTS = dict(valid_act_t_dof=[1, 1, 1], valid_act_r_dof=[1, 1, 1])


class PandaGeneric(gym.Env):
    # implement as needed
    # CONTROL_TYPES = ('delta_tool, delta_joint, pos_tool, pos_joint, vel_tool, vel_joint')
    CONTROL_TYPES = ('delta_tool')
    STS_OPTIONS = (
        'raw_image',
        'marker',
        'marker_image',
        'marker_dots',
        'flow',
        'flow_image',
        'marker_flow',
        'marker_flow_image',
        'depth',
        'depth_image',
        'contact',
        'contact_image',
    )

    def __init__(
            self,
            robot_config_file=None,  # yaml config file, anything here is added or overwrites panda_generic.yaml
            config_override_dict=None,  # dict that overrides all other default yaml files
            reset_teleop_available=False,
            success_feedback_available=False,
            info_env_only=False,  # if True, env can only be used for getting space sizes and other info
            sim_override=False,  # needed to set sim for polymetis envs
            sts_source_vid=None,  # if you want to simulate the sts sensor
            no_sensors_env=False,  # allow env to be created that would normally require sts/kinect or other connected sensors
            client_override_dict={},
        ):
        super().__init__()

        if not info_env_only and ROS_INSTALLED:
            rospy.init_node('contact_panda_gym')
            try:
                self.sim = rospy.get_param('/simulation')
            except KeyError:
                self.sim = False
        else:
            self.sim = False

        if sim_override: self.sim = True
        self.no_sensors_env = no_sensors_env

        # load config
        base_config_file = os.path.join(os.path.dirname(__file__), 'panda_generic.yaml')
        with open(base_config_file) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        if self.sim:
            # load sim values to overwrite some non-sim defaults
            base_sim_config_file = os.path.join(os.path.dirname(__file__), 'panda_generic_sim.yaml')
            with open(base_sim_config_file) as f:
                sim_cfg = yaml.load(f, Loader=yaml.FullLoader)
            for k in sim_cfg:
                self.cfg[k] = sim_cfg[k]

        if robot_config_file is not None:
            with open(robot_config_file) as f:
                new_config = yaml.load(f, Loader=yaml.FullLoader)
            for k in new_config:
                self.cfg[k] = new_config[k]

        if config_override_dict is not None:
            for k in config_override_dict:
                self.cfg[k] = config_override_dict[k]

        # for storing config
        self.cfg['env_name'] = self.__class__.__name__

        # env setup
        assert self.cfg['control_type'] in PandaGeneric.CONTROL_TYPES, '%s is not in the valid control types %s' % \
                                                                      (self.cfg['control_type'],
                                                                       PandaGeneric.CONTROL_TYPES)

        # for convenience, set all config items as attributes...leaving out for now but might be worth adding
        # for k in self.cfg:
        #     setattr(self, k, self.cfg[k])

        # control
        self.control_type = self.cfg['control_type']
        self.rot_act_rep = self.cfg['rot_act_rep']
        self.control_freq = self.cfg['control_freq']
        self.target_pose_lpf = self.cfg['target_pose_lpf']
        self.max_grip_force = self.cfg['max_grip_force']
        self.default_grip_state = self.cfg['default_grip_state']

        # env
        self.polymetis_control = self.cfg['polymetis_control']
        self.img_in_state = self.cfg['img_in_state']
        self.depth_in_state = self.cfg['depth_in_state']
        self.success_causes_done = self.cfg['success_causes_done']
        self.failure_causes_done = self.cfg['failure_causes_done']
        self.done_success = False
        self.done_failure = False
        self.done_timeout = False
        self.dense_reward = self.cfg['dense_reward']
        self.num_objs = self.cfg['num_objs']
        self.init_gripper_random_lim = self.cfg['init_gripper_random_lim']
        self.auto_reset_traj_file = self.cfg['auto_reset_traj_file']
        self.auto_reset_traj_data = None
        if self.auto_reset_traj_file != "":
            cpe_dir = os.path.dirname(contact_panda_envs.__file__)
            with open(os.path.join(cpe_dir, "envs", self.auto_reset_traj_file), 'rb') as f:
                self.auto_reset_traj_data = pickle.load(f)

        if self.no_sensors_env:
            self.img_in_state = False
            self.depth_in_state = False

        self._max_episode_steps = int(self.cfg['max_real_time'] * self.control_freq)
        self.valid_act_t_dof = np.array(self.cfg['valid_act_t_dof'])
        self.valid_act_r_dof = np.array(self.cfg['valid_act_r_dof'])
        self.t_action_base_frame = self.cfg['t_action_base_frame']
        self.r_action_base_frame = self.cfg['r_action_base_frame']
        self.grip_in_action = self.cfg['grip_in_action']
        self.state_base_frame = self.cfg['state_base_frame']
        self.obs_is_dict = self.cfg['obs_is_dict']
        self._pose_relative_to_mat_inv = np.eye(4)

        # states
        self.pose_relative = self.cfg['pose_relative']
        self.num_prev_pose = self.cfg['num_prev_pose']
        self.num_prev_grip = self.cfg['num_prev_grip']
        self.state_data = self.cfg['state_data']

        # sts
        self.sts_pysts = self.cfg['sts_pysts']
        self.sts_data = self.cfg['sts_data']
        self.sts_images = self.cfg['sts_images']
        self.sts_num_data = self.cfg['sts_num_data']
        self.sts_namespaces = self.cfg['sts_namespaces']
        self.sts_config_dirs = self.cfg['sts_config_dirs']
        self.sts_initial_mode = self.cfg['sts_initial_mode']
        self.sts_no_switch_override = self.cfg['sts_no_switch_override']
        self._has_sts = len(self.sts_data) > 0 or len(self.sts_images) > 0 or len(self.sts_num_data) > 0
        if self.no_sensors_env:
            self._has_sts = False
        self.render_stss = dict.fromkeys(self.sts_namespaces)
        self.sts_switch_in_action = self.cfg['sts_switch_in_action']

        self.act_ind_dict = dict()
        if self.grip_in_action and self.sts_switch_in_action:
            self.act_ind_dict['grip'] = -2
            self.act_ind_dict['sts'] = -1
        elif self.grip_in_action and not self.sts_switch_in_action:
            self.act_ind_dict['grip'] = -1
        elif not self.grip_in_action and self.sts_switch_in_action:
            self.act_ind_dict['sts'] = -1


        # images
        self.img_resolution = self.cfg['img_resolution']
        self.img_namespaces = self.cfg['img_namespaces']
        self.image_center_crop = self.cfg['img_center_crop']
        self.image_crop = self.cfg['img_crop']
        self.max_depth = self.cfg['depth_max_dist']
        self.require_img_depth_registration = self.cfg['require_img_depth_registration']
        self.sensor = self.cfg['sensor']
        self.cam_forward_axis = self.cfg['cam_forward_axis']
        self.render_rgbds = dict.fromkeys(self.img_namespaces)
        self.first_render = True

        # safety
        self.pos_limits = self.cfg['global_pos_limits']
        self.max_trans_vel = self.cfg['max_trans_vel']
        self.max_rot_vel = self.cfg['max_rot_vel']

        if self.rot_act_rep == 'quat':
            self._quat_in_action = True
            raise NotImplementedError('Implement if needed')
        else:
            self._quat_in_action = False

        self.torque_thresholds = self.cfg.get('torque_thresholds', None)
        self.force_thresholds = self.cfg.get('force_thresholds', None)

        # gym setup
        self._num_trans = sum(self.valid_act_t_dof)
        if sum(self.valid_act_r_dof) > 0 and self._quat_in_action:
            self._num_rot = 4  # for quat
        else:
            self._num_rot = sum(self.valid_act_r_dof)
        self._valid_act_len = self._num_trans + self._num_rot + self.grip_in_action + self.sts_switch_in_action
        self.action_space = spaces.Box(-1, 1, (self._valid_act_len,), dtype=np.float32)

        if sum(self.valid_act_r_dof) == 1:
            rot_pose_size = 2  # cos + sin
        elif sum(self.valid_act_r_dof) > 1:
            rot_pose_size = 4  # quat
        else:
            rot_pose_size = 0

        pose_size = self._num_trans + rot_pose_size

        state_sizes = dict(
            pose=pose_size,
            prev_pose=pose_size * self.num_prev_pose,
            grip_pos=self.cfg['num_grip_fingers'],
            prev_grip_pos=self.cfg['num_grip_fingers'] * self.num_prev_grip,
            obs_pos=self._num_trans * self.num_objs,
            obj_rot=rot_pose_size * self.num_objs,
            obj_rot_z=2*self.num_objs,
            obj_rot_z_90=2*self.num_objs,
            obj_rot_z_180=2*self.num_objs,
            force_torque_internal=6,
            force_torque_sensor=6,
            timestep=1,
            joint_pos=7,
            raw_world_pose=7,
        )

        state_size_all = sum([state_sizes[k] for k in self.state_data])
        state_space = spaces.Box(-np.inf, np.inf, (state_size_all,), dtype=np.float32)
        if self.img_in_state or self.depth_in_state or self._has_sts or self.obs_is_dict:
            obs_space_dict = dict()

            if self.obs_is_dict:
                for k in self.state_data:
                    obs_space_dict[k] = spaces.Box(-np.inf, np.inf, (state_sizes[k],), dtype=np.float32)
            else:
                obs_space_dict['obs'] = state_space

            if self.img_in_state and self.depth_in_state:
                for ns in self.img_namespaces:
                    w, h = self.cfg['img_resolution']
                    obs_space_dict[self.key_from_sts(ns, 'rgbd')] = spaces.Box(0, 255, (h, w, 4), dtype=np.uint16)

            elif self.img_in_state:
                for ns in self.img_namespaces:
                    w, h = self.cfg['img_resolution']
                    obs_space_dict[self.key_from_sts(ns, 'rgb')] = spaces.Box(0, 255, (h, w, 3), dtype=np.uint8)

            elif self.depth_in_state:
                for ns in self.img_namespaces:
                    w, h = self.cfg['img_resolution']
                    obs_space_dict[self.key_from_sts(ns, 'depth')] = spaces.Box(0, 1, (h, w), dtype=np.float32)

            if len(self.sts_images) > 0:
                for ns in self.sts_namespaces:
                    for sts_im_type in self.sts_images:
                        w, h = self.cfg['sts_resolution']
                        obs_space_dict[self.key_from_sts(ns, sts_im_type)] = spaces.Box(
                            0, 255, (h, w, 3), dtype=np.uint8)

            if len(self.sts_data) > 0:
                for ns in self.sts_namespaces:
                    for sts_data_type in self.sts_data:
                        w, h = self.cfg['sts_resolution']
                        if self.sts_pysts:
                            nc = STSProcessor.DTYPE_NUM_CHANNELS[sts_data_type]
                        else:
                            nc = STSClientDirect.DTYPE_NUM_CHANNELS[sts_data_type]
                        obs_space_dict[self.key_from_sts(ns, sts_data_type)] = spaces.Box(
                            -np.inf, np.inf, (h, w, nc), dtype=np.float32)

            if len(self.sts_num_data) > 0:
                for ns in self.sts_namespaces:
                    for sts_ndata_type in self.sts_num_data:
                        obs_space_dict[self.key_from_sts(ns, sts_ndata_type)] = spaces.Box(
                            -np.inf, np.inf, (STSProcessor.DTYPE_SIZE[sts_ndata_type],), dtype=np.float32)


            self.observation_space = spaces.Dict(spaces=obs_space_dict)
        else:
            self.observation_space = state_space

        if info_env_only:
            return

        # marker in rviz for pos limits
        if ROS_INSTALLED:
            self.pub_pos_limits = rospy.Publisher('pos_limits_marker', Marker, queue_size=1)
            self.pos_limits_marker = Marker()
            self.pos_limits_marker.header.frame_id = self.state_base_frame
            self.pos_limits_marker.header.stamp = rospy.Time(0)
            self.pos_limits_marker.ns = 'pos_limits'
            self.pos_limits_marker.id = 0
            self.pos_limits_marker.type = Marker.CUBE
            self.pos_limits_marker.action = Marker.ADD
            self.pos_limits_marker.pose.position.x = (self.pos_limits[0][0] + self.pos_limits[0][1]) / 2
            self.pos_limits_marker.pose.position.y = (self.pos_limits[1][0] + self.pos_limits[1][1]) / 2
            self.pos_limits_marker.pose.position.z = (self.pos_limits[2][0] + self.pos_limits[2][1]) / 2
            self.pos_limits_marker.pose.orientation.w = 1.0  # xyz default to 0
            self.pos_limits_marker.scale.x = np.abs(self.pos_limits[0][0] - self.pos_limits[0][1])
            self.pos_limits_marker.scale.y = np.abs(self.pos_limits[1][0] - self.pos_limits[1][1])
            self.pos_limits_marker.scale.z = np.abs(self.pos_limits[2][0] - self.pos_limits[2][1])
            self.pos_limits_marker.color.g = 1.0
            self.pos_limits_marker.color.a = .3
            self.pos_limits_marker.lifetime = rospy.Duration()

        # using both a fixed sleep and variable sleep to match env with and without processing time on obs
        if ROS_INSTALLED:
            self.rate = rospy.Rate(self.control_freq)
        else:
            self.rate = Rate(self.control_freq)
        assert self.cfg['max_policy_time'] < 1 / self.control_freq, "max_policy_time %.3f is not less than period " \
                                    "defined by control_freq %.3f" % (self.cfg['max_policy_time'], 1 / self.control_freq)
        self._max_policy_time = self.cfg['max_policy_time']
        self._fixed_time_after_action = round(1 / self.control_freq - self._max_policy_time, 3)
        print(f"Fixed sleep between act and obs is {self._fixed_time_after_action}")

        # resetting thresholds and parameters
        self._reset_base_tool_tf_arr = np.array(self.cfg['reset_base_tool_tf'])
        self._reset_base_tool_pose = PoseTransformer(self._reset_base_tool_tf_arr, 'euler', axes='sxyz')
        self._reset_base_tool_pose.header.frame_id = self.state_base_frame
        self._reset_joint_pos = np.array(self.cfg['reset_joint_pos'])
        self._max_reset_trans = 0.75  # meters
        self._max_reset_rot = 2.5  # radians
        self._reset_teleop_available = reset_teleop_available
        if reset_teleop_available:
            self.reset_teleop_complete = False
        self._success_feedback_available = success_feedback_available

        # set up reset pose broadcasters for viewing in rviz
        if ROS_INSTALLED:
            self.main_reset_broadcaster = tf2_ros.StaticTransformBroadcaster()
            main_reset_tf = self._reset_base_tool_pose.as_transform(child_id="main_reset_pose")
            main_reset_tf.header.stamp = rospy.Time.now()
            self.main_reset_broadcaster.sendTransform(main_reset_tf)
            self.current_reset_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # gui
        self.gui_thread = None
        self.env_to_gui_q = None
        self.gui_to_env_q = None
        self.gui_timer = None
        self.gui_dict = None
        self.gui_lock = Lock()
        self.play_pause_env = True
        self.latest_processed_img = None

        # other attributes
        self.prev_action = None
        self.prev_pose = None
        self.prev_grip_pos = None
        self.ep_timesteps = 0
        self._img_depth_registered = None
        self._env_reset_complete = False

        # compute delta pos and rot limits based on desired max speeds and control freq
        dp_lim = self.max_trans_vel / self.control_freq
        dr_lim = self.max_rot_vel / self.control_freq

        # clients for hardware
        if self.polymetis_control:
            from panda_polymetis.control.panda_client import PandaClient as PolyPandaClient
            from panda_polymetis.control.panda_gripper_client import PandaGripperClient as PolyPandaGripperClient
            from panda_polymetis.control.panda_gripper_client import FakePandaGripperClient as FakePolyPandaGripperClient

            # TODO need a way to cleanly handle server-side only options, such as bounding box of workspace
            if 'arm_client' in client_override_dict:
                self.arm_client = client_override_dict['arm_client']
            else:
                self.arm_client = PolyPandaClient(
                    server_ip="localhost" if self.sim else os.environ["NUC_IP"],
                    delta_pos_limit=dp_lim,
                    delta_rot_limit=dr_lim,
                    home_joints=self._reset_joint_pos,
                    only_positive_ee_quat=True,
                    ee_config_json=os.path.join(os.path.dirname(panda_polymetis.__file__),
                                                'conf/franka-desk/', self.cfg['polymetis_ee_json_name']),
                    sim=self.sim,
                )

            if self.sim:
                client_class = FakePolyPandaGripperClient
            else:
                client_class = PolyPandaGripperClient

            if 'gripper_client' in client_override_dict:
                self.gripper_client = client_override_dict['gripper_client']
            else:
                self.gripper_client = client_class(
                    # server_ip="localhost",  # should always be run on host machine, not nuc!
                    server_ip=os.environ["GRIPPER_IP"],  # optionally running on NUC because not currently working on thinkstation
                    open_width=self.cfg['max_grip_width'],
                    grasp_force=self.cfg['max_grip_force'],
                    grip_speed=self.cfg['grip_speed'],
                    pinch_width=self.cfg['pinch_width'],
                )

        else:
            assert ROS_INSTALLED, "Must use polymetis for control if ROS not installed/available."
            self.arm_client = PandaClient(
                # global_pos_limits=self.pos_limits,
                home_joints=self._reset_joint_pos,
                include_moveit=True,
                sim=self.sim,
                set_stiffness_to_default=False,
                only_positive_ee_quat=True,
                norm_delta_pos_limit=dp_lim,
                norm_delta_rot_limit=dr_lim,
                include_ee_vel=True
            )
            self.arm_client.update_target_pose_lpf(self.cfg['target_pose_lpf'])
            self.arm_client.update_impedance_params(**self.cfg['stiffness_params'])
            self.arm_client.update_damping_params(**self.cfg['damping_params'])
            if not self.sim:
                print("Warning: set_full_collision_behavior doesn't work on the real robot. modify "\
                    "parameters file at franka_ros/blob/bm-current-track-franka/franka_control/config/franka_control_node.yaml "
                    "on nuc instead.")
            else:
                self.arm_client.set_full_collision_behavior(**self.cfg['collision_behavior'])
            self.arm_client.moveit_interface.set_max_velocity_scaling_factor(self.cfg['moveit_max_velocity_scaling_factor'])

            self.gripper_type = self.cfg['gripper_type']
            if self.gripper_type == 'srj':
                self.gripper_client = SRJGripperClient()
            elif self.gripper_type == 'panda':
                self.gripper_client = PandaGripperClient(
                    open_width=self.cfg['max_grip_width'],
                    grasp_force=self.cfg['max_grip_force'],
                    grip_speed=self.cfg['grip_speed'],
                    grasp_width=self.cfg['target_grip_width']
                    )

        if self.img_in_state or self.depth_in_state:
            # realsense_wrapper from polymetis uses a single class for all connected cameras
            if tuple(self.img_resolution) == (212, 120):
                self._cv2_resolution_modifier = partial(cv2.resize, dsize=(212, 120))
                raw_res = (424, 240)
            else:
                self._cv2_resolution_modifier = None
                raw_res = self.img_resolution
            self._img_vertical_flip = self.cfg['img_vertical_flip']
            self._img_horizontal_flip = self.cfg['img_horizontal_flip']

            if 'camera_client' in client_override_dict:
                self.camera_client = client_override_dict['camera_client']
            else:
                self.camera_client = RealsenseAPI(height=raw_res[1], width=raw_res[0])

        if self._has_sts:
            self.sts_clients = dict()
            assert len(self.sts_namespaces) == len(self.sts_config_dirs), "each sts must have its own namespace and config dir"
            for ns, cd in zip(self.sts_namespaces, self.sts_config_dirs):
                assert ns != "", "Must pass an sts_config_dir as well to use sts sensor!"
                common_args = dict(
                    config_dir=cd,
                    allow_both_modes=True,
                    resolution=self.cfg['sts_resolution'],
                    mode_switch_opts=dict(
                        initial_mode=self.sts_initial_mode,
                        tactile_mode_object_flow_channel=self.cfg['sts_tactile_mode_object_flow_channel'],
                        no_switch_override=self.sts_no_switch_override or self.sts_switch_in_action,
                    )
                )
                if self.sts_pysts:
                    # sts mode switch opts have been greatly modified, but keeping above for backwards compatibility
                    common_args['filter_markers'] = self.cfg['sts_filter_markers']
                    common_args['mode_switch_opts']['mode_switch_req_ts'] = self.cfg['sts_mode_switch_req_ts']
                    common_args['mode_switch_opts']['tac_thresh'] = self.cfg['sts_tac_thresh']
                    common_args['mode_switch_opts']['vis_thresh'] = self.cfg['sts_vis_thresh']
                    common_args['mode_switch_opts']['mode_switch_type'] = self.cfg['sts_mode_switch_type']

                    if 'sts_clients' in client_override_dict and ns in client_override_dict['sts_clients']:
                        self.sts_clients[ns] = client_override_dict['sts_clients'][ns]
                    else:
                        if self.sim:
                            self.sts_clients[ns] = STSProcessor(**common_args, sensor_sim_vid=sts_source_vid)
                        else:
                            self.sts_clients[ns] = STSProcessor(**common_args)
                else:
                    common_args['mode_switch_opts']['contact_mode_switch'] = self.cfg['sts_contact_mode_switch']
                    common_args['mode_switch_opts']['contact_mode_switch_req_ts'] = self.cfg['sts_contact_mode_switch_req_ts']
                    common_args['mode_switch_opts']['num_contact_tac_thresh'] = self.cfg['sts_num_contact_tac_thresh']
                    common_args['mode_switch_opts']['num_contact_vis_thresh'] = self.cfg['sts_num_contact_vis_thresh']
                    common_args['mode_switch_opts']['mean_depth_tac_thresh'] = self.cfg['sts_mean_depth_tac_thresh']
                    common_args['mode_switch_opts']['mean_depth_vis_thresh'] = self.cfg['sts_mean_depth_vis_thresh']
                    assert ROS_INSTALLED, "Must have ROS installed to use STSClientDirect"
                    self.sts_clients[ns] = STSClientDirect(**common_args, sts_namespace=ns)

        if ROS_INSTALLED:
            time.sleep(1)  # Give clients some time
            self.pub_pos_limits.publish(self.pos_limits_marker)

    def key_from_topic(self, topic):
        return topic.replace('/', '-')[1:]  # remove initial /

    def key_from_sts(self, ns, type):
        return f'{ns}_{type}'

    def seed(self, seed=None):
        """ Seed for random numbers, for e.g. resetting the environment """
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def _reset_helper(self):
        """ Called within reset, but to be overridden by child classes. This should somehow help the
        experimenter reset objects to a new random pose, possibly with instructions."""
        pass
        # print('Warning: _reset_helper should be implemented by child classes.')

    def reset(self, reset_joint_position=None, auto_reset=False, collect_device:CollectDevice=None,
              no_enter_for_reset=False):
        """ Reset the environment to the beginning of an episode.
        In sim, a user could theoretically reload or otherwise move objects arbitrarily, but since the
        primary use for this class is for the real robot, this method may require interaction with a person.

        If auto_reset is True, use a predefined auto reset traj and skip extra enter press."""

        if auto_reset:
            assert self.auto_reset_traj_data is not None, "make sure auto_reset_traj_file is defined in env yaml!"
            assert collect_device is not None, "Pass the CollectDevice to allow interrupting auto reset."
            print(f"Moving to predefined joint pos from {self.auto_reset_traj_file}")
            # self.arm_client.move_to_joint_positions(self.auto_reset_traj_data['goto_joint_pos'], time_to_go=2.0)
            self.arm_client.move_to_joint_positions(self.auto_reset_traj_data['goto_joint_pos'])
            print(f"Now moving through recorded auto_reset actions.")
            self.arm_client.start_controller()
            while not self.arm_client.robot.is_running_policy():
                print("Controller not started or stopped, attempting to restart..")
                self.arm_client.start_controller()
            collect_device.reset_all_button_states()
            for act in self.auto_reset_traj_data['recorded_actions']:
                self.step(act, motion_only=True)
                collect_device.update()
                if collect_device.start_stop:
                    input(f"User called start stop with collect device. press enter to activate freedrive.")
                    self.arm_client.activate_freedrive()
                    input(f"To continue reset, move arm + env to regular, non-auto_reset pos, and press enter.")
                    self.arm_client.deactivate_freedrive()
                    break

        # self.gui_lock.acquire()
        self._env_reset_complete = False

        # For convenience, this allows a user to reset the environment using their teleop and move the EE
        # into an ideal pose to then be driven back to the initial pose.
        # A calling program resetting the environment would then do the following:
        #
        # env.reset() --> given output from _reset_help, user teleops robot to reset env
        # while not teleop_button_to_indicate_reset_done:
        #     action = get_teleop_action(current_env_pose)
        #     env.step(action)
        # env.set_reset_teleop_complete()
        # env.reset()
        if self._reset_teleop_available and not self.reset_teleop_complete:
            print("reset called with teleop available. Reset objects to new poses given by helper, "
                  "then calling program calls set_reset_teleop_complete and reset again")
            self._reset_helper()
            # self.gui_lock.release()
            return

        if self.polymetis_control:
            self.arm_client.get_and_update_state()  # updates state variables

        # first do safety checks to make sure movement isn't too dramatic
        geodesic_arm_to_init = geodesic_error(self.arm_client.EE_pose, self._reset_base_tool_pose)
        dist_arm_to_init = norm(geodesic_arm_to_init[:3])
        rot_dist_arm_to_init = geodesic_arm_to_init[3]

        if dist_arm_to_init > self._max_reset_trans:
            raise RuntimeError("EE is %.3fm from initial pose. Must be within %.3fm to reset." %
                               (dist_arm_to_init, self._max_reset_trans))

        if rot_dist_arm_to_init > self._max_reset_rot:
            raise RuntimeError("EE is %.3frad from init pose. Must be within %.3frad." %
                               (rot_dist_arm_to_init, self._max_reset_rot))

        # complete the movement
        if not self._reset_teleop_available and not auto_reset and not no_enter_for_reset:
            # self.gui_lock.release()
            input("Ensure there is a linear, collision-free path between end-effector and initial pose,"
                  "then press Enter to continue...")
            # self.gui_lock.acquire()
        if auto_reset:
            print("reset called with auto_reset=True, reset to initial pose starting.")


        if reset_joint_position is not None:
            self.arm_client.move_to_joint_positions(reset_joint_position)

        else:
            self.arm_client.move_to_joint_positions(self._reset_joint_pos)
            # self.arm_client.go_home()  # don't use this since we may be running two envs simultaneously

            # randomize initial gripper pose
            # on panda, since we have 7 joints, to create nearly repeatable joint configs between resets,
            # we'll reset to a joint config using arm_client.go_home(), and then shift the ee.
            if self.init_gripper_random_lim != [0, 0, 0, 0, 0, 0]:
                # euler is a bad idea for large sample space, but for small should be fine
                ep_reset_shift_arr = self.np_random.uniform(
                    low=-np.array(self.init_gripper_random_lim) / 2,
                    high=np.array(self.init_gripper_random_lim) / 2, size=6)

                print(f"reset arr: {ep_reset_shift_arr}")

                reset_pose_shift_mat = PoseTransformer(
                    pose=ep_reset_shift_arr, rotation_representation="rvec").get_matrix()
                new_arm_pose = PoseTransformer(
                        pose=matrix2pose(self._reset_base_tool_pose.get_matrix() @ reset_pose_shift_mat))

                # self.arm_client.move_EE_to(pose=new_arm_pose, time_to_go=1.0)
                self.arm_client.move_EE_to(pose=new_arm_pose)

        self.arm_client.reset()

        if self.default_grip_state == 'o':
            if self.polymetis_control:
                grip_resp = self.gripper_client.open(blocking=True)
            else:
                grip_resp = self.gripper_client.open(wait_for_result_time=5.0)
        elif self.default_grip_state == 'c':
            if self.polymetis_control:
                grip_resp = self.gripper_client.close(blocking=True)
            else:
                grip_resp = self.gripper_client.close(wait_for_result_time=5.0)
        else:
            raise NotImplementedError()

        if not grip_resp:
            raise ValueError(f"Grip movement did not complete on reset! Is something blocking it? Response: {grip_resp}")

        print("Reset trajectories completed.")

        if self.polymetis_control:
            self.arm_client.start_controller()

        # called after moving ee to init pose and user can now manually set up env objects
        if not self._reset_teleop_available:
            self._reset_helper()

        # if force torque is in state, reset it back to zero
        if 'force_torque_sensor' in self.state_data:
            raise NotImplementedError("No bias subtracting implemented yet. Not sure if it's needed...")
            # self.pub_ft_zero.publish(True)
            # self.gui_lock.release()
            # time.sleep(0.5)  # zero reset node needs half a second of still robot to collect data to average
            # self.gui_lock.acquire()

        # other resets
        self.ep_timesteps = 0
        self.prev_action = None
        self.prev_pose = None
        self.prev_grip_pos = None
        self.done_success = False
        self.done_failure = False
        self.done_timeout = False

        # generate observation for return -- need published trajectories above to be completed
        # if self.pose_relative == 'reset':
        #     cur_pose_pt, _ = self.arm_client.get_and_update_ee_pose()
        #     self._pose_relative_to_mat_inv = np.linalg.inv(cur_pose_pt.get_matrix())
        # else:
        #     self._pose_relative_to_mat_inv = np.eye(4)

        # reset back to initial mode + other resets
        if self._has_sts:
            for ns in self.sts_namespaces:
                self.sts_clients[ns].reset()

        obs, _ = self._prepare_obs(reset_obs=True)

        if self._reset_teleop_available:
            self.reset_teleop_complete = False

        # self.gui_lock.release()
        if ROS_INSTALLED:
            self.pub_pos_limits.publish(self.pos_limits_marker)
        self._env_reset_complete = True

        sts_debug = False
        if sts_debug:
            import cv2
            cv2.imshow('test', obs['sts_raw_image'])
            cv2.waitKey(5)

        if self.polymetis_control:
            while not self.arm_client.robot.is_running_policy():
                print("Controller not started or stopped, attempting to restart..")
                self.arm_client.start_controller()

        return obs

    def _prepare_obs(self, reset_obs=False):
        """
        Order in returned state array: 'pose', 'prev_pose', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot',
        'force_torque_sensor', 'force_torque_internal', 'timestep'
        """
        return_obs = dict()
        return_arr = []

        if self.polymetis_control:
            self.arm_client.get_and_update_state()

        if reset_obs:
            self.arm_client.reset_pose_delta_prev_pose(self.arm_client.EE_pose)

            # create matrix to modify poses if we're using relative poses
            if self.pose_relative == 'reset':
                self._pose_relative_to_mat_inv = np.linalg.inv(self.arm_client.EE_pose.get_matrix())
            else:
                self._pose_relative_to_mat_inv = np.eye(4)

            self.arm_client.reset_target_pose(self.arm_client.EE_pose)  # ensures target and current match

        cur_pos_quat = self.arm_client.EE_pose_arr  # note...wxyz!!!
        cur_pose = self.arm_client.EE_pose

        # modify pose based on configured relative pose
        cur_pose = PoseTransformer(matrix2pose(self._pose_relative_to_mat_inv @ self.arm_client.EE_pose.get_matrix()))
        cur_pos_quat = cur_pose.get_array_quat()

        # fix pose to correspond to valid dofs
        if 'pose' in self.state_data or 'prev_pose' in self.state_data:
            valid_pos = cur_pos_quat[self.valid_act_t_dof.nonzero()]
            if sum(self.valid_act_r_dof) == 1:
                cur_eul_sxyz = cur_pose.get_euler(axes='sxyz')
                valid_rot = cur_eul_sxyz[self.valid_act_r_dof.nonzero()]  # just single angle now
                rot_state = np.array([np.cos(valid_rot), np.sin(valid_rot)])
                cur_pose = np.concatenate([valid_pos, rot_state])
            elif sum(self.valid_act_r_dof) > 1:
                cur_pose = np.concatenate([valid_pos, cur_pos_quat[3:]])
            else:
                cur_pose = np.array(valid_pos)

            if 'pose' in self.state_data:
                return_obs['pose'] = cur_pose
                return_arr.append(cur_pose)

            if 'prev_pose' in self.state_data:
                if self.prev_pose is None:
                    self.prev_pose = np.tile(cur_pose, (self.num_prev_pose + 1, 1))
                self.prev_pose = np.roll(self.prev_pose, 1, axis=0)
                self.prev_pose[0] = cur_pose
            if 'prev_pose' in self.state_data:
                return_obs['prev_pose'] = self.prev_pose[1:].flatten()
                return_arr.append(return_obs['prev_pose'])

        if 'raw_world_pose' in self.state_data:
            return_obs['raw_world_pose'] = self.arm_client.EE_pose.get_array_quat()
            return_arr.append(return_obs['raw_world_pose'])

        if 'grip_pos' in self.state_data or 'prev_grip_pos' in self.state_data:
            grip_pos_raw = self.gripper_client._pos

            # normalize to be in range of [-1, 1]
            max_pos = self.cfg['max_grip_width'] / 2
            grip_pos = (grip_pos_raw / max_pos - .5) * 2

            if 'grip_pos' in self.state_data:
                return_obs['grip_pos'] = grip_pos
                return_arr.append(grip_pos)
            if 'prev_grip_pos' in self.state_data:
                if self.prev_grip_pos is None:
                    self.prev_grip_pos = np.tile(grip_pos, (self.num_prev_grip + 1, 1))
                self.prev_grip_pos = np.roll(self.prev_grip_pos, 1, axis=0)
                self.prev_grip_pos[0] = grip_pos
                return_obs['prev_grip_pos'] = np.array(self.prev_grip_pos[1:]).flatten()
                return_arr.append(return_obs['prev_grip_pos'])

        if 'joint_pos' in self.state_data:
            return_obs['joint_pos'] = np.array(self.arm_client.joint_position)
            return_arr.append(return_obs['joint_pos'])

        if 'obj_pos' in self.state_data:
            raise NotImplementedError('Object positions not implemented, need to use ARtags or some other CV method.')
        if 'obj_rot' in self.state_data:
            raise NotImplementedError('Object positions not implemented, need to use ARtags or some other CV method.')
        if 'obj_rot_z' in self.state_data:
            raise NotImplementedError('Object positions not implemented, need to use ARtags or some other CV method.')

        if 'force_torque_sensor' in self.state_data:
            raise NotImplementedError("Not implemented for physical sensor yet.")

        if 'force_torque_internal' in self.state_data:
            if self.polymetis_control:
                raise NotImplementedError("Internal force-torque yet exposed in polymetis package!")
            ft_raw = self.arm_client._force_torque

            # normalize to be in range of [-1, 1] using max forces/torques for collisions
            # force_fixed = ft_raw[:3] / self.cfg['collision_behavior']['force_torque_max'][:3]
            # torque_fixed = ft_raw[3:] / self.cfg['collision_behavior']['force_torque_max'][3:]
            force_fixed = ft_raw[:3]
            torque_fixed = ft_raw[3:]

            return_obs['force_torque_internal'] = np.concatenate([force_fixed, torque_fixed])
            return_arr.append(return_obs['force_torque_internal'])

        if 'timestep' in self.state_data:
            # adjust range of timesteps to be between -1 and 1
            adj_timestep = (self.ep_timesteps / self._max_episode_steps - .5) * 2
            return_obs['timestep'] = np.array([adj_timestep])
            return_arr.append(adj_timestep)

        # add the desired pose to the state if we're controlling that way
        if self.arm_client.target_pose is not None:
            target_pt = PoseTransformer(
                matrix2pose(self._pose_relative_to_mat_inv @ self.arm_client.target_pose.get_matrix()))
            target_pos_quat = target_pt.get_array_quat()
            return_obs['target_pose'] = target_pos_quat

        return_arr = np.concatenate(return_arr)

        # img, depth, and sts -- convert return structure to dict if they are included
        if self.img_in_state or self.depth_in_state or self._has_sts:
            return_arr = dict(obs=return_arr)

        if self.img_in_state or self.depth_in_state:
            rgbd = self.camera_client.get_rgbd()
            if self.sim and rgbd.size == 0:
                rgbd = np.zeros([rgbd.shape[0] + len(self.img_namespaces), *rgbd.shape[1:]], dtype=rgbd.dtype)

            if self._cv2_resolution_modifier is not None:
                new_w, new_h = self._cv2_resolution_modifier.keywords['dsize']
                rgbd_fixed = np.zeros([rgbd.shape[0], new_h, new_w, rgbd.shape[-1]])
                for i in range(rgbd.shape[0]):
                    rgbd_fixed[i] = self._cv2_resolution_modifier(rgbd[i])
                rgbd = rgbd_fixed

            if self._img_vertical_flip:
                rgbd = np.flip(rgbd, axis=1)

            if self._img_horizontal_flip:
                rgbd = np.flip(rgbd, axis=2)

            # going to reverse since camera comes in as bgr
            rgbd_fixed_order = rgbd.copy()
            rgbd_fixed_order[:, :, :, 0] = rgbd[:, :, :, 2]
            rgbd_fixed_order[:, :, :, 2] = rgbd[:, :, :, 0]
            rgbd = rgbd_fixed_order

            for i, ns in enumerate(self.img_namespaces):
                # store separately to take up less room on disk
                if self.img_in_state:
                    return_arr[self.key_from_sts(ns, 'rgb')] = rgbd[i, :, :, :3].astype('uint8')
                    return_obs[self.key_from_sts(ns, 'rgb')] = rgbd[i, :, :, :3].astype('uint8')
                if self.depth_in_state:
                    return_arr[self.key_from_sts(ns, 'depth')] = rgbd[i, :, :, 3]
                    return_obs[self.key_from_sts(ns, 'depth')] = rgbd[i, :, :, 3]

                # return_arr[self.key_from_sts(ns, 'rgbd')] = rgbd[i]
                # return_obs[self.key_from_sts(ns, 'rgbd')] = rgbd[i]
                self.render_rgbds[ns] = rgbd[i]

        if self._has_sts:
            for ns in self.sts_namespaces:
                sts_dict = self.sts_clients[ns].get_processed_sts(
                    modes=set(self.sts_images + self.sts_data + self.sts_num_data))
                for k in sts_dict.keys():
                    return_arr[self.key_from_sts(ns, k)] = sts_dict[k]
                    return_obs[self.key_from_sts(ns, k)] = sts_dict[k]
                self.render_stss[ns] = sts_dict['raw_image']


        if self.obs_is_dict:
            return return_obs, return_obs
        else:
            return return_arr, return_obs

    def step(self, action, ignore_motion_act=False, motion_only=False):
        """ If action space requires a quat, the quat does not have to be entered normalized to be valid.

        Action should come in as (n,) shape array, where n includes number of translational DOF, rotational DOF,
        and +1 if gripper control is included. Gripper control is a float where anything below 0
        is considered open, and anything above 0 is considered close.

        STS switch is another +1 if sts_switch_in_action is true.
        """

        # gui handling
        if self.gui_thread is not None:
            # self.gui_lock.acquire()
            if not self.play_pause_env:
                # self.gui_lock.release()
                print("Env is paused, unpause using gui.")
                # self.gui_lock.acquire()
                while not self.play_pause_env:
                    # self.gui_lock.release()
                    rospy.sleep(.1)
                    # self.gui_lock.acquire()
            # self.gui_lock.release()

        # self.gui_lock.acquire()

        # arm command
        if not ignore_motion_act:
            assert len(action) == self._valid_act_len, 'action needs %d dimensions for this env, step called with %d' % \
                                         (self._valid_act_len, len(action))
            if self.control_type == 'delta_tool':
                delta_trans = np.array([0., 0., 0.])
                delta_trans[self.valid_act_t_dof.nonzero()[0]] = action[:self._num_trans]
                rod_delta_rot = np.array([0., 0., 0.])
                rod_delta_rot[self.valid_act_r_dof.nonzero()[0]] = action[self._num_trans:(self._num_trans + self._num_rot)]

                common_args = dict(
                    translation=delta_trans,
                    base_frame=self.t_action_base_frame=='b',
                    rot_base_frame=self.r_action_base_frame=='b',
                    target_pose=True,
                )
                if self.polymetis_control:
                    self.arm_client.shift_EE_by(**common_args,
                        rotation=rod_delta_rot,
                        rotation_rep='rvec',
                    )
                else:
                    self.arm_client.shift_EE_by(**common_args,
                        rotation=rod_delta_rot,
                        interpolate=False
                    )

        # grip command -- excluded from no_act_record_motion because user can still control this properly
        if self.grip_in_action:
            if action[self.act_ind_dict['grip']] < 0: self.gripper_client.open()
            else: self.gripper_client.close()

        if self.sts_switch_in_action:
            for ns in self.sts_namespaces:
                if action[self.act_ind_dict['sts']] < 0: self.sts_clients[ns].set_mode(self.sts_clients[ns]._initial_mode)
                else: self.sts_clients[ns].set_mode(self.sts_clients[ns]._alternate_mode)

        # fixed sleep for proper mdp
        # self.gui_lock.release()
        if ROS_INSTALLED:
            rospy.sleep(self._fixed_time_after_action)
        else:
            time.sleep(self._fixed_time_after_action)
        # self.gui_lock.acquire()

        if not motion_only:
            obs, full_obs_dict = self._prepare_obs()
        else:
            obs, full_obs_dict = None, None

        # motion recording for kinesthetic teaching. timed this, it's about 0.5ms
        if self.polymetis_control:
            recorded_motion = self.arm_client.get_motion_recordings(
                base_frame=self.t_action_base_frame=='b', rot_base_frame=self.r_action_base_frame=='b',
                update_ee_pose=False)
        else:
            recorded_motion = self.arm_client.get_motion_recordings(
                base_frame=self.t_action_base_frame=='b', rot_base_frame=self.r_action_base_frame=='b')

        # variable sleep to maintain control loop consistency
        # self.gui_lock.release()
        self.rate.sleep()
        # self.gui_lock.acquire()

        if not motion_only:
            rew = self.get_reward(obs, full_obs_dict, action)
        else:
            rew = 0

        self.ep_timesteps += 1

        truncated = self.get_truncated()
        terminated = self.get_terminated()

        info = dict(done_success=False, done_failure=False)
        if terminated:
            # get feedback on success only if a timeout ocurred
            if truncated:
                if self._success_feedback_available:
                    # self.gui_lock.release()
                    if self._reset_teleop_available:
                        print("Waiting for user feedback on success: press up success, down for fail. "
                                "This must be taken care of in code handling teleop.")
                        user_success_feedback = False
                    else:
                        user_success_feedback = input("Waiting for user feedback on success: press s then enter for success, "
                                                        "or just enter for failure.")
                    # self.gui_lock.acquire()
                    if user_success_feedback == 's':
                        info['done_success'] = True
            elif self.done_success:
                info['done_success'] = True
            elif self.done_failure:
                info['done_failure'] = True

        info['recorded_motion'] = recorded_motion

        self.prev_action = action
        # self.gui_lock.release()
        return obs, rew, terminated, truncated, info

    def get_reward(self, obs, obs_dict, action):
        """ Should be overwritten by children if needed. """
        return 0.0

    def get_terminated(self):
        """ Can be overwritten by children, but this gives a default based on timeout. """
        return self.ep_timesteps == self._max_episode_steps

    def get_truncated(self):
        """ Can be overwritten by children, but this gives a default based on human feedback"""
        return (self.success_causes_done and self.done_success) or (self.failure_causes_done and self.done_failure)

    def set_sts_render(self):
        raise NotImplementedError("if implemented, should turn on rendering in sts.")

    def render(self):
        if self.img_in_state:
            for ns, rgbd in self.render_rgbds.items():
                cv2.imshow(self.key_from_sts(ns, 'rgb'), rgbd[:, :, :3])

        if self._has_sts:
            for ns, raw_img in self.render_stss.items():
                cv2.imshow(self.key_from_sts(ns, 'raw_image'), raw_img)

        if self.first_render:  # fix cv2 bug where it doesn't show up at first
            cv2.waitKey(1)
            time.sleep(.01)
            cv2.waitKey(1)
            time.sleep(.01)
            cv2.waitKey(1)
            self.first_render = False
        else:
            cv2.waitKey(1)

import numpy as np
import time 
import pickle
import random

import gym
from gym import spaces
import rospy

from place_from_pick_learning.utils.utils import generate_pregrasp, add_pose_noise
from place_from_pick_learning.grasp_correction_from_contact import GraspClassification
from control.panda_client import PandaClient
from control.srj_gripper_client import SRJGripperClient
from control.panda_gripper_client import PandaGripperClient
from perception.realsense_client import RealSenseClient
from perception.sts_client_direct import STSClientDirect
from contact_graspnet_ros1.grasp_client import ContactGraspnetClient
from transform_utils.pose_transforms import PoseTransformer, matrix2pose
from sts.scripts.contact_detection.contact_detection import ContactDetectionCreator

# For tactile sensor, M  / pixels
PIXELS_TO_M = .040 / 640

def open_pkl(save_path):
    with open(save_path, "rb") as handle:
        data = pickle.load(handle)
    return data

class PlaceFromPickEnv(gym.Env):
    def __init__(
            self, 
            grasp_pose_filename=None,
            use_pregrasp=True,
            use_compliant_grasping=True,
            use_tactile_regrasp=False,
            retrieval_pose_filename=None,
            retrieval_pose_noise=0.025,
            home_joints=[
                0.22101389, 
                -0.90719043, 
                -0.61968366, 
                -2.20749528,  
                0.24118105,  
                1.37932865,
                -0.66226584
            ], 
            img_res=(128,128),
            gripper_close_time=3.0,
            obs_act_fixed_time=0.20,
            sts_config_dir=None,
            num_sts_img_readings=10,
            dt=0.30,
        ):
        super().__init__()

        rospy.init_node('place_from_pick_env')

        #XXX: We allow step() to be used with multiple rotation representations
        # but we'll define a fixed action space based on the nominal rvec representation
        self.action_space = action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = None

        self.arm_client = PandaClient(
            robot_ip="172.16.0.4",
            home_joints=home_joints
        )
        self.gripper_client = SRJGripperClient()
        # self.gripper_client = PandaGripperClient()
        self.camera_client = RealSenseClient(size=img_res)
        if sts_config_dir:
            self.tactile_client = STSClientDirect(
                sts_config_dir,
                num_sts_img_readings=num_sts_img_readings
            )
            self.contact_back_sub = ContactDetectionCreator(sts_config_dir).create_object()
            self.grasp_classifier = GraspClassification()
        else:
            self.tactile_client = None

        time.sleep(5) # Give clients some time

        self.gripper_close_time = gripper_close_time
        self.obs_act_fixed_time = obs_act_fixed_time
        self.rate = rospy.Rate(1/dt)
        
        # Retrieval pose related stuff
        self.retrieval_pose_filename = retrieval_pose_filename
        if self.retrieval_pose_filename:
            self.retrieval_poses_list = open_pkl(retrieval_pose_filename)
        self.retrieval_pose_noise = retrieval_pose_noise
        
        # Grasp related stuff
        self.grasp_pose_filename = grasp_pose_filename
        if self.grasp_pose_filename:
            self.grasp_poses_list = open_pkl(grasp_pose_filename)
        else:
            self.cg_client = ContactGraspnetClient(
                'camera_color_optical_frame', 
                'panda_link0', 
            )
        self.use_pregrasp = use_pregrasp
        self.use_compliant_grasping = use_compliant_grasping
        self.use_tactile_regrasp = use_tactile_regrasp

    def step(self, a=np.array([0,0,0,0,0,0,-1]), ignore_motion_act=False, rotation_representation="rvec"):
        """
        Move robot end-effector by a pose delta and close or open gripper

        Args:
            a: Array of size pose_dim (e.g., 6 for rvec) for pose + 1 for gripper
        """      
        # Variable sleep to keep time steps consistent as during training / data collection
        self.rate.sleep()   
        
        # Gripper control  
        if a[-1] < 0 :
            self.gripper_client.pinch()
        elif a[-1] > 0:
            self.gripper_client.close()
        
        # Arm control
        info = {}
        if not ignore_motion_act:
            self.arm_client.shift_EE_by(
                translation=a[:3], 
                rotation=a[3:-1],
                rotation_representation=rotation_representation,
                base_frame=True,
                interpolate=False,
                target_pose=True
            ) 
        else:
            info['recorded_motion'] = self.arm_client.get_motion_recordings(base_frame=True)

        # Fixed sleep to let robot time to move
        time.sleep(self.obs_act_fixed_time)

        return self._get_obs(), self._get_reward(), False, False, info

    def reset(self, pose=None):
        """
        If a pose is provided then go to that specific pose and set
         the target pose to be that pose, else just go to home joints.
         Optionally, pick and retrieve an object
        
        Args:
             pose (PoseTransformer): Reset pose
        """
        # Move to initial pose to detect grasp
        if pose:
            self.go_to_pose(pose)
        else:
            self.arm_client.go_home()
            time.sleep(3)
            self.arm_client.reset_target_pose()

        # Pick and retrieve plate to start place task
        self.arm_client.update_impedance_params(
            translational_stiffness=550, 
            rotational_stiffness=30,
            nullspace_stiffness=0.5
        )
        high_freq_retrieval_poses, grasp_tac_img = self.pick_and_retrieve()
        obs = self._get_obs()
        
        return (obs, grasp_tac_img, high_freq_retrieval_poses)

    def pick_and_retrieve(self, v=0.125):
        """
        Scripted pick and retrieve with proper impedances.
        Stores and returns the retrieval trajectory.
        """

        # Close gripper to pinch mode to not collide
        self.gripper_client.pinch()
        # self.gripper_client.set_debug_to(True)

        # Store initial pose
        initial_pose = self.arm_client.EE_pose

        # Get grasp pose
        if self.grasp_pose_filename:
            grasp_pose = random.choice(self.grasp_poses_list)[1]
        else:
            grasp_request = self.cg_client.send_grasp_request()
            while (not grasp_request.grasp.data):
                print("Resending grasp request to cg_net")
                grasp_request = self.cg_client.send_grasp_request()
            grasp_pose = PoseTransformer(matrix2pose(np.array(grasp_request.grasp.data).reshape(4,4)))

        # Move to pre-grasp pose
        if self.use_pregrasp:
            pregrasp_pose = generate_pregrasp(grasp_pose, distance=0.35)
            self.go_to_pose(pregrasp_pose, v=v)
        
        # Move to grasp pose
        self.go_to_pose(grasp_pose, v=v)

        # Grasp
        if self.tactile_client:
            self.tactile_client.start_tactile_storage()

        # Use compliant grasping
        if self.use_compliant_grasping:
            self.arm_client.update_impedance_params(
                translational_stiffness=1e-6, 
                rotational_stiffness=1e-6,
                nullspace_stiffness=1e-6
            )

        self.gripper_client.close()
        time.sleep(self.gripper_close_time)

        # Store tactile image during grasp
        if self.tactile_client:
            if self.use_tactile_regrasp:
                self.contact_back_sub._update_ref_image(grasp_tac_img[0])
                contact_patch = self.contact_back_sub.detect(grasp_tac_img[-1])
                diff_pixels = self.grasp_classifier.classify_grasp(contact_patch)
                if diff_pixels:
                    diff_metric = diff_pixels * PIXELS_TO_M
                    print(f"Detected shallow grasp, moving {diff_pixels} pixels or {diff_metric} metres deeper")
                            
                    self.gripper_client.open()
                    time.sleep(self.gripper_close_time)

                    # Move to new grasp pose
                    current_pose = self.arm_client.EE_pose
                    new_grasp_pose = generate_pregrasp(current_pose, distance=-diff_metric)
                    self.arm_client.update_impedance_params(
                        translational_stiffness=800, 
                        rotational_stiffness=40,
                        nullspace_stiffness=0.5            
                    )
                    self.go_to_pose(new_grasp_pose, v=v)

                    # Close gripper
                    self.arm_client.update_impedance_params(
                        translational_stiffness=1e-6, 
                        rotational_stiffness=1e-6,
                        nullspace_stiffness=1e-6
                    )
                    self.gripper_client.close()
                    time.sleep(self.gripper_close_time)
            grasp_tac_img = self.tactile_client.stop_tactile_storage()
        else:
            grasp_tac_img = None

        # Get retrieval pose
        if self.retrieval_pose_filename:
            retrieval_pose = random.choice(self.retrieval_poses_list)[1]
        else:
            retrieval_pose = initial_pose
        if self.retrieval_pose_noise > 0.0:
            retrieval_pose = add_pose_noise(
                retrieval_pose,
                var_t=self.retrieval_pose_noise, 
                var_angle=0.0, 
                var_axis=0.0
            )
        self.arm_client.update_impedance_params(
            translational_stiffness=800, 
            rotational_stiffness=40,
            nullspace_stiffness=0.5            
        )

        # XXX: Hacky shift for large dish racks
        # self.arm_client.shift_EE_by(
        #     translation=np.array([0,0,0]),
        #     rotation=np.array([0,0,-0.175])
        # )
        # time.sleep(1)

        # Go back to retrieval pose while storing high frequency poses
        self.arm_client.start_ee_storage()
        if self.use_pregrasp:
            self.go_to_pose(pregrasp_pose, v=v)
        self.go_to_pose(retrieval_pose, v=0.065)
        time.sleep(0.65)
        high_freq_retrieval_poses = self.arm_client.stop_ee_storage()

        return high_freq_retrieval_poses, grasp_tac_img

    def go_to_pose(self, pose, v=0.125):
        self.arm_client.move_EE_to(
            pose,
            max_iter=32,
            N=32,
            V=v
        )
        self.arm_client.reset_target_pose(pose)
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        ee = self.arm_client.EE_pose.get_array("rvec")
        q = self.arm_client.joint_position
        rgb = self.camera_client.get_rgb_image()
        depth = self.camera_client.get_depth_image()
        gripper = np.array([
            self.gripper_client.theta,
            self.gripper_client.dtheta,
        ])
        
        if self.tactile_client:
            ret = self.tactile_client.get_processed_sts(modes={'transformed_image'})
            tac = ret["transformed_image"]
        else:
            tac = None
            
        return {
            "ee": ee, 
            "q": q, 
            "gripper": gripper,
            "rgb": rgb, 
            "depth": depth, 
            "sts_transformed_image": tac
        }

    def _get_reward(self):
        return None
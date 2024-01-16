import os
import copy
import time

import numpy as np
from scipy.ndimage import median_filter, convolve1d
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import RotationSpline, Rotation
from simple_pid import PID

import transforms3d as tf3d
from transform_utils.pose_transforms import PoseTransformer, matrix2pose
from sts.scripts.helper import read_json
from pysts.utils import eul_rot_json_to_mat, eul_rot_to_mat
from pysts.processing import STSProcessor


STS_AXIS_DICT = {'normal': 2, 'shear_x': 0, 'shear_y': 1, 'torque': 5}

class LowPassFilter:
    def __init__(self, shape, coef, val=None):
        self.coef = coef
        if val is not None:
            if type(val) == np.ndarray:
                self.shape = val.shape
                self.val = val
            else:
                self.shape = (1,)
                self.val = np.array(val)
        else:
            self.shape = shape
            self.val = np.zeros(self.shape)

    def reset(self, val=None):
        if val is not None:
            self.val = val
        else:
            self.val = np.zeros(self.shape)

    def update_and_get(self, new_val):
        self.val = self.coef * new_val + (1 - self.coef) * self.val
        return self.val


class FTAdapter:
    # def __init__(self, adapter_type, sts_processor: STSProcessor=None):
    def __init__(self, adapter_type, env):
        """ STS-based adapters read their config params from the ft_adapted_replay.json file. """
        if adapter_type == 'open_loop_delft':
            self.adapter = OpenLoopDelFTAdapter(env)
        elif adapter_type == 'closed_loop_pid':
            self.adapter = ClosedLoopPIDFTAdapter(env)
        elif adapter_type == 'binary_contact':
            self.adapter = BinaryContactAdapter(env)
        elif adapter_type == 'forward_model':
            self.adapter = ForwardModelFTAdapter(env)
        else:
            raise NotImplementedError(f"Not implemented for {adapter_type}")

    def reset(self, demo_traj):
        """
        demo_traj: full list of sas' pairs that are being used for replay.
        """
        self.adapter.reset(demo_traj=demo_traj)

    def get_action_modifier(self, cur_ts, cur_obs):
        """
        cur_ts: current timestep of replay.
        cur_obs: the current observation dictionary
        """
        return self.adapter.get_action_modifier(ts=cur_ts, obs=cur_obs)


class FTAdapterBase:
    def __init__(self, ft_to_ee_eul_rxyz):
        self.ft_to_ee_rot_mat = eul_rot_to_mat(ft_to_ee_eul_rxyz)
        self.ft_rot_mat = np.eye(6)
        self.ft_rot_mat[:3, :3] = self.ft_to_ee_rot_mat
        self.ft_rot_mat[3:, 3:] = self.ft_to_ee_rot_mat

    def get_contact(self):
        raise NotImplementedError()

    def get_action_modifier(self):
        raise NotImplementedError()


class STSFTAdapter(FTAdapterBase):
    # def __init__(self, sts_processor: STSProcessor):
    def __init__(self, env):
        if env.t_action_base_frame != 't' or env.r_action_base_frame != 't':
            raise NotImplementedError("Only implemented for both being tool frame for now! Easy enough to fix if needed.")
        sts_processor = env.sts_clients['sts']
        self.all_params = read_json(os.path.join(sts_processor._config_dir, "ft_adapted_replay.json"))
        super().__init__(self.all_params["sts_to_ee_eul_rxyz"])
        self.sts_processor = sts_processor
        self.cum_act_mod = np.zeros(6)
        self.no_contact_lpf = LowPassFilter(shape=(6,), coef=self.all_params['no_contact_lpf_coef'])
        self.demo_traj = None

    def reset(self, demo_traj):
        self.cum_act_mod = np.zeros(6)
        self.no_contact_lpf.reset()
        self.demo_traj = demo_traj

        # get relevant arrays
        self.cur_contact = []
        self.next_contact = []
        self.next_sts_ft = []
        self.cur_sts_ft = []
        self.pose_mats = []
        self.next_pose_mats = []
        self.orig_acts = []
        for sas in demo_traj:
            self.orig_acts.append(sas[1])
            self.cur_contact.append(sas[0]['sts_in_contact'])
            self.next_contact.append(sas[2]['sts_in_contact'])
            self.next_sts_ft.append(sas[2]['sts_avg_force'])
            self.cur_sts_ft.append(sas[0]['sts_avg_force'])
            cur_pt = PoseTransformer(pose=sas[0]['pose'])
            self.pose_mats.append(cur_pt.get_matrix())
            next_pt = PoseTransformer(pose=sas[2]['pose'])
            self.next_pose_mats.append(next_pt.get_matrix())

        self.orig_acts = np.array(self.orig_acts)
        self.cur_contact = np.array(self.cur_contact)
        self.next_contact = np.array(self.next_contact)
        self.next_sts_ft = np.array(self.next_sts_ft)
        self.cur_sts_ft = np.array(self.cur_sts_ft)
        self.pose_mats = np.array(self.pose_mats)
        self.next_pose_mats = np.array(self.next_pose_mats)

        # median filter on the des sts
        all_sts_ft = np.concatenate([self.cur_sts_ft, np.atleast_2d(self.next_sts_ft[-1])])
        all_sts_ft_filtered = median_filter(
            all_sts_ft, size=(self.all_params['des_ft_med_filt_kernel'], 1), mode='nearest')
        self.cur_sts_ft_filtered = all_sts_ft_filtered[:-1]
        self.next_sts_ft_filtered = all_sts_ft_filtered[1:]

    def get_contact(self):
        return self.sts_processor._in_contact  # needs to have been started with allow_both_modes set to True

    def get_action_modifier(self):
        raise NotImplementedError()

    def enforce_cum_limits(self, act_mod_raw):
        des_cum_act_mod = self.cum_act_mod + act_mod_raw

        # enforce cumulative limits
        cum_act_mod_t_norm = np.linalg.norm(des_cum_act_mod[:3])
        if cum_act_mod_t_norm > self.all_params['max_trans_mod']:
            clipped_des_cum_t_act_mod = des_cum_act_mod[:3] / cum_act_mod_t_norm * self.all_params['max_trans_mod']
            act_mod_raw[:3] = clipped_des_cum_t_act_mod - self.cum_act_mod[:3]

        # really this should be converted to axis-angle, but since sts torque is only one-dof for now, this will be fine
        cum_act_mod_r_norm = np.linalg.norm(des_cum_act_mod[3:])
        if cum_act_mod_r_norm > self.all_params['max_rot_mod']:
            clipped_des_cum_r_act_mod = des_cum_act_mod[3:] / cum_act_mod_r_norm * self.all_params['max_rot_mod']
            act_mod_raw[3:] = clipped_des_cum_r_act_mod - self.cum_act_mod[3:]

        return act_mod_raw

    def return_to_zero_lpf(self):
        # use LPF to gradually bring cum action back to 0
        cum_act_setpoint = self.no_contact_lpf.update_and_get(np.zeros(6))
        act_mod_raw = cum_act_setpoint - self.cum_act_mod
        self.cum_act_mod += act_mod_raw

        return act_mod_raw


class BinaryContactAdapter(STSFTAdapter):
    def __init__(self, env):
        super().__init__(env)
        self.params = self.all_params['binary_contact']
        self.z_addition = self.params['z_addition']
        self.backoff_ts = int(env.control_freq * self.params['backoff_time'])
        self.load_ts = int(env.control_freq * self.params['load_time'])
        self.robot_addition = self.ft_to_ee_rot_mat.T @ np.array([0, 0, self.z_addition])

    def reset(self, demo_traj):
        super().reset(demo_traj)

        # get the ts where we transition from contact to none for backoff steps
        both_contacts = np.vstack([self.cur_contact, self.next_contact]).T
        falling_edges = np.where((both_contacts == (1, 0)).all(axis=1))
        cur_contact_mod = self.cur_contact.astype(np.float32)
        next_contact_mod = self.next_contact.astype(np.float32)
        for fe in falling_edges:
            fe = fe[0]
            cur_contact_mod[fe:fe + self.backoff_ts + 1] = np.linspace(1.0, 0, self.backoff_ts + 1)
            next_contact_mod[fe-1:fe-1 + self.backoff_ts + 1] = np.linspace(1.0, 0, self.backoff_ts + 1)

        # do the same for load steps
        rising_edges = np.where((both_contacts == (0, 1)).all(axis=1))
        for re in rising_edges:
            re = re[0]
            first = max(0, re + 1 - self.load_ts)
            num_steps = re + 2 - first
            first_for_next = max(0, first - 1)
            num_steps_next = re + 1 - first_for_next
            cur_contact_mod[first:re+2] = np.linspace(0.0, 1.0, num_steps)
            next_contact_mod[first_for_next:re+1] = np.linspace(0.0, 1.0, num_steps_next)

        # in case there's a "re-contact", overwrite the backoff
        cur_contact_mod[np.argwhere(self.cur_contact)] = 1.0
        next_contact_mod[np.argwhere(self.next_contact)] = 1.0

        per_ts_additions = np.repeat(np.atleast_2d(self.robot_addition), len(demo_traj), axis=0)
        next_per_ts_additions = np.repeat(np.atleast_2d(self.robot_addition), len(demo_traj), axis=0)
        per_ts_additions *= np.atleast_2d(cur_contact_mod).T
        next_per_ts_additions *= np.atleast_2d(next_contact_mod).T

        # need to rotate the additions into the pose of each frame
        per_ts_addition_mats = np.tile(np.eye(4), [len(demo_traj), 1, 1])
        per_ts_addition_mats[:, :3, 3] = per_ts_additions
        new_pose_mats = self.pose_mats @ per_ts_addition_mats
        next_per_ts_addition_mats = np.tile(np.eye(4), [len(demo_traj), 1, 1])
        next_per_ts_addition_mats[:, :3, 3] = next_per_ts_additions
        next_new_pose_mats = self.next_pose_mats @ next_per_ts_addition_mats

        # generate new actions with deltas
        self.new_acts = []
        for mat, next_mat in zip(new_pose_mats, next_new_pose_mats):
            delt_mat = np.linalg.inv(mat) @ next_mat
            pt = PoseTransformer(matrix2pose(delt_mat))
            self.new_acts.append(pt.get_array_rvec())

        self.new_acts = np.array(self.new_acts)

    def get_action_modifier(self, ts, obs):
        return self.new_acts[ts]


class ForwardModelFTAdapter(STSFTAdapter):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, demo_traj):
        super().reset(demo_traj)

        dist_adds_sts_frame = np.zeros([len(demo_traj), 6])
        next_dist_adds_sts_frame = np.zeros([len(demo_traj), 6])

        for ty in STS_AXIS_DICT.keys():
            if not self.all_params[ty]["use"]:
                continue

            cur_force = self.cur_sts_ft_filtered[:, STS_AXIS_DICT[ty]]
            next_force = self.next_sts_ft_filtered[:, STS_AXIS_DICT[ty]]
            dists = np.zeros_like(cur_force)
            n_dists = np.zeros_like(next_force)

            max_dist = self.all_params[ty]["max_dist"]

            # invert saved params since we need dist given force
            # y = mx + b, x = (y - b) / m = y/m - b/m
            m0_inv = 1 / self.all_params[ty]["m_0"]
            b0_inv = -self.all_params[ty]["b_0"] / self.all_params[ty]["m_0"]

            if ty == 'normal':
                pos_trans_point = self.all_params[ty]['x_trans_point']
                force_trans_point = self.all_params[ty]['m_0'] * pos_trans_point + self.all_params[ty]['b_0']

                m1_inv = 1 / self.all_params[ty]["m_1"]
                b1_inv = -self.all_params[ty]["b_1"] / self.all_params[ty]["m_1"]

                l0_indices = cur_force < force_trans_point
                l1_indices = np.invert(l0_indices)
                n_l0_indices = next_force < force_trans_point
                n_l1_indices = np.invert(n_l0_indices)

                dists[l0_indices] = m0_inv * cur_force[l0_indices] + b0_inv
                dists[l1_indices] = m1_inv * cur_force[l1_indices] + b1_inv
                n_dists[n_l0_indices] = m0_inv * next_force[n_l0_indices] + b0_inv
                n_dists[n_l1_indices] = m1_inv * next_force[n_l1_indices] + b1_inv

            else:

                # handles case where b_0 is not 0 by mirroring outputs instead
                pos_indices = cur_force >= 0
                neg_indices = np.invert(pos_indices)
                dists[pos_indices] = m0_inv * cur_force[pos_indices] + b0_inv
                n_dists[pos_indices] = m0_inv * next_force[pos_indices] + b0_inv
                dists[neg_indices] = -m0_inv * (-cur_force[neg_indices]) - b0_inv
                n_dists[neg_indices] = -m0_inv * (-next_force[neg_indices]) - b0_inv

            dist_adds_sts_frame[:, STS_AXIS_DICT[ty]] = np.clip(dists, -max_dist, max_dist)
            next_dist_adds_sts_frame[:, STS_AXIS_DICT[ty]] = np.clip(n_dists, -max_dist, max_dist)

        # rotate from sts frame to robot frame
        rot_add = (self.ft_rot_mat.T @ dist_adds_sts_frame.T).T
        n_rot_add = (self.ft_rot_mat.T @ next_dist_adds_sts_frame.T).T

        # add changes to all poses
        # be careful about doing both dist mods and rot mods...dist should be done before rot (i.e. rot should
        # have no affect on xyz) -- pretty sure this is the case with post multiplied trans matrix
        self.new_acts = []
        for add_amt, n_add_amt, pose_mat, n_pose_mat in zip(rot_add, n_rot_add, self.pose_mats, self.next_pose_mats):
            add_mat = PoseTransformer(add_amt, rotation_representation='rvec').get_matrix()
            n_add_mat = PoseTransformer(n_add_amt, rotation_representation='rvec').get_matrix()
            new_pose_mat = pose_mat @ add_mat
            n_new_pose_mat = n_pose_mat @ n_add_mat
            delt_mat = np.linalg.inv(new_pose_mat) @ n_new_pose_mat
            pt = PoseTransformer(matrix2pose(delt_mat))
            self.new_acts.append(pt.get_array_rvec())

        self.new_acts = np.array(self.new_acts)

    def get_action_modifier(self, ts, obs):
        return self.new_acts[ts]


class OpenLoopDelFTAdapter(STSFTAdapter):
    def __init__(self, env):
        super().__init__(env)
        raise NotImplementedError("Need to accomodate all fixes that have been made in the PID one")
        self.params = self.all_params['open_loop_del_ft']
        self.no_contact_lpf = LowPassFilter(shape=(6,), coef=self.all_params['no_contact_lpf_coef'])

    def reset(self, demo_traj):
        super().reset(demo_traj)
        self.no_contact_lpf.reset()
        self.del_ft = self.next_sts_ft - self.cur_sts_ft
        self.act_mod_raw = self.del_ft * self.params['mults']

    def get_action_modifier(self, ts, obs):
        if self.cur_contact[ts]:
            # use del force of traj to set action
            act_mod_raw = self.enforce_cum_limits(self.act_mod_raw[ts])  # this could be precomputed but eh
            self.cum_act_mod += act_mod_raw
            self.no_contact_lpf.reset(val=self.cum_act_mod)
        else:
            # return to zero action gradually with lpf
            act_mod_raw = self.return_to_zero_lpf()

        act_mod = self.ft_rot_mat @ act_mod_raw

        print(f"del ft: {self.del_ft[ts]}, act mod: {act_mod}")

        return act_mod


class ClosedLoopPIDFTAdapter(STSFTAdapter):
    def __init__(self, env):
        super().__init__(env)
        raise NotImplementedError("This needs to be fixed if we're going to use it!")
        self.params = self.all_params['closed_loop_pid']
        self.pid = PID(**self.params['pid'], sample_time=None)
        self.cur_ft_lpf = LowPassFilter(shape=(6,), coef=self.all_params['ft_lpf_coef'])
        self.no_contact_lpf = LowPassFilter(shape=(6,), coef=self.all_params['no_contact_lpf_coef'])
        self.max_steps = self.params['future_weights']['max_steps']
        future_weights_1d = np.array(
            [self.params['future_weights']['discount'] ** n for n in range(0, self.max_steps)])
        self.future_weights_1d = future_weights_1d / future_weights_1d.sum()

        self.act_t_frame = env.t_action_base_frame
        self.act_r_frame = env.r_action_base_frame
        self.total_act_mod = np.zeros(6)
        self.reset_t_act_lpf = LowPassFilter(shape=(3,), coef=self.all_params['no_contact_lpf_coef'])

        # for rotations, will be based on angle/norm of rotation vector
        self.reset_r_act_lpf = LowPassFilter(shape=(1,), coef=self.all_params['no_contact_lpf_coef'])

        self.env = env

    def reset(self, demo_traj):
        super().reset(demo_traj)
        self.pid.reset()
        self.cur_ft_lpf.reset()
        self.no_contact_lpf.reset()
        self.total_act_mod = np.zeros(6)
        self.reset_t_act_lpf.reset()
        self.reset_r_act_lpf.reset()

        # precompute desired wrenches including weights on future fts
        # first need to rotate future wrenches into current frame for each ts
        # going to do with for loop for now because it's simpler to think about
        # takes 10ms in practice, so doesn't matter
        start_rotate = time.time()
        fixed_des_sts_ft = []
        orig_act_mats = []
        pose_mats = []
        next_pose_mats = []
        rvecs = []
        next_rvecs = []
        for t in range(len(demo_traj)):
            # get all original desired actions
            orig_act_mats.append(PoseTransformer(pose=demo_traj[t][1], rotation_representation='rvec').get_matrix())
            cur_pt = PoseTransformer(pose=demo_traj[t][0]['pose'])
            next_pt = PoseTransformer(pose=demo_traj[t][2]['pose'])
            pose_mats.append(cur_pt.get_matrix())
            next_pose_mats.append(next_pt.get_matrix())
            rvecs.append(cur_pt.get_rvec())
            next_rvecs.append(next_pt.get_rvec())

            # rotate future sts wrenches into current frame
            combined_des_sts_ft_t = self.future_weights_1d[0] * self.cur_des_sts_ft[t]
            for fut_t_i, fut_t in enumerate(range(t + 1, t + self.max_steps)):

                fut_t_to_use = min(fut_t, len(demo_traj) - 1)  # pad the final timestep with the same wrench
                R_cur_des = tf3d.quaternions.quat2mat(demo_traj[t][2]['pose'][3:])
                R_fut_des = tf3d.quaternions.quat2mat(demo_traj[fut_t_to_use][2]['pose'][3:])
                R_fut_to_cur = R_fut_des.T @ R_cur_des

                R_full = np.eye(6)
                R_full[:3, :3] = R_fut_to_cur
                R_full[3:, 3:] = R_fut_to_cur
                rotated_fut_des_sts_ft = R_full @ self.cur_des_sts_ft[fut_t_to_use]

                combined_des_sts_ft_t += self.future_weights_1d[fut_t_i + 1] * rotated_fut_des_sts_ft

            fixed_des_sts_ft.append(combined_des_sts_ft_t)

        fixed_des_sts_ft = np.array(fixed_des_sts_ft)
        self.cur_des_sts_ft = fixed_des_sts_ft
        self.orig_act_mats = np.array(orig_act_mats)
        self.pose_mats = np.array(pose_mats)
        self.next_pose_mats = np.array(next_pose_mats)
        self.rvecs = np.array(rvecs)
        self.next_rvecs = np.array(next_rvecs)
        pose_mats_all = np.concatenate([self.pose_mats, np.expand_dims(self.next_pose_mats[-1], 0)])
        self.pos_spline = make_interp_spline(range(len(demo_traj) + 1), pose_mats_all[:, :3, 3])

        rvecs_all = np.concatenate([self.rvecs, np.expand_dims(self.next_rvecs[-1], 0)])
        rots_all = Rotation.from_rotvec(rvecs_all)
        self.rot_spline = RotationSpline(range(len(demo_traj) + 1), rots_all)
        # self.rot_spline_simple = make_interp_spline(range(len(demo_traj) + 1), rvecs_all)

        # print(f"FORCE FRAME ROTATE TIME: {time.time() - start_rotate}")

        # old fast method that didn't properly handle rotations
        # origin = int(np.ceil(self.max_steps / 2) - 1)
        # # self.cur_des_sts_ft = convolve1d(
        # #     self.cur_des_sts_ft, np.flip(self.future_weights_1d), axis=0, origin=origin, mode='nearest')
        # old_fix = convolve1d(
        #     self.cur_des_sts_ft, np.flip(self.future_weights_1d), axis=0, origin=origin, mode='nearest')

    def get_action_modifier(self, ts, obs):
        start_act_modify = time.time()

        t_error_beyond_max = False
        r_error_beyond_max = False
        des_ft = self.cur_des_sts_ft[ts]
        cur_ft_raw = obs['sts_avg_force']
        cur_ft = self.cur_ft_lpf.update_and_get(cur_ft_raw)
        if not self.params['use_torque']:
            des_ft[3:] = 0

        cur_des_pose_rot_mat_inv = self.next_pose_mats[ts][:3, :3].T

        # 1. use PID based on ft to set action
        self.pid.setpoint = des_ft
        act_mod_raw = self.pid(cur_ft)
        act_mod = self.ft_rot_mat @ act_mod_raw

        # act_mod = np.zeros(6)
        # act_mod[3:] = np.zeros(3)

        # print(f"ACT MOD: {act_mod}")

        # 2. pose fixing to ensure we don't drift too far from original trajectory
        # this was flawed due to control latency, so we're trying with spline instead
        # next_expected_pose = PoseTransformer(pose=obs['pose']).get_matrix() @ self.orig_act_mats[ts]
        # expected_error = np.linalg.inv(next_expected_pose) @ self.next_pose_mats[ts]
        # print(f"EXPECTED ERROR T NORM at ts {ts}: {np.linalg.norm(expected_error[:3, 3])}")

        # TODO hardcoded values for max error, should be in config
        max_t_error = .005  # m, so 5 mm
        max_r_error = 2     # deg

        # testing pose fixing with spline
        # start with hardcoded 10 eval points per timestep and up to 3 prev timesteps
        # TODO unhardcode these values
        if ts > 0:

            first_ts = max(ts - 3, 0)
            num_pts_per_ts = 20
            num_inps = (ts - first_ts) * num_pts_per_ts
            test_inps = np.linspace(first_ts, ts, num_inps)
            pos_spl_values = self.pos_spline(test_inps)
            dists = np.linalg.norm(obs['pose'][:3] - pos_spl_values, axis=-1)
            closest = dists.argmin()
            shortest_dist = dists.min()
            print(f"SHORTEST ERROR at ts {ts}, {(num_inps - closest) / num_pts_per_ts} ts back: {shortest_dist}")

            # overwrite and reduce cumulative if beyond threshold
            if shortest_dist > max_t_error:
                t_error_beyond_max = True
                cum_t_act_setpoint = self.reset_t_act_lpf.update_and_get(np.zeros(3))
                act_mod_raw = cum_t_act_setpoint - self.total_act_mod[:3]
                # act_mod[:3] = cur_pose_rot_mat_inv.T @ act_mod_raw
                act_mod[:3] = cur_des_pose_rot_mat_inv.T @ act_mod_raw

            # TODO plan is to directly use the spline and move the arm back in that direction, instead of using the accumulation method
            # accumulation actually appears to be working now that everything is done in the frame of the desired pose, so going to
            # stick with that

            # now with rotations
            rot_spl_values = self.rot_spline(test_inps)
            rot_matrices = rot_spl_values.as_matrix()
            diff_mats = cur_des_pose_rot_mat_inv @ rot_matrices
            diff_rvecs = Rotation.from_matrix(diff_mats).as_rotvec()
            diff_rvec_angs = np.linalg.norm(diff_rvecs, axis=-1)
            rot_closest = diff_rvec_angs.argmin()
            rot_shortest_dist = diff_rvec_angs.min()
            rot_shortest_dist_deg = rot_shortest_dist * 180 / np.pi

            print(f"SHORTEST ROT ERROR at ts {ts}, {(num_inps - rot_closest) / num_pts_per_ts} ts back: {rot_shortest_dist_deg}")

            # overwrite and reduce cumulative if beyond threshold
            if rot_shortest_dist_deg > max_r_error and np.any(np.abs(self.total_act_mod[3:])):
                r_error_beyond_max = True
                cum_r_act_setpoint = self.reset_r_act_lpf.update_and_get(np.zeros(1))
                total_act_mod_rot_ang = np.linalg.norm(self.total_act_mod[3:])
                total_act_mod_rot_axis = self.total_act_mod[3:] / total_act_mod_rot_ang
                act_ang = cum_r_act_setpoint - total_act_mod_rot_ang
                act_mod_raw = total_act_mod_rot_axis * act_ang
                act_mod[3:] = cur_des_pose_rot_mat_inv.T @ act_mod_raw

        # 3. Accumulate act mod
        to_accum = np.zeros(6)
        print(f"ACT MOD: {act_mod}")
        if self.act_t_frame == 't':  # t tool, b base
            to_accum[:3] = cur_des_pose_rot_mat_inv @ act_mod[:3]
        else:
            to_accum[:3] = act_mod[:3]

        if self.act_r_frame == 't':
            to_accum[3:] = cur_des_pose_rot_mat_inv @ act_mod[3:]
        else:
            to_accum[3:] = act_mod[3:]

        self.total_act_mod[:3] += to_accum[:3]
        # self.total_act_mod[3:] += to_accum[3:]
        self.total_act_mod[3:] = (Rotation.from_rotvec(self.total_act_mod[3:]) * Rotation.from_rotvec(to_accum[3:])).as_rotvec()

        print(f"TOTAL ACT MOD: {self.total_act_mod}")

        if not t_error_beyond_max:
            self.reset_t_act_lpf.reset(self.total_act_mod[:3])
        if not r_error_beyond_max:
            # self.reset_r_act_lpf.reset(self.total_act_mod[3:])
            self.reset_r_act_lpf.reset(np.linalg.norm(self.total_act_mod[3:]))

        # print(f"ACTION MODIFY TIME: {time.time() - start_act_modify}")  # tested..appears to be about 2 seconds

        # print(f"act ft: {cur_ft}")
        # print(f"act mod: {act_mod}")
        # print(f"cum act mod: {self.cum_act_mod}")

        return act_mod


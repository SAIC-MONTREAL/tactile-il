import time
import numpy as np
import os
from PIL import Image
from multiprocessing import Pool
from itertools import repeat
import shutil
import gzip
import json
from collections import OrderedDict
import uuid
import pickle
import glob
import gzip

import gym
from gym import spaces

from transform_utils.pose_transforms import PoseTransformer, matrix2pose


def compress_outs(compress):
    if compress:
        o_func = gzip.open
        f_ext = ".pkl.gz"
    else:
        o_func = open
        f_ext = ".pkl"
    return o_func, f_ext

def get_ofunc_from_filename(filename):
    file_ext = filename.split('.')[-1]
    return gzip.open if file_ext == 'gz' else open


def save_pkl(data, save_file):
    o_func = get_ofunc_from_filename(save_file)
    with o_func(f"{save_file}", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def open_pkl(save_file):
    o_func = get_ofunc_from_filename(save_file)
    with o_func(f"{save_file}", "rb") as handle:
        data = pickle.load(handle)
    return data


def get_sorted_dict(dic):
    return OrderedDict(sorted(dic.items(), key=lambda t: t[0]))


class EnvEncoder(json.JSONEncoder):
    STR_TYPES = [spaces.Box, spaces.Dict]
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if type(obj) in EnvEncoder.STR_TYPES:
            return str(obj)
        return super(EnvEncoder, self).default(obj)

class StrEncoder(json.JSONEncoder):
    def default(self, obj):
        return str(obj)


class DictDataset:
    ENV_CFG_IGNORE_ON_LOAD = {"sts_config_dirs", "sts_config_dirs_loaded"}
    PERF_STAT_KEYS = {"return", "success"}
    RAW_DEMO_DATA_STR = 'raw_demo_data'
    def __init__(self,
                 pa_args=None,
                 dataset_name=None,
                 main_dir=None,
                 compress=False,
                 env=None,
                 load_saved_sts_config=False,
                 csv_save_keys=(),
                 mp4_save_keys=(),
                 load_env_ignore_keys=set(),
                 ):

        if main_dir is None:
            main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if dataset_name is None:
            dataset_name = f"dt-{pa_args.dt}_img-res-{pa_args.res}_episodes-{pa_args.n_episodes}_{pa_args.dataset_id}_{uuid.uuid4().hex}"

        self._dir = os.path.join(main_dir, dataset_name)
        self._ds_parameters_file = os.path.join(self._dir, "dataset_parameters.json")
        self._env_parameters_file = os.path.join(self._dir, "env_parameters.json")
        self._performance_file = os.path.join(self._dir, "performance.json")
        self._csv_save_keys = csv_save_keys
        self._mp4_save_keys = mp4_save_keys
        self._loaded = False
        if hasattr(env, 'cfg'):
            self.env_cfg = env.cfg
        else:
            self.env_cfg = {}

        # set up saving/loading sts config
        self.sts_clients = dict()
        self.sts_cfg_dir = os.path.join(self._dir, "sts_configs")
        if hasattr(env, 'tactile_client') or (hasattr(env, '_has_sts') and env._has_sts):
            if hasattr(env, 'tactile_client'):  # PlaceFromPickEnv
                self.sts_clients['sts'] = env.tactile_client

            elif hasattr(env, '_has_sts') and env._has_sts:  # contact_panda_envs
                self.sts_clients = env.sts_clients

        if os.path.exists(self._ds_parameters_file):
            print(f"Dataset at {self._dir} exists, loading!")
            self._loaded = True
            with open(self._ds_parameters_file, "r") as f:
                self._params = OrderedDict(json.load(f))

            if os.path.exists(self._performance_file):
                print(f"Opening existing performance file.")
                with open(self._performance_file, "r") as f:
                    self._performance_data = OrderedDict(json.load(f))

            if hasattr(env, 'cfg'):
                with open(self._env_parameters_file, "r") as f:
                    lep = json.load(f)

                for k in lep:
                    if k in DictDataset.ENV_CFG_IGNORE_ON_LOAD.union(load_env_ignore_keys):
                        continue
                    assert k in env.cfg, f"env config key {k} in saved env, but not in current env!"
                    assert env.cfg[k] == lep[k], f"saved env config {k}: {lep[k]}, input env: {env.cfg[k]}"
                print("Saved env config matches current env config.")
            print(f"Finished loading dataset at {self._dir} with {self._params['actual_n_episodes']} episodes.")

            # load saved sts configs(s)
            if len(self.sts_clients) > 0 and load_saved_sts_config:
                ns_sts_cfg_dirs_abs = []
                ns_sts_cfg_dirs_rel = []
                for ns in self.sts_clients:
                    # ns_sts_cfg_dir = os.path.join(sts_cfg_dir, ns)
                    ns_sts_cfg_dirs_rel.append(os.path.join('.', "sts_configs", ns))
                    ns_sts_cfg_dir_abs = os.path.join(os.path.dirname(self._ds_parameters_file), "sts_configs", ns)
                    ns_sts_cfg_dirs_abs.append(ns_sts_cfg_dir_abs)
                    self.sts_clients[ns].load_config(ns_sts_cfg_dir_abs)

                # save the "new" config file as addition to the original one to try and maintain sanity
                env.cfg['sts_config_dirs_loaded'] = ns_sts_cfg_dirs_rel
                self.save_parameters_file(self._env_parameters_file, get_sorted_dict(env.cfg))
                print(f"Loaded sts config(s) from {self.sts_cfg_dir}.")

            if 'raw_dataset_for_replay' in self._params:
                rdfr_ds = DictDataset(
                    pa_args=None,
                    dataset_name="",
                    main_dir=self._params['raw_dataset_for_replay'],
                    env=env,  # worth verifying that envs are the same
                    load_saved_sts_config=False,  # only want the trajectories
                    load_env_ignore_keys={'env_name', 'sts_initial_mode', 'sts_no_switch_override', 'state_data',
                                        'sts_switch_in_action'}  # allow tactile only env
                )
                self._rdfr = rdfr_ds

        else:
            if pa_args is not None:
                self._params = OrderedDict(sorted(pa_args.__dict__.items(), key=lambda t: t[0]))
            else:
                self._params = OrderedDict()
            self._params['actual_n_episodes'] = 0
            self._params['num_data'] = 0
            self._params['ep_lens'] = []
            self._params['zfill'] = 4
            self._params['compress'] = compress
            # self.save_ds_parameters_file()

            # if hasattr(env, 'cfg'):
            #     self.save_parameters_file(self._env_parameters_file, get_sorted_dict(env.cfg))

            # # save sts config(s)
            # if len(self.sts_clients) > 0:
            #     os.makedirs(self.sts_cfg_dir, exist_ok=True)
            #     for ns in self.sts_clients:
            #         shutil.copytree(self.sts_clients[ns]._config_dir, os.path.join(self.sts_cfg_dir, ns))

    def __len__(self):
        return self._params['actual_n_episodes']

    def attach_raw_dataset_for_replay(self, ds: 'DictDataset'):
        if 'raw_dataset_for_replay' in self._params:
            print(f"Dataset at {self._dir} already has a raw dataset for replay: {self._params['raw_dataset_for_replay']}")
            print(f"Starting raw dataset for replay at ep {self._params['rdfr_current_ep']}")
        else:
            self._params['raw_dataset_for_replay'] = ds._dir
            self._params['rdfr_current_ep'] = 0
            self._params['rdfr_ep_labels'] = []
            print(f"Attaching dataset at {self._params['raw_dataset_for_replay']} as raw dataset for replay.")

        self._rdfr = ds

    def get_rdfr_ep_params(self, ep_i=None):
        assert hasattr(self, "_rdfr"), "Call attach_raw_dataset_for_replay before attempting to get rdfr ep."
        if ep_i is None:
            ep_i = self._params['rdfr_current_ep']
        real_ep_data, raw_ep_data = self._rdfr.load_ep(ep_i, include_raw_demo=True)

        # get sts switch actions from real data, since they weren't done in raw
        if self._rdfr.env_cfg['sts_switch_in_action']:
            recorded_sts_switch_actions = []
            for sas in real_ep_data:
                recorded_sts_switch_actions.append(sas[1][-1])

        reset_joint_position = raw_ep_data[0][0]['joint_pos']
        recorded_actions = []
        for sas_i, sas in enumerate(raw_ep_data):
            recorded_actions.append(sas[1])

            if self._rdfr.env_cfg['sts_switch_in_action']:
                recorded_actions[-1][-1] = recorded_sts_switch_actions[sas_i]

        return raw_ep_data, reset_joint_position, recorded_actions

    def get_ep_file(self, i, sub_folder='data', ext='.pkl'):
        if self._params['compress'] and ext == '.pkl':
            ext += '.gz'
        return os.path.join(self._dir, sub_folder, str(i).zfill(self._params['zfill']) + ext)

    def load_ep(self, i, include_raw_demo=False):
        assert i <= self._params['actual_n_episodes'], \
            f"i {i} greater than number of eps in dataset {self._params['actual_n_episodes']}"

        data = open_pkl(self.get_ep_file(i))
        if include_raw_demo:
            return data, open_pkl(self.get_ep_file(i, sub_folder=DictDataset.RAW_DEMO_DATA_STR))
        return data

    def load_ep_as_dict_of_arrays(self, i, include_raw_demo=False):
        assert i <= self._params['actual_n_episodes'], \
            f"i {i} greater than number of eps in dataset {self._params['actual_n_episodes']}"

        returns = []

        if include_raw_demo:
            sas_list, raw_sas_list = self.load_ep(i, include_raw_demo=include_raw_demo)
        else:
            sas_list = self.load_ep(i, include_raw_demo=include_raw_demo)
            raw_sas_list = []

        for sas_l in [sas_list, raw_sas_list]:
            ret_dict = dict()
            for k in sas_l[0][0].keys():
                ret_dict[k] = []
                ret_dict[f"n_{k}"] = []
            ret_dict['act'] = []
            for sas in sas_l:
                st, ac, nst = sas
                for k in st.keys():
                    ret_dict[k].append(st[k])
                    ret_dict[f"n_{k}"].append(nst[k])
                ret_dict['act'].append(ac)

            for k in ret_dict.keys():
                ret_dict[k] = np.array(ret_dict[k])

            if len(raw_sas_list) == 0:
                return ret_dict

            returns.append(ret_dict)

        return returns

    def save_ep(self, sas_list, perf_dict=None, raw_demo_data=None):
        """ perf_dict can optionally contain performance parameters such as return, success, rewards, etc.

        raw_demo_data should contain a list, equal length of sas_list, that has various parameters from the raw
        robot trajectory (before the replay) to make repeating/analyzing easier"""

        # only create directories once we save eps
        os.makedirs(self._dir, exist_ok=True)
        os.makedirs(os.path.join(self._dir, "data"), exist_ok=True)

        if not os.path.exists(self._env_parameters_file) and len(self.env_cfg) > 0:
            self.save_parameters_file(self._env_parameters_file, get_sorted_dict(self.env_cfg))

        # save sts config(s)
        if len(self.sts_clients) > 0:
            if not os.path.exists(self.sts_cfg_dir):
                os.makedirs(self.sts_cfg_dir, exist_ok=True)
                for ns in self.sts_clients:
                    shutil.copytree(self.sts_clients[ns]._config_dir, os.path.join(self.sts_cfg_dir, ns))

        print(f"Saving episode...")
        save_pkl(sas_list, self.get_ep_file(self._params['actual_n_episodes']))
        if len(self._csv_save_keys) > 0:
            self.save_ep_csv(sas_list, self.get_ep_file(self._params['actual_n_episodes'], sub_folder='csvs', ext='.csv'))
            if raw_demo_data is not None:
                self.save_ep_csv(raw_demo_data, self.get_ep_file(self._params['actual_n_episodes'],
                                                                 sub_folder='raw_demo_csvs', ext='.csv'))
        if len(self._mp4_save_keys) > 0:
            from pysts.utils import sas_list_to_vids
            sas_list_to_vids(self._params['actual_n_episodes'], sas_list, self._mp4_save_keys, self._dir, 'data_videos')
            if raw_demo_data is not None:
                sas_list_to_vids(self._params['actual_n_episodes'], raw_demo_data, self._mp4_save_keys, self._dir,
                                 'raw_data_videos')

        if raw_demo_data is not None:
            os.makedirs(os.path.join(self._dir, DictDataset.RAW_DEMO_DATA_STR), exist_ok=True)
            save_pkl(raw_demo_data, self.get_ep_file(self._params['actual_n_episodes'], DictDataset.RAW_DEMO_DATA_STR))
        if 'raw_dataset_for_replay' in self._params and self._params['rdfr_current_ep'] < len(self._rdfr):
            self._params['rdfr_ep_labels'].append(self._params['actual_n_episodes'])
            self._params['rdfr_current_ep'] += 1
        self._params['actual_n_episodes'] += 1
        self._params['num_data'] += len(sas_list)
        self._params['ep_lens'].append(len(sas_list))
        self.save_ds_parameters_file()

        if perf_dict is not None:
            if not hasattr(self, "_performance_data"):
                self._performance_data = OrderedDict()
                for k, v in perf_dict.items():
                    self._performance_data[k] = [v]
            else:
                for k, v in perf_dict.items():
                    self._performance_data[k].append(v)
                    assert len(self._performance_data[k]) == self._params['actual_n_episodes'], \
                        f"Peformance metric {k} has length of {len(self._performance_data[k])}, "\
                        f"should be {self._params['actual_n_episodes']}"

            self.calc_perf_stats()
            self.save_perf_parameters_file()

            print("Ep performance: " + ''.join([f"{k}: {v}, " for k, v in perf_dict.items()]))
            for k in DictDataset.PERF_STAT_KEYS & self._performance_data.keys():
                print(f"{k}: Mean: {self._performance_data[k + '_mean']:.3f}, "
                      f"Std: {self._performance_data[k + '_std']:.3f}")

        if 'n_episodes' in self._params:
            print(f"Episode saved, dataset now contains {self._params['actual_n_episodes']}/{self._params['n_episodes']}.")
        else:
            print(f"Episode saved, dataset now contains {self._params['actual_n_episodes']} episodes.")

    def calc_perf_stats(self):
        for k in DictDataset.PERF_STAT_KEYS:
            if k in self._performance_data:
                arr = np.array(self._performance_data[k])
                self._performance_data[k + "_mean"] = float(arr.mean())
                self._performance_data[k + "_std"] = float(arr.std())

    def remove_last_ep(self, query_error_type=False):
        if self._params['actual_n_episodes'] == 0:
            print("No episodes to remove!")
            return

        os.remove(self.get_ep_file(self._params['actual_n_episodes'] - 1))

        raw_dd_f = self.get_ep_file(self._params['actual_n_episodes'] - 1, sub_folder=DictDataset.RAW_DEMO_DATA_STR)
        if os.path.exists(raw_dd_f):
            os.remove(raw_dd_f)

        still_using_rdfr = False
        if 'raw_dataset_for_replay' in self._params:
            still_using_rdfr = self._params['rdfr_current_ep'] < len(self._rdfr) or \
                (self._params['rdfr_current_ep'] == len(self._rdfr) and self._params['rdfr_ep_labels'][-1] != 'f')

        # if query_error_type or still_using_rdfr:
        if query_error_type:
            e_type = input(
                "What type of error?\n"
                "f: force adjustment during replay\n"
                "h: human error\n"
                "o: other\n")
            if e_type == 'f':
                if 'num_force_error_discard' not in self._params:
                    self._params['num_force_error_discard'] = 0
                self._params['num_force_error_discard'] += 1
                if still_using_rdfr:
                    self._params['rdfr_ep_labels'][-1] = 'f'  # now this rdfr ep index is labelled as force error
            elif e_type == 'h':
                if 'num_human_error_discard' not in self._params:
                    self._params['num_human_error_discard'] = 0
                self._params['num_human_error_discard'] += 1
                if still_using_rdfr:
                    self._params['rdfr_current_ep'] -= 1
                    self._params['rdfr_ep_labels'].pop()
            else:
                if still_using_rdfr:
                    self._params['rdfr_current_ep'] -= 1
                    self._params['rdfr_ep_labels'].pop()

            if still_using_rdfr:
                print(f"Current raw dataset for replay ep: {self._params['rdfr_current_ep']}")
        else:
            if still_using_rdfr:
                self._params['rdfr_current_ep'] -= 1
                self._params['rdfr_ep_labels'].pop()


        self._params['actual_n_episodes'] -= 1
        self._params['num_data'] -= self._params['ep_lens'][-1]
        self._params['ep_lens'].pop()
        self.save_ds_parameters_file()

        if hasattr(self, "_performance_data"):
            for k in self._performance_data:
                if "mean" not in k and "std" not in k:
                    self._performance_data[k].pop()
            if self._params['actual_n_episodes'] > 0:
                self.calc_perf_stats()
                self.save_perf_parameters_file()

        print(f"Episode deleted, dataset now contains {self._params['actual_n_episodes']} episodes.")

    def save_ds_parameters_file(self):
        self.save_parameters_file(self._ds_parameters_file, self._params)

    def save_perf_parameters_file(self):
        self.save_parameters_file(self._performance_file, self._performance_data)

    def save_parameters_file(self, file, params):
        with open(file, "w") as f:
            json.dump(params, f, indent=2, cls=EnvEncoder)

    def save_ep_csv(self, sas_list, file, num_dec=5, max_header_len=8, pose_as_rvec=True):
        first = True
        csv_lines = []
        for sas in sas_list:
            s_t, a_t, s_t1 = sas[0], sas[1], sas[2]
            if first:
                # get headers
                csv_header_line = []
                for k in self._csv_save_keys:
                    if k == 'action':
                        shape = np.atleast_1d(a_t).shape
                    else:
                        shape = np.atleast_1d(s_t[k]).shape

                    if 'pose' in k and pose_as_rvec:
                        shape = (6,)

                    assert len(shape) == 1, f"Got shape {shape} for k {k}, but entry must be 1-dimensional"

                    for i in range(shape[0]):
                        csv_header_line.append(f"{k[:max_header_len]}-{i}")
                csv_lines.append(",".join(csv_header_line))
                first = False

            csv_line = []
            for k in self._csv_save_keys:
                if k == 'action':
                    shape = np.atleast_1d(a_t).shape
                    data = np.atleast_1d(a_t)
                else:
                    shape = np.atleast_1d(s_t[k]).shape
                    data = np.atleast_1d(s_t[k])

                if 'pose' in k and pose_as_rvec:  # convert to rvec since they're more readable
                    pt = PoseTransformer(data)
                    data = pt.get_array_rvec()
                    shape = data.shape

                for i in range(shape[0]):
                    csv_line.append(f"{data[i]:.{num_dec}f}")

            csv_lines.append(",".join(csv_line))

        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'w') as f:
            for l in csv_lines:
                f.write(f"{l}\n")

import random
import numpy as np
import sys
import json
import os
import math

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from hydra.utils import instantiate
from omegaconf.omegaconf import open_dict
from torch.distributions import Normal, Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from hydra.utils import instantiate

from transform_utils.pose_transforms import PoseTransformer


FLOAT_IMG_KEYS = {'depth', 'sts_flow', 'sts_marker_flow'}
RGB_IMG_KEYS = {'rgb', 'sts_raw_image', 'wrist_rgb'}
IMG_KEYS = FLOAT_IMG_KEYS.union(RGB_IMG_KEYS)


def frame_stack(x, frames=1):
    """
    Given a trajectory of images with shape (n, l, c, ...) convert to
    (n, l - frames, (frames + 1) * c, ...), where the channel dimension
    contains the extra frames added.

    e.g. visualization of frames=2:
    x_{0} | x_{1} x_{2} x_{3} ... x_{l}   |
    0     | x_{0} x_{1} x_{2} ... x_{l-1} | x_{l}

    NOTE: "Index 0" is the current frame, and index 1+ is the history in the framestack channel
    """
    input_type = type(x)
    if input_type is np.ndarray:
        x = torch.from_numpy(x)

    n, l, c = x.shape[:3]
    x_stacked = torch.zeros((n, l, (frames + 1) * c, *x.shape[3:]),
                                dtype=x.dtype, device=x.device)
    x_stacked[:, :, :c] = x
    for ii in (_ + 1 for _ in range(frames)):
        pad = torch.zeros(
            (n, ii, c, *x.shape[3:]),
            dtype=x.dtype, device=x.device
        )
        x_stacked[:, :, ((ii) * c):((ii+1) * c)] = \
            torch.cat((pad, x), dim=1)[:, :l]
    # slice off the initial part of the traj w/ no history
    x_stacked = x_stacked[:, frames:]

    if input_type is np.ndarray:
        x_stacked = x_stacked.detach().cpu().numpy()
    return x_stacked

def process_obs(
    tuple_sequence,
    rotation_representation="rvec",
    stored_rotation_representation="rvec",
    n_frame_stacks=1,
    load_tensor=False,
    device="cpu"
):
    """Process observation for network."""
    first_obs, _, _ = tuple_sequence[0]

    # Store data and remove null keys
    state_dict = {k: [] for k, v in first_obs.items() if (v is not None)}
    # XXX: hack for dataset that has some data with this and some without
    if 'raw_world_pose' in state_dict:
        state_dict.pop('raw_world_pose')

    for obs, act, next_obs in tuple_sequence:
        for k, v in obs.items():
            # XXX: hack for dataset that has some data with this and some without
            if k == 'raw_world_pose':
                continue
            if v is not None:
                if k =="ee":
                    p = PoseTransformer(
                        v,
                        rotation_representation=stored_rotation_representation
                    )
                    v = p.get_array(rotation_representation)
                v = np.expand_dims(v, axis=0).astype('float32')

                if k in FLOAT_IMG_KEYS:
                    if len(v.shape) == 3:  # e.g. depth, no channel dimension
                        v = np.expand_dims(v, axis=-1).astype('float32')
                    v = v.transpose(0,3,1,2)

                if k in RGB_IMG_KEYS:
                    v = v.astype('float32')
                    v = v.transpose(0,3,1,2)
                    v = v / 255.0

                state_dict[k].append(v)

    # Concatenate list data into a single tensor
    state_dict = {k:np.concatenate(v, axis=0) for k, v in state_dict.items()}

    # Stack frames
    if n_frame_stacks >= 1:
        state_dict = {k:frame_stack(v[None], frames=n_frame_stacks)[0] for k,v in state_dict.items()}

    # Move to pytorch tensor directly
    if load_tensor:
        state_dict = {k:torch.from_numpy(v).float().to(device) for k,v in state_dict.items()}

    return state_dict

def process_action(
    tuple_sequence,
    rotation_representation="rvec",
    stored_rotation_representation="rvec",
    n_frame_stacks=1,
    load_tensor=False,
    device="cpu",
    grip_in_action=True,
    sts_switch_in_action=False
):
    """Process action for network."""
    #TODO: We can consider using one-hot-encoding for gripper control
    actions = []
    for obs, act, next_obs in tuple_sequence:
        if sts_switch_in_action and grip_in_action:
            a_m = act[:-2]
            a_g = act[-2]
            a_s = act[-1]
        elif not sts_switch_in_action and grip_in_action:
            a_m = act[:-1]
            a_g = act[-1]
        elif sts_switch_in_action and not grip_in_action:
            a_m = act[:-1]
            a_s = act[-1]
        else:
            a_m = act

        p = PoseTransformer(
            a_m,
            rotation_representation=stored_rotation_representation
        )
        a_m = p.get_array(rotation_representation)

        if sts_switch_in_action and grip_in_action:
            action_array = np.concatenate((a_m, [a_g], [a_s]))[None, ...]
        elif not sts_switch_in_action and grip_in_action:
            action_array = np.concatenate((a_m, [a_g]))[None, ...]
        elif sts_switch_in_action and not grip_in_action:
            action_array = np.concatenate((a_m, [a_s]))[None, ...]
        else:
            action_array = a_m[None, ...]  # expand dims in 0th dimension

        actions.append(action_array)

    actions = np.concatenate(actions, axis=0)

    # Account for stacked frames
    if n_frame_stacks >= 1:
        actions = actions[n_frame_stacks:]

    # Move to pytorch tensor directly
    if load_tensor:
        actions = torch.from_numpy(actions).float().to(device)

    return actions

def build_state_action_pair(
    tuple_sequence,
    obs_rotation_representation="rvec",
    stored_obs_rotation_representation="rvec",
    act_rotation_representation="rvec",
    stored_act_rotation_representation="rvec",
    n_frame_stacks=1,
    load_tensor=False,
    device="cpu",
    grip_in_action=True,
    sts_switch_in_action=False
):
    '''
    Given a sequence of tuples ( obs (dict), act(PoseStamped), next_obs(dict) ) returns a state action pair.
    State is a dictionary generated the sequence of observations (which are dictionaries)
    Action is a tuple of (PoseTransform, int).
    Input:
        :param tuple_sequence: list of tuples
        :param rotation_representation: string representing the choice of
            rotation representation from {quat, rvec, axisa, euler}
    Output:
        :param state: dictionary of numpy array or pytorch tensor
        :param action_array: numpy array or pytorch tensor
    '''
    obs = process_obs(
        tuple_sequence,
        rotation_representation=obs_rotation_representation,
        stored_rotation_representation=stored_obs_rotation_representation,
        n_frame_stacks=n_frame_stacks,
        load_tensor=load_tensor,
        device=device
    )
    act = process_action(
        tuple_sequence,
        rotation_representation=act_rotation_representation,
        stored_rotation_representation=stored_act_rotation_representation,
        n_frame_stacks=n_frame_stacks,
        load_tensor=load_tensor,
        device=device,
        grip_in_action=grip_in_action,
        sts_switch_in_action=sts_switch_in_action
    )
    return obs, act

def load_model(model_config, path=None, mode="eval", device="cuda:0", ex_obs=None, ex_act=None):
    """Load the models based on args."""
    if path is not None:
        print("Loading models in path: ", path)

    # Automatically get obseration and action shapes from an examples, if it's not
    # already saved. When loading for training, generated from obs, when loaded
    # for control, generated from saved values.
    if hasattr(model_config, "obs_key_list") and not hasattr(model_config, "obs_key_shapes_dict"):
        with open_dict(model_config):
            model_config.obs_key_shapes_dict = dict()
            for k in model_config.obs_key_list:
                model_config.obs_key_shapes_dict[k] = list(ex_obs[k].shape[1:])
            if ex_act is not None:
                model_config.act_dim = ex_act.shape[1]

    model = instantiate(model_config)
    model = model.to(device=device)

    if path is not None:
        try:
            model.load_state_dict(
                torch.load(path, map_location=device)
            )
        except Exception as e:
            print(e)

    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
    else:
        raise NotImplementedError()

    return model

def create_dataloaders(dataset_config, random_seed, n_batch, n_worker, val_split):
    """Return training and validation dataloaders."""
    def _init_fn(worker_id):
        np.random.seed(int(random_seed))

    dataset = instantiate(dataset_config)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(
        dataset,
        batch_size=n_batch,
        num_workers=n_worker,
        sampler=train_sampler,
        worker_init_fn=_init_fn
    )

    if len(val_indices) > 0:
        valid_sampler = SubsetRandomSampler(val_indices)
        val_loader = DataLoader(
            dataset,
            batch_size=n_batch,
            num_workers=n_worker,
            sampler=valid_sampler,
            worker_init_fn=_init_fn
        )
    else:
        val_loader = None

    return train_loader, val_loader

def set_seed(seed):
    """Set the same random seed for all sources of rng?"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def MultivariateNormalDiag(loc, scale, batch_dim=1):
    """Returns a diagonal multivariate normal Torch distribution."""
    return Independent(Normal(loc, scale), batch_dim)

def MixtureGaussianDiag(categorical_logits, loc, scale, batch_dim=1):
    """Returns a mixture of diagonal Gaussians."""
    return MixtureSameFamily(Categorical(logits=categorical_logits), Independent(Normal(loc, scale), batch_dim))

def add_weight_decay(model, weight_decay=0.1):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LSTM, torch.nn.GRU)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if "bias" in fpn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif "weight" in fpn and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif "weight" in fpn and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return optim_groups

def exit_if_learning_complete(save_dir, cfg, dataset_config):
    # first get the total dataset size
    dataset_dir = dataset_config.data_dirs[0]
    dataset_params_path = os.path.join(dataset_dir, '..', 'dataset_parameters.json')

    if not os.path.exists(dataset_params_path):
        return False
    with open(dataset_params_path, 'r') as f:
        dataset_params = json.load(f)

    # now get number of batches (that would be in loader)
    ep_lens = dataset_params['ep_lens']
    valid_ep_lens = ep_lens[:dataset_config.n_max_episodes]
    total_data = sum(valid_ep_lens)
    total_train_data = (1.0 - cfg.val_split) * total_data
    total_number_batches = math.ceil(total_train_data / cfg.n_batches)

    if cfg.training_limit == 'epochs':
        n_epochs = cfg.n_epochs
    if cfg.training_limit == 'grad_updates':
        n_epochs = math.ceil(cfg.n_grad_updates / total_number_batches)
    else:
        raise NotImplementedError(f"training_limit must be epochs or grad_updates, got {cfg.training_limit}")

    info_file = os.path.join(save_dir, "model-info.json")
    if os.path.exists(info_file):
        with open(os.path.join(save_dir, "model-info.json"), 'r') as f:
            info = json.load(f)
        if info['epoch'] >= n_epochs:
            print(f"Training of model at {save_dir} already complete, {info['epoch']} epochs done, "\
                    f"{n_epochs} requested, exiting.")
            sys.exit(0)
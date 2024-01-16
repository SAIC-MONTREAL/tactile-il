import os
from functools import partial
import math

import torch
import hydra
from omegaconf import OmegaConf

from place_from_pick_learning.trainer import trainer
from place_from_pick_learning.losses import bc_mse_pose_loss, bc_nll_pose_loss
from place_from_pick_learning.utils.learning_utils import (
    set_seed,
    load_model,
    create_dataloaders,
    add_weight_decay,
    exit_if_learning_complete
)

def opt_epoch_nll(loader, model, device, act_indices=None, opt=None, **epoch_vars):
    """A single training epoch with NLL loss."""
    if opt:
        model.train()
    else:
        model.eval()

    running_stats = {
        'total_nll': [],
        'total_mse': [],
        'translation_mse': [],
        'rotation_mse': [],
    }

    if act_indices is None or act_indices['gr'] is not None:
        running_stats['gripper_mse'] = []

    if act_indices is not None and 'sts' in act_indices:
        running_stats['sts_mse'] = []

    for idx, data in enumerate(loader):
        x, u = data

        # since we don't know keys, we'll just grab any
        any_obs = next(iter(x.values())).shape
        n, l = any_obs.shape[0], any_obs.shape[1]

        if u.device != device:
            x = {k:v.to(device=device) for k,v in x.items()}
            u = u.to(device=device)

        nll, (tr_mse, rot_mse, gr_mse, sts_mse)  = bc_nll_pose_loss(
            x,
            u,
            model,
            act_indices,
            return_sample_mse=True,
        )

        nll = torch.sum(nll) / (n * l)
        running_stats["total_nll"].append(nll.item())

        tr_mse = torch.sum(tr_mse) / (n * l)
        rot_mse = torch.sum(rot_mse) / (n * l)

        running_stats["translation_mse"].append(tr_mse.item())
        running_stats["rotation_mse"].append(rot_mse.item())

        total_loss_mse = tr_mse + rot_mse

        if gr_mse is not None:
            gr_mse = torch.sum(gr_mse) / (n * l)
            running_stats["gripper_mse"].append(gr_mse.item())
            total_loss_mse += gr_mse

        if sts_mse is not None:
            sts_mse = torch.sum(sts_mse) / (n * l)
            running_stats["sts_mse"].append(sts_mse.item())
            total_loss_mse += sts_mse

        running_stats["total_mse"].append(total_loss_mse.item())

        if opt:
            opt.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            nll.backward()
            opt.step()
    summary_stats = {f"avg_{k}": sum(v) / len(v) for k, v in running_stats.items()}
    return summary_stats

def opt_epoch_mse(loader, model, device, act_indices=None, opt=None, **epoch_vars):
    """A single training epoch with MSE loss."""
    if opt:
        model.train()
    else:
        model.eval()

    running_stats = {
        'total_mse': [],
        'translation_mse': [],
        'rotation_mse': []
    }

    if act_indices is None or act_indices['gr'] is not None:
        running_stats['gripper_mse'] = []

    if act_indices is not None and 'sts' in act_indices and act_indices['sts']:
        running_stats['sts_mse'] = []


    for idx, data in enumerate(loader):
        x, u = data

        # since we don't know keys, we'll just grab any
        any_obs = next(iter(x.values()))
        n, l = any_obs.shape[0], any_obs.shape[1]

        if u.device != device:
            x = {k:v.to(device=device) for k,v in x.items()}
            u = u.to(device=device)

        (tr_e, rot_e, gr_e, sts_e) = bc_mse_pose_loss(
            x,
            u,
            model,
            act_indices
        )

        tr_e = torch.sum(tr_e) / (n * l)
        rot_e = torch.sum(rot_e) / (n * l)

        running_stats["translation_mse"].append(tr_e.item())
        running_stats["rotation_mse"].append(rot_e.item())

        #TODO: Scale losses to make sense
        total_loss = tr_e + rot_e
        total_loss_opt = tr_e + rot_e

        if gr_e is not None:
            gr_e = torch.sum(gr_e) / (n * l)
            running_stats["gripper_mse"].append(gr_e.item())
            total_loss += gr_e
            total_loss_opt += gr_e

        if sts_e is not None:
            sts_e = torch.sum(sts_e) / (n * l)
            running_stats["sts_mse"].append(sts_e.item())
            total_loss += sts_e
            total_loss_opt += sts_e

        running_stats["total_mse"].append(total_loss.item())

        if opt:
            opt.zero_grad(set_to_none=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_loss_opt.backward()
            opt.step()
    summary_stats = {f"avg_{k}": sum(v) / len(v) for k, v in running_stats.items()}
    return summary_stats

def get_epoch_vars(epoch):
    """
    Returns the extra arguments (**epoch_vars) to
    opt_epoch based on the current epoch (e.g., annealing values).
    """
    return {}

hydra_main_args = dict(config_path='cfgs', config_name='cfg')
if hydra.__version__ != '1.0.6':
    hydra_main_args['version_base'] = None

@hydra.main(**hydra_main_args)
def train_vanilla_bc(cfg):
    # Save and checkpoint dirs
    save_dir = os.getcwd()
    checkpoint_dir = os.path.join(save_dir, "checkpoints/")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Fix random seed
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    set_seed(cfg.random_seed)
    device = torch.device(cfg.device)
    dataset_config = cfg.dataset.dataset_config if hydra.__version__ != '1.0.6' else cfg.dataset_config

    # Exit if complete -- do this before dataloaders because initial data load on lambda blade is slow
    exit_if_learning_complete(save_dir, cfg, dataset_config)

    train_loader, val_loader = create_dataloaders(
        dataset_config=dataset_config,
        random_seed=cfg.random_seed,
        n_batch=cfg.n_batches,
        n_worker=cfg.n_workers,
        val_split=cfg.val_split
    )
    ex_obs = train_loader.dataset[0][0]
    ex_act = train_loader.dataset[0][1]

    #Load model(s)
    model = load_model(
        model_config=cfg.model.model_config if hydra.__version__ != '1.0.6' else cfg.model_config,
        path=cfg.initial_weights_path,
        device=device,
        mode="train",
        ex_obs=ex_obs,
        ex_act=ex_act,
    )

    # Fix weight decay
    params = add_weight_decay(model, weight_decay=cfg.weight_decay)

    # Load optimizer
    opt = torch.optim.AdamW(
        params,
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=0
    )

    # Choose training limiter, set params based on this
    if cfg.training_limit == 'epochs':
        cfg.n_grad_updates = cfg.n_epochs * len(train_loader)
    elif cfg.training_limit == 'grad_updates':
        cfg.n_epochs = math.ceil(cfg.n_grad_updates / len(train_loader))
        cfg.n_scheduler_epochs = math.ceil(cfg.n_scheduler_grad_updates / len(train_loader))
    else:
        raise NotImplementedError(f"training_limit must be epochs or grad_updates, got {cfg.training_limit}")

    # Load scheduler
    if cfg.n_scheduler_epochs > 0:
        sched = torch.optim.lr_scheduler.StepLR(opt, cfg.n_scheduler_epochs, gamma=0.5)
    else:
        sched = None

    # Loss function

    # If available, grab trans, rot, and gr indices from env config
    act_indices = None
    if hasattr(train_loader.dataset, "env_config"):
        act_indices = {}
        ec = train_loader.dataset.env_config
        num_t_dof = sum(ec['valid_act_t_dof'])
        act_indices['tr'] = slice(0, num_t_dof)
        if cfg.act_rotation_representation in {'quat', 'axisa'}:
            num_r_dof = 4
        else:
            num_r_dof = sum(ec['valid_act_r_dof'])
        act_indices['rot'] = slice(num_t_dof, num_t_dof + num_r_dof)
        act_indices['gr'] = slice(num_t_dof + num_r_dof, None) if ec['grip_in_action'] else None
        act_indices['sts'] = slice(num_t_dof + num_r_dof + ec['grip_in_action'], None) if ec['sts_switch_in_action'] else None

    if cfg.action_distribution == "deterministic":
        opt_epoch = partial(opt_epoch_mse, act_indices=act_indices)
    elif cfg.action_distribution in ["gaussian", "mixture"]:
        opt_epoch = partial(opt_epoch_nll, act_indices=act_indices)

    # Save model config - before starting training but after all pre-config/setup
    with open("model-config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # Train BC policy
    trainer(
        model=model,
        train_loader=train_loader,
        opt=opt,
        opt_epoch=opt_epoch,
        get_epoch_vars=get_epoch_vars,
        device=device,
        save_dir=save_dir,
        model_id=cfg.id,
        checkpoint_dir=checkpoint_dir,
        n_epoch=cfg.n_epochs,
        n_checkpoint_epoch=cfg.n_checkpoint_epochs,
        val_loader=val_loader,
        sched=sched
    )

if __name__ == "__main__":
    train_vanilla_bc()
import torch
import time


ACT_INDICES_DEFAULT = dict(
    tr=slice(0, 3),
    rot=slice(3, -1),
    gr=slice(-1, None)
)

def common_pose_loss(a, u, act_indices=None):
    ai = ACT_INDICES_DEFAULT if act_indices is None else act_indices

    tr = torch.sum((a[..., ai['tr']] - u[..., ai['tr']])**2)
    rot = torch.sum((a[..., ai['rot']] - u[..., ai['rot']])**2)

    if ai['gr'] is not None:
        gr = torch.sum((a[..., ai['gr']] - u[..., ai['gr']])**2)
    else:
        gr = None

    if ai['sts'] is not None:
        sts = torch.sum((a[..., ai['sts']] - u[..., ai['sts']])**2)
    else:
        sts = None

    return (tr, rot, gr, sts)


def bc_mse_pose_loss(x, u, model, act_indices=None):
    """
    Compute MSE loss but split over translation,
        rotation and gripper command
    """
    a, _ = model(x)

    return common_pose_loss(a, u, act_indices)


def bc_nll_pose_loss(x, u, model, act_indices=None, return_sample_mse=False):
    """
    Compute loglikelihood loss but split over translation,
        rotation and gripper command
    """

    a, pa_o = model(x)
    nll = -pa_o.log_prob(u)

    mse_tr, mse_rot, mse_gr, mse_sts = common_pose_loss(x, a, act_indices)

    if return_sample_mse:
        return nll, (mse_tr, mse_rot, mse_gr, mse_sts)
    else:
        return nll

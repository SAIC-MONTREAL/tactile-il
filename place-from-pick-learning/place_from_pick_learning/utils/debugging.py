import numpy as np
import torch


def nice_print(prec=4):
    np.set_printoptions(suppress=True, precision=prec)
    torch.set_printoptions(sci_mode=False, precision=prec)


def a2t(tens):
    return tens.detach().cpu().numpy().squeeze()
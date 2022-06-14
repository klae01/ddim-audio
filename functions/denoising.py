from argparse import Namespace

import torch
import torch.nn as nn
from torch import Tensor

from . import utils


@torch.no_grad()
def generalized_steps(
    x: Tensor,
    model: nn.Module,
    a_sqrt: Tensor,
    a_coeff_sqrt: Tensor,
    t: Tensor,
    select_index: list,
    spec: Namespace,
    mapping: Namespace,
):
    num_steps = len(a_sqrt)
    xs = []
    time_push = torch.empty(x.size(0)).type(x.type())
    for index in range(num_steps):
        a_s, a_cs, ct = a_sqrt[-index - 1], a_coeff_sqrt[-index - 1], t[-index - 1]
        if index != 0:
            x.mul_(a_s).add_(y, alpha=a_cs)
        if mapping.log_polar:
            x = utils.angle_normalize(x)

        time_push[...] = ct
        y = model(x, time_push)
        if mapping.gaussian:
            y, sig_y = y
            y.addcmul_(torch.randn_like(y), sig_y)

        x.add_(y, alpha=-a_cs).div_(a_s)

        if (
            select_index is None
            or index in select_index
            or index - num_steps in select_index
        ):
            x_save = utils.log_polar_invert(x, spec) if mapping.log_polar else x
            xs.append(x_save.to("cpu"))
    return xs

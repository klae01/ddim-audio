from argparse import Namespace

import torch
import torch.nn as nn
from torch import Tensor


def log_polar_convert(x: Tensor, config: Namespace) -> Tensor:
    """_summary_

    Args:
        x (Tensor): Torch tensor with shape=[..., 2].
                          This tensor was created from a complex array.
        config (_type_): log_data_spec

    Returns:
        Tensor: concat([log_scale, theta], dim=-1)
    """
    scale = torch.linalg.norm(x, ord=2, axis=-1)
    log_scale = (torch.log(scale + config.eps) - config.mean) / config.std
    return torch.stack(
        [log_scale, torch.acos(x[..., 0] / scale) * torch.sign(x[..., 1])],
        dim=-1,
    )


def log_polar_invert(x: Tensor, config: Namespace) -> Tensor:
    """_summary_

    Args:
        x (Tensor): Torch tensor with shape=[..., 2].
                          concat([log_scale, theta], dim=-1)
        config (_type_): log_data_spec

    Returns:
        Tensor: A real array represent the original complex array
    """
    log_scale = x[..., 0] * config.std + config.mean
    scale = torch.nn.functional.relu(torch.exp(log_scale) - config.eps)
    theta = x[..., 1]
    return torch.stack(
        [scale * torch.cos(theta), scale * torch.sin(theta)],
        dim=-1,
    )


def __angle_range_fit(x: Tensor) -> Tensor:
    # config is log_data_spec
    half_range = torch.pi
    full_range = half_range * 2
    Y = torch.fmod(x + half_range, full_range)
    Z = torch.fmod(Y + full_range, full_range) - half_range
    return Z


def __angle_normalize(x: Tensor) -> Tensor:
    # config is log_data_spec
    std, mean = torch.std_mean(x)
    x.sub_(mean).div_(std)
    return x


@torch.no_grad()
def angle_normalize(x: Tensor) -> Tensor:
    # config is log_data_spec
    angle = x[..., 1].clone()
    for _ in range(4):
        angle = __angle_normalize(angle)
        angle = __angle_range_fit(angle)
    # while True:
    #     angle = __angle_normalize(angle)
    #     r_angle = __angle_range_fit(angle)
    #     print((r_angle != angle).sum().item())
    #     if (r_angle == angle).all().item():
    #         break
    #     angle = r_angle
    x[..., 1] = angle
    return x


@torch.no_grad()
def angle_centering(x: Tensor, move_mean) -> Tensor:
    # config is log_data_spec
    angle = x[..., 1].clone()
    if move_mean:
        for _ in range(4):
            angle = angle - angle.mean()
            angle = __angle_range_fit(angle)
        # while True:
        #     angle = angle - angle.mean()
        #     r_angle = __angle_range_fit(angle)
        #     if (r_angle == angle).all().item():
        #         break
        #     angle = r_angle
    else:
        angle = __angle_range_fit(angle)
    x[..., 1] = angle
    return x

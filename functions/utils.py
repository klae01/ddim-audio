import torch


def log_polar_convert(x: torch.Tensor, config) -> torch.Tensor:
    """_summary_

    Args:
        x (torch.Tensor): Torch tensor with shape=[..., 2].
                          This tensor was created from a complex array.
        config (_type_): log_data_spec

    Returns:
        torch.Tensor: concat([log_scale, theta], dim=-1)
    """
    scale = torch.linalg.norm(x, ord=2, axis=-1)
    log_scale = (torch.log(scale + config.eps) - config.mean) / config.std
    return torch.stack(
        [log_scale, torch.acos(x[..., 0] / scale) * torch.sign(x[..., 1])],
        dim=-1,
    )


def log_polar_invert(x: torch.Tensor, config) -> torch.Tensor:
    """_summary_

    Args:
        x (torch.Tensor): Torch tensor with shape=[..., 2].
                          concat([log_scale, theta], dim=-1)
        config (_type_): log_data_spec

    Returns:
        torch.Tensor: A real array represent the original complex array
    """
    log_scale = x[..., 0] * config.std + config.mean
    scale = torch.nn.functional.relu(torch.exp(log_scale) - config.eps)
    theta = x[..., 1]
    return torch.stack(
        [scale * torch.cos(theta), scale * torch.sin(theta)],
        dim=-1,
    )


def log_polar_noise_processing(x: torch.Tensor, config) -> torch.Tensor:
    # config is log_data_spec
    std, mean = torch.std_mean(x[..., 1])
    x[..., 1].sub_(mean).div_(std)
    return x


def angle_processing(x: torch.Tensor) -> torch.Tensor:
    # config is log_data_spec
    half_range = torch.pi
    full_range = half_range * 2
    Y = torch.fmod(x + half_range, full_range)
    Z = torch.fmod(Y + full_range, full_range) - half_range
    return Z


def log_polar_state_processing(x: torch.Tensor, config) -> torch.Tensor:
    # config is log_data_spec
    x[..., 1] = angle_processing(x[..., 1])
    return x

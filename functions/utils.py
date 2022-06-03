import torch


def log_polar_convert(x: torch.Tensor, config):
    # x shape: [..., 2]
    # config is log_data_spec
    scale = torch.linalg.norm(x, ord=2, axis=-3, keepdims=True)
    log_scale = (torch.log(scale + config.eps) - config.mean) / config.std
    return torch.concat(
        [log_scale, torch.acos(x[..., 0] / scale) * torch.sign(x[..., 1])],
        axis=-1,
    )


def log_polar_invert(x: torch.Tensor, config):
    # x shape: [..., 2]
    log_scale = x[..., 0] * config.std + config.mean
    scale = torch.exp(log_scale)
    theta = x[..., 1]
    return torch.concat(
        [scale * torch.cos(theta), scale * torch.sin(theta)],
        axis=-1,
    )


def log_polar_noise_processing(x: torch.Tensor, config):
    std, mean = torch.std_mean(x[..., 1])
    x[..., 1].sub_(mean).div_(std)


def log_polar_state_processing(x: torch.Tensor, config):
    x[..., 1] = (x[..., 1] + torch.pi).fmod(torch.pi * 2) - torch.pi

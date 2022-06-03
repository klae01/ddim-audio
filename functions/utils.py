import torch


def log_polar_convert(x: torch.Tensor):
    # x shape: [..., 2]
    scale = torch.linalg.norm(x, ord=2, axis=-3, keepdims=True)
    log_scale = torch.log(scale + torch.exp(-16)) / 8 + 1
    print(log_scale.min(), log_scale.max())
    return torch.concat(
        [log_scale, torch.acos(x[..., 0] / scale) * torch.sign(x[..., 1])],
        axis=-1,
    )

def log_polar_invert(x: torch.Tensor):
    # x shape: [..., 2]
    log_scale = x[..., 0]
    scale = torch.exp((log_scale-1)*8)
    theta = x[..., 1]
    return torch.concat(
        [scale * torch.cos(theta), scale * torch.sin(theta)],
        axis=-1,
    )

def log_polar_noise_processing(x: torch.Tensor):
    std, mean = torch.std_mean(x[..., 1])
    x[..., 1].sub_(mean).div_(std)

def log_polar_state_processing(x: torch.Tensor):
    x[..., 1] = (x[..., 1]+torch.pi).fmod(torch.pi*2)-torch.pi

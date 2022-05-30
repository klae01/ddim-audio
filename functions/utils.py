import torch


def conj2polar(x: torch.Tensor):
    # x.shape = [*, 2, -1, -1]
    scale = torch.linalg.norm(x, ord=2, axis=-3, keepdims=True)
    return torch.concat(
        [scale, torch.cos(x[..., 1:, :, :] / scale) * torch.sign(x[..., :1, :, :])],
        axis=-2,
    )


def polar2conj(x: torch.Tensor):
    # x.shape = [*, 2, -1, -1]
    scale = x[..., :1, :, :]
    theta = x[..., 1:, :, :]
    theta = [scale * torch.sin(theta), scale * torch.cos(theta)]

    scale = torch.linalg.norm(x, ord=2, axis=-3, keepdims=True)
    return torch.concat(
        [scale, torch.cos(x[..., 1:, :, :] / scale) * torch.sign(x[..., :1, :, :])],
        axis=-2,
    )

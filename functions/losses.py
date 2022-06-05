import torch
from . import utils


def model_loss_evaluation(
    model,
    x0: torch.Tensor,
    e: torch.Tensor,
    a_sqrt: torch.Tensor,
    a_coeff_sqrt: torch.Tensor,
    config,
):
    detail = {}
    a_sqrt = a_sqrt.view(-1, 1, 1, 1)
    a_coeff_sqrt = a_coeff_sqrt.view(-1, 1, 1, 1)
    x0_lp = utils.log_polar_convert(x0, config)
    e = utils.log_polar_noise_processing(e, config)

    x = x0_lp * a_sqrt + e * a_coeff_sqrt
    y, sig_y = model(x, a_sqrt)

    avg_diff = e - y
    sig_eps = sig_y + 1e-4

    # scalar loss (NLL)
    # Original design: log(std).mean() + (diff / std).square().mean() / 2
    diff = avg_diff[..., 0]
    sig = sig_eps[..., 0]
    detail["loss_scalar"] = torch.log(sig).mean() + (diff / sig).square().mean() / 2
    detail["loss"] = detail["loss_scalar"]

    # angular loss (NLL)
    # Original design: log(std).mean() + ((diff mod 2 pi) / std).square().mean() / 2
    diff = avg_diff[..., 1]
    sig = sig_eps[..., 1]
    with torch.no_grad():
        target_diff = utils.angle_processing(diff)
        moving_diff = diff - target_diff
    diff = diff - moving_diff
    detail["loss_angular"] = torch.log(sig).mean() + (diff / sig).square().mean() / 2
    detail["loss"] = detail["loss"] + detail["loss_angular"]

    with torch.no_grad():
        x_hat = utils.log_polar_invert(
            x0_lp + (a_coeff_sqrt / a_sqrt) * (e - y), config
        )

        signal = x0.norm(dim=-1, p=2)
        noise = x_hat.norm(dim=-1, p=2) - signal
        detail["SNR"] = 10 * (
            signal.square().mean().clamp(min=1e-20).log10()
            - noise.square().mean().clamp(min=1e-20).log10()
        )

    return detail

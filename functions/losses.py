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
    if config:
        x0_lp = utils.log_polar_convert(x0, config)
        x0_lp = utils.angle_centering(x0_lp, move_mean=True)
        e = utils.angle_normalize(e)
        x = x0_lp * a_sqrt + e * a_coeff_sqrt
    else:
        x = x0 * a_sqrt + e * a_coeff_sqrt

    y, sig_y = model(x, a_sqrt)

    avg_diff = e - y
    sig_eps = sig_y + 1e-4

    if config:
        # scalar loss (NLL)
        # Original design: log(std).mean() + (diff / std).square().mean() / 2
        diff = avg_diff[..., 0]
        sig = sig_eps[..., 0]
        loss = torch.log(sig).mean() + (diff / sig).square().mean() / 2
        detail["loss_scalar"] = loss.detach().clone()
        detail["loss"] = loss

        # angular loss (NLL)
        # Original design: log(std).mean() + ((diff mod 2 pi) / std).square().mean() / 2
        diff = avg_diff[..., 1]
        sig = sig_eps[..., 1]
        with torch.no_grad():
            target_diff = utils.angle_centering(diff.clone(), move_mean=False)
            moving_diff = diff - target_diff
        diff = diff - moving_diff
        loss = torch.log(sig).mean() + (diff / sig).square().mean() / 2
        detail["loss_angular"] = loss.detach().clone()
        detail["loss"] += loss
    else:
        detail["loss"] = (
            torch.log(sig_eps).mean() + (avg_diff / sig_eps).square().mean() / 2
        )

    with torch.no_grad():
        x_hat = x0_lp + (a_coeff_sqrt / a_sqrt) * (e - y)
        if config:
            x_hat = utils.log_polar_invert(x_hat, config)

        signal = x0.norm(dim=-1, p=2)
        noise = x_hat.norm(dim=-1, p=2) - signal
        detail["SNR"] = 10 * (
            signal.square().mean().clamp(min=1e-20).log10()
            - noise.square().mean().clamp(min=1e-20).log10()
        )

    return detail

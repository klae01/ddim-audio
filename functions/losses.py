import torch
from . import utils


def model_loss_evaluation(
    model,
    x0: torch.Tensor,
    e: torch.Tensor,
    a_sqrt: torch.Tensor,
    config,
):
    a_sqrt = a_sqrt.view(-1, 1, 1, 1)
    a = a_sqrt.square()
    x0_lp = utils.log_polar_convert(x0, config)
    e = utils.log_polar_noise_processing(e, config)

    x = x0_lp * a_sqrt + e * (1 - a).sqrt()
    y, sig_y = model(x, a_sqrt)

    avg_diff = e - y
    sig_eps = sig_y + 1e-3

    # scalar loss (NLL)
    diff = avg_diff[..., 0]
    sig = sig_eps[..., 0]
    loss = torch.log(sig).mean() + (diff / sig).square().mean() / 2

    # angular loss (NLL)
    diff = avg_diff[..., 1]
    sig = sig_eps[..., 1]
    var = sig.square()
    loss += torch.log1p(-torch.exp(-var / 2) * torch.cos(diff)).mean()

    with torch.no_grad():
        x_hat = utils.log_polar_invert(x0_lp + (1 / a - 1).sqrt() * (e - y), config)

        signal = x0.norm(dim=-1, p=2)
        noise = x_hat.norm(dim=-1, p=2) - signal
        SNR = 10 * (
            signal.square().mean().clamp(min=1e-20).log10()
            - noise.square().mean().clamp(min=1e-20).log10()
        )

    return {"loss": loss, "SNR": SNR}

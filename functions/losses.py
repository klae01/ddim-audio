import torch
from . import utils

def model_loss_evaluation(
    model,
    x0: torch.Tensor,
    e: torch.Tensor,
    a_sqrt: torch.Tensor,
):
    a = a_sqrt.square()
    x0_lp = utils.log_polar_convert(x0)
    e = utils.log_polar_noise_processing(e)

    x = x0_lp * a_sqrt + e * (1 - a).sqrt()
    y, sig_y = model(x, a_sqrt)

    # scalar loss (NLL)
    a = e[..., 0]
    b = y[..., 0]
    sig = sig_y[..., 0]
    var = sig.square()
    loss = torch.log1p(-torch.exp(-var/2) * torch.cos(a - b)).mean()

    # angular loss (NLL direct)
    a = e[..., 1]
    b = y[..., 1]
    sig = sig_y[..., 1] + 1e-3
    loss += torch.log(sig).mean() + ((a-b)/sig).square().mean()/2
    
    with torch.no_grad():
        x_hat = utils.log_polar_invert(x0_lp + (1 / a - 1).sqrt() * (e - y))

        signal = x0.norm(dim=-1, p=2)
        noise = x_hat.norm(dim=-1, p=2) - signal
        SNR = 10 * (
            signal.square().mean().clamp(min=1e-20).log10()
            - noise.square().mean().clamp(min=1e-20).log10()
        )

    return {"loss": loss, "SNR": SNR}

import torch


def noise_estimation_loss(
    model,
    x0: torch.Tensor,
    t: torch.LongTensor,
    e: torch.Tensor,
    a: torch.Tensor,
):
    a = a.index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    y = model(x, t.long())

    with torch.no_grad():
        x_hat = x0 + (1 / a - 1).sqrt() * (e - y)
        signal = x0.norm(dim=-3, p=2)
        noise = x_hat.norm(dim=-3, p=2) - signal
        SNR = 10 * (
            signal.square().mean().clamp(min=1e-20).log10()
            - noise.square().mean().clamp(min=1e-20).log10()
        )

    loss = (e - y).square().sum(dim=(1, 2, 3)).mean()
    return {"loss": loss, "SNR": SNR}


loss_registry = {
    "simple": noise_estimation_loss,
}

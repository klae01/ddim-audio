import torch
from . import utils


@torch.no_grad()
def generalized_steps(
    x: torch.Tensor, model, a_sqrt, a_coeff_sqrt, select_index, config
):
    num_steps = len(a_sqrt)
    xs = []
    alpha_push = torch.empty(x.size(0)).type(x.type())
    for index in range(num_steps):
        a_s, a_cs = a_sqrt[-index - 1], a_coeff_sqrt[-index - 1]
        if index != 0:
            x.mul_(a_s).add_(y, alpha=a_cs)
        alpha_push[...] = a_s
        if config:
            x = utils.angle_normalize(x)
        y, sig_y = model(x, alpha_push)
        # x.addcmul_(torch.randn_like(mu_y), sig_y, value=-a_cs).div_(a_s)
        y.addcmul_(torch.randn_like(y), sig_y)
        x.add_(y, alpha=-a_cs).div_(a_s)

        if (
            select_index is None
            or index in select_index
            or index - num_steps in select_index
        ):
            xs.append((utils.log_polar_invert(x, config) if config else x).to("cpu"))
    return xs

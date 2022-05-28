import torch


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, alpha, select_index, **kwargs):
    with torch.no_grad():
        alpha = [1.] + alpha.to("cpu", torch.float32).numpy().tolist()

        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        xt = x.type("torch.cuda.FloatTensor")
        t = torch.zeros(n).type("torch.cuda.LongTensor")

        for index, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t[...] = i
            at = alpha[int(i)+1]
            at_next = alpha[int(j)+1]

            et = model(xt, t.long())
            xt.add_(et, alpha = -(1 - at) ** 0.5).div_(at**0.5)

            if (
                select_index is None
                or index in select_index
                or index - len(seq) in select_index
            ):
                x0_preds.append(xt.to("cpu"))

            c1 = (
                kwargs.get("eta", 0)
                * ((1 - at / at_next) * (1 - at_next) / (1 - at)) ** 0.5
            )
            c2 = ((1 - at_next) - c1**2) ** 0.5
            xt.mul_(at_next**0.5).add_(et, alpha=c2).add_(
                torch.randn_like(x), alpha=c1
            )

            if (
                select_index is None
                or index in select_index
                or index - len(seq) in select_index
            ):
                xs.append(xt.to("cpu"))

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, select_index, **kwargs):
    if select_index is not None:
        raise NotImplementedError(
            "Specifying select_index is not implemented in ddpm_steps."
        )
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to("cuda")

            output = model(x, t.long())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to("cpu"))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e
                + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to("cpu"))
    return xs, x0_preds

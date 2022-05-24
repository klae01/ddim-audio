import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(config, parameters):
    if config.optimizer == "Adam":
        return optim.Adam(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.beta,
            amsgrad=config.amsgrad,
            eps=config.eps,
        )
    elif config.optimizer == "AdamW":
        return optim.AdamW(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.beta,
            amsgrad=config.amsgrad,
            eps=config.eps,
        )
    elif config.optimizer == "AdaBelief":
        import sys

        sys.path.append("External/step-clip-optimizer")
        from clip_opt import AdaBelief

        return AdaBelief(
            parameters,
            lr=config.lr,
            betas=config.beta,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad,
            weight_decouple=True,
            fixed_decay=False,
            rectify=False,
            clip_step=config.clip_step,
            norm_ord=config.norm_ord,
        )
    elif config.optimizer == "RMSProp":
        return optim.RMSprop(
            parameters, lr=config.lr, weight_decay=config.weight_decay
        )
    elif config.optimizer == "SGD":
        return optim.SGD(parameters, lr=config.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            "Optimizer {} not understood.".format(config.optimizer)
        )


def get_scheduler(config, optimizer):
    return LambdaLR(
        optimizer,
        lambda step: min(
            ((1 + step) / config.warmup) ** -0.5, (1 + step) / config.warmup
        ),
    )

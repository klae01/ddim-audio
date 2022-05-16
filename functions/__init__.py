import torch.optim as optim


def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'AdaBelief':
        import sys
        sys.path.append('External/step-clip-optimizer')
        from clip_opt import AdaBelief
        return AdaBelief(parameters, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                 weight_decay=config.optim.weight_decay, amsgrad=config.optim.amsgrad, weight_decouple=True, fixed_decay=False, rectify=False,
                 clip_step = config.optim.clip_step, norm_ord = config.optim.norm_ord)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))

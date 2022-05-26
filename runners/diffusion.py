import os
import sys

sys.path.append("External")

import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from PIL import Image
from scipy.io.wavfile import write as WAV_write

from UPU.signal.denoise import denoise_2d
from SST.utils.wav2img import limit_length_img, pfft2img, pfft2wav
from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer, get_scheduler
from functions.losses import loss_registry
from datasets import get_dataset
from functions.ckpt_util import get_ckpt_path
from utils import dict2namespace


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class parameter_option:
    config = {}
    params = []


def classify_group(config, model):
    param_top_level = {}
    param_group = {}
    for group_name, sub_config in vars(config).items():
        param_group[group_name] = []
        sub_config = vars(sub_config)
        for I in sub_config.pop("top_level_name"):
            param_top_level[I] = group_name
        param_group[group_name] = parameter_option()
        param_group[group_name].config = dict2namespace(sub_config)

    for name, param in model.named_parameters():
        top_level_name = name.split(".")[0]
        group_name = param_top_level.get(top_level_name, "default")
        param_group[group_name].params.append(param)

    return param_group


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        alphas = np.concatenate([[1], 1.0 - betas], axis=-1)
        alphas = torch.from_numpy(alphas).type(self.config.model.dtype)
        betas = self.betas = torch.from_numpy(betas).type(self.config.model.dtype)
        self.num_timesteps = betas.shape[0]

        self.alphas = alphas.cumprod(dim=0)
        alphas_cumprod = self.alphas[1:]
        alphas_cumprod_prev = self.alphas[:-1]
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
        if self.config.model.dtype:
            self.alphas = self.alphas.type(self.config.model.dtype)
            self.betas = self.betas.type(self.config.model.dtype)

    def train_step(
        self, model, x, optimizers, schedulers, grad_group, ema_helper, step, epoch
    ):
        n = x.size(0)
        model.train()

        x = x.to(self.device)
        e = torch.randn_like(x)
        a = self.alphas

        # antithetic sampling
        t = torch.randint(low=0, high=self.num_timesteps, size=((n + 1) // 2,)).to(
            self.device
        )
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        loss = loss_registry[self.config.model.type](model, x, t, e, a)

        self.config.tb_logger.add_scalar("loss", loss.item(), global_step=step)

        loggings = {
            "step": step,
            "loss": loss.item(),
        }
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        loss.backward()

        for name, p_opt in grad_group.items():
            if p_opt.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    p_opt.params,
                    p_opt.config.grad_clip,
                )

        for name, optimizer in optimizers.items():
            step_output = optimizer.step()
            if type(step_output) is dict:
                loggings.update(
                    {
                        f"{K}_{name}": V
                        for K, V in step_output.items()
                        if V is not None and K != "loss"
                    }
                )
        for scheduler in schedulers.values():
            scheduler.step()

        logging.info(
            ", ".join(
                (f"{K}: {V:.4f}" if type(V) is float else f"{K}: {V}")
                for K, V in loggings.items()
            )
        )

        if self.config.model.ema:
            ema_helper.update(model)

        if step % self.config.training.snapshot_freq == 0 or step == 1:
            states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
            if self.config.model.ema:
                states.append(ema_helper.state_dict())

            torch.save(
                states,
                os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
            )
            torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

    def train(self):
        assert (self.config.training.n_epochs is not None) != (
            self.config.training.n_iters is not None
        )

        dataset, test_dataset = get_dataset(self.args, self.config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )
        model = Model(self.config)

        model = model.to(self.device)
        # model = torch.nn.DataParallel(model)

        optimizers = {}
        schedulers = {}
        param_group = classify_group(self.config.optimization.optimizer, model)
        for name, p_opt in param_group.items():
            optimizers[name] = optimizer = get_optimizer(p_opt.config, p_opt.params)
            scheduler = get_scheduler(p_opt.config, optimizer)
            if scheduler:
                schedulers[name] = scheduler

        grad_group = {}
        param_group = classify_group(self.config.optimization.grad_norm, model)
        for name, p_opt in param_group.items():
            grad_group[name] = p_opt

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = dict(
                zip(
                    ["model", "optimizer", "epoch", "step", "ema_helper"],
                    torch.load(os.path.join(self.args.log_path, "ckpt.pth")),
                )
            )
            model.load_state_dict(states["model"])

            states["optimizer"]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states["optimizer"])
            start_epoch = states["epoch"]
            step = states["step"]
            if self.config.model.ema:
                ema_helper.load_state_dict(states["ema_helper"])
            del states

        if self.config.training.n_epochs is not None:
            for epoch in range(start_epoch, self.config.training.n_epochs):
                for x, y in train_loader:
                    step += 1
                    self.train_step(
                        model,
                        x,
                        optimizers,
                        schedulers,
                        grad_group,
                        ema_helper,
                        step,
                        epoch,
                    )
        else:
            epoch = start_epoch
            while step < self.config.training.n_iters:
                for x, y in train_loader:
                    step += 1
                    self.train_step(
                        model,
                        x,
                        optimizers,
                        schedulers,
                        grad_group,
                        ema_helper,
                        step,
                        epoch,
                    )
                    if step >= self.config.training.n_iters:
                        break
                epoch += 1

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            # model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
            del states
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            # model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence is not None:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model, select_index=[-1])[0]
                x = inverse_data_transform(
                    config, x, as_uint8=(self.config.data.dataset not in ["AUDIO"])
                )

                for i in range(n):
                    path = os.path.join(self.args.image_folder, f"{img_id}")
                    if self.config.data.dataset == "AUDIO":
                        raise NotImplementedError(
                            "sample_fid with AUDIO dataset is not implemented"
                        )
                    else:
                        Image.fromarray(x[i]).save(path + ".png")
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.model.channels,
            config.model.t_size,
            config.model.f_size,
            device=self.device,
        )

        if self.args.sequence in [-1, 0]:
            index = range(self.args.timesteps)
        else:
            index = np.linspace(
                1, self.args.timesteps, self.args.sequence, dtype=np.int32
            )
            index = set((self.args.timesteps - index).tolist())

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            x_, x = self.sample_image(x, model, select_index=index)

        if self.config.sampling.denoise:
            x = [denoise_2d(y) for y in x]
        x = [y.permute(0, 3, 2, 1).to("cpu").numpy() for y in x]
        digits = np.ceil(np.log10(len(x) + 1)).astype(np.int32).tolist()

        for i in range(len(x)):
            for j, img in enumerate(x[i]):
                path = os.path.join(self.args.image_folder, f"{j}_{i:0{digits}d}")
                if self.config.data.dataset == "AUDIO":
                    Image.fromarray(limit_length_img(pfft2img(img))).save(path + ".png")
                    wav = pfft2wav(
                        img,
                        self.config.sampling.virtual_samplerate,
                        dtype=np.int32,
                        HPI=self.config.sampling.HPI,
                    )
                    WAV_write(
                        path + ".wav",
                        self.config.data.dataset_kwargs.virtual_samplerate,
                        wav,
                    )
                else:
                    Image.fromarray(img).save(path + ".png")

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model, select_index=[-1])[0])
        x = inverse_data_transform(
            config,
            torch.cat(xs, dim=0),
            as_uint8=(self.config.data.dataset not in ["AUDIO"]),
        )
        digits = np.ceil(np.log10(x.size(0) + 1)).astype(np.int32).tolist()
        for i in range(x.size(0)):
            path = os.path.join(self.args.image_folder, f"{i:0{digits}d}")
            if self.config.data.dataset == "AUDIO":
                raise NotImplementedError(
                    "sample_interpolation with AUDIO dataset is not implemented"
                )
            else:
                Image.fromarray(x[i]).save(path + ".png")

    def sample_image(self, x, model, select_index=None):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(
                x, seq, model, self.alphas, eta=self.args.eta, select_index=select_index
            )
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas, select_index=select_index)
        else:
            raise NotImplementedError
        return x

    def test(self):
        pass

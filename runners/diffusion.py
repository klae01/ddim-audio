import glob
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.utils.data as data
import tqdm
from datasets import get_dataset
from functions import get_optimizer, get_scheduler
from functions.losses import model_loss_evaluation
from models.diffusion import Model
from models.ema import EMAHelper
from PIL import Image
from scipy.io.wavfile import write as WAV_write
from scipy.interpolate import interp1d

sys.path.append("External")
from SST.utils import config as SST_config
from SST.utils.wav2img import limit_length_img, pfft2img, pfft2wav
from UPU.signal.denoise import denoise_2d
from utils import dict2namespace


class parameter_option:
    def __init__(self):
        self.config = {}
        self.params = []
        self.named_params = {}


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
        param_group[group_name].named_params[name] = param

    return {K: V for K, V in param_group.items() if V.params}


class Diffusion(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.mapping = config.mapping

    def build_variable_from_beta(self):
        # Beta Derived Variable Construction
        self.alphas = 1 - self.betas
        self.alphas_cumprod = self.alphas.cumprod()
        self.num_timesteps = len(self.betas)
        self.alphas_cumprod_interpolation = interp1d(
            self.alphas_cumprod, np.arange(self.num_timesteps)
        )

    def get_mix_ratio(self, time: np.ndarray):
        alphas_cumprod = self.alphas_cumprod_interpolation(time)
        alphas_cumprod_sqrt = np.sqrt(alphas_cumprod)
        alphas_cumprod_coeff_sqrt = np.sqrt(1 - alphas_cumprod)

        return [
            torch.from_numpy(I)
            for I in [alphas_cumprod_sqrt, alphas_cumprod_coeff_sqrt]
        ]

    def train_step(
        self, model, x, optimizers, schedulers, grad_group, ema_helper, step, epoch
    ):
        n = x.size(0)
        e = torch.randn_like(x)
        t = torch.rand(n, device="cpu") * (self.num_timesteps - 1)
        a, a_coeff, t = [
            I.type(self.config.model.dtype) for I in [*self.get_mix_ratio(t.numpy()), t]
        ]

        process_info = model_loss_evaluation(
            model, x, e, a, a_coeff, t, spec=self.log_data_spec, mapping=self.mapping
        )

        loss = process_info["loss"]
        for K, V in process_info.items():
            self.config.tb_logger.add_scalar(K, V.item(), global_step=step)
        for K, V in model.named_parameters():
            if "rezero" in K:
                if V.shape:
                    self.config.tb_logger.add_scalar(
                        f"{K}_std", V.std().item(), global_step=step
                    )
                else:
                    self.config.tb_logger.add_scalar(f"{K}", V.item(), global_step=step)

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
                self.log_data_spec,
                ema_helper,
            ]

            torch.save(
                states,
                os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
            )
            torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

    def train(self):
        assert (self.config.training.n_epochs is not None) != (
            self.config.training.n_iters is not None
        )
        # build beta & alpha
        self.betas = eval(self.config.diffusion.betas)
        self.build_variable_from_beta()

        # config dataset
        dataset, test_dataset, log_data_spec = get_dataset(
            self.args, self.config.data, self.config.mapping
        )
        train_loader = data.DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )
        self.log_data_spec = (
            dict2namespace(log_data_spec) if log_data_spec else log_data_spec
        )

        # config model
        model = Model(self.config)

        # config optimizer & scheduler
        optimizers = {}
        schedulers = {}
        param_group = classify_group(self.config.optimization.optimizer, model)
        for name, p_opt in param_group.items():
            params = {
                "rezero": {"params": [], "weight_decay": 0},
                "default": {"params": []},
            }

            for name, param in p_opt.named_params.items():
                group_name = "default"
                for K, V in params.items():
                    if K in name:
                        group_name = K
                        break
                params[group_name]["params"].append(param)

            optimizers[name] = optimizer = get_optimizer(
                p_opt.config, list(params.values())
            )
            scheduler = get_scheduler(p_opt.config, optimizer)
            if scheduler:
                schedulers[name] = scheduler

        # config grad norm group
        grad_group = {}
        param_group = classify_group(self.config.optimization.grad_norm, model)
        for name, p_opt in param_group.items():
            grad_group[name] = p_opt

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        # prepare training
        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = dict(
                zip(
                    [
                        "model",
                        "optimizer",
                        "epoch",
                        "step",
                        "log_data_spec",
                        "ema_helper",
                    ],
                    torch.load(os.path.join(self.args.log_path, "ckpt.pth")),
                )
            )
            model.load_state_dict(states["model"])
            optimizer.load_state_dict(states["optimizer"])
            start_epoch = states["epoch"]
            step = states["step"]
            self.log_data_spec = states["log_data_spec"]
            if self.config.model.ema:
                ema_helper.load_state_dict(states["ema_helper"])
            del states

        # training
        step_args = [optimizers, schedulers, grad_group, ema_helper, step, epoch]
        if self.config.training.n_epochs is not None:
            for epoch in range(start_epoch, self.config.training.n_epochs):
                model.train()
                for x, y in train_loader:
                    step += 1
                    self.train_step(model, x, *step_args)
        else:
            epoch = start_epoch
            while step < self.config.training.n_iters:
                model.train()
                for x, y in train_loader:
                    step += 1
                    self.train_step(model, x, *step_args)
                    if step >= self.config.training.n_iters:
                        break
                epoch += 1

    def sample(self):
        # build beta & alpha
        self.betas = eval(self.config.diffusion.betas)
        self.build_variable_from_beta()

        # config model
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
            model.load_state_dict(states[0], strict=True)
            self.log_data_spec = states[4]

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
            del states
        else:
            raise NotImplementedError("unknown option for pretrained model")

        model.eval()

        if self.args.sequence is not None:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_sequence(self, model):
        config = self.config
        sst_config = SST_config(**vars(self.config.data.dataset_kwargs))

        x = torch.randn(
            config.sampling.num_samples,
            config.sampling.t_size,
            config.model.io.f_size,
            config.model.io.channels,
        ).type(self.config.model.dtype)

        if self.args.sequence in [-1, 0]:
            index = range(self.args.timesteps)
        else:
            index = np.linspace(
                1, self.args.timesteps, self.args.sequence, dtype=np.int32
            )
            index = set((self.args.timesteps - index).tolist())

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            x = self.sample_image(x, model, select_index=index)

        if self.config.sampling.denoise:
            x = (
                y.to(self.config.sampling.denoise_device).permute(0, 3, 1, 2) for y in x
            )
            x = [denoise_2d(y).permute(0, 2, 3, 1).to("cpu").numpy() for y in x]
        else:
            x = [y.numpy() for y in x]
        digits = np.ceil(np.log10(len(x) + 1)).astype(np.int32).tolist()

        for i in range(len(x)):
            for j, img in enumerate(x[i]):
                path = os.path.join(self.args.image_folder, f"{j}_{i:0{digits}d}")
                if self.config.data.dataset == "AUDIO":
                    Image.fromarray(limit_length_img(pfft2img(img, sst_config))).save(
                        path + ".png"
                    )
                    wav = pfft2wav(img, sst_config)
                    WAV_write(
                        path + ".wav",
                        self.config.data.dataset_kwargs.samplerate,
                        wav,
                    )
                else:
                    Image.fromarray(img).save(path + ".png")

    def sample_image(self, x, model, select_index=None):
        if self.args.skip_type == "uniform":
            t = np.linspace(0, self.num_timesteps - 1, self.args.timesteps)
        elif self.args.skip_type == "quad":
            t = (
                np.linspace(
                    0, np.sqrt((self.num_timesteps - 1) * 0.8), self.args.timesteps
                )
                ** 2
            )
        else:
            raise NotImplementedError

        if self.args.sample_type == "generalized":
            from functions.denoising import generalized_steps

            a, a_coeff, t = [
                I.type(self.config.model.dtype)
                for I in [*self.get_mix_ratio(t), torch.from_numpy(t)]
            ]

            return generalized_steps(
                x,
                model,
                a,
                a_coeff,
                t,
                select_index=select_index,
                spec=self.log_data_spec,
                mapping=self.mapping,
            )
        else:
            raise NotImplementedError

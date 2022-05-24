import math
import sys

import torch
import torch.nn as nn

sys.path.append("External")

from UPU.layers.normalize.groupnorm import GroupNorm1D


class Residual_Block(nn.Module):
    def __init__(self, *, channels, kernel_size=3):
        super().__init__()
        self.channels = channels

        self.norm = nn.Sequential(*[
            torch.nn.GroupNorm(
                num_groups=4, num_channels=channels, eps=1e-6, affine=True
            )
            for _ in range(4)
        ])
        torch.nn.init.zeros_(self.norm[-1].weight)
        self.norm[-1].register_parameter("bias", None)

        self.conv = nn.Sequential(*[
            torch.nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=True,
            )
            for _ in range(2)
        ])

    def forward(self, input, temb):
        NORM = iter(self.norm)
        CONV = iter(self.conv)
        x = input

        x = next(NORM)(x)
        x = torch.nn.functional.silu(x)
        x = next(NORM)(x)
        x = x + temb[..., None, None]
        x = next(CONV)(x)
        x = torch.nn.functional.silu(x)
        x = next(NORM)(x)
        x = next(CONV)(x)
        x = torch.nn.functional.silu(x)
        x = next(NORM)(x)

        return input + x


@torch.no_grad()
def Add_Encoding(data):
    length = data.shape[-2]
    channel = data.shape[-1]
    position = torch.arange(length, dtype=data.dtype, device=data.device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, channel, 2, dtype=data.dtype, device=data.device)
        * (-math.log(10000.0) / channel)
    )
    x = position * div_term
    data[..., 0::2] += torch.sin(x)
    data[..., 1::2] += torch.cos(x)


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        channel_sz = config.model.transformers.channels
        seq_length = config.model.t_size
        te = torch.zeros(seq_length, channel_sz)
        Add_Encoding(te)
        self.register_buffer("te", te)

        self.norm = nn.Sequential(*[
            torch.nn.LayerNorm(channel_sz, eps=1e-05, elementwise_affine=True)
            for _ in range(1)
        ])
        self.weight = nn.Sequential(*[
            torch.nn.Linear(channel_sz, channel_sz, bias=True) for _ in range(1)
        ])

    def forward(self, input):
        NORM = iter(self.norm)
        WIGT = iter(self.weight)
        x = input

        x = next(WIGT)(x)
        x = torch.nn.functional.silu(x)
        x = next(NORM)(x)

        if x.shape[2] != self.te.shape[1]:
            Add_Encoding(x)
        else:
            x += self.te

        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        raw_channels = config.model.channels

        PR = config.model.preprocessing_residual
        embedding_size = [PR.channels] * len(PR.kernal)
        PR = config.model.postprocessing_residual
        embedding_size += [PR.channels] * len(PR.kernal)
        self.embedding_size = embedding_size
        self.temb = torch.nn.Embedding(
            self.config.diffusion.num_diffusion_timesteps, sum(embedding_size)
        )

        PR = config.model.preprocessing_residual
        self.conv_in = torch.nn.Conv2d(
            raw_channels, PR.channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.preprocessing_residual = nn.Sequential(*[
            Residual_Block(channels=PR.channels, kernel_size=krn) for krn in PR.kernal
        ])

        PR = config.model.postprocessing_residual
        self.postprocessing_residual = nn.Sequential(*[
            Residual_Block(channels=PR.channels, kernel_size=krn) for krn in PR.kernal
        ])
        self.conv_out = torch.nn.Conv2d(
            PR.channels, raw_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        TR = config.model.transformers
        exec(TR.imports)
        self.transformer_embedding = Embedding(config)
        self.transformer = eval(TR.module)(eval(TR.config)(**vars(TR.kwargs)))
        self.fft_mapper = nn.Linear(
            config.model.transformers.channels, PR.channels * config.model.f_size, bias=False
        )

        if self.config.model.dtype:
            self.type(self.config.model.dtype)

    def forward(self, input, t):
        # input: [B, C, T, F]
        # transformer input: [B, T, F*C]
        # transformer output: [B, T, F*C]
        # transformer style: [B, conv_channels, T, F]
        # conv input: [B, C, T, F]

        if self.config.model.transformers.dtype:
            self.transformer_embedding.type(self.config.model.transformers.dtype)
            self.transformer.type(self.config.model.transformers.dtype)
            self.fft_mapper.type(self.config.model.transformers.dtype)
            x = input.type(self.config.model.transformers.dtype)
        else:
            x = input

        x = torch.permute(x, (0, 2, 1, 3))
        x = x.reshape(*x.shape[:2], -1)
        x = self.transformer_embedding(x)
        x = self.transformer(
            x,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
        x = self.fft_mapper(x)
        x = x.view(*x.shape[:2], -1, self.config.model.f_size)
        x = torch.permute(x, (0, 2, 1, 3))
        style = x

        temb = self.temb(t)
        temb = torch.split(temb, self.embedding_size, dim=-1)
        temb = iter(temb)
        x = input.type(self.config.model.dtype)
        x = self.conv_in(x)
        for RES in self.preprocessing_residual:
            x = RES(x, next(temb))
        x = x + style.type(x.type())
        for RES in self.postprocessing_residual:
            x = RES(x, next(temb))
        x = self.conv_out(x)
        return x

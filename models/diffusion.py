import math
import sys

import torch
import torch.nn as nn

sys.path.append("External")

from UPU.layers import Gate


def Normalize(channels, affine=True, group_cnt=None):
    if group_cnt is None:
        group_cnt = max(channels // 16, 1)
    return nn.GroupNorm(group_cnt, channels, eps=1e-8, affine=affine)


class Residual_Block(nn.Module):
    def __init__(self, *, channels, kernel_size=3):
        super().__init__()
        self.channels = channels

        self.norm = nn.Sequential(*[Normalize(channels) for _ in range(3)])
        nn.init.zeros_(self.norm[-1].weight)
        self.norm[-1].register_parameter("bias", None)

        self.conv = nn.Sequential(
            *[
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=bias,
                )
                for bias in [False, True]
            ]
        )

    def forward(self, input, temb):
        NORM = iter(self.norm)
        CONV = iter(self.conv)
        x = input

        x = next(NORM)(x)
        x = nn.functional.silu(x)
        x = next(CONV)(x) + temb[..., None, None]
        x = nn.functional.silu(x)
        x = next(NORM)(x)
        x = next(CONV)(x)
        x = nn.functional.silu(x)
        x = next(NORM)(x)

        return input + x


class Upsample(nn.Module):
    def __init__(self, *, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, *, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


@torch.no_grad()
def get_timestep_embedding(time, embedding_dim):
    assert embedding_dim % 2 == 0
    # 9.210340372 = ln(10000)
    emb = (
        -9.210340372 * torch.linspace(0, 1, embedding_dim // 2, device=time.device)
    ).exp()
    x = time[..., None].float() * emb
    return torch.cat([x.sin(), x.cos()], dim=-1)


class BetaEmbedding(nn.Module):
    def __init__(self, channel_sz):
        super().__init__()
        self.pos_ch = 128
        emb_ch = 512

        self.weight = nn.Sequential(
            nn.Linear(self.pos_ch, emb_ch, bias=True),
            nn.Linear(emb_ch, emb_ch, bias=True),
            nn.Linear(emb_ch, channel_sz, bias=True),
        )

    def forward(self, input):
        WIGT = iter(self.weight)

        x = get_timestep_embedding(input, self.pos_ch)
        x = next(WIGT)(x)
        x = nn.functional.silu(x)
        x = next(WIGT)(x)
        x = nn.functional.silu(x)
        x = next(WIGT)(x)

        return x


class Transformer_Module(nn.Module):
    def __init__(self, io_channels, config):
        # config contains only model.transformers
        super().__init__()

        exec(config.imports)
        self.projection = nn.Sequential(
            nn.Linear(io_channels, config.channels, bias=False),
            nn.LayerNorm(config.channels),
        )
        self.encoder = eval(config.module)(eval(config.config)(**vars(config.kwargs)))
        self.compute_out = nn.Sequential(
            nn.Linear(config.channels, io_channels, bias=False),
            nn.LayerNorm(io_channels),
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.encoder(
            x,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state

        x = self.compute_out(x)
        return x


class G_mapping(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Conv2d(in_features, out_features * 2, 3, 1, 1, bias=True)
        self.weight.bias.data.zero_()

    def forward(self, input):
        x = input
        x = self.weight(x)
        μ, σ = x.chunk(2, 1)
        return μ, torch.sigmoid(σ)


class Model(nn.Module):
    def __init__(self, config):
        # get full config

        self.config = config.model
        self.mapping = config.mapping
        assert len(self.config.ch) == len(self.config.krn) == len(self.config.res)
        super().__init__()

        embedding_size = [
            temb_ch
            for res_cnt, temb_ch in zip(self.config.res, self.config.ch)
            for _ in range(res_cnt)
        ]
        embedding_size = embedding_size + embedding_size[::-1]
        self.embedding_size = embedding_size
        self.temb = BetaEmbedding(sum(embedding_size))

        self.down_modules = nn.ModuleList()
        self.down_modules.append(
            nn.Conv2d(
                self.config.io.channels,
                self.config.ch[0],
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )
        self.up_modules = nn.ModuleList()
        self.up_modules.append(
            nn.Conv2d(
                self.config.ch[0],
                self.config.io.channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            if not self.mapping.gaussian
            else G_mapping(self.config.ch[0], self.config.io.channels)
        )
        for prev_ch, ch, krn, res in zip(
            [-1] + self.config.ch[:-1], self.config.ch, self.config.krn, self.config.res
        ):
            m_list = (
                []
                if prev_ch == -1
                else [Downsample(in_channels=prev_ch, out_channels=ch)]
            )
            m_list += [Residual_Block(channels=ch, kernel_size=krn) for _ in range(res)]
            self.down_modules.append(nn.ModuleList(m_list))

            m_list = (
                []
                if prev_ch == -1
                else [Upsample(in_channels=ch, out_channels=prev_ch)]
            )
            m_list += [Residual_Block(channels=ch, kernel_size=krn) for _ in range(res)]
            self.up_modules.append(nn.ModuleList(m_list[::-1]))
        self.up_modules = self.up_modules[::-1]

        self.transformer = Transformer_Module(
            self.config.ch[-1]
            * (self.config.io.f_size // (2 ** (len(self.config.ch) - 1))),
            self.config.transformers,
        )
        if self.config.dtype:
            self.type(self.config.dtype)

    def forward(self, input, t):
        # input: [B, T, F, C]
        # transformer i/o: [B, T, C * F]
        # conv input: [B, C, T, F]

        if (
            self.config.transformers.dtype
            and self.config.transformers.dtype != self.config.dtype
        ):
            self.transformer.type(self.config.transformers.dtype)

        temb = self.temb(t)
        temb = torch.split(temb, self.embedding_size, dim=-1)
        temb = iter(temb)

        x = input.permute(0, 3, 1, 2)
        # downsampling
        hidden = []
        for lays in self.down_modules:
            if type(lays) != nn.modules.container.ModuleList:
                lays = [lays]

            for lay in lays:
                if type(lay) == Residual_Block:
                    x = lay(x, next(temb))
                else:
                    x = lay(x)
            hidden.append(x)

        # transformer
        type_reserve = x.type()
        if (
            self.config.transformers.dtype
            and self.config.transformers.dtype != self.config.dtype
        ):
            x = x.type(self.config.transformers.dtype)
        B, C, T, F = x.shape
        x = torch.permute(x, (0, 2, 1, 3))
        x = x.reshape(B, T, C * F)
        x = self.transformer(x)
        x = x.reshape(B, T, C, F)
        x = torch.permute(x, (0, 2, 1, 3))
        x = x.type(type_reserve)

        # upsampling
        hidden = iter(hidden[::-1])
        for lays in self.up_modules:
            x = x + next(hidden)
            if type(lays) != nn.modules.container.ModuleList:
                lays = [lays]

            for lay in lays:
                if type(lay) == Residual_Block:
                    x = lay(x, next(temb))
                else:
                    x = lay(x)
        if self.mapping.gaussian:
            return [I.permute(0, 2, 3, 1) for I in x]
        else:
            return x.permute(0, 2, 3, 1)

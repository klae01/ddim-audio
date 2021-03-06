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

        self.norm = nn.Sequential(
            *[
                torch.nn.GroupNorm(
                    num_groups=8, num_channels=channels, eps=1e-6, affine=True
                )
                for _ in range(3)
            ]
        )
        torch.nn.init.zeros_(self.norm[-1].weight)
        self.norm[-1].register_parameter("bias", None)

        self.conv = nn.Sequential(
            *[
                torch.nn.Conv2d(
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
        x = torch.nn.functional.silu(x)
        x = next(CONV)(x) + temb[..., None, None]
        x = torch.nn.functional.silu(x)
        x = next(NORM)(x)
        x = next(CONV)(x)
        x = torch.nn.functional.silu(x)
        x = next(NORM)(x)

        return input + x


class Upsample(nn.Module):
    def __init__(self, *, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, *, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


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


class BetaEmbedding(nn.Module):
    def __init__(self, seq_length, channel_sz):
        super().__init__()
        pos_ch = 128
        emb_ch = 512
        te = torch.zeros(seq_length, pos_ch)
        Add_Encoding(te)
        self.register_buffer("te", te)

        self.weight = nn.Sequential(
            torch.nn.Linear(pos_ch, emb_ch, bias=True),
            torch.nn.Linear(emb_ch, emb_ch, bias=True),
            torch.nn.Linear(emb_ch, channel_sz, bias=True),
        )

    def forward(self, input):
        WIGT = iter(self.weight)

        x = self.te.index_select(0, input)
        x = next(WIGT)(x)
        x = torch.nn.functional.silu(x)
        x = next(WIGT)(x)
        x = torch.nn.functional.silu(x)
        x = next(WIGT)(x)

        return x


class TransformerEmbedding(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.te = None
        self.LayerNorm = nn.LayerNorm(in_channels, eps=config.kwargs.layer_norm_eps)
        self.projection = nn.Linear(in_channels, config.channels)
        self.dropout = nn.Dropout(config.kwargs.hidden_dropout_prob)

    def forward(self, input):
        if self.te == None or self.te.size(0) > input.size(1):
            size = input.size(1)
            size = 2 ** math.ceil(math.log2(size))
            self.te = torch.zeros(
                size, input.size(2), dtype=input.dtype, device=input.device
            )
            Add_Encoding(self.te)

        x = input + self.te[: input.size(1)]

        x = self.LayerNorm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class Transformer_Module(nn.Module):
    def __init__(self, io_channels, config):
        # config contains only model.transformers
        super().__init__()

        exec(config.imports)
        self.embedding = TransformerEmbedding(io_channels, config)
        self.encoder = eval(config.module)(eval(config.config)(**vars(config.kwargs)))
        self.compute_out = nn.Linear(config.channels, io_channels)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(
            x,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state

        x = self.compute_out(x)
        return x


class Model(nn.Module):
    def __init__(self, config):
        # get full config

        self.config = config.model
        assert len(self.config.ch) == len(self.config.krn) == len(self.config.res)
        super().__init__()

        embedding_size = [
            temb_ch
            for res_cnt, temb_ch in zip(self.config.res, self.config.ch)
            for _ in range(res_cnt)
        ]
        embedding_size = embedding_size + embedding_size[::-1]
        self.embedding_size = embedding_size
        self.temb = BetaEmbedding(
            config.diffusion.num_diffusion_timesteps, sum(embedding_size)
        )

        self.down_modules = nn.ModuleList()
        self.down_modules.append(
            torch.nn.Conv2d(
                self.config.channels,
                self.config.ch[0],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.up_modules = nn.ModuleList()
        self.up_modules.append(
            torch.nn.Conv2d(
                self.config.ch[0],
                self.config.channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
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
            * (self.config.f_size // (2 ** (len(self.config.ch) - 1))),
            self.config.transformers,
        )
        if self.config.dtype:
            self.type(self.config.dtype)

    def forward(self, input, t):
        # input: [B, C, T, F]
        # transformer i/o: [B, T, F*C]
        # conv input: [B, C, T, F]

        if (
            self.config.transformers.dtype
            and self.config.transformers.dtype != self.config.dtype
        ):
            self.transformer.type(self.config.transformers.dtype)

        temb = self.temb(t)
        temb = torch.split(temb, self.embedding_size, dim=-1)
        temb = iter(temb)

        x = input
        # downsampling
        hidden = []
        for lays in self.down_modules:
            if type(lays) != torch.nn.modules.container.ModuleList:
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
        x = torch.permute(x, (0, 2, 1, 3))
        shape_reserve = x.shape
        x = x.reshape(*x.shape[:2], -1)
        x = self.transformer(x)
        x = x.reshape(*shape_reserve)
        x = torch.permute(x, (0, 2, 1, 3))
        x = x.type(type_reserve)

        # upsampling
        hidden = iter(hidden[::-1])
        for lays in self.up_modules:
            x = x + next(hidden)
            if type(lays) != torch.nn.modules.container.ModuleList:
                lays = [lays]

            for lay in lays:
                if type(lay) == Residual_Block:
                    x = lay(x, next(temb))
                else:
                    x = lay(x)

        return x

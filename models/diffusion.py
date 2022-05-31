import math
import sys

import torch
import torch.nn as nn
import transformers.models.fnet.modeling_fnet as FNET

sys.path.append("External")

from UPU.layers.normalize.groupnorm import GroupNorm1D


class Residual_Block(nn.Module):
    def __init__(self, *, channels, kernel_size=3):
        super().__init__()
        self.channels = channels

        self.norm = nn.ModuleList(
            [
                nn.GroupNorm(num_groups=8, num_channels=channels, eps=1e-6, affine=True)
                for _ in range(2)
            ]
        )

        self.conv = nn.ParameterList(
            [
                nn.Parameter(torch.empty((channels, channels, 3, 3))),
                nn.Parameter(torch.empty((channels, channels, 3, 3))),
            ]
        )
        for I in self.conv:
            nn.init.kaiming_normal_(I)
        self.rezero = nn.Parameter(torch.zeros(1))

    def forward(self, input, temb):
        NORM = iter(self.norm)
        CONV = iter(self.conv)
        x = input

        x = next(NORM)(x)
        x = nn.functional.silu(x)
        x = nn.functional.conv2d(x, next(CONV), stride=1, padding=1)
        x = x + temb[..., None, None]
        x = nn.functional.silu(x)
        x = next(NORM)(x)
        x = nn.functional.conv2d(x, next(CONV) * self.rezero, stride=1, padding=1)

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
            nn.Linear(pos_ch, emb_ch, bias=True),
            nn.Linear(emb_ch, emb_ch, bias=True),
            nn.Linear(emb_ch, channel_sz, bias=True),
        )

    def forward(self, input):
        WIGT = iter(self.weight)

        x = self.te.index_select(0, input)
        x = next(WIGT)(x)
        x = nn.functional.silu(x)
        x = next(WIGT)(x)
        x = nn.functional.silu(x)
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


class PoolingAttention(nn.Module):
    def __init__(self, in_features: int, attention_features: int, segments: int, max_pool_kernel: int):
        super(PoolingAttention, self).__init__()
        self.attn = nn.Linear(in_features, attention_features * 6)
        self.segments = segments
        self.max_pool_kernel = max_pool_kernel

    def forward(self, inp: torch.Tensor):  # Shape: [Batch, Sequence, Features]
        batch, sequence, features = inp.size()
        assert sequence % self.segments == 0

        qry, key, val, seg, loc, out = self.attn(inp).chunk(6, 2)  # 6x Shape: [Batch, Sequence, AttentionFeatures]
        
        aggregated = qry.mean(1, keepdim=True)  # Shape: [Batch, AttentionFeatures]
        aggregated = torch.einsum("ba,bsa->bs", aggregated, key)  # Shape: [Batch, Sequence]
        aggregated = nn.functional.softmax(aggregated, 1)
        aggregated = torch.einsum("bs,bsa,bza->bza", aggregated, val, out)  # Shape: [Batch, Sequence, AttentionFeatures]

        segment_max_pooled = seg.view(batch, sequence // self.segments, self.segments, -1)
        segment_max_pooled = segment_max_pooled.max(2, keepdim=True)  # Shape: [Batch, PooledSequence, 1, AttentionFeatures]
        segment_max_pooled = segment_max_pooled * out.view(batch, sequence // self.segments, self.segments, -1)  # Shape: [Batch, PooledSequence, PoolSize, AttentionFeatures]
        segment_max_pooled = segment_max_pooled.view(batch, sequence, -1)  # Shape: [Batch, Sequence, AttentionFeatures]
        
        loc = loc.transpose(1, 2)  # Shape: [Batch, AttentionFeatures, Sequence]
        local_max_pooled = nn.functional.max_pool1d(loc, self.max_pool_kernel, 1, self.max_pool_kernel // 2)
        local_max_pooled = local_max_pooled.transpose(1, 2)  # Shape: [Batch, Sequence, AttentionFeatures]
        
        return aggregated + segment_max_pooled + local_max_pooled

class FNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Attention = PoolingAttention(config.hidden_size, config.hidden_size, 16, 3)
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.rezero = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states):
        stream = x = hidden_states
        x = self.Attention(x)
        stream = stream + x * self.rezero

        x = stream
        x = self.dense_1(x)
        x = nn.functional.silu(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        stream = stream + x * self.rezero

        return (stream,)


FNET.FNetLayer = FNetLayer


class Transformer_Module(nn.Module):
    def __init__(self, io_channels, config):
        # config contains only model.transformers
        super().__init__()

        self.embedding = TransformerEmbedding(io_channels, config)
        self.encoder = FNET.FNetEncoder(FNET.FNetConfig(**vars(config.kwargs)))
        self.LayerNorm = nn.LayerNorm(config.channels, eps=config.kwargs.layer_norm_eps)
        self.compute_out = nn.Linear(config.channels, io_channels)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(
            x,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
        x = self.LayerNorm(x)
        x = nn.functional.silu(x)
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
            nn.Conv2d(
                self.config.channels,
                self.config.ch[0],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.up_modules = nn.ModuleList()
        self.up_modules.append(
            nn.Conv2d(
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
        self.rezero = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in self.down_modules]
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
        x = torch.permute(x, (0, 2, 1, 3))
        shape_reserve = x.shape
        x = x.reshape(*x.shape[:2], -1)
        x = self.transformer(x)
        x = x.reshape(*shape_reserve)
        x = torch.permute(x, (0, 2, 1, 3))
        x = x.type(type_reserve)

        # upsampling
        hidden = iter(hidden[::-1])
        rezero = iter(self.rezero)
        for lays in self.up_modules:
            # x = x + next(rezero) * next(hidden)
            x = next(rezero) * x + next(hidden)
            if type(lays) != nn.modules.container.ModuleList:
                lays = [lays]

            for lay in lays:
                if type(lay) == Residual_Block:
                    x = lay(x, next(temb))
                else:
                    x = lay(x)

        return x

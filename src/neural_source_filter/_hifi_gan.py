from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from TTS.vocoder.models.hifigan_generator import LRELU_SLOPE, HifiganGenerator

from ._model import SineGen, SourceModuleHnNSF


class NSFHifiganGenerator(HifiganGenerator):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resblock_type: str,
        resblock_dilation_sizes: Sequence[Sequence[int]],
        resblock_kernel_sizes: Sequence[int],
        upsample_kernel_sizes: Sequence[int],
        upsample_initial_channel: int,
        upsample_factors: Sequence[int],
        inference_padding: int = 5,
        cond_channels: int = 0,
        conv_pre_weight_norm: bool = True,
        conv_post_weight_norm: bool = True,
        conv_post_bias: bool = True,
        sine_gen: SineGen = SineGen(22050),
    ):
        super().__init__(
            in_channels,
            out_channels,
            resblock_type,
            resblock_dilation_sizes,
            resblock_kernel_sizes,
            upsample_kernel_sizes,
            upsample_initial_channel,
            upsample_factors,
            inference_padding,
            cond_channels,
            conv_pre_weight_norm,
            conv_post_weight_norm,
            conv_post_bias,
        )

        self.f0_upsamp = nn.Upsample(scale_factor=np.prod(upsample_factors))
        self.m_source = SourceModuleHnNSF(sine_gen)
        self.noise_convs = nn.ModuleList()
        for i in range(len(self.ups)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < self.num_upsamples:
                stride_f0 = np.prod(upsample_factors[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None = None,
        f0: torch.Tensor = torch.Tensor([]),
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): feature input tensor.
            g (Tensor): global conditioning input tensor.
            f0 (Tensor): fundamental frequency input tensor.

        Returns:
            Tensor: output waveform.

        Shapes:
            x: [B, C, T]
            g: [B, C_g, T]
            f0: [B, 1, T] or [B, T]
        """
        if f0.dim() == 2:  # modified
            f0 = f0.unsqueeze(1)  # modified

        f0 = self.f0_upsamp(f0).transpose(1, 2)  # modified
        har_source = self.m_source(f0)[0].transpose(1, 2)  # modified

        o = self.conv_pre(x)
        if hasattr(self, "cond_layer"):
            if g is None:  # modified
                raise RuntimeError(
                    "Global conditioning is not given although cond_channels > 0."
                )  # modified
            o = o + self.cond_layer(g)
        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o) + self.noise_convs[i](har_source)  # modified
            z_sum: torch.Tensor | None = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i * self.num_kernels + j](o)
                else:
                    z_sum += self.resblocks[i * self.num_kernels + j](o)
            if z_sum is None:
                raise RuntimeError("No residual blocks found.")
            o = z_sum / self.num_kernels
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

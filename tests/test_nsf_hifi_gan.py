from unittest import TestCase

import torch
import torch.testing
from TTS.vocoder.models.hifigan_generator import HifiganGenerator

from neural_source_filter import NSFHifiganGenerator


class TestHifiGan(TestCase):
    def setUp(self) -> None:
        self.g = HifiganGenerator(
            in_channels=192,
            out_channels=1,
            resblock_type="1",
            resblock_dilation_sizes=[[1, 3, 5]] * 3,
            resblock_kernel_sizes=[3, 7, 11],
            upsample_kernel_sizes=[16, 16, 4, 4, 4],
            upsample_initial_channel=512,
            upsample_factors=[8, 8, 2, 2, 2],
            inference_padding=5,
            cond_channels=256,
            conv_pre_weight_norm=True,
            conv_post_weight_norm=True,
            conv_post_bias=True,
        )

    def test_forward(self):
        x = torch.randn(3, 192, 100)
        g = torch.randn(3, 256, 100)
        y = self.g(x, g)
        torch.testing.assert_allclose(
            y.shape, torch.Size([3, 1, 100 * 8 * 8 * 2 * 2 * 2])
        )


class TestNSFHifigan(TestCase):
    def setUp(self) -> None:
        self.g = NSFHifiganGenerator(
            in_channels=192,
            out_channels=1,
            resblock_type="1",
            resblock_dilation_sizes=[[1, 3, 5]] * 3,
            resblock_kernel_sizes=[3, 7, 11],
            upsample_kernel_sizes=[16, 16, 4, 4, 4],
            upsample_initial_channel=512,
            upsample_factors=[8, 8, 2, 2, 2],
            inference_padding=5,
            cond_channels=256,
            conv_pre_weight_norm=True,
            conv_post_weight_norm=True,
            conv_post_bias=True,
        )

    def test_forward(self):
        x = torch.randn(3, 192, 100)
        f0 = torch.randn(3, 1, 100)
        g = torch.randn(3, 256, 100)
        y = self.g(x, g, f0)
        torch.testing.assert_allclose(
            y.shape, torch.Size([3, 1, 100 * 8 * 8 * 2 * 2 * 2])
        )

# Copyright 2025 Wasserstein Distortion authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language governing
# permissions and limitations under the License.
# ========================================================================================
"""Computes features for perceptual representations."""

from typing import override
import torch
from torch import nn
from torchvision import models as tv
from . import helpers

Tensor = torch.Tensor


class VGG16Features(nn.Module):
    """Multiscale VGG-16 features of an image.

    Computes VGG-16 activations on the image and on ``num_scales - 1`` successively
    downsampled versions of it. Max pooling is replaced by average pooling.
    """

    mean: Tensor
    std: Tensor

    # Indices into `torchvision.models.vgg16().features` after which outputs are taken.
    LAYERS = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu4_3": 22, "relu5_3": 29}

    def __init__(
        self,
        num_scales: int = 3,
        num_layers: int = 5,
        avg_pooling: bool = True,
        pretrained: bool = True,
    ):
        """Initializer.

        Parameters
        ----------
        num_scales
            The number of scales of the image the features should be computed on.
        num_layers
            The number of VGG layers to output (between 1 and 5). Truncating the network
            saves computation.
        avg_pooling
            Whether to replace max pooling with average pooling.
        pretrained
            Whether to load pretrained ImageNet weights. If ``False``, the weights are
            randomly initialized, which is useful for unit testing.
        """
        super().__init__()
        if not 1 <= num_layers <= len(self.LAYERS):
            raise ValueError(
                f"`num_layers` must be between 1 and {len(self.LAYERS)}, inclusive, "
                f"but is {num_layers}."
            )
        self.num_scales = num_scales
        self.layer_names = list(self.LAYERS)[:num_layers]

        vgg = tv.vgg16(weights="IMAGENET1K_V1" if pretrained else None).features
        modules = [
            nn.AvgPool2d(kernel_size=2, stride=2)
            if avg_pooling and isinstance(m, nn.MaxPool2d)
            else m
            for m in vgg  # pyright: ignore[reportGeneralTypeIssues]
        ]
        # Split the network into consecutive blocks, each ending at an output layer.
        self.blocks = nn.ModuleList()
        start = 0
        for name in self.layer_names:
            end = self.LAYERS[name] + 1
            self.blocks.append(nn.Sequential(*modules[start:end]))
            start = end
        self.requires_grad_(False)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, 3, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1))

    @override
    def forward(self, image: Tensor) -> list[Tensor]:
        """Extracts VGG features from an image.

        Parameters
        ----------
        image
            Image tensor of shape ``(batch, 3, height, width)``. The pixel format is
            assumed to be: sRGB colorspace, with pixel values linearly scaled to floats in
            the range [0, 1].

        Returns
        -------
        list
            The image itself, followed by the VGG activations for each scale, each a
            tensor of shape ``(batch, channels, height, width)``.
        """
        features = [image]
        x = (image - self.mean) / self.std
        for _ in range(self.num_scales):
            h = x
            for block in self.blocks:
                h = block(h)
                features.append(h)
            x = helpers.lowpass(x, stride=2)
        return features

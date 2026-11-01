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

import jax
import jax.numpy as jnp
import flaxmodels as fm
from . import helpers

Array = jax.Array
ArrayLike = jax.typing.ArrayLike

_MODELS = {}


def load_vgg16_model(pretrained: bool = True) -> None:
    """Loads VGG-16 model, by default with pretrained ImageNet weights.

    Parameters
    ----------
    pretrained
        Whether to load pretrained ImageNet weights. If ``False``, the weights are
        randomly initialized, which is useful for unit testing.
    """
    vgg = fm.VGG16(
        output="activations",
        pretrained="imagenet" if pretrained else None,  # pyright: ignore[reportArgumentType]
        normalize=False,
        pooling="avg_pool",
        include_head=False,
    )
    dummy_input = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
    params = vgg.init(jax.random.key(0), dummy_input)
    _MODELS["vgg16"] = (vgg, params)


def compute_vgg16_features(image: ArrayLike, num_scales: int = 3) -> list[Array]:
    """Extracts VGG features from an image.

    Parameters
    ----------
    image
        Image of shape ``(3, height, width)``. The pixel format is assumed to be: sRGB
        colorspace, with pixel values linearly scaled to floats in the range [0, 1].
    num_scales
        The number of scales of the image the features should be computed on. The image
        will be downsampled ``num_scales - 1`` times and VGG features computed on the
        original image plus the downsampled versions.

    Returns
    -------
    list
        The image itself, followed by the VGG activations for each scale, each an array
        of shape ``(channels, height, width)``.
    """
    try:
        vgg, params = _MODELS["vgg16"]
    except KeyError:
        raise RuntimeError("Please call `load_vgg16_model` first.")
    layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
    features: list[Array] = [image]  # type:ignore
    mean = jnp.array([0.485, 0.456, 0.406]).reshape((-1, 1, 1)).astype(image.dtype)
    std = jnp.array([0.229, 0.224, 0.225]).reshape((-1, 1, 1)).astype(image.dtype)
    image = (image - mean) / std
    for _ in range(num_scales):
        activations = vgg.apply(params, image.transpose((1, 2, 0))[None], train=False)
        features.extend(activations[l].squeeze(0).transpose((2, 0, 1)) for l in layers)
        image = helpers.lowpass(image, 2)
    return features

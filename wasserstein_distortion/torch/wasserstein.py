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
"""Implementation of Wasserstein Distortion in PyTorch."""

import math
from collections.abc import Callable, Sequence
from typing import override
import torch
from torch import nn
from torch.nn import functional as F
from . import helpers
from . import pretrained_features

Tensor = torch.Tensor
FeatureExtractor = Callable[[Tensor], Sequence[Tensor]]


def compute_multiscale_stats(
    features: Tensor,
    num_levels: int,
) -> tuple[list[Tensor], list[Tensor]]:
    """Computes local mean and variance of a feature tensor."""
    # Filter features and their squares together, which halves the number of calls.
    stacked = torch.stack([features, torch.square(features)])
    means = []
    variances = []
    for _ in range(num_levels):
        filtered = helpers.lowpass(stacked, stride=1)
        m, p = filtered.unbind()
        means.append(m)
        variances.append(p - torch.square(m))
        stacked = filtered[..., ::2, ::2]
    return means, variances


def wasserstein_distortion(
    features_a: Tensor,
    features_b: Tensor,
    log2_sigma: Tensor,
    *,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
) -> Tensor:
    """Evaluates Wasserstein Distortion between two feature tensors.

    Parameters
    ----------
    features_a
        Tensor of shape ``(batch, channels, height, width)``. The first feature tensor to
        be compared.
    features_b
        Tensor of shape ``(batch, channels, height, width)``. The second feature tensor to
        be compared.
    log2_sigma
        Tensor of shape ``(batch, 1, height, width)``. The base two logarithm of the sigma
        map, which indicates the amount of summarization in each location. Must have the
        same height and width as the feature tensors.
    num_levels
        The number of multi-scale levels of the feature statistics to compute. Must be
        greater than or equal to the maximum of `log2_sigma` (not checked, to avoid a
        device synchronization).
    sqrt_grad_limit
        Upper limit for the gradient of the square root applied to the empirical feature
        variance estimates, for numerical stability.

    Returns
    -------
    Tensor
        Scalar distortion value, averaged over the batch.

    Notes
    -----
    For an introduction to Wasserstein Distortion, refer to [1]_. For a description of
    this implementation, refer to [2]_. Please cite these papers if you use this code for
    scientific or research work.

    .. [1] Y. Qiu, A. B. Wagner, J. Ballé, L. Theis: "Wasserstein Distortion: Unifying
       Fidelity and Realism," 2024 58th Ann. Conf. on Information Sciences and Systems
       (CISS), 2024. https://arxiv.org/abs/2310.03629
    .. [2] J. Ballé, L. Versari, E. Dupont, H. Kim, M. Bauer: "Good, Cheap,
       and Fast: Overfitted Image Compression with Wasserstein Distortion," 2025 IEEE/CVF
       Conf. on Computer Vision and Pattern Recognition (CVPR), 2025.
       https://arxiv.org/abs/2412.00505
    """
    if features_a.shape != features_b.shape:
        raise ValueError(
            f"`features_a` and `features_b` must have same shape, but received "
            f"{features_a.shape} and {features_b.shape}, respectively."
        )
    if features_a.shape[-2:] != log2_sigma.shape[-2:]:
        raise ValueError(
            f"features and `log2_sigma` must have same spatial shape, but received "
            f"{features_a.shape[-2:]} and {log2_sigma.shape[-2:]}, respectively."
        )

    means_a, variances_a = compute_multiscale_stats(features_a, num_levels)
    means_b, variances_b = compute_multiscale_stats(features_b, num_levels)

    wd_maps = [torch.square(features_a - features_b)]
    for ma, mb, va, vb in zip(means_a, means_b, variances_a, variances_b):
        # Variance estimates can turn out slightly negative due to numerics. This brings
        # such estimates up to zero, but passes through a useful gradient.
        va = helpers.lower_limit(va, 0.0)
        vb = helpers.lower_limit(vb, 0.0)
        # The square root has unbounded gradients near zero. This limits the gradient to a
        # finite value.
        sa = helpers.safe_sqrt(va, sqrt_grad_limit)
        sb = helpers.safe_sqrt(vb, sqrt_grad_limit)
        wd_maps.append(torch.square(ma - mb) + torch.square(sa - sb))

    dist = features_a.new_zeros(())
    for i, wd_map in enumerate(wd_maps):
        weight = F.relu(1 - torch.abs(log2_sigma - i))
        dist = dist + torch.mean(weight * wd_map)
        if i > 0:
            log2_sigma = helpers.lowpass(log2_sigma, stride=2)
    return dist


def multi_wasserstein_distortion(
    features_a: Sequence[Tensor],
    features_b: Sequence[Tensor],
    log2_sigma: Tensor,
    *,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
) -> Tensor:
    """Evaluates Wasserstein Distortion between multiple feature tensors of two images.

    This function accepts more than one feature tensor per image. The tensors don't need
    to be all the same shape, but the nth tensor in `features_a` must have the same shape
    as the nth tensor in `features_b`. The aspect ratio of all the tensors should be
    approximately the same.

    Parameters
    ----------
    features_a
        Multiple feature tensors of shape ``(batch, channels, height, width)``,
        corresponding to the first image to be compared.
    features_b
        Multiple feature tensors of shape ``(batch, channels, height, width)``,
        corresponding to the second image to be compared.
    log2_sigma
        Tensor of shape ``(batch, 1, height, width)``. The base two logarithm of the sigma
        map, which indicates the amount of summarization in each location. Doesn't have
        to have the same shape as the feature tensors.
    num_levels
        The number of multi-scale levels of the feature statistics to compute. Must be
        greater than or equal to the maximum of `log2_sigma`.
    sqrt_grad_limit
        Upper limit for the gradient of the square root applied to the empirical feature
        variance estimates, for numerical stability.

    Returns
    -------
    Tensor
        Scalar distortion value, averaged over the batch.
    """
    if len(features_a) != len(features_b):
        raise ValueError(
            f"`features_a` and `features_b` must have same length, but received "
            f"{len(features_a)} and {len(features_b)}, respectively."
        )

    dist = log2_sigma.new_zeros(())
    for fa, fb in zip(features_a, features_b):
        # Resize the sigma map to match the feature tensors.
        ls = F.interpolate(
            log2_sigma, size=fa.shape[-2:], mode="bilinear", antialias=True
        )
        # Rescale sigma to match the feature tensors. For example, if a feature tensor has
        # a very low spatial resolution, we make sigma correspondingly smaller, because
        # each element in the feature tensor covers a larger portion of the image. Since
        # we are in log space, we subtract the log of the size ratio and then cap at zero.
        log_ratio_h = math.log2(log2_sigma.shape[-2] / fa.shape[-2])
        log_ratio_w = math.log2(log2_sigma.shape[-1] / fa.shape[-1])
        mean_log_ratio = (log_ratio_h + log_ratio_w) / 2
        ls = F.relu(ls - mean_log_ratio)
        dist = dist + wasserstein_distortion(
            fa, fb, ls, num_levels=num_levels, sqrt_grad_limit=sqrt_grad_limit
        )
    return dist


class WassersteinDistortion(nn.Module):
    """Wasserstein Distortion between two images, using a pluggable feature extractor.

    The feature extractor can be any callable (typically an `nn.Module`) mapping an image
    tensor to a sequence of feature tensors of shape ``(batch, channels, height, width)``.
    """

    def __init__(
        self,
        features: FeatureExtractor,
        num_levels: int = 5,
        sqrt_grad_limit: float = 1e6,
    ):
        """Initializer.

        Parameters
        ----------
        features
            Feature extractor, mapping an image to a sequence of feature tensors.
        num_levels
            The number of multi-scale levels of the feature statistics to compute. Must
            be greater than or equal to the maximum of the `log2_sigma` passed to
            `forward`.
        sqrt_grad_limit
            Upper limit for the gradient of the square root applied to the empirical
            feature variance estimates, for numerical stability.
        """
        super().__init__()
        self.features = features
        self.num_levels = num_levels
        self.sqrt_grad_limit = sqrt_grad_limit

    @override
    def forward(self, image_a: Tensor, image_b: Tensor, log2_sigma: Tensor) -> Tensor:
        """Computes the distortion.

        Parameters
        ----------
        image_a
            First image to be compared, in the format expected by the feature extractor.
        image_b
            Second image to be compared, in the format expected by the feature extractor.
        log2_sigma
            Tensor of shape ``(batch, 1, height, width)``. The base two logarithm of the
            sigma map, which indicates the amount of summarization in each location.
            Doesn't have to have the same shape as the images.

        Returns
        -------
        Tensor
            Scalar distortion value, averaged over the batch.
        """
        if image_a.shape != image_b.shape:
            raise ValueError(
                f"Images must have the same shape, but received {image_a.shape} and "
                f"{image_b.shape}."
            )
        return multi_wasserstein_distortion(
            self.features(image_a),
            self.features(image_b),
            log2_sigma,
            num_levels=self.num_levels,
            sqrt_grad_limit=self.sqrt_grad_limit,
        )


class VGG16WassersteinDistortion(WassersteinDistortion):
    """VGG-16 Wasserstein Distortion between two images.

    Images are expected to have shape ``(batch, 3, height, width)``, in sRGB colorspace,
    with pixel values linearly scaled to floats in the range [0, 1].

    Notes
    -----
    This is the distortion loss function used in [1]_. Please cite the paper if you use
    this code for scientific or research work.

    .. [1] J. Ballé, L. Versari, E. Dupont, H. Kim, M. Bauer: "Good, Cheap,
       and Fast: Overfitted Image Compression with Wasserstein Distortion," 2025 IEEE/CVF
       Conf. on Computer Vision and Pattern Recognition (CVPR), 2025.
       https://arxiv.org/abs/2412.00505
    """

    def __init__(
        self,
        num_scales: int = 3,
        num_levels: int = 5,
        sqrt_grad_limit: float = 1e6,
        pretrained: bool = True,
    ):
        """Initializer.

        Parameters
        ----------
        num_scales
            The number of scales of the image the features should be computed on. The
            image will be downsampled ``num_scales - 1`` times and VGG features computed
            on the original image plus the downsampled versions.
        num_levels
            The number of multi-scale levels of the feature statistics to compute. Must
            be greater than or equal to the maximum of the `log2_sigma` passed to
            `forward`.
        sqrt_grad_limit
            Upper limit for the gradient of the square root applied to the empirical
            feature variance estimates, for numerical stability.
        pretrained
            Whether to load pretrained ImageNet weights. If ``False``, the weights are
            randomly initialized, which is useful for unit testing.
        """
        super().__init__(
            pretrained_features.VGG16Features(
                num_scales=num_scales, pretrained=pretrained
            ),
            num_levels=num_levels,
            sqrt_grad_limit=sqrt_grad_limit,
        )

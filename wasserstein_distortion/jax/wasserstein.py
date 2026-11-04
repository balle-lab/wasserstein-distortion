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
"""Implementation of Wasserstein Distortion in JAX."""

import collections
from collections.abc import Collection, Sequence
from typing import overload, Any, Literal
import jax
from jax import numpy as jnp
from jax.scipy import stats
from . import helpers
from . import pretrained_features

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


def compute_multiscale_stats(
    features: ArrayLike,
    num_levels: int,
) -> tuple[list[Array], list[Array]]:
    """Computes local mean and variance of a feature array."""
    # Filter features and their squares together, which halves the number of calls.
    stacked = jnp.stack([features, jnp.square(features)])
    means = []
    variances = []
    for _ in range(num_levels):
        filtered = helpers.lowpass(stacked, stride=1)
        m, p = jnp.unstack(filtered)
        means.append(m)
        variances.append(p - jnp.square(m))
        stacked = filtered[..., ::2, ::2]
    return means, variances


@overload
def wasserstein_distortion(
    features_a: ArrayLike,
    features_b: ArrayLike,
    log2_sigma: Array,
    *,
    weight: ArrayLike = 1.0,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_intermediates: Literal[False] = False,
) -> Array: ...


@overload
def wasserstein_distortion(
    features_a: ArrayLike,
    features_b: ArrayLike,
    log2_sigma: Array,
    *,
    weight: ArrayLike = 1.0,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_intermediates: Literal[True],
) -> tuple[Array, dict[str, Any]]: ...


def wasserstein_distortion(
    features_a: ArrayLike,
    features_b: ArrayLike,
    log2_sigma: Array,
    *,
    weight: ArrayLike = 1.0,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_intermediates: bool = False,
) -> Array | tuple[Array, dict[str, Any]]:
    """Evaluates Wasserstein Distortion between two feature arrays.

    Parameters
    ----------
    features_a
        Array of shape ``(channels, height, width)``. The first feature array to be
        compared.
    features_b
        Array of shape ``(channels, height, width)``. The second feature array to be
        compared.
    log2_sigma
        Array of shape ``(height, width)``. The base two logarithm of the sigma map, which
        indicates the amount of summarization in each location. Must have the same height
        and width as the feature arrays.
    weight
        Scalar or vector of shape ``(channels,)``. If a vector, each element is applied to
        the distortion in the corresponding feature channel. The distortion is the mean
        of the weighted per-channel distortions.
    num_levels
        The number of multi-scale levels of the feature statistics to compute. Must be
        greater than or equal to the maximum of `log2_sigma`.
    sqrt_grad_limit
        Upper limit for the gradient of the square root applied to the empirical feature
        variance estimates, for numerical stability.
    return_intermediates
        If ``True``, returns intermediate computations in a dictionary, besides the
        distortion value.

    Returns
    -------
    Array
        Scalar distortion value.
    dict
        Dictionary containing intermediate computations (if `return_intermediates`).

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
    if features_a.shape[-2:] != log2_sigma.shape:
        raise ValueError(
            f"features and `log2_sigma` must have same spatial shape, but received "
            f"{features_a.shape[-2:]} and {log2_sigma.shape}, respectively."
        )

    if not isinstance(log2_sigma, jax.core.Tracer) and jnp.max(log2_sigma) > num_levels:
        raise ValueError(f"Must have max(log2_sigma) <= {num_levels}.")
    weight = jnp.asarray(weight)
    if weight.ndim > 1 or (weight.ndim == 1 and weight.shape[0] != features_a.shape[-3]):
        raise ValueError(
            f"`weight` must be a scalar or a vector with one element per channel "
            f"({features_a.shape[-3]}), but received shape {weight.shape}."
        )

    means_a, variances_a = compute_multiscale_stats(features_a, num_levels)
    means_b, variances_b = compute_multiscale_stats(features_b, num_levels)
    assert len(means_a) == len(means_b) == len(variances_a) == len(variances_b)

    wd_maps = [jnp.square(features_a - features_b)]  # type: ignore
    for ma, mb, va, vb in zip(means_a, means_b, variances_a, variances_b):
        assert ma.shape == mb.shape == va.shape == vb.shape
        # Variance estimates can turn out slightly negative due to numerics. This brings
        # such estimates up to zero, but passes through a useful gradient.
        va = helpers.lower_limit(va, 0)
        vb = helpers.lower_limit(vb, 0)
        # The square root has unbounded gradients near zero. This limits the gradient to a
        # finite value.
        sa = helpers.safe_sqrt(va, sqrt_grad_limit)
        sb = helpers.safe_sqrt(vb, sqrt_grad_limit)
        wd_maps.append(jnp.square(ma - mb) + jnp.square(sa - sb))
    assert len(wd_maps) == num_levels + 1

    # Accumulate the distortion per channel, so the channel weights can be applied once.
    channel_dist = jnp.zeros(features_a.shape[-3])
    intermediates = collections.defaultdict(list)
    intermediates.update(wd_maps=wd_maps)
    for i, wd_map in enumerate(wd_maps):
        assert wd_map.shape[-2:] == log2_sigma.shape
        level_weight = jax.nn.relu(1 - abs(log2_sigma - i))
        intermediates["weights"].append(level_weight)
        channel_dist += jnp.mean(level_weight * wd_map, axis=(-2, -1))
        if i > 0:
            log2_sigma = helpers.lowpass(log2_sigma[None], stride=2).squeeze(0)
    dist = jnp.mean(weight * channel_dist)

    if return_intermediates:
        return dist, intermediates
    return dist


@overload
def multi_wasserstein_distortion(
    features_a: Collection[Array],
    features_b: Collection[Array],
    log2_sigma: Array,
    *,
    weights: Sequence[ArrayLike] | None = None,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_intermediates: Literal[False] = False,
) -> Array: ...


@overload
def multi_wasserstein_distortion(
    features_a: Collection[Array],
    features_b: Collection[Array],
    log2_sigma: Array,
    *,
    weights: Sequence[ArrayLike] | None = None,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_intermediates: Literal[True],
) -> tuple[Array, dict[str, Any]]: ...


def multi_wasserstein_distortion(
    features_a: Collection[Array],
    features_b: Collection[Array],
    log2_sigma: Array,
    *,
    weights: Sequence[ArrayLike] | None = None,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_intermediates: bool = False,
) -> Array | tuple[Array, dict[str, Any]]:
    """Evaluates Wasserstein Distortion between multiple feature arrays of two images.

    This function accepts more than one feature array per image. The arrays don't need to
    be all the same shape, but the nth array in `features_a` must have the same shape as
    the nth array in `features_b`. The aspect ratio of all the arrays should be
    approximately the same.

    Parameters
    ----------
    features_a
        Multiple feature arrays of shape ``(channels, height, width)``, corresponding to
        the first image to be compared.
    features_b
        Multiple feature arrays of shape ``(channels, height, width)``, corresponding to
        the second image to be compared.
    log2_sigma
        Array of shape ``(height, width)``. The base two logarithm of the sigma map, which
        indicates the amount of summarization in each location. Doesn't have to have the
        same shape as the feature arrays.
    weights
        One weight per feature array, used to form the weighted sum of the distortion
        values across feature arrays. Each weight can be a scalar, or a vector of shape
        ``(channels,)`` which weights each channel of the corresponding feature array
        separately (see `wasserstein_distortion`). Scalars and vectors can be mixed. If
        ``None``, all weights are 1.
    num_levels
        The number of multi-scale levels of the feature statistics to compute. Must be
        greater than or equal to the maximum of `log2_sigma`.
    sqrt_grad_limit
        Upper limit for the gradient of the square root applied to the empirical feature
        variance estimates, for numerical stability.
    return_intermediates
        If ``True``, returns intermediate computations in a dictionary, besides the
        distortion value.

    Returns
    -------
    Array
        Scalar distortion value.
    dict
        Dictionary containing intermediate computations (if `return_intermediates`).

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
    if len(features_a) != len(features_b):
        raise ValueError(
            f"`features_a` and `features_b` must have same length, but received "
            f"{len(features_a)} and {len(features_b)}, respectively."
        )
    if weights is None:
        weights = [1.0] * len(features_a)
    elif len(weights) != len(features_a):
        raise ValueError(
            f"`weights` must have the same length as the feature lists, but received "
            f"{len(weights)} and {len(features_a)}, respectively."
        )

    dist = jnp.zeros(())
    intermediates = collections.defaultdict(list)
    for fa, fb, w in zip(features_a, features_b, weights):
        if fa.shape != fb.shape:
            raise ValueError(
                f"Found feature arrays with incompatible shapes. "
                f"A: {fa.shape}, B: {fb.shape}."
            )
        # Resize the sigma map to match the feature arrays.
        ls = jax.image.resize(log2_sigma, fa.shape[-2:], "linear", antialias=True)
        # Rescale sigma to match the feature arrays. For example, if a feature array has a
        # very low spatial resolution, we make sigma correspondingly smaller, because each
        # element in the feature array covers a larger portion of the image. Since we are
        # in log space, we subtract the log of the size ratio and then cap at zero.
        log_ratio_h = jnp.log2(log2_sigma.shape[-2] / fa.shape[-2])
        log_ratio_w = jnp.log2(log2_sigma.shape[-1] / fa.shape[-1])
        mean_log_ratio = (log_ratio_h + log_ratio_w) / 2
        ls = jax.nn.relu(ls - mean_log_ratio)
        d, s = wasserstein_distortion(
            fa,
            fb,
            ls,
            weight=w,
            num_levels=num_levels,
            sqrt_grad_limit=sqrt_grad_limit,
            return_intermediates=True,
        )
        dist += d
        for i in s:
            intermediates[i].append(s[i])

    if return_intermediates:
        return dist, intermediates
    return dist


def vgg16_wasserstein_distortion(
    image_a: ArrayLike,
    image_b: ArrayLike,
    log2_sigma: Array,
    *,
    num_scales: int = 3,
    image_weight: ArrayLike = 20.0,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
) -> Array:
    """Evaluates VGG-16 Wasserstein Distortion between two images.

    Parameters
    ----------
    image_a
        First image to be compared, of shape ``(3, height, width)``. The pixel format is
        assumed to be the same as for VGG: sRGB colorspace, with pixel values linearly
        scaled to floats in the range [0, 1].
    image_b
        Second image to be compared, of shape ``(3, height, width)``. The pixel format is
        assumed to be the same as for VGG: sRGB colorspace, with pixel values linearly
        scaled to floats in the range [0, 1].
    log2_sigma
        Array of shape ``(height, width)``. The base two logarithm of the sigma map, which
        indicates the amount of summarization in each location. Doesn't have to have the
        same shape as the image arrays.
    num_scales
        The number of scales of the image the features should be computed on. The image
        will be downsampled ``num_scales - 1`` times and VGG features computed on the
        original image plus the downsampled versions. The concatenated list of all
        features will be used to compute the distortion.
    image_weight
        Weight of the distortion computed on the image itself (i.e., on pixels), relative
        to the VGG feature maps, which each have weight 1. Can be a scalar, or a vector of
        shape ``(3,)`` to weight each color channel separately.
    num_levels
        The number of multi-scale levels of the feature statistics to compute. Must be
        greater than or equal to the maximum of `log2_sigma`.
    sqrt_grad_limit
        Upper limit for the gradient of the square root applied to the empirical feature
        variance estimates, for numerical stability.

    Returns
    -------
    Array
        Scalar distortion value.

    Notes
    -----
    This is the distortion loss function used in [1]_. Please cite the paper if you use
    this code for scientific or research work.

    .. [1] J. Ballé, L. Versari, E. Dupont, H. Kim, M. Bauer: "Good, Cheap,
       and Fast: Overfitted Image Compression with Wasserstein Distortion," 2025 IEEE/CVF
       Conf. on Computer Vision and Pattern Recognition (CVPR), 2025.
       https://arxiv.org/abs/2412.00505
    """
    features_a = pretrained_features.compute_vgg16_features(
        image_a, num_scales=num_scales
    )
    features_b = pretrained_features.compute_vgg16_features(
        image_b, num_scales=num_scales
    )

    return multi_wasserstein_distortion(
        features_a,
        features_b,
        log2_sigma,
        weights=[image_weight] + [1.0] * (len(features_a) - 1),
        num_levels=num_levels,
        sqrt_grad_limit=sqrt_grad_limit,
        return_intermediates=False,
    )


def discrete_wasserstein_distortion(
    features_a: Collection[Array],
    features_b: Collection[Array],
    centers_w: Array,
    centers_h: Array,
    sigmas: Collection[Array],
) -> Array:
    """Evaluates Wasserstein Distortion for a discrete number of receptive fields.

    This function accepts more than one feature array per image. The arrays don't need to
    be all the same shape, but the nth array in `features_a` must have the same shape as
    the nth array in `features_b`.

    Parameters
    ----------
    features_a
        Multiple feature arrays of shape ``(channels, height, width)``, corresponding to
        the first image to be compared.
    features_b
        Multiple feature arrays of shape ``(channels, height, width)``, corresponding to
        the second image to be compared.
    centers_w
        Horizontal coordinates for all feature arrays, in relative coordinates: the left
        is 0, the right is 1. The argument is interpreted as a vector, specifying one or
        more locations of receptive fields within each feature array.
    centers_h
        Vertical coordinates for all feature arrays, in relative coordinates: the top is
        0, the bottom is 1. The argument is interpreted as a vector, specifying one or
        more locations of receptive fields within each feature array.
    sigmas
        List of receptive field sizes for each feature array. The number of list elements
        must correspond to the number of feature arrays. Each element of the list is
        interpreted as a vector, specifying one or more sizes of receptive fields within
        the corresponding feature array. The sizes are scaled with the geometric mean of
        the dimensions of each feature array.

    Returns
    -------
    Array
        Scalar distortion value.

    Notes
    -----
    For an introduction to Wasserstein Distortion, refer to [1]_. Please cite this paper
    if you use this code for scientific or research work.

    .. [1] Y. Qiu, A. B. Wagner, J. Ballé, L. Theis: "Wasserstein Distortion: Unifying
       Fidelity and Realism," 2024 58th Ann. Conf. on Information Sciences and Systems
       (CISS), 2024. https://arxiv.org/abs/2310.03629
    """
    if not len(features_a) == len(features_b) == len(sigmas):
        raise ValueError("The list of feature maps and sigmas must have same length.")

    dist = jnp.zeros(())
    for fa, fb, ss in zip(features_a, features_b, sigmas):
        if fa.shape != fb.shape:
            raise ValueError(
                f"Found feature arrays with incompatible shapes. "
                f"A: {fa.shape}, B: {fb.shape}."
            )
        if not centers_w.ndim == centers_h.ndim == ss.ndim == 1:
            raise ValueError(
                f"Centers and sigma arrays must be vectors. "
                f"centers_w: {centers_w.shape}, centers_h: {centers_h.shape}, "
                f"sigmas: {ss.shape})"
            )
        if not centers_w.shape == centers_h.shape == ss.shape:
            raise ValueError(
                f"Found centers or sigma vectors with different lengths. "
                f"centers_w: {centers_w.size}, centers_h: {centers_h.size}, "
                f"sigmas: {ss.size})"
            )

        sigma_factor = jnp.sqrt(fa.shape[-2] * fa.shape[-1])
        window_w = stats.norm.pdf(
            jnp.linspace(0.0, fa.shape[-1], fa.shape[-1])[None, :],
            centers_w[:, None] * fa.shape[-1],
            ss[:, None] * sigma_factor,
        )
        window_h = stats.norm.pdf(
            jnp.linspace(0.0, fa.shape[-2], fa.shape[-2])[None, :],
            centers_h[:, None] * fa.shape[-2],
            ss[:, None] * sigma_factor,
        )
        window_w /= jnp.sum(window_w, axis=-1, keepdims=True)
        window_h /= jnp.sum(window_h, axis=-1, keepdims=True)

        mean_a = jnp.einsum("chw,rw,rh->rc", fa, window_w, window_h)
        mean_b = jnp.einsum("chw,rw,rh->rc", fb, window_w, window_h)

        squared_fa = jnp.square(fa - mean_a[:, :, None, None])
        squared_fb = jnp.square(fb - mean_b[:, :, None, None])

        std_a = jnp.sqrt(jnp.einsum("rchw,rw,rh->rc", squared_fa, window_w, window_h))
        std_b = jnp.sqrt(jnp.einsum("rchw,rw,rh->rc", squared_fb, window_w, window_h))

        d = jnp.sqrt(jnp.square(mean_a - mean_b) + jnp.square(std_a - std_b))
        d = d.mean(axis=0).sum()
        dist += d

    return dist

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
import math
from collections.abc import Collection, Sequence
from typing import overload, Any, Literal
import jax
from jax import numpy as jnp
from jax.scipy import stats
import numpy as np
from . import helpers
from . import pretrained_features

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


def _check_norm(name: str, value: float) -> None:
    """Raises if a norm exponent is not positive and finite."""
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"`{name}` must be positive and finite, but received {value}.")


def _concrete_max(x: Array) -> float | None:
    """Returns the maximum of `x` computed on the host, or ``None`` if `x` is traced.

    Uses NumPy rather than `jnp`, because under `jax.jit`, `jnp` operations are staged
    into the compiled computation even when their inputs are concrete. This way, the check
    runs at most once at trace time and doesn't add work to the compiled code.
    """
    if isinstance(x, jax.core.Tracer):
        return None
    return float(np.max(np.asarray(x)))


def compute_multiscale_stats(
    features: Array,
    num_levels: int,
) -> tuple[list[Array], list[Array]]:
    """Computes local mean and variance of a feature array."""
    # Filter features and their squares together, which halves the number of calls.
    stacked = jnp.stack([features, jnp.square(features)])
    means = []
    variances = []
    for i in range(num_levels):
        filtered = helpers.lowpass(stacked)
        m, p = jnp.unstack(filtered)
        means.append(m)
        variances.append(p - jnp.square(m))
        if i < num_levels - 1:
            stacked = helpers.subsample(filtered, helpers.subsampling_offset(i))
    return means, variances


@overload
def wasserstein_distortion(
    features_a: Array,
    features_b: Array,
    log2_sigma: Array,
    *,
    norm_spatial: float = 2.0,
    norm_channels: float = 2.0,
    weight: ArrayLike = 1.0,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_wd_map: tuple[int, int] | None = None,
    return_intermediates: Literal[False] = False,
) -> Array: ...


@overload
def wasserstein_distortion(
    features_a: Array,
    features_b: Array,
    log2_sigma: Array,
    *,
    norm_spatial: float = 2.0,
    norm_channels: float = 2.0,
    weight: ArrayLike = 1.0,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_wd_map: tuple[int, int] | None = None,
    return_intermediates: Literal[True],
) -> tuple[Array, dict[str, Any]]: ...


def wasserstein_distortion(
    features_a: Array,
    features_b: Array,
    log2_sigma: Array,
    *,
    norm_spatial: float = 2.0,
    norm_channels: float = 2.0,
    weight: ArrayLike = 1.0,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_wd_map: tuple[int, int] | None = None,
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
    norm_spatial
        The exponent :math:`p` of the :math:`L_p` norm taken over spatial locations. Must
        be positive and finite. See Notes.
    norm_channels
        The exponent :math:`q` of the :math:`L_q` norm taken over channels. Must be
        positive and finite. See Notes.
    weight
        Nonnegative scalar or vector of shape ``(channels,)``. If a vector, each element
        is applied to the distortion in the corresponding feature channel. The distortion
        is the mean of the weighted per-channel distortions.
    num_levels
        The number of multi-scale levels of the feature statistics to compute. Must be
        greater than or equal to the maximum of `log2_sigma`.
    sqrt_grad_limit
        Upper limit for the gradient magnitude of the square root applied to the empirical
        feature variance estimates, and of fractional powers taken by the norms, for
        numerical stability.
    return_wd_map
        If ``None``, returns the scalar distortion value. If a spatial shape ``(height,
        width)``, returns a distortion map of that shape instead, which aggregates the
        distortion across channels and multi-scale levels in the same way as the scalar
        value, but not across space. The distortion maps of all multi-scale levels are
        resized to this shape before being summed. Each channel contributes to the map in
        proportion to the local contributions to its spatial norm, such that the mean of
        the map is approximately equal to the scalar value.
    return_intermediates
        If ``True``, returns intermediate computations in a dictionary, besides the
        distortion value.

    Returns
    -------
    Array
        Scalar distortion value, or distortion map (if `return_wd_map` is a shape).
    dict
        Dictionary containing intermediate computations (if `return_intermediates`).

    Notes
    -----
    This is a feature-agnostic implementation of the distortion loss used in [1]_. For the
    specific version used in the paper, see `vgg16_wasserstein_distortion`. For a
    theoretical introduction to Wasserstein Distortion, refer to [2]_. Please cite these
    papers if you use this code for scientific or research work.

    Let :math:`d_c(x)` be the local Wasserstein distance in channel :math:`c` at location
    :math:`x` (i.e., the square root of the corresponding entry of the multi-scale
    distortion maps), and :math:`w_c` the channel weight. With :math:`p` = `norm_spatial`
    and :math:`q` = `norm_channels`, the distortion is

    .. math:: \\operatorname{mean}_c \\, w_c \\left( \\operatorname{mean}_x \\, d_c(x)^p
       \\right)^{q/p},

    i.e., the :math:`q`-th power of a weighted, mean-normalized composite
    :math:`L_q`/:math:`L_p` norm. Taking the power rather than the norm itself means the
    defaults :math:`p = q = 2` yield the weighted mean of the squared distances. For
    non-integer `log2_sigma`, :math:`d_c(x)^p` is linearly interpolated between adjacent
    multi-scale levels.

    .. [1] J. Ballé, L. Versari, E. Dupont, H. Kim, M. Bauer: "Good, Cheap,
       and Fast: Overfitted Image Compression with Wasserstein Distortion," 2025 IEEE/CVF
       Conf. on Computer Vision and Pattern Recognition (CVPR), 2025.
       https://arxiv.org/abs/2412.00505
    .. [2] Y. Qiu, A. B. Wagner, J. Ballé, L. Theis: "Wasserstein Distortion: Unifying
       Fidelity and Realism," 2024 58th Ann. Conf. on Information Sciences and Systems
       (CISS), 2024. https://arxiv.org/abs/2310.03629
    """
    _check_norm("norm_spatial", norm_spatial)
    _check_norm("norm_channels", norm_channels)
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

    # Tolerate rounding errors, e.g. from resizing in `multi_wasserstein_distortion`. The
    # weight of the missing level beyond `num_levels` is at most the excess.
    sigma_max = _concrete_max(log2_sigma)
    if sigma_max is not None and sigma_max > num_levels + 1e-5:
        raise ValueError(f"Must have max(log2_sigma) <= {num_levels}.")
    if return_wd_map is not None and len(return_wd_map) != 2:
        raise ValueError(
            f"`return_wd_map` must be None or a spatial shape (height, width), but "
            f"received {return_wd_map}."
        )
    weight = jnp.asarray(weight)
    if weight.ndim > 1 or (weight.ndim == 1 and weight.shape[0] != features_a.shape[-3]):
        raise ValueError(
            f"`weight` must be a scalar or a vector with one element per channel "
            f"({features_a.shape[-3]}), but received shape {weight.shape}."
        )

    means_a, variances_a = compute_multiscale_stats(features_a, num_levels)
    means_b, variances_b = compute_multiscale_stats(features_b, num_levels)
    assert len(means_a) == len(means_b) == len(variances_a) == len(variances_b)

    wd_maps = [jnp.square(features_a - features_b)]
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

    # Weight the p-th powers of the distances for each level. The maps contain squared
    # distances, so raising them to the power p/2 yields the p-th power of the distances.
    level_dists = []
    intermediates = collections.defaultdict(list)
    intermediates.update(wd_maps=wd_maps)
    for i, wd_map in enumerate(wd_maps):
        assert wd_map.shape[-2:] == log2_sigma.shape
        level_weight = jax.nn.relu(1 - abs(log2_sigma - i))
        intermediates["weights"].append(level_weight)
        level_dists.append(
            level_weight * helpers.safe_pow(wd_map, norm_spatial / 2, sqrt_grad_limit)
        )
        if 0 < i < len(wd_maps) - 1:
            # The statistics of the next level were subsampled in step `i - 1`.
            log2_sigma = helpers.lowpass(
                log2_sigma, downsample=helpers.subsampling_offset(i - 1)
            )

    # Convert the p-th power of the spatial norm per channel to its q-th power, and
    # apply the channel weights.
    exponent = norm_channels / norm_spatial
    if return_wd_map is None:
        channel_dist = sum(jnp.mean(d, axis=(-2, -1)) for d in level_dists)
        channel_dist = helpers.safe_pow(channel_dist, exponent, sqrt_grad_limit)  # pyright: ignore[reportArgumentType]
        dist = jnp.mean(weight * channel_dist)
    else:
        shape = (features_a.shape[-3], *return_wd_map)
        channel_map = sum(
            jax.image.resize(d, shape, "linear", antialias=True) for d in level_dists
        )
        if exponent != 1:
            # Scale each channel's map so that its mean is the q-th power of the spatial
            # norm. Since q/p - 1 > -1, flooring the norm at sqrt(tiny) keeps the scale
            # below 1/sqrt(tiny), which is finite. Where the norm is below the floor, the
            # map is (nearly) zero anyway.
            channel_dist = jnp.mean(channel_map, axis=(-2, -1))
            floor = jnp.sqrt(jnp.finfo(channel_dist.dtype).tiny)
            weight = weight * helpers.safe_pow(
                jnp.maximum(channel_dist, floor), exponent - 1, sqrt_grad_limit
            )
        dist = jnp.mean(weight[..., None, None] * channel_map, axis=0)

    if return_intermediates:
        return dist, intermediates
    return dist


@overload
def multi_wasserstein_distortion(
    features_a: Collection[Array],
    features_b: Collection[Array],
    log2_sigma: Array,
    *,
    norm_spatial: float = 2.0,
    norm_channels: float = 2.0,
    weights: Sequence[ArrayLike] | None = None,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_wd_map: tuple[int, int] | None = None,
    return_intermediates: Literal[False] = False,
) -> Array: ...


@overload
def multi_wasserstein_distortion(
    features_a: Collection[Array],
    features_b: Collection[Array],
    log2_sigma: Array,
    *,
    norm_spatial: float = 2.0,
    norm_channels: float = 2.0,
    weights: Sequence[ArrayLike] | None = None,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_wd_map: tuple[int, int] | None = None,
    return_intermediates: Literal[True],
) -> tuple[Array, dict[str, Any]]: ...


def multi_wasserstein_distortion(
    features_a: Collection[Array],
    features_b: Collection[Array],
    log2_sigma: Array,
    *,
    norm_spatial: float = 2.0,
    norm_channels: float = 2.0,
    weights: Sequence[ArrayLike] | None = None,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_wd_map: tuple[int, int] | None = None,
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
        the first image to be compared. No feature array may be larger than the image in
        either spatial dimension.
    features_b
        Multiple feature arrays of shape ``(channels, height, width)``, corresponding to
        the second image to be compared. No feature array may be larger than the image in
        either spatial dimension.
    log2_sigma
        Array of shape ``(height, width)``. The base two logarithm of the sigma map, which
        indicates the amount of summarization in each location, in units of pixels of the
        image. It must have the shape of the image the features were computed on. It is
        resized to the shape of each feature array, and rescaled to its resolution.
    norm_spatial
        The exponent of the :math:`L_p` norm taken over spatial locations (see
        `wasserstein_distortion`).
    norm_channels
        The exponent of the :math:`L_p` norm taken over channels (see
        `wasserstein_distortion`). The resulting distortion values are summed across
        feature arrays.
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
        Upper limit for the gradient magnitude of the square root applied to the empirical
        feature variance estimates, and of fractional powers taken by the norms, for
        numerical stability.
    return_wd_map
        If ``None``, returns the scalar distortion value. If a spatial shape ``(height,
        width)``, returns a distortion map of that shape instead, which aggregates the
        distortion across feature arrays, channels and multi-scale levels in the same way
        as the scalar value, but not across space (see `wasserstein_distortion`). A
        natural choice is the shape of the largest feature array, or of `log2_sigma`. The
        mean of the map is approximately equal to the scalar value.
    return_intermediates
        If ``True``, returns intermediate computations in a dictionary, besides the
        distortion value.

    Returns
    -------
    Array
        Scalar distortion value, or distortion map (if `return_wd_map` is a shape).
    dict
        Dictionary containing intermediate computations (if `return_intermediates`).

    Notes
    -----
    This is a feature-agnostic implementation of the distortion loss used in [1]_. For the
    specific version used in the paper, see `vgg16_wasserstein_distortion`. For a
    theoretical introduction to Wasserstein Distortion, refer to [2]_. Please cite these
    papers if you use this code for scientific or research work.

    .. [1] J. Ballé, L. Versari, E. Dupont, H. Kim, M. Bauer: "Good, Cheap,
       and Fast: Overfitted Image Compression with Wasserstein Distortion," 2025 IEEE/CVF
       Conf. on Computer Vision and Pattern Recognition (CVPR), 2025.
       https://arxiv.org/abs/2412.00505
    .. [2] Y. Qiu, A. B. Wagner, J. Ballé, L. Theis: "Wasserstein Distortion: Unifying
       Fidelity and Realism," 2024 58th Ann. Conf. on Information Sciences and Systems
       (CISS), 2024. https://arxiv.org/abs/2310.03629
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

    # Since no feature array is larger than the sigma map, the rescaling below can only
    # decrease sigma, and it suffices to check the sigma map itself.
    sigma_max = _concrete_max(log2_sigma)
    if sigma_max is not None and sigma_max > num_levels:
        raise ValueError(f"Must have max(log2_sigma) <= {num_levels}.")

    dist = jnp.zeros(() if return_wd_map is None else return_wd_map)
    intermediates = collections.defaultdict(list)
    for fa, fb, w in zip(features_a, features_b, weights):
        if fa.shape != fb.shape:
            raise ValueError(
                f"Found feature arrays with incompatible shapes. "
                f"A: {fa.shape}, B: {fb.shape}."
            )
        if fa.shape[-2] > log2_sigma.shape[-2] or fa.shape[-1] > log2_sigma.shape[-1]:
            raise ValueError(
                f"Feature arrays must not be larger than `log2_sigma` in either spatial "
                f"dimension, but received {fa.shape[-2:]} and {log2_sigma.shape}."
            )
        # Resize the sigma map to match the feature arrays.
        ls = jax.image.resize(log2_sigma, fa.shape[-2:], "linear", antialias=True)
        # Rescale sigma to match the feature arrays. For example, if a feature array has a
        # very low spatial resolution, we make sigma correspondingly smaller, because each
        # element in the feature array covers a larger portion of the image. Since we are
        # in log space, we subtract the log of the size ratio and then cap at zero.
        log_ratio_h = math.log2(log2_sigma.shape[-2] / fa.shape[-2])
        log_ratio_w = math.log2(log2_sigma.shape[-1] / fa.shape[-1])
        mean_log_ratio = (log_ratio_h + log_ratio_w) / 2
        ls = jax.nn.relu(ls - mean_log_ratio)
        d, s = wasserstein_distortion(
            fa,
            fb,
            ls,
            norm_spatial=norm_spatial,
            norm_channels=norm_channels,
            weight=w,
            num_levels=num_levels,
            sqrt_grad_limit=sqrt_grad_limit,
            return_wd_map=return_wd_map,
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
    norm_spatial: float = 2.0,
    norm_channels: float = 2.0,
    image_weight: ArrayLike = 20.0,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_wd_map: tuple[int, int] | None = None,
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
        indicates the amount of summarization in each location, in units of pixels. It
        must have the same shape as the image arrays.
    num_scales
        The number of scales of the image the features should be computed on. The image
        will be downsampled ``num_scales - 1`` times and VGG features computed on the
        original image plus the downsampled versions. The concatenated list of all
        features will be used to compute the distortion.
    norm_spatial
        The exponent of the :math:`L_p` norm taken over spatial locations (see
        `wasserstein_distortion`).
    norm_channels
        The exponent of the :math:`L_p` norm taken over channels (see
        `wasserstein_distortion`).
    image_weight
        Weight of the distortion computed on the image itself (i.e., on pixels), relative
        to the VGG feature maps, which each have weight 1. Can be a scalar, or a vector of
        shape ``(3,)`` to weight each color channel separately.
    num_levels
        The number of multi-scale levels of the feature statistics to compute. Must be
        greater than or equal to the maximum of `log2_sigma`.
    sqrt_grad_limit
        Upper limit for the gradient magnitude of the square root applied to the empirical
        feature variance estimates, and of fractional powers taken by the norms, for
        numerical stability.
    return_wd_map
        If ``None``, returns the scalar distortion value. If a spatial shape ``(height,
        width)``, returns a distortion map of that shape instead, which aggregates the
        distortion in the same way as the scalar value, but not across space (see
        `multi_wasserstein_distortion`). A natural choice is the shape of the images. The
        mean of the map is approximately equal to the scalar value.

    Returns
    -------
    Array
        Scalar distortion value, or distortion map (if `return_wd_map` is a shape).

    Notes
    -----
    This is the distortion loss function used in [1]_. For a theoretical introduction to
    Wasserstein Distortion, refer to [2]_. Please cite these papers if you use this code
    for scientific or research work.

    .. [1] J. Ballé, L. Versari, E. Dupont, H. Kim, M. Bauer: "Good, Cheap,
       and Fast: Overfitted Image Compression with Wasserstein Distortion," 2025 IEEE/CVF
       Conf. on Computer Vision and Pattern Recognition (CVPR), 2025.
       https://arxiv.org/abs/2412.00505
    .. [2] Y. Qiu, A. B. Wagner, J. Ballé, L. Theis: "Wasserstein Distortion: Unifying
       Fidelity and Realism," 2024 58th Ann. Conf. on Information Sciences and Systems
       (CISS), 2024. https://arxiv.org/abs/2310.03629
    """
    if jnp.shape(image_a)[-2:] != jnp.shape(log2_sigma):
        raise ValueError(
            f"`log2_sigma` must have the same spatial shape as the images, but received "
            f"{jnp.shape(log2_sigma)} and {jnp.shape(image_a)[-2:]}, respectively."
        )
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
        norm_spatial=norm_spatial,
        norm_channels=norm_channels,
        weights=[image_weight] + [1.0] * (len(features_a) - 1),
        num_levels=num_levels,
        sqrt_grad_limit=sqrt_grad_limit,
        return_wd_map=return_wd_map,
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
    This is a Gaussianized version of the distortion loss function described in [1]_.
    Please cite this paper if you use this code for scientific or research work.

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

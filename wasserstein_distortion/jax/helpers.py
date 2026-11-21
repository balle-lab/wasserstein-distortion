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
"""Helper operations."""

import functools
import jax
import jax.numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


@jax.custom_vjp
def lower_limit(inputs: ArrayLike, limit: ArrayLike) -> Array:
    """Limits an array from below, with faked gradients.

    In contrast to `jnp.maximum`, this function never returns a gradient for `limit`, and
    for values in `inputs` which are below the limit, it returns a gradient if and only if
    that gradient is negative.

    Parameters
    ----------
    inputs
        The input array.
    limit
        Lower limit for `inputs`.

    Returns
    -------
    Array
        ``jnp.maximum(inputs, limit)``.
    """
    return jnp.maximum(inputs, limit)


def lower_limit_fwd(inputs, limit):
    """Forward pass for `lower_limit`."""
    return lower_limit(inputs, limit), (inputs >= limit,)


def lower_limit_bwd(res, grad):
    """Backward pass for `lower_limit`."""
    (limit_inactive,) = res
    pass_through_if = jnp.logical_or(limit_inactive, grad < 0.0)
    inputs_grad = jnp.where(pass_through_if, grad, 0.0)
    return inputs_grad, None


lower_limit.defvjp(lower_limit_fwd, lower_limit_bwd)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def safe_sqrt(inputs: ArrayLike, limit: float) -> Array:
    """`jnp.sqrt` with gradient capped at `limit`."""
    del limit  # unused in forward pass
    return jnp.sqrt(inputs)


@safe_sqrt.defjvp
def safe_sqrt_jvp(limit, primals, tangents):
    """Gradient for `safe_sqrt`."""
    (x,) = primals
    (x_dot,) = tangents
    sqrt_dot = jnp.minimum(0.5 / jnp.sqrt(x), limit) * x_dot
    return safe_sqrt(x, limit), sqrt_dot


def safe_pow(inputs: Array, exponent: float, limit: float) -> Array:
    """``inputs ** exponent`` for nonnegative inputs, with gradient magnitude capped.

    For ``exponent == 1``, returns `inputs` unchanged. For ``exponent < 1``, the
    gradient, which is unbounded near zero, is clipped to ``[-limit, limit]``.
    """
    if exponent == 1:
        return inputs
    return _safe_pow(inputs, exponent, limit)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _safe_pow(inputs: Array, exponent: float, limit: float) -> Array:
    """Implementation of `safe_pow`."""
    del limit  # unused in forward pass
    return jnp.power(inputs, exponent)


@_safe_pow.defjvp
def _safe_pow_jvp(exponent, limit, primals, tangents):
    """Gradient for `_safe_pow`."""
    (x,) = primals
    (x_dot,) = tangents
    grad = exponent * jnp.power(x, exponent - 1)
    if exponent < 1:
        grad = jnp.clip(grad, -limit, limit)
    return _safe_pow(x, exponent, limit), grad * x_dot


def subsampling_offset(step: int) -> tuple[int, int]:
    """Returns the subsampling offset in height and width for a pyramid step.

    The offset of each dimension alternates between 0 and 1 from step to step, so that
    the shifts of the sample positions relative to block-centered sampling partially
    cancel across pyramid levels.
    """
    return step % 2, (step + 1) % 2


def _effective_offsets(
    height: int, width: int, offset: tuple[int, int]
) -> tuple[int, int]:
    """Validates subsampling offsets, using offset 0 for dimensions of length one."""
    if any(o not in (0, 1) for o in offset):
        raise ValueError(f"`offset` must contain only 0 or 1, but received {offset}.")
    return (offset[0] if height > 1 else 0, offset[1] if width > 1 else 0)


def subsample(inputs: Array, offset: tuple[int, int] = (0, 0)) -> Array:
    """Subsamples an array of shape ``(..., height, width)`` by a factor of two.

    Parameters
    ----------
    inputs
        The input array of shape ``(..., height, width)``.
    offset
        The index of the first retained element in height and width, each 0 or 1.
        Dimensions of length one always use offset 0, so that they don't become empty.

    Returns
    -------
    Array
        ``inputs[..., offset[0]::2, offset[1]::2]``.
    """
    offset_h, offset_w = _effective_offsets(*inputs.shape[-2:], offset)
    return inputs[..., offset_h::2, offset_w::2]


def lowpass(inputs: ArrayLike, downsample: tuple[int, int] | None = None) -> Array:
    """Lowpass filters an array of shape ``(..., height, width)``.

    Parameters
    ----------
    inputs
        The input array of shape ``(..., height, width)``. Each 2D slice is filtered
        separately.
    downsample
        If ``None``, the output has the same shape as the input. Otherwise, the output
        is downsampled by a factor of two, and this is the index of the input element on
        which the first output element is centered, in height and width, each 0 or 1.
        For example, ``(0, 0)`` is consistent with ``x[..., ::2, ::2]``. The result is
        the same as ``subsample(lowpass(inputs), downsample)``, but cheaper to compute.
        Dimensions of length one always use offset 0, so that they don't become empty.

    Returns
    -------
    Array
        The lowpass filtered array of shape ``(..., height, width)``, or of shape
        ``(..., height', width')`` with about half the height and width if downsampling.
    """
    inputs = jnp.asarray(inputs)
    # Lowpass filter. Weights sum to one so that the output dynamic range is the same as
    # the input's. Its frequency response will cause some alias when downsampling, but we
    # accept that because a 3x3 filter is reasonably fast.
    kernel = jnp.array([0.25, 0.5, 0.25], dtype=inputs.dtype)
    kernel = jnp.outer(kernel, kernel)[None, None]  # shape: (1, 1, 3, 3)
    *batch, height, width = inputs.shape
    if downsample is None:
        stride, offset_h, offset_w = 1, 0, 0
    else:
        stride = 2
        offset_h, offset_w = _effective_offsets(height, width, downsample)
    # Pad explicitly rather than using "same" padding, which with stride 2 and even sizes
    # pads only at the end. Padding by one at the end, and also at the start for offset 0,
    # centers the outputs on the input elements with the given offsets.
    padding = ((1 - offset_h, 1), (1 - offset_w, 1))
    outputs = jax.lax.conv(
        inputs.reshape(-1, 1, height, width),
        kernel,
        window_strides=(stride, stride),
        padding=padding,  # pyright: ignore[reportArgumentType]
    )
    return outputs.reshape(*batch, *outputs.shape[-2:])

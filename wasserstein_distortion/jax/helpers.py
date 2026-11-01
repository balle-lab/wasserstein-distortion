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


def lowpass(inputs: ArrayLike, stride: int) -> Array:
    """Lowpass filters an array of shape ``(..., height, width)``.

    Parameters
    ----------
    inputs
        The input array of shape ``(..., height, width)``. Each 2D slice is filtered
        separately.
    stride
        The stride length of the convolution. Typically either 1 or 2.

    Returns
    -------
    Array
        The lowpass filtered array of shape ``(..., height, width)``. Height and width
        are the same as the input array if stride is 1.
    """
    inputs = jnp.asarray(inputs)
    # Lowpass filter. Weights sum to one so that the output dynamic range is the same as
    # the input's. Its frequency response will cause some alias when downsampling, but we
    # accept that because a 3x3 filter is reasonably fast.
    kernel = jnp.array([0.25, 0.5, 0.25], dtype=inputs.dtype)
    kernel = jnp.outer(kernel, kernel)[None, None]  # shape: (1, 1, 3, 3)
    # Pad symmetrically rather than using "same" padding. With stride 2 and even sizes,
    # "same" padding pads only at the end, which centers the outputs on odd pixels. This
    # keeps the outputs centered on even pixels, consistent with `x[..., ::2, ::2]`.
    *batch, height, width = inputs.shape
    outputs = jax.lax.conv(
        inputs.reshape(-1, 1, height, width),
        kernel,
        window_strides=(stride, stride),
        padding=((1, 1), (1, 1)),  # pyright: ignore[reportArgumentType]
    )
    return outputs.reshape(*batch, *outputs.shape[-2:])

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
from typing import override
import torch
from torch.nn import functional as F

Tensor = torch.Tensor


class _LowerLimit(torch.autograd.Function):
    """Limits a tensor from below, with faked gradients.

    The forward pass is ``torch.clamp(inputs, min=limit)``. In the backward pass, the
    gradient is passed through where the limit is inactive, or where the gradient is
    negative (i.e., where gradient descent would push the input back above the limit).
    """

    @override
    @staticmethod
    def forward(ctx, inputs: Tensor, limit: float) -> Tensor:  # pylint: disable=arguments-differ
        ctx.save_for_backward(inputs >= limit)
        return torch.clamp(inputs, min=limit)

    @override
    @staticmethod
    def backward(ctx, grad: Tensor):  # pylint: disable=arguments-differ
        (limit_inactive,) = ctx.saved_tensors
        pass_through = torch.logical_or(limit_inactive, grad < 0)
        return torch.where(pass_through, grad, torch.zeros_like(grad)), None


def lower_limit(inputs: Tensor, limit: float) -> Tensor:
    """Limits `inputs` from below by `limit`, with faked gradients."""
    return _LowerLimit.apply(inputs, limit)


class _SafeSqrt(torch.autograd.Function):
    """`torch.sqrt` with gradient capped at `limit`."""

    @override
    @staticmethod
    def forward(ctx, inputs: Tensor, limit: float) -> Tensor:  # pylint: disable=arguments-differ
        outputs = torch.sqrt(inputs)
        ctx.save_for_backward(outputs)
        ctx.limit = limit
        return outputs

    @override
    @staticmethod
    def backward(ctx, grad: Tensor):  # pylint: disable=arguments-differ
        (outputs,) = ctx.saved_tensors
        return torch.clamp(0.5 / outputs, max=ctx.limit) * grad, None


def safe_sqrt(inputs: Tensor, limit: float) -> Tensor:
    """`torch.sqrt` with gradient capped at `limit`."""
    return _SafeSqrt.apply(inputs, limit)


class _SafePow(torch.autograd.Function):
    """``inputs ** exponent``, with gradient magnitude capped if ``exponent < 1``."""

    @override
    @staticmethod
    def forward(ctx, inputs: Tensor, exponent: float, limit: float) -> Tensor:  # pylint: disable=arguments-differ
        ctx.save_for_backward(inputs)
        ctx.exponent = exponent
        ctx.limit = limit
        return torch.pow(inputs, exponent)

    @override
    @staticmethod
    def backward(ctx, grad: Tensor):  # pylint: disable=arguments-differ
        (inputs,) = ctx.saved_tensors
        inputs_grad = ctx.exponent * torch.pow(inputs, ctx.exponent - 1)
        if ctx.exponent < 1:
            inputs_grad = torch.clamp(inputs_grad, -ctx.limit, ctx.limit)
        return inputs_grad * grad, None, None


def safe_pow(inputs: Tensor, exponent: float, limit: float) -> Tensor:
    """``inputs ** exponent`` for nonnegative inputs, with gradient magnitude capped.

    For ``exponent == 1``, returns `inputs` unchanged. For ``exponent < 1``, the
    gradient, which is unbounded near zero, is clipped to ``[-limit, limit]``.
    """
    if exponent == 1:
        return inputs
    return _SafePow.apply(inputs, exponent, limit)


@functools.cache
def _lowpass_kernel(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Returns the lowpass kernel, cached to avoid a host-to-device copy on every call."""
    # Lowpass filter. Weights sum to one so that the output dynamic range is the same as
    # the input's. Its frequency response will cause some alias when downsampling, but we
    # accept that because a 3x3 filter is reasonably fast.
    # If first called in inference mode, the kernel must still be usable with autograd.
    with torch.inference_mode(False):
        kernel = torch.tensor([0.25, 0.5, 0.25], dtype=dtype, device=device)
        return torch.outer(kernel, kernel)[None, None]  # shape: (1, 1, 3, 3)


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


def subsample(inputs: Tensor, offset: tuple[int, int] = (0, 0)) -> Tensor:
    """Subsamples a tensor of shape ``(..., height, width)`` by a factor of two.

    Parameters
    ----------
    inputs
        The input tensor of shape ``(..., height, width)``.
    offset
        The index of the first retained element in height and width, each 0 or 1.
        Dimensions of length one always use offset 0, so that they don't become empty.

    Returns
    -------
    Tensor
        ``inputs[..., offset[0]::2, offset[1]::2]``.
    """
    offset_h, offset_w = _effective_offsets(*inputs.shape[-2:], offset)
    return inputs[..., offset_h::2, offset_w::2]


def lowpass(inputs: Tensor, downsample: tuple[int, int] | None = None) -> Tensor:
    """Lowpass filters a tensor of shape ``(..., height, width)``.

    Parameters
    ----------
    inputs
        The input tensor of shape ``(..., height, width)``. Each 2D slice is filtered
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
    Tensor
        The lowpass filtered tensor of shape ``(..., height, width)``, or of shape
        ``(..., height', width')`` with about half the height and width if downsampling.
    """
    kernel = _lowpass_kernel(inputs.device, inputs.dtype)
    *batch, height, width = inputs.shape
    inputs = inputs.reshape(-1, 1, height, width)
    if downsample is None:
        outputs = F.conv2d(inputs, kernel, padding=1)  # pylint: disable=not-callable
    else:
        offset_h, offset_w = _effective_offsets(height, width, downsample)
        # Padding by one at the end, and also at the start for offset 0, centers the
        # outputs on the input elements with the given offsets.
        if (offset_h, offset_w) == (0, 0):
            outputs = F.conv2d(  # pylint: disable=not-callable
                inputs, kernel, stride=2, padding=1
            )
        else:
            inputs = F.pad(inputs, (1 - offset_w, 1, 1 - offset_h, 1))
            outputs = F.conv2d(inputs, kernel, stride=2)  # pylint: disable=not-callable
    return outputs.reshape(*batch, *outputs.shape[-2:])

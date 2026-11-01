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


def lowpass(inputs: Tensor, stride: int) -> Tensor:
    """Lowpass filters a tensor of shape ``(..., height, width)``.

    Parameters
    ----------
    inputs
        The input tensor of shape ``(..., height, width)``. Each 2D slice is filtered
        separately.
    stride
        The stride length of the convolution. Typically either 1 or 2.

    Returns
    -------
    Tensor
        The lowpass filtered tensor of shape ``(..., height, width)``. Height and width
        are the same as the input tensor if stride is 1.
    """
    kernel = _lowpass_kernel(inputs.device, inputs.dtype)
    # Pad symmetrically, which keeps the outputs centered on even pixels, consistent with
    # `x[..., ::2, ::2]`.
    *batch, height, width = inputs.shape
    outputs = F.conv2d(  # pylint: disable=not-callable
        inputs.reshape(-1, 1, height, width), kernel, stride=stride, padding=1
    )
    return outputs.reshape(*batch, *outputs.shape[-2:])

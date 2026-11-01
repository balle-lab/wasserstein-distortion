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
"""Tests of Wasserstein Distortion metric (PyTorch)."""

# pylint: disable=wrong-import-position
from typing import override
import pytest
torch = pytest.importorskip("torch")
from torch import nn
from wasserstein_distortion.torch import helpers
from wasserstein_distortion.torch import pretrained_features
from wasserstein_distortion.torch import wasserstein


def test_lowpass_behavior():
    """Verifies that lowpass() correctly performs spatial downsampling.

    Checks
    ------
    - The spatial resolution is halved when using stride=2.
    - The value range stays between 0 and 1, the same as the input.
    """
    noise = torch.rand(1, 1, 64, 64, generator=torch.Generator().manual_seed(0))
    filtered = helpers.lowpass(noise, stride=2)
    assert filtered.shape == (1, 1, 32, 32)
    assert torch.all(filtered >= 0) and torch.all(filtered <= 1)


def test_lowpass_preserves_leading_dims_and_dtype():
    """Verifies that lowpass() filters each 2D slice separately.

    Checks
    ------
    - Arbitrary leading dimensions are preserved.
    - Each slice is filtered independently of the others.
    - The dtype is preserved.
    """
    x = torch.randn(2, 3, 4, 16, 16, dtype=torch.float64)
    filtered = helpers.lowpass(x, stride=1)
    assert filtered.shape == x.shape
    assert filtered.dtype == torch.float64
    torch.testing.assert_close(filtered[1, 2], helpers.lowpass(x[1, 2], stride=1))


def test_lowpass_preserves_constant_in_interior():
    """Verifies that the lowpass filter weights sum to one."""
    x = torch.full((1, 1, 16, 16), 3.0)
    filtered = helpers.lowpass(x, stride=1)
    torch.testing.assert_close(filtered[..., 1:-1, 1:-1], x[..., 1:-1, 1:-1])


def test_lowpass_kernel_cached_in_inference_mode_supports_autograd():
    """Verifies that a kernel first cached in inference mode still works with autograd."""
    helpers._lowpass_kernel.cache_clear()  # pylint: disable=protected-access  # pyright: ignore[reportPrivateUsage]
    with torch.inference_mode():
        helpers.lowpass(torch.ones(1, 8, 8), stride=1)
    x = torch.ones(1, 8, 8, requires_grad=True)
    helpers.lowpass(x, stride=1).sum().backward()
    assert x.grad is not None


def test_lower_limit_forward_and_gradient():
    """Verifies the forward pass and the faked gradients of lower_limit().

    Checks
    ------
    - The forward pass clamps from below.
    - Gradients pass through where the limit is inactive.
    - Where the limit is active, only negative gradients pass through.
    """
    x = torch.tensor([-1.0, -1.0, 1.0, 1.0], requires_grad=True)
    y = helpers.lower_limit(x, 0.0)
    torch.testing.assert_close(y, torch.tensor([0.0, 0.0, 1.0, 1.0]))
    y.backward(torch.tensor([1.0, -1.0, 1.0, -1.0]))
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.tensor([0.0, -1.0, 1.0, -1.0]))


def test_safe_sqrt_forward_and_gradient():
    """Verifies the forward pass and the capped gradient of safe_sqrt().

    Checks
    ------
    - The forward pass equals the square root.
    - The gradient equals that of the square root where it is below the limit.
    - The gradient is capped at the limit, including at zero.
    """
    x = torch.tensor([0.0, 1e-8, 4.0], requires_grad=True)
    y = helpers.safe_sqrt(x, 100.0)
    torch.testing.assert_close(y, torch.sqrt(x.detach()))
    y.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.tensor([100.0, 100.0, 0.25]))


def test_compute_multiscale_stats_behavior():
    """Verifies that compute_multiscale_stats() behaves correctly on constant input.

    Checks
    ------
    - Variance is near-zero at all pyramid levels (away from the borders).
    - The mean equals the constant (away from the borders).
    - The pyramid resolution halves at each level as expected.
    """
    constant_value = 5.0
    x = torch.full((1, 1, 64, 64), constant_value)

    means, variances = wasserstein.compute_multiscale_stats(x, num_levels=3)

    assert len(means) == len(variances) == 3
    for level, (m, v) in enumerate(zip(means, variances)):
        size = 64 // 2**level
        assert m.shape == v.shape == (1, 1, size, size)
        # Zero padding affects one more border pixel at each level.
        margin = level + 1
        interior = (..., slice(margin, -margin), slice(margin, -margin))
        torch.testing.assert_close(
            v[interior], torch.zeros_like(v[interior]), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            m[interior], torch.full_like(m[interior], constant_value)
        )


def test_compute_multiscale_stats_non_constant():
    """Verifies variance computation on non-uniform input (checkerboard pattern).

    Checks
    ------
    - Variance is non-negative (up to numerics) and positive somewhere at each level.
    - The spatial resolution halves at each level.
    """
    x = torch.zeros(1, 1, 64, 64)
    x[..., 0::2, 0::2] = 1.0
    x[..., 1::2, 1::2] = 1.0

    means, variances = wasserstein.compute_multiscale_stats(x, num_levels=3)

    for level in range(3):
        assert torch.all(variances[level] >= -1e-5)
        assert torch.any(variances[level] > 1e-2)
        size = 64 // 2**level
        assert means[level].shape == (1, 1, size, size)


def test_wasserstein_distortion_behavior():
    """Verifies the correct behavior of wasserstein_distortion().

    Checks
    ------
    - The distance is zero when comparing identical features.
    - The distance is positive when comparing different features.
    - The function raises an error if the feature shapes don't match.
    - The function raises an error if the sigma map shape doesn't match.
    """
    features_a = torch.ones(1, 1, 32, 32)
    log2_sigma = torch.zeros(1, 1, 32, 32)

    result = wasserstein.wasserstein_distortion(features_a, features_a, log2_sigma)
    assert result.shape == ()
    assert result == 0.0

    result = wasserstein.wasserstein_distortion(features_a, 2 * features_a, log2_sigma)
    assert result > 0.0

    with pytest.raises(ValueError):
        wasserstein.wasserstein_distortion(
            features_a, torch.ones(1, 1, 16, 16), log2_sigma
        )
    with pytest.raises(ValueError):
        wasserstein.wasserstein_distortion(
            features_a, features_a, torch.zeros(1, 1, 16, 16)
        )


def test_wasserstein_distortion_sigma_zero_is_mse():
    """Verifies that with zero sigma, the distortion reduces to mean squared error."""
    generator = torch.Generator().manual_seed(0)
    features_a = torch.randn(2, 4, 16, 16, generator=generator, dtype=torch.float64)
    features_b = torch.randn(2, 4, 16, 16, generator=generator, dtype=torch.float64)
    log2_sigma = torch.zeros(2, 1, 16, 16, dtype=torch.float64)

    result = wasserstein.wasserstein_distortion(features_a, features_b, log2_sigma)
    torch.testing.assert_close(result, torch.mean(torch.square(features_a - features_b)))


def test_wasserstein_distortion_is_symmetric():
    """Verifies that swapping the two feature tensors doesn't change the distortion."""
    generator = torch.Generator().manual_seed(0)
    features_a = torch.randn(1, 4, 16, 16, generator=generator)
    features_b = torch.randn(1, 4, 16, 16, generator=generator)
    log2_sigma = 4 * torch.rand(1, 1, 16, 16, generator=generator)

    result_ab = wasserstein.wasserstein_distortion(features_a, features_b, log2_sigma)
    result_ba = wasserstein.wasserstein_distortion(features_b, features_a, log2_sigma)  # pylint: disable=arguments-out-of-order
    torch.testing.assert_close(result_ab, result_ba)


def test_wasserstein_distortion_batch_is_mean_over_examples():
    """Verifies that a batched call equals the mean of per-example calls."""
    generator = torch.Generator().manual_seed(0)
    features_a = torch.randn(3, 4, 16, 16, generator=generator, dtype=torch.float64)
    features_b = torch.randn(3, 4, 16, 16, generator=generator, dtype=torch.float64)
    log2_sigma = 4 * torch.rand(3, 1, 16, 16, generator=generator, dtype=torch.float64)

    batched = wasserstein.wasserstein_distortion(features_a, features_b, log2_sigma)
    individual = [
        wasserstein.wasserstein_distortion(
            features_a[i : i + 1], features_b[i : i + 1], log2_sigma[i : i + 1]
        )
        for i in range(3)
    ]
    torch.testing.assert_close(batched, torch.stack(individual).mean())


def test_wasserstein_distortion_gradients_are_finite():
    """Verifies that gradients are finite even where feature variances are zero."""
    features_a = torch.ones(1, 2, 16, 16)
    features_b = torch.ones(1, 2, 16, 16, requires_grad=True)
    log2_sigma = torch.full((1, 1, 16, 16), 3.0)

    wasserstein.wasserstein_distortion(features_a, features_b, log2_sigma).backward()
    assert features_b.grad is not None
    assert torch.all(torch.isfinite(features_b.grad))


def test_multi_wasserstein_distortion_behavior():
    """Verifies the correct behavior of multi_wasserstein_distortion().

    Checks
    ------
    - The distance is zero when features are identical.
    - The distance is positive when features differ.
    - The function raises an error if:
        - The list lengths differ.
        - Any feature shape in the lists doesn't match.
    """
    features_a = [torch.ones(1, 1, 32, 32), torch.ones(1, 1, 16, 16)]
    log2_sigma = torch.zeros(1, 1, 64, 64)

    result = wasserstein.multi_wasserstein_distortion(features_a, features_a, log2_sigma)
    assert result == 0.0

    features_b = [2 * f for f in features_a]
    result = wasserstein.multi_wasserstein_distortion(features_a, features_b, log2_sigma)
    assert result > 0.0

    with pytest.raises(ValueError):
        wasserstein.multi_wasserstein_distortion(
            features_a, features_a[:1], log2_sigma
        )
    with pytest.raises(ValueError):
        wasserstein.multi_wasserstein_distortion(
            features_a, [torch.ones(1, 1, 32, 32), torch.ones(1, 1, 8, 8)], log2_sigma
        )


def test_multi_wasserstein_sigma_scaling(monkeypatch):
    """Validates sigma map scaling for feature tensors with different resolutions.

    Checks
    ------
    - If a feature tensor is smaller (e.g., 32x32 vs. original 64x64 sigma map),
      log2_sigma is adjusted by subtracting log2(size_ratio).
    """
    log2_sigma = torch.full((1, 1, 64, 64), 4.0)
    features = [torch.ones(1, 1, 32, 32)]
    # Expected adjusted sigma: 4 - log2(64/32) = 3
    expected_ls = torch.full((1, 1, 32, 32), 3.0)
    calls = []

    def check_sigma(fa, fb, ls, **kwargs):
        del fa, fb, kwargs
        calls.append(ls)
        return torch.zeros(())

    monkeypatch.setattr(wasserstein, "wasserstein_distortion", check_sigma)
    wasserstein.multi_wasserstein_distortion(features, features, log2_sigma)

    assert len(calls) == 1
    torch.testing.assert_close(calls[0], expected_ls, atol=0.1, rtol=0)


class _PixelsAndPooled(nn.Module):
    """Simple feature extractor for testing."""

    @override
    def forward(self, image):
        return [image, nn.functional.avg_pool2d(image, 2)]  # pylint: disable=not-callable


def test_wasserstein_distortion_module_with_custom_features():
    """Verifies the WassersteinDistortion module with a custom feature extractor.

    Checks
    ------
    - The result equals multi_wasserstein_distortion() on the extracted features.
    - The feature extractor is registered as a submodule.
    - The module raises an error if the image shapes don't match.
    """
    generator = torch.Generator().manual_seed(0)
    image_a = torch.rand(2, 3, 32, 32, generator=generator)
    image_b = torch.rand(2, 3, 32, 32, generator=generator)
    log2_sigma = 3 * torch.rand(2, 1, 32, 32, generator=generator)
    features = _PixelsAndPooled()

    module = wasserstein.WassersteinDistortion(features, num_levels=4)
    result = module(image_a, image_b, log2_sigma)
    expected = wasserstein.multi_wasserstein_distortion(
        features(image_a), features(image_b), log2_sigma, num_levels=4
    )
    torch.testing.assert_close(result, expected)

    assert dict(module.named_children())["features"] is features

    with pytest.raises(ValueError):
        module(image_a, image_b[..., :16], log2_sigma)


def test_wasserstein_distortion_module_accepts_plain_function():
    """Verifies that the feature extractor can be a plain function."""
    image = torch.rand(1, 3, 16, 16)
    module = wasserstein.WassersteinDistortion(lambda x: [x])
    result = module(image, image, torch.zeros(1, 1, 16, 16))
    assert result == 0.0


@pytest.mark.parametrize("num_layers", [1, 3, 5])
@pytest.mark.parametrize("num_scales", [1, 2])
def test_vgg16_features_shapes(num_layers, num_scales):
    """Verifies the number and shapes of the VGG-16 features (random weights).

    Checks
    ------
    - The image itself is returned first, followed by `num_layers` features per scale.
    - Channels and spatial sizes match the VGG-16 architecture.
    """
    features = pretrained_features.VGG16Features(
        num_scales=num_scales, num_layers=num_layers, pretrained=False
    )
    image = torch.rand(2, 3, 64, 64)
    outputs = features(image)

    assert len(outputs) == 1 + num_layers * num_scales
    channels = [64, 128, 256, 512, 512]
    for scale in range(num_scales):
        for layer in range(num_layers):
            size = 64 // 2 ** (scale + layer)
            output = outputs[1 + scale * num_layers + layer]
            assert output.shape == (2, channels[layer], size, size)


def test_vgg16_features_configuration():
    """Verifies the configuration of the VGG-16 feature extractor (random weights).

    Checks
    ------
    - Parameters don't require gradients.
    - Max pooling is replaced by average pooling, unless disabled.
    - An invalid number of layers raises an error.
    """
    features = pretrained_features.VGG16Features(pretrained=False)
    assert not any(p.requires_grad for p in features.parameters())
    modules = list(features.modules())
    assert not any(isinstance(m, nn.MaxPool2d) for m in modules)
    assert any(isinstance(m, nn.AvgPool2d) for m in modules)

    features = pretrained_features.VGG16Features(avg_pooling=False, pretrained=False)
    modules = list(features.modules())
    assert any(isinstance(m, nn.MaxPool2d) for m in modules)
    assert not any(isinstance(m, nn.AvgPool2d) for m in modules)

    for num_layers in [0, 6]:
        with pytest.raises(ValueError):
            pretrained_features.VGG16Features(num_layers=num_layers, pretrained=False)


def test_vgg16_wasserstein_distortion_can_be_called():
    """Tests that VGG16 Wasserstein Distortion can be computed and differentiated."""
    module = wasserstein.VGG16WassersteinDistortion(pretrained=False)
    image_a = torch.rand(1, 3, 64, 64)
    image_b = torch.rand(1, 3, 64, 64, requires_grad=True)
    log2_sigma = torch.full((1, 1, 64, 64), 3.0)

    assert module(image_a, image_a, log2_sigma) == 0.0

    dist = module(image_a, image_b, log2_sigma)
    assert dist > 0.0
    dist.backward()
    assert image_b.grad is not None
    assert torch.all(torch.isfinite(image_b.grad))

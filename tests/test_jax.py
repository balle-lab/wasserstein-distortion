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
"""Tests of Wasserstein Distortion metric (JAX)."""

# pylint: disable=wrong-import-position
import pytest
jax = pytest.importorskip("jax")
import jax.numpy as jnp
from wasserstein_distortion.jax import helpers
from wasserstein_distortion.jax import pretrained_features
from wasserstein_distortion.jax import wasserstein


def test_lowpass_behavior():
    """Verifies that lowpass() correctly performs spatial downsampling.

    Checks
    ------
    - The spatial resolution is halved when using stride=2.
    - The value range stays between 0 and 1, the same as the input.
    """
    # Create random noise [0, 1]
    noise = jax.random.uniform(jax.random.key(0), (1, 64, 64))
    # Apply lowpass with stride=2
    filtered = helpers.lowpass(noise, stride=2)

    # Check shape halved
    assert filtered.shape == (1, 32, 32)

    # Check values still in [0, 1]
    assert jnp.all(filtered >= 0) and jnp.all(filtered <= 1)


def test_lowpass_preserves_leading_dims():
    """Verifies that lowpass() filters each 2D slice separately.

    Checks
    ------
    - Arbitrary leading dimensions are preserved.
    - Each slice is filtered independently of the others.
    """
    x = jax.random.normal(jax.random.key(0), (2, 3, 4, 16, 16))
    filtered = helpers.lowpass(x, stride=1)
    assert filtered.shape == x.shape
    assert jnp.allclose(filtered[1, 2], helpers.lowpass(x[1, 2], stride=1))
    assert helpers.lowpass(x, stride=2).shape == (2, 3, 4, 8, 8)


def test_lowpass_preserves_constant_in_interior():
    """Verifies that the lowpass filter weights sum to one."""
    x = jnp.full((1, 16, 16), 3.0)
    filtered = helpers.lowpass(x, stride=1)
    assert jnp.allclose(filtered[..., 1:-1, 1:-1], 3.0)


def test_lower_limit_forward_and_gradient():
    """Verifies the forward pass and the faked gradients of lower_limit().

    Checks
    ------
    - The forward pass clamps from below.
    - Gradients pass through where the limit is inactive.
    - Where the limit is active, only negative gradients pass through.
    """
    x = jnp.array([-1.0, -1.0, 1.0, 1.0])
    y, vjp = jax.vjp(lambda x: helpers.lower_limit(x, 0.0), x)
    assert jnp.array_equal(y, jnp.array([0.0, 0.0, 1.0, 1.0]))
    (grad,) = vjp(jnp.array([1.0, -1.0, 1.0, -1.0]))
    assert jnp.array_equal(grad, jnp.array([0.0, -1.0, 1.0, -1.0]))


def test_safe_sqrt_forward_and_gradient():
    """Verifies the forward pass and the capped gradient of safe_sqrt().

    Checks
    ------
    - The forward pass equals the square root.
    - The gradient equals that of the square root where it is below the limit.
    - The gradient is capped at the limit, including at zero.
    """
    x = jnp.array([0.0, 1e-8, 4.0])
    y, grad = jax.value_and_grad(lambda x: helpers.safe_sqrt(x, 100.0).sum())(x)
    assert jnp.allclose(y, jnp.sqrt(x).sum())
    assert jnp.allclose(grad, jnp.array([100.0, 100.0, 0.25]))


def test_compute_multiscale_stats_behavior():
    """Verifies that compute_multiscale_stats() behaves correctly.

    Verifies that compute_multiscale_stats(): Correctly generates a pyramid of decreasing
    resolution. Preserves spatial uniformity (low variance) when given a constant input.

    Checks
    ------
    - Variance is near-zero at all pyramid levels.
    - The output stays spatially uniform, regardless of any global mean shift.
    - The pyramid resolution halves at each level as expected.
    """
    # Constant input
    constant_value = 5.0
    constant_array = jnp.full((1, 64, 64), constant_value, dtype=jnp.float32)

    means, variances = wasserstein.compute_multiscale_stats(constant_array, num_levels=3)

    assert len(means) == len(variances) == 3
    for level, (m, v) in enumerate(zip(means, variances)):
        # Check shape is halved at each level
        size = 64 // 2**level
        assert m.shape == v.shape == (1, size, size)
        # Zero padding affects one more border pixel at each level.
        margin = level + 1
        interior = (..., slice(margin, -margin), slice(margin, -margin))
        # Variance should be zero
        assert jnp.allclose(v[interior], 0.0, atol=1e-5)
        # Mean should equal the constant value
        assert jnp.allclose(m[interior], constant_value, atol=1e-5)


def test_wasserstein_distortion_behavior():
    """Verifies the correct behavior of wasserstein_distortion().

    Checks
    ------
    - The distance is zero when comparing identical features.
    - The distance is positive when comparing different features.
    - The function raises an error if the feature shapes don't match.
    """
    # Create two identical feature arrays
    feature_a = jnp.ones((1, 32, 32))
    feature_b = jnp.ones((1, 32, 32))

    log2_sigma = jnp.zeros((32, 32))

    # Should return 0 for identical features
    result = wasserstein.wasserstein_distortion(feature_a, feature_b, log2_sigma)
    assert jnp.isclose(result, 0.0)

    # Create different features
    feature_b_different = jnp.ones((1, 32, 32)) * 2.0

    # Should return positive value
    result = wasserstein.wasserstein_distortion(
        feature_a, feature_b_different, log2_sigma
    )
    assert result > 0

    # Should raise ValueError for shape mismatch of features
    with pytest.raises(ValueError):
        _ = wasserstein.wasserstein_distortion(
            feature_a, jnp.ones((1, 16, 16)), log2_sigma
        )

    # Should raise ValueError for shape mismatch of sigma map
    with pytest.raises(ValueError):
        _ = wasserstein.wasserstein_distortion(
            feature_a, feature_b, jnp.zeros((16, 16))
        )


def test_wasserstein_distortion_sigma_zero_is_mse():
    """Verifies that with zero sigma, the distortion reduces to mean squared error."""
    key_a, key_b = jax.random.split(jax.random.key(0))
    features_a = jax.random.normal(key_a, (4, 16, 16))
    features_b = jax.random.normal(key_b, (4, 16, 16))
    log2_sigma = jnp.zeros((16, 16))

    result = wasserstein.wasserstein_distortion(features_a, features_b, log2_sigma)
    assert jnp.allclose(result, jnp.mean(jnp.square(features_a - features_b)))


def test_wasserstein_distortion_is_symmetric():
    """Verifies that swapping the two feature arrays doesn't change the distortion."""
    key_a, key_b, key_s = jax.random.split(jax.random.key(0), 3)
    features_a = jax.random.normal(key_a, (4, 16, 16))
    features_b = jax.random.normal(key_b, (4, 16, 16))
    log2_sigma = 4 * jax.random.uniform(key_s, (16, 16))

    result_ab = wasserstein.wasserstein_distortion(features_a, features_b, log2_sigma)
    result_ba = wasserstein.wasserstein_distortion(features_b, features_a, log2_sigma)  # pylint: disable=arguments-out-of-order
    assert jnp.allclose(result_ab, result_ba)


def test_wasserstein_distortion_gradients_are_finite():
    """Verifies that gradients are finite even where feature variances are zero."""
    features = jnp.ones((2, 16, 16))
    log2_sigma = jnp.full((16, 16), 3.0)

    grad = jax.grad(wasserstein.wasserstein_distortion, argnums=1)(
        features, features, log2_sigma
    )
    assert jnp.all(jnp.isfinite(grad))


def test_multi_wasserstein_distortion_behavior():
    """Verifies the correct behavior of multi_wasserstein_distortion().

    Checks
    ------
    - The distance is non-negative (ideally zero) when features are identical.
    - The distance is positive when features differ.
    - The function raises an error if:
        - The list lengths differ.
        - Any feature shape in the lists doesn't match.
    """
    # Matching feature lists
    features_a = [jnp.ones((1, 32, 32)), jnp.ones((1, 16, 16))]
    features_b_same = [jnp.ones((1, 32, 32)), jnp.ones((1, 16, 16))]

    log2_sigma = jnp.zeros((64, 64))

    # Should return 0 for identical lists
    result = wasserstein.multi_wasserstein_distortion(
        features_a, features_b_same, log2_sigma
    )
    assert result == 0.

    # Different features
    features_b_different = [jnp.ones((1, 32, 32)) * 2.0, jnp.ones((1, 16, 16)) * 2.0]
    result = wasserstein.multi_wasserstein_distortion(
        features_a, features_b_different, log2_sigma
    )
    assert result > 0.

    # Should raise ValueError for list length mismatch
    with pytest.raises(ValueError):
        _ = wasserstein.multi_wasserstein_distortion(
            features_a, [jnp.ones((1, 32, 32))], log2_sigma
        )

    # Should raise ValueError for shape mismatch in one of the pairs
    with pytest.raises(ValueError):
        _ = wasserstein.multi_wasserstein_distortion(
            features_a, [jnp.ones((1, 32, 32)), jnp.ones((1, 8, 8))], log2_sigma
        )


def test_multi_wasserstein_distortion_weights():
    """Verifies the weighting of feature arrays in multi_wasserstein_distortion().

    Checks
    ------
    - The result equals the weighted sum of the per-array distortions.
    - Unit weights give the same result as no weights.
    - The function raises an error if the number of weights is wrong.
    """
    keys = jax.random.split(jax.random.key(0), 5)
    shapes = [(2, 32, 32), (2, 16, 16)]
    features_a = [jax.random.uniform(k, s) for k, s in zip(keys[:2], shapes)]
    features_b = [jax.random.uniform(k, s) for k, s in zip(keys[2:4], shapes)]
    log2_sigma = 2 * jax.random.uniform(keys[4], (32, 32))

    parts = [
        wasserstein.multi_wasserstein_distortion([fa], [fb], log2_sigma)
        for fa, fb in zip(features_a, features_b)
    ]
    result = wasserstein.multi_wasserstein_distortion(
        features_a, features_b, log2_sigma, weights=[0.5, 3.0]
    )
    assert jnp.allclose(result, 0.5 * parts[0] + 3.0 * parts[1])

    unweighted = wasserstein.multi_wasserstein_distortion(
        features_a, features_b, log2_sigma
    )
    result = wasserstein.multi_wasserstein_distortion(
        features_a, features_b, log2_sigma, weights=[1.0, 1.0]
    )
    assert jnp.allclose(result, unweighted)

    with pytest.raises(ValueError):
        _ = wasserstein.multi_wasserstein_distortion(
            features_a, features_b, log2_sigma, weights=[1.0]
        )


def test_compute_multiscale_stats_non_constant():
    """Verifies variance computation on non-uniform input (checkerboard pattern).

    This is a high-variance pattern with sharp spatial contrast, perfect for testing
    whether compute_multiscale_stats detects local variance.

    Checks
    ------
    - Variance Positivity: Ensures variance is non-negative at all pyramid levels.
    - Shape Halving: Validates spatial resolution reduction at each level (e.g., 64x64 →
      32x32 → 16x16).
    """
    # Create a checkerboard pattern
    input_arr = jnp.zeros((1, 64, 64))
    input_arr = input_arr.at[:, 0::2, 0::2].set(1.0)
    input_arr = input_arr.at[:, 1::2, 1::2].set(1.0)

    means, variances = wasserstein.compute_multiscale_stats(input_arr, num_levels=3)

    for level in range(3):
        # Check variance is positive and reasonable
        assert jnp.all(variances[level] >= -1e-5)
        assert jnp.any(variances[level] > 1e-2)
        # Check shape halving
        expected_shape = (1, 64 // (2**level), 64 // (2**level))
        assert means[level].shape == expected_shape


def test_wasserstein_distortion_intermediates():
    """Test if intermediate results are returned when return_intermediates=True.

    Checks
    ------
    - Dictionary Structure: Ensures intermediates include wd_maps (Wasserstein distortion
      maps).
    - Distortion Value: Checks that identical features yield zero distortion.
    """
    features = jnp.ones((1, 32, 32))
    log2_sigma = jnp.zeros((32, 32))

    dist, intermediates = wasserstein.wasserstein_distortion(
        features, features, log2_sigma, return_intermediates=True
    )

    assert isinstance(intermediates, dict), "Intermediates should be a dict"
    assert "wd_maps" in intermediates, "Missing wd_maps in intermediates"
    assert len(intermediates["wd_maps"]) == 5 + 1, "Incorrect number of wd_maps"
    assert jnp.isclose(dist, 0.0), "Distortion should be zero for identical features"


def test_wasserstein_distortion_jit_compilable():
    """Verify that wasserstein_distortion can be compiled with jax.jit.

    No errors should occur due to jnp.max(log2_sigma) check.
    """
    features = jnp.ones((1, 32, 32))
    log2_sigma = jnp.zeros((32, 32))  # max = 0, should not raise ValueError

    # JIT compile
    compiled_fn = jax.jit(
        wasserstein.wasserstein_distortion, static_argnames=["num_levels"]
    )

    # Should not raise any error
    result = compiled_fn(features, features, log2_sigma, num_levels=3)
    assert jnp.isclose(result, 0.0)


def test_num_levels_handling():
    """Validates error handling for insufficient num_levels.

    Checks
    ------
    - Valid Case: num_levels=3 works when log2_sigma has a max value of 2.
    - Invalid Case: num_levels=1 triggers an error when log2_sigma exceeds it.
    """
    features = jnp.ones((1, 32, 32))
    log2_sigma = jnp.full((32, 32), 3.0)

    # num_levels=3 should handle this without error
    dist = wasserstein.wasserstein_distortion(
        features, features, log2_sigma, num_levels=3
    )
    assert jnp.isclose(dist, 0.0)

    # num_levels=2 should fail
    with pytest.raises(ValueError):
        _ = wasserstein.wasserstein_distortion(
            features, features, log2_sigma, num_levels=2
        )


def test_multi_wasserstein_sigma_scaling(monkeypatch):
    """Validates sigma map scaling for feature arrays with different resolutions.

    Checks
    ------
    - Scaling Logic: If a feature array is smaller (e.g., 32x32 vs. original 64x64 sigma
      map), log2_sigma is adjusted by subtracting log2(size_ratio).
    """

    # Original sigma map (64x64)
    log2_sigma = jnp.full((64, 64), 4.0)
    # Feature with smaller spatial dim (32x32)
    features = [jnp.ones((1, 32, 32))]
    # Expected adjusted sigma: 4 - log2(64/32) = 3
    expected_ls = jnp.full((32, 32), 3.0)

    def check_sigma(fa, fb, ls, **kwargs):
        del fa, fb, kwargs
        assert jnp.allclose(ls, expected_ls, atol=0.1), "Sigma not scaled correctly"
        return jnp.zeros(()), {"dummy": []}  # Return a dummy intermediates dict

    # Monkeypatch to intercept call to wasserstein_distortion
    monkeypatch.setattr(wasserstein, "wasserstein_distortion", check_sigma)

    # Run function under test
    _ = wasserstein.multi_wasserstein_distortion(
        features, features, log2_sigma, return_intermediates=False
    )


@pytest.mark.parametrize("num_scales", [1, 2])
def test_vgg16_features_shapes(num_scales):
    """Verifies the number and shapes of the VGG-16 features (random weights).

    Checks
    ------
    - The image itself is returned first, followed by 5 features per scale.
    - Channels and spatial sizes match the VGG-16 architecture.
    """
    pretrained_features.load_vgg16_model(pretrained=False)
    image = jax.random.uniform(jax.random.key(0), (3, 64, 64))
    outputs = pretrained_features.compute_vgg16_features(image, num_scales=num_scales)

    assert len(outputs) == 1 + 5 * num_scales
    assert outputs[0] is image
    channels = [64, 128, 256, 512, 512]
    for scale in range(num_scales):
        for layer in range(5):
            size = 64 // 2 ** (scale + layer)
            output = outputs[1 + scale * 5 + layer]
            assert output.shape == (channels[layer], size, size)


def test_vgg16_wasserstein_distortion_can_be_called():
    """Tests that VGG16 Wasserstein Distortion can be computed and differentiated."""
    pretrained_features.load_vgg16_model(pretrained=False)

    key_a, key_b = jax.random.split(jax.random.key(0))
    image_a = jax.random.uniform(key_a, (3, 64, 64))
    image_b = jax.random.uniform(key_b, (3, 64, 64))
    log2_sigma = jnp.full((64, 64), 3.0)

    dist = wasserstein.vgg16_wasserstein_distortion(image_a, image_a, log2_sigma)
    assert jnp.isclose(dist, 0.0)

    dist, grad = jax.value_and_grad(wasserstein.vgg16_wasserstein_distortion, argnums=1)(
        image_a, image_b, log2_sigma
    )
    assert dist > 0.0
    assert jnp.all(jnp.isfinite(grad))


def test_vgg16_wasserstein_distortion_image_weight():
    """Verifies that `image_weight` only scales the distortion on the image itself."""
    pretrained_features.load_vgg16_model(pretrained=False)

    key_a, key_b = jax.random.split(jax.random.key(0))
    image_a = jax.random.uniform(key_a, (3, 64, 64))
    image_b = jax.random.uniform(key_b, (3, 64, 64))
    log2_sigma = jnp.full((64, 64), 3.0)

    base = wasserstein.vgg16_wasserstein_distortion(
        image_a, image_b, log2_sigma, num_scales=2, image_weight=1.0
    )
    weighted = wasserstein.vgg16_wasserstein_distortion(
        image_a, image_b, log2_sigma, num_scales=2, image_weight=3.0
    )
    pixel_dist = wasserstein.multi_wasserstein_distortion(
        [image_a], [image_b], log2_sigma
    )
    assert jnp.allclose(weighted, base + 2.0 * pixel_dist, rtol=1e-5)


def test_discrete_wasserstein_distortion_behavior():
    """Tests that discrete Wasserstein Distortion returns expected results."""
    # Matching feature lists
    features_a = [jnp.ones((1, 32, 32)), jnp.ones((1, 16, 16))]
    features_b_same = [jnp.ones((1, 32, 32)), jnp.ones((1, 16, 16))]

    centers_w = jnp.array([0.9, 0.1])
    centers_h = jnp.array([0.1, 0.9])
    sigmas = [jnp.array([0.25, 0.5]), jnp.array([0.25, 0.5])]

    # Should return 0 for identical lists
    result = wasserstein.discrete_wasserstein_distortion(
        features_a, features_b_same, centers_w, centers_h, sigmas
    )
    assert result == 0.

    # Different features
    features_b_different = [jnp.ones((1, 32, 32)) * 2.0, jnp.ones((1, 16, 16)) * 2.0]
    result = wasserstein.discrete_wasserstein_distortion(
        features_a, features_b_different, centers_w, centers_h, sigmas
    )
    assert result > 0.

    # Should raise ValueError for list length mismatch
    with pytest.raises(ValueError):
        _ = wasserstein.discrete_wasserstein_distortion(
            features_a, features_b_different[:-1], centers_w, centers_h, sigmas
        )
    with pytest.raises(ValueError):
        _ = wasserstein.discrete_wasserstein_distortion(
            features_a, features_b_same, centers_w, centers_h, sigmas[:-1]
        )

    # Should raise ValueError for shape mismatch in one of the pairs
    with pytest.raises(ValueError):
        _ = wasserstein.discrete_wasserstein_distortion(
            features_a,
            [jnp.ones((1, 32, 32)), jnp.ones((1, 8, 8))],
            centers_w,
            centers_h,
            sigmas,
        )

    # Should raise ValueError for non-vector arguments
    with pytest.raises(ValueError):
        _ = wasserstein.discrete_wasserstein_distortion(
            features_a,
            features_b_same,
            centers_w[None],
            centers_h,
            sigmas,
        )

    # Should raise ValueError for different vector lengths
    with pytest.raises(ValueError):
        _ = wasserstein.discrete_wasserstein_distortion(
            features_a,
            features_b_same,
            centers_w,
            centers_h,
            2 * [jnp.array([0.25, 0.5, 1.0])],
        )

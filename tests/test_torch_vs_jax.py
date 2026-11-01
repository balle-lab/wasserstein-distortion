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
"""Tests that PyTorch and JAX implementations of Wasserstein Distortion agree."""

# pylint: disable=wrong-import-position
import pathlib
import numpy as np
import pytest
jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")
import jax.numpy as jnp
from wasserstein_distortion import jax as wd_jax
from wasserstein_distortion import torch as wd_torch

EXAMPLE_DIR = pathlib.Path(__file__).parent.parent / "example"


def random_features(seed, shape):
    """Returns random features as float64 numpy array."""
    return np.random.default_rng(seed).normal(size=shape)


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("size", [(16, 16), (15, 17)])
def test_lowpass_matches(stride, size):
    """Compares lowpass filters, including odd sizes."""
    x = random_features(0, (3, *size)).astype(np.float32)
    out_jax = wd_jax.lowpass(jnp.array(x), stride)
    out_torch = wd_torch.lowpass(torch.from_numpy(x), stride)
    np.testing.assert_allclose(out_torch.numpy(), np.asarray(out_jax), atol=1e-6)


def test_wasserstein_distortion_matches():
    """Compares distortion values and gradients on random features."""
    fa = random_features(0, (4, 32, 32)).astype(np.float32)
    fb = random_features(1, (4, 32, 32)).astype(np.float32)
    ls = (4 * np.random.default_rng(2).uniform(size=(32, 32))).astype(np.float32)

    loss_jax, grad_jax = jax.value_and_grad(wd_jax.wasserstein_distortion, argnums=1)(
        jnp.array(fa), jnp.array(fb), jnp.array(ls)
    )

    fb_torch = torch.from_numpy(fb[None]).requires_grad_(True)
    loss_torch = wd_torch.wasserstein_distortion(
        torch.from_numpy(fa[None]), fb_torch, torch.from_numpy(ls[None, None])
    )
    loss_torch.backward()

    np.testing.assert_allclose(loss_torch.item(), loss_jax.item(), rtol=1e-5)
    assert fb_torch.grad is not None
    np.testing.assert_allclose(
        fb_torch.grad[0].numpy(), np.asarray(grad_jax), rtol=1e-4, atol=1e-8
    )


def test_multi_wasserstein_distortion_matches():
    """Compares distortion values for features at multiple resolutions."""
    shapes = [(3, 64, 48), (8, 32, 24), (16, 16, 12)]
    fa = [random_features(i, s).astype(np.float32) for i, s in enumerate(shapes)]
    fb = [random_features(i + 10, s).astype(np.float32) for i, s in enumerate(shapes)]
    ls = (5 * np.random.default_rng(20).uniform(size=(64, 48))).astype(np.float32)

    loss_jax = wd_jax.multi_wasserstein_distortion(
        [jnp.array(f) for f in fa], [jnp.array(f) for f in fb], jnp.array(ls)
    )
    loss_torch = wd_torch.multi_wasserstein_distortion(
        [torch.from_numpy(f[None]) for f in fa],
        [torch.from_numpy(f[None]) for f in fb],
        torch.from_numpy(ls[None, None]),
    )
    np.testing.assert_allclose(loss_torch.item(), loss_jax.item(), rtol=1e-4)


def load_image(path):
    """Loads an image as a float array of shape (1, C, H, W) in [0, 1]."""
    image = np.asarray(Image.open(path), dtype=np.float32) / 255.0
    return image.transpose((2, 0, 1))[None]


def test_torch_matches_jax_full():
    """Compares PyTorch implementation to JAX implementation."""
    img1 = load_image(EXAMPLE_DIR / "reference.png")
    img2 = load_image(EXAMPLE_DIR / "example.png")
    log2_sigma = np.full_like(img1[:, 0:1], 2.0)

    wd_jax.load_vgg16_model()
    loss_jax = wd_jax.vgg16_wasserstein_distortion(
        jnp.array(img1[0]), jnp.array(img2[0]), jnp.array(log2_sigma[0, 0]), num_scales=3
    )

    wloss_torch = wd_torch.VGG16WassersteinDistortion(num_scales=3)
    loss_torch = wloss_torch(
        torch.from_numpy(img1),
        torch.from_numpy(img2),
        torch.from_numpy(log2_sigma),
    )

    np.testing.assert_allclose(loss_torch.item(), loss_jax.item(), rtol=1e-3)

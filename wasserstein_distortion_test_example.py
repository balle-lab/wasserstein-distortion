# Copyright 2025 Yueyu Hu, Jona Ballé.
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
"""Example code for Wasserstein Distortion."""

from PIL import Image
import torch
import torch.optim as optim
import numpy as np
from wasserstein_distortion import VGG16WassersteinDistortion


def im2tensor(image, cent=0.0, factor=255.0):
    return torch.Tensor(
        (image / factor - cent)[:, :, :, None].transpose((3, 2, 0, 1))
    )


def convert_to_numpy_image(x: torch.Tensor) -> np.ndarray:
    x = torch.clamp(x, 0, 1).permute(0, 2, 3, 1)[0]
    y = x.cpu().detach().numpy()
    y = y * 255
    y = y.astype(np.uint8)
    return y


def optimize_noise():
    """Creates an adversarial attack on VGG-16 Wasserstein Distortion."""
    im1 = Image.open("./example/example.png")
    im1_tensor = im2tensor(np.asarray(im1)).cuda()
    im2_tensor = torch.nn.Parameter(
        torch.randn_like(im1_tensor), requires_grad=True
    ).cuda()
    optimizer = optim.Adam([im2_tensor], lr=0.1)
    wloss = VGG16WassersteinDistortion().cuda()
    # To test the Wasserstein distortion, we construct a manual log2_sigma map
    # with globally log2_sigma = 4
    log2_sigma = torch.zeros_like(im1_tensor[:, 0:1, ...]) + 4
    constant_log2_sigma = log2_sigma.cuda()
    for i in range(200):
        optimizer.zero_grad()
        loss = wloss(im1_tensor, im2_tensor, constant_log2_sigma)
        if i % 20 == 0:
            print(loss.item())
            im_pred = convert_to_numpy_image(im2_tensor)
            Image.fromarray(im_pred).save("./example/output.png")
            diff = convert_to_numpy_image((im1_tensor - im2_tensor) / 2)
            Image.fromarray(diff).save("./example/diff_map.png")
        loss.backward()
        optimizer.step()


def test_sample():
    """Compares PyTorch implementation to JAX implementation."""
    img1 = Image.open("./example/reference.png")
    img1_tensor = im2tensor(np.asarray(img1))
    img2 = Image.open("./example/example.png")
    img2_tensor = im2tensor(np.asarray(img2))

    import codex.loss
    import jax.numpy as jnp

    wloss_pytorch = VGG16WassersteinDistortion()

    log2_sigma = torch.zeros_like(img1_tensor[:, 0:1, ...]) + 2

    img1_jax_array = jnp.array(img1_tensor.cpu().numpy())
    img2_jax_array = jnp.array(img2_tensor.cpu().numpy())
    log2_sigma_jax = jnp.array(log2_sigma.cpu().numpy())

    codex.loss.load_vgg16_model(mock=False)

    loss_jax = codex.loss.vgg16_wasserstein_distortion(
        img1_jax_array[0], img2_jax_array[0], log2_sigma_jax[0, 0], num_scales=3
    )

    loss_pytorch = wloss_pytorch(img1_tensor, img2_tensor, log2_sigma, num_scales=3)
    print("PyTorch Loss:", loss_pytorch.item())
    print("JAX Loss:", loss_jax.item())


if __name__ == "__main__":
    test_sample()
    optimize_noise()

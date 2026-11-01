# Wasserstein Distortion
This is an implementation of VGG-16 Wasserstein Distortion in JAX and PyTorch.

Wasserstein Distortion was introduced in:

> Y. Qiu, A. B. Wagner, J. Ballé, L. Theis: "Wasserstein Distortion: Unifying Fidelity and
> Realism," 2024 58th Ann. Conf. on Information Sciences and Systems (CISS), 2024.
> https://arxiv.org/abs/2310.03629

For a description of this implementation, refer to:

> J. Ballé, L. Versari, E. Dupont, H. Kim, M. Bauer: "Good, Cheap, and Fast: Overfitted
> Image Compression with Wasserstein Distortion," 2025 IEEE/CVF Conf. on Computer Vision
> and Pattern Recognition (CVPR), 2025. https://arxiv.org/abs/2412.00505

Please cite these papers if you use this code for scientific work.

## Authors
Jona Ballé (@jonaballe), Yueyu Hu (@huzi96) and Ruizhe Huang (@Ritchie-R-H).

## Install
Install the package with the extra for the framework you want to use:

```
pip install "wasserstein-distortion[jax] @ git+https://github.com/balle-lab/wasserstein-distortion.git"
pip install "wasserstein-distortion[torch] @ git+https://github.com/balle-lab/wasserstein-distortion.git"
```

Use `[all]` to install both, or `[jax,test]` / `[torch,test]` to also install pytest.

## Usage
Import the implementation for your framework:

```python
from wasserstein_distortion.jax import vgg16_wasserstein_distortion  # JAX
from wasserstein_distortion.torch import VGG16WassersteinDistortion  # PyTorch
```

Only the framework you import needs to be installed. See
`example_torch.ipynb` or `example_jax.ipynb` for a complete example.

### Custom features (PyTorch)
`VGG16WassersteinDistortion` is a `WassersteinDistortion` with VGG-16 features plugged in.
To use other features, pass any module that maps an image to a list of feature tensors
of shape `(batch, channels, height, width)`:

```python
from wasserstein_distortion.torch import WassersteinDistortion

class MyFeatures(torch.nn.Module):
    def forward(self, image):
        return [image, ...]

loss_fn = WassersteinDistortion(MyFeatures())
loss = loss_fn(image_a, image_b, log2_sigma)  # log2_sigma: (batch, 1, height, width)
```

If you already have feature tensors, call `multi_wasserstein_distortion` (or
`wasserstein_distortion` for a single tensor) directly.

### Custom features (JAX)
`vgg16_wasserstein_distortion` computes VGG-16 features and passes them to
`multi_wasserstein_distortion`. To use other features, compute them yourself as a list
of arrays of shape `(channels, height, width)` and call `multi_wasserstein_distortion`
directly:

```python
from wasserstein_distortion.jax import multi_wasserstein_distortion

def my_features(image):
    return [image, ...]

loss = multi_wasserstein_distortion(
    my_features(image_a), my_features(image_b), log2_sigma  # log2_sigma: (height, width)
)
```

The JAX functions operate on a single image; use `jax.vmap` for batches. For a single
feature array, use `wasserstein_distortion`.

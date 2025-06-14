# wasserstein-distortion
This is an implementation of VGG-16 Wasserstein Distortion in PyTorch.

Wasserstein Distortion was introduced in [1]_. For a description of this implementation,
refer to [2]_. Please cite these papers if you use this code for scientific work.

A more comprehensive implementation in JAX can be found in
[CoDeX](https://github.com/balle-lab/codex).

.. [1] Y. Qiu, A. B. Wagner, J. Ballé, L. Theis: "Wasserstein Distortion: Unifying
    Fidelity and Realism," 2024 58th Ann. Conf. on Information Sciences and Systems
    (CISS), 2024. https://arxiv.org/abs/2310.03629
.. [2] J. Ballé, L. Versari, E. Dupont, H. Kim, M. Bauer: "Good, Cheap, and Fast:
    Overfitted Image Compression with Wasserstein Distortion," 2025 IEEE/CVF Conf. on
    Computer Vision and Pattern Recognition (CVPR), 2025. https://arxiv.org/abs/2412.00505

## Authors
Yueyu Hu [huzi96](huzi96) and Jona Ballé [jonaballe](jonaballe).

## Install
```pip install git+https://github.com/balle-lab/wasserstein-distortion.git```

## Usage
Check ```wasserstein_distorion_test_example.py```.

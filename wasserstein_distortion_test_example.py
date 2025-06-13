from PIL import Image
import torch
import torch.optim as optim
import numpy as np
from wasserstein_distortion import WassersteinDistortion

def im2tensor(image, imtype=np.uint8, cent=0., factor=255.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def convert_to_numpy_image(x: torch.Tensor) -> np.ndarray:
    x = torch.clamp(x, 0, 1).permute(0, 2, 3, 1)[0]
    y = x.cpu().detach().numpy()
    y = y * 255
    y = y.astype(np.uint8)
    return y

def optimize_noise():
    im1 = Image.open('./example/example.png')
    im1_tensor = im2tensor(np.asarray(im1)).cuda()
    im2_tensor = torch.nn.Parameter(
        torch.randn_like(im1_tensor),
        requires_grad=True
    ).cuda()
    optimizer = optim.Adam([im2_tensor], lr=0.1)
    wloss = WassersteinDistortion().cuda()
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
            Image.fromarray(im_pred).save('./example/output.png')
            diff = convert_to_numpy_image((im1_tensor - im2_tensor) / 2)
            Image.fromarray(diff).save('./example/diff_map.png')
        loss.backward()
        optimizer.step()

def test_sample():
    img1 = Image.open('./example/reference.png')
    img1_tensor = im2tensor(np.asarray(img1))
    img2 = Image.open('./example/example.png')
    img2_tensor = im2tensor(np.asarray(img2))

    from codex.loss import wasserstein
    import jax
    wloss_pytorch = WassersteinDistortion()

    log2_sigma = torch.zeros_like(img1_tensor[:, 0:1, ...]) + 2

    img1_jax_array = jax.numpy.array(img1_tensor.cpu().numpy())
    img2_jax_array = jax.numpy.array(img2_tensor.cpu().numpy())
    log2_sigma_jax = jax.numpy.array(log2_sigma.cpu().numpy())

    from codex.loss import pretrained_features
    pretrained_features.load_vgg16_model(mock=False)


    loss_jax = wasserstein.vgg16_wasserstein_distortion(img1_jax_array[0], img2_jax_array[0], log2_sigma_jax[0, 0], num_scales=3)

    loss_pytorch = wloss_pytorch(img1_tensor, img2_tensor, log2_sigma, num_scales=3)
    print("Pytorch Loss:", loss_pytorch.item())
    print("JAX Loss:", loss_jax.item())

if __name__ == '__main__':
    test_sample()
    optimize_noise()

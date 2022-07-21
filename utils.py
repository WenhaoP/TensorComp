import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

def plot_channel(Psi, image):
    Psi_zeros = np.zeros(image.shape)
    Psi_r = Psi_zeros.copy()
    Psi_g = Psi_zeros.copy()
    Psi_b = Psi_zeros.copy()
    Psi_r[..., 0] = Psi[..., 0]
    Psi_g[..., 1] = Psi[..., 1]
    Psi_b[..., 2] = Psi[..., 2]

    image_zeros = np.zeros(image.shape)
    image_r = image_zeros.copy()
    image_g = image_zeros.copy()
    image_b = image_zeros.copy()
    image_r[..., 0] = image[..., 0]
    image_g[..., 1] = image[..., 1]
    image_b[..., 2] = image[..., 2]

    fig, axes = plt.subplots(1, 6, figsize=(25, 6))
    axes[0].imshow(Psi_r)
    axes[1].imshow(image_r)
    axes[2].imshow(Psi_g)
    axes[3].imshow(image_g)
    axes[4].imshow(Psi_b)
    axes[5].imshow(image_b)
    plt.show()
import numpy as np


def generate_patches(image, patch_size=224, margin=0):
    """
    Generates patches for an image with tags
    Tag is (x, y) as the origin
    """
    image = np.asarray(image)
    size = image.shape[:2]

    kernel_patch_size = patch_size - margin * 2

    for i in range(0, size[0], kernel_patch_size):
        for j in range(0, size[1], kernel_patch_size):
            patch = image[
                i: i + patch_size,
                j: j + patch_size,
            ]
            yield ((i, j), patch)


def generate_coords(image, patch_size=224, margin=0):
    image = np.asarray(image)
    size = image.shape[:2]

    kernel_patch_size = patch_size - margin * 2

    for i in range(0, size[0], kernel_patch_size):
        for j in range(0, size[1], kernel_patch_size):
            yield (i, j)

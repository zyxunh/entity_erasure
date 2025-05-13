import numpy as np


__all__ = ['mask_proportion', 'acquire_images_size']


def mask_proportion(mask_area, image_shape):
    image_area = image_shape[0] * image_shape[1]
    ratio_length = (mask_area / image_area) ** 0.5
    return ratio_length


def acquire_images_size(names):
    import imagesize
    sizes = []
    for name in names:
        size_wh = imagesize.get(name)
        sizes.append(size_wh)
    return sizes
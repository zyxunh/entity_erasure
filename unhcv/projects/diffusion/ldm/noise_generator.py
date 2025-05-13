import torch
import math
from typing import Optional, List, Union, Sequence
from functools import partial


def patch_noise(shape, patch_size=3, seed=None):
    assert seed is None
    if seed is not None:
        seed = seed + patch_size
        generator = torch.Generator().manual_seed(
            seed)
    else:
        generator = None
    if patch_size == 1:
        return torch.randn(shape, generator=generator)
    raise NotImplementedError
    N, C, H, W = shape
    H_down = math.ceil(H / patch_size)
    W_down = math.ceil(W / patch_size)
    noise = torch.randn([N, C, H_down, W_down], generator=generator)
    noise = noise[:, :, :, None, :, None].repeat(1, 1, 1, patch_size, 1, patch_size)
    noise = noise.flatten(-2).flatten(-3, -2)
    noise = noise[:, :, :H, :W]
    return noise


def multi_patch_noise(shape, patch_sizes=(1, 3), probability=None, seed=None):
    if isinstance(patch_sizes, int):
        noise = patch_noise(shape, patch_size=patch_sizes, seed=seed)
    else:
        assert len(patch_sizes) > 1
        noises = []
        for patch_size in patch_sizes:
            noises.append(patch_noise(shape, patch_size, seed=seed))
        noise = merge_noise(noises, probability)
        noise = torch.stack(noises).sum(0) / len(noises) ** 0.5
    return noise


def merge_noise(noises, probability: Optional[List]=None):
    noises = torch.stack(noises)
    noise_for_idx = torch.randn_like(noises)
    if probability is not None:
        probability: torch.Tensor = noise_for_idx.new_tensor(probability)
        while probability.dim() < noise_for_idx.dim():
            probability = probability[..., None]
        noise_for_idx = probability * noise_for_idx
    idx = noise_for_idx.argmin(0)
    noise = torch.gather(noises, dim=0, index=idx[None])[0]
    return noise


class NoiseGenerator:
    def __init__(self, noise_mode="multi_patch_noise", patch_sizes: Union[int, Sequence]=1, probability=None):
        self.noise_function = partial(globals()[noise_mode], patch_sizes=patch_sizes, probability=probability)

    def __call__(self, shape, seed=None):
        return self.noise_function(shape=shape, seed=seed)


__all__ = ["multi_patch_noise", "NoiseGenerator"]


if __name__ == "__main__":
    pass
import torch.nn as nn
import torch
from diffusers import AutoencoderKL


class VAE(nn.Module):
    def __init__(self, pretrained_model_name_or_path=None, config=None):
        super().__init__()
        self.model = AutoencoderKL.from_pretrained(pretrained_model_name_or_path)

    def encode(self, x):
        latents = self.model.encode(x).latent_dist.sample()
        latents = latents * self.model.config.scaling_factor
        return latents

    def decode(self, x, clip=True):
        image = self.model.decode(x / self.model.config.scaling_factor).sample
        if clip:
            image = image.clip(-1, 1)
        return image

if __name__ == '__main__':
    vae = VAE(pretrained_model_name_or_path="/home/yixing/model/stable-diffusion-v1-5-inpainting/vae")
    print(vae)
    print(vae.model)
    input = torch.randn(1, 3, 64, 64).clamp(min=-1, max=1)
    latents = vae.encode(input)
    print(latents.shape)
    image = vae.decode(latents)
    print(image.shape)
    breakpoint()
from typing import Union

import diffusers
from diffusers import DPMSolverMultistepScheduler
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
# from diffusers.schedulers import get_betas
import torch
from math import floor

from unhcv.common.types import DataDict
from unhcv.common.image.visual import imwrite_tensor
# from unhcv.projects.diffusion.inference import DiffusionInferencePipeline

class DiffusionInferencePipeline(object):
    def __init__(self):
        pass
        # ddim_train_steps = 1000

        # ddim_betas = get_betas(
        #     name=args.schedule_type,
        #     num_steps=ddim_train_steps,
        #     shift_snr=args.schedule_shift_snr)

        # ddim_scheduler = DDIMScheduler(
        #     betas=ddim_betas,
        #     num_train_timesteps=ddim_train_steps,
        #     num_inference_timesteps=args.sample_steps,
        #     device=args.device)

    @staticmethod
    def delete_noise(
            scheduler: DDIMScheduler,
            noised_image: torch.Tensor,
            noise_pred: torch.Tensor,
            timesteps: torch.IntTensor,
            pred_original_sample = None,
            reverse=False
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        alpha_prod_t = scheduler.alphas_cumprod[timesteps].cuda()
        while (alpha_prod_t.dim()) < noised_image.dim():
            alpha_prod_t = alpha_prod_t[..., None]
        beta_prod_t = 1 - alpha_prod_t

        if not reverse:
            if scheduler.config.prediction_type == "epsilon":
                pred_original_sample = (noised_image - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
            elif scheduler.config.prediction_type == "sample":
                pred_original_sample = noise_pred
            elif scheduler.config.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t ** 0.5) * noised_image - (beta_prod_t ** 0.5) * noise_pred

            return pred_original_sample
        else:
            if scheduler.config.prediction_type == "epsilon":
                noise_pred = (noised_image - pred_original_sample * alpha_prod_t ** (0.5)) / beta_prod_t ** (0.5)
            elif scheduler.config.prediction_type == "sample":
                raise NotImplementedError
                pred_original_sample = noise_pred
            elif scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
                pred_original_sample = (alpha_prod_t ** 0.5) * noised_image - (beta_prod_t ** 0.5) * noise_pred

            return noise_pred
        denoise_out = scheduler.step(model_output=noise_pred, sample=noised_image, timestep=timesteps)
        return denoise_out.pred_original_sample

        alphas_cumprod = scheduler.alphas_cumprod.to(device=noised_image.device)
        alphas_cumprod = alphas_cumprod.to(dtype=noised_image.dtype)
        # timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(noised_image.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noise_pred.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # noised_image = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise_pred
        original_samples = (noised_image - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
        return original_samples

    @staticmethod
    def inference_function(ddim_scheduler: Union[diffusers.DDIMScheduler, DPMSolverMultistepScheduler],
                           unet=None,
                           noise=None,
                           unet_forward_function=None,
                           unet_conv_extra_input=None,
                           mask=None,
                           masked_latents=None,
                           encoder_hidden_states=None,
                           attention_mask=None,
                           strength=1,
                           input_latent=None,
                           guidance_scale=1,
                           num_inference_steps=None,
                           unet_extra_input={}, collect_original_latent=False,
                           container_dict=None, masked_consistent=None):
        # unet.eval()
        if num_inference_steps is not None:
            ddim_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        timesteps = ddim_scheduler.timesteps.to(noise.device)
        latent = noise
        if strength != 1:
            timesteps = DiffusionInferencePipeline.get_timesteps(timesteps, strength)
            noise_timestep = timesteps[0]
            latent = ddim_scheduler.add_noise(input_latent, noise, torch.tensor([noise_timestep]))
        else:
            latent = noise

        original_latents = []
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=noise.dtype):
            for i_timestep, timestep in enumerate(timesteps):
                if container_dict is not None:
                    container_dict['timestep'] = timestep
                    container_dict['i_timestep'] = i_timestep
                if mask is not None:
                    pred_latent = torch.cat([latent, mask, masked_latents], dim=1)
                elif unet_conv_extra_input is not None:
                    pred_latent = torch.cat([latent, unet_conv_extra_input], dim=1)
                else:
                    pred_latent = latent
                if guidance_scale > 1:
                    pred_latent = pred_latent.repeat(2, 1, 1, 1)
                # torch.cat([GLOBAL_ITEM.noisy_latents[0:1], unet_conv_extra_input], dim=1)
                if unet_forward_function is not None:
                    noise_pred_redundancy = noise_pred = unet_forward_function(pred_latent, timestep.repeat(len(pred_latent)), encoder_hidden_states, attention_mask=attention_mask, **unet_extra_input).sample
                else:
                    noise_pred_redundancy = noise_pred = unet(pred_latent, timestep, encoder_hidden_states, attention_mask=attention_mask, **unet_extra_input).sample
                # imwrite_tensor("/home/tiger/code/unhcv/show1.jpg", noise_pred[:, 4:])

                if noise_pred.size(1) > latent.size(1):
                    extra_pred = noise_pred[:, latent.size(1):]
                else:
                    extra_pred = None
                noise_pred = noise_pred[:, :latent.size(1)]
                if guidance_scale > 1:
                    if (i_timestep + 1) / len(timesteps) < 0.5:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        if extra_pred is not None:
                            extra_pred = extra_pred.chunk(2)[1]
                    else:
                        noise_pred = noise_pred.chunk(2)[1]
                    # redundancy_latent, original_redundancy_latent = ddim_scheduler.step(
                    #     model_output=noise_pred_redundancy,
                    #     timestep=timestep,
                    #     sample=pred_latent).to_tuple()

                # original_latent_ = ddim_scheduler.step(
                #     model_output=noise_pred_uncond,
                #     timestep=timestep,
                #     sample=latent).to_tuple()[1]

                step_output = ddim_scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=latent).to_tuple()
                if len(step_output) == 1:
                    latent = step_output[0]
                    original_latent = None
                else:
                    latent, original_latent = step_output
                # if masked_consistent is not None:
                #     latent_consistent = masked_consistent.latent
                #     if i_timestep < len(timesteps) - 1:
                #         noise_timestep = timesteps[i_timestep + 1]
                #         latent_consistent = ddim_scheduler.add_noise(latent_consistent, noise,
                #                                                      torch.tensor([noise_timestep]))
                #     latent = (1 - masked_consistent.mask) * latent_consistent + masked_consistent.mask * latent

                # if timestep < 100:
                #     breakpoint()
                # from unhcv.common.image.visual import imwrite_tensor
                # imwrite_tensor("/home/tiger/code/unhcv/show.jpg", original_redundancy_latent[:, :3])

                # if collect_original_latent:
                #     collect_data = DataDict(original_latent=original_latent.cpu())
                #     if extra_pred is not None:
                #         collect_data.extra_pred = extra_pred
                #     original_latents.append(collect_data)
                # original_latent = original_latent_
                if collect_original_latent:
                    original_latent = original_latent.cpu()
                    # original_collector = DataDict(original_latent=original_latent.cpu())
                    if extra_pred is not None:
                        original_latent = torch.cat([original_latent, extra_pred.cpu()], dim=1)
                        # original_collector.original_latent = torch.cat([original_latent, extra_pred.cpu()], dim=1)
                    # original_collector.original_redundancy_latent = original_redundancy_latent
                    original_latents.append(original_latent)
        return latent, original_latents

    @staticmethod
    def get_timesteps(timesteps, strength):
        # get the original timestep using init_timestep
        num_inference_steps = len(timesteps)
        init_timestep = max(min(int(num_inference_steps * strength), num_inference_steps), 1)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]

        return timesteps

if __name__ == '__main__':

    # import sys; sys.path.insert(0, '/root/code/latent-diffusion/examples')
    # from text_to_image.schedulers.ddim import DDIMScheduler
    # from text_to_image.schedulers.utils import get_betas
    # breakpoint()

    # ddim_train_steps = 1000

    # ddim_betas = get_betas(
    #     name='squared_linear',
    #     num_steps=ddim_train_steps,
    #     shift_snr=1)

    # ddim_scheduler = DDIMScheduler(
    #     betas=ddim_betas,
    #     num_train_timesteps=ddim_train_steps,
    #     num_inference_timesteps=20,
    #     device='cuda')

    breakpoint()

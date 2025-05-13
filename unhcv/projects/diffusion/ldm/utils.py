from typing import Union, Dict, Type

import torch
from diffusers import DDIMScheduler, UNet2DConditionModel, PNDMScheduler
from torch import nn

from unhcv.common.utils import obj_load


__all__ = ["add_cond_wrapper", "ode", "build_scheduler"]


class CondWrapper:

    def forward(
        self,
        sample: torch.Tensor,
        *args,
        conditioning: Dict[str, torch.Tensor] = None,
        **kwargs
    ):
        class_labels = conditioning.get("vector", None)
        crossattn = conditioning.get("crossattn", None)
        concat = conditioning.get("concat", None)

        # concat conditioning
        if concat is not None:
            sample = torch.cat([sample, concat], dim=1)

        return self.model(sample, *args, encoder_hidden_states=crossattn, class_labels=class_labels, **kwargs).sample


def add_cond_wrapper(cls) -> Type[UNet2DConditionModel]:
    class WrappedClass(cls):
        def forward(
                self,
                sample: torch.Tensor,
                *args,
                conditioning: Dict[str, torch.Tensor] = None,
                **kwargs):
            """

            Args:
                sample:
                *args:
                conditioning: dict(encoder_hidden_states, concat, class_labels, conv_in_residual)
                **kwargs:

            Returns:

            """
            conditioning = conditioning.copy()
            concat = conditioning.pop("concat", None)

            # concat conditioning
            if concat is not None:
                sample = torch.cat([sample, concat], dim=1)

            return super().forward(sample, *args, **conditioning, **kwargs).sample

    return WrappedClass

class CondWrapper_(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        sample: torch.Tensor,
        *args,
        conditioning: Dict[str, torch.Tensor] = None,
        **kwargs
    ):
        class_labels = conditioning.get("vector", None)
        crossattn = conditioning.get("crossattn", None)
        concat = conditioning.get("concat", None)

        # concat conditioning
        if concat is not None:
            sample = torch.cat([sample, concat], dim=1)

        return self.model(sample, *args, encoder_hidden_states=crossattn, class_labels=class_labels, **kwargs).sample

    def __getattr__(self, item):
        print(item)
        return self.model.item


def ode(scheduler: DDIMScheduler, denoiser, noisy_sample, conditioning, unconditional_conditioning=None,
        guidance_scale=1, start_idx=0, steps=None) -> torch.Tensor:
    """

    Args:
        scheduler:
        denoiser:
        noisy_sample:
        conditioning: dict(crossattn, concat)
        unconditional_conditioning:
        guidance_scale:
        start_idx:

    Returns:

    """

    if steps is not None and scheduler.num_inference_steps != steps:
        scheduler.set_timesteps(steps)
    for t in scheduler.timesteps[start_idx:]:
        timestep = torch.tensor([t], device=noisy_sample.device).repeat(noisy_sample.shape[0])
    
        noisy_sample_ = scheduler.scale_model_input(noisy_sample, t)
    
        # Denoise sample
        noise_pred = cond_noise_pred = denoiser(
            sample=noisy_sample_,
            timestep=timestep,
            conditioning=conditioning,
        )

        if guidance_scale > 1:
            uncond_noise_pred = denoiser(
                sample=noisy_sample_,
                timestep=timestep,
                conditioning=unconditional_conditioning,
            )

            # Make CFG
            noise_pred = (
                    guidance_scale * cond_noise_pred
                    + (1 - guidance_scale) * uncond_noise_pred
            )
    
        # Make one step on the reverse diffusion process
        noisy_sample = scheduler.step(
            noise_pred, t, noisy_sample, return_dict=False
        )[0]

        noisy_sample = noisy_sample.to(cond_noise_pred)

    return noisy_sample


def extract_into_tensor(a, t, x_dim):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (x_dim - 1)))


def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    sample_dim = sample.dim()
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample_dim)
        alphas = extract_into_tensor(alphas, timesteps, sample_dim)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample_dim)
        alphas = extract_into_tensor(alphas, timesteps, sample_dim)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


def build_scheduler(scheduler_name_or_path) -> DDIMScheduler:
    config = obj_load(scheduler_name_or_path)
    class Scheduler(eval(config["_class_name"])):
        pass
    return Scheduler.from_pretrained(scheduler_name_or_path)


if __name__ == "__main__":
    scheduler = build_scheduler("/home/yixing/model/stable-diffusion-v1-5-inpainting/scheduler/scheduler_config.json")
    # scheduler = Scheduler("/home/yixing/model/stable-diffusion-v1-5-inpainting/scheduler/scheduler_config.json")
    breakpoint()
    pass
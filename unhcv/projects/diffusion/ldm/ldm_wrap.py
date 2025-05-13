import dataclasses
import glob
import importlib
import os.path

import PIL.Image
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel, Transformer2DModel, \
    DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor, \
    CLIPTextModelWithProjection, T5EncoderModel
from transformers.utils import ContextManagers
try:
    from unhcv.projects.segmentation.custom_mask2former import Mask2FormerInput
except:
    pass

from unhcv.common.utils import obj_load, write_im, walk_all_files_with_suffix, attach_home_root, find_path
from unhcv.nn.utils import load_checkpoint
from unhcv.common.types import DataDict
from unhcv.nn.utils.analyse import cal_para_num, sum_para_value
from unhcv.nn.utils.checkpoint import save_checkpoint
from unhcv.projects.diffusion.inference import DiffusionInferencePipeline
from unhcv.projects.diffusion.ldm.controlnet import ControlNet, ControlNetConfig, EntityInpaintingCond, \
    EntityInpaitingControlNet, AttentionControlNet
from unhcv.projects.diffusion.ldm.noise_generator import NoiseGenerator
from unhcv.core.train import AccelerateTrain
from typing import Optional, List, Union, overload, Dict
from torchvision.transforms import Compose, ToTensor, Normalize


@dataclasses.dataclass
class TextEmbedding:
    encoder_hidden_states: Union[torch.Tensor, None]
    attention_mask: Union[torch.Tensor, None] = None


class LdmWrap:

    def init_vae_text_encoder(self, models_root, unet_extra_config={}, scheduler_config={}):
        config_dict: Dict = DiffusionPipeline.load_config(models_root)
        modules = []

        def get_modules(modules_str, fp32=False, maybe_custom=False, extra_config={}):
            modules_tmp = []
            for module_str in modules_str:
                if not self.use_text_encoder and module_str in ["text_encoder", "tokenizer"]:
                    modules_tmp.append(None)
                    continue
                if module_str in config_dict:
                    library = importlib.import_module(config_dict[module_str][0])
                    module_cls = getattr(library, config_dict[module_str][1])

                    kwargs = {}
                    if not fp32:
                        kwargs['torch_dtype'] = self.dtype
                    # breakpoint()
                    if module_str == "scheduler":
                        scheduler_cls = scheduler_config.get("cls", None)
                        if scheduler_cls is not None:
                            module_cls = getattr(library, scheduler_cls)

                    if maybe_custom:
                        model_config_path = walk_all_files_with_suffix(os.path.join(models_root, module_str), suffixs=(".json", ".yml"))
                        model_config = obj_load(model_config_path[0])
                        model_config.update(extra_config)
                        instance = module_cls.from_config(model_config, **kwargs)
                        checkpoint = walk_all_files_with_suffix(os.path.join(models_root, module_str), suffixs=(".safetensors", ".bin"))
                        checkpoint = [var for var in checkpoint if os.path.getsize(var) > 1000]
                        if len(checkpoint) > 0:
                            load_checkpoint(instance, checkpoint[0])
                    else:
                        instance = module_cls.from_pretrained(os.path.join(models_root, module_str), **kwargs)
                    modules_tmp.append(instance)
            return modules_tmp

        modules_str = ["scheduler", "text_encoder", "tokenizer", "vae"]
        with ContextManagers(AccelerateTrain.deepspeed_zero_init_disabled_context_manager()):
            modules.extend(get_modules(modules_str))

        modules_str = ["transformer", "unet"]
        modules.extend(get_modules(modules_str, fp32=True, maybe_custom=True, extra_config=unet_extra_config))
        return modules

        if self.is_pix_art:
            text_encoder = CLIPTextModel.from_pretrained(os.path.join(models_root, 'text_encoder'))
        else:
            text_encoder = CLIPTextModel.from_pretrained(os.path.join(models_root, 'text_encoder'))
            # vae = AutoencoderKL(**obj_load(os.path.join(models_root, 'vae/config.json')))
            vae = AutoencoderKL.from_pretrained(os.path.join(models_root, 'vae'))
            # load_checkpoint(vae, os.path.join(models_root, 'vae/diffusion_pytorch_model.fp16.bin'))
            tokenizer = CLIPTokenizer.from_pretrained(models_root, subfolder="tokenizer")
        return vae, text_encoder, tokenizer

    text_attn_mask = False
    def __init__(self, dtype=torch.float16, device='cuda', unet_extra_config={},
                 guidance_scale=7.5, vae_channels=4, predict_image_num=1,
                 unet_checkpoint=None,
                 ddpm_config={}, num_inference_steps=12, unet_cls=None,
                 noise_config=dict(noise_mode="multi_patch_noise", patch_sizes=1), accelerator=None,
                 models_root="/home/tiger/model/stable-diffusion-v1-5", collector={}, text_max_length=None,
                 use_text_encoder=True, scheduler_config={}):
        models_root = find_path(models_root)
        self.collector = collector
        self.use_text_encoder = use_text_encoder
        self.device = device
        self.dtype = dtype
        self.is_pix_art = unet_cls in ["pix_art", "AttentionControlNetPixArt"]
        if self.is_pix_art:
            self.text_attn_mask = True
        self.unet_cls = unet_cls
        self.text_max_length = text_max_length

        if unet_checkpoint is None:
            unet_checkpoint = os.path.join(models_root, "unet/diffusion_pytorch_model.bin")

        unet_extra_config_birth = unet_extra_config.pop("birth", {})
        ddim_noise_scheduler, text_encoder, tokenizer, vae, unet = self.init_vae_text_encoder(models_root=models_root, unet_extra_config=unet_extra_config_birth, scheduler_config=scheduler_config)

        # vae, text_encoder, tokenizer = self.init_vae_text_encoder(models_root=models_root)
        model_config_path = os.path.join(models_root, 'unet/config.json')
        if not os.path.exists(model_config_path):
            model_config_path = os.path.join(models_root, 'transformer/config.json')
        unet_kwargs = obj_load(model_config_path)
        unet_kwargs.update(unet_extra_config)

        if unet_cls == "BrushNet":
            from diffusers import BrushNetModel
            unet_base = UNet2DConditionModel(**obj_load(os.path.join(models_root, 'unet/config.json')))
            if unet_checkpoint is not None:
                load_checkpoint(unet_base, unet_checkpoint, mismatch_shape=True)
            self.unet_base = unet_base.to(device=device, dtype=dtype)
            self.unet_base.requires_grad_(False)
            unet = BrushNetModel.from_unet(unet_base, **unet_extra_config)
        else:
            use_brushnet_diffusers = False
            try:
                from diffusers import BrushNetModel
                use_brushnet_diffusers = True
            except Exception:
                pass
            assert not use_brushnet_diffusers, "used brushnet_diffusers"
            if self.is_pix_art:
                if unet_cls == "AttentionControlNetPixArt":
                    unet = AttentionControlNetPixArt(unet, collector=collector, **unet_extra_config)
                pass
            elif unet_cls in ("ControlNet", "EntityInpaitingControlNet"):
                if unet_cls == "ControlNet":
                    self.controlnet = ControlNet(models_root=models_root, **unet_extra_config, dtype=dtype,
                                                 collector=collector)
                elif unet_cls == "EntityInpaitingControlNet":
                    self.controlnet = EntityInpaitingControlNet(**unet_extra_config)
                self.controlnet.control_branch = unet = self.controlnet.control_branch.cuda()
                self.controlnet.main_branch = self.controlnet.main_branch.to(device=device, dtype=dtype)
            elif unet_cls == 'attention_controlnet':
                model_config = dict(models_root=models_root, conditioning_channels=5, controlnet_up=False,
                                    controlnet_down=False, controlnet_mid=False, collector=collector)
                model_config.update(unet_extra_config)
                unet = AttentionControlNet(**model_config)
            else:
                if unet_cls is None:
                    unet = UNet2DConditionModel(**unet_kwargs)
                else:
                    unet = unet_cls(**unet_kwargs)
                if unet_checkpoint is not None:
                    load_checkpoint(unet, unet_checkpoint, mismatch_shape=True)

        self.unet_cls = unet_cls
        self.ddpm_noise_scheduler = DDPMScheduler.from_pretrained(models_root, subfolder="scheduler", **ddpm_config)
        # inference_scheduler_config = dict(self.ddpm_noise_scheduler.config)
        # inference_scheduler_config.update(ddpm_config)
        # ddim_noise_scheduler = DDIMScheduler.from_config(inference_scheduler_config)
        self.num_inference_steps = num_inference_steps
        ddim_noise_scheduler.set_timesteps(num_inference_steps)
        self.text_encoder = text_encoder.to(device=device).eval() if text_encoder is not None else text_encoder
        self.tokenizer = tokenizer
        self.vae = vae.to(device=device, dtype=dtype).eval()
        self.unet = unet.to(device=device)
        self.vae_encode = lambda x: vae.encode(x.to(dtype)).latent_dist.sample().mul_(vae.config.scaling_factor)
        self.vae_decode_lambda = lambda x: vae.decode(x.to(dtype) / vae.config.scaling_factor).sample.clip(-1, 1)
        self.ddim_noise_scheduler = ddim_noise_scheduler
        # self.noise_generator = NoiseGenerator(**noise_config)

        self.guidance_scale = guidance_scale
        self.vae_channels = vae_channels
        self.predict_image_num = predict_image_num

        # preprocess_image
        self.transform = Compose([ToTensor(), Normalize(mean=0.5, std=0.5)])

    @property
    def frozen_models(self):
        if self.unet_cls in ("EntityInpaitingControlNet", "ControlNet"):
            return [self.controlnet.main_branch]
        return []

    def reset_model(self, unet):
        self.unet = unet
        if self.unet_cls in ("EntityInpaitingControlNet", "ControlNet"):
            self.controlnet.control_branch = unet
        elif self.unet_cls == "attention_controlnet":
            self.unet = unet

    def unet_forward_function(self, sample: torch.Tensor,
                              timestep: Union[torch.Tensor, float, int],
                              encoder_hidden_states: Union[torch.Tensor, TextEmbedding],
                              attention_mask=None, **kwargs
                              ):
        if self.is_pix_art:
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
            kwargs = dict(timestep=timestep, added_cond_kwargs=added_cond_kwargs, **kwargs)
            if encoder_hidden_states is not None:
                kwargs.update(dict(encoder_hidden_states=encoder_hidden_states.encoder_hidden_states, encoder_attention_mask=encoder_hidden_states.attention_mask))
            out = self.unet(sample, **kwargs)
            out.sample = out.sample[:, :sample.shape[1]]
            return out
        elif self.unet_cls in ("EntityInpaitingControlNet", "ControlNet"):
            return self.controlnet(sample, timestep, encoder_hidden_states, **kwargs)
        elif self.unet_cls == "attention_controlnet":
            # if hasattr(self, 'segmentation_model'):
            #     model_input = Mask2FormerInput(sample=sample,
            #                                    controlnet_cond=kwargs['controlnet_cond'],
            #                                    timestep=timestep, inpainting_mask=self.collector['inpainting_mask'])
            #     segmentation_output = self.segmentation_model(model_input)
            #     panoptic_seg = torch.stack([var['panoptic_seg'][0] for var in segmentation_output], dim=0).cpu()
            #     self.collector['segmentation_outputs'].append(panoptic_seg)

            return self.unet(sample, timestep, encoder_hidden_states, **kwargs)
        elif hasattr(self, "unet_base"):
            conditioning_latents = sample[:, -5:]
            conditioning_latents = torch.cat([conditioning_latents[:, 1:], 1 - conditioning_latents[:, 0:1]], dim=1)

            sample = sample[:, :-5]
            down_block_res_samples, mid_block_res_sample, up_block_res_samples = self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                brushnet_cond=conditioning_latents,
                return_dict=False,
            )

            # Predict the noise residual
            model_pred = self.unet_base(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                down_block_add_samples=[
                    sample.to(dtype=self.dtype) for sample in down_block_res_samples
                ],
                mid_block_add_sample=mid_block_res_sample.to(dtype=self.dtype),
                up_block_add_samples=[
                    sample.to(dtype=self.dtype) for sample in up_block_res_samples
                ],
                return_dict=True,
            )
            return model_pred
        else:
            return self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states, )

    def preprocess_image(self, image: PIL.Image.Image):
        image = image.resize((512, 512))
        im_tensor: torch.Tensor = self.transform(image)[None].to(self.dtype).cuda()
        return im_tensor

    def generate_noise(self, shape, seed=None):
        if seed is None:
            noise = torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            noise = torch.randn(shape, dtype=self.dtype, device=self.device,
                                generator=torch.Generator(device=self.device).manual_seed(seed))
        return noise

    def vae_decode(self, x):
        if x.shape[1] == self.vae_channels:
            return self.vae_decode_lambda(x)
        else:
            out = []
            for i in range(x.shape[1] // self.vae_channels):
                out.append(self.vae_decode_lambda(x[:, i*self.vae_channels: (i+1)*self.vae_channels]))
            return torch.cat(out, dim=1)

    @property
    def modules(self):
        return (
            self.text_encoder, self.tokenizer, self.vae, self.unet, self.ddim_noise_scheduler,
            self.ddpm_noise_scheduler)

    def text_encode(self, texts):
        if not self.use_text_encoder:
            return None
        max_length = self.tokenizer.model_max_length
        if self.text_max_length is not None:
            max_length = min(max_length, self.text_max_length)
        kwargs = dict(max_length=max_length, padding="max_length",
                      truncation=True,
                      return_tensors="pt")
        if self.is_pix_art:
            kwargs['add_special_tokens'] = True
        text_input = self.tokenizer(texts, **kwargs)
        if self.text_attn_mask:
            attention_mask = text_input.attention_mask.to(self.device)
            encoder_hidden_states = \
                self.text_encoder(text_input.input_ids.to(self.device), attention_mask=attention_mask)[0]
            encoder_hidden_states = TextEmbedding(encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)
        else:
            encoder_hidden_states = self.text_encoder(text_input.input_ids.to(self.device))[0]
            # encoder_hidden_states = TextEmbedding(encoder_hidden_states=encoder_hidden_states)
        return encoder_hidden_states

    def prepare_common_input_for_inference(self, texts, image_size, seed, guidance_scale=None, negative_texts=None):
        batch_size = len(texts)
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        if guidance_scale > 1:
            if negative_texts is None:
                negative_texts = [""] * len(texts)
            texts = negative_texts + texts
        noise = self.generate_noise([batch_size, self.predict_image_num * self.vae_channels, *image_size], seed).to(
            self.device, self.dtype)
        # noise = torch.randn(
        #     size=[batch_size, self.predict_image_num * self.vae_channels, *image_size],
        #     device=self.device,
        #     generator=torch.Generator(self.device).manual_seed(
        #         seed), dtype=self.dtype)
        encoder_hidden_states = self.text_encode(texts)

        return noise, encoder_hidden_states, guidance_scale

    def vae_decode_with_original_latent(self, output_vae, output_original_vaes):
        output_decode = torch.cat([self.vae_decode(var) for var in output_vae.split(self.vae_channels, dim=1)], dim=1)
        for i_out in range(len(output_original_vaes)):
            output_original_vaes[i_out] = torch.cat([self.vae_decode(var.cuda()).cpu() for var in
                                                     output_original_vaes[i_out].split(self.vae_channels, dim=1)],
                                                    dim=1)
        return DataDict(output_decode=output_decode, output_originals=output_original_vaes)

    def text2image(self, texts: List, image_size=(64, 64), seed=1234, guidance_scale=None):
        noise, encoder_hidden_states, guidance_scale = self.prepare_common_input_for_inference(texts=texts, image_size=image_size,
                                                                               seed=seed, guidance_scale=guidance_scale)
        output_vae = DiffusionInferencePipeline.inference_function(noise=noise,
                                                                   ddim_scheduler=self.ddim_noise_scheduler,
                                                                   unet=self.unet,
                                                                   encoder_hidden_states=encoder_hidden_states,
                                                                   guidance_scale=guidance_scale)
        output_decode = self.vae_decode(output_vae)
        return output_decode

    @torch.no_grad()
    def inpaint(self, images: torch.Tensor, masks: torch.Tensor, texts: List[str], seed=1234, guidance_scale=None,
                collect_original_latent=False, container_dict=None, rebase=None, strength=1, output_channels=None,
                unet_extra_input={}):
        # images_vae = self.vae_encode(images)
        # images = images * (1 - masks)
        masked_images_vae = self.vae_encode(images)
        if rebase is not None:
            rebase_vae = self.vae_encode(rebase)
        else:
            rebase_vae = None
        masks = F.interpolate(masks, size=masked_images_vae.shape[2:])
        noise, encoder_hidden_states, guidance_scale = self.prepare_common_input_for_inference(texts=texts,
                                                                                               image_size=
                                                                                               masked_images_vae.shape[2:],
                                                                                               seed=seed,
                                                                                               guidance_scale=
                                                                                               guidance_scale)
        output_vae, output_original_vaes = DiffusionInferencePipeline.inference_function(noise=noise,
                                                                                         mask=masks,
                                                                                         masked_latents=masked_images_vae,
                                                                                         ddim_scheduler=self.ddim_noise_scheduler,
                                                                                         unet=self.unet,
                                                                                         input_latent=rebase_vae,
                                                                                         encoder_hidden_states=encoder_hidden_states,
                                                                                         guidance_scale=guidance_scale,
                                                                                         collect_original_latent=collect_original_latent,
                                                                                         container_dict=container_dict,
                                                                                         strength=strength,
                                                                                         unet_extra_input=unet_extra_input,
                                                                                         unet_forward_function=self.unet_forward_function)
        output_decode = self.vae_decode_with_original_latent(output_vae, output_original_vaes)
        return output_decode

    @torch.no_grad()
    def condition2image(self, noise_shape, texts: List[str], seed=1234, guidance_scale=None,
                        collect_original_latent=False, container_dict=None, rebase=None, strength=1,
                        unet_extra_input={}, negative_texts=None, **kwargs):
        noise, encoder_hidden_states, guidance_scale = \
            self.prepare_common_input_for_inference(texts=texts,
                                                    image_size=noise_shape,
                                                    seed=seed,
                                                    guidance_scale=guidance_scale,
                                                    negative_texts=negative_texts)
        output_vae, output_original_vaes = \
            DiffusionInferencePipeline.inference_function(noise=noise,
                                                          ddim_scheduler=self.ddim_noise_scheduler,
                                                          unet=self.unet,
                                                          encoder_hidden_states=encoder_hidden_states,
                                                          guidance_scale=guidance_scale,
                                                          collect_original_latent=collect_original_latent,
                                                          container_dict=self.collector,
                                                          strength=strength,
                                                          unet_extra_input=unet_extra_input,
                                                          unet_forward_function=self.unet_forward_function,
                                                          num_inference_steps=self.num_inference_steps,
                                                          **kwargs)
        output_decode = self.vae_decode_with_original_latent(output_vae, output_original_vaes)
        return output_decode

    def image2image(self, image: Optional[torch.Tensor], image_vae: Optional[torch.Tensor], texts: Optional[List[str]],
                    seed=1234, guidance_scale=None) -> Union[List[torch.Tensor], torch.Tensor]:
        if image_vae is None:
            image_vae = self.vae_encode(image)
        noise, encoder_hidden_states, guidance_scale = self.prepare_common_input_for_inference(texts=texts,
                                                                                               image_size=
                                                                                               image_vae.shape[2:],
                                                                                               seed=seed,
                                                                                               guidance_scale=
                                                                                               guidance_scale)
        output_vae, output_original_vaes = DiffusionInferencePipeline.inference_function(noise=noise,
                                                                   unet_conv_extra_input=image_vae,
                                                                   ddim_scheduler=self.ddim_noise_scheduler,
                                                                   unet=self.unet,
                                                                   encoder_hidden_states=encoder_hidden_states,
                                                                   guidance_scale=guidance_scale, collect_original_latent=True)

        output_decode = torch.cat([self.vae_decode(var) for var in output_vae.split(self.vae_channels, dim=1)], dim=1)
        for i_out in range(len(output_original_vaes)):
            output_original_vaes[i_out] = torch.cat([self.vae_decode(var.cuda()).cpu() for var in output_original_vaes[i_out].split(self.vae_channels, dim=1)], dim=1)
        return DataDict(output_decode=output_decode, output_originals=output_original_vaes)

        if self.predict_image_num == 1:
            output_decode = self.vae_decode(output_vae)
            output_originals = []
            for i_out in range(len(output_original_vaes)):
                output_original_vaes[i_out].original_latent = self.vae_decode(output_original_vaes[i_out].original_latent.cuda()).cpu()
            return DataDict(output_decode=output_decode, output_originals=output_original_vaes)
        else:
            output_decodes = []
            output_originals = [[] for _ in range(len(output_original_vaes))]
            for i_predict_image in range(self.predict_image_num):
                output_decode = self.vae_decode(
                    output_vae[:, i_predict_image * self.vae_channels:(i_predict_image + 1) * self.vae_channels])
                output_decodes.append(output_decode)
                for i_out in range(len(output_original_vaes)):
                    output_originals[i_out].append(self.vae_decode(output_original_vaes[i_out][:,
                                                                   i_predict_image * self.vae_channels:(
                                                                                                               i_predict_image + 1) * self.vae_channels].cuda()).cpu())
            return DataDict(output_decode=output_decodes, output_originals=output_originals)


if __name__ == "__main__":
    DEBUG_NAME = "adapter" # inpainting, t2i, adapter, controlnet, ip_adapter
    from torchvision.transforms import Compose, ToTensor, Normalize
    from unhcv.common.image.visual import imwrite_tensor, visual_tensor
    import numpy as np
    config_path = 'unhcv/projects/diffusion/inpainting/configs/unet_controlnet/ip_adapter.py'
    config = obj_load(config_path)
    if DEBUG_NAME == "adapter":
        state_dict = torch.load("/home/zhuyixing/model/IP-Adapter/ip-adapter-full-face_sd15.bin", map_location='cpu')
        ip_adapter_weight = {}
        for key, value in state_dict["ip_adapter"].items():
            key = key.split(".")
            key_i = str((int(key[0]) - 1) // 2)
            key = ".".join([key_i, *key[1:]])
            key = key.replace("to_k_ip", "to_k_control").replace("to_v_ip", "to_v_control")
            ip_adapter_weight[key] = value
        breakpoint()
        unet_extra_config = config.get("unet_extra_config")
        ldm_wrap_config = config.get("ldm_wrap_config")
        unet_cls = config.get("unet_cls")
        ldm_wrap_config['unet_cls'] = unet_cls
        ldm_wrap = LdmWrap(device="cuda", unet_extra_config=unet_extra_config, **ldm_wrap_config)
        # load ip_adapter
        ldm_wrap.unet.ip_adapter.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = ldm_wrap.unet.main_branch.attn_processors.values()
        ip_layers = [var for var in ip_layers if isinstance(var, torch.nn.Module)]
        ip_layers = torch.nn.ModuleList(ip_layers)
        breakpoint()
        ip_layers.load_state_dict(ip_adapter_weight)
        save_checkpoint(ldm_wrap.unet, attach_home_root("model/IP-Adapter/unet_ip-adapter-full-face_sd15.bin"))

        ldm_wrap.unet.main_branch.get_
        ldm_wrap.collector["entity_mask"] = torch.ones([1, 1, 512, 512], device="cuda", dtype=torch.float)
        ldm_wrap.collector["inpainting_mask"] = torch.ones([1, 1, 512, 512], device="cuda", dtype=torch.float)
        noise_shape = [64, 64]
        text = [""]
        condition = torch.zeros([1, 5, 64, 64], dtype=torch.float).cuda()
        ip_adapter_image = torch.zeros([1, 3, 224, 224], dtype=torch.float).cuda()
        # entity_condition = torch.zeros([1, 30, 512, 512], dtype=torch.float).cuda()
        unet_extra_input = dict(controlnet_cond=condition, ip_adapter_image=ip_adapter_image)
        sample = torch.zeros([1, 4, 64, 64], dtype=torch.float)

        out = ldm_wrap.condition2image(noise_shape=noise_shape, texts=text, unet_extra_input=unet_extra_input, guidance_scale=1)

        breakpoint()
    elif DEBUG_NAME == "controlnet":
        ldm_wrap = LdmWrap(device="cuda", unet_extra_config=dict(entity_in_channels=30),
                           unet_cls="EntityInpaitingControlNet", models_root="/home/tiger/model/realisticVisionV60B1_v51VAE")
        noise_shape = [64, 64]
        text = [""]
        condition = torch.zeros([1, 5, 64, 64], dtype=torch.float).cuda()
        entity_condition = torch.zeros([1, 30, 512, 512], dtype=torch.float).cuda()
        unet_extra_input = dict(controlnet_cond=EntityInpaintingCond(entity=entity_condition, outpainting_image=condition))
        sample = torch.zeros([1, 4, 64, 64], dtype=torch.float)

        out = ldm_wrap.condition2image(noise_shape=noise_shape, texts=text, unet_extra_input=unet_extra_input)

        breakpoint()

    elif DEBUG_NAME == "brushnet":
        transform = Compose([ToTensor(), Normalize(mean=0.5, std=0.5)])
        im = obj_load("/mnt/bn/zhuyixing/workspace/datasets/demo/inst_demo/panoptic_demo/Demo_0.jpg")
        im = np.array(im.resize((512, 512)))
        # im_tensor: torch.Tensor = transform(im)[None].to(torch.float16).cuda()
        mask = np.zeros([*im.shape[:2]])
        mask[10:100, 100:300] = 1
        # mask[...] = 1
        im = im * (1 - mask[..., None])
        im_tensor = (torch.from_numpy(im).to(torch.float16).permute(2, 0, 1)[None].cuda() - 127.5) / 127.5
        mask = torch.from_numpy(mask).to(torch.float16)[None, None].cuda()

        imwrite_tensor("/home/tiger/code/unhcv_researsh/show_input.jpg", im_tensor, max_value=1, min_value=-1, image_mode="rgb")
        ldm_wrap = LdmWrap(device="cuda", unet_extra_config=dict(),
                           unet_checkpoint="/home/tiger/model/realisticVisionV60B1_v51VAE/unet/diffusion_pytorch_model.bin",
                           unet_cls="BrushNet", models_root="/home/tiger/model/realisticVisionV60B1_v51VAE")
        load_checkpoint(ldm_wrap.unet, "/home/tiger/model/brushnet/random_mask/diffusion_pytorch_model.safetensors")
        out = ldm_wrap.inpaint(images=im_tensor, masks=mask, texts=["high quality"], guidance_scale=7.5)
        output_decode = out[0]
        imwrite_tensor("/home/tiger/code/unhcv_researsh/show_output.jpg", out[0], max_value=1, min_value=-1, image_mode="rgb")

        breakpoint()
        def mismatch_resolve_function(key, state_parameter, model_parameter):
            model_parameter[:, :4] = state_parameter[:, :4]
            model_parameter[:, 4:] = state_parameter[:, 5:]
            return model_parameter
    elif DEBUG_NAME == "adapter":
        transform = Compose([ToTensor(), Normalize(mean=0.5, std=0.5)])
        im = obj_load("/mnt/bn/zhuyixing/workspace/datasets/demo/inst_demo/panoptic_demo/Demo_0.jpg")
        im = im.resize((512, 512))
        im_tensor: torch.Tensor = transform(im)[None].to(torch.float16).cuda()
        mask = im_tensor.new_zeros([1, 1, *im_tensor.shape[2:]])
        mask[:, :, 10:100, 100:300] = 1
        # mask[...] = 1
        im_tensor = im_tensor * (1 - mask)
        imwrite_tensor("/home/tiger/code/unhcv/show_input.jpg", im_tensor, max_value=1, min_value=-1, image_mode="rgb")
        ldm_wrap = LdmWrap(device="cuda", unet_extra_config=dict(in_channels=12, out_channels=8),
                           unet_checkpoint=None, predict_image_num=2)

        def mismatch_resolve_function(key, state_parameter, model_parameter):
            if "_out" in key:
                model_parameter[:4] = state_parameter
            else:
                model_parameter[:, :4] = state_parameter[:, :4]
                model_parameter[:, 8:12] = state_parameter[:, 5:]
            return model_parameter
        load_checkpoint(ldm_wrap.unet,
                        "/mnt/bn/inpainting-bytenas-lq/zyx/models/sd-v1.5/stable-diffusion-inpainting/unet/diffusion_pytorch_model.bin",
                        mismatch_shape=True, mismatch_resolve_function=mismatch_resolve_function)
        with torch.no_grad():
            out = ldm_wrap.image2image(None, image_vae=ldm_wrap.vae_encode(im_tensor), texts=["clock"],
                                       guidance_scale=7.5)
        breakpoint()
        imwrite_tensor("/home/tiger/code/unhcv/show_output_0.jpg", out[1], max_value=1, min_value=-1, image_mode="rgb")
    elif DEBUG_NAME == "inpainting":
        transform = Compose([ToTensor(), Normalize(mean=0.5, std=0.5)])
        im = obj_load("/mnt/bn/zhuyixing/workspace/datasets/demo/inst_demo/panoptic_demo/Demo_0.jpg")
        im = im.resize((512, 512))
        im_tensor: torch.Tensor = transform(im)[None].to(torch.float16).cuda()
        mask = im_tensor.new_zeros([1, 1, *im_tensor.shape[2:]])
        mask[:, :, 10:100, 100:300] = 1
        # mask[...] = 1
        im_tensor = im_tensor * (1 - mask)
        imwrite_tensor("/home/tiger/code/unhcv/show_input.jpg", im_tensor, max_value=1, min_value=-1, image_mode="rgb")
        ldm_wrap = LdmWrap(device="cuda", unet_extra_config=dict(in_channels=9),
                           unet_checkpoint="/mnt/bn/inpainting-bytenas-lq/zyx/models/sd-v1.5/stable-diffusion-inpainting/unet/diffusion_pytorch_model.bin")
        breakpoint()

        def mismatch_resolve_function(key, state_parameter, model_parameter):
            model_parameter[:, :4] = state_parameter[:, :4]
            model_parameter[:, 4:] = state_parameter[:, 5:]
            return model_parameter


        # load_checkpoint(ldm_wrap.unet,
        #                 None,
        #                 mismatch_shape=True, mismatch_resolve_function=mismatch_resolve_function)
        out = ldm_wrap.inpaint(images=im_tensor, masks=mask, texts=["a girl, real scenes"], guidance_scale=7.5)
        imwrite_tensor("/home/tiger/code/unhcv/show_output.jpg", out, max_value=1, min_value=-1, image_mode="rgb")
        breakpoint()
    else:
        ldm_wrap = LdmWrap(device="cuda")
        out = ldm_wrap.text2image(["clock on desert"])
        imwrite_tensor("/home/tiger/code/unhcv/show.jpg", out, max_value=1, min_value=-1, image_mode="rgb")
    pass
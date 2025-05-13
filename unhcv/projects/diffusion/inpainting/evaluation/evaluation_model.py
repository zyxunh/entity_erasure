import os

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor

from unhcv.common import visual_mask, write_im, remove_dir
from unhcv.common.image import visual_tensor, concat_differ_size, gray2color
from unhcv.common.utils import find_path, obj_load, attach_home_root, ProgressBar, ProgressBarTqdm
from unhcv.datasets.common_datasets import DatasetWithPreprocess, Dataset, SizeBucket
from unhcv.datasets.transforms.torchvision_transforms import RandomResizedCrop
from unhcv.nn.utils import load_checkpoint
from unhcv.projects.diffusion.ldm import LdmWrap
from unhcv.projects.segmentation.custom_mask2former import build_mask2former_model, CustomExtraMask2FormerConfig, \
    Mask2FormerInput


class EvaluationModel:
    def __init__(self, generation_model_config, generation_checkpoint, segmentation_checkpoint, guidance_scale=3):

        guidance_scale = guidance_scale
        weight_dtype = torch.float16
        inference_timesteps = 20
        seed = 1234

        generation_checkpoint = find_path(generation_checkpoint)
        segmentation_checkpoint = find_path(segmentation_checkpoint)

        self.weight_dtype = weight_dtype
        self.inference_timesteps = inference_timesteps
        self.seed = seed
        self.guidance_scale = guidance_scale

        self.segmentation_model = build_mask2former_model(
            find_path("code/unhcv/unhcv/projects/segmentation/config/maskformer2_R50_bs16_50ep.yaml"),
            extra_config=CustomExtraMask2FormerConfig(mask_projection_upscale=2)).cuda().eval()

        load_checkpoint(self.segmentation_model, segmentation_checkpoint)

        generation_model_config = obj_load(generation_model_config)
        unet_cls = generation_model_config.pop("unet_cls")

        self.collector = {}

        ldm_wrap_config = dict(dtype=weight_dtype, device="cuda",
                               unet_cls=unet_cls, unet_checkpoint=None,
                               unet_extra_config=generation_model_config.get("unet_extra_config", {}),
                               ddpm_config=generation_model_config.get("ddpm_config", {}),
                               guidance_scale=guidance_scale,
                               num_inference_steps=20, collector=self.collector,
                               models_root=os.environ["MODEL_ROOT"])
        ldm_wrap_config.update(generation_model_config.get("ldm_wrap_config", {}))

        self.ldm_wrap = LdmWrap(**ldm_wrap_config)
        self.ldm_wrap.segmentation_model = self.segmentation_model
        load_checkpoint(self.ldm_wrap.unet, generation_checkpoint)

    @torch.no_grad()
    def __call__(self, data, seed=None):
        if seed is None:
            seed = self.seed
        image = data['image'].to(dtype=self.weight_dtype, device="cuda")
        inpainting_mask = data['inpainting_mask'].to(dtype=self.weight_dtype, device="cuda")
        image = image / 127.5 - 1
        inpainting_mask /= 255
        image_masked = image * (1 - inpainting_mask)

        image_vaed = self.ldm_wrap.vae_encode(image)
        image_masked_vaed = self.ldm_wrap.vae_encode(image_masked)
        image = self.ldm_wrap.vae_decode(image_vaed)
        image_masked = self.ldm_wrap.vae_decode(image_masked_vaed)
        inpainting_mask_latent_size = F.interpolate(inpainting_mask, image_masked_vaed.shape[2:])
        noisy_latents = torch.randn(image_masked_vaed.shape, device="cuda", dtype=image_masked_vaed.dtype,
                            generator=torch.Generator(device="cuda").manual_seed(seed))
        controlnet_cond = torch.cat([image_masked_vaed, inpainting_mask_latent_size], dim=1)
        self.collector['segmentation_outputs'] = []

        # 分割部分
        model_input = Mask2FormerInput(sample=noisy_latents,
                                       controlnet_cond=controlnet_cond,
                                       timestep=torch.tensor([999], dtype=torch.long, device='cuda'),
                                       inpainting_mask=inpainting_mask)

        with torch.autocast(dtype=torch.float16, device_type="cuda"):
            model_output = self.segmentation_model(model_input)
        panoptic_seg = torch.stack([var['panoptic_seg'][0] for var in model_output], dim=0)

        # 生成部分
        self.collector['visual_attn'] = True
        self.collector["inpainting_mask"] = inpainting_mask
        self.collector["entity_mask"] = panoptic_seg[:, None].to(self.weight_dtype)
        batch_size, _, input_height, input_width = image.shape
        text = ["best quality, high quality"] * batch_size
        if self.guidance_scale != 1:
            controlnet_cond = torch.repeat_interleave(controlnet_cond, dim=0, repeats=2)
            controlnet_cond[0, :4] = 0

        ip_adapter_image = None

        inpainting_output = self.ldm_wrap.condition2image(noise_shape=(input_height // 8, input_width // 8),
                                                          texts=text,
                                                          unet_extra_input=dict(controlnet_cond=controlnet_cond,
                                                                                ip_adapter_image=ip_adapter_image),
                                                          guidance_scale=self.guidance_scale,
                                                          collect_original_latent=True)

        inpainting_output_show = visual_tensor(inpainting_output.output_decode, max_value=1, min_value=-1,
                                               reverse=True)
        return dict(inpainting_output_show=inpainting_output_show, panoptic_seg=panoptic_seg)

if __name__ == '__main__':
    generation_checkpoint = "amodal_completion_model.bin"
    segmentation_checkpoint = "amodal_segmentation_model.bin"
    generation_model_config = "unhcv/projects/diffusion/inpainting/configs/entity_erasure.py"
    guidance_scale = 3
    show_root = "/home/yixing/show"
    image_path = "unhcv/projects/diffusion/inpainting/evaluation/image.png"
    mask_path = "unhcv/projects/diffusion/inpainting/evaluation/mask.png"
    image = obj_load(image_path)
    inpainting_mask = obj_load(mask_path)

    size_bucket = SizeBucket(stride=1)
    hw = size_bucket.match((image.height, image.width))
    ratio = image.width / image.height
    random_resized_crop = RandomResizedCrop(size=hw, ratio=(ratio, ratio), scale=(1, 1))
    image, inpainting_mask = random_resized_crop((image, inpainting_mask), interpolations=("bicubic", "nearest-exact"))
    image = pil_to_tensor(image)
    inpainting_mask = pil_to_tensor(inpainting_mask)
    data = dict(image=image[None], inpainting_mask=inpainting_mask[None])

    model = EvaluationModel(generation_model_config=generation_model_config,
                            generation_checkpoint=generation_checkpoint,
                            segmentation_checkpoint=segmentation_checkpoint,
                            guidance_scale=guidance_scale)

    out = model(data)
    inpainting_output_show = out['inpainting_output_show']
    panoptic_seg_color = gray2color(out['panoptic_seg'][0])
    write_im(os.path.join(show_root, "seg.png"), panoptic_seg_color.cpu().numpy())
    write_im(os.path.join(show_root, "result.jpg"), inpainting_output_show)


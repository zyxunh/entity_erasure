import kornia.morphology
import math
import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms.functional import pil_to_tensor, to_tensor, normalize
from PIL import Image
from transformers import CLIPImageProcessor
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from typing import Any, Optional, Tuple, Union

from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPOutput, clip_loss
from transformers.utils import ContextManagers
import torch
import torch.nn as nn

from unhcv.datasets.segmentation import Entity2Rgb
from unhcv.projects.diffusion.inpainting.dataset import InpaintingDatasetWithMask

from unhcv.math import torch_corroef
from diffusers.models.attention import BasicTransformerBlock, FeedForward

from unhcv.common.types import DataDict, ListDict
from unhcv.nn.utils.checkpoint import load_checkpoint
from unhcv.common.image.visual import imwrite_tensor, concat_differ_size, visual_tensor, putText, putTextBatch
from unhcv.core.train.accelerate_train import AccelerateTrain
from unhcv.common.utils import obj_load, write_im, obj_dump, write_txt, remove_dir, walk_all_files_with_suffix, \
    find_path
from unhcv.common.image import visual_mask, concat_differ_size_tensor
from unhcv.projects.diffusion.inference import DiffusionInferencePipeline
from unhcv.projects.diffusion.ldm import LdmWrap, UNet2DConditionModelExtraOut, UNet2DConditionModel
from unhcv.datasets.common_datasets import ConcatDataset, sharder_worker_init_fn
from unhcv.datasets.common_datasets import DatasetWithPreprocess
from unhcv.common.image import ratio_length2hw
from unhcv.common.fileio.hdfs import listdir
from unhcv.nn.loss import PointSampleLoss
from unhcv.projects.diffusion.ldm.inpainting_attention import InpaintingAttnProcessor2_0, set_attn_processor
import bson


def mismatch_resolve_function(key, state_parameter, model_parameter, random_init=True):
    if not random_init:
        model_parameter[...] = 0
    if "_out" in key:
        model_parameter[:4] = state_parameter
    else:
        model_parameter[:, :4] = state_parameter[:, :4]
        model_parameter[:, -5:] = state_parameter[:, 4:]
    return model_parameter


def missing_resolve_function(key, state_dict):
    breakpoint()
    pass


def decode_methods_default(name, data, root_id):
    new_data = {}
    data = bson.loads(data)
    image = obj_load('data.jpg', buffer=data['image'])
    new_data['image'] = image
    new_data['text'] = data['text']
    new_data['aesthetic'] = data['aesthetic']
    # new_data['similarity'] = data['similarity']
    # new_data['width'] = data['width']
    # new_data['height'] = data['height']
    return new_data


def data_indexes_filter(data_indexes):
    new_data_indexes = [var for var in data_indexes if var['intro']['aesthetic'] >= 5]
    return new_data_indexes


class TrainInpainting(AccelerateTrain):
    collector: Optional[dict] = {}
    ldm_wrap: Optional[LdmWrap] = None
    point_sample_loss: Optional[PointSampleLoss] = None
    erase_prompt: Optional[str] = None
    prepare_train_dataloader = False
    task = "inpainting"
    controlnet_cond_drop_ratio = 0.05
    text_cond_drop_ratio = 0.05
    all_cond_drop_ratio = 0.05
    entity_mask_drop_ratio = 0.0

    def __init__(self, *args, **kwargs):
        default_dataset_common_kwargs = dict(collect_keys=('image', 'mask', 'inpainting_mask', 'text'),
                                             iou_filter_thres=0.3, image_modes=("RGB", "L"))
        default_dataset_kwargs = [
            # dict(data_indexes_path="/home/tiger/dataset/Adobe_EntitySeg/image_indexes/train_lr_indexes.yml",
            #      segmentation_dc_id=0, data_root="/home/tiger/dataset/Adobe_EntitySeg"),
            # dict(data_indexes_path="/home/tiger/dataset/entity_seg_mdb_indexes/train_image_indexes.msgpack",
            #      segmentation_dc_id=0, data_root="/home/tiger/dataset/entity_openimage_mdb")
        ]

        # for var in default_dataset_kwargs:
        #     var.update(default_dataset_common_kwargs)
        self.default_dataset_kwargs = default_dataset_kwargs
        self.dataset_class = ConcatDataset
        self.demo_dataset_class = DatasetWithPreprocess
        self.text_encoder: Optional[nn.Module] = None
        self.tokenizer: Optional[nn.Module] = None
        self.vae: Optional[nn.Module] = None
        self.unet: Optional[nn.Module] = None
        self.ddim_noise_scheduler: Optional[nn.Module] = None
        self.ddpm_noise_scheduler: Optional[nn.Module] = None
        self.entity_attention = False
        self.train_step1000_with_other_step = False
        super().__init__(*args, **kwargs)

    def init_train_dataset(self):
        args = self.args
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        dataset_config = self.parse_dataset_config()
        train_dataset_kwargs = dataset_config.get("train_dataset_kwargs", {})

        data_root = dataset_config.pop("data_root",
                                                 "hdfs://haruna/home/byte_labcv_gan/bang/data/laion_small_pic_collect/laion_2b_en_512plus_buckets")
        data_indexes_root = find_path(dataset_config.pop("data_indexes_root",
                                                         "hdfs://haruna/dp/mloops/datasets/unh/dataset/laion/laion_2b_en_512plus_buckets_indexes"))
        # data_path = "hdfs://haruna/home/byte_labcv_gan/bang/data/laion_small_pic_collect/laion_2b_en_512plus_buckets/002540-10000-768-1344_00254_00004"
        # data_indexes_path = "hdfs://haruna/dp/mloops/datasets/unh/dataset/laion/laion_2b_en_512plus_buckets_indexes_sorted/002540-10000-768-1344_00254_00004_catalog.bson_catalog.bson"

        # data_indexes_paths = listdir(data_indexes_root, kv=True)
        # data_indexes_paths = [var for var in data_indexes_paths if "_catalog.bson" in var]
        data_indexes_suffix = dataset_config.pop("data_indexes_suffix", '_catalog.bson')
        data_indexes_paths = walk_all_files_with_suffix(data_indexes_root, suffixs=(data_indexes_suffix,), sort=True)

        transforms_kwargs = dict(interpolations='bicubic', max_stride=64)
        parallel_read_num = 100
        parallel_read_num = math.ceil(parallel_read_num / args.train_batch_size) * args.train_batch_size
        common_config = dict(image_keys='image', transforms_kwargs=transforms_kwargs,
                             batch_size=parallel_read_num,
                             backend_config=dict(decode_methods=dict(default=decode_methods_default)),
                             image_modes="RGB", parallel_read_num=parallel_read_num,
                             data_indexes_filter=data_indexes_filter)
        common_config.update(train_dataset_kwargs)

        custom_config = []
        for data_indexes_path in data_indexes_paths:
            data_path = data_indexes_path[:-len(data_indexes_suffix)]
            if isinstance(data_root, str):
                data_path = data_path.replace(data_indexes_root, data_root)
            else:
                data_path = [data_path.replace(data_indexes_root, var) for var in data_root]
            # length = int(os.path.basename(data_indexes_path).replace("-", "_").split("_")[1])
            file_informs = os.path.basename(data_indexes_path).replace("-", "_").split("_")
            length = int([var for var in file_informs if 'num' in var][0].replace('num', ''))
            custom_config.append(dict(data_indexes_path=data_indexes_path, data_root=data_path, length=length))
        dataset = ConcatDataset(InpaintingDatasetWithMask, common_config=common_config, custom_configs=custom_config)

        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            prefetch_factor=8 if args.dataloader_num_workers > 0 else None,
            collate_fn=torch.utils.data.default_collate,
            pin_memory=True,
            worker_init_fn=sharder_worker_init_fn)

        self.logger.info("***** Running training *****")
        self.logger.info(
            f"world_size is {world_size}, batch_size is {world_size * args.train_batch_size * args.gradient_accumulation_steps}, "
            f"num_epoch is {world_size * args.train_batch_size * args.train_steps * args.gradient_accumulation_steps / len(dataset)}")
        time.sleep(10)
        self.train_dataloader = train_dataloader

    @staticmethod
    def custom_wrap_policy(module, recurse, unwrapped_params=None, nonwrapped_numel=None):
        return recurse or isinstance(module, (
        # Discriminator, # VisionTransformer, # FeatureNetwork, # , FeatureNetwork #, PatchEmbed, Block #, VisionTransformer
        ))

    def parser_add_argument(self):
        parser = super().parser_add_argument()
        parser.add_argument("--guidance_scale", type=float, default=3)
        parser.add_argument("--seed", type=int, default=1234)
        return parser

    def init_model(self):
        super().init_model()
        weight_dtype = self.weight_dtype
        accelerator = self.accelerator
        args = self.args

        model_config = self.parse_model_config()
        unet_extra_config = model_config.get('unet_extra_config', {})
        self.task = model_config.get("task", "inpainting")
        self.erase_prompt = model_config.get("erase_prompt", "")
        self.unet_cls = unet_cls = model_config.get("unet_cls", "UNet2DConditionModel")
        self.use_controlnet_cond = unet_cls in ["attention_controlnet", "AttentionControlNetPixArt", "ControlNet"]
        self.ip_adapter_on = model_config.get("unet_extra_config", {}).get("ip_adapter_on", False)
        if self.ip_adapter_on:
            mean = torch.tensor(OPENAI_CLIP_MEAN, dtype=weight_dtype, device="cuda").view(1, -1, 1, 1)
            std = torch.tensor(OPENAI_CLIP_STD, dtype=weight_dtype, device="cuda").view(1, -1, 1, 1)
            self.ip_adapter_image_processor = lambda x: ((x + 1) / 2 - mean) / std
        self.controlnet_cond_drop_ratio = model_config.get("controlnet_cond_drop_ratio", self.controlnet_cond_drop_ratio)
        self.all_cond_drop_ratio = model_config.get("all_cond_drop_ratio", self.all_cond_drop_ratio)
        self.text_cond_drop_ratio = model_config.get("text_cond_drop_ratio", self.text_cond_drop_ratio)
        self.entity_mask_drop_ratio = model_config.get("entity_mask_drop_ratio", self.entity_mask_drop_ratio)

        self.entity_rgb_cond_on = unet_extra_config.get("entity_rgb_cond_on", False)

        ldm_wrap_config = dict(dtype=weight_dtype, device=accelerator.device,
                                       unet_cls=unet_cls, unet_checkpoint=None,
                                       unet_extra_config=model_config.get("unet_extra_config", {}),
                                       ddpm_config=model_config.get("ddpm_config", {}),
                                       guidance_scale=args.guidance_scale, accelerator=accelerator,
                                       num_inference_steps=20, collector=self.collector, models_root=os.environ["MODEL_ROOT"])
        ldm_wrap_config.update(model_config.get("ldm_wrap_config", {}))
        

        ldm_wrap = LdmWrap(**ldm_wrap_config)
        (self.text_encoder, self.tokenizer, self.vae, self.unet, self.ddim_noise_scheduler,
         self.ddpm_noise_scheduler) = ldm_wrap.modules
        self.model = self.unet
        self.frozen_models = [self.text_encoder, self.vae]

        if self.args.checkpoint is not None:
            load_checkpoint(self.model, mismatch_shape=False,
                            state_dict=self.args.checkpoint, log_missing_keys=True)
        self.ldm_wrap = ldm_wrap

    def init_for_train(self):
        super().init_for_train()
        self.unet = self.model
        self.ldm_wrap.reset_model(self.unet)

    def init_for_eval(self):
        pass

    def get_mse_loss(self, unet_input, timesteps, encoder_hidden_states, target, out_name="sample"):
        outputs = self.ldm_wrap.unet_forward_function(unet_input, timesteps, encoder_hidden_states)
        loss = F.mse_loss(outputs[out_name], target, reduction="mean")
        return loss

    def backward(self, loss_dict):
        losses = []
        for key, value in loss_dict.items():
            if value.requires_grad:
                losses.append(value)
            loss_dict[key] = value.detach()
        losses = sum(losses)
        self.accelerator.backward(losses)
        return loss_dict

    def get_loss(self, batch):
        self.collector.clear()
        (text_encoder, tokenizer, vae, unet, ddim_noise_scheduler,
         ddpm_noise_scheduler,
         ldm_wrap) = (self.text_encoder, self.tokenizer, self.vae, self.unet, self.ddim_noise_scheduler,
                      self.ddpm_noise_scheduler, self.ldm_wrap)
        accelerator = self.accelerator

        image = batch["image"].to(self.weight_dtype).cuda()
        entity_mask = batch["mask"].to(self.weight_dtype)[:, None].cuda()
        inpainting_mask = batch["inpainting_mask"].to(self.weight_dtype)[:, None].cuda()
        self.collector["inpainting_mask"] = inpainting_mask
        self.collector["entity_mask"] = entity_mask
        image_masked = image * (1 - inpainting_mask)
        if self.ip_adapter_on:
            ip_adapter_image = image
            ip_adapter_image = F.interpolate(ip_adapter_image, (224, 224), mode='bicubic')
            ip_adapter_image = self.ip_adapter_image_processor(ip_adapter_image)
        if self.task == "segmentation":
            image = batch["mask_rgb"].to(self.weight_dtype).cuda()
        text = batch["text"]

        if self.global_step % self.args.train_visual_steps == 1:
            self.save_for_training_show_tensors = DataDict()

        with torch.no_grad():
            latents = ldm_wrap.vae_encode(image)
            latents_masked = ldm_wrap.vae_encode(image_masked)
            if self.task == "image_and_segmentation":
                mask_rgb = batch["mask_rgb"].to(self.weight_dtype).cuda()
                latents_mask_rgb = ldm_wrap.vae_encode(mask_rgb)
                latents = torch.cat([latents, latents_mask_rgb], dim=1)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, ddpm_noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = ddpm_noise_scheduler.add_noise(latents, noise, timesteps)

        if ddpm_noise_scheduler.config.prediction_type == "epsilon":
            unet_target = noise
        elif ddpm_noise_scheduler.config.prediction_type == "v_prediction":
            unet_target = ddpm_noise_scheduler.get_velocity(latents, noise, timesteps)
        elif ddpm_noise_scheduler.config.prediction_type == "sample":
            unet_target = latents
        else:
            raise ValueError(f"Unknown prediction type {ddpm_noise_scheduler.config.prediction_type}")

        unet_extra_input = {}
        inpainting_mask_latent_size = F.interpolate(inpainting_mask, noisy_latents.shape[2:])
        rand_nums = torch.rand(len(latents_masked), device=latents_masked.device)
        ratio_floor = 0

        def need_drop(ratio, ratio_floor):
            ratio_ceil = ratio_floor + ratio
            return (rand_nums >= ratio_floor) & (rand_nums < ratio_ceil), ratio_ceil

        text_drop_flag, ratio_floor = need_drop(self.text_cond_drop_ratio, ratio_floor)
        if self.all_cond_drop_ratio > 0:
            all_cond_drop_flag, ratio_floor = need_drop(self.all_cond_drop_ratio, ratio_floor)
            text_drop_flag = text_drop_flag | all_cond_drop_flag
        if self.use_controlnet_cond:
            if self.controlnet_cond_drop_ratio > 0:
                controlnet_drop_flag, ratio_floor = need_drop(self.controlnet_cond_drop_ratio, ratio_floor)
            if self.entity_mask_drop_ratio > 0:
                entity_mask_drop_flag, ratio_floor = need_drop(self.entity_mask_drop_ratio, ratio_floor)
            if self.all_cond_drop_ratio > 0:
                if self.controlnet_cond_drop_ratio > 0:
                    controlnet_drop_flag = controlnet_drop_flag | all_cond_drop_flag
                if self.entity_mask_drop_ratio > 0:
                    entity_mask_drop_flag = entity_mask_drop_flag | all_cond_drop_flag
            if self.controlnet_cond_drop_ratio > 0:
                latents_masked[controlnet_drop_flag] = 0
            if self.entity_mask_drop_ratio > 0:
                self.collector["entity_mask"][entity_mask_drop_flag] = 1
            unet_extra_input.update(
                dict(controlnet_cond=torch.cat([latents_masked, inpainting_mask_latent_size], dim=1)))
            if self.entity_rgb_cond_on:
                unet_extra_input["controlnet_cond"] = (batch['mask_rgb'].to(self.weight_dtype).cuda(), unet_extra_input["controlnet_cond"])
        if self.ip_adapter_on:
            if self.controlnet_cond_drop_ratio > 0:
                ip_adapter_drop_flag, ratio_floor = need_drop(self.controlnet_cond_drop_ratio, ratio_floor)
                if self.all_cond_drop_ratio > 0:
                    ip_adapter_drop_flag = ip_adapter_drop_flag | all_cond_drop_flag
                ip_adapter_image[ip_adapter_drop_flag] = 0
                batch["ip_adapter_image"] = unet_extra_input['ip_adapter_image'] = ip_adapter_image

        batch["text"] = text = ["" if flag else text_one for text_one, flag in zip(text, text_drop_flag)]

        with torch.no_grad():
            encoder_hidden_states = self.ldm_wrap.text_encode(text)

        loss_dict = {}

        outputs = self.ldm_wrap.unet_forward_function(noisy_latents, timesteps, encoder_hidden_states, **unet_extra_input)

        if isinstance(outputs, torch.Tensor):
            noise_pred = outputs
        else:
            noise_pred = outputs.sample
        loss = F.mse_loss(noise_pred.float(), unet_target.float(), reduction="mean")
        loss_dict.update(dict(mse_loss=loss))
        # loss_dict.update(self.get_entity_loss(mask=mask, inpainting_mask=inpainting_mask))

        if self.save_for_training_show_tensors is not None:
            self.save_for_training_show_tensors.update(noisy_latents=noisy_latents, outputs=outputs,
                                                       timesteps=timesteps, batch=batch, mask=entity_mask,
                                                       latents_masked=latents_masked, latents=latents)
        self.collector.clear()
        return loss_dict

    def visual_training_result(self):
        noisy_latents, outputs, timesteps, batch, mask, latents_masked, latents = self.save_for_training_show_tensors.get_keys("noisy_latents", "outputs", "timesteps", "batch", "mask", "latents_masked", "latents")
        extra_out = self.save_for_training_show_tensors.get("extra_out", None)
        show_root = os.path.join(self.args.show_dir, 'training/{:06}'.format(self.global_step))
        remove_dir(show_root)
        shows = []
        with torch.no_grad():
            delete_noise_out = DiffusionInferencePipeline.delete_noise(self.ddpm_noise_scheduler, noisy_latents,
                                                                       outputs.sample[:, :noisy_latents.size(1)], timesteps)
            decode_latents = lambda x: torch.cat([self.ldm_wrap.vae_decode(var) for var in x.split(4, dim=1)], dim=-1)
            delete_noise_out_decode = decode_latents(delete_noise_out)
            noisy_image = decode_latents(noisy_latents)
            latents_masked_decode = self.ldm_wrap.vae_decode(latents_masked)
            latents_decode = decode_latents(latents)
            delete_noise_out_decode_show = visual_tensor(delete_noise_out_decode, max_value=1, min_value=-1, reverse=True,
                                                         max_num=self.max_visual_num_in_training)
            noisy_image_show = visual_tensor(noisy_image, max_value=1, min_value=-1, reverse=True,
                                                         max_num=self.max_visual_num_in_training)
            latents_masked_decode_show = visual_tensor(latents_masked_decode, max_value=1, min_value=-1, reverse=True,
                                             max_num=self.max_visual_num_in_training)
            latents_decode_show = visual_tensor(latents_decode, max_value=1, min_value=-1, reverse=True,
                                             max_num=self.max_visual_num_in_training)
            shows.extend([latents_decode_show, latents_masked_decode_show, noisy_image_show, delete_noise_out_decode_show])
            mask_show = visual_tensor(mask, max_value=255, min_value=0, max_num=self.max_visual_num_in_training)
            mask_show = visual_mask(latents_decode_show, mask_show, stack_axis=1)[-1]
            shows.append(mask_show)
            if extra_out is not None:
                extra_out_decode = self.ldm_wrap.vae_decode(extra_out)
                extra_out_decode_show = visual_tensor(extra_out_decode, max_value=1, min_value=-1, reverse=True,
                                             max_num=self.max_visual_num_in_training)
                shows.append(extra_out_decode_show)
            if "ip_adapter_image" in batch:
                ip_adapter_image_show = visual_tensor(batch['ip_adapter_image'], mean=OPENAI_CLIP_MEAN,
                                                      std=OPENAI_CLIP_STD, reverse=True,
                                                      max_num=self.max_visual_num_in_training)
                shows.append(ip_adapter_image_show)

            text_show = [putText(np.zeros([*delete_noise_out_decode.shape[2:], 3], dtype=np.uint8), show_texts=text) for text in batch["text"][:self.max_visual_num_in_training]]
            text_show = concat_differ_size(text_show, axis=0)
            shows.append(text_show)
            shows = concat_differ_size(shows, axis=1)
            write_im(os.path.join(show_root, "shows.jpg"), shows)
            self.log_images(shows, tag="train_data", reverse_color=True)
            information = ["timesteps: {}".format(timesteps.tolist()[:self.max_visual_num_in_training])]
            obj_dump(os.path.join(show_root, "shows.txt"), information)
            pass

    def debug_for_dataset(self):
        show_root = os.path.join(self.args.show_dir, 'train_dataset')
        remove_dir(show_root)
        train_dataiter = iter(self.train_dataloader)
        # train_dataiter = iter(self.train_dataloader.dataset)
        for i in range(10000):
            print(f"debug {i}")
            data = next(train_dataiter)
            image = visual_tensor(data['image'], max_value=1, min_value=-1, max_num=5)
            mask = visual_tensor(data['mask'][:, None], max_value=255, min_value=0, max_num=5)
            inpainting_mask = visual_tensor(data['inpainting_mask'][:, None], max_value=255, min_value=0, max_num=5)
            mask_show = visual_mask(image, mask, stack_axis=1)[-1]
            inpainting_mask_show = visual_mask(image, inpainting_mask, stack_axis=1)[-1]
            text_show = putTextBatch(show_texts=data["text"][:5], img_size=data['image'].shape[-2:][::-1])
            shows = [image, mask_show, inpainting_mask_show, text_show]
            shows = concat_differ_size(shows)
            write_im(os.path.join(show_root, f"{i}.jpg"), shows)
            if i == 0:
                print(data)

if __name__ == "__main__":
    TrainInpainting().main()

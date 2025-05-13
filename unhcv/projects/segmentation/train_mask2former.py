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
from diffusers import AutoencoderKL, DDPMScheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms.functional import pil_to_tensor, to_tensor, normalize
from PIL import Image
from transformers import CLIPImageProcessor, Mask2FormerConfig
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

from unhcv.datasets.transforms.torchvision_transforms import RandomResizedCrop
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
from unhcv.projects.diffusion.inpainting.dataset import InpaintingDatasetWithMask
from unhcv.projects.diffusion.inpainting.train_inpainting import TrainInpainting
from unhcv.datasets.common_datasets import ConcatDataset, sharder_worker_init_fn, SizeBucket
from unhcv.datasets.common_datasets import DatasetWithPreprocess
from unhcv.common.image import ratio_length2hw
from unhcv.common.fileio.hdfs import listdir
from unhcv.nn.loss import PointSampleLoss
from unhcv.projects.diffusion.ldm.inpainting_attention import InpaintingAttnProcessor2_0, set_attn_processor
import bson

from unhcv.projects.segmentation.custom_mask2former import CustomExtraMask2FormerConfig, build_mask2former_model, Mask2FormerInput
from unhcv.projects.segmentation.unet_2d_backbone import LDMInput


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


class TrainMask2former(TrainInpainting):
    collector: Optional[dict] = {}
    ldm_wrap: Optional[LdmWrap] = None
    point_sample_loss: Optional[PointSampleLoss] = None
    erase_prompt: Optional[str] = None
    prepare_train_dataloader = False
    task = "inpainting"

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

    def init_model(self):
        super(TrainInpainting, self).init_model()
        weight_dtype = self.weight_dtype
        accelerator = self.accelerator
        args = self.args
        if args.model_config is not None:
            model_config = obj_load(args.model_config)
        else:
            model_config = {}

        self.train_999_step_ratio = model_config.get('train_999_step_ratio', None)

        ldm_models_root = os.environ["MODEL_ROOT"]
        self.model = build_mask2former_model(
            find_path("code/unhcv/unhcv/projects/segmentation/config/maskformer2_R50_bs16_50ep.yaml"),
            extra_config=CustomExtraMask2FormerConfig(mask_projection_upscale=2))
        # self.model = CustomMask2FormerForUniversalSegmentation(
        #     Mask2FormerConfig.from_json_file("/home/tiger/code/unhcv_researsh/unhcv/projects/segmentation/config/mask2former-swin-small-coco-panoptic_config.json"),
        #     CustomExtraMask2FormerConfig(mask_projection_upscale=2)).cuda()
        vae = AutoencoderKL.from_pretrained(os.path.join(ldm_models_root, "vae")).to(weight_dtype).cuda()
        self.vae_encode = lambda x: vae.encode(x.to(weight_dtype)).latent_dist.sample().mul_(vae.config.scaling_factor)
        self.vae_decode = lambda x: vae.decode(x.to(weight_dtype) / vae.config.scaling_factor).sample.clip(-1, 1)
        self.ddpm_noise_scheduler = DDPMScheduler.from_pretrained(ldm_models_root, subfolder="scheduler")

        self.frozen_models = [vae]

        if self.args.checkpoint is not None:
            load_checkpoint(self.model, mismatch_shape=False,
                            state_dict=self.args.checkpoint, log_missing_keys=True)

    def init_for_train(self):
        super(TrainInpainting, self).init_for_train()

    def init_for_eval(self):
        self.model = self.model.cuda()

    def get_mse_loss(self, unet_input, timesteps, encoder_hidden_states, target, out_name="sample"):
        outputs = self.ldm_wrap.unet_forward_function(unet_input, timesteps, encoder_hidden_states)
        loss = F.mse_loss(outputs[out_name], target, reduction="mean")
        return loss

    def get_entity_loss(self, mask, inpainting_mask, tag="_loss"):
        loss_dict = {}
        for key in list(self.collector.keys()):
            if key.endswith(".entity_attn_map"):
                attn_map = self.collector.pop(key)
                scale = int((mask.size(2) * mask.size(3) / attn_map.size(2)) ** 0.5)
                attn_map_h, attn_map_w = mask.size(2) // scale, mask.size(3) // scale
                mask_downsample = F.interpolate(mask, (attn_map_h, attn_map_w))
                mask_downsample = mask_downsample.flatten(2)
                mask_downsample_cross_match = mask_downsample[:, :, :, None] == mask_downsample[:, :, None]
                mask_downsample_cross_match_valid = (mask_downsample[:, :, :, None] != 0) & (mask_downsample[:, :, None] != 0)
                if self.entity_attention_loss_tgt_outside or self.entity_attention_loss_src_inside:
                    inpainting_mask_downsample: torch.Tensor = F.interpolate(inpainting_mask, (attn_map_h, attn_map_w))
                    inpainting_mask_downsample = inpainting_mask_downsample.flatten(2)
                    outside_mask_downsample = inpainting_mask_downsample == 0
                    if self.entity_attention_loss_tgt_outside:
                        mask_downsample_cross_match_valid = mask_downsample_cross_match_valid & outside_mask_downsample[:, :, None]
                    if self.entity_attention_loss_src_inside:
                        mask_downsample_cross_match_valid = mask_downsample_cross_match_valid & (inpainting_mask_downsample[..., None] == 1)
                if self.point_sample_loss is not None:
                    loss = self.point_sample_loss(attn_map.float(), mask_downsample_cross_match.float(), mask_downsample_cross_match_valid.float())
                else:
                    loss = F.binary_cross_entropy_with_logits(attn_map, mask_downsample_cross_match.to(attn_map), reduction='none')
                    loss = (loss * mask_downsample_cross_match_valid).mean() * self.entity_attention_loss_weight
                loss_dict.update({key + tag: loss})
        return loss_dict

    def get_loss(self, batch):
        image = batch["image"].to(self.weight_dtype).cuda()
        mask = batch["mask"].cuda()
        inpainting_mask = batch["inpainting_mask"].to(self.weight_dtype)[:, None].cuda()
        # self.collector["inpainting_mask"] = inpainting_mask
        image_masked = image * (1 - inpainting_mask)

        ddpm_noise_scheduler = self.ddpm_noise_scheduler
        accelerator = self.accelerator

        if self.global_step % self.args.train_visual_steps == 1:
            self.save_for_training_show_tensors = DataDict()

        with torch.no_grad():
            image_masked_vaed = self.vae_encode(image_masked)
            image_vaed = self.vae_encode(image)

        noise = torch.randn_like(image_masked_vaed)
        bsz = image_masked_vaed.shape[0]
        timesteps = torch.randint(0, ddpm_noise_scheduler.config.num_train_timesteps, (bsz,), device=image_masked_vaed.device)
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        if self.train_999_step_ratio is not None:
            random_num = torch.rand(len(timesteps), device='cuda')
            timesteps[random_num < self.train_999_step_ratio] = 999
        noisy_latents = ddpm_noise_scheduler.add_noise(image_vaed, noise, timesteps)
        inpainting_mask_latent_size = F.interpolate(inpainting_mask, noisy_latents.shape[2:])
        loss_dict = {}


        class_labels = []
        mask_labels = []
        for mask_one in mask:
            mask_labels.append(F.one_hot(mask_one).permute(2, 0, 1).to(self.weight_dtype)[1:])
            class_labels.append(torch.zeros(mask_labels[-1].size(0), dtype=torch.long, device="cuda"))
        model_input = Mask2FormerInput(sample=noisy_latents,
                                       controlnet_cond=torch.cat([image_masked_vaed, inpainting_mask_latent_size],
                                                                 dim=1),
                                       timestep=timesteps, class_labels=class_labels, mask_labels=mask_labels)

        loss_dict = self.model(model_input)

        if self.save_for_training_show_tensors is not None:
            self.save_for_training_show_tensors.update(noisy_latents=noisy_latents, image_masked_vaed=image_masked_vaed, image_vaed=image_vaed, timesteps=timesteps, batch=batch,
                                                       mask=mask, model_input=model_input)
        self.collector.clear()
        return loss_dict

    def visual_training_result(self):
        model = self.accelerator.unwrap_model(self.model)
        noisy_latents, image_masked_vaed, image_vaed, timesteps, batch, mask, model_input = \
            self.save_for_training_show_tensors.get_keys(
                'noisy_latents', 'image_masked_vaed', 'image_vaed', 'timesteps', 'batch', 'mask', 'model_input')
        show_root = os.path.join(self.args.show_dir, 'training/{:06}'.format(self.global_step))
        remove_dir(show_root)
        with torch.no_grad():
            model.eval()
            with torch.no_grad():
                model_output = model(model_input)
            model.train()
            image_masked = self.vae_decode(image_masked_vaed)
            noisy_image = self.vae_decode(noisy_latents)
            image = self.vae_decode(image_vaed)
            image_masked_show = visual_tensor(image_masked, max_value=1, min_value=-1, reverse=True, max_num=self.max_visual_num_in_training)
            noisy_image_show = visual_tensor(noisy_image, max_value=1, min_value=-1, reverse=True, max_num=self.max_visual_num_in_training)
            image_show = visual_tensor(image, max_value=1, min_value=-1, reverse=True, max_num=self.max_visual_num_in_training)
            mask_show = visual_tensor(mask[:, None], max_value=255, min_value=0, max_num=self.max_visual_num_in_training)

            panoptic_seg = torch.stack([var['panoptic_seg'][0] for var in model_output], dim=0)
            panoptic_seg_show = visual_tensor(panoptic_seg[:, None], max_value=255, min_value=0, max_num=self.max_visual_num_in_training)
            image_mask_show = visual_mask(image_show, mask_show, stack_axis=1)[-1]
            panoptic_seg_show = visual_mask(image_show, panoptic_seg_show, stack_axis=1)[-1]
            shows = [image_show, image_masked_show, noisy_image_show, image_mask_show, panoptic_seg_show]
            shows = concat_differ_size(shows, axis=1)
            write_im(os.path.join(show_root, "shows.jpg"), shows)
            self.log_images(shows, tag="train_data", reverse_color=True)
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
        pass

    def debug_for_demo(self):
        show_root = os.path.join(self.args.show_dir, 'demo_dataset')
        remove_dir(show_root)
        for i_data, data in enumerate(self.demo_dataloader):
            image = visual_tensor(data['image'], max_value=255, min_value=0, max_num=5)[..., ::-1]
            # mask = visual_tensor(data['mask'][:, None], max_value=255, min_value=0, max_num=5)
            inpainting_mask = visual_tensor(data['inpainting_mask'], max_value=255, min_value=0, max_num=5)
            # mask_show = visual_mask(image, mask, stack_axis=1)[-1]
            inpainting_mask_show = visual_mask(image, inpainting_mask, stack_axis=1)[-1]
            # text_show = putTextBatch(show_texts=data["text"][:5], img_size=data['image'].shape[-2:][::-1])
            shows = [image, inpainting_mask_show]
            shows = concat_differ_size(shows)
            write_im(os.path.join(show_root, f"{i_data}.jpg"), shows)

if __name__ == "__main__":
    TrainMask2former().main()
